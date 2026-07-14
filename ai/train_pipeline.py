"""
JellyDay 통합 학습 파이프라인

기존 03_drugbank_lookup.py를 기반으로 한다.
폐기 이력:
  - 01_lstm_pattern.py: CASAS 데이터셋 미보유로 위치/행동 패턴 추적 제외
  - 02_bert_intent.py: 감성대화 말뭉치가 실제 통증/불편 호소와 의미가 어긋나고,
    신고/알림 액션도 없애기로 하면서 의도 분류 자체가 불필요해짐
  - 04_xgboost_risk.py: MIMIC-III 학습 데이터에서 drug_risk_score/
    has_major_interaction이 항상 0(자리채움값)이라 정작 이 시스템이 필요로
    하는 "약물 상호작용 -> 위험도" 관계를 XGBoost가 전혀 학습하지 못함.
    위험도는 DrugAnalyzer(DB 룩업) + DDI 임베딩 모델(2부)만으로 산출하기로
    하면서 통째로 제외.

실행 흐름 (main):
  1부. 약물 상호작용 DB 구축   (data/ 의 DDInter + 식약처 DUR CSV -> drug_data.db)
  2부. DDI 임베딩 모델 학습    (1부의 drug_data.db 필요, DB 미등록 약물 조합 위험도 추정)

DrugAnalyzer.analyze()가 반환하는 drug_risk_score(0~1)가 이 시스템의
최종 위험도다 — DUR/DDInter DB 확정 결과가 우선이고, DB에 없는 조합은
DDI 임베딩 모델의 추정치로 보조한다.
"""

import glob
import re
import sqlite3
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from gpu_config import setup_rtx3060

try:
    from rapidfuzz import fuzz as _rfuzz
    _RAPIDFUZZ_AVAILABLE = True
except ImportError:
    _RAPIDFUZZ_AVAILABLE = False

AI_DIR   = Path(__file__).resolve().parent
DATA_DIR = AI_DIR.parent / 'data'


# ============================================================
# 1부. 약물 상호작용 DB  (구 03_drugbank_lookup.py)
# ============================================================

# 성분명 정규화 (DUR ↔ DDInter 매칭용)

_AS_PATTERN = re.compile(r'\(as ([^)]+)\)', re.I)
_SALT_SUFFIX = re.compile(
    r'\s*(hydrochloride|hydrobromide|sulfate|sulphate|sodium|potassium|calcium|maleate|'
    r'mesylate|besylate|tartrate|citrate|acetate|succinate|fumarate|phosphate|nitrate|'
    r'bitartrate|hyclate|besilate|tosylate|camsylate|edisylate|gluconate|lactate|'
    r'benzoate|palmitate|stearate|valerate|dipropionate|propionate|decanoate|enanthate|'
    r'pamoate|embonate|hemifumarate|monohydrate|dihydrate|trihydrate|hydrate|anhydrous|'
    r'etexilate|lysinate|orotate|cilexetil|medoxomil|isopropyl)\b.*$', re.I)


def normalize_ingredient(raw: str) -> List[str]:
    """
    식약처 DUR 성분명을 DDInter Drug명과 매칭 가능한 base 성분명으로 정규화.

    DUR 성분명은 "epinephrine bitartrate (as epinephrine)"처럼 염(salt) 형태나
    "evogliptin+metformin hydrochloride"처럼 복합제 표기를 쓰는 반면, DDInter는
    "epinephrine"처럼 활성 성분 단일명만 사용한다.
      1. '+' 로 복합제 성분 분리
      2. '(as X)' 표기 시 실제 활성 성분(X) 추출
      3. 염 접미사 제거 (hydrochloride, sodium, tartrate 등)

    반환: 복합제는 성분별로 분리된 리스트, 단일 성분은 길이 1 리스트.
    """
    if not raw or not isinstance(raw, str):
        return []
    name = raw.lower().strip()
    if name in ('', 'nan'):
        return []
    out = []
    for part in name.split('+'):
        part = part.strip()
        m = _AS_PATTERN.search(part)
        base = m.group(1).strip() if m else part
        base = _SALT_SUFFIX.sub('', base).strip()
        if base:
            out.append(base)
    return out


class DrugNameNormalizer:
    """OCR 약물명 정규화 (브랜드명/처방전 텍스트용)."""
    _FORM = re.compile(
        r'(정|캡슐|캡|주사|시럽|연고|크림|패치|좌약|과립|액|앰플'
        r'|tablet|capsule|injection|mg|ml|mcg|%)\b', re.I)
    _NUM = re.compile(r'\d+[\.\d]*')

    def normalize(self, raw: str) -> str:
        if not isinstance(raw, str):
            return ''
        t = unicodedata.normalize('NFC', raw.strip())
        t = self._FORM.sub('', t)
        t = self._NUM.sub('', t)
        return re.sub(r'\s+', ' ', t).strip().lower()

    def normalize_list(self, raws: List[str]) -> List[str]:
        return [self.normalize(r) for r in raws]


_PAREN_OR_UNDERSCORE = re.compile(r'[_(].*$')
_name_normalizer = DrugNameNormalizer()


def clean_product_name(raw: str) -> str:
    """
    DUR 원본 제품명(예: '제클라정(클래리트로마이신)_(0.25g/1정)')에서
    실제 처방전/약봉투 OCR에 나타나는 브랜드명만 추출.
    """
    if not isinstance(raw, str):
        return ''
    name = _PAREN_OR_UNDERSCORE.sub('', raw).strip()
    return _name_normalizer.normalize(name)


class KoreanDrugDataLoader:
    """
    DDInter(국제 약물-약물 상호작용 DB, 영문 성분명 기준)와
    식약처 DUR(병용금기/용량주의/투여기간주의, 국문 CSV) 데이터를 SQLite로 통합 적재.
    """
    DDINTER_LEVEL_RISK = {'Major': 1.0, 'Moderate': 0.55, 'Minor': 0.2, 'Unknown': 0.1}

    def __init__(self, db_path: str = 'drug_data.db'):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute('PRAGMA journal_mode=WAL')
        self.conn.execute('PRAGMA synchronous=NORMAL')
        self.conn.execute('PRAGMA cache_size=-65536')
        self._init_tables()

    def _init_tables(self):
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS ingredients (
            ingredient        TEXT PRIMARY KEY,
            max_dose_mg_day   REAL,
            max_duration_days INTEGER
        );
        CREATE TABLE IF NOT EXISTS product_alias (
            alias        TEXT PRIMARY KEY,
            ingredient   TEXT,
            product_code TEXT
        );
        CREATE TABLE IF NOT EXISTS dur_interactions (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            ingredient_a TEXT,
            ingredient_b TEXT,
            reason       TEXT,
            notice_no    TEXT,
            notice_date  TEXT
        );
        CREATE TABLE IF NOT EXISTS ddinter_interactions (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            ingredient_a TEXT,
            ingredient_b TEXT,
            level        TEXT,
            risk_score   REAL
        );
        CREATE INDEX IF NOT EXISTS idx_alias     ON product_alias(alias);
        CREATE INDEX IF NOT EXISTS idx_dur_a     ON dur_interactions(ingredient_a);
        CREATE INDEX IF NOT EXISTS idx_dur_b     ON dur_interactions(ingredient_b);
        CREATE INDEX IF NOT EXISTS idx_ddi_a     ON ddinter_interactions(ingredient_a);
        CREATE INDEX IF NOT EXISTS idx_ddi_b     ON ddinter_interactions(ingredient_b);
        """)
        self.conn.commit()

    def _register_alias(self, cur, raw_product_name, ingredient, product_code) -> int:
        alias = clean_product_name(raw_product_name)
        if not alias or not ingredient:
            return 0
        cur.execute(
            "INSERT OR IGNORE INTO product_alias(alias,ingredient,product_code) VALUES (?,?,?)",
            (alias, ingredient, str(product_code)))
        return 1

    def load_ddinter(self, glob_pattern: str = None):
        glob_pattern = glob_pattern or str(DATA_DIR / 'ddinter_downloads_code_*.csv')
        paths = sorted(glob.glob(glob_pattern))
        cur = self.conn.cursor()
        total = 0
        for path in paths:
            df = pd.read_csv(path)
            df['ingredient_a'] = df['Drug_A'].str.lower().str.strip()
            df['ingredient_b'] = df['Drug_B'].str.lower().str.strip()
            df['risk_score'] = df['Level'].map(self.DDINTER_LEVEL_RISK).fillna(0.1)
            rows = list(df[['ingredient_a', 'ingredient_b', 'Level', 'risk_score']]
                        .itertuples(index=False, name=None))
            cur.executemany(
                "INSERT INTO ddinter_interactions"
                "(ingredient_a,ingredient_b,level,risk_score) VALUES (?,?,?,?)", rows)
            total += len(rows)
        self.conn.commit()
        print(f"[DDInter] 파일 {len(paths)}개, {total}쌍 적재 완료")

    def load_dur_contraindications(self, csv_path: str = None, encoding: str = 'cp949'):
        csv_path = csv_path or str(DATA_DIR / '한국의약품안전관리원_병용금기약물_20240625.csv')
        df = pd.read_csv(csv_path, encoding=encoding, dtype=str)
        df = df.rename(columns={
            '성분명1': 'ing1', '제품명1': 'prod1', '제품코드1': 'code1',
            '성분명2': 'ing2', '제품명2': 'prod2', '제품코드2': 'code2',
            '금기사유': 'reason', '공고번호': 'notice_no', '공고일자': 'notice_date',
        }).dropna(subset=['ing1', 'ing2'])

        cur = self.conn.cursor()

        pairs = df[['ing1', 'ing2', 'reason', 'notice_no', 'notice_date']] \
            .drop_duplicates(subset=['ing1', 'ing2', 'reason'])
        n_inter = 0
        for row in pairs.itertuples(index=False):
            for a in normalize_ingredient(row.ing1):
                for b in normalize_ingredient(row.ing2):
                    cur.execute(
                        "INSERT INTO dur_interactions"
                        "(ingredient_a,ingredient_b,reason,notice_no,notice_date) VALUES (?,?,?,?,?)",
                        (a, b, row.reason, row.notice_no, row.notice_date))
                    n_inter += 1

        n_alias = 0
        for prod_col, ing_col, code_col in (('prod1', 'ing1', 'code1'), ('prod2', 'ing2', 'code2')):
            uniq = df[[prod_col, ing_col, code_col]].drop_duplicates(subset=[prod_col])
            for row in uniq.itertuples(index=False):
                names = normalize_ingredient(getattr(row, ing_col))
                if names:
                    n_alias += self._register_alias(
                        cur, getattr(row, prod_col), names[0], getattr(row, code_col))

        self.conn.commit()
        print(f"[DUR 병용금기] 성분쌍 {n_inter}건, 제품 alias {n_alias}건 적재 완료")

    def load_dose_caution(self, csv_path: str = None, encoding: str = 'cp949'):
        csv_path = csv_path or str(DATA_DIR / '한국의약품안전관리원_용량주의약물_20240501.csv')
        df = pd.read_csv(csv_path, encoding=encoding, dtype={'제품코드': str})
        df = df.rename(columns={
            '제품명': 'prod', '성분명': 'ing', '제품코드': 'code',
            '1일최대 투여기준량': 'max_dose',
        })

        cur = self.conn.cursor()

        per_ing = df[['ing', 'max_dose']].dropna().groupby('ing')['max_dose'].max().reset_index()
        for row in per_ing.itertuples(index=False):
            base = normalize_ingredient(row.ing)
            key = base[0] if base else row.ing.lower()
            cur.execute(
                "INSERT INTO ingredients(ingredient,max_dose_mg_day) VALUES (?,?)"
                " ON CONFLICT(ingredient) DO UPDATE SET max_dose_mg_day=excluded.max_dose_mg_day",
                (key, float(row.max_dose)))

        n_alias = 0
        uniq = df[['prod', 'ing', 'code']].drop_duplicates(subset=['prod'])
        for row in uniq.itertuples(index=False):
            names = normalize_ingredient(row.ing)
            if names:
                n_alias += self._register_alias(cur, row.prod, names[0], row.code)

        self.conn.commit()
        print(f"[DUR 용량주의] 성분 {len(per_ing)}건, 제품 alias {n_alias}건 적재 완료")

    def load_duration_caution(self, csv_path: str = None, encoding: str = 'cp949'):
        csv_path = csv_path or str(DATA_DIR / '한국의약품안전관리원_투여기간주의약물_20231108.csv')
        df = pd.read_csv(csv_path, encoding=encoding, dtype={'제품코드': str})
        df = df.rename(columns={
            '제품명': 'prod', '성분명': 'ing', '제품코드': 'code',
            '최대투여기간일수': 'max_days',
        })

        cur = self.conn.cursor()

        per_ing = df[['ing', 'max_days']].dropna().groupby('ing')['max_days'].max().reset_index()
        for row in per_ing.itertuples(index=False):
            base = normalize_ingredient(row.ing)
            key = base[0] if base else row.ing.lower()
            cur.execute(
                "INSERT INTO ingredients(ingredient,max_duration_days) VALUES (?,?)"
                " ON CONFLICT(ingredient) DO UPDATE SET max_duration_days=excluded.max_duration_days",
                (key, int(row.max_days)))

        n_alias = 0
        uniq = df[['prod', 'ing', 'code']].drop_duplicates(subset=['prod'])
        for row in uniq.itertuples(index=False):
            names = normalize_ingredient(row.ing)
            if names:
                n_alias += self._register_alias(cur, row.prod, names[0], row.code)

        self.conn.commit()
        print(f"[DUR 투여기간주의] 성분 {len(per_ing)}건, 제품 alias {n_alias}건 적재 완료")

    def load_all(self):
        self.load_ddinter()
        self.load_dur_contraindications()
        self.load_dose_caution()
        self.load_duration_caution()


class FuzzyDrugMatcher:
    """
    OCR 오인식 대응 Fuzzy 약물명 매칭.
    SequenceMatcher(항상 사용 가능) + rapidfuzz(설치 시, ~100x 빠름) 병행.
    """

    def __init__(self, conn: sqlite3.Connection,
                 threshold: float = 0.75,
                 top_k: int = 3):
        self.conn = conn
        self.threshold = threshold
        self.top_k = top_k
        self._cache: Optional[dict] = None

    def _load_cache(self) -> dict:
        if self._cache is None:
            rows = self.conn.execute(
                "SELECT alias, ingredient FROM product_alias"
            ).fetchall()
            self._cache = {r[0]: r[1] for r in rows}
            print(f"[FuzzyMatcher] alias 캐시 로드: {len(self._cache)}건")
        return self._cache

    def invalidate_cache(self):
        self._cache = None

    @staticmethod
    def _similarity_sequence(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

    @staticmethod
    def _similarity_rapidfuzz(a: str, b: str) -> float:
        return _rfuzz.token_sort_ratio(a, b) / 100.0

    def _similarity(self, a: str, b: str) -> float:
        seq_score = self._similarity_sequence(a, b)
        if _RAPIDFUZZ_AVAILABLE:
            rf_score = self._similarity_rapidfuzz(a, b)
            return max(seq_score, rf_score)
        return seq_score

    def find_candidates(self, norm: str) -> List[dict]:
        cache = self._load_cache()
        method = 'rapidfuzz' if _RAPIDFUZZ_AVAILABLE else 'sequence_matcher'
        scored = []

        for alias, ingredient in cache.items():
            score = self._similarity(norm, alias)
            if score >= self.threshold:
                scored.append({
                    'alias': alias,
                    'ingredient': ingredient,
                    'match_score': round(score, 4),
                    'match_method': method,
                })

        scored.sort(key=lambda x: x['match_score'], reverse=True)
        return scored[:self.top_k]


class DrugAnalyzer:
    def __init__(self, loader: KoreanDrugDataLoader,
                 fuzzy_threshold: float = 0.75,
                 fuzzy_top_k: int = 3,
                 ddi_predictor: 'Optional[DDIRiskPredictor]' = None):
        self.conn = loader.conn
        self.normalizer = DrugNameNormalizer()
        self.fuzzy = FuzzyDrugMatcher(
            conn=self.conn,
            threshold=fuzzy_threshold,
            top_k=fuzzy_top_k,
        )
        # DB에 없는 미등록 약물 조합의 위험도를 추정하는 학습된 임베딩 모델(선택).
        # attach_ddi_predictor()로 나중에 붙일 수도 있다.
        self.ddi_predictor = ddi_predictor

    def lookup(self, norm: str) -> Optional[str]:
        row = self.conn.execute(
            "SELECT ingredient FROM product_alias WHERE alias=?", (norm,)
        ).fetchone()
        return row[0] if row else None

    def get_info(self, ingredient: str) -> dict:
        row = self.conn.execute(
            "SELECT ingredient, max_dose_mg_day, max_duration_days FROM ingredients WHERE ingredient=?",
            (ingredient,)
        ).fetchone()
        if row:
            return {'id': row[0], 'max_dose_mg_day': row[1], 'max_duration_days': row[2]}
        return {'id': ingredient, 'max_dose_mg_day': None, 'max_duration_days': None}

    def check_interactions(self, ingredients: List[str]) -> List[dict]:
        out, seen = [], set()
        for i, a in enumerate(ingredients):
            for b in ingredients[i + 1:]:
                key = tuple(sorted((a, b)))
                if key in seen:
                    continue
                seen.add(key)

                dur_row = self.conn.execute(
                    "SELECT reason FROM dur_interactions"
                    " WHERE (ingredient_a=? AND ingredient_b=?) OR (ingredient_a=? AND ingredient_b=?)"
                    " LIMIT 1", (a, b, b, a)
                ).fetchone()
                if dur_row:
                    out.append({'drug_a': a, 'drug_b': b, 'source': 'DUR',
                                'severity': 'contraindicated', 'risk_score': 1.0,
                                'description': dur_row[0]})
                    continue

                ddi_row = self.conn.execute(
                    "SELECT level, risk_score FROM ddinter_interactions"
                    " WHERE (ingredient_a=? AND ingredient_b=?) OR (ingredient_a=? AND ingredient_b=?)"
                    " ORDER BY risk_score DESC LIMIT 1", (a, b, b, a)
                ).fetchone()
                if ddi_row:
                    out.append({'drug_a': a, 'drug_b': b, 'source': 'DDInter',
                                'severity': ddi_row[0].lower(), 'risk_score': ddi_row[1],
                                'description': f'DDInter {ddi_row[0]} interaction'})
                    continue

                # DUR·DDInter 둘 다 미등록 조합 -> 임베딩 모델로 추정치 제공 (붙어있을 때만)
                if self.ddi_predictor is not None:
                    predicted = self.ddi_predictor.predict(a, b)
                    if predicted is not None and predicted >= self.ddi_predictor.report_threshold:
                        out.append({'drug_a': a, 'drug_b': b, 'source': 'model',
                                    'severity': 'predicted', 'risk_score': predicted,
                                    'description': 'DB 미등록 조합 - 학습된 임베딩 모델 추정치'})
        return out

    def analyze(self, ocr_names: List[str]) -> dict:
        norms = self.normalizer.normalize_list(ocr_names)
        identified, unidentified = {}, []

        for norm, raw in zip(norms, ocr_names):
            ingredient = self.lookup(norm)
            if ingredient:
                identified[raw] = self.get_info(ingredient)
            else:
                candidates = self.fuzzy.find_candidates(norm)
                auto_matched = None
                if candidates and candidates[0]['match_score'] >= 0.90:
                    best = candidates[0]
                    info = self.get_info(best['ingredient'])
                    info['fuzzy_matched'] = True
                    info['match_score'] = best['match_score']
                    info['match_method'] = best['match_method']
                    info['matched_alias'] = best['alias']
                    identified[raw] = info
                    auto_matched = best
                    print(f"[FuzzyMatcher] '{raw}' -> '{best['alias']}' "
                          f"(score={best['match_score']:.3f}, method={best['match_method']})")

                if auto_matched is None:
                    unidentified.append({
                        'raw': raw,
                        'normalized': norm,
                        'fuzzy_candidates': candidates,
                    })

        ids = [v['id'] for v in identified.values() if v]
        inters = self.check_interactions(ids)
        risk = self._risk(identified, inters)

        return {
            'identified': identified,
            'unidentified': unidentified,
            'interactions': inters,
            'drug_risk_score': round(risk, 3),
        }

    def _risk(self, identified, inters) -> float:
        if not identified:
            return 0.0
        inter_score = max((i['risk_score'] for i in inters), default=0.0)
        poly = max(0.0, (len(identified) - 4) * 0.08)
        contraindicated = any(i['severity'] == 'contraindicated' for i in inters)
        return min(max(inter_score + poly, 1.0 if contraindicated else 0.0), 1.0)


def build_drug_db(db_path: str = None) -> DrugAnalyzer:
    """data/ 의 DDInter + 식약처 DUR CSV를 SQLite로 적재하고 DrugAnalyzer를 반환."""
    db_path = db_path or str(DATA_DIR / 'drug_data.db')
    loader = KoreanDrugDataLoader(db_path)
    loader.load_all()
    return DrugAnalyzer(loader, fuzzy_threshold=0.75, fuzzy_top_k=3)


# ============================================================
# 2부. 약물 상호작용 임베딩 모델 (DDI 위험도 예측)
#
# DrugAnalyzer.check_interactions()의 DUR/DDInter 룩업은 DB에 등록된
# 조합만 알 수 있다. DDInter 22만 쌍(1,939개 고유 약물)의 (약물A, 약물B,
# 위험등급) 라벨로 임베딩 기반 모델을 학습시켜, DB에 없는 새 조합에 대해
# "비슷한 약물들의 상호작용 패턴" 기반 추정 위험도를 제공한다.
#
# 주의: 음성(비상호작용) 샘플은 DDInter에 실려있지 않은 조합을 무작위로
# 뽑아 risk_score=0으로 사용한다 — "안전이 확인됨"이 아니라 "DDInter에
# 등재되지 않음"이라는 뜻이므로, 예측치는 DB 확정 결과보다 신뢰도가 낮다.
# ============================================================

class DDIEmbeddingModel(torch.nn.Module):
    """
    약물 임베딩 2개를 대칭 결합(합/절대차/곱)해 상호작용 위험도(0~1)를 예측.
    concat이 아니라 대칭 연산을 쓰는 이유: 상호작용은 순서가 없으므로
    (A,B)와 (B,A)의 예측이 항상 같아야 한다.
    """

    def __init__(self, n_drugs: int, embed_dim: int = 32):
        super().__init__()
        self.embed = torch.nn.Embedding(n_drugs, embed_dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embed_dim * 3, 64), torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 1),
        )

    def forward(self, a_idx: torch.Tensor, b_idx: torch.Tensor) -> torch.Tensor:
        ea, eb = self.embed(a_idx), self.embed(b_idx)
        combined = torch.cat([ea + eb, torch.abs(ea - eb), ea * eb], dim=-1)
        return torch.sigmoid(self.mlp(combined).squeeze(-1))


class DDIRiskPredictor:
    """학습된 DDIEmbeddingModel + 어휘사전을 감싸 (약물명, 약물명) -> 위험도 추론을 제공."""

    def __init__(self, model: DDIEmbeddingModel, vocab: dict,
                 report_threshold: float = 0.3):
        self.model = model.eval()
        self.vocab = vocab   # ingredient(str) -> idx
        self.report_threshold = report_threshold   # 이 이상일 때만 check_interactions에 보고

    def predict(self, drug_a: str, drug_b: str) -> Optional[float]:
        """어휘사전에 없는(OOV) 약물이 하나라도 있으면 추정 불가 -> None."""
        ia, ib = self.vocab.get(drug_a), self.vocab.get(drug_b)
        if ia is None or ib is None:
            return None
        with torch.no_grad():
            score = self.model(torch.tensor([ia]), torch.tensor([ib]))
        return round(float(score.item()), 3)

    def save(self, path: str):
        torch.save({
            'state_dict': self.model.state_dict(),
            'vocab': self.vocab,
            'embed_dim': self.model.embed.embedding_dim,
            'report_threshold': self.report_threshold,
        }, path)
        print(f"저장 완료: {path}")

    @staticmethod
    def load(path: str) -> 'DDIRiskPredictor':
        ckpt = torch.load(path, map_location='cpu')
        model = DDIEmbeddingModel(len(ckpt['vocab']), ckpt['embed_dim'])
        model.load_state_dict(ckpt['state_dict'])
        return DDIRiskPredictor(model, ckpt['vocab'], ckpt.get('report_threshold', 0.3))


def train_ddi_model(db_path: str = None,
                    embed_dim: int = 32,
                    epochs: int = 15,
                    neg_ratio: float = 1.0,
                    seed: int = 42) -> DDIRiskPredictor:
    """
    drug_data.db의 ddinter_interactions 테이블(양성 샘플)에 무작위 미등록
    조합(음성 샘플, risk_score=0)을 섞어 DDIEmbeddingModel을 학습시킨다.
    """
    db_path = db_path or str(DATA_DIR / 'drug_data.db')
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT ingredient_a, ingredient_b, MAX(risk_score) FROM ddinter_interactions"
        " GROUP BY ingredient_a, ingredient_b"
    ).fetchall()
    conn.close()

    drugs = sorted({r[0] for r in rows} | {r[1] for r in rows})
    vocab = {d: i for i, d in enumerate(drugs)}
    n_drugs = len(vocab)
    print(f"[DDI] 양성 쌍 {len(rows)}개, 고유 약물 {n_drugs}개")

    rng = np.random.default_rng(seed)
    pos_a = np.array([vocab[r[0]] for r in rows], dtype=np.int64)
    pos_b = np.array([vocab[r[1]] for r in rows], dtype=np.int64)
    pos_y = np.array([r[2] for r in rows], dtype=np.float32)

    known = {(min(a, b), max(a, b)) for a, b in zip(pos_a.tolist(), pos_b.tolist())}
    n_neg = int(len(rows) * neg_ratio)
    neg_pairs = set()
    while len(neg_pairs) < n_neg:
        a, b = int(rng.integers(0, n_drugs)), int(rng.integers(0, n_drugs))
        if a == b:
            continue
        key = (min(a, b), max(a, b))
        if key not in known:
            neg_pairs.add(key)
    neg_a = np.array([p[0] for p in neg_pairs], dtype=np.int64)
    neg_b = np.array([p[1] for p in neg_pairs], dtype=np.int64)
    neg_y = np.zeros(len(neg_pairs), dtype=np.float32)
    print(f"[DDI] 음성(미등록) 쌍 {len(neg_pairs)}개 무작위 샘플링")

    a_idx = np.concatenate([pos_a, neg_a])
    b_idx = np.concatenate([pos_b, neg_b])
    y = np.concatenate([pos_y, neg_y])

    perm = rng.permutation(len(y))
    a_idx, b_idx, y = a_idx[perm], b_idx[perm], y[perm]
    split = int(0.9 * len(y))

    device = setup_rtx3060()
    model = DDIEmbeddingModel(n_drugs, embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = torch.nn.MSELoss()

    def to_tensors(idx):
        return (torch.tensor(a_idx[idx], device=device),
                torch.tensor(b_idx[idx], device=device),
                torch.tensor(y[idx], device=device))

    tr_idx, val_idx = np.arange(split), np.arange(split, len(y))
    batch_size = 4096

    for epoch in range(1, epochs + 1):
        model.train()
        rng.shuffle(tr_idx)
        total_loss = 0.0
        for start in range(0, len(tr_idx), batch_size):
            batch = tr_idx[start:start + batch_size]
            a_b, b_b, y_b = to_tensors(batch)
            optimizer.zero_grad(set_to_none=True)
            pred = model(a_b, b_b)
            loss = criterion(pred, y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)

        model.eval()
        with torch.no_grad():
            a_v, b_v, y_v = to_tensors(val_idx)
            val_mae = torch.mean(torch.abs(model(a_v, b_v) - y_v)).item()
        print(f"  Epoch {epoch:2d} | train_mse={total_loss/len(tr_idx):.4f} | val_mae={val_mae:.4f}")

    predictor = DDIRiskPredictor(model.cpu(), vocab)
    predictor.save(str(AI_DIR / 'ddi_model.pt'))
    return predictor


# ============================================================
# 통합 실행
# ============================================================

def main():
    print("=" * 60)
    print("[1/2] 약물 상호작용 DB 구축")
    print("=" * 60)
    db_path = str(DATA_DIR / 'drug_data.db')
    analyzer = build_drug_db(db_path)

    print("=" * 60)
    print("[2/2] DDI 임베딩 모델 학습")
    print("=" * 60)
    ddi_predictor = train_ddi_model(db_path)
    analyzer.ddi_predictor = ddi_predictor

    print("=" * 60)
    print("완료:",
          f"drug_analyzer={'OK' if analyzer else 'X'}",
          f"ddi_model={'OK' if ddi_predictor else 'X'}")
    print("=" * 60)
    return analyzer, ddi_predictor


if __name__ == '__main__':
    main()
