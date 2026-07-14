"""
JellyDay 통합 학습 파이프라인

기존 03_drugbank_lookup.py + 04_xgboost_risk.py 를 하나로 합쳤다.
폐기 이력: 01_lstm_pattern.py(CASAS 데이터셋 미보유),
02_bert_intent.py(감성대화 말뭉치가 실제 통증/불편 호소와 의미가 어긋나고,
신고/알림 액션도 없애기로 하면서 의도 분류 자체가 불필요해짐).

실행 흐름 (main):
  1부. 약물 상호작용 DB 구축   (data/ 의 DDInter + 식약처 DUR CSV -> drug_data.db)
  2부. XGBoost 위험도 모델 학습 (data/MIMIC -III (10000 patients)/ 있을 때만)

각 부는 필요한 입력이 없으면 자동으로 건너뛴다.
"""

import glob
import json
import re
import sqlite3
import unicodedata
from collections import deque
from dataclasses import dataclass, asdict, field
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Deque, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

from gpu_config import print_vram, setup_rtx3060

try:
    from rapidfuzz import fuzz as _rfuzz
    _RAPIDFUZZ_AVAILABLE = True
except ImportError:
    _RAPIDFUZZ_AVAILABLE = False

try:
    import xgboost as xgb
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False

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
                 fuzzy_top_k: int = 3):
        self.conn = loader.conn
        self.normalizer = DrugNameNormalizer()
        self.fuzzy = FuzzyDrugMatcher(
            conn=self.conn,
            threshold=fuzzy_threshold,
            top_k=fuzzy_top_k,
        )

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
# 2부. XGBoost 위험도 모델  (구 04_xgboost_risk.py)
#
# 폐기 이력:
#   - LSTM(01_lstm_pattern.py): pattern_anomaly_score, pattern_variability,
#     zone_transition_abnormal, time_in_unusual_zone 피처 제거
#   - BERT(구 02_bert_intent.py): 감성대화 말뭉치가 일상 감정 대화라 노인 돌봄
#     통증/불편 호소와 의미가 어긋나고, 신고/알림 액션도 없애기로 하면서
#     intent_risk_score, pain_score, distress_score 피처 및 BERT 파트 전체 제거
#   - 신고(알림) 액션: RiskXGBoost._action()의 notify/contact 로직 제거 —
#     이제 risk_level/raw_score만 반환하는 순수 점수 모델
# 13개 -> 4개 피처로 축소.
# ============================================================

@dataclass
class RiskFeatures:
    drug_risk_score        : float
    n_drugs                 : float   # / 10
    has_major_interaction  : float
    acceleration_anomaly    : float

    def to_array(self) -> np.ndarray:
        return np.array([
            self.drug_risk_score, self.n_drugs,
            self.has_major_interaction, self.acceleration_anomaly,
        ], dtype=np.float32)

    @staticmethod
    def names() -> List[str]:
        return [
            'drug_risk', 'n_drugs', 'has_major_interaction',
            'acceleration_anomaly',
        ]


@dataclass
class FeedbackSample:
    """
    사용자가 오탐을 교정한 단일 샘플.

    predicted_risk : XGBoost가 예측한 위험도
    corrected_risk : 사용자 피드백으로 보정된 위험도
    features       : 해당 시점의 RiskFeatures 배열 (길이 4)
    timestamp      : 피드백 수신 시각
    feedback_type  : 'false_alarm'(오탐) | 'missed'(미탐) | 'manual'(수동 입력)
    """
    predicted_risk : float
    corrected_risk : float
    features        : List[float]
    timestamp        : str = field(default_factory=lambda: datetime.now().isoformat())
    feedback_type    : str = 'false_alarm'


class FeedbackBuffer:
    """
    사용자 피드백 샘플을 누적하는 순환 큐.
      - maxlen으로 오래된 피드백 자동 만료
      - persist_path 지정 시 JSONL로 영속화 -> 재시작 후에도 복원 가능
      - get_batch()로 numpy 배열 변환 후 IncrementalTrainer에 전달
    """

    def __init__(self, maxlen: int = 500, persist_path: str = "feedback_buffer.jsonl"):
        self._buf: Deque[FeedbackSample] = deque(maxlen=maxlen)
        self.persist_path = Path(persist_path)
        self._load()

    def add(self, sample: FeedbackSample) -> None:
        self._buf.append(sample)
        self._append_jsonl(sample)
        print(f"[FeedbackBuffer] 샘플 추가 | 누적={len(self._buf)} "
              f"| type={sample.feedback_type} "
              f"| predicted={sample.predicted_risk:.1f} → corrected={sample.corrected_risk:.1f}")

    def __len__(self) -> int:
        return len(self._buf)

    def get_batch(self) -> tuple:
        X = np.array([s.features for s in self._buf], dtype=np.float32)
        y = np.array([s.corrected_risk for s in self._buf], dtype=np.float32)
        return X, y

    def clear(self) -> None:
        self._buf.clear()

    def _append_jsonl(self, sample: FeedbackSample) -> None:
        try:
            with open(self.persist_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[FeedbackBuffer] 저장 실패: {e}")

    def _load(self) -> None:
        if not self.persist_path.exists():
            return
        try:
            with open(self.persist_path, "r", encoding="utf-8") as f:
                for line in f:
                    d = json.loads(line.strip())
                    self._buf.append(FeedbackSample(**d))
            print(f"[FeedbackBuffer] 이전 피드백 {len(self._buf)}개 복원")
        except Exception as e:
            print(f"[FeedbackBuffer] 복원 실패: {e}")


class MIMICPreprocessor:
    """
    MIMIC-III 전처리.

    data/MIMIC -III (10000 patients)/ 는 표준 MIMIC-III 배포 형식이 아니라
    테이블별 하위 폴더 + '_random'/'_sorted' 두 개의 CSV로 나뉘어 있고,
    (실측 확인 결과) CHARTEVENTS(활력징후) 테이블 자체가 없다.

    -> _read_table()이 두 CSV를 합쳐 HADM_ID 중복 제거 후 사용하고,
       CHARTEVENTS가 없으면 건너뛴다 (vitals 피처는 _features()의 기존
       fallback으로 자동 0 처리됨 — 컬럼이 없으면 norm()이 zeros 반환).
    """
    VITAL_ITEMS = {211: 'hr', 51: 'sbp', 8368: 'dbp', 646: 'spo2', 615: 'rr'}
    TYPE_RISK = {'ELECTIVE': 2, 'NEWBORN': 1, 'URGENT': 6, 'EMERGENCY': 9}

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def _read_table(self, table: str, usecols: list = None) -> pd.DataFrame:
        folder = self.data_dir / table
        parts = sorted(folder.glob(f'{table}_*.csv')) if folder.is_dir() else []
        flat = self.data_dir / f'{table}.csv'

        if parts:
            dfs = [pd.read_csv(p, usecols=usecols, low_memory=False) for p in parts]
            df = pd.concat(dfs, ignore_index=True)
            if 'HADM_ID' in df.columns:
                df = df.drop_duplicates(subset='HADM_ID')
            return df
        if flat.exists():
            return pd.read_csv(flat, usecols=usecols, low_memory=False)
        raise FileNotFoundError(f"{table} 데이터 없음: {folder} / {flat}")

    def load(self) -> tuple:
        admissions = self._read_table('ADMISSIONS', usecols=['HADM_ID', 'ADMISSION_TYPE', 'DIAGNOSIS'])
        prescriptions = self._read_table('PRESCRIPTIONS', usecols=['HADM_ID', 'DRUG', 'DOSE_VAL_RX'])

        try:
            chartevents = self._read_table('CHARTEVENTS', usecols=['HADM_ID', 'ITEMID', 'VALUENUM'])
        except FileNotFoundError:
            print("[MIMICPreprocessor] CHARTEVENTS 없음 -> 활력징후 피처는 0으로 대체")
            chartevents = None

        dc = prescriptions.groupby('HADM_ID')['DRUG'].nunique().rename('n_drugs')

        df = admissions
        if chartevents is not None:
            vitals = chartevents[chartevents['ITEMID'].isin(self.VITAL_ITEMS)].copy()
            vitals['vital'] = vitals['ITEMID'].map(self.VITAL_ITEMS)
            vp = vitals.groupby(['HADM_ID', 'vital'])['VALUENUM'].mean().unstack(fill_value=np.nan)
            df = df.join(vp, on='HADM_ID', how='left')

        df = df.join(dc, on='HADM_ID', how='left')
        df['n_drugs'] = df['n_drugs'].fillna(0)
        df['risk'] = df['ADMISSION_TYPE'].map(self.TYPE_RISK).fillna(5).astype(int)
        df = self._adjust_diagnosis(df)

        X = self._features(df)
        y = df['risk'].clip(1, 10).values
        valid = ~np.isnan(X).any(axis=1)
        return X[valid].astype(np.float32), y[valid]

    def _adjust_diagnosis(self, df):
        keywords = ['sepsis', 'cardiac arrest', 'respiratory failure',
                    'stroke', 'acute myocardial', 'hemorrhage']
        diag = df['DIAGNOSIS'].str.lower().fillna('')
        for kw in keywords:
            df.loc[diag.str.contains(kw), 'risk'] = \
                df.loc[diag.str.contains(kw), 'risk'].clip(lower=8)
        return df

    def _features(self, df):
        """
        MIMIC 자체엔 drug_risk 등 실제 RiskFeatures 신호가 없으므로,
        생체 신호 기반 종합 위험도(comp)를 acceleration_anomaly 자리에 프록시로 넣어
        XGBoost가 RiskFeatures와 동일한 4차원 벡터 구조를 먼저 학습하게 한다.
        이후 IncrementalTrainer가 실제 피드백으로 미세조정한다.
        """
        def norm(col, lo, hi):
            if col not in df.columns:
                return np.zeros(len(df))
            v = df[col].values.astype(float)
            return np.clip(np.where(v < lo, (lo - v) / lo,
                           np.where(v > hi, (v - hi) / hi, 0.0)), 0, 1)

        hr = norm('hr', 60, 100)
        sbp = norm('sbp', 90, 140)
        spo2 = norm('spo2', 95, 100)
        rr = norm('rr', 12, 20)
        nd = np.clip(df['n_drugs'].values / 10.0, 0, 1)
        comp = (hr + sbp + spo2 + rr) / 4.0

        return np.column_stack([
            np.zeros(len(df)),   # drug_risk_score: MIMIC엔 상호작용 정보 없음
            nd,
            np.zeros(len(df)),   # has_major_interaction
            comp,                # acceleration_anomaly 자리의 생체 신호 프록시
        ])


class RiskXGBoost:
    def __init__(self):
        self.model = xgb.XGBRegressor(
            device='cuda',
            tree_method='hist',
            max_bin=256,

            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            gamma=0.1,
            objective='reg:squarederror',
            eval_metric='mae',
            early_stopping_rounds=30,
            random_state=42,
        )
        self.is_trained = False

    def train(self, X_tr, y_tr, X_val, y_val):
        print_vram('XGBoost 학습 전')
        self.model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            sample_weight=compute_sample_weight('balanced', y_tr.astype(int)),
            verbose=50,
        )
        self.is_trained = True
        torch.cuda.empty_cache()
        print_vram('XGBoost 학습 후')

    def predict(self, features: RiskFeatures) -> dict:
        """
        신고/알림 액션 없이 risk_level(1~10)과 raw_score만 반환하는
        순수 점수 모델. 알림·119 연동이 필요해지면 이 반환값을 소비하는
        쪽에서 임계값 기반으로 판단한다.
        """
        x = features.to_array().reshape(1, -1)
        raw = float(self.model.predict(x)[0])
        level = int(np.clip(round(raw), 1, 10))
        return {
            'risk_level': level,
            'raw_score': round(raw, 3),
        }

    def save(self, path: str):
        self.model.save_model(path)
        print(f"저장 완료: {path}")

    def load(self, path: str):
        self.model.load_model(path)
        self.is_trained = True


class IncrementalTrainer:
    """
    FeedbackBuffer가 min_samples 이상 쌓이면 기존 XGBoost 모델에
    새 트리를 추가(boosting 계속)하여 온라인 학습을 수행한다.
    xgb_model=기존_booster 로 전달하면 기존 트리를 유지한 채 잔차만 학습한다
    (catastrophic forgetting 없음).
    """

    def __init__(
        self,
        risk_model: RiskXGBoost,
        buffer: FeedbackBuffer,
        min_samples: int = 30,
        incremental_trees: int = 20,
        feedback_lr: float = 0.01,
        snapshot_dir: str = "model_snapshots",
        clear_after_fit: bool = False,
    ):
        self.risk_model = risk_model
        self.buffer = buffer
        self.min_samples = min_samples
        self.incremental_trees = incremental_trees
        self.feedback_lr = feedback_lr
        self.snapshot_dir = Path(snapshot_dir)
        self.clear_after_fit = clear_after_fit
        self.retrain_count = 0

        self.snapshot_dir.mkdir(exist_ok=True)

    def try_incremental_fit(self) -> bool:
        if len(self.buffer) < self.min_samples:
            print(f"[IncrementalTrainer] 샘플 부족 "
                  f"({len(self.buffer)}/{self.min_samples}) → 학습 보류")
            return False

        if not self.risk_model.is_trained:
            print("[IncrementalTrainer] 기반 모델 미학습 → 증분 학습 불가")
            return False

        self._run(self.buffer.get_batch())
        return True

    def _run(self, batch: tuple) -> None:
        X_fb, y_fb = batch
        print(f"\n[IncrementalTrainer] 증분 학습 시작 | "
              f"피드백 샘플={len(X_fb)} | "
              f"추가 트리={self.incremental_trees}")

        snapshot_path = self._save_snapshot()

        try:
            booster = self.risk_model.model.get_booster()

            incremental_model = xgb.XGBRegressor(
                device='cuda',
                tree_method='hist',
                max_bin=256,
                n_estimators=self.incremental_trees,
                max_depth=4,
                learning_rate=self.feedback_lr,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.5,
                reg_lambda=2.0,
                objective='reg:squarederror',
                eval_metric='mae',
                random_state=42,
            )
            incremental_model.fit(
                X_fb, y_fb,
                xgb_model=booster,
                verbose=False,
            )

            self.risk_model.model = incremental_model
            self.retrain_count += 1

            preds = np.clip(np.round(incremental_model.predict(X_fb)), 1, 10)
            mae = mean_absolute_error(y_fb, preds)
            print(f"[IncrementalTrainer] 재학습 완료 | "
                  f"횟수={self.retrain_count} | "
                  f"피드백 MAE={mae:.3f} | "
                  f"스냅샷={snapshot_path.name}")

            updated_path = f"xgboost_risk_v{self.retrain_count}.json"
            self.risk_model.save(updated_path)

            if self.clear_after_fit:
                self.buffer.clear()
                print("[IncrementalTrainer] 버퍼 초기화 완료")

            torch.cuda.empty_cache()

        except Exception as e:
            print(f"[IncrementalTrainer] 재학습 실패: {e}")
            print(f"[IncrementalTrainer] 스냅샷({snapshot_path})으로 롤백 가능")

    def _save_snapshot(self) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.snapshot_dir / f"snapshot_{ts}_v{self.retrain_count}.json"
        self.risk_model.model.save_model(str(path))
        return path

    def rollback(self, snapshot_path: str) -> None:
        self.risk_model.load(snapshot_path)
        print(f"[IncrementalTrainer] 롤백 완료 → {snapshot_path}")

    def list_snapshots(self) -> List[str]:
        return sorted(str(p) for p in self.snapshot_dir.glob("snapshot_*.json"))


class FinalRiskPipeline:
    """
    DrugAnalyzer(약물 상호작용) 결과를 받아 XGBoost로 최종 위험도를 계산한다.
    LSTM·BERT 폐기로 인해 lstm_result/bert_result 입력은 없앴다.
    accel_anomaly는 필요 시 다른 가속도 센서 파이프라인에서 별도로 주입한다.
    """

    def __init__(self, model: RiskXGBoost,
                 min_samples: int = 30,
                 incremental_trees: int = 20):
        self.xgb = model
        self.buffer = FeedbackBuffer()
        self.trainer = IncrementalTrainer(
            risk_model=model,
            buffer=self.buffer,
            min_samples=min_samples,
            incremental_trees=incremental_trees,
        )
        self._last_features: Optional[RiskFeatures] = None

    def compute(self, drug_result: dict, accel_anomaly: float = 0.0) -> dict:
        n_drugs = len(drug_result.get('identified', {}))
        feat = RiskFeatures(
            drug_risk_score=drug_result.get('drug_risk_score', 0.0),
            n_drugs=min(n_drugs / 10.0, 1.0),
            has_major_interaction=float(any(
                i['severity'] in ('contraindicated', 'major')
                for i in drug_result.get('interactions', [])
            )),
            acceleration_anomaly=accel_anomaly,
        )
        self._last_features = feat
        result = self.xgb.predict(feat)
        result['incremental_ready'] = len(self.buffer) >= self.trainer.min_samples
        return result

    def user_feedback(
        self,
        feedback_type: str = 'false_alarm',
        corrected_risk: Optional[float] = None,
        features: Optional[RiskFeatures] = None,
    ) -> dict:
        """
        사용자 피드백 수신 -> 버퍼 적재 -> 임계치 도달 시 자동 재학습.

        feedback_type  : 'false_alarm' | 'missed' | 'manual'
        corrected_risk : None이면 false_alarm->1, missed->9 자동 할당
        features       : None이면 마지막 compute() 호출 때의 피처 재사용
        """
        feat = features or self._last_features
        if feat is None:
            return {'status': 'error', 'message': 'compute()를 먼저 호출하세요'}

        if corrected_risk is None:
            corrected_risk = 1.0 if feedback_type == 'false_alarm' else 9.0

        pred_result = self.xgb.predict(feat)
        predicted_risk = pred_result['raw_score']

        sample = FeedbackSample(
            predicted_risk=predicted_risk,
            corrected_risk=float(np.clip(corrected_risk, 1, 10)),
            features=feat.to_array().tolist(),
            feedback_type=feedback_type,
        )
        self.buffer.add(sample)

        retrained = self.trainer.try_incremental_fit()

        return {
            'status': 'retrained' if retrained else 'buffered',
            'buffer_size': len(self.buffer),
            'min_samples': self.trainer.min_samples,
            'retrain_count': self.trainer.retrain_count,
            'predicted_risk': predicted_risk,
            'corrected_risk': corrected_risk,
        }


def train_xgboost(mimic_dir: str) -> RiskXGBoost:
    setup_rtx3060()

    prep = MIMICPreprocessor(mimic_dir)
    X, y = prep.load()
    print(f"MIMIC-III: {X.shape}, 위험도={np.bincount(y)[1:]}")

    skf, maes = StratifiedKFold(5, shuffle=True, random_state=42), []
    for fold, (tr, val) in enumerate(skf.split(X, y), 1):
        m = RiskXGBoost()
        m.train(X[tr], y[tr], X[val], y[val])
        pred = np.clip(np.round(m.model.predict(X[val])), 1, 10)
        mae = mean_absolute_error(y[val], pred)
        maes.append(mae)
        print(f"  Fold {fold} MAE: {mae:.3f}")
        torch.cuda.empty_cache()

    print(f"평균 MAE: {np.mean(maes):.3f} ± {np.std(maes):.3f}")

    split = int(0.85 * len(X))
    final = RiskXGBoost()
    final.train(X[:split], y[:split], X[split:], y[split:])
    final.save(str(AI_DIR / 'xgboost_risk_final.json'))
    return final


# ============================================================
# 통합 실행
# ============================================================

def main():
    print("=" * 60)
    print("[1/2] 약물 상호작용 DB 구축")
    print("=" * 60)
    analyzer = build_drug_db()

    xgb_model = None
    mimic_dir = DATA_DIR / 'MIMIC -III (10000 patients)'
    print("=" * 60)
    print("[2/2] XGBoost 위험도 모델 학습")
    print("=" * 60)
    if not _XGBOOST_AVAILABLE:
        print("[건너뜀] xgboost 미설치 (pip install xgboost)")
    elif not mimic_dir.exists():
        print(f"[건너뜀] {mimic_dir} 없음")
    else:
        xgb_model = train_xgboost(str(mimic_dir))

    pipeline = FinalRiskPipeline(xgb_model) if xgb_model is not None else None

    print("=" * 60)
    print("완료:",
          f"drug_analyzer={'OK' if analyzer else 'X'}",
          f"xgboost={'OK' if xgb_model else 'X'}")
    print("=" * 60)
    return analyzer, pipeline


if __name__ == '__main__':
    main()
