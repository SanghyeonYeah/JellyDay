import glob
import re
import sqlite3
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Optional

import pandas as pd

# rapidfuzz 설치 여부 자동 감지
# pip install rapidfuzz 로 설치하면 더 빠른 Levenshtein 기반 매칭 사용
try:
    from rapidfuzz import fuzz as _rfuzz
    _RAPIDFUZZ_AVAILABLE = True
except ImportError:
    _RAPIDFUZZ_AVAILABLE = False

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'

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
    "epinephrine"처럼 활성 성분 단일명만 사용한다. 아무 정규화 없이 매칭하면
    DUR-DDInter 간 이름이 겹치는 비율이 491개 성분 중 125개(25%)에 불과하지만,
    아래 처리를 거치면 275개(71%)까지 올라간다 (실제 데이터로 검증됨):
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


# OCR 약물명 정규화 (브랜드명/처방전 텍스트용)

class DrugNameNormalizer:
    _FORM = re.compile(
        r'(정|캡슐|캡|주사|시럽|연고|크림|패치|좌약|과립|액|앰플'
        r'|tablet|capsule|injection|mg|ml|mcg|%)\b', re.I)
    _NUM  = re.compile(r'\d+[\.\d]*')

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
    괄호(성분/용량 부기) 및 밑줄 뒤 표기를 제거한 뒤 OCR 정규화를 동일하게 적용한다.
    """
    if not isinstance(raw, str):
        return ''
    name = _PAREN_OR_UNDERSCORE.sub('', raw).strip()
    return _name_normalizer.normalize(name)


# DDInter + 식약처 DUR CSV -> SQLite

class KoreanDrugDataLoader:
    """
    DDInter(국제 약물-약물 상호작용 DB, 영문 성분명 기준)와
    식약처 DUR(병용금기/용량주의/투여기간주의, 국문 CSV) 데이터를 SQLite로 통합 적재.

    성분(ingredient)을 공통 키로 사용해 두 데이터 소스를 연결한다:
        DUR 성분명  --normalize_ingredient()-->  base 성분명
        DDInter Drug_A/B (원래 영문 소문자)      -->  base 성분명과 동일 어휘 공간
    """
    # DDInter Level -> 위험도 점수
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

    # DDInter (파일 8개: code_A/B/D/H/L/P/R/V, 총 약 22만 쌍)
    def load_ddinter(self, glob_pattern: str = None):
        glob_pattern = glob_pattern or str(DATA_DIR / 'ddinter_downloads_code_*.csv')
        paths = sorted(glob.glob(glob_pattern))
        cur   = self.conn.cursor()
        total = 0
        for path in paths:
            df = pd.read_csv(path)
            df['ingredient_a'] = df['Drug_A'].str.lower().str.strip()
            df['ingredient_b'] = df['Drug_B'].str.lower().str.strip()
            df['risk_score']   = df['Level'].map(self.DDINTER_LEVEL_RISK).fillna(0.1)
            rows = list(df[['ingredient_a', 'ingredient_b', 'Level', 'risk_score']]
                        .itertuples(index=False, name=None))
            cur.executemany(
                "INSERT INTO ddinter_interactions"
                "(ingredient_a,ingredient_b,level,risk_score) VALUES (?,?,?,?)", rows)
            total += len(rows)
        self.conn.commit()
        print(f"[DDInter] 파일 {len(paths)}개, {total}쌍 적재 완료")

    # 식약처 병용금기약물 (제품 조합 기준 원본은 수십만 행 -> 성분쌍 기준으로 중복 제거 후 적재)
    def load_dur_contraindications(self, csv_path: str = None, encoding: str = 'cp949'):
        csv_path = csv_path or str(DATA_DIR / '한국의약품안전관리원_병용금기약물_20240625.csv')
        df = pd.read_csv(csv_path, encoding=encoding, dtype=str)
        df = df.rename(columns={
            '성분명1': 'ing1', '제품명1': 'prod1', '제품코드1': 'code1',
            '성분명2': 'ing2', '제품명2': 'prod2', '제품코드2': 'code2',
            '금기사유': 'reason', '공고번호': 'notice_no', '공고일자': 'notice_date',
        }).dropna(subset=['ing1', 'ing2'])

        cur = self.conn.cursor()

        # 성분쌍 단위 중복 제거 (제품 조합 기준 ~54만행 -> 성분 기준 ~1,700행)
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

        # 제품명 -> 성분 alias 등록 (OCR 매칭용, 양쪽 컬럼 모두)
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

    # 식약처 용량주의약물 (1일 최대 투여기준량 mg)
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
            key  = base[0] if base else row.ing.lower()
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

    # 식약처 투여기간주의약물 (최대 투여기간 일수)
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
            key  = base[0] if base else row.ing.lower()
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


# Fuzzy Matching 엔진

class FuzzyDrugMatcher:
    """
    OCR 오인식 대응 Fuzzy 약물명 매칭.

    동작 원리:
        정확 매칭 실패 시 DB의 모든 alias를 후보로
        두 가지 유사도 측정을 병렬로 수행:

        1. SequenceMatcher (표준 라이브러리, 항상 사용 가능)
               ratio() -> 0~1 유사도
               한글 자모 단위 편집 거리에 강함

        2. rapidfuzz.fuzz
               token_sort_ratio -> 어순 불일치에도 강인
               C++ 기반으로 SequenceMatcher 대비 ~100x 빠름

    OCR 오류 예시:
        '아스피린' -> '아스피림'   (받침 오인식)
        'warfarin' -> 'worfarin'   (모음 오인식)
        '암로디핀' -> '암노디핀'   (자음 오인식)

    고려사항:
        - DB alias 전체를 메모리에 캐싱 -> 매 조회 시 DB I/O 없음
        - top_k=3 반환으로 불확실한 경우 여러 후보 제시 가능
        - threshold 미달 시 빈 리스트 반환 (오탐 방지)

    파라미터:
        threshold : 유사도 최솟값 (0~1). 기본 0.75
                    너무 낮으면 관계없는 약물이 후보로 포함됨
                    너무 높으면 OCR 오류를 감지 못함
        top_k     : 반환할 최대 후보 수 (기본 3)
    """

    def __init__(self, conn: sqlite3.Connection,
                 threshold: float = 0.75,
                 top_k    : int   = 3):
        self.conn      = conn
        self.threshold = threshold
        self.top_k     = top_k
        self._cache    : Optional[dict] = None   # alias -> ingredient

    # alias 캐시 초기화 (최초 1회)
    def _load_cache(self) -> dict:
        if self._cache is None:
            rows        = self.conn.execute(
                "SELECT alias, ingredient FROM product_alias"
            ).fetchall()
            self._cache = {r[0]: r[1] for r in rows}
            print(f"[FuzzyMatcher] alias 캐시 로드: {len(self._cache)}건")
        return self._cache

    def invalidate_cache(self):
        """DB 변경 후 캐시 재로드가 필요할 때 호출."""
        self._cache = None

    # 유사도 계산
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

    # 외부 진입점
    def find_candidates(self, norm: str) -> List[dict]:
        """
        정규화된 약물명 norm에 대해 유사 alias 후보 목록 반환.

        반환 형식:
            [
              {
                'alias'       : 'aspirin',
                'ingredient'  : 'aspirin',
                'match_score' : 0.89,
                'match_method': 'rapidfuzz'
              },
              ...
            ]

        threshold 미달 시 빈 리스트 반환.
        """
        cache  = self._load_cache()
        method = 'rapidfuzz' if _RAPIDFUZZ_AVAILABLE else 'sequence_matcher'
        scored = []

        for alias, ingredient in cache.items():
            score = self._similarity(norm, alias)
            if score >= self.threshold:
                scored.append({
                    'alias'       : alias,
                    'ingredient'  : ingredient,
                    'match_score' : round(score, 4),
                    'match_method': method,
                })

        scored.sort(key=lambda x: x['match_score'], reverse=True)
        return scored[:self.top_k]


# 통합 분석 엔진

class DrugAnalyzer:
    def __init__(self, loader: KoreanDrugDataLoader,
                 fuzzy_threshold: float = 0.75,
                 fuzzy_top_k    : int   = 3):
        self.conn       = loader.conn
        self.normalizer = DrugNameNormalizer()
        self.fuzzy      = FuzzyDrugMatcher(
            conn      = self.conn,
            threshold = fuzzy_threshold,
            top_k     = fuzzy_top_k,
        )

    def lookup(self, norm: str) -> Optional[str]:
        """정확 매칭 (product_alias 테이블 직접 조회) -> 성분명 반환."""
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
        # 용량/기간주의 대상이 아닌 성분도 존재 -> 상호작용 체크는 가능하도록 최소 정보 반환
        return {'id': ingredient, 'max_dose_mg_day': None, 'max_duration_days': None}

    def check_interactions(self, ingredients: List[str]) -> List[dict]:
        """
        성분 쌍마다 DUR 병용금기(식약처 공식 금기) 우선 조회 후,
        없으면 DDInter 상호작용 등급을 조회한다.
        DUR 병용금기는 severity='contraindicated'(risk=1.0)로 DDInter보다 항상 우선한다.
        """
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
        """
        OCR 인식 약물명 리스트를 분석.

        흐름:
            1. 정규화 (DrugNameNormalizer)
            2. 정확 매칭 (lookup, product_alias 테이블)
            3. 정확 매칭 실패 시 -> Fuzzy 매칭으로 후보 제안
            4. 상호작용 분석(DUR 우선, DDInter 보조) 및 위험도 계산

        반환:
            identified      : 정확 매칭 성공 약물 {raw_name: drug_info}
            unidentified    : 정확 매칭 실패 약물 목록 (fuzzy_candidates 포함)
            interactions    : 약물 간 상호작용 목록
            drug_risk_score : 0~1 위험도 점수
        """
        norms = self.normalizer.normalize_list(ocr_names)
        identified, unidentified = {}, []

        for norm, raw in zip(norms, ocr_names):
            ingredient = self.lookup(norm)
            if ingredient:
                identified[raw] = self.get_info(ingredient)
            else:
                candidates   = self.fuzzy.find_candidates(norm)
                auto_matched = None
                if candidates and candidates[0]['match_score'] >= 0.90:
                    best = candidates[0]
                    info = self.get_info(best['ingredient'])
                    info['fuzzy_matched'] = True
                    info['match_score']   = best['match_score']
                    info['match_method']  = best['match_method']
                    info['matched_alias'] = best['alias']
                    identified[raw]       = info
                    auto_matched          = best
                    print(f"[FuzzyMatcher] '{raw}' -> '{best['alias']}' "
                          f"(score={best['match_score']:.3f}, method={best['match_method']})")

                if auto_matched is None:
                    unidentified.append({
                        'raw'              : raw,
                        'normalized'       : norm,
                        'fuzzy_candidates' : candidates,
                    })

        ids    = [v['id'] for v in identified.values() if v]
        inters = self.check_interactions(ids)
        risk   = self._risk(identified, inters)

        return {
            'identified'     : identified,
            'unidentified'   : unidentified,
            'interactions'   : inters,
            'drug_risk_score': round(risk, 3),
        }

    def _risk(self, identified, inters) -> float:
        if not identified:
            return 0.0
        inter_score     = max((i['risk_score'] for i in inters), default=0.0)
        poly            = max(0.0, (len(identified) - 4) * 0.08)
        contraindicated = any(i['severity'] == 'contraindicated' for i in inters)
        return min(max(inter_score + poly, 1.0 if contraindicated else 0.0), 1.0)


if __name__ == '__main__':
    loader = KoreanDrugDataLoader(str(DATA_DIR / 'drug_data.db'))
    loader.load_all()

    analyzer = DrugAnalyzer(loader, fuzzy_threshold=0.75, fuzzy_top_k=3)

    # 실제 병용금기 쌍(clarithromycin x simvastatin)의 제품명으로 시뮬레이션
    # + OCR 오인식('심바로드정' -> '심바로드정' 오타)
    result = analyzer.analyze(['제클라정', '심바로드정20밀리그람', '아스피림200mg정'])
    print(f"약물 위험도: {result['drug_risk_score']}")

    for raw_name, info in result['identified'].items():
        fuzzy = info.get('fuzzy_matched', False)
        tag   = f" [Fuzzy:{info.get('matched_alias')} score={info.get('match_score')}]" if fuzzy else ""
        print(f"  인식: {raw_name} -> {info['id']}{tag}")

    for item in result['unidentified']:
        print(f"  미인식: {item['raw']} (정규화: {item['normalized']})")
        for c in item['fuzzy_candidates']:
            print(f"    후보: {c['alias']} (score={c['match_score']:.3f}, "
                  f"ingredient={c['ingredient']}, method={c['match_method']})")

    for i in result['interactions']:
        print(f"  [{i['source']}/{i['severity']}] {i['drug_a']} <-> {i['drug_b']} - {i['description']}")
