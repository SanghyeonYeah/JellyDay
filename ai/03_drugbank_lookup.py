import sqlite3
import re
import json
import unicodedata
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
from typing import List, Optional

NS = 'http://www.drugbank.ca'

# rapidfuzz 설치 여부 자동 감지
# pip install rapidfuzz 로 설치하면 더 빠른 Levenshtein 기반 매칭 사용
try:
    from rapidfuzz import fuzz as _rfuzz
    _RAPIDFUZZ_AVAILABLE = True
except ImportError:
    _RAPIDFUZZ_AVAILABLE = False

# DrugBank XML -> SQLite

class DrugBankLoader:
    def __init__(self, db_path: str = 'drugbank.db'):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute('PRAGMA journal_mode=WAL')
        self.conn.execute('PRAGMA synchronous=NORMAL')
        self.conn.execute('PRAGMA cache_size=-65536')
        self._init_tables()

    def _init_tables(self):
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS drugs (
            drugbank_id     TEXT PRIMARY KEY,
            name_en         TEXT NOT NULL,
            name_ko         TEXT,
            atc_code        TEXT,
            max_dose_mg_day REAL,
            half_life_hours REAL
        );
        CREATE TABLE IF NOT EXISTS ingredients (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            drugbank_id TEXT,
            ingredient  TEXT,
            FOREIGN KEY(drugbank_id) REFERENCES drugs(drugbank_id)
        );
        CREATE TABLE IF NOT EXISTS drug_names_alias (
            alias       TEXT PRIMARY KEY,
            drugbank_id TEXT,
            FOREIGN KEY(drugbank_id) REFERENCES drugs(drugbank_id)
        );
        CREATE TABLE IF NOT EXISTS interactions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            drug_a_id   TEXT,
            drug_b_id   TEXT,
            severity    TEXT,
            description TEXT,
            risk_score  REAL
        );
        CREATE INDEX IF NOT EXISTS idx_alias   ON drug_names_alias(alias);
        CREATE INDEX IF NOT EXISTS idx_inter_a ON interactions(drug_a_id);
        CREATE INDEX IF NOT EXISTS idx_inter_b ON interactions(drug_b_id);
        """)
        self.conn.commit()

    def build_from_xml(self, xml_path: str, commit_every: int = 500):
        """iterparse 스트리밍 파싱 -> RAM 사용 최소화."""
        print(f"DrugBank XML 파싱: {xml_path}")
        cur   = self.conn.cursor()
        count = 0
        SEVERITY = {'major':1.0,'serious':0.9,'moderate':0.55,'minor':0.2}

        for _, elem in ET.iterparse(xml_path, events=('end',)):
            if elem.tag != f'{{{NS}}}drug':
                continue
            if elem.attrib.get('type') != 'small molecule':
                elem.clear(); continue

            did = self._text(elem, f'{{{NS}}}drugbank-id[@primary="true"]')
            if not did:
                elem.clear(); continue

            name_en = self._text(elem, f'{{{NS}}}name') or ''
            atc     = self._text(elem, f'{{{NS}}}atc-codes/{{{NS}}}atc-code') or ''
            hl_raw  = self._text(elem, f'{{{NS}}}half-life') or ''
            hl      = float(re.search(r'([\d.]+)', hl_raw).group(1)) if re.search(r'([\d.]+)', hl_raw) else None

            max_dose = None
            for d in elem.findall(f'{{{NS}}}dosages/{{{NS}}}dosage'):
                s = self._text(d, f'{{{NS}}}strength') or ''
                m = re.search(r'([\d.]+)\s*mg', s)
                if m:
                    max_dose = max(max_dose or 0, float(m.group(1)))

            cur.execute("INSERT OR IGNORE INTO drugs VALUES (?,?,?,?,?,?)",
                        (did, name_en, None, atc, max_dose, hl))

            for p in elem.findall(f'{{{NS}}}international-brands/{{{NS}}}international-brand'):
                b = self._text(p, f'{{{NS}}}name')
                if b:
                    cur.execute("INSERT OR IGNORE INTO drug_names_alias VALUES (?,?)",
                                (b.lower(), did))

            for p in elem.findall(f'{{{NS}}}calculated-properties/{{{NS}}}property'):
                if self._text(p, f'{{{NS}}}kind') == 'IUPAC Name':
                    v = self._text(p, f'{{{NS}}}value')
                    if v:
                        cur.execute("INSERT OR IGNORE INTO ingredients(drugbank_id,ingredient) VALUES (?,?)",
                                    (did, v.lower()))

            for inter in elem.findall(f'{{{NS}}}drug-interactions/{{{NS}}}drug-interaction'):
                oid  = self._text(inter, f'{{{NS}}}drugbank-id')
                desc = self._text(inter, f'{{{NS}}}description') or ''
                dl   = desc.lower()
                sev, risk = 'minor', 0.2
                for kw, sc in SEVERITY.items():
                    if kw in dl:
                        sev, risk = kw, sc; break
                if oid:
                    cur.execute(
                        "INSERT OR IGNORE INTO interactions"
                        "(drug_a_id,drug_b_id,severity,description,risk_score) VALUES (?,?,?,?,?)",
                        (did, oid, sev, desc, risk))

            elem.clear()
            count += 1
            if count % commit_every == 0:
                self.conn.commit()
                print(f"  {count}개 처리...")

        self.conn.commit()
        print(f"완료: {count}개 적재")

    def add_korean_alias(self, csv_path: str):
        import pandas as pd
        df  = pd.read_csv(csv_path)
        cur = self.conn.cursor()
        for _, row in df.iterrows():
            cur.execute("INSERT OR IGNORE INTO drug_names_alias VALUES (?,?)",
                        (row['alias_ko'].lower().strip(), row['drugbank_id']))
        self.conn.commit()
        print(f"한국어 alias {len(df)}건 추가")

    @staticmethod
    def _text(elem, xpath) -> Optional[str]:
        n = elem.find(xpath)
        return n.text.strip() if n is not None and n.text else None

# OCR 약물명 정규화

class DrugNameNormalizer:
    _FORM = re.compile(
        r'(정|캡슐|캡|주사|시럽|연고|크림|패치|좌약|과립|액|앰플'
        r'|tablet|capsule|injection|mg|ml|mcg|%)\b', re.I)
    _NUM  = re.compile(r'\d+[\.\d]*')

    def normalize(self, raw: str) -> str:
        t = unicodedata.normalize('NFC', raw.strip())
        t = self._FORM.sub('', t)
        t = self._NUM.sub('', t)
        return re.sub(r'\s+', ' ', t).strip().lower()

    def normalize_list(self, raws: List[str]) -> List[str]:
        return [self.normalize(r) for r in raws]

# 3. Fuzzy Matching 엔진

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
        self._cache    : dict[str, str] | None = None   # alias -> drugbank_id

    # alias 캐시 초기화 (최초 1회)
    def _load_cache(self) -> dict[str, str]:
        if self._cache is None:
            rows         = self.conn.execute(
                "SELECT alias, drugbank_id FROM drug_names_alias"
            ).fetchall()
            self._cache  = {r[0]: r[1] for r in rows}
            print(f"[FuzzyMatcher] alias 캐시 로드: {len(self._cache)}건")
        return self._cache

    def invalidate_cache(self):
        """add_korean_alias 등 DB 변경 후 캐시 재로드가 필요할 때 호출."""
        self._cache = None

    # 유사도 계산
    @staticmethod
    def _similarity_sequence(a: str, b: str) -> float:
        """SequenceMatcher 기반 유사도 (표준 라이브러리)."""
        return SequenceMatcher(None, a, b).ratio()

    @staticmethod
    def _similarity_rapidfuzz(a: str, b: str) -> float:
        """rapidfuzz token_sort_ratio (설치 시 사용). 0~100 -> 0~1 정규화."""
        return _rfuzz.token_sort_ratio(a, b) / 100.0

    def _similarity(self, a: str, b: str) -> float:
        """
        rapidfuzz 설치 여부에 따라 자동 선택.
        두 점수 중 높은 값을 반환하여 다양한 OCR 패턴에 강인하게 대응.
        """
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
                'alias'       : 'aspirin',       # DB에 있는 alias
                'drugbank_id' : 'DB00945',
                'match_score' : 0.89,            # 0~1 유사도
                'match_method': 'rapidfuzz'      # 또는 'sequence_matcher'
              },
              ...
            ]

        threshold 미달 시 빈 리스트 반환.
        """
        cache    = self._load_cache()
        method   = 'rapidfuzz' if _RAPIDFUZZ_AVAILABLE else 'sequence_matcher'
        scored   = []

        for alias, did in cache.items():
            score = self._similarity(norm, alias)
            if score >= self.threshold:
                scored.append({
                    'alias'       : alias,
                    'drugbank_id' : did,
                    'match_score' : round(score, 4),
                    'match_method': method,
                })

        # 유사도 내림차순 정렬 후 top_k 반환
        scored.sort(key=lambda x: x['match_score'], reverse=True)
        return scored[:self.top_k]

# DB Lookup 엔진

class DrugAnalyzer:
    SEV = {'major':1.0,'serious':0.9,'moderate':0.55,'minor':0.2}

    def __init__(self, loader: DrugBankLoader,
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
        """정확 매칭 (alias 테이블 직접 조회)."""
        row = self.conn.execute(
            "SELECT drugbank_id FROM drug_names_alias WHERE alias=?", (norm,)
        ).fetchone()
        return row[0] if row else None

    def get_info(self, did: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM drugs WHERE drugbank_id=?", (did,)
        ).fetchone()
        if not row: return None
        ings = [r[0] for r in self.conn.execute(
            "SELECT ingredient FROM ingredients WHERE drugbank_id=?", (did,)
        ).fetchall()]
        return {'id':row[0],'name_en':row[1],'name_ko':row[2],
                'atc_code':row[3],'max_dose':row[4],'half_life':row[5],'ingredients':ings}

    def check_interactions(self, ids: List[str]) -> List[dict]:
        out = []
        for i, a in enumerate(ids):
            for b in ids[i+1:]:
                row = self.conn.execute(
                    "SELECT severity,description,risk_score FROM interactions"
                    " WHERE (drug_a_id=? AND drug_b_id=?) OR (drug_a_id=? AND drug_b_id=?)",
                    (a,b,b,a)
                ).fetchone()
                if row:
                    out.append({'drug_a':a,'drug_b':b,
                                'severity':row[0],'description':row[1],'risk_score':row[2]})
        return out

    def analyze(self, ocr_names: List[str]) -> dict:
        """
        OCR 인식 약물명 리스트를 분석.

        흐름:
            1. 정규화 (DrugNameNormalizer)
            2. 정확 매칭 (lookup)
            3. 정확 매칭 실패 시 -> Fuzzy 매칭으로 후보 제안
            4. 상호작용 분석 및 위험도 계산

        반환:
            identified      : 정확 매칭 성공 약물 {raw_name: drug_info}
            unidentified    : 정확 매칭 실패 약물 목록
                              각 항목에 'fuzzy_candidates' 포함
            interactions    : 약물 간 상호작용 목록
            drug_risk_score : 0~1 위험도 점수
        """
        norms = self.normalizer.normalize_list(ocr_names)
        identified, unidentified = {}, []

        for norm, raw in zip(norms, ocr_names):
            did = self.lookup(norm)
            if did:
                # 정확 매칭 성공
                identified[raw] = self.get_info(did)
            else:
                # 정확 매칭 실패 -> Fuzzy 후보 탐색
                candidates = self.fuzzy.find_candidates(norm)

                # 최고 점수 후보가 threshold 이상이면 자동 적용 (신뢰도 보고 포함)
                auto_matched = None
                if candidates and candidates[0]['match_score'] >= 0.90:
                    best = candidates[0]
                    info = self.get_info(best['drugbank_id'])
                    if info:
                        info['fuzzy_matched']  = True
                        info['match_score']    = best['match_score']
                        info['match_method']   = best['match_method']
                        info['matched_alias']  = best['alias']
                        identified[raw]        = info
                        auto_matched           = best
                        print(f"[FuzzyMatcher] '{raw}' -> '{best['alias']}' "
                              f"(score={best['match_score']:.3f}, "
                              f"method={best['match_method']})")

                if auto_matched is None:
                    # 자동 적용 불가 -> 후보만 제안
                    unidentified.append({
                        'raw'              : raw,
                        'normalized'       : norm,
                        'fuzzy_candidates' : candidates,  # 사용자/상위 레이어에 제시
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
        if not identified: return 0.0
        inter_score = max((self.SEV.get(i['severity'],0) for i in inters), default=0.0)
        poly        = max(0.0, (len(identified)-4) * 0.08)
        major       = any(i['severity'] in ('major','serious') for i in inters)
        return min(max(inter_score+poly, 0.6 if major else 0.0), 1.0)


if __name__ == '__main__':
    loader = DrugBankLoader('drugbank.db')
    loader.build_from_xml('full_database.xml')
    loader.add_korean_alias('korean_aliases.csv')

    analyzer = DrugAnalyzer(loader, fuzzy_threshold=0.75, fuzzy_top_k=3)

    # OCR 오인식 시뮬레이션: '아스피린' -> '아스피림', 'warfarin' -> 'worfarin'
    result = analyzer.analyze(['암로디핀5mg정', '와파린2mg', '아스피림200mg정'])
    print(f"약물 위험도: {result['drug_risk_score']}")

    for raw_name, info in result['identified'].items():
        fuzzy = info.get('fuzzy_matched', False)
        tag   = f" [Fuzzy:{info.get('matched_alias')} score={info.get('match_score')}]" if fuzzy else ""
        print(f"  인식: {raw_name} -> {info['name_en']}{tag}")

    for item in result['unidentified']:
        print(f"  미인식: {item['raw']} (정규화: {item['normalized']})")
        for c in item['fuzzy_candidates']:
            print(f"    후보: {c['alias']} (score={c['match_score']:.3f}, "
                  f"id={c['drugbank_id']}, method={c['match_method']})")

    for i in result['interactions']:
        print(f"  [{i['severity']}] {i['drug_a']} ↔ {i['drug_b']}")
