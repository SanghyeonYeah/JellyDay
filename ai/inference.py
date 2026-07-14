"""
JellyDay 추론 모듈 — 앱에서 실제로 쓰는 런타임 전용 코드.

학습/DB 구축(CSV 파싱, DDI 모델 학습)은 train_pipeline.py가 담당한다.
이 모듈은 이미 만들어진 drug_data.db(+ 선택적으로 ddi_model.pt)를 불러와
약물 상호작용을 조회/추정하는 데만 쓴다. 앱 배포 시엔 이 파일 + drug_data.db
(+ ddi_model.pt)만 있으면 되고, train_pipeline.py는 필요 없다.

사용 예:
    from inference import load_pipeline
    analyzer = load_pipeline('data/drug_data.db', 'ai/ddi_model.pt')
    result = analyzer.analyze(ocr_names)   # OCR 결과 -> 식별/상호작용/위험도
"""

import re
import sqlite3
import unicodedata
from difflib import SequenceMatcher
from typing import List, Optional

import torch

try:
    from rapidfuzz import fuzz as _rfuzz
    _RAPIDFUZZ_AVAILABLE = True
except ImportError:
    _RAPIDFUZZ_AVAILABLE = False


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


class DrugAnalyzer:
    def __init__(self, conn: sqlite3.Connection,
                 fuzzy_threshold: float = 0.75,
                 fuzzy_top_k: int = 3,
                 ddi_predictor: Optional[DDIRiskPredictor] = None):
        self.conn = conn
        self.normalizer = DrugNameNormalizer()
        self.fuzzy = FuzzyDrugMatcher(
            conn=self.conn,
            threshold=fuzzy_threshold,
            top_k=fuzzy_top_k,
        )
        # DB에 없는 미등록 약물 조합의 위험도를 추정하는 학습된 임베딩 모델(선택).
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


def load_pipeline(db_path: str, ddi_model_path: Optional[str] = None) -> DrugAnalyzer:
    """
    이미 만들어진 drug_data.db(+ 선택적으로 ddi_model.pt)를 불러와
    바로 쓸 수 있는 DrugAnalyzer를 반환한다. 앱에서는 이 함수 하나만 부르면 된다.
    """
    conn = sqlite3.connect(db_path, check_same_thread=False)
    analyzer = DrugAnalyzer(conn)
    if ddi_model_path:
        analyzer.ddi_predictor = DDIRiskPredictor.load(ddi_model_path)
    return analyzer
