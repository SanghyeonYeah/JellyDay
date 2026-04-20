import json
import numpy as np
import pandas as pd
import xgboost as xgb
from collections import deque
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.utils.class_weight import compute_sample_weight
from typing import List, Optional, Deque

import torch
from gpu_config import setup_rtx3060, print_vram

# 피처 정의

@dataclass
class RiskFeatures:
    pattern_anomaly_score    : float
    pattern_variability      : float
    zone_transition_abnormal : float
    time_in_unusual_zone     : float
    intent_risk_score        : float
    pain_score               : float
    distress_score           : float
    emergency_score          : float
    drug_risk_score          : float
    n_drugs                  : float   # / 10
    has_major_interaction    : float
    anomaly_composite        : float
    acceleration_anomaly     : float

    def to_array(self) -> np.ndarray:
        return np.array([
            self.pattern_anomaly_score, self.pattern_variability,
            self.zone_transition_abnormal, self.time_in_unusual_zone,
            self.intent_risk_score, self.pain_score,
            self.distress_score, self.emergency_score,
            self.drug_risk_score, self.n_drugs,
            self.has_major_interaction,
            self.anomaly_composite, self.acceleration_anomaly,
        ], dtype=np.float32)

    @staticmethod
    def names() -> List[str]:
        return [
            'pattern_anomaly','pattern_variability',
            'zone_transition_abnormal','time_in_unusual_zone',
            'intent_risk','pain','distress','emergency',
            'drug_risk','n_drugs','has_major_interaction',
            'anomaly_composite','acceleration_anomaly',
        ]

# 피드백 버퍼

@dataclass
class FeedbackSample:
    """
    사용자가 오탐을 교정한 단일 샘플.

    predicted_risk : XGBoost가 예측한 위험도
    corrected_risk : 사용자 피드백으로 보정된 위험도
    features       : 해당 시점의 RiskFeatures 배열 (길이 13)
    timestamp      : 피드백 수신 시각
    feedback_type  : 'false_alarm'(오탐) | 'missed'(미탐) | 'manual'(수동 입력)
    """
    predicted_risk : float
    corrected_risk : float
    features       : List[float]
    timestamp      : str = field(default_factory=lambda: datetime.now().isoformat())
    feedback_type  : str = 'false_alarm'


class FeedbackBuffer:
    """
    사용자 피드백 샘플을 누적하는 순환 큐.

    설계:
      - maxlen으로 오래된 피드백 자동 만료 (오래된 행동 패턴 희석 방지)
      - persist_path 지정 시 JSONL로 영속화 → 재시작 후에도 복원 가능
      - get_batch()로 numpy 배열 변환 후 IncrementalTrainer에 전달
    """
    def __init__(self, maxlen: int = 500, persist_path: str = "feedback_buffer.jsonl"):
        self._buf          : Deque[FeedbackSample] = deque(maxlen=maxlen)
        self.persist_path  = Path(persist_path)
        self._load()          # 이전 세션 피드백 복원

    # 외부 인터페이스
    def add(self, sample: FeedbackSample) -> None:
        self._buf.append(sample)
        self._append_jsonl(sample)
        print(f"[FeedbackBuffer] 샘플 추가 | 누적={len(self._buf)} "
              f"| type={sample.feedback_type} "
              f"| predicted={sample.predicted_risk:.1f} → corrected={sample.corrected_risk:.1f}")

    def __len__(self) -> int:
        return len(self._buf)

    def get_batch(self) -> tuple[np.ndarray, np.ndarray]:
        """버퍼 전체를 (X, y) numpy 배열로 반환."""
        X = np.array([s.features       for s in self._buf], dtype=np.float32)
        y = np.array([s.corrected_risk for s in self._buf], dtype=np.float32)
        return X, y

    def clear(self) -> None:
        self._buf.clear()

    # 영속화
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

# 3. MIMIC-III 전처리

class MIMICPreprocessor:
    VITAL_ITEMS = {211:'hr', 51:'sbp', 8368:'dbp', 646:'spo2', 615:'rr'}
    TYPE_RISK   = {'ELECTIVE':2, 'NEWBORN':1, 'URGENT':6, 'EMERGENCY':9}

    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load(self) -> tuple:
        admissions    = pd.read_csv(f'{self.data_dir}/ADMISSIONS.csv',
                                    usecols=['HADM_ID','ADMISSION_TYPE','DIAGNOSIS'])
        chartevents   = pd.read_csv(f'{self.data_dir}/CHARTEVENTS.csv',
                                    usecols=['HADM_ID','ITEMID','VALUENUM'])
        prescriptions = pd.read_csv(f'{self.data_dir}/PRESCRIPTIONS.csv',
                                    usecols=['HADM_ID','DRUG','DOSE_VAL_RX'])

        vitals = chartevents[chartevents['ITEMID'].isin(self.VITAL_ITEMS)].copy()
        vitals['vital'] = vitals['ITEMID'].map(self.VITAL_ITEMS)
        vp = vitals.groupby(['HADM_ID','vital'])['VALUENUM'].mean().unstack(fill_value=np.nan)

        dc = prescriptions.groupby('HADM_ID')['DRUG'].nunique().rename('n_drugs')

        df = admissions.join(vp, on='HADM_ID', how='left').join(dc, on='HADM_ID', how='left')
        df['n_drugs'] = df['n_drugs'].fillna(0)
        df['risk']    = df['ADMISSION_TYPE'].map(self.TYPE_RISK).fillna(5).astype(int)
        df = self._adjust_diagnosis(df)

        X     = self._features(df)
        y     = df['risk'].clip(1, 10).values
        valid = ~np.isnan(X).any(axis=1)
        return X[valid].astype(np.float32), y[valid]

    def _adjust_diagnosis(self, df):
        keywords = ['sepsis','cardiac arrest','respiratory failure',
                    'stroke','acute myocardial','hemorrhage']
        diag = df['DIAGNOSIS'].str.lower().fillna('')
        for kw in keywords:
            df.loc[diag.str.contains(kw), 'risk'] = \
                df.loc[diag.str.contains(kw), 'risk'].clip(lower=8)
        return df

    def _features(self, df):
        def norm(col, lo, hi):
            if col not in df.columns:
                return np.zeros(len(df))
            v = df[col].values.astype(float)
            return np.clip(np.where(v < lo, (lo-v)/lo,
                           np.where(v > hi, (v-hi)/hi, 0.0)), 0, 1)

        hr   = norm('hr',   60, 100)
        sbp  = norm('sbp',  90, 140)
        dbp  = norm('dbp',  60,  90)
        spo2 = norm('spo2', 95, 100)
        rr   = norm('rr',   12,  20)
        nd   = np.clip(df['n_drugs'].values / 10.0, 0, 1)
        comp = (hr + sbp + spo2 + rr) / 4.0

        return np.column_stack([
            comp, rr, sbp, dbp, spo2, hr, rr, comp,
            np.zeros(len(df)), nd, np.zeros(len(df)), comp, hr,
        ])

# XGBoost

class RiskXGBoost:
    def __init__(self):
        self.model = xgb.XGBRegressor(
            device      = 'cuda',
            tree_method = 'hist',
            max_bin     = 256,

            n_estimators          = 500,
            max_depth             = 6,
            learning_rate         = 0.05,
            subsample             = 0.8,
            colsample_bytree      = 0.8,
            min_child_weight      = 3,
            reg_alpha             = 0.1,
            reg_lambda            = 1.0,
            gamma                 = 0.1,
            objective             = 'reg:squarederror',
            eval_metric           = 'mae',
            early_stopping_rounds = 30,
            random_state          = 42,
        )
        self.is_trained = False

    def train(self, X_tr, y_tr, X_val, y_val):
        print_vram('XGBoost 학습 전')
        self.model.fit(
            X_tr, y_tr,
            eval_set      = [(X_val, y_val)],
            sample_weight = compute_sample_weight('balanced', y_tr.astype(int)),
            verbose       = 50,
        )
        self.is_trained = True
        torch.cuda.empty_cache()
        print_vram('XGBoost 학습 후')

    def predict(self, features: RiskFeatures) -> dict:
        x     = features.to_array().reshape(1, -1)
        raw   = float(self.model.predict(x)[0])
        level = int(np.clip(round(raw), 1, 10))
        return {
            'risk_level'   : level,
            'raw_score'    : round(raw, 3),
            'action'       : self._action(level),
            'top_features' : self._top_features(x),
        }

    def _action(self, level: int) -> dict:
        if level <= 2:
            return {'type':'normal',         'notify':False, 'relearn':True}
        elif level <= 4:
            return {'type':'monitor',        'notify':False, 'ask_user':True}
        elif level <= 6:
            return {'type':'alert_mild',     'notify':True,  'escalate':False}
        elif level <= 8:
            return {'type':'alert_moderate', 'notify':True,  'contact':'caregiver'}
        else:
            return {'type':'emergency',      'notify':True,  'contact':'119'}

    def _top_features(self, x, n=3) -> list:
        imp  = self.model.feature_importances_
        idx  = np.argsort(imp)[::-1][:n]
        names = RiskFeatures.names()
        return [{'feature':names[i], 'importance':round(float(imp[i]),3),
                 'value':round(float(x[0,i]),3)} for i in idx]

    def save(self, path: str):
        self.model.save_model(path)
        print(f"저장 완료: {path}")

    def load(self, path: str):
        self.model.load_model(path)
        self.is_trained = True

# 온라인(증분) 학습 트레이너

class IncrementalTrainer:
    """
    FeedbackBuffer가 min_samples 이상 쌓이면 기존 XGBoost 모델에
    새 트리를 추가(boosting 계속)하여 온라인 학습을 수행한다.

    동작 원리:
    XGBoost는 xgb_model 파라미터로 기존 모델을 전달하면
    해당 모델의 트리를 그대로 유지하면서 추가 트리만 학습한다.
    즉 기존 지식을 잃지 않고(catastrophic forgetting 없이)
    새 피드백 패턴을 점진적으로 흡수한다.

    재학습 흐름:
    1. buffer.get_batch()   -> 피드백 (X_fb, y_fb) 추출
    2. 기존 모델로 예측      -> residual = y_fb - y_pred  (잔차)
    3. XGBRegressor(새 인스턴스, n_estimators=incremental_trees)
       .fit(..., xgb_model=기존_booster)
       -> 잔차를 줄이는 방향으로 추가 트리 학습
    4. 스냅샷 저장(타임스탬프) -> 롤백 가능
    5. 버퍼 초기화(선택적)

    파라미터:
    min_samples       : 재학습 트리거 임계치 (기본 30)
                        너무 낮으면 과적합, 너무 높으면 반응 지연
    incremental_trees : 1회 재학습 시 추가할 트리 수 (기본 20)
                        기존 500트리 대비 ~4% 추가로 안전한 수준
    feedback_lr       : 피드백 트리 학습률. 초기 학습률보다 낮게 설정해
                        기존 모델을 과도하게 덮어쓰지 않음
    snapshot_dir      : 재학습 전 모델 스냅샷 저장 경로 (롤백 용도)
    clear_after_fit   : 재학습 후 버퍼 초기화 여부
                        True  → 새 피드백만 반영 (드리프트 추적)
                        False → 모든 피드백 누적 반영 (안정적)
    """

    def __init__(
        self,
        risk_model        : RiskXGBoost,
        buffer            : FeedbackBuffer,
        min_samples       : int   = 30,
        incremental_trees : int   = 20,
        feedback_lr       : float = 0.01,
        snapshot_dir      : str   = "model_snapshots",
        clear_after_fit   : bool  = False,
    ):
        self.risk_model        = risk_model
        self.buffer            = buffer
        self.min_samples       = min_samples
        self.incremental_trees = incremental_trees
        self.feedback_lr       = feedback_lr
        self.snapshot_dir      = Path(snapshot_dir)
        self.clear_after_fit   = clear_after_fit
        self.retrain_count     = 0

        self.snapshot_dir.mkdir(exist_ok=True)

    # 외부 진입점
    def try_incremental_fit(self) -> bool:
        """
        버퍼 크기를 확인하고, 임계치 이상이면 증분 학습 실행.
        학습이 실행되면 True, 건너뛰면 False 반환.
        """
        if len(self.buffer) < self.min_samples:
            print(f"[IncrementalTrainer] 샘플 부족 "
                  f"({len(self.buffer)}/{self.min_samples}) → 학습 보류")
            return False

        if not self.risk_model.is_trained:
            print("[IncrementalTrainer] 기반 모델 미학습 → 증분 학습 불가")
            return False

        self._run(self.buffer.get_batch())
        return True

    # 재학습
    def _run(self, batch: tuple[np.ndarray, np.ndarray]) -> None:
        X_fb, y_fb = batch
        print(f"\n[IncrementalTrainer] 증분 학습 시작 | "
              f"피드백 샘플={len(X_fb)} | "
              f"추가 트리={self.incremental_trees}")

        # 기존 모델 스냅샷 저장
        snapshot_path = self._save_snapshot()

        # 피드백 데이터로 추가 boosting
        #    xgb_model=booster -> 기존 트리 유지 + 추가 트리만 학습
        #    학습률을 낮게 유지하여 기존 지식 과소멸 방지
        try:
            booster = self.risk_model.model.get_booster()

            incremental_model = xgb.XGBRegressor(
                device            = 'cuda',
                tree_method       = 'hist',
                max_bin           = 256,
                n_estimators      = self.incremental_trees,
                max_depth         = 4,           # 과적합 방지: 6보다 얕게
                learning_rate     = self.feedback_lr,
                subsample         = 0.8,
                colsample_bytree  = 0.8,
                reg_alpha         = 0.5,         # 정규화 강화: 소규모 피드백 과적합 방지
                reg_lambda        = 2.0,
                objective         = 'reg:squarederror',
                eval_metric       = 'mae',
                random_state      = 42,
            )
            incremental_model.fit(
                X_fb, y_fb,
                xgb_model     = booster,         # <- 기존 모델 위에 boosting 계속
                verbose       = False,
            )

            # 모델 교체
            self.risk_model.model = incremental_model
            self.retrain_count   += 1

            # 성능 확인
            preds = np.clip(np.round(incremental_model.predict(X_fb)), 1, 10)
            mae   = mean_absolute_error(y_fb, preds)
            print(f"[IncrementalTrainer] 재학습 완료 | "
                  f"횟수={self.retrain_count} | "
                  f"피드백 MAE={mae:.3f} | "
                  f"스냅샷={snapshot_path.name}")

            # 재학습 모델 저장
            updated_path = f"xgboost_risk_v{self.retrain_count}.json"
            self.risk_model.save(updated_path)

            # 버퍼 초기화(설정에 따라)
            if self.clear_after_fit:
                self.buffer.clear()
                print("[IncrementalTrainer] 버퍼 초기화 완료")

            torch.cuda.empty_cache()

        except Exception as e:
            print(f"[IncrementalTrainer] 재학습 실패: {e}")
            print(f"[IncrementalTrainer] 스냅샷({snapshot_path})으로 롤백 가능")

    def _save_snapshot(self) -> Path:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.snapshot_dir / f"snapshot_{ts}_v{self.retrain_count}.json"
        self.risk_model.model.save_model(str(path))
        return path

    def rollback(self, snapshot_path: str) -> None:
        """지정한 스냅샷 파일로 모델 롤백."""
        self.risk_model.load(snapshot_path)
        print(f"[IncrementalTrainer] 롤백 완료 → {snapshot_path}")

    def list_snapshots(self) -> List[str]:
        return sorted(str(p) for p in self.snapshot_dir.glob("snapshot_*.json"))

# 통합 파이프라인

class FinalRiskPipeline:

    def __init__(self, model: RiskXGBoost,
                 min_samples: int = 30,
                 incremental_trees: int = 20):
        self.xgb     = model
        self.buffer  = FeedbackBuffer()
        self.trainer = IncrementalTrainer(
            risk_model        = model,
            buffer            = self.buffer,
            min_samples       = min_samples,
            incremental_trees = incremental_trees,
        )
        self._last_features: Optional[RiskFeatures] = None   # 피드백 매칭용

    def compute(self, lstm_result: dict, bert_result: dict,
                drug_result: dict, accel_anomaly: float = 0.0) -> dict:
        n_drugs = len(drug_result.get('identified', {}))
        feat = RiskFeatures(
            pattern_anomaly_score    = lstm_result.get('anomaly_score', 0.0),
            pattern_variability      = lstm_result.get('variability', 0.0),
            zone_transition_abnormal = lstm_result.get('zone_transition_abnormal', 0.0),
            time_in_unusual_zone     = lstm_result.get('time_in_unusual_zone', 0.0),
            intent_risk_score        = bert_result.get('intent_risk_score', 0.0),
            pain_score               = bert_result.get('pain', 0.0),
            distress_score           = bert_result.get('distress', 0.0),
            emergency_score          = bert_result.get('emergency', 0.0),
            drug_risk_score          = drug_result.get('drug_risk_score', 0.0),
            n_drugs                  = min(n_drugs / 10.0, 1.0),
            has_major_interaction    = float(any(
                i['severity'] in ('major','serious')
                for i in drug_result.get('interactions', [])
            )),
            anomaly_composite = float(np.mean([
                lstm_result.get('anomaly_score', 0.0),
                bert_result.get('intent_risk_score', 0.0),
                accel_anomaly,
            ])),
            acceleration_anomaly = accel_anomaly,
        )
        self._last_features = feat       # 피드백 시 재참조
        result = self.xgb.predict(feat)
        result['incremental_ready'] = len(self.buffer) >= self.trainer.min_samples
        return result

    def user_feedback(
        self,
        feedback_type  : str            = 'false_alarm',
        corrected_risk : Optional[float] = None,
        features       : Optional[RiskFeatures] = None,
    ) -> dict:
        """
        사용자 피드백 수신 -> 버퍼 적재 -> 임계치 도달 시 자동 재학습.

        호출 예시:
        # "지금은 위급 상황이 아니야" (오탐 피드백)
        pipeline.user_feedback(feedback_type='false_alarm')

        # "이건 정말 위험해" (미탐 피드백 + 수동 위험도 입력)
        pipeline.user_feedback(feedback_type='missed', corrected_risk=9.0)

        파라미터:
        feedback_type  : 'false_alarm' | 'missed' | 'manual'
        corrected_risk : 명시적 보정 위험도.
                         None이면 false_alarm->1, missed->9 자동 할당
        features       : 피드백 시점의 RiskFeatures.
                         None이면 마지막 compute() 호출 때의 피처 재사용
        """
        feat = features or self._last_features
        if feat is None:
            return {'status': 'error', 'message': 'compute()를 먼저 호출하세요'}

        # corrected_risk 자동 할당
        if corrected_risk is None:
            corrected_risk = 1.0 if feedback_type == 'false_alarm' else 9.0

        # 예측값 조회
        pred_result    = self.xgb.predict(feat)
        predicted_risk = pred_result['raw_score']

        sample = FeedbackSample(
            predicted_risk = predicted_risk,
            corrected_risk = float(np.clip(corrected_risk, 1, 10)),
            features       = feat.to_array().tolist(),
            feedback_type  = feedback_type,
        )
        self.buffer.add(sample)

        # 임계치 도달 시 자동 재학습
        retrained = self.trainer.try_incremental_fit()

        return {
            'status'         : 'retrained' if retrained else 'buffered',
            'buffer_size'    : len(self.buffer),
            'min_samples'    : self.trainer.min_samples,
            'retrain_count'  : self.trainer.retrain_count,
            'predicted_risk' : predicted_risk,
            'corrected_risk' : corrected_risk,
        }

# 학습 실행

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
        mae  = mean_absolute_error(y[val], pred)
        maes.append(mae)
        print(f"  Fold {fold} MAE: {mae:.3f}")
        torch.cuda.empty_cache()

    print(f"평균 MAE: {np.mean(maes):.3f} ± {np.std(maes):.3f}")

    split = int(0.85 * len(X))
    final = RiskXGBoost()
    final.train(X[:split], y[:split], X[split:], y[split:])
    final.save('xgboost_risk_final.json')
    return final


if __name__ == '__main__':
    # 초기 학습
    model    = train_xgboost('mimic_iii_10k/')
    pipeline = FinalRiskPipeline(model, min_samples=30, incremental_trees=20)

    # 피드백 시뮬레이션
    dummy_lstm = {'anomaly_score':0.7, 'variability':0.4,
                  'zone_transition_abnormal':0.3, 'time_in_unusual_zone':0.2}
    dummy_bert = {'intent_risk_score':0.6, 'pain':0.5, 'distress':0.3, 'emergency':0.1}
    dummy_drug = {'identified':{}, 'interactions':[], 'drug_risk_score':0.2}

    result = pipeline.compute(dummy_lstm, dummy_bert, dummy_drug)
    print("예측:", result)

    # 사용자: "지금은 위급 상황이 아니야"
    fb = pipeline.user_feedback(feedback_type='false_alarm')
    print("피드백:", fb)
