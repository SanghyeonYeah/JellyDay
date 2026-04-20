import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from filterpy.kalman import KalmanFilter
from scipy.io import arff
from collections import deque
from datetime import datetime, timedelta
import sqlite3
import json

from gpu_config import setup_rtx3060, get_scaler, amp_dtype, print_vram, BATCH, NUM_WORKERS, PIN_MEMORY


# 데이터 로드

def load_casas(arff_path: str) -> pd.DataFrame:
    data, _ = arff.loadarff(arff_path)
    df = pd.DataFrame(data)
    for col in df.select_dtypes(['object']):
        df[col] = df[col].str.decode('utf-8')
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df.sort_values('datetime').reset_index(drop=True)


def casas_to_matrix(df, rssi_cols, accel_cols):
    rssi  = df[rssi_cols].values.astype(np.float32)
    accel = df[accel_cols].values.astype(np.float32)
    rssi  = np.where(np.isnan(rssi), -100.0, rssi)
    return rssi, accel, df['datetime']


# RSSI 전처리

def apply_ema(rssi: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """
    지수 평활법
    매우 빠른 스파이크를 alpha 비율로 감쇠.
    alpha가 작을수록 더 강하게 평활.

    공식: s_t = alpha * x_t + (1 - alpha) * s_{t-1}
    """
    out = np.empty_like(rssi)
    out[0] = rssi[0]
    for t in range(1, len(rssi)):
        out[t] = alpha * rssi[t] + (1.0 - alpha) * out[t - 1]
    return out


def apply_moving_average(rssi: np.ndarray, window: int = 5) -> np.ndarray:
    """
    이동 평균
    EMA 이후 남은 중간 주파수 노이즈 제거.
    window가 클수록 더 부드럽지만 반응 지연 증가.
    """
    kernel = np.ones(window) / window
    # 각 AP 채널별로 독립적으로 적용
    return np.column_stack([
        np.convolve(rssi[:, i], kernel, mode='same')
        for i in range(rssi.shape[1])
    ])


def apply_kalman(rssi: np.ndarray) -> np.ndarray:
    """
    칼만 필터
    EMA -> MA로 스파이크 제거 후 투입하여
    선형 상태 추정의 정확도 향상.
    """
    n = rssi.shape[1]
    kf = KalmanFilter(dim_x=n, dim_z=n)
    kf.F = np.eye(n); kf.H = np.eye(n)
    kf.R = np.eye(n) * 5.0
    kf.Q = np.eye(n) * 0.1
    kf.P = np.eye(n) * 10.0
    kf.x = rssi[0].reshape(-1, 1)
    out = []
    for obs in rssi:
        kf.predict(); kf.update(obs.reshape(-1, 1))
        out.append(kf.x.flatten())
    return np.array(out)


def preprocess_rssi(rssi_raw: np.ndarray,
                    ema_alpha: float = 0.3,
                    ma_window: int   = 5) -> np.ndarray:

    step1 = apply_ema(rssi_raw, alpha=ema_alpha)
    step2 = apply_moving_average(step1, window=ma_window)
    step3 = apply_kalman(step2)
    return step3


# 가속도 주파수 도메인 특징 추출

def extract_accel_features(accel: np.ndarray, window: int = 20) -> np.ndarray:
    """
    가속도 데이터에서 주파수 도메인 특징 추출.

    추가 특징 (채널별 -> 총 accel_channels * 3개 추가):
      - ZCR (Zero Crossing Rate): 신호가 0을 교차하는 비율
            -> 높을수록 동적(걷기/운동), 낮을수록 정적(수면/앉기)
      - Energy: 윈도우 내 신호 에너지 (x²의 합)
            -> 활동량 척도
      - SMA (Signal Magnitude Area): |x| + |y| + |z| 의 평균
            -> 3축 합산 활동량

    반환 shape: (T, accel_channels * 3)
    기존 accel(T, 3)과 concatenate하여 사용.
    """
    T, C = accel.shape
    zcr    = np.zeros((T, C), dtype=np.float32)
    energy = np.zeros((T, C), dtype=np.float32)

    for t in range(T):
        start = max(0, t - window + 1)
        seg   = accel[start:t + 1]            # (window, C)

        # ZCR: 부호 변화 횟수 / 윈도우 길이
        signs  = np.sign(seg)
        zcr[t] = np.sum(np.abs(np.diff(signs, axis=0)), axis=0) / max(len(seg) - 1, 1)

        # Energy: sum(x²) / 윈도우 길이
        energy[t] = np.sum(seg ** 2, axis=0) / len(seg)

    # SMA: |x| + |y| + |z| 의 윈도우 평균 -> 단일 컬럼으로 확장
    sma = np.zeros((T, 1), dtype=np.float32)
    for t in range(T):
        start  = max(0, t - window + 1)
        seg    = accel[start:t + 1]
        sma[t] = np.mean(np.sum(np.abs(seg), axis=1))

    return np.concatenate([zcr, energy, sma], axis=1)  # (T, C*2+1)


# 상태 전이 행렬

class ZoneTransitionMatrix:
    """
    학습 데이터로부터 구역 간 전이 확률을 추정하고,
    추론 시 "이전 구역 -> 현재 구역" 확률을 입력 피처로 제공.

    Laplace Smoothing (smooth=1):
        한 번도 관측되지 않은 전이도 0이 되지 않도록 최소 확률 부여.
    """
    def __init__(self, n_zones: int, smooth: float = 1.0):
        self.n_zones = n_zones
        self.smooth  = smooth
        # 전이 카운트 행렬: [from_zone, to_zone]
        self._counts = np.zeros((n_zones, n_zones), dtype=np.float64)

    def fit(self, zone_labels: np.ndarray) -> 'ZoneTransitionMatrix':
        for i in range(len(zone_labels) - 1):
            f = int(zone_labels[i])
            t = int(zone_labels[i + 1])
            if 0 <= f < self.n_zones and 0 <= t < self.n_zones:
                self._counts[f, t] += 1

        # Laplace Smoothing 후 행 정규화 -> 확률 행렬
        smoothed = self._counts + self.smooth
        self._prob = smoothed / smoothed.sum(axis=1, keepdims=True)
        return self

    def transition_prob(self, from_zone: int) -> np.ndarray:
        """
        from_zone에서 각 구역으로 갈 확률 벡터 반환.
        입력 피처로 사용: shape (n_zones,)
        """
        if not hasattr(self, '_prob'):
            return np.ones(self.n_zones) / self.n_zones   # 미학습 시 균등 분포
        from_zone = int(np.clip(from_zone, 0, self.n_zones - 1))
        return self._prob[from_zone].astype(np.float32)

    def save(self, path: str):
        np.save(path, self._counts)

    def load(self, path: str):
        self._counts = np.load(path)
        smoothed     = self._counts + self.smooth
        self._prob   = smoothed / smoothed.sum(axis=1, keepdims=True)


# DBSCAN 클러스터링 + 주기적 Refit 감지

class ZoneClusterer:
    """
    Refit 신호 로직:
      - predict_zone() 호출마다 최근 W개 결과를 추적
      - 'unknown' 비율 > refit_ratio  ->  환경 변화 감지 -> refit_needed = True
      - 클러스터 중심과의 평균 거리가 drift_threshold 초과해도 refit 신호
    """
    def __init__(self, eps=3.0, min_samples=10,
                 refit_window=200, refit_ratio=0.25, drift_threshold=5.0):
        self.eps             = eps
        self.min_samples     = min_samples
        self.dbscan          = DBSCAN(eps=eps, min_samples=min_samples)
        self.scaler          = StandardScaler()
        self.zone_map        : dict = {}
        self.cluster_centers : dict = {}

        self.refit_window    = refit_window
        self.refit_ratio     = refit_ratio
        self.drift_threshold = drift_threshold
        self._predict_log    = deque(maxlen=refit_window)
        self._dist_log       = deque(maxlen=refit_window)
        self.refit_needed    = False
        self.refit_count     = 0
        self.fit_timestamp   = None

    def fit(self, rssi: np.ndarray, timestamps: pd.Series) -> np.ndarray:
        scaled = self.scaler.fit_transform(rssi)
        labels = self.dbscan.fit_predict(scaled)
        for lab in set(labels) - {-1}:
            mask = labels == lab
            self.cluster_centers[lab] = rssi[mask].mean(axis=0)
            h   = pd.to_datetime(timestamps[mask]).dt.hour.mode()[0]
            tag = 'morning' if 6 <= h < 10 else ('bedroom' if h >= 22 or h < 6 else 'living')
            self.zone_map[lab] = f'zone_{lab}_{tag}'

        self.fit_timestamp = datetime.now().isoformat()
        self._predict_log.clear()
        self._dist_log.clear()
        self.refit_needed = False
        return labels

    def refit(self, rssi_recent: np.ndarray, timestamps_recent: pd.Series):
        print(f"[ZoneClusterer] Refit #{self.refit_count + 1} 시작")
        old_centers = dict(self.cluster_centers)
        self.dbscan          = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.scaler          = StandardScaler()
        self.zone_map        = {}
        self.cluster_centers = {}
        new_labels = self.fit(rssi_recent, timestamps_recent)

        if old_centers:
            for new_lab, new_center in self.cluster_centers.items():
                best_old, best_dist = None, float('inf')
                for old_lab, old_center in old_centers.items():
                    d = np.linalg.norm(new_center - old_center)
                    if d < best_dist:
                        best_dist, best_old = d, old_lab
                if best_dist < self.drift_threshold * 2:
                    old_name = f'zone_{best_old}_'
                    matched  = next((v for k, v in self.zone_map.items()
                                     if k == new_lab), None)
                    if matched and old_name not in matched:
                        self.zone_map[new_lab] = matched

        self.refit_count += 1
        self.refit_needed = False
        print(f"[ZoneClusterer] Refit 완료: {len(self.cluster_centers)}개 구역")
        return new_labels

    def predict_zone(self, rssi_vec: np.ndarray) -> str:
        if not self.cluster_centers:
            self._predict_log.append('unknown')
            return 'unknown'

        sv = self.scaler.transform(rssi_vec.reshape(1, -1))[0]
        best, bd = -1, float('inf')
        for lab, c in self.cluster_centers.items():
            sc = self.scaler.transform(c.reshape(1, -1))[0]
            d  = np.linalg.norm(sv - sc)
            if d < bd:
                bd, best = d, lab

        zone = self.zone_map.get(best, 'unknown')
        self._predict_log.append(zone)
        self._dist_log.append(bd)

        if len(self._predict_log) >= self.refit_window:
            unknown_ratio = list(self._predict_log).count('unknown') / self.refit_window
            mean_dist     = float(np.mean(list(self._dist_log)))
            if unknown_ratio > self.refit_ratio:
                print(f"[ZoneClusterer] Refit 신호: unknown={unknown_ratio:.2%}")
                self.refit_needed = True
            elif mean_dist > self.drift_threshold:
                print(f"[ZoneClusterer] Refit 신호: drift={mean_dist:.2f}")
                self.refit_needed = True

        return zone

    def save(self, path: str):
        json.dump({
            'zone_map'       : {str(k): v for k, v in self.zone_map.items()},
            'cluster_centers': {str(k): v.tolist() for k, v in self.cluster_centers.items()},
            'scaler_mean'    : self.scaler.mean_.tolist(),
            'scaler_scale'   : self.scaler.scale_.tolist(),
            'fit_timestamp'  : self.fit_timestamp,
            'refit_count'    : self.refit_count,
        }, open(path, 'w'), indent=2)

    def load(self, path: str):
        state = json.load(open(path))
        self.zone_map        = {int(k): v for k, v in state['zone_map'].items()}
        self.cluster_centers = {int(k): np.array(v)
                                for k, v in state['cluster_centers'].items()}
        self.scaler.mean_    = np.array(state['scaler_mean'])
        self.scaler.scale_   = np.array(state['scaler_scale'])
        self.scaler.var_     = self.scaler.scale_ ** 2
        self.scaler.n_features_in_ = len(self.scaler.mean_)
        self.fit_timestamp   = state.get('fit_timestamp')
        self.refit_count     = state.get('refit_count', 0)


# LSTM

class PatternLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim=128, n_layers=2, n_zones=10):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers,
                            batch_first=True, dropout=0.3)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc   = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(64, n_zones),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.norm(out[:, -1, :]))


class ZoneDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float16)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


def build_sequences(feat, labels, seq_len=20):
    X, y = [], []
    for i in range(len(feat) - seq_len):
        X.append(feat[i:i+seq_len])
        y.append(labels[i+seq_len])
    return np.array(X), np.array(y)


# 7. 추론 Warmup

def warmup_model(model: PatternLSTM, device, input_dim: int, seq_len: int = 20):
    """더미 텐서로 forward 3회 실행 -> torch.compile 그래프 사전 컴파일."""
    print("[Warmup] LSTM 추론 사전 컴파일 시작...")
    model.eval()
    dummy = torch.zeros(1, seq_len, input_dim, dtype=amp_dtype(), device=device)
    with torch.no_grad(), autocast(device_type='cuda', dtype=amp_dtype()):
        for _ in range(3):
            _ = model(dummy)
            torch.cuda.synchronize()
    print("[Warmup] 완료 — 이후 첫 추론 지연 없음")


# 8. SQLite 이상치 히스토리 DB (데이터 로테이션 포함)

class AnomalyHistoryDB:
    """
    SQLite 이상치 기록 영속 저장.

    데이터 로테이션 (rotate_old_logs):
        - retention_days(기본 90일) 이전 로그를 통계로 집약 후 삭제
        - anomaly_log_archive 테이블에 월별 요약 저장
        - 조회 속도 저하 방지 + 장기 운영 디스크 절약

    스키마:
        anomaly_log         : 실시간 원본 로그
        anomaly_log_archive : 월별 집약 아카이브
    """
    def __init__(self, db_path: str = 'anomaly_history.db'):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute('PRAGMA journal_mode=WAL')
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS anomaly_log (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                ts            TEXT    NOT NULL,
                anomaly_score REAL    NOT NULL,
                actual_zone   TEXT,
                threshold     REAL,
                is_anomaly    INTEGER NOT NULL DEFAULT 0,
                variability   REAL,
                alert_level   TEXT    DEFAULT 'none',
                note          TEXT
            )
        """)
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ts ON anomaly_log(ts)"
        )
        # 월별 집약 아카이브 테이블
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS anomaly_log_archive (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                month           TEXT    NOT NULL,    -- 'YYYY-MM'
                actual_zone     TEXT,
                total_records   INTEGER,
                anomaly_count   INTEGER,
                avg_score       REAL,
                avg_variability REAL,
                archived_at     TEXT
            )
        """)
        self.conn.commit()

    def record(self, anomaly_score: float, actual_zone: str,
               threshold: float, is_anomaly: bool,
               variability: float, alert_level: str = 'none', note: str = ''):
        self.conn.execute(
            """INSERT INTO anomaly_log
               (ts, anomaly_score, actual_zone, threshold, is_anomaly, variability, alert_level, note)
               VALUES (?,?,?,?,?,?,?,?)""",
            (datetime.now().isoformat(), anomaly_score, actual_zone,
             threshold, int(is_anomaly), variability, alert_level, note)
        )
        self.conn.commit()

    def add_note(self, record_id: int, note: str):
        self.conn.execute(
            "UPDATE anomaly_log SET note=? WHERE id=?", (note, record_id)
        )
        self.conn.commit()

    def last_id(self) -> int:
        row = self.conn.execute(
            "SELECT id FROM anomaly_log ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return row[0] if row else -1

    # 데이터 로테이션
    def rotate_old_logs(self, retention_days: int = 90) -> int:
        """
        retention_days(기본 90일) 이전 로그를 월별로 집약해 아카이브하고 삭제.

        동작 순서:
          1. cutoff 날짜 산출 (오늘 - retention_days)
          2. cutoff 이전 데이터를 'YYYY-MM' × actual_zone 단위로 집약
          3. anomaly_log_archive 에 삽입 (이미 아카이브된 월은 IGNORE)
          4. 원본 anomaly_log에서 해당 행 삭제
          5. VACUUM으로 디스크 공간 회수

        반환: 삭제된 행 수
        """
        cutoff = (datetime.now() - timedelta(days=retention_days)).isoformat()

        # 집약 쿼리 (월별 × 구역별)
        rows = self.conn.execute("""
            SELECT SUBSTR(ts, 1, 7) AS month,
                   actual_zone,
                   COUNT(*)              AS total_records,
                   SUM(is_anomaly)       AS anomaly_count,
                   AVG(anomaly_score)    AS avg_score,
                   AVG(variability)      AS avg_variability
            FROM anomaly_log
            WHERE ts < ?
            GROUP BY month, actual_zone
        """, (cutoff,)).fetchall()

        if not rows:
            print(f"[DB Rotation] {retention_days}일 이전 데이터 없음 — 건너뜀")
            return 0

        archived_at = datetime.now().isoformat()
        self.conn.executemany("""
            INSERT OR IGNORE INTO anomaly_log_archive
            (month, actual_zone, total_records, anomaly_count,
             avg_score, avg_variability, archived_at)
            VALUES (?,?,?,?,?,?,?)
        """, [(r[0], r[1], r[2], r[3], r[4], r[5], archived_at) for r in rows])

        # 원본 삭제
        cur = self.conn.execute(
            "DELETE FROM anomaly_log WHERE ts < ?", (cutoff,)
        )
        deleted = cur.rowcount
        self.conn.commit()

        # 디스크 공간 회수
        self.conn.execute("VACUUM")
        print(f"[DB Rotation] {deleted}행 삭제 -> {len(rows)}개 월별 아카이브 저장 완료")
        return deleted

    # 리포트 쿼리
    def query_anomalies(self, limit: int = 100) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT * FROM anomaly_log WHERE is_anomaly=1 ORDER BY ts DESC LIMIT ?",
            self.conn, params=(limit,)
        )

    def query_by_zone(self) -> pd.DataFrame:
        return pd.read_sql_query(
            """SELECT actual_zone,
                      COUNT(*)            AS total_records,
                      SUM(is_anomaly)     AS anomaly_count,
                      ROUND(AVG(anomaly_score), 4) AS avg_score,
                      ROUND(AVG(variability), 4)   AS avg_variability
               FROM anomaly_log GROUP BY actual_zone ORDER BY anomaly_count DESC""",
            self.conn
        )

    def query_time_heatmap(self) -> pd.DataFrame:
        return pd.read_sql_query(
            """SELECT CAST(SUBSTR(ts, 12, 2) AS INTEGER) AS hour,
                      COUNT(*) AS total,
                      SUM(is_anomaly) AS anomalies,
                      ROUND(AVG(anomaly_score), 4) AS avg_score
               FROM anomaly_log GROUP BY hour ORDER BY hour""",
            self.conn
        )

    def export_report(self, out_path: str = 'anomaly_report.json'):
        report = {
            'generated_at'    : datetime.now().isoformat(),
            'by_zone'         : self.query_by_zone().to_dict(orient='records'),
            'time_heatmap'    : self.query_time_heatmap().to_dict(orient='records'),
            'recent_anomalies': self.query_anomalies(50).to_dict(orient='records'),
        }
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[Report] 저장 완료: {out_path}")


# 9. 이상치 감지 — 다중 임계값 + 시간대 가중치 + 피드백 루프

# 시간대별 가중치 테이블
# 새벽(0~5시)에 이상치 민감도를 높여 야간 사고 조기 감지
_HOUR_WEIGHT = {
    **{h: 1.5  for h in range(0,  6)},   # 새벽 (0~5시)  : 고감도
    **{h: 1.0  for h in range(6, 22)},   # 주간 (6~21시) : 기본
    **{h: 1.25 for h in range(22, 24)},  # 야간 (22~23시): 중간
}


class PatternAnomalyDetector:
    """
    동적 임계값 + 다중 Alert 레벨 + 시간대 가중치 + 피드백 온라인 학습.

    Alert 레벨:
        none   : 정상
        yellow : 단순 패턴 이탈 (관찰 권고)
                 조건: weighted_score > yellow_threshold
        red    : 명확한 이상 (즉시 알림)
                 조건: weighted_score > red_multiplier * yellow_threshold

    시간대 가중치:
        raw_score * HOUR_WEIGHT[현재 시각] 로 보정 후 임계값과 비교.
        새벽 시간대 계수 1.5 -> 낮은 raw_score도 yellow/red로 분류 가능.

    피드백 루프 (online_feedback):
        사용자 "이상 없음" -> is_exception=True 로 update_pattern() 호출
        -> loss 가중치 0.3 적용 소량 Fine-tuning
        -> 개인 기준선이 점진적으로 갱신됨
    """
    # Red Alert = yellow_threshold * RED_MULTIPLIER
    RED_MULTIPLIER = 1.6

    def __init__(self, model: PatternLSTM, device,
                 init_threshold : float = 0.35,
                 baseline_window: int   = 200,
                 k_base         : float = 2.0,
                 k_var          : float = 1.5,
                 history_db_path: str   = 'anomaly_history.db'):

        self.model            = model
        self.device           = device
        self.init_threshold   = init_threshold
        self.baseline_window  = baseline_window
        self.k_base           = k_base
        self.k_var            = k_var

        self._baseline_buf: list = []
        self.baseline_mean : float | None = None
        self.baseline_std  : float | None = None
        self._recent  = deque(maxlen=50)
        self.history  = deque(maxlen=100)
        self.db       = AnomalyHistoryDB(history_db_path)

    # 동적 임계값
    def _compute_threshold(self) -> float:
        if self.baseline_mean is None:
            return self.init_threshold
        variability = self.variability_score()
        return (
            self.baseline_mean
            + self.k_base * self.baseline_std
            + self.k_var  * variability
        )

    def variability_score(self) -> float:
        return float(np.std(list(self._recent))) if len(self._recent) >= 5 else 0.0

    @property
    def threshold(self) -> float:
        return self._compute_threshold()

    # 시간대 가중치 적용 점수
    @staticmethod
    def _weighted_score(raw_score: float, hour: int | None = None) -> float:
        """
        시간대(hour)에 따라 raw_score에 가중치를 곱해 반환.
        hour=None 이면 현재 시각을 자동으로 사용.
        """
        if hour is None:
            hour = datetime.now().hour
        weight = _HOUR_WEIGHT.get(hour, 1.0)
        return raw_score * weight

    # Alert 레벨 판정
    def _alert_level(self, weighted_score: float) -> str:
        """
        yellow_threshold = self.threshold
        red_threshold    = self.threshold * RED_MULTIPLIER

        반환값: 'none' | 'yellow' | 'red'
        """
        thr = self._compute_threshold()
        if weighted_score > thr * self.RED_MULTIPLIER:
            return 'red'
        elif weighted_score > thr:
            return 'yellow'
        return 'none'

    # 이상치 점수 계산
    def score(self, seq: torch.Tensor, actual_zone: str | int,
              hour: int | None = None) -> dict:
        """
        반환 dict:
            raw_score     : 모델 예측 기반 원시 점수
            weighted_score: 시간대 가중치 적용 점수
            alert_level   : 'none' | 'yellow' | 'red'
            threshold     : 당시 동적 임계값
        """
        self.model.eval()
        with torch.no_grad(), autocast(device_type='cuda', dtype=amp_dtype()):
            probs = torch.softmax(
                self.model(seq.unsqueeze(0).to(self.device).to(amp_dtype())), dim=-1
            )[0]

        zone_id = actual_zone if isinstance(actual_zone, int) else 0
        raw     = 1.0 - probs[zone_id].item()

        # 시간대 가중치 적용
        weighted = self._weighted_score(raw, hour)

        self._recent.append(raw)
        self.history.append(raw)

        # 기준선 수집
        if self.baseline_mean is None:
            self._baseline_buf.append(raw)
            if len(self._baseline_buf) >= self.baseline_window:
                arr = np.array(self._baseline_buf)
                self.baseline_mean = float(arr.mean())
                self.baseline_std  = float(arr.std())
                print(f"[Detector] 기준선 확정: mean={self.baseline_mean:.4f} "
                      f"std={self.baseline_std:.4f} "
                      f"-> threshold={self._compute_threshold():.4f}")

        thr         = self._compute_threshold()
        alert       = self._alert_level(weighted)
        is_anomaly  = alert != 'none'
        zone_str    = actual_zone if isinstance(actual_zone, str) else str(actual_zone)

        self.db.record(
            anomaly_score = raw,
            actual_zone   = zone_str,
            threshold     = thr,
            is_anomaly    = is_anomaly,
            variability   = self.variability_score(),
            alert_level   = alert,
        )

        return {
            'raw_score'     : raw,
            'weighted_score': weighted,
            'alert_level'   : alert,
            'threshold'     : thr,
        }

    def should_ask_user(self, result: dict) -> bool:
        """yellow 이상이면 사용자에게 확인 요청."""
        return result['alert_level'] in ('yellow', 'red')

    def annotate_last(self, note: str):
        self.db.add_note(self.db.last_id(), note)

    # 피드백 루프 온라인 학습
    def online_feedback(self, seq: torch.Tensor, zone,
                        optimizer, is_false_alarm: bool = True):
        """
        사용자 피드백 "이상 없음" -> Fine-tuning 트리거.

        is_false_alarm=True  -> is_exception=True (loss 가중치 0.3)
                               오탐 패턴을 정상으로 소량 학습
        is_false_alarm=False -> 일반 강도 재학습 (미탐 교정)

        self.db의 마지막 레코드에 'user_ok' 노트도 함께 기록.
        """
        self.annotate_last('user_feedback:no_emergency' if is_false_alarm else 'user_feedback:confirm')
        self._update_pattern(seq, zone, optimizer, is_exception=is_false_alarm)

    def _update_pattern(self, seq, zone, optimizer, is_exception=False):
        """
        is_exception=True : 낮은 위험도 예외 케이스 -> loss 가중치 0.3
        is_exception=False: 일반 재학습
        """
        self.model.train()
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type='cuda', dtype=amp_dtype()):
            zone_id = zone if isinstance(zone, int) else 0
            loss = nn.CrossEntropyLoss()(
                self.model(seq.unsqueeze(0).to(self.device).to(amp_dtype())),
                torch.tensor([zone_id], device=self.device),
            ) * (0.3 if is_exception else 1.0)
        loss.backward()
        optimizer.step()


# 10. 체크포인트 저장/로드

def save_checkpoint(model: PatternLSTM, optimizer, scheduler,
                    epoch: int, loss: float,
                    path: str = 'lstm_pattern_best.pt'):
    """
    정전·재시작 후 학습 흐름 완벽 복구를 위해
    model state 외에 optimizer·scheduler state도 함께 저장.

    저장 항목:
        epoch         : 현재 epoch (재시작 시 이어받기)
        model_state   : 모델 가중치
        optimizer_state: AdamW 모멘텀 등 누적 상태
        scheduler_state: CosineAnnealing T_cur 등
        best_loss     : 최적 loss 기록
        saved_at      : 저장 시각
    """
    torch.save({
        'epoch'          : epoch,
        'model_state'    : model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'best_loss'      : loss,
        'saved_at'       : datetime.now().isoformat(),
    }, path)
    print(f"[Checkpoint] 저장 완료: {path} (epoch={epoch}, loss={loss:.4f})")


def load_checkpoint(model: PatternLSTM, optimizer, scheduler,
                    path: str = 'lstm_pattern_best.pt') -> int:
    """
    체크포인트 로드 후 시작 epoch 반환.
    파일이 없으면 0 반환 (처음부터 학습).
    """
    import os
    if not os.path.exists(path):
        print(f"[Checkpoint] {path} 없음 — 처음부터 학습")
        return 0

    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    scheduler.load_state_dict(ckpt['scheduler_state'])
    start_epoch = ckpt['epoch'] + 1
    print(f"[Checkpoint] 복원 완료: epoch={ckpt['epoch']} "
          f"loss={ckpt['best_loss']:.4f} "
          f"saved_at={ckpt.get('saved_at','?')} "
          f"-> epoch {start_epoch}부터 재개")
    return start_epoch


# 11. 학습 루프

def train_lstm(arff_path: str, rssi_cols: list, accel_cols: list,
               n_zones=10, seq_len=20, epochs=30,
               resume: bool = True) -> PatternLSTM:

    device = setup_rtx3060()

    df = load_casas(arff_path)
    rssi_raw, accel, ts = casas_to_matrix(df, rssi_cols, accel_cols)

    # RSSI 전처리
    rssi_f = preprocess_rssi(rssi_raw, ema_alpha=0.3, ma_window=5)

    # 구역 클러스터링
    clusterer   = ZoneClusterer()
    zone_labels = clusterer.fit(rssi_f, ts)
    clusterer.save('zone_clusterer.json')

    # 상태 전이 행렬 학습
    trans_mat = ZoneTransitionMatrix(n_zones)
    trans_mat.fit(zone_labels)
    trans_mat.save('zone_transition.npy')

    n   = len(df)

    # One-hot 구역
    oh  = np.zeros((n, n_zones), dtype=np.float32)
    for i, lab in enumerate(zone_labels):
        if 0 <= lab < n_zones:
            oh[i, lab] = 1.0

    # 시간 sin/cos
    hrs = ts.dt.hour.values
    tfs = np.stack([np.sin(2*np.pi*hrs/24), np.cos(2*np.pi*hrs/24)], 1).astype(np.float32)

    # 가속도 주파수 특징 추가 (ZCR, Energy, SMA)
    accel_feat = extract_accel_features(accel, window=seq_len)  # (T, C*2+1)

    # 전이 확률 피처 추가
    trans_feat = np.zeros((n, n_zones), dtype=np.float32)
    for i in range(1, n):
        prev = int(zone_labels[i - 1]) if zone_labels[i - 1] >= 0 else 0
        trans_feat[i] = trans_mat.transition_prob(prev)

    # 전체 피처 결합:
    # [one_hot(n_zones) | time_sin_cos(2) | accel_raw(3) | accel_feat(C*2+1) | trans_prob(n_zones)]
    feat = np.concatenate([oh, tfs, accel, accel_feat, trans_feat], axis=1)

    valid = zone_labels >= 0
    feat, zone_labels = feat[valid], zone_labels[valid]
    input_dim = feat.shape[1]

    X, y = build_sequences(feat, zone_labels, seq_len)
    loader = DataLoader(
        ZoneDataset(X, y),
        batch_size         = BATCH['lstm'],
        shuffle            = True,
        num_workers        = NUM_WORKERS,
        pin_memory         = PIN_MEMORY,
        persistent_workers = True,
    )

    model     = PatternLSTM(input_dim, 128, 2, n_zones).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4, fused=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 체크포인트 복원
    start_epoch = 0
    if resume:
        start_epoch = load_checkpoint(model, optimizer, scheduler)

    try:
        model = torch.compile(model, mode='max-autotune')
        print("torch.compile 적용 완료")
    except Exception:
        print("torch.compile 미지원 — 일반 모드 실행")

    warmup_model(model, device, input_dim, seq_len)

    criterion = nn.CrossEntropyLoss().to(device)
    scaler    = get_scaler()

    print_vram('학습 시작')
    best_loss = float('inf')

    for epoch in range(start_epoch + 1, epochs + 1):
        model.train()
        total = 0.0

        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True).to(amp_dtype())
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type='cuda', dtype=amp_dtype()):
                loss = criterion(model(xb), yb)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()

        scheduler.step()
        avg = total / len(loader)

        if avg < best_loss:
            best_loss = avg
            # optimizer·scheduler state 포함 체크포인트 저장
            save_checkpoint(model, optimizer, scheduler, epoch, best_loss)

        if epoch % 5 == 0:
            print_vram(f'epoch {epoch}')
        print(f"  Epoch {epoch:3d} | loss={avg:.4f} | best={best_loss:.4f}")

    print("학습 완료")
    return model


if __name__ == '__main__':
    train_lstm(
        arff_path  = 'hh101/hh101.arff',
        rssi_cols  = ['AP1','AP2','AP3','AP4','AP5','AP6','AP7','AP8'],
        accel_cols = ['acc_x','acc_y','acc_z'],
        resume     = True,
    )
