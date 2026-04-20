import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import pandas as pd
import yaml
from collections import Counter
from pathlib import Path
from typing import List, Tuple

from gpu_config import setup_rtx3060, get_scaler, amp_dtype, print_vram, BATCH, NUM_WORKERS, PIN_MEMORY

INTENT_LABELS   = {0:'normal', 1:'pain', 2:'distress', 3:'emergency', 4:'activity'}
N_CLASSES       = len(INTENT_LABELS)
NORMAL_LABELS   = {0, 4}
ABNORMAL_LABELS = {1, 2, 3}

# 설정 로더

def load_config(config_path: str = "bert_config.yaml") -> dict:
    """
    YAML 설정 파일을 로드한다.
    파일이 없으면 코드 내 기본값(fallback)을 반환하여 하위 호환성 유지.
    """
    path = Path(config_path)
    if not path.exists():
        print(f"[경고] {config_path} 없음 -> 코드 내 기본값 사용")
        return _default_config()

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print(f"[설정] {config_path} 로드 완료 | alpha_mode={cfg['focal_loss']['alpha_mode']}")
    return cfg


def _default_config() -> dict:
    """bert_config.yaml 이 없을 때 사용하는 코드 내 기본값 (fallback)."""
    return {
        "model":      {"name": "klue/bert-base", "dropout": 0.3,
                       "max_len": 128, "gradient_checkpointing": True},
        "training":   {"epochs": 10, "weight_decay": 0.01,
                       "lr_bert": 2e-5, "lr_head": 1e-4, "warmup_ratio": 0.1},
        "focal_loss": {"gamma": 2.0, "alpha_mode": "manual",
                       "alpha_manual": {"normal":1.0,"pain":2.0,
                                        "distress":2.5,"emergency":3.0,"activity":1.2},
                       "dynamic_normalize": True, "dynamic_scale": 1.0},
        "risk_weights":{"pain": 0.6, "distress": 0.8, "emergency": 1.0},
        "router":     {"threshold": 0.5},
        "yamnet":     {"conf_threshold": 0.6,
                       "trigger_events": ["Groan","Shout","Crying","Screaming","Wheezing"]},
    }

# Dynamic Alpha 계산

def compute_dynamic_alpha(labels: List[int], cfg: dict) -> torch.Tensor:
    """
    데이터 분포 기반 자동 alpha 계산.

    원리:
        클래스 i의 샘플 수를 n_i 라 할 때,
            raw_alpha_i = 1 / n_i          (샘플이 적을수록 가중치 높게)

        dynamic_normalize=True 이면:
            alpha_i = raw_alpha_i / sum(raw_alpha) * N_CLASSES
            -> 가중치 합이 N_CLASSES로 정규화되어 loss 스케일 유지

        dynamic_scale 로 전체 배율 추가 조정 가능.

    예시 (emergency 500개, normal 5000개):
        raw  = [1/5000, 1/..., 1/500, 1/...]
        정규화 후 emergency 가중치 ≈ normal의 10배
    """
    fl_cfg    = cfg["focal_loss"]
    normalize = fl_cfg.get("dynamic_normalize", True)
    scale     = fl_cfg.get("dynamic_scale", 1.0)

    counts = Counter(labels)
    # 한 번도 등장하지 않은 클래스는 전체 샘플 수로 대체 (0 나눔 방지)
    total  = len(labels)
    raw    = torch.tensor(
        [1.0 / counts.get(i, total) for i in range(N_CLASSES)],
        dtype=torch.float32,
    )

    if normalize:
        alpha = raw / raw.sum() * N_CLASSES
    else:
        alpha = raw

    alpha = alpha * scale

    label_names = [INTENT_LABELS[i] for i in range(N_CLASSES)]
    print("[Dynamic Alpha] 클래스별 샘플 수:", dict(counts))
    print("[Dynamic Alpha] 계산된 alpha   :",
          {name: f"{val:.4f}" for name, val in zip(label_names, alpha.tolist())})
    return alpha


def build_alpha(labels: List[int], cfg: dict) -> torch.Tensor:
    """
    alpha_mode에 따라 manual 또는 dynamic alpha를 반환.
    """
    mode = cfg["focal_loss"]["alpha_mode"]

    if mode == "dynamic":
        return compute_dynamic_alpha(labels, cfg)

    # manual: YAML의 alpha_manual 딕셔너리 -> 클래스 순서대로 텐서 변환
    alpha_dict = cfg["focal_loss"]["alpha_manual"]
    alpha = torch.tensor(
        [alpha_dict[INTENT_LABELS[i]] for i in range(N_CLASSES)],
        dtype=torch.float32,
    )
    print("[Manual Alpha]", {INTENT_LABELS[i]: f"{v:.4f}"
                              for i, v in enumerate(alpha.tolist())})
    return alpha

# Focal Loss

class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017) — 클래스 불균형 대응

    CrossEntropyLoss에 (1 - p_t)^gamma 항을 곱해
    이미 잘 분류된 다수 클래스(normal, activity)의 기여도를 자동으로 낮추고,
    어려운 소수 클래스(emergency 등)에 학습이 집중되게 한다.

    파라미터:
        gamma : 포커싱 강도. 0이면 CrossEntropy와 동일.
                불균형이 심할수록 높게 (권장 2.0).
        alpha : 클래스별 사전 가중치 (Tensor). None이면 균등.
                Focal Loss와 alpha를 함께 쓰면 이중 보정 효과.

    작동 원리:
        p_t  = 정답 클래스의 예측 확률
        loss = -alpha_t * (1 - p_t)^gamma * log(p_t)

        p_t -> 1 (쉬운 샘플): focal_weight -> 0  -> 기여 감소
        p_t -> 0 (어려운 샘플): focal_weight -> 1 -> 기여 유지
    """
    def __init__(self, gamma: float = 2.0,
                 alpha: 'torch.Tensor | None' = None,
                 reduction: str = 'mean'):
        super().__init__()
        self.gamma     = gamma
        self.alpha     = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_prob = torch.nn.functional.log_softmax(logits, dim=-1)
        prob     = log_prob.exp()

        log_pt = log_prob.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt     = prob.gather(1,     targets.unsqueeze(1)).squeeze(1)

        focal_weight = (1.0 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_t      = self.alpha.to(logits.device)[targets]
            focal_weight = focal_weight * alpha_t

        loss = -focal_weight * log_pt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

# 데이터셋

class IntentDataset(Dataset):
    def __init__(self, csv_path: str, tokenizer, max_len=128):
        df = pd.read_csv(csv_path)
        self.texts  = df['text'].tolist()
        self.labels = df['label'].tolist()
        self.tok    = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(self.texts[i], max_length=self.max_len,
                       padding='max_length', truncation=True, return_tensors='pt')
        return {
            'input_ids'     : enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label'         : torch.tensor(self.labels[i], dtype=torch.long),
        }

# 2. 모델 (gradient checkpointing)

class IntentBERT(nn.Module):
    def __init__(self, model_name='klue/bert-base', dropout=0.3,
                 gradient_checkpointing=True):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)

        # gradient checkpointing: 순전파 중간 활성화를 저장하지 않고
        # 역전파 시 재계산 -> VRAM ~40% 절약, 속도 약 20% 감소
        if gradient_checkpointing:
            self.bert.gradient_checkpointing_enable()

        hidden = self.bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, N_CLASSES),
        )

    def forward(self, input_ids, attention_mask):
        # gradient checkpointing 사용 시 use_cache=False 필수
        cls = self.bert(input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False).last_hidden_state[:, 0, :]
        return self.classifier(cls)

    def get_scores(self, input_ids, attention_mask) -> dict:
        self.eval()
        with torch.no_grad(), autocast(device_type='cuda', dtype=amp_dtype()):
            probs = torch.softmax(
                self(input_ids, attention_mask), dim=-1
            )[0].float().cpu().numpy()
        return {INTENT_LABELS[i]: float(p) for i, p in enumerate(probs)}

# 라우터

class IntentRouter:
    def __init__(self, model: IntentBERT, tokenizer, device,
                 threshold: float = 0.5,
                 risk_weights: dict = None):
        self.model        = model
        self.tok          = tokenizer
        self.device       = device
        self.threshold    = threshold
        # YAML에서 주입된 위험도 가중치 (없으면 기본값)
        self.risk_weights = risk_weights or {"pain": 0.6, "distress": 0.8, "emergency": 1.0}

    def route(self, text: str) -> Tuple[str, float, dict]:
        enc = self.tok(text, max_length=128, padding='max_length',
                       truncation=True, return_tensors='pt')
        ids  = enc['input_ids'].to(self.device)
        mask = enc['attention_mask'].to(self.device)
        scores = self.model.get_scores(ids, mask)

        top    = max(scores, key=scores.get)
        top_id = next(k for k, v in INTENT_LABELS.items() if v == top)
        conf   = scores[top]

        route = 'lstm' if (top_id in NORMAL_LABELS and conf >= self.threshold) else 'anomaly'

        # 위험도 점수: YAML 가중치 사용
        rw = self.risk_weights
        scores['intent_risk_score'] = (
            scores.get('pain',      0) * rw.get('pain',      0.6) +
            scores.get('distress',  0) * rw.get('distress',  0.8) +
            scores.get('emergency', 0) * rw.get('emergency', 1.0)
        )
        return route, conf, scores

# YamNet 연동
def yamnet_callback(event: str, conf: float, router: IntentRouter,
                    stt_text: str = None,
                    conf_threshold: float = 0.6,
                    trigger_events: set = None) -> dict:
    """
    conf_threshold, trigger_events 를 인자로 받아 하드코딩 제거.
    기본값은 기존 동작과 동일하게 유지.
    """
    trigger_events = trigger_events or {'Groan', 'Shout', 'Crying', 'Screaming', 'Wheezing'}

    if event in trigger_events and conf > conf_threshold:
        text = stt_text or f"이상 소리: {event}"
        route, c, scores = router.route(text)
        return {'route': route, 'scores': scores, 'triggered_by': event}
    return {'route': 'normal', 'scores': {}, 'triggered_by': None}

# 학습 루프

def train_bert(csv_path: str,
               config_path: str = "bert_config.yaml",
               model_name: str = None,
               epochs: int = None,
               batch_size: int = None) -> Tuple[IntentBERT, object]:

    cfg = load_config(config_path)

    device     = setup_rtx3060()
    model_name = model_name or cfg["model"]["name"]
    epochs     = epochs     or cfg["training"]["epochs"]
    batch_size = batch_size or BATCH['bert']

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset   = IntentDataset(csv_path, tokenizer,
                              max_len=cfg["model"]["max_len"])
    loader    = DataLoader(
        dataset,
        batch_size         = batch_size,
        shuffle            = True,
        num_workers        = NUM_WORKERS,
        pin_memory         = PIN_MEMORY,
        persistent_workers = True,
    )

    model = IntentBERT(
        model_name,
        dropout                = cfg["model"]["dropout"],
        gradient_checkpointing = cfg["model"]["gradient_checkpointing"],
    ).to(device)

    # torch.compile (BERT 인코더 그래프 최적화)
    try:
        model = torch.compile(model, mode='reduce-overhead')
        print("torch.compile 적용")
    except Exception:
        pass

    # BERT 하위 레이어 lr 낮게, 헤드 lr 높게
    t_cfg = cfg["training"]
    optimizer = torch.optim.AdamW([
        {'params': model.bert.parameters(),       'lr': t_cfg["lr_bert"]},
        {'params': model.classifier.parameters(), 'lr': t_cfg["lr_head"]},
    ], weight_decay=t_cfg["weight_decay"], fused=True)

    total_steps = len(loader) * epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = int(total_steps * t_cfg["warmup_ratio"]),
        num_training_steps = total_steps,
    )

    # Focal Loss 설정
    # alpha_mode = "dynamic" : 데이터 분포로 자동 계산 (소수 클래스 -> 높은 가중치)
    # alpha_mode = "manual"  : YAML alpha_manual 값 그대로 사용
    # gamma, alpha 병용 -> 소수 클래스 + 어려운 샘플 이중 보정
    alpha = build_alpha(dataset.labels, cfg)
    criterion = FocalLoss(
        gamma = cfg["focal_loss"]["gamma"],
        alpha = alpha,
    )
    scaler = get_scaler()

    print_vram('BERT 학습 시작')
    best_loss = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        total, correct = 0.0, 0

        for batch in loader:
            ids  = batch['input_ids'].to(device, non_blocking=True)
            mask = batch['attention_mask'].to(device, non_blocking=True)
            lab  = batch['label'].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type='cuda', dtype=amp_dtype()):
                logits = model(ids, mask)
                loss   = criterion(logits, lab)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total   += loss.item()
            correct += (logits.argmax(1) == lab).sum().item()

        avg = total / len(loader)
        acc = correct / len(dataset)
        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), 'bert_intent_best.pt')
        if epoch % 2 == 0:
            print_vram(f'BERT epoch {epoch}')
        print(f"  Epoch {epoch:2d} | loss={avg:.4f} | acc={acc:.3f}")

    print("저장 완료: bert_intent_best.pt")
    return model, tokenizer


if __name__ == '__main__':
    train_bert('intent_data.csv')
