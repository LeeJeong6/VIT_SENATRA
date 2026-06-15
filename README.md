# VIT-SENATRA + RoPE

Vision Transformer (ViT)에 **SENATRA** (Semantic Token Reduction Architecture) 모듈을 삽입하여 progressive token downsampling을 수행하는 ImageNet 학습 코드입니다.

기존 token merging 연구들(ToMe 등)은 classification만 지원했지만, SENATRA는 각 reduction 단계의 assignment matrix를 Markov chain으로 곱해 **원본 해상도의 segmentation map까지 복원**할 수 있습니다.

위치 인코딩은 **2D Axial RoPE** (Rotary Position Embedding)를 사용합니다. 토큰 수가 단계적으로 감소해도 각 단계의 격자 좌표를 RoPE에 직접 전달하므로 보간이 필요 없고, 학습 파라미터도 추가되지 않습니다.

---

## 핵심 아이디어

### 문제 정의

표준 ViT는 모든 transformer block을 통해 동일한 수의 patch token을 처리합니다. 224×224 이미지에 patch_size=16을 쓰면 **196개(14×14) 토큰**이 12개 block 내내 유지됩니다. 이는 계산 비용이 높고, 계층적 spatial 표현을 만들기 어렵습니다.

### SENATRA의 해법

특정 transformer block 이후에 `SenatraTokenReducer`를 삽입하여 토큰 격자를 단계적으로 축소합니다.

```
Input: 224×224 image
  ↓ PatchEmbed (16×16 patch)
14×14 = 196 tokens   ← RoPE with (row,col) ∈ [0,13]²

  ↓ RoPEBlockWithKey (block 3)  ← key 벡터 추출
  ↓ SenatraTokenReducer  →  aups1 [B, 196, 144] 저장
12×12 = 144 tokens   ← RoPE with (row,col) ∈ [0,11]²

  ↓ RoPEBlockWithKey (block 6)
  ↓ SenatraTokenReducer  →  aups2 [B, 144, 100] 저장
10×10 = 100 tokens   ← RoPE with (row,col) ∈ [0,9]²

  ↓ RoPEBlockWithKey (block 9)
  ↓ SenatraTokenReducer  →  aups3 [B, 100, 25] 저장
5×5 = 25 tokens      ← RoPE with (row,col) ∈ [0,4]²

  ↓ RoPEBlocks 10~12 → LayerNorm → [CLS] token → head
Classification logits
```

---

## Positional Encoding: 2D Axial RoPE

### 기존 Absolute Pos Embed vs RoPE

| | Absolute Pos Embed (구버전) | RoPE (현재) |
|---|---|---|
| 저장 방식 | 학습 파라미터 `[1, N+1, C]` | 파라미터 없음 (좌표에서 직접 계산) |
| 해상도 변화 시 | bicubic interpolation 필요 | 새 격자 좌표 그대로 사용 |
| 추가 파라미터 | `pos_embed` 텐서 | 없음 |
| SENATRA 후 처리 | 보간 후 재주입 | 다음 block이 새 해상도 freqs 사용 |

### 동작 방식

```python
# __init__: 각 해상도별 freqs 사전 계산 (학습 파라미터 아님)
self._rope_freqs[(14,14)] = _compute_axial_cis(head_dim, 14, 14, theta=100.0)  # [196, head_dim//2]
self._rope_freqs[(12,12)] = _compute_axial_cis(head_dim, 12, 12, theta=100.0)  # [144, head_dim//2]
self._rope_freqs[(10,10)] = _compute_axial_cis(head_dim, 10, 10, theta=100.0)  # [100, head_dim//2]
self._rope_freqs[( 5, 5)] = _compute_axial_cis(head_dim,  5,  5, theta=100.0)  # [ 25, head_dim//2]

# forward: SENATRA reduction 직후 현재 해상도의 freqs로 교체
current_res = self.initial_resolution           # (14, 14)
freqs_cis = self._rope_freqs[current_res]
...
# reducer 통과 후
current_res = self.senatra_resolutions[reducer_idx]   # (12, 12)
freqs_cis = self._rope_freqs[current_res]             # ← 다음 block부터 이 freqs 사용
```

### RoPE 적용 범위

CLS 토큰은 공간 위치 개념이 없으므로 RoPE에서 제외합니다. Patch 토큰에만 회전을 적용합니다.

```python
# RoPEAttention.forward
n = self.num_prefix_tokens   # 1 (CLS 사용) 또는 0
q_r, k_r = _apply_rotary_emb(q[:, :, n:], k[:, :, n:], freqs_cis)
q = torch.cat([q[:, :, :n], q_r], dim=2)
k = torch.cat([k[:, :, :n], k_r], dim=2)
```

### SENATRA 내부 relative bias

SENATRA reducer 내부의 cross-attention (K는 입력 해상도, Q는 출력 해상도)에는 RoPE를 적용하지 않습니다. 서로 다른 해상도를 가진 K-Q 쌍에 RoPE를 정의하면 상대 위치 의미가 모호해지기 때문입니다. 대신 기존과 동일하게 **learnable relative bias**를 사용합니다.

```python
# dense 모드: 전체 N_in × N_out 상대 바이어스
self.rel_bias = nn.Parameter(torch.randn(n_in, n_out) * 0.01)

# local 모드 (NATTEN): 윈도우 내 2D 상대 위치 바이어스
self.rel_bias_local = nn.Parameter(torch.zeros(num_branches, 2*k-1, 2*k-1))
```

---

## Segmentation 복원: Markov Chain

각 reduction 단계는 **assignment matrix** (`aups`)를 저장합니다.

| 단계 | 형태 | 의미 |
|------|------|------|
| `aups1` | `[B, 196, 144]` | 원본 토큰 → 1차 축소 클러스터 소속 확률 |
| `aups2` | `[B, 144, 100]` | 1차 클러스터 → 2차 축소 클러스터 소속 확률 |
| `aups3` | `[B, 100, 25]` | 2차 클러스터 → 최종 클러스터 소속 확률 |

이를 **Markov chain으로 연쇄 곱** 하면:

```python
chain = aups1 @ aups2 @ aups3  # [B, 196, 25]
labels = chain.argmax(dim=-1).view(B, 14, 14)  # [B, 14, 14]
```

`chain[b, i, j]`는 원본 token i가 최종 cluster j에 속할 확률입니다.

---

## Key Vector 활용

SENATRA reducer 직전 block은 `RoPEBlockWithKey`로 교체됩니다. `return_key=True` 호출 시 self-attention의 key 벡터(post-RoPE)를 추출해 SENATRA의 `external_keys`로 전달합니다.

```python
# forward_features
x, key_tokens = blk(x, freqs_cis, return_key=True)   # key 추출
x, aups, adown = reducer(patch_tokens,
                          external_keys=key_tokens)    # semantic grouping
```

Key 벡터는 이미 semantic 정보를 인코딩하고 있어, 별도의 for-loop 없이 의미론적 k-means clustering에 가까운 downsampling이 한 번에 완료됩니다.

---

## 파일 구조

```
VIT_SENATRA/
├── train.py              # 메인 학습 스크립트 (VisionTransformerSenatra + RoPE)
├── senatra.py            # SENATRA 핵심 모듈
├── vision_transformer.py # 표준 ViT 구현 (timm 0.4.12 기반)
├── utils.py              # 옵티마이저, 스케줄러, 체크포인트 유틸리티
└── rope-vit/             # RoPE 참고 구현 (naver-ai)
```

### train.py 주요 클래스

| 클래스 | 역할 |
|--------|------|
| `RoPEAttention` | 2D Axial RoPE를 Q·K에 적용; `return_key=True`로 key 벡터 반환 가능 |
| `RoPEBlock` | RoPEAttention + MLP; `freqs_cis`를 인자로 받음 |
| `RoPEBlockWithKey` | RoPEBlock + key 반환 기능 (SENATRA reducer 직전 위치) |
| `VisionTransformerSenatra` | ViT + SENATRA + RoPE 통합 모델 |

### senatra.py 주요 함수

| 이름 | 역할 |
|------|------|
| `SenatraTokenReducer` | 단일 token reduction 모듈 (`external_keys` 지원) |
| `resolve_reducer_grouping_mode` | 각 stage의 grouping mode 결정 |
| `compose_membership_map` | Markov chain으로 `[B, N_final, N_initial]` 반환 |
| `segmentation_labels_from_aups` | argmax로 segmentation label map `[B, H, W]` 생성 |

---

## 학습 실행

### 환경

```bash
pip install torch torchvision timm yacs termcolor
# local grouping 사용 시 (권장):
pip install natten
```

### 기본 실행 (ViT-Small, 8 GPU)

```bash
torchrun --nproc_per_node=8 train.py \
    --data-path /path/to/imagenet \
    --model vit_small \
    --batch-size 128 \
    --output output/ \
    --tag run1
```

### 주요 인자

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--model` | `vit_small` | `vit_tiny`, `vit_small`, `vit_base` |
| `--rope-theta` | `100.0` | RoPE base frequency θ |
| `--senatra-resolutions` | `12x12 10x10 5x5` | reduction 목표 해상도 목록 |
| `--senatra-insert-blocks` | 자동(depth 기반) | reducer를 삽입할 block 인덱스 (1-based) |
| `--senatra-grouping-mode` | `auto` | `auto`(NATTEN 있으면 local), `local`, `dense` |
| `--local-window-size` | `3` | local grouping 윈도우 크기 |
| `--num-iters` | `1` | iterative grouping 반복 횟수 |
| `--use-cls-token` / `--no-cls-token` | CLS 사용 | CLS 토큰 사용 여부 |
| `--vis-freq` | `1` | grouping map 시각화 주기 (epoch) |
| `--save-best-only` | off | 최고 acc 체크포인트만 저장 |

### ViT-Base 예시

```bash
torchrun --nproc_per_node=8 train.py \
    --data-path /path/to/imagenet \
    --model vit_base \
    --batch-size 64 \
    --senatra-resolutions 12x12 10x10 5x5 \
    --senatra-insert-blocks 3 6 9 \
    --rope-theta 100.0 \
    --output output/ \
    --tag vit_base_senatra_rope
```

---

## 학습 설정 상세

### LR 스케일링

```
effective_lr = base_lr × batch_size × world_size / 512
```

기본: `base_lr=5e-4`, batch=128×8GPU → `lr = 1e-3`

### 기본 하이퍼파라미터

| 항목 | 값 |
|------|-----|
| Optimizer | AdamW (β=(0.9, 0.999), eps=1e-8) |
| Weight decay | 0.05 (1D param, bias 제외) |
| LR schedule | Cosine (warmup 20 epoch) |
| Epochs | 300 |
| Drop path rate | 0.3 |
| Augmentation | RandAugment + Mixup(α=0.8) + CutMix(α=1.0) |
| Label smoothing | 0.1 |

### Weight decay 제외 대상

`cls_token`, `rel_bias`, `rel_bias_local`  
(`pos_embed` 없음 — RoPE 사용)

---

## 출력 디렉토리 구조

```
output/
└── vit_small_patch16_224_senatra_rope/
    └── run1/
        ├── config.json
        ├── log_rank0.txt
        ├── ckpt_epoch_0.pth
        ├── best.pth             # --save-best-only 사용 시
        └── vis_grouping/
            ├── epoch_000/
            │   ├── sample_00.png
            │   └── ...
```

---

## 평가

```bash
torchrun --nproc_per_node=1 train.py \
    --data-path /path/to/imagenet \
    --model vit_small \
    --resume output/.../best.pth \
    --eval
```

---

## Segmentation Map 추출 (추론 예시)

```python
from senatra import segmentation_labels_from_aups

model.eval()
with torch.no_grad():
    logits, aups_list, _ = model(images, return_assignments=True)

# 14×14 해상도의 hard segmentation label map
labels = segmentation_labels_from_aups(aups_list, model.initial_resolution)
# labels: [B, 14, 14]

# 원본 이미지 크기로 업샘플
seg_map = F.interpolate(labels.unsqueeze(1).float(),
                         size=(224, 224), mode='nearest').squeeze(1)
```

---

## 기존 연구와의 비교

| 방법 | Token 감소 | Classification | Segmentation 복원 | 위치 인코딩 |
|------|-----------|---------------|------------------|------------|
| ToMe | 점진적 merge | ✅ | ❌ | Absolute |
| Swin PatchMerging | 2×2 → 1 | ✅ | △ | Relative bias |
| **SENATRA (구버전)** | 임의 격자 축소 | ✅ | ✅ | Absolute (bicubic 보간) |
| **SENATRA + RoPE (현재)** | 임의 격자 축소 | ✅ | ✅ | RoPE (파라미터 없음) |

**RoPE 도입으로 개선된 점:**
- `pos_embed` 학습 파라미터 제거 → 더 적은 파라미터
- SENATRA reduction 후 bicubic interpolation 불필요 → 깔끔한 forward 흐름
- 해상도가 단계적으로 변화해도 각 격자 좌표를 그대로 사용 → 정확한 위치 인코딩
