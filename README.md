# 🚀 AR·자율주행·AI 시대를 위한 차세대 이동통신 예측 기술  
### Next-Generation Wireless Channel Prediction using Large Wireless Model (LWM)

> **충남대학교 2025 창의SW·AI 축전 창의작품경진대회 출품작**  
> 참가 부문: 학술  
> 팀명: **LWM**  
> 지도교수: **양희철 교수님**

---

## 📘 개요 (Overview)

본 프로젝트는 초고속 이동통신 환경에서 발생하는 **채널 에이징(Channel Aging)** 문제를 해결하기 위해  
**거대무선모델(Large Wireless Model, LWM)** 기반 트랜스포머를 적용한  
**무선 채널 시계열 예측(Sequential Channel Prediction)** 기술을 제시합니다.

LWM은 **채널 마스킹(Masked Channel Modeling, MCM)** 으로 사전학습된 **파운데이션 모델**이며,  
RNN/GRU/LSTM/일반 Transformer와 **NMSE(dB)** 기준 성능을 비교했습니다.

---

## 🎯 연구 목표 (Objectives)

- 시간 변화로 인한 **채널 에이징 완화**
- **LWM Transformer 백본**을 채널 예측에 적용
- **RNN/GRU/LSTM/Transformer vs LWM** 성능 비교
- **User / Scene / Subcarrier** 기준 **3가지 데이터 분할 실험**
- 전이학습 전략 비교  
  1) **LWM_Finetune**: 사전학습 백본 미세조정  
  2) **LWM_Freeze**: 백본 동결, 출력 헤드만 학습  
  3) **LWM_FromScratch**: 사전학습 없이 처음부터 학습


---

### 🧪 실험 데이터 (Dataset) 요약
| 항목 | 값 |
|---|---|
| 시나리오 | **O2_dyn_3p5** (O2 Dynamic, Sub-6 GHz) |
| 지형/레이아웃 | 도심 도로 + 교차로(동적 환경) |
| 기지국 수 | 2 (BS#1, BS#2) |
| 후보 사용자 | ≈ 115,000 (격자) |
| 시간 샘플 간격 | 100 ms *(문서 기준)* |
| 총 장면 수 | ≈ 1,000 scenes *(문서 기준)* |
| 주파수 | 3.5 GHz |
| 제공 데이터 | OFDM **서브캐리어별 주파수-도메인 CSI**(복소 채널 계수) |

이후 대역폭, 서브캐리어 수, 안테나 배열, 경로 수 등은 실험 목적에 맞게 아래에서 설정

```python
scene = 각 데이터 분할에 맞게 설정
parameters['dataset_folder'] = '/home/dlghdbs200/LWM/scenarios'
parameters['scenario'] = 'O2_dyn_3p5'
parameters['dynamic_scenario_scenes'] = np.arange(scene)
parameters['num_paths'] = 10
parameters['user_rows'] = np.arange(100)
parameters['user_subsampling'] = 0.01
parameters['active_BS'] = np.array([1])
parameters['activate_OFDM'] = 1
parameters['OFDM']['bandwidth'] = 0.05
parameters['OFDM']['subcarriers'] = 512
parameters['OFDM']['selected_subcarriers'] = np.arange(0, 64, 1)
parameters['ue_antenna']['shape'] = np.array([1, 1])
parameters['bs_antenna']['shape'] = np.array([1, 32])

```

---

## 🔀 데이터 분할 전략 (Split Strategies)

세 가지 분할 축에서 **일반화 성능**을 평가합니다.

### 1) Scene Split (장면 기반)
과거 Scene으로 미래 Scene을 예측합니다. (3:1 = 75%:25%)

```python
seq_len = 14
split_idx = 26  # 예: 총 44개 Scene → 0~25 Train / 26~43 Val
train_ds = dataset[:split_idx]
val_ds   = dataset[split_idx:]
```
### 🧍‍♂️ Table 1. 사용자 단위 분할 파라미터
| Parameter | Value |
|------------|-------|
| Total Scene (N_scene) | 30 |
| Users (U_total) | 727 |
| Train users (U_train) | 545 |
| Validation users (U_val) | 182 |
| Sequence length (T) | 14 |
| Prediction horizon (h) | 1 |
| Number of valid sequence (M) | 16 |
| Subcarrier (S) | 64 |
| Train samples | 558,080 |
| Validation samples | 186,368 |

---

### 2) user split (user 기반)
유저를 3:1로 분할하고 train에 대해서 추가적으로 아래 더 학습 사용자 비율로 분할 합니다.
```pyhton
U = dataset[0][0]['user']['channel'].shape[0]
user_ids = np.arange(U)
random.shuffle(user_ids)

cut = int(len(user_ids) * 0.75)          # 75% 학습 도메인
## 학습 사용자 비율을 아래코드에서 선택 가능
cut_1pt = max(1, math.floor(cut * 0.01))
cut_3pt = max(1, math.floor(cut * 0.03))
cut_5pt = max(1, math.floor(cut * 0.05))
cut_10pt = max(1, math.floor(cut * 0.1))
cut_30pt = max(1, math.floor(cut * 0.3))
cut_50pt = max(1, math.floor(cut * 0.5))
cut_100pt = cut

train_users = set(user_ids[:cut_50pt])
val_users   = set(user_ids[cut:])
```

### 🌆 Table 2. 장면 단위 분할 파라미터
| Parameter | Value |
|------------|-------|
| Total Scene (N_scene) | 44 |
| User (U) | 727 |
| Train scene | 26 |
| Validation scene | 18 |
| Sequence length (T) | 14 |
| Prediction horizon (h) | 1 |
| Train sequence (M_train) | 12 |
| Validation sequence (M_val) | 4 |
| Subcarrier (S) | 64 |
| Train samples | 566,016 |
| Validation samples | 188,672 |


### 3) subcarrier Split (서브캐리어 분할)
캐리어를 64개로 분할 한 후 train : val = 3 : 1로 분할하였습니다.
``` pyhton
import numpy as np, random

S = dataset[0][0]['user']['channel'].shape[3]  # 예: 64
sc_ids = np.arange(S)
random.shuffle(sc_ids)

cut = int(S * 0.75)
train_sc = set(sc_ids[:cut])
val_sc   = set(sc_ids[cut:])
```
### 📡 Table 3. 서브캐리어 단위 분할 파라미터
| Parameter | Value |
|------------|-------|
| Total Scene (N_scene) | 30 |
| User (U) | 727 |
| Total subcarrier (S) | 64 |
| Train subcarrier (S_train) | 48 |
| Validation subcarrier (S_val) | 16 |
| Sequence length (T) | 14 |
| Prediction horizon (h) | 1 |
| Number of valid sequence (M) | 16 |
| Train samples | 566,016 |
| Validation samples | 188,672 |

---
# 🧠 LWM 모델 구조
Input → Projection → Patch/Positional Embedding → LWM Backbone → Output Head → Prediction
![LWM Architecture](https://github.com/yuknow01/LWM/blob/main/LWM_architecture.png?raw=true)

## ⚙️ 코드 구조 (Code)

```python
class LWMWithHead(nn.Module):
    def __init__(
        self,
        input_dim: int,                 # 실제 입력 차원 (예: 64)
        patch_length: int,              # 백본에서 기대하는 패치 길이 (예: 16)
        d_model: int = 64,              # LWM의 hidden size
        max_len: int = 129,             # positional encoding의 최대 길이
        n_layers: int = 12,             # Transformer encoder 층 수
        out_dim: int = 64,              # FC head 출력 차원
        freeze_backbone: bool = True,   # 백본 파라미터 동결 여부
        checkpoint_path: str | None = "./model_weights.pth",
        device: str = "cuda"
    ):
        super().__init__()

        # 입력을 백본의 patch 크기에 맞게 투영
        # ⭐ 만일 가중치를 사용하지 않으면 아래 코드는 주석처리 ⭐
        self.input_proj = nn.Linear(input_dim, patch_length)

        # ⭐ 백본 초기화 (사전학습된 모델 불러오기 or 랜덤 초기화) ⭐
        if checkpoint_path is None:
            self.backbone = lwm(
                element_length=patch_length,
                d_model=d_model,
                max_len=max_len,
                n_layers=n_layers
            ).to(device)
        else:
            self.backbone = lwm.from_pretrained(
                ckpt_name=checkpoint_path,
                device=device
            )

        # 백본 파라미터 동결 (Freeze)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # FC Head 구성 (1 layer)
        self.head = nn.Sequential(
            nn.Linear(d_model, out_dim),
        )

    def forward(self, input_ids: torch.Tensor, masked_pos: torch.Tensor) -> torch.Tensor:
        # ⭐ 만일 가중치를 사용하지 않으면 아래 코드는 주석처리 ⭐
        x = self.input_proj(input_ids)               # 입력 투영
        _, enc_output = self.backbone(x, masked_pos) # 백본 통과
        feat = enc_output[:, 0, :]                   # CLS 토큰 피처 추출
        out = self.head(feat)                        # FC Head 출력
        return out
```
---



## 🧠 학습 전략 (Training Strategies)

본 프로젝트에서는 사전학습된 LWM 백본의 활용 여부와 미세조정 범위에 따라  
다음 세 가지 학습 전략을 비교했습니다.

| 전략명 | 설명 | 학습 범위 |
|--------|------|-----------|
| **LWM_Finetune** | 사전학습된 백본을 불러와 전체 파라미터를 미세조정 (Full Fine-tuning) | 투영 + 백본 + 출력 헤드 |
| **LWM_Freeze** | 사전학습된 백본을 동결하고, 출력 레이어(헤드)만 학습 | 출력 헤드만 |
| **LWM_FromScratch** | 사전학습 없이 처음부터 모델을 학습 | 전층 |

> 💡 모든 실험은 동일한 데이터셋(DeepMIMO v3)과 전처리 방식,  
> 평가 지표(NMSE[dB])를 기준으로 수행되었습니다.

---

## 📊 실험 결과 (Results)
![LWM NMSE 성능평가](https://github.com/yuknow01/LWM/blob/main/%EC%84%B1%EB%8A%A5%ED%8F%89%EA%B0%80.png)

> NMSE(dB)는 **값이 더 작을수록(더 음수)** 좋습니다. (아래 수치는 반올림)

| 분할 기준 | 최적 모델 | NMSE(dB) | 해석 |
|---|---:|:--:|---|
| **User Split** | **GRU** | **-23.40** | 사용자 ID가 바뀌어도 일반화가 가장 좋음 |
| **Scene Split** | **LWM_Finetune** | **≈ -18.24** | 시간축(장면 변화) 분리에 가장 강건 |
| **Subcarrier Split** | **LWM_FromScratch** | **-25.71** | 주파수 도메인 변화에 최적 |

### 분할별 코멘트
- **User Split**: RNN 계열(특히 **GRU**) 우세 → **개별 사용자 일반화** 강함.  
- **Scene Split**: **사전학습 백본 미세조정(LWM_Finetune)**이 최고 → **시간적 분포 이동**에 robust.  
- **Subcarrier Split**: **LWM_FromScratch**가 최고, **LWM_Finetune** 차순위 → **주파수 도메인 적응력** 우수.

---

### 📊 추가 실험 — 유저 분할(학습 사용자 비율)
![LWM NMSE 유저분할](https://github.com/yuknow01/LWM/blob/main/%EC%B6%94%EA%B0%80%20%EC%84%B1%EB%8A%A5%ED%8F%89%EA%B0%80%20%EC%9C%A0%EC%A0%80%EB%B6%84%ED%95%A0.png)

- **≤ 1% (저데이터)**: **LWM_FromScratch** 선도 → **데이터 효율성** 강점  
- **3–30%**: **RNN**이 구간별 최고치 다수  
- **≥ 50%**: **GRU** 최상 성능

---

### ⚡ 추론 시간 & 파라미터
![inference 및 parameters](https://github.com/yuknow01/LWM/blob/main/inference%20%EB%B0%8F%20parameters.png)

- **RNN 계열**: ≈ **0.85–0.91 ms/sample**, **≤ 0.10M** params → **빠르고 가벼움**  
- **LWM 계열**: ≈ **14.5–16.5 ms/sample**, **~0.61M** params → 느리지만 **Scene/Subcarrier 적응력** 우수

---

### ✅ 한 줄 요약
**사용자 일반화 → GRU**, **시간 변화 → LWM_Finetune**, **주파수 변화 → LWM_FromScratch**.  
**속도/경량성은 RNN·GRU**, **도메인 적응·저데이터 효율은 LWM**이 강점.


## 💡 기대 효과 (Expected Impact)

- **LWM 파운데이션 모델의 무선 채널 예측 적용 가능성 검증**  
- 다양한 사용자 환경에서도 **높은 일반화 성능** 입증  
- 실제 통신 시스템 적용 시 **정확도 ↔ 연산 복잡도** 트레이드오프 가이드 제공  
- **AR·자율주행·AI 시대**의 차세대 이동통신 기술 발전에 기여  
- 향후 연구로 **도메인 적응형 모델(Domain Adaptive Model)** 및  
  **경량화(Lightweight Fine-tuning)** 전략 개발 가능성 제시  

---



