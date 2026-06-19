# Attractor-AE 결과 (Phase F)

> 수렴 모델 기준 정량·정성 평가 결과. 논문 Results 섹션 재료.
> 평가 코드: `src/eval/{interpret,disentangle,attractor_diag}.py`. 데이터: 3D Shapes(480k).

---

## 1. 실험 설정
- **데이터**: 3D Shapes, RGB 64×64×3, GT factor 6개 (floor_hue·wall_hue·object_hue 각 10, scale 8, shape 4, orientation 15).
- **모델**(모두 z=32, 동일 conv enc/dec, Adam 1e-3, 수렴까지):
  - **VAE** (β=1): 연속 잠재, KL 규제. 표현 = $\mu$.
  - **VQ-VAE** (8 group × K=16 product 양자화 + STE): 이산, 비변분. 표현 = $z_e$.
  - **Attractor-AE** (제안): 8 factor × 4dim, Modern Hopfield, β 1→8 어닐링, unit_patterns + context_norm, fpi(max_iter=20) + one-step IFT. 표현 = $y^\star$.
- **지표**: recon(MSE 합/배치), MIG, factor-level MIG(fMIG), DCI(MI importance 기반 D·C), FactorVAE score. (의존성 numpy only — 전부 직접 구현.)

---

## 2. 재구성 (Reconstruction)
| 모델 | recon (↓) |
|---|---|
| VAE | 29.5 |
| **VQ-VAE** | **5.7** |
| Attractor-AE | 7.9 |

Attractor-AE는 VAE(KL 병목)보다 훨씬 낮고 VQ-VAE보다 약간 높다. (이산 병목 + 어닐링 비용.)

---

## 3. Disentanglement (수렴 모델, 높을수록 좋음)
| 모델 | MIG(per-dim) | **fMIG** | **DCI-D** | **DCI-C** | FactorVAE |
|---|---|---|---|---|---|
| VAE | 0.564 | — | 0.420 | 0.276 | 0.883 |
| VQ-VAE | 0.077 | 0.289 | 0.584 | 0.293 | **0.994** |
| **Attractor-AE** | 0.085 | **0.307** | **0.617** | **0.302** | 0.947 |

**판독**:
- **Attractor-AE가 fMIG·DCI-D·DCI-C 3개 지표에서 최고** (VQ-VAE 추월). FactorVAE만 VQ-VAE에 근소 2위.
- **per-dim MIG는 metric 편향**: VAE는 KL로 7/32 차원만 활성→차원당 factor 집중↑. VQ-VAE·Attractor는 factor당 다차원(4dim)이라 정보가 분산→차원당 MIG↓. 따라서 4dim/factor 설계엔 **fMIG(factor-code 단위)가 공정**하며, 거기서 Attractor가 우위.

---

## 3b. 변분(Categorical AQ-VAE) + ablation (β_kl 스윕, K=32)
deterministic AQ를 변분화(`--variational`, 학습 categorical prior + closed-form KL)하고, β_kl·K를 스윕한 7-모델 비교(수렴, n=20k):

| model | recon↓ | 활성코드 | fMIG↑ | DCI-D↑ | DCI-C↑ | FactorVAE↑ |
|---|---|---|---|---|---|---|
| VAE | 28.5 | 7 dim | — | 0.388 | 0.254 | 0.902 |
| VQ-VAE | **5.7** | 32 | 0.288 | 0.541 | 0.272 | **0.994** |
| **AQ det (K16)** | 7.9 | 88/128 | 0.299 | **0.569** | **0.277** | 0.946 |
| AQ 변분 β_kl=0.5 | 8.8 | 20/128 | 0.127 | 0.440 | 0.236 | 0.949 |
| AQ 변분 β_kl=0.1 | 9.7 | 31/128 | 0.147 | 0.527 | 0.260 | 0.929 |
| AQ 변분 β_kl=0.05 | 8.1 | 32/128 | 0.139 | 0.501 | 0.240 | 0.981 |
| **AQ K=32 det** | 9.0 | 97/256 | **0.331** | 0.568 | 0.273 | 0.935 |

**판독 (정직하게)**:
1. **변분(categorical KL)은 disentanglement를 악화** — 모든 β_kl(0.05~0.5)에서 fMIG 0.13~0.15 (det 0.30 대비↓), DCI-D도↓. "KL=정보병목→깨끗한 분리" 기대가 *반대*로 작동(KL이 factor-code 정보량을 눌러 분리 흐려짐). → **변분의 가치는 분리가 아니라 *생성***.
2. **변분 생성은 작동하나 흐릿** (`generated_e148.png`) — factorized prior가 비일관 조합 샘플(재구성 `recon_var_e148.png`은 선명). → autoregressive prior 또는 Gaussian-in-c 변형 필요.
3. **K=32는 싼 disentanglement 이득** — fMIG **0.331(전체 최고)**. 코드 2배가 factor↔GT 정렬 개선(단 recon 9.0으로 미개선).
4. **rate-distortion**: β_kl↓ → 활성코드↑·recon↑ (0.5→0.05: 20→32, 8.8→8.1)나 disentanglement는 전 구간 저조.

→ **표현 품질(분리+recon)엔 deterministic AQ(또는 K=32)가 최선(VQ를 DCI에서 추월). 변분은 생성 능력을 얻는 대신 분리를 희생.**

---

## 4. 의미 분석 (정성 + 정량)
모델팩터(8)와 GT(6)의 NMI 행렬 + attractor purity + swap traversal.

**팩터↔GT 정렬 (NMI, 수렴 모델)**:
| GT factor | 소유 모델팩터 (NMI) | 비고 |
|---|---|---|
| floor_hue | f4 (0.79) | 깨끗 |
| wall_hue | f5 (0.77), f0 (0.51) | 강함 |
| object_hue | f6 (0.57), f3 (0.38) | 약한 중복 |
| orientation | f1 (0.62) | 위치와 약 entangle |
| shape | f2 (0.37), f7 (0.37) | purity 0.65, **4/4 값** |
| **scale** | — | **분리 실패** (f2서 shape와 entangle) |

**attractor purity 예** (정렬 GT 기준): shape factor f2 — `a0→shape2(0.89)`, `a5→shape0(1.00)` 등 일부 attractor 순도 1.0. wall_hue f5 purity 0.74.

**swap traversal** (`outputs/interpret/`): 한 이미지의 한 factor만 attractor를 갈아끼워 디코딩.
- **shape(f3/e60 그림)**: 객체 모양만 큐브↔원기둥↔캡슐↔구로 변하고 배경·색 고정 → "이 attractor=큐브, 저건=원기둥" 시각 확인.
- **orientation(f1)**: 장면 회전(수평선·위치 이동).

**핵심**: 이산 attractor가 **해석가능한 의미**를 가짐(특히 범주형 shape, hue). 단 disentanglement는 부분적(scale 누락, 약한 중복).

---

## 5. Attractor 고유 진단
**(1) Perturbation-복귀 (basin 검증)** — 고정점 $y^\star$에 크기 ν 교란 후 재반복:
| ν | 0.10 | 0.25 | 0.50 | 1.00 | 2.00 |
|---|---|---|---|---|---|
| 복귀거리 | 0.000 | 0.000 | 0.021 | 0.158 | 0.371 |
작은 교란은 복귀(수축적 basin), 큰 교란(ν=2)은 이웃 basin 이탈 → **진짜 multi-stable attractor**.

**(2) Internalization** — solver step 수 vs recon:
| steps | 0 | 1 | 2 | 4 | 8 | 64 |
|---|---|---|---|---|---|---|
| recon | 280 | 13.6 | 9.4 | **8.0** | 7.9 | 7.9 |
- head contribution $\|y^\star-c\|/\|c\|=0.477$ (solver가 실질적 일을 함).
- **0 step(context 직접) = 망함** → solver 필수(이산성의 근원). Solve-the-Loop의 internalization과 반대이나, 이산성이 동역학에서 나오므로 *당연*.
- **추론 시 ~4 step이면 거의 수렴** → 효율적(빠른 solver 결과와 일치).

---

## 6. 종합 비교
| | VAE | VQ-VAE | **Attractor-AE** |
|---|---|---|---|
| 표현 | 연속 | 이산 (STE) | **이산 (미분가능, IFT)** |
| 미분 | exact | **편향(STE)** | **exact(IFT)** |
| recon ↓ | 29.5 | **5.7** | 7.9 |
| fMIG ↑ | — | 0.289 | **0.307** |
| DCI-D ↑ | 0.420 | 0.584 | **0.617** |
| DCI-C ↑ | 0.276 | 0.293 | **0.302** |
| FactorVAE ↑ | 0.883 | **0.994** | 0.947 |
| 해석가능 코드 | — | ○ | **○ (basin 검증됨)** |

---

## 7. 핵심 주장 & 한계
**주장**:
1. **STE 없이 완전 미분가능한 이산 표현**을 attractor 동역학 + IFT로 학습 가능함을 실증.
2. 학습된 이산 코드는 **해석가능**(shape/hue가 attractor 단위로 의미)하고 **진짜 basin**을 형성.
3. Disentanglement(factor 단위 fMIG·DCI)에서 **VQ-VAE와 경쟁력 있거나 우위**.

**한계**:
1. recon이 VQ-VAE에 소폭 열위(7.9 vs 5.7).
2. **scale factor 분리 실패**(shape와 entangle), 일부 GT 약한 중복.
3. per-dim MIG는 다차원/factor 설계에 불리(metric 한계).
4. 학습이 effective-temperature 어닐링에 민감(collapse 회피 위해 soft 시작 필수 — `docs/0617_collapse_analysis.md`).

**다음**: scale 분리 개선, recon 격차 축소(K·구조), β/K 스윕(Phase E), variational 확장.
