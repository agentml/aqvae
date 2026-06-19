# attractor_vae — 설계 문서 (실무)

> 논문 서술은 `docs/0617_proposal.md`. 본 문서는 **결정사항 · 코드구조 · 로드맵 · 미결**에 집중한다.
> 코드 구현은 별도 허락 후 시작.

---

## 1. 목표
이산(discrete) 표현을 **완전 미분가능(fully differentiable)** 하게 학습 → **VQ-VAE / straight-through / Gumbel의 대체제**.
핵심: **attractor = discrete concept**. 잠재 동역학이 유한 개 고정점 중 하나로 수렴(예: shape → 삼각형/네모 attractor), 그 고정점을 디코딩·재구성하고 **Solve-the-Loop의 analytic IFT gradient**로 backprop. STE 같은 근사 불필요.

---

## 2. 확정된 결정
| 항목 | 결정 | 비고 |
|---|---|---|
| 데이터셋 | **3D Shapes** (RGB 64×64×3, 480k, factor 6개) | 최대 cardinality 15 → 4D basin 배치 현실적. h5py 필요 |
| 잠재 구조 | **z = 32 = 8 factor × 4 dim** | factor당 4D, slot 8개(실제 factor 6 + 여유 2) |
| factor 동역학 | **factor별 독립 attractor** $f_i$ (8개) | 4차원씩 끊어 별도 dynamics |
| gradient | **IFT (Solve-the-Loop 이식)** | one-step Neumann 기본, 메모리 $O(1)$ |
| 손실 | Recon = **Gaussian/MSE** (RGB) + β·KL | BCE 아님 |
| 시각화 | factor 4D → **PCA 2D**, ground-truth로 색칠 | basin↔이산 레벨 정렬 진단 |
| 비교군 | **VQ-VAE**, 표준 VAE | 동일 backbone·예산 |

---

## 3. 구조 개요
```
x(RGB) ─conv enc─▶ c ∈ R^32 ──(8×4 분할)──▶ c_1..c_8
   factor i:  y_i* = RootFind  f_i(y,c_i)=Head_i(y+c_i)−y      # multi-stable
              (no-grad Anderson solve → IFT로 grad 부착)
   y* = concat(y_1*..y_8*) ─▶ (mu,logvar 또는 직접) ─reparam─▶ z ─conv dec─▶ x̂
손실: MSE(x̂,x) + β·KL
```

---

## 4. Multi-stability — 핵심 미결 (별도 논의)
단순 수축 Head는 *유일 연속 고정점*만 줌. 이산성엔 **multi-stable**(여러 basin) 필요. 후보:
- **① differentiable VQ well**: 학습 codebook $\{a_k\}$ + 에너지 $E=-\tau\log\sum e^{-\|y-a_k\|^2/\tau}$ gradient flow. $\tau\to0$ = VQ이되 동역학이라 IFT 미분가능. $K$ 지정.
- **② Modern Hopfield (유력)**: $y\leftarrow X\,\mathrm{softmax}(\beta X^\top y)$, 저장패턴=attractor. **$\beta$(=stiffness 노브)**: 크면 $K$개 sharp, 작으면 병합돼 적은 수 → option1(개수 지정)+option2(stiffness 조절) 통합.
- **③ stiffness regularizer**: 자유 Head + Jacobian/Lipschitz/곡률 페널티 $\lambda$. 직관적이나 개수 통제 간접적.

**잠정 방침**: $K$ 상한 고정(예: 16) + $\beta$ 노브(②/①). $\beta$ 스윕/어닐링으로 활성 개수 관찰.
**IFT 주의**: stable 점에선 수축적(IFT OK), **basin 경계/saddle**에서 $(I-J^\top)^{-1}$ ill-conditioned → `adjoint_clip`.

---

## 5. Solve-the-Loop 이식 부품
| 부품 | 원본 | 이식 |
|---|---|---|
| 고정점 solver | `attractor/models/attractor/solvers.py` (`anderson_solve`,`fpi_solve`) | 거의 그대로 (범용 `(B,…)`) |
| Implicit diff | `attractor.py::_IFTAttach`,`_maybe_clip_jv` | 단순화(단일 GPU), one-step Neumann |
| attractor map | `attractor.py::_f` (`Head(y+c)−y`) | Head를 factor별 모듈로 |
| persistent injection | `_f` 의 `y+c` | 동일 |
| 수축 게이트 | `FixedPointBlock.raw_gamma` | 스칼라/채널 게이트, 작게 초기화 |
| adjoint clip | `adjoint_grad_clip` | 그대로 |
> 원본 MIT — 이식 코드에 출처 표기.

---

## 6. 코드 구조 (제안, 구현은 허락 후)
```
src/
  model.py              # 기존 VAE (baseline, 불변)
  dataset.py            # dSprites (sanity) + 3D Shapes(h5py) 추가
  train.py              # --arch {vae,vqvae,attractor} 플래그
  model_attractor.py    # AttractorVAE: enc → factor별 attractor → dec
  fixed_point/
    solvers.py          # anderson/fpi (이식)
    implicit.py         # IFTAttach (이식·단순화)
    multistable.py      # ①well / ②hopfield / ③regularizer
  eval/                 # MIG·DCI·FactorVAE + attractor 진단(perturb·internalize)
docs/{0617_proposal.md,0617_design.md}
```
- baseline(`model.py`,`train.py`) 최소 변경·격리.

---

## 7. 평가
- **품질**: ELBO/recon/KL, 재구성. VQ-VAE·VAE 동일조건.
- **Disentanglement**: MIG + DCI + FactorVAE (표현 = 고정점/μ).
- **이산성·attractor 진단**: 활성 attractor 수·dead 비율, solver iters/residual/converged, **perturbation-복귀**(basin), **internalization**($\|y^\star-c\|/\|c\|$ · solver step 축소 강건성).

---

## 8. 로드맵
- **P0**(완료): VAE 파이프라인 sanity — dSprites 표준 VAE 수렴, loss 33.97 @ e121.
- **P1**: 3D Shapes 전환(RGB·MSE·h5py) + VAE/VQ-VAE 베이스라인.
- **P2**: factor별 attractor + multi-stability(§4) 구현 → 완전 미분가능 이산 학습 검증.
- **P3**: disentanglement 비교, β/stiffness 스윕, 이산성·internalization 진단.
- **P4**(확장): stochastic·multimodal(Nam 방향) 연결.

---

## 9. 미결 사항
1. **Multi-stability 메커니즘** ①/②/③ — 핵심, 별도 논의.
2. **factor당 $K$**: 고정(16) vs emergent; factor별 cardinality(4~15).
3. **Backward**: one-step vs 정확 adjoint.
4. **확률화 위치**: $y^\star$→(mu,logvar) vs $y^\star$ 직접 디코딩.
5. **격리 방식**: `--arch` 플래그 vs 별도 파일.
