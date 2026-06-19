# Attractor-VAE: Attractor 동역학과 Implicit Differentiation을 통한 완전 미분가능 이산 표현 학습

> **문서 성격**: 연구 제안(design proposal) — 논문 형식. 일부는 확정, 일부는 *미결(Open)*로 표기.
> **관련 노트**: `related_paper/solve_the_loop.md`, `related_paper/Discrete, compositional, and symbolic representations through attractor dynamics.md`
> **참조 구현**: `related_paper/Attractor/` (Solve-the-Loop 공식 소스, MIT)

---

## 초록 (Abstract)

이산(discrete) 잠재 표현은 해석가능성·압축·조합성(compositionality) 측면에서 유용하지만, 기존 방법(VQ-VAE의 nearest-neighbor lookup, straight-through estimator, Gumbel-Softmax)은 **이산화 지점에서 미분이 끊겨** straight-through 같은 *편향된 근사 추정기*로 우회한다. 본 연구는 이산화를 **attractor 동역학의 고정점(fixed point)** 으로 재정의하여, 이 한계를 제거하는 것을 목표로 한다. 핵심 발상은 **attractor를 하나의 discrete concept으로 보는 것**이다 — 예컨대 모양(shape) factor는 삼각형 attractor·네모 attractor 등 유한 개의 안정 고정점 중 하나로 수렴한다. 수렴한 고정점을 디코딩해 입력을 재구성하고, 그 고정점의 gradient를 **implicit function theorem(IFT)** 으로 *해석적(analytic)* 으로 계산한다(Solve-the-Loop). 이로써 이산 표현 학습 전체가 **완전 미분가능(fully differentiable)** 해지며, straight-through 근사가 필요 없다. 본 문서는 PoC(proof-of-concept) 단계의 설계를 기술한다: 데이터셋은 **3D Shapes**, 잠재는 **z=32 = 8 factor × 4 dim**(factor별 독립 attractor 동역학), multi-stability 생성 방법(미결의 핵심 연구 문제)에 대한 후보군과 평가 계획을 포함한다.

---

## 1. 서론 (Introduction)

### 1.1 문제 의식
표현 학습에서 **이산 잠재 변수**는 여러 장점을 갖는다 — 개념의 명시적 분리, 낮은 비트 표현, 조합적 일반화. 그러나 이산화 연산(예: argmin codebook 선택)은 본질적으로 **미분 불가능**하다. 대표적 우회책:
- **VQ-VAE** (van den Oord et al., 2017): 연속 인코딩을 codebook의 최근접 코드로 양자화. 최근접 선택이 미분 불가 → **straight-through estimator(STE)** 로 gradient를 *복사*해 통과시킴(편향).
- **Gumbel-Softmax / Concrete** (Jang et al., 2017; Maddison et al., 2017): 이산 분포를 연속 완화 + 온도 어닐링. 여전히 완화·STE에 의존.
- **STE** (Bengio et al., 2013): forward는 이산, backward는 항등 — 근본적으로 *근사 gradient*.

이들의 공통 한계는 **forward(이산)와 backward(gradient)가 불일치**한다는 것이다. 즉 진정한 의미의 end-to-end 미분이 아니다.

### 1.2 핵심 발상: attractor = discrete concept
본 연구는 이산화를 **동역학계의 수렴**으로 본다. 연속 잠재 공간 위에 **multi-stable attractor 동역학**을 두면, 상태는 유한 개의 안정 고정점(attractor) 중 하나로 수렴한다. **각 attractor를 하나의 이산 개념에 대응**시킨다:

> 예) shape factor의 부분 공간에서 동역학을 굴리면, 입력이 삼각형이면 "삼각형 attractor"로, 네모면 "네모 attractor"로 수렴한다.

수렴한 고정점 $y^\star$ 를 디코더에 넣어 재구성하고, **전 과정을 backprop**한다. 이때 결정적 요소는 고정점의 gradient를 **해석적으로** 얻는 것이다.

### 1.3 왜 미분가능한가 — Implicit Differentiation
고정점 $y^\star$ 는 명시적 공식이 아니라 *방정식의 해*($y^\star = \mathcal{T}(y^\star)$)로 정의된다. **음함수 정리(IFT)** 를 쓰면 반복 궤적을 펼치지(unroll) 않고도 $y^\star$ 를 파라미터에 대해 미분할 수 있다(Bai et al., 2019의 Deep Equilibrium Models; Solve-the-Loop, 2026). 결과:
- **STE 같은 근사가 불필요** — gradient가 고정점에서 *정확*하다.
- solver 반복 수와 **무관하게 메모리 $O(1)$**.
- forward(이산화)와 backward(정확 gradient)가 **일치**.

따라서 본 방법은 **VQ-VAE·STE·Gumbel의 대체제**로, "이산 표현을 완전 미분가능하게 학습"한다.

### 1.4 기여 (목표)
1. 이산화를 attractor 고정점으로 정식화하고, IFT로 **완전 미분가능**하게 학습하는 프레임(Attractor-VAE) 제안.
2. **factor별 독립 attractor 동역학**(2D→확장 4D, 8 factor)으로 disentangle된 이산 표현 유도.
3. multi-stability(이산 attractor 다수 생성)를 위한 방법론 정리 및 PoC 설계.

---

## 2. 배경 및 관련 연구 (Background)

### 2.1 Solve-the-Loop — Attractor Models (Fein-Ashley & Rashidinejad, 2026)
looped/recurrent 모델을 유한 unroll하지 않고 **고정점을 직접 solve**한다. backbone이 초기 추측 $\tilde y_0$ 를 제안하고(warm start), 작은 weight-tied attractor head $\mathcal{T}_{\theta_a}$ 를 고정점까지 반복:
$$\tilde y^\star = \mathcal{T}_{\theta_a}(\tilde y^\star, \tilde y_0), \qquad f(y,c) = \mathrm{Head}(y+c) - y,\ \ y^\star = f(y^\star,c) \Leftrightarrow \mathrm{Head}(y^\star+c)=y^\star+c.$$
- **persistent injection**: context $c$ 를 매 step 재주입 → 입력 무관 고정점으로의 붕괴 방지.
- **solver**: Anderson acceleration, 상대 잔차 $\|f\|/\|y\|<\varepsilon$ 로 적응적 종료.
- **backward(IFT)**: $\partial L/\partial\theta = u^\top \partial f/\partial\theta$, $u=(I-J^\top)^{-1}v$. **one-step(Neumann-1) 근사** $u\approx v + J^\top v$ → vector–Jacobian product 한 번, 메모리 $O(1)$.
- **안정성**: $(I-J^\top)^{-1}$ 가 비수축 영역에서 폭발 → gradient가 그 영역을 자동 회피. LayerScale 게이트를 작게 초기화해 시작부터 수축적. `adjoint_clip` 안전장치.
- **equilibrium internalization**: 학습이 진행되면 backbone의 $\tilde y_0$ 가 $y^\star$ 에 근접 → 추론 시 solver를 거의 빼도 성능 유지.

본 연구는 이 중 **고정점 solver + IFT + persistent injection + 수축 게이트**를 이식한다(참조 소스 `attractor/models/attractor/solvers.py`, `attractor.py::_IFTAttach`).

### 2.2 Attractor 동역학을 통한 이산·조합 표현 (Nam et al., 2024)
연속 잠재 위 **stochastic attractor 동역학**으로 공간을 discrete basin들로 자생 분할하고, 각 attractor를 symbolic sequence에 대응시킨다. discreteness를 아키텍처(slot/codebook)에 *박지 않고* dynamics에서 *emerge*시킨다는 점이 본 연구와 철학을 공유한다. 단 그쪽은 학습이 **GFlowNet-EM**(분포 학습)인 반면, 본 연구는 **결정론적 고정점 + IFT**(정확 gradient)를 택한다 — 두 노트의 §8.4에서 지적된 교차점이다. (벤치마크로 dSprites+color를 VAE pre-train 위에 사용한 선례.)

### 2.3 VAE와 disentanglement
VAE(Kingma & Welling, 2014)와 β-VAE(Higgins et al., 2017)를 기반으로 한다. disentanglement 정량 지표로 MIG(Chen et al., 2018), DCI(Eastwood & Williams, 2018), FactorVAE score(Kim & Mnih, 2018)를 사용한다(§5).

---

## 3. 방법 (Method)

### 3.1 전체 구조
```
x(RGB 64×64×3) ──conv encoder──▶ c(x) ∈ R^32          # backbone: 초기 context / warm start
                                   │ factor별로 분할 (8 × 4dim)
        각 factor i: y_i* = AttractorDynamics_i(y_i, c_i)  # multi-stable, 고정점=이산 개념
                                   │ (no-grad solve → IFT로 grad 부착)
                  y* = concat(y_1*, …, y_8*) ∈ R^32
                                   │
            (확률화) mu, logvar 또는 y* 직접 → reparam → z
                                   │
                         z ──conv decoder──▶ x̂        # RGB
손실: Recon(Gaussian/MSE) + β·KL(q(z|x)‖N(0,I))
```

### 3.2 잠재 구조 — factor별 독립 attractor (확정)
- **z = 32 = 8 factor × 4 dim.** 처음부터 4차원씩 끊어 **factor마다 별도 attractor 동역학** $f_i$ 를 돌린다.
- **factor 차원 = 4** 로 확장(2D는 최대 cardinality 15(orientation)를 담기엔 too compact). 4D는 多 basin 분리에 여유.
- 3D Shapes의 실제 factor는 6개 → 8 slot은 여유 2개. (잉여 slot이 trivial attractor로 collapse하는지 관찰 가능.)
- **시각화**: 각 factor 4D를 **PCA로 2D 투영**, ground-truth factor 값으로 색칠해 "basin이 이산 factor 레벨과 정렬되는가" 진단.

### 3.3 Attractor map과 고정점 (이식)
factor별로 Solve-the-Loop과 동일한 사상:
$$f_i(y, c_i) = \mathrm{Head}_i(y + c_i) - y, \qquad y_i^\star = \text{RootFind}(f_i;\ y_0=c_i).$$
- forward: `c.detach()` 로 no-grad Anderson 반복(`solvers.py` 이식).
- backward: `_IFTAttach` 로 IFT grad 부착, one-step Neumann 기본(`implicit.py` 이식·단순화). per-factor Jacobian은 4×4로 가벼움.

### 3.4 Multi-stability — 핵심 연구 문제 (★미결, 별도 논의 예정)
단순 수축적 Head는 *유일한 연속 고정점*만 준다. 이산성을 위해서는 동역학이 **multi-stable**(여러 basin)이어야 하며, 연속 입력 $c$ 가 한 basin을 *선택*해 같은 attractor로 snap해야 한다. 후보 방법:

**① 명시적 codebook + well 에너지 (differentiable VQ).**
factor마다 학습 가능한 점 $\{a_1,\dots,a_K\}$, 에너지 $E(y)=-\tau\log\sum_k e^{-\|y-a_k\|^2/\tau}$ 의 gradient flow. $\tau\to0$ 이면 hard nearest-neighbor = VQ이지만, lookup이 아니라 동역학이라 **IFT로 미분 가능**. $K$ 는 미리 지정.

**② Modern Hopfield Network (Ramsauer et al., 2020) — 유력 후보.**
저장 패턴 $X\in\mathbb{R}^{4\times K}$(=학습 codebook), 업데이트 $y \leftarrow X\,\mathrm{softmax}(\beta X^\top y)$. 저장 패턴이 곧 attractor.
- **inverse temperature $\beta$ = "stiffness 노브"**: $\beta$ 크면 각 패턴이 개별 sharp attractor($K$개), 작으면 패턴이 **병합돼 적은 수**의 metastable attractor.
- 즉 "**개수 미리 지정(option 1)**"과 "**stiffness로 개수 조절(option 2)**"을 한 틀에서 통합: $K$ 는 상한, $\beta$ 가 실제 활성 개수를 연속 조절.

**③ Emergent + stiffness regularizer (option 2 순수형).**
codebook 없이 자유 Head + Jacobian/Lipschitz/에너지 곡률 페널티 $\lambda$. $\lambda$ 크면 단일 고정점, 작으면 多 attractor. 직관에 부합하나 "$\lambda \to$ 정확한 개수" 보장이 어렵고 통제가 간접적.

**현재 방침(잠정)**: $K$ 를 상한으로 **고정(예: 16/factor)** 하고 **temperature $\beta$(stiffness)를 노브**로 두는 ②/① 조합을 PoC 시작점으로 본다. $\beta$ 스윕/어닐링으로 실제 활성 attractor 수를 관찰. (최종 확정은 별도 논의.)

**IFT 주의**: multi-stable 시스템은 각 stable 고정점에서 국소 수축적 → IFT 적용 가능. 단 **basin 경계/saddle 근처**에서 $J$ 스펙트럴 반경→1 → $(I-J^\top)^{-1}$ ill-conditioned → `adjoint_clip` 필요.

### 3.5 손실
- **Recon**: 3D Shapes는 RGB 연속값 → Bernoulli BCE 대신 **Gaussian/MSE**(또는 logistic).
- **KL**: 표준 $D_{\mathrm{KL}}(q(z|x)\|\mathcal{N}(0,I))$, β 가중.
- attractor 관련 추가 손실(halting 등)은 두지 않음(Solve-the-Loop 철학).

---

## 4. 실험 설계 (Experimental Setup)

### 4.1 데이터셋 — 3D Shapes (확정)
DeepMind 3D Shapes: 64×64×**3(RGB)**, 480,000장, **6개 독립 factor, fully crossed**:

| factor | cardinality |
|---|---|
| floor hue | 10 |
| wall hue | 10 |
| object hue | 10 |
| scale | 8 |
| shape | 4 |
| orientation | 15 |

- 최대 cardinality 15 → 4D attractor 부분공간에 basin 배치가 현실적(dSprites의 orientation=40/위치=32보다 유리).
- 모든 factor가 본질적 이산값 → "attractor=discrete concept" 의미적 정합.
- 포맷: HDF5(`3dshapes.h5`, GCS 호스팅) → **h5py 의존성** 필요, 원본 수 GB.

> dSprites는 PoC 대상이 아니라 **파이프라인 sanity 및 참고 베이스라인**으로만 사용(§6).

### 4.2 모델
- conv encoder(64→4, ×4) → flat → $c\in\mathbb{R}^{32}$, RGB 입력(in_channels=3).
- factor별 attractor head ×8(각 4D), §3.4의 multi-stability 메커니즘.
- conv decoder(out_channels=3).
- baseline: 동일 backbone의 **표준 VAE** 및 **VQ-VAE**(직접 비교 대상).

### 4.3 평가 지표
**(a) 표현 품질**: ELBO/recon/KL, 재구성 샘플. VQ-VAE·표준 VAE와 동일 조건 비교.

**(b) Disentanglement** (3D Shapes ground-truth factor): **MIG + DCI + FactorVAE score**. 표현은 attractor 고정점(또는 $\mu$).

**(c) 이산성·attractor 진단**:
- 활성 attractor 수(factor별), codebook 사용률(dead attractor 비율).
- 수렴: solver iters, rel_residual, converged 비율.
- **perturbation-복귀**(Nam, §4.1): $y^\star$ 교란 후 재수렴 거리 → basin 검증.
- **equilibrium internalization**(Solve-the-Loop): $\|y^\star-c\|/\|c\|$, 추론 시 solver step 축소에도 성능 유지 여부.

---

## 5. 평가 프로토콜과 비교군 (Evaluation Protocol)
- **주 비교**: Attractor-VAE vs VQ-VAE vs (Gumbel-Softmax VAE) — 동일 데이터·backbone·latent 예산에서 recon·disentanglement·gradient 충실도 비교.
- **핵심 가설**: Attractor-VAE는 STE 없이(완전 미분가능) VQ-VAE에 준하거나 우월한 이산 표현·disentanglement를 달성한다.

---

## 6. 예비 결과 (Preliminary Results)
파이프라인 검증을 위해 **dSprites(1채널)** 에서 표준 VAE(β=1, latent_dim=10)를 수렴까지 학습:
- early stopping으로 epoch 131에 종료, **best loss 33.97** @ epoch 121 (recon ≈ 12.3, KL ≈ 21.6).
- 에폭당 ~42초(RTX 4090 단일). 데이터 로더·학습·체크포인트·재구성 저장 파이프라인 정상 동작 확인.

> 이는 **sanity check**이며, 실제 PoC 베이스라인은 3D Shapes(RGB)에서 재수립한다.

---

## 7. 단계별 로드맵 (Roadmap)
- **Phase 0** (완료): VAE 파이프라인 sanity(dSprites).
- **Phase 1**: 3D Shapes 전환(RGB·MSE·h5py) + 표준 VAE/VQ-VAE 베이스라인.
- **Phase 2**: factor별 attractor 동역학 + multi-stability 메커니즘(§3.4) 구현 → 완전 미분가능 이산 학습 검증.
- **Phase 3**: disentanglement(MIG/DCI/FactorVAE) 비교, $\beta$/stiffness 스윕, 이산성·internalization 진단.
- **Phase 4**(확장): stochastic·multimodal 일반화(Nam 방향)와의 연결.

---

## 8. 미결 사항 (Open Questions)
1. **Multi-stability 메커니즘**(핵심): ① VQ-well / ② Modern Hopfield+β / ③ stiffness regularizer 중 채택 — 별도 논의.
2. **factor당 attractor 수 $K$**: 고정(예: 16) vs emergent. factor마다 다른 cardinality(4~15) 처리.
3. **Backward**: one-step Neumann(기본) vs 정확 adjoint(Anderson).
4. **확률화 위치**: 고정점 $y^\star$ → (mu, logvar) vs $y^\star$ 직접 디코딩. reparam과 고정점의 분리.
5. **격리/구현 구조**: baseline 코드(`src/model_vae.py`, `src/train.py`)와의 분리 방식.

---

## 참고문헌 (References)
- Fein-Ashley, J., Rashidinejad, P. (2026). *Solve the Loop: Attractor Models for Language and Reasoning.* arXiv:2605.12466.
- Nam, A. J., Elmoznino, E., Malkin, N., McClelland, J. L., Lajoie, G., Bengio, Y. (2024). *Discrete, compositional, and symbolic representations through attractor dynamics.* arXiv:2310.01807.
- van den Oord, A., Vinyals, O., Kavukcuoglu, K. (2017). *Neural Discrete Representation Learning (VQ-VAE).* NeurIPS.
- Jang, E., Gu, S., Poole, B. (2017). *Categorical Reparameterization with Gumbel-Softmax.* ICLR.
- Maddison, C. J., Mnih, A., Teh, Y. W. (2017). *The Concrete Distribution.* ICLR.
- Bengio, Y., Léonard, N., Courville, A. (2013). *Estimating or Propagating Gradients Through Stochastic Neurons (STE).* arXiv:1308.3432.
- Bai, S., Kolter, J. Z., Koltun, V. (2019). *Deep Equilibrium Models.* NeurIPS.
- Ramsauer, H., et al. (2020). *Hopfield Networks is All You Need.* arXiv:2008.02217.
- Kingma, D. P., Welling, M. (2014). *Auto-Encoding Variational Bayes.* ICLR.
- Higgins, I., et al. (2017). *β-VAE.* ICLR.
- Burgess, C. P., Kim, H. (2018). *3D Shapes Dataset.* / Burgess et al., *Understanding disentangling in β-VAE.*
- Chen, R. T. Q., et al. (2018). *Isolating Sources of Disentanglement in VAEs (MIG).* NeurIPS.
- Eastwood, C., Williams, C. K. I. (2018). *A Framework for Quantifying Disentanglement (DCI).* ICLR.
- Kim, H., Mnih, A. (2018). *Disentangling by Factorising (FactorVAE).* ICML.
