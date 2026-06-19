# AQ vs (Straight-Through) Gumbel-Softmax — 관계 분석

> 세 방법 모두 "**미분가능한 이산 잠재**"를 다룬다. AQ가 Gumbel-Softmax·ST-Gumbel과
> 어떻게 같고 다른지, 그리고 왜 AQ는 **샘플링 없이** 작동하는지 분석.
> 관련: `0617_proposal.md`, `0618_categorical_aqvae_design.md`, `0617_collapse_analysis.md`(포화-소실).

---

## 0. 공통 문제
이산 잠재(어느 코드 $k$)를 신경망에 넣고 학습하려면 **argmax/샘플링이 미분 불가**라는 벽을 넘어야 한다. 세 방법(+AQ)은 이 벽을 다른 방식으로 우회한다.

## 1. Straight-Through Estimator (STE) — Bengio et al. 2013
- **forward**: hard(argmax → one-hot).
- **backward**: 항등으로 *가정*(gradient를 그냥 통과).
- 성격: **편향(biased)이지만 nonzero** gradient. VQ-VAE가 이걸 씀.

## 2. Gumbel-Softmax / Concrete — Jang et al. 2017; Maddison et al. 2017
- **Gumbel-Max trick**: $k=\arg\max_k(\log\pi_k+g_k),\ g_k\sim\mathrm{Gumbel}$ = Categorical($\pi$)의 *정확한 샘플*.
- **완화(relaxation)**: $y_k=\mathrm{softmax}_k\!\big((\log\pi_k+g_k)/\tau\big)$ — 샘플을 **연속**으로 풀어줌(심플렉스 위).
- **reparameterization**: 무작위성($g$)이 외부라 softmax로 gradient가 흐름 → **저분산 reparam gradient**.
- 온도 $\tau$: $\tau\!\to\!0$ one-hot(고분산), 크면 smooth(완화 편향). **forward가 soft 샘플**(이산 아님).

## 3. Straight-Through Gumbel-Softmax (ST-GS)
- **forward**: Gumbel-softmax 샘플의 **hard argmax**(one-hot, 이산).
- **backward**: soft Gumbel-softmax의 gradient(soft로 *가정*).
- 성격: **이산 forward + reparam(저분산) gradient**, 단 여전히 **편향**(soft≠hard).

## 4. AQ (Attractor Quantization)
- **forward**: attractor 동역학의 **고정점** $y^\star=\sum_k w^\star_k X_k$, $w^\star=\mathrm{softmax}(\beta X^\top(y^\star+c))$.
  - **샘플링 없음**. $y^\star$는 categorical $w^\star$ 하의 **패턴 기댓값(posterior 평균)** = 결정론적.
  - softmax가 *한 번*이 아니라 **수렴까지 반복**(고정점)된다는 점이 Gumbel과 다름.
- **gradient**: 고정점에서 **IFT(음함수 정리)로 정확(exact)**. STE의 가짜도, Gumbel의 완화 편향도 아님 — *진짜* gradient.
- **이산성**: $\beta$(=inverse temperature, stiffness)를 **어닐링**하면 $w^\star\!\to$ one-hot → $y^\star\!\to$ 단일 패턴(이산). 즉 이산성은 **다중안정 동역학으로의 수렴 + β 어닐링**에서 옴(노이즈+완화 아님).
- **노이즈/무작위성 없음**: 잠재 무작위성은 (변분 시) **prior 샘플(생성)** 에만, **KL은 closed-form**.

## 5. 한눈 비교
| | forward | gradient | 이산성 출처 | 샘플링 | 편향 |
|---|---|---|---|---|---|
| **STE** | hard | 항등(가짜) | argmax | (보통) 없음 | 편향·nonzero |
| **Gumbel-Softmax** | soft 샘플 | reparam(정확, soft 목적) | 노이즈+softmax 완화 | ✓(Gumbel) | 완화 편향 |
| **ST-Gumbel** | hard | soft(가짜) | 노이즈+argmax | ✓(Gumbel) | 편향 |
| **AQ** | **soft 기댓값**(결정론) | **IFT 정확** | **attractor 수렴 + β어닐링** | **✗**(생성 prior만) | 무편향(단 §7) |

## 6. 핵심 개념 차이
1. **샘플 vs 기댓값**: Gumbel/ST-GS는 *샘플*(stochastic). AQ는 *기댓값 디코딩*(결정론, mean-field). Categorical AQ-VAE는 "샘플하는 categorical VAE"가 아니라 **mean-field categorical VAE**.
2. **완화 vs 동역학**: Gumbel의 softness = 노이즈+온도의 *일회성 softmax 완화*. AQ의 softness = **반복 attractor 동역학의 고정점**(다중안정 basin). 이산성이 *완화*가 아니라 *수렴*에서 emerge.
3. **logit 출력 vs 검색(retrieval)**: Gumbel은 망이 logit $\pi$를 *직접* 출력. AQ의 "logit"은 학습된 저장패턴과의 **유사도 $X^\top(y+c)$**(associative memory) — 즉 **content-addressable 검색**.
4. **읽기 구조의 유사점**: $y^\star=\sum_k w^\star_k X_k$는 형식상 "**codebook 위 Gumbel-softmax 가중 lookup**"과 닮음 — 단 (a) 노이즈 없음, (b) 일회성 아닌 고정점, (c) IFT 정확 gradient. AQ ≈ *"노이즈 없는, 동역학 기반, 정확-gradient 버전의 codebook attention"*.

## 7. ★정직한 핵심 뉘앙스 — "무편향"의 함정
"AQ는 STE를 피한다"가 헤드라인이지만, **이산 극한($\beta\!\to\!\infty$)에서 AQ의 *정확* gradient는 0이 된다**(softmax 포화, `0617_collapse_analysis.md`). 반면 **STE의 *편향* gradient는 그 극한에서도 nonzero** — 바로 그 편향이 이산 병목을 학습시키는 힘이다.
> 즉 AQ는 "더 나은 gradient"가 아니라 **"정확하지만 사라지는 gradient"**, STE는 **"편향됐지만 살아있는 gradient"**. 둘 다 이산 극한에 약점이 있고, AQ의 약점은 **soft 영역 학습 + 어닐링**으로 푼다 — 이는 사실 **Gumbel의 온도 어닐링과 같은 처방**이다.
- 차이점: AQ는 매 온도에서 gradient가 *정확*(완화 편향 없음)하고 forward가 *결정론적*(샘플 분산 없음). Gumbel은 저분산이나 완화 편향, ST-GS는 이산 forward이나 backward 편향.

## 8. AQ를 stochastic하게 만들 수 있나?
가능하다 — $w^\star$에서 **Gumbel/ST로 코드를 샘플**해 패턴을 디코딩하면 진짜 stochastic VAE가 된다. 그러나 그 순간 **샘플링-gradient(완화/STE 편향)가 재도입**되어 AQ의 정체성(무편향·결정론·IFT)을 잃는다. 그래서 본 설계는 **의도적으로 기댓값 디코딩 + analytic KL**을 택한다.

## 9. 위치 짓기 (positioning)
- VQ-VAE = hard argmax + **STE**(편향).
- Gumbel-Softmax VAE = **샘플**(완화 편향, 저분산).
- ST-Gumbel = 이산 샘플 + 편향 backward.
- **AQ-VAE = 결정론적 기댓값(attractor 고정점) + IFT 정확 gradient + β어닐링 이산화**. 무작위성은 생성 prior로 분리, KL closed-form.

## 10. 한 줄
**AQ는 "noise-injected relaxation(Gumbel)"도 "fake-gradient(STE)"도 아닌, *attractor 동역학의 고정점을 IFT로 정확히 미분하는 결정론적 이산화*다. 이산성은 완화가 아니라 수렴에서, gradient는 가정이 아니라 음함수정리에서 온다. 대가는 이산 극한의 gradient 소실 — Gumbel식 온도(β) 어닐링으로 완화.**

## 참고
- Bengio, Léonard, Courville (2013) — Straight-Through Estimator.
- Jang, Gu, Poole (2017) — Categorical Reparameterization with Gumbel-Softmax.
- Maddison, Mnih, Teh (2017) — The Concrete Distribution.
- van den Oord et al. (2017) — VQ-VAE(STE codebook).
- Bai, Kolter, Koltun (2019) — Deep Equilibrium Models(IFT).
- Ramsauer et al. (2020) — Modern Hopfield(검색 readout).
