# Gaussian-AQ-VAE — context에 Gaussian 노이즈를 주는 stochastic AQ 설계

> AQ를 stochastic·생성 모델로 만드는 또 다른 길: 이산 코드를 샘플하는 대신
> **연속 context $c$에 Gaussian 노이즈(reparameterization)** 를 주고, attractor가 이를 이산화.
> 상태: **설계(미구현)**. 비교 대안: `categorical_aqvae_design.md`(이산 직접 샘플 대신 기댓값).
> 관련: `collapse_analysis.md`(magnitude 규제), `aq_vs_gumbel.md`, `neuroscience_significance.md`.

---

## 1. 동기
이산 코드 $k$에서 직접 샘플하면 Gumbel/STE 편향이 재도입된다(`aq_vs_gumbel.md`). 대신 **확률성을 연속 공간($c$)에 두면** — Gaussian **reparameterization**으로 *무편향·저분산* gradient를 쓰고, attractor가 그 연속 잠재를 이산화한다. 즉 "샘플 → 이산화"가 아니라 "**연속 샘플 → attractor 수렴 → 이산**".

## 2. 구조 (생성 과정)
$$x\xrightarrow{\text{enc}}(\mu_c,\sigma_c)\xrightarrow{\text{reparam}} c=\mu_c+\sigma_c\odot\varepsilon\ (\varepsilon\sim N(0,I))\xrightarrow{\text{attractor}} y^\star(c)\xrightarrow{\text{dec}}\hat x$$
$$\mathcal{L}=\underbrace{\mathrm{recon}(\hat x,x)}_{\text{decode}(y^\star)}+\beta_{\mathrm{KL}}\,\mathrm{KL}\big(N(\mu_c,\sigma_c^2)\,\|\,N(0,I)\big)$$
- 인코더 마지막 Linear → $(\mu_c,\log\sigma_c^2)\in\mathbb{R}^{32}\times\mathbb{R}^{32}$(factor별 슬라이스로 해석).
- **KL closed-form**: $\tfrac12\sum_d(\mu_d^2+\sigma_d^2-\log\sigma_d^2-1)$, 배치 평균.
- attractor 부분은 그대로(IFT 정확 gradient). 노이즈는 forward의 $c$ 한 곳에만.

## 3. ★핵심 시너지
1. **Gumbel/STE 없는 stochastic** — Gaussian reparam = 무편향·저분산. AQ 정체성 유지하며 *진짜 VAE*.
2. **collapse의 magnitude 규제를 KL이 겸함** — Gaussian KL의 $\mu^2$ 항이 $\|\mu_c\|$를 **unit-Gaussian shell(반경 $\sqrt d$)** 로 당김. *0이 아니라* 적정 반경 → 우리가 collapse 막으려 넣은 **`context_norm`을 KL이 대체**. (`collapse_analysis.md` §5의 "KL은 0으로 당겨 나쁘다"는 *연속 c*에서 해소: $N(0,I)$ 질량은 0이 아닌 반경 $\sqrt d$.) **σ가 collapse(→0)해도 $\mu^2$ 항이 magnitude를 계속 규제**.
3. **calibrated 불확실성** — $\sigma_c$가 입력별로 학습 → **basin 경계 근처 애매한 입력은 큰 σ(코드 흔들림=stochastic), basin 깊은 확신 입력은 작은 σ(결정론)**. 범주지각의 불확실성과 호응(`neuroscience_significance.md`).
4. **생성 단순** — $c\sim N(0,I)$ → attractor → decode. categorical의 학습 prior/autoregressive 불필요. 이산 코드 분포는 $N(0,I)$의 basin pushforward(자동).

## 4. ★기하 결정 — context_norm 대체 (추천 A)
현 `context_norm`은 $c$를 *구면*에 둠 → Gaussian과 불일치. 두 선택:
- **(A) context_norm 제거 + Gaussian $c$ + $N(0,I)$ prior** — KL이 magnitude 자동 규제(§3.2). **가장 깔끔**, 추천.
- (B) **구면 유지 + von Mises–Fisher**(Hyperspherical VAE, Davidson et al. 2018): 방향 평균 $\mu$ + 집중도 $\kappa$, KL(vMF‖uniform). context_norm과 정합하나 구현 복잡(vMF 샘플·KL).

→ **A 채택**. effective temperature $=\beta\|X\|\|c\|$에서 $\|c\|\!\sim\!\sqrt d\!\approx\!2$(KL이 고정), 이산화는 기존 **β 어닐링**이 담당.

## 5. 생성
```
c ~ N(0, I)  →  y* = attractor.infer(c)  →  decode  →  새 이미지
```
- (옵션) factor별 독립 N(0,I). 이산 코드 분포는 basin 기하가 정함.

## 6. 구현 변경
- 인코더 출력 차원 $32\to 64$ ($\mu,\log\sigma^2$). `--variational gaussian` 모드.
- forward: `c = mu + exp(0.5*logvar)*eps`(train), eval은 `c=mu`. context_norm off(모드 A).
- 손실 `gaussian_aqvae_loss`: recon + β_KL·Gaussian-KL.
- `generate`: `c=randn(...)` → infer → decode.
- train.py: `--variational {categorical, gaussian}` 로 일반화(현 categorical과 병렬).

## 7. categorical 버전과 비교
| | Categorical | **Gaussian-in-c** |
|---|---|---|
| 확률 잠재 | 이산(직접) | 연속 $c$ (이산은 파생) |
| reparam | 없음(기댓값 디코딩) | **Gaussian(무편향·저분산)** |
| prior | 학습 categorical | **N(0,I)** (단순) |
| collapse 규제 | 별도 context_norm | **KL이 겸함** |
| 불확실성 | $w^\star$ 엔트로피 | **입력별 σ(calibrated)** |
| 생성 | factorized(흐릿 관측) | $c\sim N(0,I)$ |
| 이산 코드 분포 | 명시적 $p(k)$ | 암묵적(basin pushforward) |
→ **상보적**. 둘을 *대안 모드*로 두고 recon·disentanglement·생성에서 비교하는 게 흥미로운 결과.

## 8. 스펙트럼 — Nam 2024로의 확장
- **노이즈를 $c$(입력)에만** = 본 설계(가장 단순한 stochastic AQ).
- **노이즈를 매 solver step에** = SDE/Langevin = **Nam 2024 stochastic attractor** 방향.
→ 본 설계는 그 *가벼운 입구*. 자연스럽게 step-wise 노이즈(SDE)로 확장 가능.

## 9. 리스크·미결
- **σ collapse**(β_KL↓ → σ→0 → 결정론, stochasticity 소실). 단 $\mu^2$ 규제는 유지 → magnitude collapse는 아님. β_KL 워밍업으로 균형.
- **σ vs basin 폭**: σ가 너무 작으면 attractor가 노이즈를 씻어내 코드 안 흔들림. 경계 stochasticity 위해 σ~basin 간격(β·σ 조율).
- **β 어닐링 × Gaussian KL** 상호작용(두 정규화) 튜닝.
- 모드 A에서 **context_norm 제거**가 학습 안정성에 미치는 영향(KL이 충분히 규제하는지) 검증 필요.

## 10. 단계
1. `--variational gaussian` 구현(μ/σ 인코더, reparam, Gaussian KL, context_norm off).
2. 스모크: KL·recon·σ 거동, generate.
3. 본 학습 → categorical·deterministic과 **recon·disentanglement·생성** 3자 비교.
4. (확장) vMF(모드 B) 또는 step-wise 노이즈(SDE).

## 11. 미결 결정
1. 기하: **(A) Gaussian + context_norm 제거**(추천) vs (B) vMF 구면.
2. σ 파라미터화: factor별 vs dim별; β_KL 스케줄.
3. categorical과 **통합 플래그**(`--variational {categorical,gaussian}`) 구조.
