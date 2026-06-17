# Attractor-AE의 Representation Collapse: 진단·기전·해결

> **문서 성격**: 연구 노트(논문 작성용 상세 기록). deterministic attractor-AE 학습 초기에 관측된
> representation collapse를, 가설→실험→기전분석→해결 순으로 빠짐없이 기록한다.
> **관련**: `docs/proposal.md`, `docs/design.md`, 부품 검증 `src/fixed_point/_verify_c*.py`,
> 본 분석 실험 `src/_smoke_anneal.py`, `src/_smoke_ctxnorm.py`.
> **환경**: 3D Shapes(RGB 64×64×3, 480k), z=32=8 factor×4dim, Modern Hopfield(K=16), RTX 4090.

---

## 초록 (Abstract)

완전 미분가능 이산 표현을 위한 **deterministic attractor-AE**(VQ-VAE 대체 목표)를 조립(D1)하자,
학습 즉시 **representation collapse**가 발생했다: factor당 활성 attractor가 1개로 붕괴하고
(8/128), 재구성 손실이 데이터 분산 수준(recon≈1500)에서 정체했다. 본 노트는 이 collapse를
세 단계로 규명한다. (1) **β 어닐링 가설**은 기각된다 — 인코더가 context 크기 $\|c\|$를 폭발시켜
($c\_std$ 74→542) β 통제를 무력화한다. (2) **context 정규화**(magnitude 제한)만으로도 부족하다 —
$\|c\|$를 묶어도(1.47) collapse가 유지된다. (3) 근본 원인은 **softmax 포화**이며, 포화 지점에서
**exact IFT gradient가 ≈0**이 되어 인코더가 학습 신호를 잃는다. 이는 "정확한 gradient"가 이산
극한에서 **편향됐지만 nonzero인 STE보다 오히려 불리**함을 시사한다. 해결책은 **effective
temperature**($\beta\cdot\|X\|\cdot\|c\|$)를 통제 대상으로 삼아 **비포화(soft) 영역에서 학습 후
이산으로 어닐링**하는 것이다. soft 영역 실험에서 collapse가 깨지고(recon 1500→333, 활성
8→36/128) 학습이 회복됨을 실증했다. 나아가 **β 어닐링(soft→discrete)으로 이를 완결**했다:
패턴·context를 단위/고정 노름으로 묶어 effective temperature를 β 단일 노브로 만들고, soft
warmup 후 β를 1→8로 올리자 **재collapse 없이** 이산화(dist→pat 0.61→0.10)·다양성 증가(활성
8→90/128)·recon 개선(1500→182)이 동시에 일어났다. 즉 "train soft, anneal to discrete"가
end-to-end로 작동한다.

---

## 1. 배경: deterministic attractor-AE

이산 잠재 표현을 **완전 미분가능**하게 학습하는 것이 목표다(VQ-VAE의 argmin+STE 대체). 구조:
$$x \xrightarrow{\text{conv enc}} c\in\mathbb{R}^{32} \xrightarrow{\text{FactorizedHopfield}} y^\star \xrightarrow{\text{conv dec}} \hat{x}, \quad L=\mathrm{MSE}(\hat{x},x).$$
factor $i$(4차원)마다 Modern Hopfield 사상으로 고정점을 푼다(persistent injection):
$$T_i(y;c_i) = X_i\,\mathrm{softmax}\big(\beta\,X_i^\top(y+c_i)\big), \qquad y_i^\star = T_i(y_i^\star;c_i).$$
- $X_i\in\mathbb{R}^{4\times K}$: 학습 가능한 저장패턴(=이산 개념), $K=16$.
- 디코더에는 **고정점 $y^\star$ 자체**를 넣는다(argmin snap 아님) → IFT로 정확 미분, STE 불필요.
- 부품(solver·IFT·Hopfield)은 단독 검증 완료(Phase B/C): IFT gradient가 닫힌형과 1e-14 일치,
  multi-stability·β노브·factor 독립성 확인.

**비포화 vs 포화 — 사전 관측(Phase C2)**: 단독 모듈에서 이미 *discreteness↔gradient* trade-off를
관측했다. β=2(패턴노름3)에서 $|\partial L/\partial c|=30$(gradient 양호)·dist→pat=0.11(준이산),
β≥5에서 dist→pat→0(이산)·$|\partial L/\partial c|\to0$(gradient 소멸). 이 trade-off가 *학습
동역학*으로 발현된 것이 본 collapse다.

---

## 2. 현상: D1에서의 즉시 collapse

모델 조립 직후 스모크(3D Shapes 배치, β=2 고정, context 정규화 없음, Adam 1e-3):

| 지표 | 관측 |
|---|---|
| 형상·gradient | recon_logits (B,3,64,64), $y^\star$(B,32) 정상; **encoder·patterns 모두 grad 흐름** ✓ |
| recon | 1977 → 1888 (8 step, 미미) |
| **dist→pat** | **0.000** (= $y^\star$가 패턴에 완전히 붙음 = 포화) |
| **factor별 활성 attractor** | **[1,1,1,1,1,2,1,2]** (factor당 ~1개) |

즉 미분 경로는 정상이나(STE 없이 인코더까지 grad), **표현이 factor당 1개 코드로 붕괴**했다.
이는 VQ-VAE의 codebook collapse와 동형이다. 이하 원인을 추적한다.

---

## 3. 가설 1 — β 어닐링 (기각)

**가설**: C2의 trade-off에 근거, 저 β로 시작하면 gradient가 흘러 인코더가 잠재를 조직화하고,
β를 올리며 이산화하면 collapse를 피한다.

**실험**(`_smoke_anneal.py`, 각 400 step): 고정 β=2.0 / 고정 β=0.5 / 어닐링 β 0.3→4.0.

| 스케줄 | 최종 recon | 활성(/128) | **c_std** | dist→pat |
|---|---|---|---|---|
| 고정 β=2.0 | 1497 | 8 | **74.2** | 0.000 |
| 고정 β=0.5 | 1491 | 8 | **542.8** | 0.000 |
| 어닐링 0.3→4.0 | 1499 | 8 | **486.1** | 0.000 |

**결과: 전부 collapse(8/128), recon 정체.** 가설 기각.

**핵심 단서 — $c\_std$ 폭발**: β를 낮춰도 $\|c\|$가 폭증(β=0.5에서 542)했다. 즉 인코더가 context
크기로 효과를 상쇄해 β 통제를 무력화했다. → §5.3에서 기전 설명.

---

## 4. 가설 2 — context 정규화 (부분 기각)

**가설**: collapse는 $\|c\|$ 폭발(→포화 강제) 때문이다. factor별 $c$를 고정 노름으로 묶으면
인코더는 *크기*가 아니라 *방향*으로 패턴을 골라야 해 다양성이 강제된다.

**구현**: `_norm_context` — $c_\text{factor} \leftarrow c_\text{factor}/\|c_\text{factor}\|\cdot s$.

**실험**(`_smoke_ctxnorm.py`, 각 300 step):

| config | recon | 활성(/128) | c_std | dist→pat |
|---|---|---|---|---|
| ctxnorm s=3, β=4 | 1508 | 8 | 1.47 | 0.000 |
| ctxnorm s=3, β=2 | 1505 | 9 | 1.50 | 0.000 |
| ctxnorm s=1.5, β=6 | 1500 | 8 | 0.75 | 0.000 |

**결과: $\|c\|$는 잘 묶였으나(1.47) collapse 유지(8~9/128).** 가설 부분 기각 — **magnitude가
유일한 원인이 아니다.** `dist→pat=0.000`이 그대로 → 여전히 완전 포화. 패턴노름3+c노름3+β≥2면
logit 스프레드 $\sim\!36$이라 $\|c\|$를 묶어도 포화한다. → 근본 원인은 **포화 자체**(§5).

---

## 5. 기전 분석: softmax 포화와 gradient 소실

### 5.1 effective temperature
softmax가 one-hot이 되는지는 **logit들의 스프레드**가 결정한다. Hopfield logit:
$$z_k = \beta\,\langle X_k,\, y+c\rangle, \qquad \text{스프레드} \sim \beta\cdot\|X\|\cdot\|y+c\|.$$
$\|y\|\le\|X\|$(conv hull로 유계)이지만 $\|c\|$는 인코더가 자유 조절. 따라서
$$\boxed{\text{effective temperature} \sim \beta\cdot\|X\|\cdot\|c\|}$$
이 크면 포화(one-hot), 작으면 soft. **β·패턴노름·context노름 셋 다** 온도를 키운다.

### 5.2 포화 → gradient 소실 → trivial 해 (vicious cycle)
1. one-hot 자체는 무해(입력마다 다른 패턴이면 OK). 문제는 **모든 입력→같은 패턴**.
2. recon 최소화의 가장 쉬운 국소해: $y^\star$를 상수로 두고 디코더가 **데이터 평균 이미지** 출력.
   → recon ≈ 데이터 분산 ≈ **1500**(관측 정체값). trivial local optimum.
3. 더 잘하려면 입력별 다른 패턴 선택이 필요한데, 포화 softmax에서
   $$\frac{\partial\,\mathrm{softmax}(z)}{\partial z}\Big|_{\text{one-hot}} \approx 0 \;\Rightarrow\; \frac{\partial y^\star}{\partial c}\approx 0.$$
4. → recon gradient가 "다른 패턴을 골라라"를 $c$로 전달 못 함 → **collapse가 얼어붙음.**
   `dist→pat=0.000`이 완전 포화의 직접 증거.

### 5.3 β–‖c‖ 상호대체성 (왜 β 어닐링이 실패했나)
effective temperature는 $\beta$와 $\|c\|$의 **곱**이다. β를 낮춰도 인코더가 $\|c\|$를 키워 온도를
유지할 수 있다(§3에서 β=0.5일 때 $c\_std=542$로 폭발). 따라서 **$\|c\|$가 자유로우면 β 단독
통제는 무의미**하다. 이것이 β 어닐링 기각의 정확한 원인이다.

### 5.4 "exact gradient = 0"의 역설 (vs STE)
IFT는 고정점의 gradient를 *정확히* 준다. 그러나 포화에서는 **참 Jacobian이 실제로 ≈0**이므로
정확한 gradient도 0이다. 반면 VQ-VAE의 **STE는 편향됐지만 nonzero**($\partial z_q/\partial z_e=I$로
가정) — 그래서 이산 병목에도 학습된다. 결론:
> **"완전 미분가능(exact gradient)"이 "유용한 gradient"를 뜻하지 않는다.** 함수가 이산 극한에서
> 진짜로 평평하면 정확한 gradient는 0이고, 이는 학습에서 **편향-nonzero STE보다 불리**할 수 있다.

이는 본 접근(IFT로 STE 대체)의 가장 본질적 난점이며, **비포화 영역에서 학습**해야 하는
이유를 규정한다.

### 5.5 KL 부재와의 관계
collapse는 "VAE의 KL이 없어 잠재 magnitude 규제가 사라진 현상"으로도 볼 수 있다. KL의 $-\mu^2$
항은 $\|z\|$를 누른다. 다만 KL은 $\|z\|\to0$(prior 중심)으로 당기는데, 본 readout에서 $c\to0$은
**무정보 collapse**(패턴 선택 불가, centroid 단일 attractor)를 부른다. 즉 잠재 크기는 *너무 크면
포화, 너무 작으면 무정보* → 필요한 것은 **0이 아니라 특정 nonzero 스케일 고정**(=context 정규화).
KL을 그대로 넣는 것은 적절치 않고, target 스케일을 향한 소프트 규제가 deterministic 대안이다.

---

## 6. 해결: soft 영역 학습 + 어닐링

### 6.1 soft(저온) 영역 실험
§5.1에 따라 effective temperature를 낮추려면 **패턴노름·context노름·β를 함께 작게** 둔다.

**실험**(`_smoke_ctxnorm.py`, soft configs, 각 300 step):

| config | recon (100→300 step) | 활성(/128) | dist→pat | c_batch_std |
|---|---|---|---|---|
| **pat=1, s=1, β=1** | 1164 → 757 → **333** | 27→33→**36** | 0.47 | 0.26 |
| pat=1, s=1, β=0.5 | 1357 → … → 685 | 25 | 0.43 | 0.30 |
| pat=0.5, s=0.5, β=2 | 1328 → … → 365 | 29 | 0.24 | 0.13 |

**결과: collapse 해소.** 최고(pat=1,s=1,β=1)에서 **recon 1500→333**(아직 하강 중), **활성
8→36/128**, 비포화(dist→pat 0.47)로 gradient 정상 흐름. 인코더가 실제로 학습하고 코드를 다양하게
쓴다.

**단서**: dist→pat=0.47 ≠ 0 → $y^\star$가 패턴에 안 붙은 **blend**. 즉 *작동하는 오토인코더이되
코드가 아직 이산이 아니다.*

### 6.2 원리: train soft, anneal to discrete
검증된 그림:
- **포화 영역**(또렷한 이산 스케일: 큰 패턴·β) = gradient 0 = **학습 불가**(collapse).
- **비포화 soft 영역**(작은 패턴·c·β) = gradient 흐름 = **학습 가능**(recon↓, 다양성↑) — 단 코드 연속.
- → **저온에서 인코더가 잠재공간을 조직화 → effective temperature를 점진 상승 → $y^\star$를 패턴으로
  sharpen.** Gumbel-softmax 온도 어닐링과 동형이나, **매 온도에서 gradient가 정확(IFT)**이라는 차이.

핵심 통제량은 단일 β가 아니라 **effective temperature $\beta\cdot\|X\|\cdot\|c\|$**이다(패턴 정규화 +
context 정규화 + β 스케줄을 함께).

### 6.3 어닐링 실증 (D3)
**설정**: `unit_patterns=True`(패턴 단위노름) + `context_norm`($\|c\|{=}1$) → effective temp $={}$β·1·1
$={}$β로 **단일화**. soft warmup(앞 30%, β=1) 후 β를 선형으로 $\beta_\text{end}$까지 상승(`_smoke_d3.py`,
각 1500 step).

| 스케줄 | 활성(/128) | dist→pat | recon |
|---|---|---|---|
| β 1→4 | 80 | 0.250 | 194 |
| **β 1→8** | **90** | 0.099 | **182** |
| β 1→16 | 82 | **0.049** | 193 |

**궤적(1→8)**: β 1.0→8.0 상승에 따라 dist→pat **0.61→0.10**(이산화), 활성 **65→90**(↑), recon
**403→182**(↓) — *세 지표 동시 개선*.

**핵심 결과**:
1. **재collapse 없음** — β를 16까지 올려도 활성 80~90 유지. soft warmup에서 인코더가 잠재공간을
   조직화한 뒤라 고온에서도 무너지지 않음.
2. **이산성 달성** — dist→pat 0.05~0.10 ($y^\star$가 패턴에 근접 = 사실상 이산).
3. **다양성 증가** — collapse(활성 8) 대비 90/128. 어닐링이 코드 사용을 *늘림*.
4. **sweet spot $\beta_\text{end}{=}8$**: 활성 90(최고)·dist→pat 0.10·recon 182(최저).

→ §8의 "남은 과제(soft→discrete 어닐링)"는 **해결**됐다. 잔여는 충분 학습 시 recon이 VQ-VAE
수준(5.7)에 얼마나 근접하는가다(§8 갱신).

---

## 7. 정량 요약 (한눈에)

| 영역/처치 | recon | 활성(/128) | dist→pat | 학습? |
|---|---|---|---|---|
| 포화·정규화X (β=2) | ~1500 | 8 | 0.000 | ✗ collapse |
| 포화·β어닐링 | ~1498 | 8 | 0.000 | ✗ ($c\_std$ 486 폭발) |
| 포화·ctxnorm (s=3,β≥2) | ~1505 | 8~9 | 0.000 | ✗ (여전히 포화) |
| **비포화·soft (pat1,s1,β1)** | **333** | **36** | 0.47 | ✓ 학습(코드 연속) |
| **어닐링 β1→8 (unit_pat+ctxnorm)** | **182** | **90** | 0.099 | ✓✓ 학습 + **이산화** |

---

## 8. 남은 과제 & 설계 함의

**해결됨 (D3)**: **재collapse 없는 soft→discrete 어닐링** — §6.3에서 실증. soft warmup 후 β를
단일 온도 노브로 1→8 상승시키면 dist→pat→0.10, 활성 90/128, recon 182로 *동시* 개선. collapse
문제의 핵심 해법 확정.

**남은 과제**: (a) **충분 학습 시 recon 격차** — 현재 182(1500 step)는 VQ-VAE 베이스라인
5.7(108 epoch)과 큼. 본 학습(D5) 수렴 후 이산 코드로 공정 비교 필요. recon이 평탄해지면 이산
병목·코드수(K) 영향 분석. (b) **어닐링 스케줄 민감도**(β_end·warmup·속도)의 체계적 탐색.
(c) disentanglement 평가(MIG/DCI/FactorVAE)로 표현 품질 검증.

**설계 함의(확정)**:
1. **effective temperature($\beta\|X\|\|c\|$)가 1차 통제량.** β 단독 불충분.
2. **패턴 정규화 + context 정규화**로 $\|X\|,\|c\|$를 고정, 온도를 β(또는 scale) 스케줄로만 변동.
3. **저온 시작 필수**: C1의 "또렷한 이산" 스케일은 학습엔 부적합(포화). 학습은 soft에서 시작.
4. 잠재 크기는 **nonzero 고정 스케일**(KL식 →0 아님)이 readout 특성에 맞음. 소프트 target 규제도 후보.
5. **이론적 리스크 기록**: 이산 극한에서 exact gradient→0. 어닐링이 충분히 느려야 인코더가
   조직화를 유지. 어닐링이 본질적 한계에 부딪히면, (a) STE-혼합(저온 IFT+고온 STE), (b) 정규화로
   비포화에서 *유사 이산성* 확보, (c) 보조 commitment/usage-entropy 손실 등을 고려.

**비교 기준(향후)**: VQ-VAE 베이스라인 recon=5.7(수렴). 현재 soft recon=333은 300 step·연속코드라
직접 비교 불가 — 어닐링·충분 학습 후 이산 코드로 비교해야 한다.

---

## 부록 A. 실험 설정·코드 포인터
- 모델: `src/model_attractor_vae.py` (`AttractorVAE`, `attractor_loss`, `_norm_context`).
- 부품: `src/fixed_point/{solvers,implicit,multistable}.py` (검증 `_verify_b/_verify_c1/_c2/_c3.py`).
- collapse 실험: `src/_smoke_anneal.py`(β 어닐링), `src/_smoke_ctxnorm.py`(context 정규화·soft 영역).
- 공통: 3D Shapes 256배치, Adam 1e-3, fpi/anderson solver, one-step IFT backward, adjoint_clip=10.
- 진단량: `dist→pat`(=평균 $\|y^\star-\text{nearest pattern}\|$, 0이면 완전 이산/포화),
  `활성 attractor`(=배치에서 factor별 unique argmin 패턴 수 합, /128), `c_std`/`c_batch_std`.

## 부록 B. 핵심 명제 (논문용 한 줄들)
- P1. attractor-AE의 collapse는 softmax 포화에서 $\partial y^\star/\partial c\to0$이 되어 발생한다.
- P2. effective temperature는 $\beta\cdot\|X\|\cdot\|c\|$이며, 인코더는 $\|c\|$로 β 통제를 우회한다.
- P3. exact IFT gradient는 이산 극한에서 0 → 편향-nonzero STE보다 학습에 불리할 수 있다.
- P4. 따라서 학습은 비포화(soft) 영역에서 시작하고 이산은 어닐링으로 달성해야 한다.
- P5. 잠재 magnitude는 0이 아닌 nonzero 고정 스케일로 규제해야 한다(KL의 →0과 구별).
