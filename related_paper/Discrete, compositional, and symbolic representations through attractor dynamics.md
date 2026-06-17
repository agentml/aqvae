> # **논문**: Discrete, compositional, and symbolic representations through attractor dynamics
> **저자**: Andrew J. Nam, Eric Elmoznino, Nikolay Malkin, James L. McClelland, Guillaume Lajoie, Yoshua Bengio
> **기관**: Princeton · Mila/Université de Montréal · University of Edinburgh · Stanford
> **출처**: arXiv:2310.01807v2 [cs.AI], 2024-09-26 (Preprint)

> [!info] 공부 노트
> AIR와 스텝바이스텝으로 공부하며 작성 중. 섹션 단위로 아래로 자랍니다.
> 관련: [[2605 - Solve the Loop - Attractor Models for Language and Reasoning]] — 둘 다 **attractor dynamics**를 다루지만 목적이 다릅니다(저쪽은 LM/reasoning의 효율적 inference, 이쪽은 인지모델로서 symbol의 emergence).

---

## 0. 한눈에 (TL;DR)
**문제의식**: 인간 인지의 **symbol**(언어·수학)은 전적으로 neural connection 위에서 돌아갑니다. 그런데 기존 neural 모델이 symbol을 다루는 방식 — VQ-VAE의 vector quantization, Gumbel-Softmax — 은 **discretization을 "가정"하지 "유도"하지 않습니다.** 즉 discrete slot을 아키텍처에 박아넣을 뿐, *분산된 neural activity로부터 symbol이 어떻게 emerge하는지*를 설명하지 못합니다.

**핵심 발상**: continuous latent space 위에 **attractor dynamics**를 학습시켜, 공간을 discrete basin들로 자연스럽게 분할합니다. 각 attractor state가 하나의 **symbolic sequence**(토큰열)에 대응됩니다. discretization이 아키텍처 제약이 아니라 **stochastic dynamics가 평형으로 수렴하는 과정**에서 emergent하게 나옵니다 — 사전 정의된 primitive 없이 **unsupervised learning**으로.

**무엇을 주장하나**: 이 모델은 **probabilistic Language of Thought (PLoT)**를 neural dynamics로 구현합니다.
1. attractor state가 sentence embedding 주위에 형성되어 discrete symbolic representation으로 매핑됨.
2. 이 attractor들이 **grounded semantics·compositionality** 같은 symbolic 성질을 (학습으로) 획득.
3. 입력 $x$와 symbolic encoding 사이의 mutual information을 반영하는 **다양한 분포(Bayesian posterior)**를 sampling.
- VQ-VAE·emergent communication 모델과 비등한 성능을, *명시적 lookup table 없이* attractor dynamics만으로 달성.

**왜 중요한가 (인지/신경 plausibility)**: 뇌는 attractor dynamics로 discrete state를 유지한다는 신경과학적 증거(원숭이 MT 영역의 integrator neuron, 쥐 ALM의 optogenetic 개입으로 확인된 attractor basin·decision boundary)가 있습니다. 이 모델은 그 메커니즘과 일관된 계산 모델입니다.

---

## 1. Introduction — symbol은 neural activity에서 어떻게 emerge하는가
### 1.1 오래된 논쟁: symbolic vs sub-symbolic
- **Symbolic 진영** (Fodor & Pylyshyn 1988, Newell, Marcus): 인지의 핵심은 언어처럼 **systematic·compositional** 구조를 가진 *규칙과 symbol의 집합*(= Language of Thought, LoT). Newell의 General Problem Solver처럼 graph search로 사고를 시뮬레이션.
- **Sub-symbolic/connectionist 진영**: continuous·distributed 표현. 실제 인간 추론이 순수 symbolic 계산에서 벗어나는 많은 방식을 포착하지만, "symbol이 어디서 오나"가 모호.
- LLM의 등장은 "복잡한 neural connection 망이 symbol manipulation을 *할 수 있다*"는 강한 증거. 하지만 정교한 LLM도 수학·reasoning 같은 symbolic 문제에선 여전히 고전.

### 1.2 PLoT — 확률적 사고의 언어
**Probabilistic Language of Thought (PLoT)** (Goodman et al. 2015): LoT를 Bayesian statistics로 확장. 핵심은 **단일 예측이 아니라 표현·가설에 대한 "분포"를 포착**한다는 점 — 인간 인지의 graded·noisy·multimodal 특성에 적합. rule induction, structural form discovery, pragmatic language, causal reasoning 등 다양한 인지과정 모델링에 사용됨.

LoT/PLoT의 장점은 **compositional 구조**(primitive를 조합해 복잡한 사고 생성, novel 입력에 robust generalization). 단점은 **symbol과 process를 미리 명시**해야 한다는 것 — symbolic하게 표현 가능한 domain에 갇히고, 그 표현을 떠받치는 *sub-symbolic 메커니즘을 추상화해 버림*.

### 1.3 기존 neurosymbolic discretization의 4가지 한계 (이 논문이 공격하는 지점)
VQ-VAE(vector quantization), Gumbel-Softmax 같은 "discrete bottleneck" 기법들의 문제:
1. **discreteness를 아키텍처에 박아넣음** — globally accessible한 discrete slot 배열(codebook)을 가정. 인간 마음이 명시적 slot에 mental state를 담고 있을 것 같진 않음 → 인지 모델로 부적절.
2. **discretization이 "어떻게" 일어나는지 설명 없음** — 그냥 quantized 값을 상속받아 *일어난다*. 순수 symbolic 모델과 같은 한계: computational level에서 목표(discretization)만 정의하고 algorithmic level의 메커니즘은 못 잡음.
3. **원자극의 흔적(trace)을 못 담음** — 사람은 discrete-like 행동을 보이면서도 원래 자극의 흔적이 시간에 따라 *서서히 감쇠*하는데(McMurray et al. 2002), discrete state로 딱 매핑하는 모델은 이를 못 담음.
4. **원리적인 diverse 분포를 학습할 장치가 없음** — PLoT·인간 데이터의 특징인 "다양한 discrete state의 분포"를 향한 메커니즘 부재 → uncertainty를 잘 못 잡는 임의적 해로 귀결.

### 1.4 이 논문의 해법: attractor dynamics로 discretization을 emergent하게
**neural stochastic dynamical systems model**을 제안. 핵심:
- continuous latent space를 **discrete basin**으로 분할, **attractor state = symbolic sequence**의 embedding.
- discretization은 아키텍처 제약(slot)이 아니라, **energy landscape를 학습하고 거기서 sampling**하는 *완전히 sub-symbolic·continuous한 과정*에서 나옴.
- 입력 $x$에 조건화된 **token sequence에 대한 복잡한 Bayesian posterior**를 학습 (VAE·normalizing flow·diffusion처럼 분포를 sampling하되, *continuous state와 discrete symbol을 동시에* 다루는 점이 고유).
- symbolic 구조·latent semantics·attractor dynamics 그 무엇도 사전 지정 안 함 → 전부 **unsupervised learning**으로 emerge. "discrete token을 출력하라"는 압력 아래 이 성질들이 자생.

> **신경과학적 영감**: 뇌가 attractor dynamics로 discrete state를 달성·유지한다는 증거 — 원숭이 MT integrator neuron이 2AFC 과제에서 두 구별되는 state로 수렴(drift diffusion model과 유사), 쥐 ALM의 optogenetic 개입이 attractor basin·decision boundary의 인과적 증거 제공(작은 perturbation엔 robust, 큰 perturbation은 trajectory를 다른 attractor로 이동).

> **우리 연구 연결 (NEO/NES/WTM · LoT 관점, 초벌)**: 이건 연구자님의 *"Language of Thought의 목적은 고차원 출력이 아니다"*, *sensor-in/control-out* 단상과 직접 맞닿습니다. 고차원 input $x$를 저차원 discrete code로 압축하는 과정을 *attractor로의 수렴*으로 본 것이고, 그 code가 compositional하게 emerge합니다. NES(neurosymbolic) 쪽에서 "symbol을 박지 않고 dynamics에서 끌어내는" 설계 원형으로 참고 가치가 큽니다. (Model 섹션에서 GFlowNet-EM 학습으로 구체화)

---

## 2. Model — 생성 과정: 입력에서 attractor로, attractor에서 token으로
### 2.1 한 문장 목표
입력 $x$(예: 이미지)에 대해, latent space $\mathcal{Z}$ 위의 **marginal 분포 $P_\theta(z_T \mid x)$** 를 학습합니다. 이 분포는 token sequence $s$의 embedding vector $\hat{z}_s$ 주위에 **cluster(basin)** 를 이루며, 그 밀도는 *"$s$가 $x$를 얼마나 잘 묘사하는가"에 비례*합니다.

쉽게 말하면: "빨간 사각형" 이미지를 주면, latent 공간에서 `top right large red square`에 해당하는 attractor 근처가 가장 자주 찍히고, `heart`에 해당하는 곳은 거의 안 찍히도록 분포를 만든다는 겁니다(Fig.1).

> [!question]+ 비판적 질문 — "full 문장마다 attractor가 하나면 mode가 지수적으로 많아야 하고, 그럼 compositionality가 없는 lookup 아닌가?"
>
> 연구자님 질문: feature가 $K$개, 각 $V$값이면 가능한 description은 $V^K$개 → attractor도 $V^K$개 필요 → 사실상 거대한 lookup table 아니냐.
>
> **핵심 반론: "mode가 지수적으로 많다"는 compositionality의 *부재*가 아니라 productivity(생산성) — compositionality의 정의 그 자체다.** compositionality는 "mode가 적다"가 아니라 *"적은 수의 primitive와 규칙으로 무한/지수적으로 많은 표현을 생성한다(finite means, infinite ends)"*를 뜻한다. 그러니 $V^K$개 basin이 있는 것은 문제가 아니고, 질문은 **"그 basin들이 독립적으로 저장되었나, 아니면 공유 primitive로 *생성*되었나"**다.
>
> **이 모델에서 compositionality가 사는 곳** (= 왜 lookup이 아닌가):
> 1. attractor embedding $\hat{z}_s = e_\phi(s)$는 **자유 벡터가 아니다.** sentence encoder $e_\phi$가 *토큰열 $s$로부터* 만든다. 즉 `red square`의 embedding은 토큰 `red`·`square`의 처리에서 조립된다. 저장되는 것은 $V^K$개 벡터가 아니라 **소수의 토큰 표현 + 작은 network**이고, $V^K$개 attractor는 거기서 *생성*된다.
> 2. discretizer $P_\theta(s_i\mid s_{:i}, z)$도 **shared weight**로 토큰을 하나씩 붙인다 → systematic한 production rule.
> 3. §2.2(다음 섹션)에서 명시: attractor dynamics *단독*으로는 discreteness만 줄 뿐 compositional 구조는 안 나온다. **symbolic form(토큰) + shared weight + prior $P(s)$가 compositionality의 inductive bias를 공급**한다.
>
> **결정적 경험 증거(§4에서 상세):**
> - **OOD 일반화**: held-out feature 조합(예: magenta oval, 특정 위치)에서도 거의 완전한 일반화. *순수 lookup은 본 적 없는 조합에 절대 일반화 못 한다* → lookup 가설의 직접 반증.
> - **Feature-wise composition (FWC)**: description을 *토큰 하나씩, feature 하나씩* 조립해도 모델이 직접 sampling한 문장만큼 feature를 잘 예측 → 생성 규칙이 compositional.
> - **RSA/TopSim 0.525** (VQ-VAE 0.314, EC 0.156보다 높음): 표현 기하가 feature 기하를 반영.
> - HBV 토큰 설계를 *일부러* 12토큰·6쌍으로 묶어, branch마다 고유 토큰(`A,B,C,D`)을 주는 비-compositional 해를 **막고** 조합(`AA,AB,BA,BB`)을 강제.
>
> **그래도 남는 진짜 tension (질문의 정당한 핵심):**
> - *dynamics/sampling 부담*: SDE가 실제로 $V^K$개 basin을 새기고, 그 위 posterior를 올바르게 방문·추정해야 한다. 논문도 §4.3에서 "combinatorially 큰 sentence space의 posterior 적합도 평가는 어렵다"며 **작은 HBV에서만** goodness-of-fit을 검증한다. 즉 *지수적 mode를 제대로 다 잡았는지의 검증은 제한적*.
> - compositionality가 **아키텍처로 보장**되는 게 아니라 **inductive bias로 유도(emergent)**된다. 모델이 이탈 가능 — 실제로 synonym basin이 합쳐지거나(§4.1), OOD에서 FWC가 모델보다 더 잘 맞는 등 규칙에서 벗어나는 흔적이 관측된다.
>
> **한 줄 결론**: 지수적 mode는 버그가 아니라 feature(productivity)다. 진짜 쟁점은 "그 mode들이 공유 토큰 표현으로 생성되는가"이고, OOD 일반화·FWC가 *그렇다*고 강하게 시사한다. 다만 "지수적 posterior를 정확히 학습·검증"하는 부분은 이 논문의 명시적 한계다.

### 2.2 모듈 구성 (Table 1)
파라미터를 **두 그룹**으로 나눕니다 — 학습 단계가 다릅니다(§2.4의 EM):
- **$\phi$ (M-step, encoder/decoder 류)**: 무엇이 무엇을 의미하는지(semantics)를 정함.
- **$\theta$ (E-step, dynamics/sampling 류)**: 어떻게 sampling할지(policy)를 정함.

| 모듈 | 역할 | 표기 |
|------|------|------|
| Input Encoder ($\phi$) | 입력 $x$를 정보 풍부한 latent 상태 $z_0$로 인코딩 | $P_\phi(z_0 \mid x)$ |
| Input Decoder ($\phi$) | latent $z$를 입력 $x$로 복원 (MI 확보용) | $P_\phi(x \mid z)$ |
| **Forward Dynamics** ($\theta$) | $z_t \to z_{t+1}$ stochastic 전이 (←핵심) | $P_\theta(z_{t+1} \mid z_t)$ |
| Backward Dynamics ($\theta$) | 역방향 전이 (학습 보조) | $P_\theta(z_t \mid z_{t+1}, t{+}1, x)$ |
| Sentence Encoder ($\phi$) | token sequence $s$ → embedding $\hat{z}_s$ | $\hat{z}_s \leftarrow e_\phi(s)$ |
| **Sentence Decoder = Discretizer** ($\theta$) | latent $z$ → token $s_i$ 자기회귀 sampling | $P_\theta(s_i \mid s_{:i}, z)$ |
| Flow Correction ($\theta$) | 상태 $z_t$의 'flow'(미래 기대보상) 추정 | $g \leftarrow g_\theta(z_t, t, x)$ |

> 굵게 표시한 **Forward Dynamics + Discretizer 둘만 있으면 discretization은 작동**합니다. 나머지는 학습·분석용 보조 모듈입니다.

### 2.3 생성 trajectory: $z_0 \rightsquigarrow z_T \to s$
실제 추론(sampling) 과정은 세 단계입니다:

1. **초기 인코딩**: 입력 $x$와 mutual information이 높은 시작점 $z_0 \sim P_\phi(z_0 \mid x)$를 뽑습니다.
2. **dynamics로 굴리기**: forward dynamics로 다음 점을 반복 sampling
$$z_{t+1} \sim P_\theta(z_{t+1} \mid z_t), \qquad z_0 \rightsquigarrow z_T = z_0, z_1, \dots, z_T$$
수렴 조건까지 굴리면, 종착점 $z_T$는 어떤 $s$의 embedding $\hat{z}_s$ 근처의 **attractor point**에 도달합니다.
3. **discretize**: 빈 시퀀스에서 시작해 sentence decoder로 토큰을 하나씩 자기회귀로 붙입니다
$$s_i \sim P_\theta(s_i \mid s_{:i}, z_T), \quad \text{EOS까지} \;\Rightarrow\; s = s_1, \dots, s_k, s_{eos}$$
핵심 직관: **stochasticity가 다양한 trajectory를 낳고**, **attractor dynamics가 그 trajectory들을 sentence embedding $\hat{z}_s$ 쪽으로 끌어당깁니다.** 그래서 종착점 분포 $P_\theta(z_T \mid x)$가 $\hat{z}_s$ 주위의 discrete basin들로 갈라집니다. (영어 단어는 해석을 위한 표기일 뿐, 모델은 *어떤 formal language도 본 적 없이* 임의 토큰에 emergent semantics를 붙입니다.)

### 2.4 왜 "neural stochastic 미분방정식(SDE)"인가
forward·backward·input dynamics는 모두 **conditional Gaussian**이고, 그 평균·분산을 neural network가 출력합니다. 그러면:
- 이 policy는 **discretized neural SDE** (Tzen & Raginsky 2019)와 동치이고,
- sampling 과정은 SDE를 수치적으로 푸는 **Euler–Maruyama method**와 동치입니다.

쉽게 말하면: 각 step은 "현재 위치에서 신경망이 가리키는 방향(평균)으로 조금 이동 + 약간의 noise(분산)"입니다 — diffusion/Langevin 같은 그림. 또 *gradual change*를 위해 전이 분포의 평균에 **bound(smoothness prior)** 를 걸어, 한 step에 너무 멀리 못 뛰게 합니다. 이 덕에 trajectory가 매끄럽게 attractor로 흘러듭니다.

> # **요약**: Model = (입력→$z_0$) + (Gaussian SDE로 $z_0\rightsquigarrow z_T$, attractor로 수렴) + (discretizer로 $z_T\to$ 토큰열). 다음 스텝(§2 reward)에서 *"무슨 분포로 수렴해야 좋은가"*를 정의하는 **reward function**(Eq.1)을 봅니다.

---

## 3. Reward function (Eq.1) — "어떤 분포로 수렴해야 좋은가"
§2의 dynamics는 *어떻게* sampling하는지를 정합니다. 그런데 *무엇을 향해* sampling해야 좋은지(=목표 밀도)는 따로 정의해야 합니다. 그게 **reward** $R_\phi$입니다. 모델은 종착점·토큰열의 결합분포가 이 reward에 **비례**하도록 학습됩니다:
$$P_\theta(z_T, s \mid x) \;\propto\; R_\phi(z_T, s; x) \;=\; \underbrace{\exp\!\big(-D_{\mathrm{KL}}(P_\phi(z_0\mid x)\,\Vert\, P_\phi(z_0\mid \hat{z}_s))\big)}_{\text{(1) fidelity}} \cdot \underbrace{P_\phi(s\mid z_T)}_{\text{(2) synonym}} \cdot \underbrace{P(s)}_{\text{(3) prior}} \tag{1}$$
"비례한다(∝)"가 핵심입니다 — 어떤 $s$가 reward가 2배 높으면, 그 attractor가 2배 자주 sampling되어야 합니다. (단일 best $s$만 고르는 게 아니라 **분포**를 맞춤 = PLoT의 Bayesian posterior.) 세 항을 하나씩 봅니다.

### (1) Fidelity 항 — "그 symbol이 입력을 얼마나 잘 설명하나"
$$\exp\!\big(-D_{\mathrm{KL}}(P_\phi(z_0\mid x)\,\Vert\, P_\phi(z_0\mid \hat{z}_s))\big)$$
- $P_\phi(z_0\mid x)$: *실제 입력 $x$*가 만드는 시작점 분포.
- $P_\phi(z_0\mid \hat{z}_s)$: *symbol $s$의 embedding $\hat{z}_s$만 보고* 복원한 "이 symbol이 가리키는 대상"의 분포.
- 둘 사이 KL이 작다 = "$s$만 알아도 $x$를 거의 그대로 떠올릴 수 있다" = **discretization으로 잃은 정보가 적다.** KL의 음수를 exp → fidelity가 높을수록 reward↑.

쉽게 말하면: 사과 이미지를 보고 `red square`라고 압축했을 때, 나중에 `red square`만 듣고 떠올린 그림이 원본과 비슷할수록 좋은 symbol. (정보 손실 = 나쁜 reward.)

### (2) Synonym 항 — 같은 뜻의 여러 문장 보정
$$P_\phi(s\mid z_T)$$
같은 embedding에서 나올 수 있는 *서로 다른 문장들*의 likelihood. $x$와 무관하게, **synonym이 embedding을 공유하는 것**을 보정합니다.
- 직관: 사과를 보고 'red'를 떠올리든 'scarlet'을 떠올리든 사고 내용은 거의 안 바뀜. 둘은 거의 같은 latent(redness)에 매핑됨.
- 문제: 한 attractor 근처에 synonym embedding이 10개 몰려 있으면, 단순히 밀도만 보면 그 개념이 *10배 과대평가*됨. 이 항은 "$z_T$ 하나에 가까운 $\hat{z}_s$가 1개든 10개든 sampling이 달라지지 않도록" reward를 scale해, **개념(meaning) 기준으로 공정하게** 만듭니다.

### (3) Prior 항 $P(s)$ — 문장 자체에 대한 선호
$\mathcal{S}$ 위의 prior. 가장 단순하게는 **짧은 시퀀스 선호(Gricean)** — "필요 이상으로 길게 말하지 마라". symbolic form이라서 *길이·복잡도 비용*을 걸기 쉽습니다(continuous vector로는 어려움). 실험에선 uniform prior도 사용.

### Energy-based 해석
각주: $R = e^{-\mathcal{E}}$, 즉 reward는 energy-based model의 **energy function $\mathcal{E}$**와 $R=e^{-\mathcal{E}}$ 관계입니다.
→ "$R$에 비례하게 sampling" = "energy $\mathcal{E}$가 낮은 곳(=좋은 symbol의 attractor)에 더 자주 머묾". §1.4에서 말한 *"energy landscape를 학습하고 거기서 sampling"*의 정확한 형태가 이것입니다. attractor basin = energy의 낮은 골짜기.

> **우리 연구 연결**: reward를 (fidelity) × (synonym 보정) × (description prior)로 **분해**한 게 설계적으로 깔끔합니다. NES/WTM에서 latent code의 품질을 "복원 충실도 × 표현 간결성"으로 정의하고 *분포로* 맞추고 싶을 때 그대로 차용 가능. 특히 (1)의 *"symbol만으로 입력을 재구성했을 때의 KL"*은 latent symbol의 grounding을 정량화하는 일반적 도구로 쓸 만합니다.

---

## 4. Training — GFlowNet-EM (§2.1)
§3에서 "$P_\theta(z_T, s\mid x) \propto R$"라는 *목표*를 세웠습니다. 그런데 "분포에 비례하게 sampling"을 어떻게 *학습*할까요? 답은 **continuous Generative Flow Network (GFlowNet/GFN)**을 **EM loop**로 학습하는 것입니다.

> [!note]+ 보충 — GFlowNet이란? (왜 여기 쓰나)
>
> **한 줄**: GFlowNet은 *"terminal state의 분포가 주어진 unnormalized reward $R$에 비례하도록"* trajectory를 만드는 sampler를 학습하는 방법입니다.
>
> - 일반 RL/최적화는 보통 reward를 **최대화**(단일 best mode로 수렴)합니다. GFlowNet은 대신 reward에 **비례하게 sampling** — reward가 높은 mode를 *여러 개* 비율대로 방문합니다. 그래서 **multimodal·diverse** 분포(=PLoT가 원하는 Bayesian posterior)에 딱 맞습니다.
> - 작동 원리: 상태들을 노드로 보는 DAG에서, 각 상태로 들어오고 나가는 **"flow"가 보존**되도록(유량 보존) policy를 학습합니다. 흐름이 reward에 맞게 분배되면 terminal 분포가 $\propto R$이 됩니다. 강물이 여러 골짜기로 reward 비율만큼 갈라져 흐르는 그림.
> - 이 논문의 trajectory $z_0\rightsquigarrow z_T$가 바로 그 "flow를 따라 흐르는 경로"이고, 종착점 $z_T$가 terminal state입니다. **continuous** GFN(Lahlou et al. 2023)이라 상태가 연속(SDE)인 점만 표준 GFN과 다릅니다.

### 4.1 GFN-EM: 왜 EM인가
문제 구조: 입력 $x$는 관측, 그런데 "$x$를 묘사하는 symbol $s$"는 **latent 변수**(정답 라벨 없음)입니다. latent 변수를 추정하며 생성 모델을 학습하는 고전 도구가 **EM(Expectation–Maximization)**입니다. 두 step을 번갈아 돕니다(Hu et al. 2023, GFN-EM):

- **E-step — "어떤 문장으로 묘사할까?"** (labeling)
  현재의 *semantics를 고정*한 채, 각 관측에 어떤 문장 $s$를 붙일지(sampling policy)를 학습. → 파라미터 $\theta$ 최적화.
- **M-step — "그 문장들이 무슨 뜻이어야 할까?"** (semantics)
  E-step이 붙인 문장들을 *고정*한 채, 그 문장이 가리키는 의미(embedding·encoder/decoder)를 학습. → 파라미터 $\phi$ 최적화.

쉽게 말하면: E-step은 "이 사과 그림엔 `red`라고 하자"라고 라벨을 달고, M-step은 "그럼 `red`는 이런 latent 영역을 뜻하는 걸로 하자"라고 의미를 조정합니다. 둘이 번갈아 수렴하면 **emergent language**가 자리 잡습니다.

### 4.2 E-step 디테일 — forward/backward 궤적 매칭
E-step은 sampling policy $\theta$ (= forward dynamics $P_\theta(z_{t+1}\mid z_t)$ + discretizer $P_\theta(s_i\mid s_{:i},z)$)를 학습합니다. GFlowNet의 학습 신호는 **"앞으로 만든 궤적의 확률 = 뒤로 만든 궤적의 확률"**이 되도록 맞추는 것입니다. 두 방향:

**Forward**(입력 embedding $z_0$에서 출발해 토큰까지):
$$P(z_0,\dots,z_T,s\mid x) = P(z_0\mid x)\cdot P(z_1\mid z_0)\cdots P(z_T\mid z_{T-1})\cdot P(s\mid z_T) \tag{2}$$
**Backward**(토큰 $s$에서 출발해 거꾸로 궤적을 복원):
$$P(z_0,\dots,z_T,s\mid x) = P(s\mid x)\cdot P(z_T\mid \hat{z}_s, T{+}1, x)\cdot P(z_{T-1}\mid z_T, T, x)\cdots P(z_0\mid z_1, 1, x) \tag{3}$$
직관: 같은 (궤적, 토큰열) 결합확률을 **앞에서 굴린 값**과 **뒤에서 굴린 값**이 일치하도록 강제합니다. 이게 맞으면 종착점 분포가 reward에 비례하게 됩니다(GFlowNet의 핵심 등식). 그래서 **backward dynamics**가 학습 보조로 필요합니다(추론 땐 안 씀).

- 구체적으로는 **detailed balance with forward-looking flow parametrization (FL-DB)** objective를 씁니다. 여기서 **flow correction $g_\theta(z_t, t, x)$**가 상태 $z_t$의 'flow'(= 그 상태에서 기대되는 미래 reward)를 추정합니다 — GFN 특유의 양.
- discretizer도 (토큰을 꼭 왼→오 순서로 붙이지 않도록) backward token policy $P_\theta(s_i\mid s_{:i+1}, z, x)$와 함께 학습. reward가 $z$를 직접 안 쓰므로 "이 $s$가 어느 $z$에서 나왔나"를 복원하는 $P_\theta(z\mid s, x)$도 필요한데, 이는 latent 공간의 첫 backward step $P(z\mid \hat{z}_s, x)$와 같아 backward dynamics로 계산 가능(별도 모듈 불필요).

### 4.3 M-step 디테일 — semantics(=reward 파라미터 $\phi$) 학습
M-step은 reward function에 들어가는 모듈 $\phi$를 학습합니다:
- $P_\phi(z_0\mid x)$ (입력 인코더)와 $P_\phi(z_0\mid \hat{z}_s)$ ("symbol $s$가 가리키는 대상") — 둘 다 평균·분산을 신경망이 내는 **Gaussian**.
- 보조 reconstruction 모델 $P_\phi(x\mid z_0)$ — $x$와 $z_0$의 **mutual information을 높이도록** 압박(autoencoding).
- 절차: 각 $x$에 대해 $z_0\sim P_\phi(z_0\mid x)$, 궤적 $z_0\rightsquigarrow z_T$, 토큰 $s\sim P_\theta(s\mid z_T,x)$를 sampling → 튜플 $(x, z_0, s)$로 **VAE**(Kingma & Welling 2014) + $x$–$s$ MI를 키우는 추가 KL 항으로 $P_\phi(z_0\mid x)$를 최적화.

> **요약**: 학습 = continuous GFlowNet을 EM으로. **E-step**(θ): forward=backward 궤적확률 매칭(FL-DB + flow $g_\theta$)으로 *"무엇을 sampling할지"*. **M-step**(φ): VAE류로 *"symbol이 무슨 뜻인지"*. 둘을 번갈아 → reward에 비례하는 multimodal posterior + emergent compositional language.

> **우리 연구 연결**: "reward에 비례해 *분포로* sampling(GFlowNet) + latent를 EM으로 의미 부여"는 NES에서 *discrete symbol을 라벨 없이 자생시키는* 학습 레시피로 직접 후보입니다. 특히 forward/backward 매칭으로 latent trajectory를 학습하는 구조는 world model의 latent rollout 학습과 형식이 비슷해, *"goal/조건부 posterior를 diverse하게 sampling"* 하고 싶을 때 (단일 MAP가 아니라) GFlowNet objective를 얹는 그림이 그려집니다.

### 4.4 Pseudo-code (논문 모듈 기준 재구성)
논문에 흩어진 모듈·objective를 하나로 모은 **개념적 의사코드**입니다(Appendix B의 정확한 loss 식이 아니라 흐름 중심). 표기는 노트와 동일: `z0`=입력 인코딩, `zT`=종착점(attractor 근처), `z_hat_s`=$\hat{z}_s$, `s`=토큰열.

**(1) Sampling = inference (학습된 모델로 $x$ → symbolic code 뽑기)**
```python
def sample(x):
    z = sample(P_phi.z0_given_x(x))          # 시작점 z0  (입력 인코딩, Gaussian)
    traj = [z]
    for t in range(T):                       # forward dynamics 로 굴리기
        z = sample(P_theta.fwd(z))           #   z_{t+1} ~ N(mu_θ(z), σ_θ(z))  : Euler–Maruyama 1-step
        traj.append(z)
    zT = z                                    # attractor basin 안의 종착점

    s = []                                    # discretizer: 토큰 자기회귀 sampling
    while True:
        s_i = sample(P_theta.tok(s, zT))      #   s_i ~ P_θ(s_i | s_{:i}, zT)
        if s_i == EOS: break
        s.append(s_i)
    return traj, zT, s                        # traj = z0⇝zT,  s = 떠오른 symbolic 사고
```

**(2) Reward (Eq.1) — 한 (zT, s)이 얼마나 좋은가**
```python
def reward(x, zT, s):                         # R_φ(zT, s; x),  energy E = -log R
    z_hat_s = e_phi(s)                                       # sentence embedding
    fidelity = exp(-KL(P_phi.z0_given_x(x) || P_phi.z0_given_zhat(z_hat_s)))  # 정보 손실 ↓ → 보상 ↑
    synonym  = P_phi.s_given_z(s, zT)                        # 같은 뜻 여러 문장 보정
    prior    = P_of_s(s)                                     # 짧은 문장 선호 등
    return fidelity * synonym * prior
```

**(3) Training = GFlowNet-EM (θ와 φ를 번갈아 학습)**
```python
init θ = {fwd dynamics, bwd dynamics, discretizer(+bwd), flow g_θ}
init φ = {input enc/dec P_phi.z0_given_x, P_phi.x_given_z0,
          sentence enc e_phi, P_phi.z0_given_zhat, P_phi.s_given_z}

while not converged:
    # ---------- E-step : sampling policy θ 학습 (φ·reward 고정) ----------
    for x in minibatch:
        traj, zT, s = sample(x)                       # forward 궤적 (Eq.2 경로)
        R = reward(x, zT, s)

        logP_fwd = log P_phi.z0_given_x(traj[0], x) \
                 + sum(log P_theta.fwd(traj[t+1] | traj[t]) for t) \
                 + log P_theta.tok_seq(s | zT)                       # Eq.2
        logP_bwd = log P_of_s(s) \
                 + log P_theta.bwd(zT | z_hat_s) \
                 + sum(log P_theta.bwd(traj[t] | traj[t+1], x) for t)# Eq.3

        # forward-looking detailed balance: flow g_θ 가 미래 reward를 매개
        L_E += FL_DB_loss(logP_fwd, logP_bwd, g_theta(traj), R)
    θ ← θ - η · ∇θ L_E      # 핵심: terminal 분포가 R 에 비례하도록 flow 매칭

    # ---------- M-step : semantics φ 학습 (θ 고정) ----------
    for x in minibatch:
        traj, zT, s = sample(x)                       # 현재 policy로 (x, z0, s) 튜플 생성
        # VAE: z0 가 x 를 잘 복원하도록 + x·s 의 mutual information ↑
        L_M += VAE_loss(x, traj[0], decoder=P_phi.x_given_z0) \
             + KL_term_raising_MI(x, s)               # P_phi.z0_given_x, P_phi.z0_given_zhat 갱신
    φ ← φ - η · ∇φ L_M
```

> **읽는 포인트**: 추론은 (1)뿐 — *forward dynamics + discretizer 둘이면 작동*. backward dynamics·flow $g_\theta$·reconstruction은 전부 **학습용 보조**라 (3)에만 등장합니다. E-step이 "어떻게 sampling"(θ)을, M-step이 "무슨 뜻"(φ)을 맡는 EM 구조가 코드에서 두 루프로 분리돼 보입니다.

---

## 5. Explicit symbolic form의 역할 (§2.2)
학습엔 명시적 토큰열 $s$가 필요합니다. 그런데 *추론* 땐 dynamics만으로 되는데(§4.4-(1)), **왜 학습에 굳이 symbolic form을 끼워 넣나?** 논문은 두 역할 + 한 가지 미묘한 단서를 답합니다.

### 5.1 역할 ① — attractor의 "닻"이자 backward의 repulsor
symbolic form은 모델이 학습한 **stable concept들의 집합**을 나타내고, 각 토큰열 $s$가 latent 공간의 *좁은 영역* $\hat{z}_s$로 매핑되도록 제약합니다. 이 $\hat{z}_s$가 핵심적인 이중 역할을 합니다:
- **forward dynamics 관점**: $\hat{z}_s$는 trajectory가 빨려드는 **attractor**(끌어당김).
- **backward dynamics 관점**: $\hat{z}_s$는 역방향 궤적의 *초기 조건*이 되고, forward와 같은 smoothness prior를 받아 작은 step으로 퍼져 나갑니다. 즉 $\hat{z}_s$에서 **밀려 나가는 repulsor**(밀어냄).

쉽게 말하면: 같은 점 $\hat{z}_s$가 **앞으로 가면 골짜기(attractor), 뒤로 가면 봉우리(repulsor)**입니다. forward는 그 점으로 모이고 backward는 그 점에서 흩어지죠. 이 대칭이 GFlowNet의 forward=backward 매칭(§4.2)을 기하학적으로 떠받칩니다. symbolic form이 *그 닻의 위치*를 박아 줍니다.

### 5.2 역할 ② — systematicity·compositionality의 inductive bias 공급원
이게 §2.1 비판적 질문 박스의 답과 직결됩니다. **compositional 구조는 attractor dynamics가 아니라 symbolic form에서 옵니다:**
- 토큰열을 처리하는 **shared model weight**가 의미·구문의 일관성을 강제하고, 그 일관성이 latent 공간으로 *전파*되어 linguistic 구조를 선호하는 inductive bias가 됨.
- prior $P(s)$로 **복잡도(예: 길이) 비용**을 걸기 쉬움 — 이건 symbolic form이라 가능하지, continuous vector로는 어려움.
- **핵심 문장**: *symbolic form이 없으면, attractor dynamics 단독으로는 "discrete 표현"이라는 bias만 줄 뿐, 추가적인 compositional 구조는 보장하지 않는다.* → "discreteness"와 "compositionality"는 별개 성질이고, 전자는 dynamics가, 후자는 symbolic scaffold가 책임집니다.

### 5.3 미묘한 단서 — 그래도 "핵심 메커니즘"은 sub-symbolic이다
중요한 균형추: 위 제약들은 *학습*엔 필요하지만, **discretization의 근본 메커니즘 자체는 diffusion dynamics만으로 충분**합니다. 따라서 학습된 모델은 **마지막 tokenization 단계를 떼고 완전히 sub-symbolic하게 해석**해도, symbolic 표현과 일관된 특성을 유지합니다.
→ 즉 토큰은 *학습을 조형하는 scaffold(비계)*이지, *추론에 필수인 부품*이 아닙니다. "symbol은 sub-symbolic dynamics 위에 emerge한 그림자"라는 이 논문의 철학이 여기 응축됩니다. (Discussion에서 이 구분과 한계를 다시 다룸.)

> **요약**: symbolic form은 (①) attractor/repulsor의 닻을 박고 (②) compositional inductive bias를 공급한다. 단 (③) discretization 자체는 dynamics만으로 충분 — 토큰은 *학습용 scaffold*, 추론에선 떼어낼 수 있다.

> **우리 연구 연결**: "compositionality는 dynamics가 아니라 *symbolic scaffold + shared weight + 복잡도 prior*에서 온다"는 분해는 NES 설계에 직접적인 처방입니다. discreteness(저차원·안정 code)는 latent dynamics로 얻되, *compositional 구조는 별도의 토큰-수준 inductive bias로 주입*해야 한다는 것. 또 "추론 시 토큰을 떼어낼 수 있다"는 점은, world model 내부에선 sub-symbolic으로 빠르게 돌리고 *필요할 때만* symbolic readout을 붙이는 hybrid 추론(연구자님 sensor-in/control-out 그림)과 정확히 호응합니다.

---

## 6. Experiments & Results ① — 셋업과 attractor dynamics 검증 (§3, §4.1)
### 6.1 데이터셋 두 개
구현은 dynamics·discretizer·flow를 모두 **3-layer MLP**로, encoder/decoder는 먼저 **VAE로 pre-train**. trajectory 길이 $T=20$, 분석마다 모델 변형당 5 instance.

**(a) Hierarchical Binary Vectors (HBV)** — 계층 구조 검증용. 이상화된 phylogenetic tree(식물/동물 → 나무/꽃/물고기/새 …)를 $N$-bit 벡터로 추상화. 루트는 전부 1, 각 노드는 부모의 1들을 좌/우 절반씩 비겹치게 상속하며 깊이 $D$까지 이분. 실험은 **128-bit, depth-6**. 토큰을 **12개·6쌍**으로 묶어 각 쌍에서 하나만 쓰게 함 → branch마다 고유 토큰을 주는 *비-compositional 지름길 차단*(§2.1 박스). 총 $3^6=216$개 시퀀스(최대 길이 6).

**(b) dSprites (+color)** — 풍부한 modality 검증용. 도형 이미지에 색(7 RGB) 추가, 위치를 4×4 그리드로, shape·color·size·position 4 feature(orientation 제외). 토큰 32개, 시퀀스 최대 7.

**비교 모델**: **VQ-VAE**(codebook nearest 선택), **Emergent Communication(EC)**(speaker→token→listener, Gumbel-Softmax). 참조선 **FC**(완전 compositional = 상한), **NC**(비-compositional = 하한).

### 6.2 §4.1 — attractor는 정말 "끌어당기나"? (contractive 검증)
종착점 $z_T$가 진짜 attractor(수축점)인지 **perturbation 실험**으로 시험:
1. $x$로 trajectory를 굴려 $z_T$·토큰 $s$·embedding $\hat{z}_s$를 얻음.
2. $\hat{z}_s$에 무작위 방향·크기 $\nu$의 perturbation을 더해 새 출발점 $z_0' = \hat{z}_s + \nu$.
3. 다시 굴려 $z_{T'}'$가 $\hat{z}_s$로 **되돌아오는지** 거리 $\lVert z_{T'}' - \hat{z}_s\rVert$ 측정.

**결과(Fig.4–5)**:
- perturbation이 작으면 $\hat{z}_s$ 근처로 **복귀** → 수축성 확인. stochastic이라 정확히 한 점엔 안 모이지만, 중앙값 거리가 임계 크기까지 안정적이다 **inflection point**에서 꺾임 = **basin 경계**.
- perturbation이 커도 *가장 가까운* $\hat{z}_n$까지 거리는 그대로 → basin 탈출해도 **이웃 basin에 정착**(공간 전체가 basin으로 덮임).
- dSprites에선 perturbation 없이도 다른 $\hat{z}_n$에 더 가깝게 끝나는 경우 → **synonym embedding이 한 basin에 뭉치고 각 $\hat{z}_s$는 미세 국소 함몰**. Fig.5: $\hat{z}_s$·$\hat{z}_n$의 예측 feature가 강하게 상관 = 의미적으로 같은 개념. (§3 reward의 synonym 항이 기하로 드러남.)

> **요약**: perturbation-복귀로 $z_T$가 진짜 attractor임을 확인. 작은 교란엔 복귀, 큰 교란엔 이웃 basin, synonym은 한 basin에 뭉침.

> **우리 연구 연결**: "perturbation 후 복귀"로 attractor를 검증하는 프로토콜은 우리 latent code의 **robustness 정량화** 도구로 그대로 쓸 수 있습니다. basin 경계 inflection point로 code 분해능을, synonym basin 뭉침으로 *code collapse vs 의미적 동치*를 구분.

---

## 7. Results ② — emergent language와 sample diversity (§4.2–§4.3)
### 7.1 Feature decodability (§4.2.1) — code에 정보가 들었나
$z_T$·$\hat{z}_s$에서 **feature label을 예측하는 decoder**를 학습. HBV는 루트→prototype 경로(첫 4 branch), dSprites는 color·shape(정확도)와 size·position(RMSE). **within-distribution(WD)** vs **out-of-distribution(OOD)**(일부 prototype/feature 조합 held-out)로 평가.

**결과(Fig.6, red bars)**: attractors·VQ-VAE·EC 모두 NC 하한보다 우수(FC 상한엔 못 미침). 대부분 비등하나 — **dSprites OOD에서 attractors가 두드러지게 우월, withheld shape에서 거의 완전한 일반화.** (§2.1 박스의 "OOD 일반화 = lookup 반증"의 실측.)

### 7.2 RSA / TopSim (§4.2.2) — 표현 기하가 feature 기하를 닮았나
RSA(=TopSim): 입력쌍 거리행렬과 토큰열쌍 거리행렬의 Spearman 상관. **dSprites RSA: attractors 0.525 > VQ-VAE 0.314 > EC 0.156.** attractors의 표현 기하가 가장 compositional.

단 **RSA의 한계**(논문이 직접 지적): 선형 관계를 가정해 계층 구조엔 부적합. 예) base-2의 0,7,8 = `0000,0111,1000` — 단순 bit 비교는 "0과 8(3 bit 공유)이 7과 8(0 bit 공유)보다 비슷"하다고 *오판*. → HBV의 hierarchical relation엔 RSA 부적절 → FWC 도입.

### 7.3 Feature-wise composition (FWC, §4.2.3) — compositionality의 핵심 시험
선형성 가정을 피해, **"한 번에 feature 하나씩, 토큰 하나씩"** 묘사를 *조립*할 수 있는지 본다.
- HBV: top-level feature $y_1$에 대해 $P(y_1\mid w_1)$ 최대화 토큰 $w_1$ 선택 → 트리 따라 내려가며 누적 likelihood $P(y_1,\dots,y_i\mid w_1,\dots,w_i)$ 최대화 토큰 추가.
- dSprites: 계층 의존 없으니 7 feature 무작위 순서.
- 편향 제거 위해 **생성용·평가용 decoder를 따로** 학습. FWC 문장 vs 모델이 직접 sampling한 문장의 feature 예측력 비교.

**결과(Fig.6)**: WD에선 FWC 문장과 모델 문장의 정확도 차이가 **작거나 없음** → 모델의 정확한 묘사가 **고도로 compositional한 production rule과 일치**. OOD에선 FWC도 떨어지지만 *때로 모델보다 덜 떨어짐* — 경직된 production rule이 오히려 regularize.

### 7.4 Sample diversity (§4.3) — PLoT의 핵심(분포를 배우나)
PLoT은 점추정이 아니라 **분포**를 배워야 한다. 세 가지 신중한 측정:
1. 전체 $H(\cdot)$가 아니라 **입력별 조건부 entropy $H(\cdot\mid x)$** (한 입력에 대한 묘사 다양성).
2. 문장 다양성이 아니라 **의미(semantic content)의 entropy** (synonym `red`/`scarlet`은 무시, $P(\text{feature}\mid x)$ 선호).
3. **concept diversity**: 단일 개념 vs 여러 개념 구분 (Beta 예: 항상 $\{1,1\}$=균일분포 1개 개념 vs $\{1000,1\}$·$\{1,1000\}$=저-entropy지만 2개 개념). → feature predictor 출력 분포에 직접 entropy(Kozachenko–Leonenko estimator).

**결과**: attractors가 비교 모델보다 **훨씬 다양**.
- HBV: 평균 differential entropy **−18.198**(σ≈0.088) vs EC **−173.66**(σ≈0). VQ-VAE는 deterministic → 정의 불가($-\infty$).
- dSprites: **−9.415** vs EC **−49.973**.

**정보 손실 분포(Fig.7)**: $z_0$에서 맞춘 bit $c_0$, $z_T$에서 맞춘 bit $c_T$, 손실 $d=c_0-c_T$. bit 하나당 entropy 2배 감소 → 가설 **Geometric(0.5)**. WD 경험치 $p=0.480$(거의 0.5), KL=0.041로 **이론과 거의 일치**. OOD는 손실 적은 쪽 더 선호($p=0.621$, KL=0.183).

> **요약**: attractors는 (decodability) 정보 보존·OOD 일반화, (RSA 0.525·FWC) compositional 생성, (entropy·Geom(0.5)) PLoT다운 diverse posterior를 *명시적 codebook 없이* 달성. VQ-VAE는 diversity 0, EC는 훨씬 낮음.

> **우리 연구 연결**: "FWC로 compositionality 직접 측정", "조건부·의미 entropy로 diversity 측정", "정보 손실을 Geometric으로 모델링"은 우리 latent code 품질 평가에 **즉시 차용 가능한 metric 세트**. 특히 *생성용/평가용 decoder 분리*는 self-evaluation 편향을 막는 좋은 실험 습관.

---

## 8. Discussion · 한계 · 최종 종합 (§5)

### 8.1 이 논문이 다리를 놓은 곳
저자들의 큰 그림: **connectionist(sub-symbolic) 진영과 Bayesian(symbolic) 진영 사이의 오랜 간극에 다리를 놓는다.**
- 다른 emergent language 모델처럼 도메인 지식·사전 primitive 없이 discrete token 표현을 만들되, **discretization 방식이 근본적으로 다르다** — lookup table에서 벡터를 *고르는* 게 아니라, 정보 풍부한 state에서 출발해 **system dynamics가 점진적으로 attractor 영역에 정착**한다. 그래서 *자극 onset부터 안정 encoding까지의 중간 trajectory*가 드러나는 **통합 framework**(symbolic + sub-symbolic).
- 이 점이 뇌 기전(short-term memory, head direction, pattern separation, decision making)과 정렬됨 → **neuro-plausibility**.
- 동시에 입력별로 **diverse한 분포(=MI 반영 Bayesian posterior)**를 학습 — VAE/diffusion엔 흔하나 discrete sampler엔 보장되지 않는 성질. 이로써 **PLoT 전통**과도 정렬. → "continuous neural 기전"과 "PLoT" 양쪽에 동시 정렬한 게 핵심 기여.

### 8.2 한계 (저자 인정 + §2.1 비판과 연결)
- **가장 큰 한계 — 여전히 학습에 symbol을 *전제*한다.** latent state를 token 시퀀스로 번역하는 discretization(=decoding) 메커니즘이 학습 과정에 필요하다. "구별성을 강조하는 환경(예: 음운 primitive)에서 emerge하는 출력 시스템"으로 해석할 순 있으나, **더 완전한 설명은 명시적 discretization step 없이 end-to-end로 symbol을 만들어내야** 한다. → *미래 과제*: 학습 중에도 명시적 tokenization 없이 symbol을 생성하는 방법.
- (§2.1 박스와 연결) compositionality·posterior 적합은 **단순 domain(HBV·dSprites)**에서만 검증. combinatorially 큰 sentence space의 posterior가 제대로 학습됐는지는 *원리적으로 확인 어려움*.
- 더 정교한 task·dataset으로의 확장은 열린 문제.

### 8.3 한 페이지 종합 — 무엇을 가져갈 것인가
**이 논문의 한 문장**: *discrete·compositional·symbolic 표현을, slot을 박지 않고 **stochastic attractor dynamics(continuous GFlowNet)**로 자생시켜, PLoT를 neural하게 구현했다.*

핵심 takeaway 4가지:
1. **discretization ≠ 아키텍처**: discrete state는 energy landscape의 attractor basin으로 *emerge*시킬 수 있다(VQ/Gumbel처럼 박을 필요 없음).
2. **discreteness와 compositionality는 분리**: 전자는 dynamics가, 후자는 *symbolic scaffold + shared weight + 복잡도 prior*가 책임진다(§5).
3. **점추정이 아니라 분포**: GFlowNet으로 reward에 *비례*해 sampling → PLoT다운 multimodal posterior.
4. **trajectory의 투명성**: 입력→discrete code의 *중간 경로*가 보여, lossy 인지 과정(categorical perception 등) 모델링에 적합.

### 8.4 두 attractor 논문 비교 — [[2605 - Solve the Loop - Attractor Models for Language and Reasoning]]
| | 이 논문 (Nam et al. 2024) | Solve the Loop (2026) |
|---|---|---|
| 목적 | **인지 모델** — symbol의 emergence·PLoT | **효율적 inference** — LM/reasoning |
| attractor 정식화 | stochastic SDE의 수렴 basin, **GFlowNet**으로 분포 학습 | 결정론적 **fixed point**, **implicit diff**로 미분 |
| 무엇이 attractor인가 | sentence embedding $\hat{z}_s$ (discrete code) | 출력 embedding $\tilde{y}^\star$ (다음 토큰) |
| 다양성 | **핵심** (multimodal posterior) | 무관 (단일 고정점) |
| 공통점 | "continuous dynamics가 attractor로 수렴 → 거기서 의미를 읽는다" |

→ 두 논문을 합치면 흥미로운 가설: *latent reasoning(고정점 수렴)과 symbol emergence(stochastic basin)를 한 틀에서?* 결정론적 fixed point를 **stochastic·multimodal**로 일반화하면, "여러 그럴듯한 사고를 분포로 내는 reasoner"가 된다. (NEO/NES에서 탐색할 만한 교차점.)

> **우리 연구 연결 (종합)**: NES의 설계 원형으로 **(1)** symbol을 박지 말고 dynamics에서 끌어내되 **(2)** compositional 구조는 토큰-scaffold로 따로 주입하고 **(3)** GFlowNet으로 *분포*를 학습해 diversity를 확보하며 **(4)** 추론 시엔 토큰을 떼고 sub-symbolic으로 빠르게 — 가 한 묶음으로 따라온다. 연구자님의 *LoT/sensor-in-control-out* 그림과 정확히 맞물리며, 한계(학습에 symbol 전제)는 곧 *open problem이자 우리가 파고들 지점*이다.

---

> [!done] 공부 완료 (260615)
> 전 8스텝 완주. Model(생성·reward·GFN-EM·symbolic form) → Experiments/Results(attractor 검증·emergent language·diversity) → Discussion·한계·종합. 핵심 비판(지수적 mode vs compositionality)·IFT 없는 학습·pseudo-code·두 attractor 논문 비교 포함. 