> **논문**: Solve the Loop — Attractor Models for Language and Reasoning
> **저자**: Jacob Fein-Ashley, Paria Rashidinejad
> **기관**: University of Southern California (USC)
> **출처**: arXiv:2605.12466v1 [cs.LG], 2026-05-12 · [website](https://attractor-models.github.io/) · [code](https://github.com/jacobfa/Attractor)

> [!info] 공부 노트
> 이 노트는 AIR와 스텝바이스텝으로 공부하며 작성 중. 진행에 따라 아래로 자란다.

---

## 0. 한눈에 (TL;DR)
**문제**: Looped/recurrent Transformer는 "한 토큰을 내놓기 전에 내부 표현을 여러 번 다듬는다"는 매력적인 아이디어지만, 실제로는 ① 학습이 불안정하고 ② 메모리가 loop 수에 비례해 커지고 ③ 학습 때 정한 loop 횟수에 묶인다(추론 때 더 돌리면 오히려 성능 저하).

**핵심 아이디어**: loop를 *유한 번 펼치지(unroll) 말고*, loop가 수렴하는 **고정점(fixed point)을 직접 풀어버리자(solve)**. 즉 recurrence를 "몇 번 돌릴까"의 문제에서 "어디로 수렴하는가"의 문제로 바꾼다. 이게 **Attractor Model**.

**결과 헤드라인**:
1. **언어모델링**: 같은 파라미터 수에서 표준 Transformer·안정화 looped 모델(Parcae)을 Pareto 우위로 능가. Lambada perplexity 최대 −46.6%, 학습 FLOPs 25–31% 절감. 770M 모델이 2배 토큰으로 학습한 1.3B Transformer와 맞먹음.
2. **하드 추론(작은 모델)**: 27M 파라미터 + 약 1000 예제로 Sudoku-Extreme 91.4%, Maze-Hard 93.1%. 같은 문제에서 DeepSeek R1·Claude·o3-mini는 0%, 특화 recursive 모델(TRM)은 키우면 붕괴(27M에서 0%).
3. **새로운 현상 — Equilibrium internalization**: 고정점으로 학습하면 backbone의 *초기 추측*이 점점 고정점에 가까워져서, 추론 때 solver를 거의/완전히 빼도 성능이 거의 안 떨어짐. recurrence가 스스로 backbone 안으로 "내재화"됨.

---

## 1. Introduction — 문제의식과 핵심 발상
### 1.1 출발 질문
표준 Transformer는 토큰 하나를 **단 한 번의 feed-forward 패스**로 만든다. 자연스러운 의문: *토큰을 확정하기 전에, 모델이 자기 내부 예측을 여러 번 다듬을 수 있어야 하지 않나?*

- **Chain-of-thought(CoT)**도 일종의 refinement지만, "생각"을 *이산 토큰 채널*에 글로 써야 한다(중간 토큰을 뱉고 다시 입력으로 먹임). 내부 표현만 다듬으면 될 때도 굳이 토큰으로 써야 하니 비효율.
- 그래서 **latent/implicit thinking** + **architectural recurrence** 계열이 부상: universal/looped/recurrent-depth Transformer, looped LM(Ouro, Parcae), continuous CoT 등. 원리적으로 반복·알고리즘적 절차를 표현할 수 있고, weight sharing으로 깊이를 흉내 내며, 토큰 기반 추론의 context 비용을 줄임.

### 1.2 왜 recurrence가 실전에서 어려운가 (이 논문이 공격하는 지점)
recurrence를 "안정적인 빌딩블록"으로 쓰기 어려운 세 가지 이유:
1. **불안정한 학습** — BPTT(시간/깊이 역전파)가 필요하고, 세심한 안정화 기법 없이는 깨짐.
2. **메모리 선형 증가** — recurrent step 수 $T$에 비례해 gradient 저장 메모리가 커짐. 큰 $T$에서 memory-limited. (Geiping et al. 2025: recurrent 모델 학습이 10배 큰 feed-forward 모델만큼 FLOPs를 먹기도.)
3. **고정된 loop 횟수** — $T$를 학습 때 정해야 하고, 추론 때 $T$를 바꾸면 train–test mismatch로 성능 저하. 작은 recursive reasoner는 심지어 **"less is more"** — 모델을 키우면 성능이 *붕괴*하는 역설.

### 1.3 핵심 발상: "solve the loop"
관찰(Blayney et al. 2026): looped LM에서 **대다수 토큰의 recurrent 궤적은 결국 한 고정점으로 수렴**한다. 즉 weight-tied recurrent 블록은 사실 *어떤 고정점 계산을 근사*하고 있고, 단지 $T$ step에서 잘라낸 것뿐.

→ 그러면 유한 번 펼칠 게 아니라, **그 수렴점(고정점)을 직접 풀자.** 이는 **Deep Equilibrium Models(DEQ)**의 발상. 이름 "Attractor"는 동역학계에서 시스템이 빨려드는 상태 집합(attractor)에서 따옴.

직관적 그림: **backbone이 초기 추측을 던지고 → attractor 모듈이 그 추측을 고정점까지 다듬고 → 그 평형 상태를 출력 분포로 디코딩.** "각 토큰을 내놓기 전에 생각하는" 모델로 볼 수 있다.
$$\tilde{y}_{t+1} = \mathcal{T}_{\theta_a}(\tilde{y}_t,\, \tilde{y}_0), \qquad \tilde{y}^\star \ \text{s.t.}\ \mathcal{T}_{\theta_a}(\tilde{y}^\star, \tilde{y}_0) = \tilde{y}^\star$$
쉽게 말하면: $\tilde{y}_0$는 backbone의 첫 제안, attractor 모듈 $\mathcal{T}_{\theta_a}$를 반복 적용하면 더 이상 변하지 않는 상태 $\tilde{y}^\star$(고정점)에 도달 — 그게 최종 답. "더 이상 변하지 않는 지점 = loop의 해(solution)"라서 *solve the loop*.

### 1.4 네 가지 설계 목표 (Contributions)
Attractor Model이 노리는 것:
- **(i) 안정적 학습**, **(ii) refinement step 수에 무관한 상수 메모리**, **(iii) 명시적 unrolling보다 훨씬 싼 학습**, **(iv) 효율적 추론**, **(v) 대규모 LM과 작은 모델 하드추론 양쪽에서 강한 성능**.
- 메모리/싼 학습의 비결: 고정점을 **implicit differentiation**(음함수 정리)으로 미분 → 모든 solver step의 activation을 저장할 필요가 없음(§3.2에서 깊게).
- step 수는 **수렴 기준(residual < ε)**으로 적응적으로 결정 → 학습/추론에서 같은 모델로 step 수를 바꿀 수 있음(train–test mismatch 해소).

> **우리 연구 연결 (NEO/NES/WTM 관점, 초벌)**: "출력 임베딩 공간에서의 고정점 = 평형으로의 반복적 refinement"는 latent에서의 추론/계획과 결이 맞는다. 특히 *test-time에 step 수를 조절 가능한 적응적 깊이*, *implicit diff로 메모리 상수화*는 latent dynamics/world model의 iterative inference에 그대로 이식할 여지가 있다. (뒤 섹션에서 구체화)

---

## 2. Background — Looped Architectures (기존 looped 모델의 일반형)
Attractor Model이 무엇을 바꾸는지 이해하려면, 먼저 기존 **looped 모델의 표준 골격**을 알아야 한다.

### 2.1 표기
- 입력 시퀀스 $x = (x_1, \dots, x_n) \in \mathcal{V}^n$ (어휘 $\mathcal{V}$ 위의 토큰열), 모델 폭 $d$.
- 목표: 각 위치에서 다음 토큰 분포 $p \in \Delta(\mathcal{V})^n$ 출력.

### 2.2 looped 모델 = 세 유닛의 합성 (prelude · recurrent · coda)
거의 모든 looped 모델은 다음 세 부분으로 쓸 수 있다:
$$\tilde{x} = \mathcal{P}(x) \in \mathbb{R}^{n\times d}, \qquad h_{t+1} = \mathcal{R}(h_t,\, \tilde{x}), \qquad p = \mathcal{C}(h_T) \in \Delta(\mathcal{V})^n$$
- **Prelude** $\mathcal{P}$: 입력을 표현 $\tilde{x}$로 인코딩 (한 번 실행).
- **Recurrent** $\mathcal{R}$: **weight-tied(가중치 공유) 블록**을 잠재 상태 $h_t \in \mathbb{R}^{n\times d}$에 $T$번 반복 적용. ← 여기가 "loop".
- **Coda** $\mathcal{C}$: 마지막 상태 $h_T$를 출력 확률로 디코딩 (한 번 실행).

쉽게 말하면: 입력을 한 번 준비(prelude)하고 → 같은 블록을 $T$번 돌려 상태를 갈고닦고(recurrent) → 마지막 상태를 답으로 읽어낸다(coda). "깊은 네트워크"를 만드는 대신 *얕은 블록을 여러 번 재사용*해 깊이를 흉내 낸다.

### 2.3 기존 모델들의 두 가지 설계 선택지 (knobs)
모델들(Parcae, Huginn, Ouro 등)이 서로 다른 건 주로 이 둘을 어떻게 정하느냐다:

**(a) 잠재 상태 초기화 $h_0$** — 대개 **정보 없는(uninformative) 값**에서 시작:
$$h_0 = 0 \quad \text{또는} \quad h_0 \sim \mathcal{N}(0, \sigma^2 I)$$
"백지/노이즈"에서 출발해 loop로 점점 의미 있는 상태로 만들어간다.
→ *Attractor Model과 갈리는 첫 번째 포인트*: Attractor는 $h_0$ 대신 backbone이 만든 **의미 있는 초기 추측 $\tilde{y}_0$**에서 출발한다(warm start).

또 입력 $\tilde{x}$를 recurrent에 **첫 step에만** 주입할지(Ouro) **매 step마다** 주입할지(Huginn, Parcae)도 갈린다(덧셈 또는 concat).

**(b) recurrence 깊이 $T$** — 핵심 설계 변수. 정하는 방식 세 가지:
1. **고정(fixed)** $T$ (예: TRM).
2. **학습 중 샘플링** — step 수를 매번 랜덤하게 뽑아 학습(Huginn, Parcae 등).
3. **halting 메커니즘** — 보조 정지 헤드가 "이제 충분하다" 판단(Ouro 등).

### 2.4 그래서 무엇이 문제인가 (Step 1의 세 문제를 이 골격으로 재확인)
학습은 보통 데이터 분포 + 선택한 $T$-메커니즘에 대해 평균낸 손실을 **깊이 방향 역전파(BPTT)**로 최소화한다. 여기서 두 비용이 **$T$에 직접 묶인다**:

- **학습 FLOPs** $\propto$ recurrent step 수.
- **gradient 메모리** $\propto$ recurrent step 수 (unroll된 모든 step의 activation을 저장해야 하므로).

그리고 결정적으로 — **추론 때 $T$를 바꾸면 train–test mismatch**가 생긴다. 학습 때와 *다른 계산 그래프*로 평가되니 성능이 떨어진다. (Step 1의 "loop 횟수에 묶임" 문제의 정확한 출처.)

> **요약**: 기존 looped 모델 = (백지 초기화) + (유한 $T$ unroll) + (BPTT 학습). 세 약점(불안정·메모리 선형·$T$ 고정)이 모두 *"$T$를 미리 정하고 그만큼 펼쳐서 BPTT 한다"*는 한 점에서 나온다. → Step 3에서 Attractor Model은 이 셋을 (의미있는 초기화) + (고정점 직접 solve) + (implicit diff)로 정면 교체한다.

---

## 3. Solve the Loop with Attractor Models (§3.1 모듈 정의)
핵심 동기 한 줄: looped 모델의 recurrent 궤적은 어차피 고정점으로 수렴하니, **유한 unroll 대신 그 고정점을 직접 푼다.** 모델은 두 모듈로 구성된다 — **backbone**(초기 추측 제안) + **attractor**(고정점까지 refine).

### 3.1 공통 임베딩: tied embedding/unembedding
입력 토큰을 임베딩으로 매핑한다:
$$\tilde{x} = E(x) \in \mathbb{R}^{n\times d}$$
여기서 $E$는 **tied embedding/unembedding** — 입력 임베딩과 출력 디코딩(unembedding)이 **같은 행렬을 공유**한다. 이게 중요한 이유: 모든 중간 상태가 *출력 임베딩 공간*에 살아서, 언제든 $E^\top$로 디코딩하면 토큰 분포가 된다. 즉 refinement 궤적의 매 점 $\{y_0, y_1, \dots, y^\star\}$이 *이미 "읽어낼 수 있는 예측"*이다.

### 3.2 Backbone 모듈 — 초기 "추측" 출력 임베딩 제안
backbone $\mathcal{T}_{\theta_b}$ (보통 **큰 causal Transformer**)가 입력으로부터 초기 제안을 만든다:
$$\tilde{y}_0 = \mathcal{T}_{\theta_b}(\tilde{x}), \qquad \tilde{x} = E(x) \tag{1}$$
쉽게 말하면: 표준 Transformer가 평소처럼 "다음 토큰 예측 임베딩"을 한 번 뽑는다. 단, 이걸 *최종 답*이 아니라 **refinement의 출발점**으로 쓴다.
→ 기존 looped의 $h_0 = 0$/노이즈와 정반대. **이미 의미 있는 예측**에서 시작하므로(warm start) refinement가 안정적이다. (DEQ는 0에서 시작해 학습 후반 iteration이 폭증하는데, 여기선 후반에 오히려 안정화 — Fig 6(b).)

### 3.3 Attractor 모듈 — 고정점까지 refine
attractor $\mathcal{T}_{\theta_a}$ (보통 **더 작은** weight-tied 네트워크)가 제안을 반복적으로 다듬는다:
$$\tilde{y}_{t+1} = \mathcal{T}_{\theta_a}(\tilde{y}_t,\, \tilde{y}_0), \qquad \tilde{y}_0 = \mathcal{T}_{\theta_b}(\tilde{x}) \tag{2}$$
**핵심 디테일 — persistent injection**: 매 step마다 초기 제안 $\tilde{y}_0$를 *계속* 집어넣는다(둘째 인자). 왜? $\tilde{y}_0$ 없이 $\tilde{y}_{t+1}=\mathcal{T}_{\theta_a}(\tilde{y}_t)$만 돌리면 attractor가 *입력과 무관한* 고정점(상수 함수처럼)으로 붕괴할 수 있다. $\tilde{y}_0$를 매번 주입하면 고정점이 **제안(=입력)에 의존**하도록 고정된다. (§4에서 ablation.)

### 3.4 고정점을 "푼다"는 것 — RootFind
유한 번 굴리는 대신, attractor 사상의 **고정점**을 직접 찾는다. 잔차(residual) 함수를 정의하면:
$$A_{\theta_a}(\tilde{y}^\star, \tilde{y}_0) := \mathcal{T}_{\theta_a}(\tilde{y}^\star, \tilde{y}_0) - \tilde{y}^\star = 0 \;\;\Rightarrow\;\; \tilde{y}^\star = \mathrm{RootFind}\big(A_{\theta_a}(\cdot, \tilde{y}_0);\, y_0\big) \tag{3}$$
쉽게 말하면: "$\mathcal{T}_{\theta_a}$를 적용해도 더 이상 안 변하는 점($\mathcal{T}_{\theta_a}(\tilde y^\star,\tilde y_0)=\tilde y^\star$)"을 찾는 것 = "$A_{\theta_a}=0$인 근(root)을 찾는 것". $A=$ (한 번 적용한 값) − (자기 자신).

- **Solver**: RootFind는 **Anderson acceleration** 사용 — 과거 몇 개 iterate·residual을 묶어 단순 반복보다 빨리 고정점으로 수렴.
- **종료 조건**: 상대 잔차가 작아지면 멈춤 —
$$\frac{\lVert A_{\theta_a}(\tilde{y}_t, \tilde{y}_0)\rVert_2}{\lVert \tilde{y}_t \rVert_2} < \varepsilon \quad \text{또는} \quad t = T_{\max}$$
→ step 수가 **수렴(residual)으로 적응적으로** 결정된다. learned halting head나 미리 정한 loop 수가 아님. 그래서 **추론 때 모델을 안 바꾸고도** step 수가 달라질 수 있다(train–test mismatch 해소).

### 3.5 디코딩 & 파라미터
수렴한 평형 $\tilde{y}^\star$를 tied unembedding으로 디코딩:
$$\ell = \tilde{y}^\star E^\top \;\;\to\;\; p = \mathrm{Softmax}(\ell)$$
전체 파라미터: $\theta := (\theta_a, \theta_b, E)$ (attractor, backbone, tied 임베딩).

> **looped와의 차이 한 장 요약**: Attractor Model은 recurrence의 **시작점**과 **끝점**을 둘 다 바꾼다.
> - 시작점: $h_0=0/\text{noise}$ → backbone 제안 $\tilde{y}_0$ (warm start).
> - 끝점: 유한 unroll $h_T$ → 고정점 $\tilde{y}^\star$ (수렴까지 solve).
> - 공간: 잠재 hidden 공간 → **tied 출력 임베딩 공간**(매 iterate가 디코딩 가능).

> **우리 연구 연결**: persistent injection(=조건 $\tilde y_0$를 매 step 유지)은 goal/condition을 latent refinement 내내 잃지 않게 하는 장치로 읽힌다. world model의 latent planning에서 *goal-conditioning이 iteration 동안 희석되는 문제*에 같은 트릭(매 step goal 재주입 + 고정점 정식화)을 대볼 만하다.

---

## 4. Training & Inference (§3.2) — implicit differentiation이 핵심
여기서 "메모리가 step 수와 무관하게 상수"라는 가장 중요한 이점이 나온다. 이유는 **고정점을 미분하는 방식**에 있다.

### 4.1 문제: 고정점을 어떻게 미분하나?
표준 looped 모델은 $T$ step을 펼쳐(unroll) **그 계산 그래프 전체를 따라 역전파(BPTT)** 한다 → 모든 중간 step의 activation을 저장해야 해서 **메모리 $\propto T$**.

Attractor는 출력 $\tilde{y}^\star$가 *방정식의 해*($A_{\theta_a}(\tilde{y}^\star,\tilde{y}_0)=0$)로 정의된다. 그러면 **굳이 궤적을 펼치지 않고도** 음함수 정리(implicit function theorem, IFT)로 $\tilde{y}^\star$를 파라미터에 대해 미분할 수 있다.

### 4.2 Implicit Function Theorem 적용
손실을 $L$, 그리고 평형에서의 손실 기울기를 $v = \partial L / \partial \tilde{y}^\star$라 하자. $A_{\theta_a}(\tilde{y}^\star, \tilde{y}_0)=0$에 IFT를 적용하면:
$$\frac{\partial L}{\partial \theta} = u^\top \frac{\partial \mathcal{T}_{\theta_a}(\tilde{y}^\star, \tilde{y}_0)}{\partial \theta}, \qquad u = \big(I - J_{\tilde{y}}^\top\big)^{-1} v, \qquad J_{\tilde{y}} = \left.\frac{\partial \mathcal{T}_{\theta_a}(\tilde{y}, \tilde{y}_0)}{\partial \tilde{y}}\right|_{\tilde{y}=\tilde{y}^\star} \tag{4}$$
쉽게 말하면:
- $J_{\tilde{y}}$ = 고정점에서 attractor 사상의 **야코비안**(한 step이 출력을 어떻게 흔드는가).
- $(I - J_{\tilde{y}}^\top)^{-1}$ = "무한 번 반복했을 때의 누적 효과"를 한 방에 닫힌 형태로 준 것. (기하급수 $\sum_k (J^\top)^k = (I-J^\top)^{-1}$의 직관 — unroll을 무한대로 보낸 극한을 *역행렬 하나*로 대체.)
- 결정적으로 이 식엔 **$\tilde{y}^\star$ 하나만** 들어간다. $\tilde{y}_1, \tilde{y}_2, \dots$ 같은 중간 iterate가 **전혀 필요 없다.**

$\theta=(\theta_a,\theta_b,E)$에 대한 미분은 (i) attractor 직접 의존 $\theta_a$, 그리고 (ii) 초기화 $\tilde{y}_0 = \mathcal{T}_{\theta_b}(E(x))$를 통한 backbone $\theta_b$·tied $E$ 의존을 모두 포함한다 → backbone도 평형 기울기로 함께 학습된다(§5의 equilibrium internalization 복선).

> [!note]+ 보충 — 음함수 정리(Implicit Function Theorem)란?
>
> **한 줄**: $y$를 명시적 공식 $y=f(x)$로 못 풀어도, $y$가 *방정식 $g(x,y)=0$의 해*로 숨어 정의돼 있으면, **그 방정식을 그대로 미분해서 $\dfrac{dy}{dx}$를 구할 수 있다**는 정리.
>
> **그림으로(직관)**: 원 $x^2+y^2=1$을 생각하자. $y$를 $x$의 단일 함수로 전역적으로 못 쓴다($\pm\sqrt{1-x^2}$). 그래도 한 점 근처에서 기울기 $dy/dx$는 구할 수 있다 — 식을 통째로 미분:
> $$2x + 2y\frac{dy}{dx}=0 \;\Rightarrow\; \frac{dy}{dx}=-\frac{x}{y}$$
> $y$를 *풀지 않고* 제약식만 미분해서 기울기를 얻었다. 이게 IFT의 정신.
>
> **일반형**: $g(x^\star,y^\star)=0$이고 그 점에서 $\partial g/\partial y$가 가역이면, 국소적으로 $y(x)$가 존재하고
> $$\frac{dy}{dx} = -\left(\frac{\partial g}{\partial y}\right)^{-1}\frac{\partial g}{\partial x}$$
> (유도: $g(x,y(x))=0$을 $x$로 미분 → $\dfrac{\partial g}{\partial x} + \dfrac{\partial g}{\partial y}\dfrac{dy}{dx}=0$ → 풀면 끝.)
>
> **언제·왜 쓰나 (used for what & when)**: $y$가 *명시적 공식이 아니라 "어떤 조건의 해"*로 정의될 때 — 즉 **근(root) / 고정점(fixed point) / argmin(최적해) / 평형(equilibrium)** — 그리고 그 $y$의 파라미터에 대한 **기울기가 필요할 때**.
> - 예: DEQ·Attractor의 고정점, bilevel 최적화(하이퍼파라미터 gradient), optimization layer(OptNet), 물리 제약 레이어.
> - **결정적 이점**: 해를 구한 *과정(반복 step들)을 거슬러 역전파하지 않아도* 기울기를 얻는다 → 반복 횟수와 무관한 **메모리 $O(1)$**. (이게 이 논문이 looped 모델의 메모리 $\propto T$를 깨는 정확한 이유.)
>
> **이 논문에 대입**: $g \leftrightarrow A_{\theta_a}$, "$y$" $\leftrightarrow \tilde{y}^\star$, "$x$" $\leftrightarrow \theta$. 여기서 $A_{\theta_a}=\mathcal{T}_{\theta_a}(\tilde{y})-\tilde{y}$이므로 $\dfrac{\partial A}{\partial \tilde{y}} = J_{\tilde{y}}-I$. IFT 공식의 $(\partial g/\partial y)^{-1}$가 곧 식 (4)의 $-(I-J_{\tilde{y}}^\top)^{-1}$로 나타난다. 즉 **식 (4) = 고정점 방정식에 IFT를 적용한 결과**일 뿐이다.

### 4.3 One-step approximation: 더 싸게
$u = (I-J_{\tilde{y}}^\top)^{-1}v$를 정확히 구하려면 선형 해(linear solve)가 또 필요하다. 선행연구(Geng et al. 2022; Fung et al. 2021)를 따라 **1-step 근사**를 쓴다:
$$u \approx v$$
즉 역행렬 항을 항등으로 근사. 그러면 backward가 $A_{\theta_a}$를 통한 **단 한 번의 vector–Jacobian product**로 줄어든다.

- **메모리 $O(1)$**: solver step마다 역전파하지 않으므로, attractor 블록 메모리가 **forward iteration 수와 무관**하게 상수. (looped는 $\propto T$.)
- 논문 §4 ablation: Anderson solver의 품질 이득은 미미한 반면, 1-step 근사는 학습을 **훨씬 싸게** 만든다 → 대규모 LM 실험에서 채택.
- 단, **작은 데이터 하드추론(Sudoku/Maze)**에서는 1-step이 너무 거친 신호라 **phantom-gradient(Geng et al. 2022)**를 대신 씀(§6). TRM도 1-step으로 바꾸면 Sudoku 87.4%→56.5%로 떨어진다고 보고 — regime에 따라 backward 정밀도가 중요.

### 4.4 Algorithm 1 (요지)
**Forward**: $\tilde{x}\leftarrow E(x)$ → $\tilde{y}_0 \leftarrow \mathcal{T}_{\theta_b}(\tilde{x})$ → $A_{\theta_a}(\tilde{y};\tilde{y}_0)\leftarrow \mathcal{T}_{\theta_a}(\tilde{y},\tilde{y}_0)-\tilde{y}$ → $\tilde{y}^\star \leftarrow \mathrm{RootFind}$ → $p\leftarrow \mathrm{Softmax}(\tilde{y}^\star E^\top)$.
**Backward** ($v=\partial L/\partial\tilde{y}^\star$ 주어짐): $(I-J_{\tilde{y}}^\top)u=v$를 RootFind로 풀고(또는 1-step $u\approx v$) → $\partial L/\partial\theta \leftarrow u^\top\, \partial \mathcal{T}_{\theta_a}(\tilde{y}^\star,\tilde{y}_0)/\partial\theta$.

학습 목적함수는 **평범한 next-token cross-entropy**(고정점 출력 $y^\star$에 적용). 특별한 loss·정규화·halting loss 없음.

### 4.5 추론 & DEQ와의 차이
- **추론**: 같은 평형 계산을 재사용. peak 메모리는 attractor 한 번 forward로 bound, backbone엔 표준 KV-caching 적용. $\varepsilon$·$T_{\max}$는 **추론 시 하이퍼파라미터** — 재학습 없이 test-time 계산 예산을 조절(평형에 얼마나 다가갈지). 흥미롭게도 학습된 모델은 test-time refinement가 거의 필요 없어진다(§5).
- **vs DEQ**: Attractor도 implicit equilibrium 모델이지만 평형의 역할이 다르다. 고전 DEQ는 *예측 네트워크 자체*를 hidden-state 평형 $z^\star$로 바꾸고 별도 출력 헤드로 디코딩. 반면 Attractor는 **표준 causal Transformer backbone을 그대로 두고 그 예측 상태 위에 평형 refinement 블록을 얹는다.** 실용적 차이 셋:
  1. solver를 *의미 있는 제안 $\tilde{y}_0$*에서 초기화 (DEQ는 0 등 무정보값).
  2. 추론 정지를 *잔차 허용오차 $\varepsilon$*로 (고정 깊이·halting head 아님).
  3. DEQ는 블록을 늘리면 성능이 나빠지는데, Attractor는 임의 깊이 backbone + 가변 solver 블록을 허용.

> **우리 연구 연결**: IFT로 "iterative inference를 메모리 $O(1)$로 미분"하는 레시피는, latent world model에서 *반복적 추론/계획을 end-to-end로 학습*할 때 BPTT 메모리 병목을 푸는 직접적 도구다. 특히 NEO류의 latent planning이 inner-loop 최적화를 포함한다면, 그 inner-loop을 고정점으로 보고 implicit diff로 바깥 학습에 붙이는 구도가 가능. one-step vs phantom-gradient의 regime 의존성(데이터·모델 크기)도 우리 실험 설계에 참고점.

---

## 5. Equilibrium Internalization & Stability (§3.3)
이 논문의 가장 흥미로운 통찰. 예측을 평형 $\tilde{y}^\star$로 정의했는데, **학습이 끝나면 backbone의 초기 제안 $\tilde{y}_0$가 이미 평형 $\tilde{y}^\star$ 근처에 가 있더라**(Fig 6, 7). 즉 solver가 할 일이 거의 없어진다. 이 현상을 **equilibrium internalization(평형 내재화)**이라 부른다.

### 5.1 왜 일어나나 — "움직이는 교사(moving teacher)"
직관: attractor 모듈이 backbone에게 **계속 정답 위치를 가리켜 주는 움직이는 교사** 역할을 한다(일종의 **automatic curriculum**, 자기-증류 self-distillation).

메커니즘을 단계로:
1. **학습 초기**: $\tilde{y}_0$는 엉성한 예측 → solver가 $\tilde{y}^\star$까지 *상당한 refinement*를 해야 함(많은 step).
2. **핵심**: $\tilde{y}_0$와 $\tilde{y}^\star$는 **같은 tied 출력 임베딩 공간**에 산다(§3.1). 그리고 평형을 통과한 기울기(식 4)는 $\tilde{y}_0 = \mathcal{T}_{\theta_b}(E(x))$를 거쳐 **backbone $\theta_b$까지 흘러간다.**
3. 그 기울기는 backbone에게 *"solver가 도달했을 그 상태($\tilde{y}^\star$)를 처음부터 내놓아라"*라고 가르친다.
4. backbone이 충분히 표현력 있으면 → **예측의 대부분을 $\tilde{y}_0$ 안으로 흡수**, solver는 가벼운 마무리 refinement만 담당.

쉽게 말하면: solver가 매번 "정답은 여기야"라고 끌어주고, backbone이 그걸 보고 "그럼 처음부터 거기로 갈게"라고 학습한다. 학습이 진행될수록 backbone의 출발점이 도착점에 붙어서, **필요한 solver step 수가 저절로 줄어든다.** recurrence가 *moving training target*으로 작동해 backbone에 자기 계산을 증류하는 셈.

> **실용적 귀결(§6에서 실측)**: 학습 후엔 추론 때 **solver를 거의/완전히 제거해도 성능 저하가 작다.** 즉 "생각(refinement)"을 학습 중에 backbone weight 안으로 내면화 → 추론은 거의 표준 1-pass Transformer처럼 싸진다.

### 5.2 안정성 — $(I-J_{\tilde{y}}^\top)^{-1}$가 만드는 "방어벽"
평형 학습은 recurrent 사상을 **수렴하는(contractive) 동역학** 쪽으로 *자동으로* 민다. 이유는 implicit gradient(식 4) 안의 역행렬 인자:
$$\big(I - J_{\tilde{y}}^\top\big)^{-1}$$
- attractor 사상이 **비수축(non-contractive)** 영역에 가까워지면(야코비안 $J_{\tilde{y}}$의 스펙트럴 반경이 1에 근접) 이 역행렬이 **ill-conditioned(폭발적으로 커짐)** → 기울기가 거대해져 학습이 그 영역을 **회피**한다.
- 즉 식 자체에 **"불안정한(발산하는) 고정점으로 가지 마라"는 장벽**이 내장돼 있다.

**fixed-loop 학습과의 대비**: 유한 $T$ unroll 학습은 *정확히 $T$ step에서만 맞는* 궤적을 배울 수 있다 — 추론 때 loop를 더 돌리면 오히려 발산/성능 저하(train–test mismatch). Attractor는 처음부터 "수렴하는 고정점"을 목표로 하므로 step 수에 robust. (이론은 Appendix B, B.3.)

> **세 줄 요약**: ① 평형으로 학습하면 backbone이 solver를 내재화(internalization)해 추론이 싸진다. ② tied 공간 + 평형 기울기가 backbone까지 흘러서 가능. ③ $(I-J^\top)^{-1}$ 항이 비수축 동역학을 자동으로 벌주어 안정성을 준다.

> **우리 연구 연결**: "iterative refinement를 학습 중에 한-패스 표현으로 증류"는 곧 *test-time compute를 train-time으로 옮기는* 메커니즘. world model/planner를 무겁게 추론하다가, 학습이 익으면 amortized inference로 싸지는 그림과 정확히 같다. internalization을 *의도적으로 끝까지 밀면* solver-free 추론 모델이 되고, 덜 밀면 test-time에 더 "생각"하는 모델이 된다 — 이 trade-off 노브를 우리 latent planner에 노출시키는 설계가 흥미롭다.

---
