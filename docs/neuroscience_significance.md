# Attractor Quantization의 계산신경과학적 의미

> 입장(position)·논의 문서. AQ(Attractor Quantization) 모델 — perception으로부터 supervision 없이
> 이산·조합적(compositional) 개념을 attractor 동역학으로 형성 — 을 계산신경과학 관점에서 논한다.
> 관련: `proposal.md`, `categorical_aqvae_design.md`, `related_paper/Discrete...attractor dynamics.md`(Nam 2024).

---

## 초록 (Abstract)
뇌는 연속적인 감각 입력으로부터 **명시적 라벨 없이** 이산적이고 조합 가능한 개념(음소·색·물체·범주)을 형성한다. 본 모델(AQ)은 이 과정을 **연속 잠재 공간 위 attractor 동역학의 고정점**으로 구현한다: 지각 입력이 학습된 **저장 패턴(Modern Hopfield associative memory)** 중 하나로 수렴하고, 그 수렴 상태가 하나의 이산 개념이 된다. 학습은 재구성(자기지도)만으로 이뤄져 **개념이 감각 통계로부터 자생**한다. 잠재를 factor별로 분해(8 factor × 4D, 각자 자체 attractor 집합)함으로써 표현은 **조합적**이 된다 — 장면 = 모양 × 색 × 위치의 이산 개념 조합. 본 글은 이 구조가 (1) **attractor dynamics**(작업기억·결정·head-direction·범주지각), (2) **associative memory**(해마 CA3 pattern completion, Modern Hopfield), (3) **unsupervised 개념 형성**(예측부호화/오토인코딩), (4) **factorized·compositional 코드**(disentangled IT 표상), (5) **평형 기반 학습**(equilibrium propagation)과 어떻게 정렬되는지를 논하고, 검증 가능한 예측과 한계를 제시한다.

---

## 1. 서론 — 핵심 질문
인지의 근본 수수께끼 하나: *뇌는 연속·고차원·노이즈 섞인 감각으로부터 어떻게 안정적이고 이산적인 개념을, 그것도 라벨 없이, 그리고 조합 가능하게 만들어내는가?* 상징주의(symbolic)는 이산·조합 구조를 가정하지만 "그 상징이 어디서 오는가"를 답하지 못하고, 연결주의(connectionist)는 연속 분산 표현을 주지만 이산성·조합성의 출처가 모호하다(Fodor & Pylyshyn 1988). AQ는 이 간극을 **동역학계의 수렴**으로 메운다: 이산성은 아키텍처에 *박힌(slot)* 것이 아니라 attractor basin으로 *생성(emergent)* 된다(cf. Nam et al. 2024).

## 2. 배경 — 뇌의 attractor 동역학
**attractor network**는 신경 활동이 빨려드는 안정 상태(고정점/다양체)로 정보를 유지·범주화하는 계산 원리로, 다수의 뇌 영역에서 증거가 축적돼 있다:
- **작업기억**: 전전두엽의 지속발화(persistent activity)는 이산 bump-attractor로 모델링됨(Wang 1999; Compte et al. 2000).
- **결정**: 누적·선택은 line/point attractor로 — 특히 쥐 **ALM(anterior lateral motor cortex)** 의 광유전 교란 실험이 **attractor basin과 결정 경계의 인과적 증거**를 제공(Inagaki et al. 2019; Finkelstein et al. 2021): 작은 교란엔 복귀, 큰 교란은 다른 attractor로 이동.
- **연속 attractor**: head-direction의 ring attractor(초파리 ellipsoid body; Seelig & Jayaraman 2015), grid/place cell의 연속 attractor(공간 부호화).
- **범주지각(categorical perception)**: 연속 자극을 이산 범주로(음소 Liberman; 색). 자극 onset부터 안정 부호화까지 *점진적 궤적 후 범주화*가 관측됨(McMurray et al. 2002).

→ **이산 attractor = 이산 정보를 유지하는 안정 신경 상태**라는 그림이 뇌 전반에 걸쳐 정립돼 있다. AQ의 고정점이 바로 이것이다.

## 3. 배경 — associative memory와 Modern Hopfield
Hopfield network(1982)는 **content-addressable(내용주소) 연상기억**의 고전 모델 — 부분/노이즈 입력에서 저장 패턴으로 *pattern completion*. 해마 **CA3**의 recurrent collateral이 autoassociative memory로 기능한다는 이론(Marr 1971; Treves & Rolls 1994)과 직접 맞닿는다. **Modern Hopfield**(Ramsauer et al. 2020)는 지수적 저장 용량 + softmax 검색을 주며 transformer attention과 등가다.

→ AQ의 핵심 층(`AQLayer`)은 **학습된 저장 패턴을 가진 Modern Hopfield 연상기억**이다. 지각 context가 query가 되어 **가장 가까운 개념을 retrieval**한다 — CA3식 pattern completion의 직접 대응.

## 4. 모델 — 계산신경과학적 해석
### 4.1 이산 개념 = attractor (범주지각의 기전)
AQ map $T(y)=X\,\mathrm{softmax}(\beta X^\top(y+c))$의 고정점은 저장 패턴(개념) 중 하나다. 연속 입력 $c$가 들어오면 동역학이 한 basin으로 **수렴(snap)** — 이는 **범주지각** 그 자체다. 더 나아가:
- **수렴 *궤적*(transient)**: 입력→고정점까지의 중간 경로는 *자극 onset 후 점진적 변화 → 범주 확정*(McMurray)의 신경 상관으로 읽힌다. AQ는 이 궤적을 명시적으로 노출한다(viz: context→attractor 수렴).
- **basin 구조**: 우리 perturbation-복귀 실험(작은 교란 복귀, 큰 교란 이웃 basin 이탈)은 ALM 광유전 결과와 *형식적으로 동형*이다.
- **stiffness $\beta$**: 범주 경계의 날카로움(within-category 압축 vs between-category 분리)을 한 노브로 제어 — 범주지각의 "warping"에 대응.

### 4.2 associative memory: 지각 → 검색
인코더(감각 처리) → context $c$ → Hopfield retrieval → 개념. 이는 **지각이 기억된 prototype으로의 attractor 수렴**이라는 고전적 관점(Hopfield; CA3)을 학습 가능한 형태로 구현. 저장 패턴 = *prototype/개념*, retrieval = *재인(recognition)*.

### 4.3 unsupervised 개념 형성 (자기지도/예측)
결정적으로, AQ의 개념(저장 패턴)은 **재구성 손실만으로** 학습된다 — 라벨 없음. 이는 뇌가 **감각 예측/재구성**(예측부호화 Rao & Ballard 1999; Friston 2010; 오토인코딩 cortex)으로 지각 범주를 *자생*시킨다는 가설의 구현이다. "discrete token을 내라"는 압력(이산 병목) 아래 **범주가 감각 통계로부터 emerge**한다.

### 4.4 compositionality = factorized attractors
AQ는 잠재를 **독립 factor**로 분해하고 각 factor가 *자체 attractor 집합*을 가진다 → 장면 = factor별 이산 개념의 **조합**(shape × hue × scale × …). 신경과학적 함의:
- **factorized/disentangled 신경 코드**: IT 얼굴 patch 뉴런이 β-VAE식 disentangled 축과 정렬된다는 발견(Higgins et al. 2021)과 결을 같이함 — 뇌도 분리된 factor 축으로 부호화.
- **결합 문제(binding)**: 분리된 factor를 조합해 객체를 구성하는 것은 feature binding의 한 계산적 형태.
- **구조적/factorized 표현**: grid cell·Tolman-Eichenbaum Machine(Whittington et al. 2020)의 factorized 구조와 통한다.
- **PLoT**: Categorical AQ-VAE의 *factorized categorical prior*는 **확률적 사고의 언어(Goodman et al. 2015)** — 개념 분포·조합 생성 — 의 neural 구현(Nam 2024와 동일 정신, 결정론적 IFT 학습으로).

### 4.5 energy landscape와 평형 기반 학습
Hopfield attractor는 **에너지 지형**을 가지며, basin = 낮은 골짜기 = 개념. 학습 = 에너지 지형을 데이터에 맞게 조형. 결정적으로 AQ의 gradient는 **고정점(평형)에서 implicit differentiation(IFT)** 으로 계산된다 — 이는 **equilibrium propagation**(Scellier & Bengio 2017; 평형 상태 차이로 gradient를 얻는 *생물학적으로 그럴듯한* 학습 규칙)과 같은 수학적 가족이다. 즉 "평형을 미분한다"는 AQ의 학습 원리는 backprop의 비국소성 문제를 **국소·평형 기반 가소성**으로 옮길 다리를 품고 있다(§7 한계 참조).

### 4.6 symbolic ↔ sub-symbolic 가교
AQ에서 **상징(이산 개념)은 sub-symbolic 연속 동역학 위에 emerge한 안정 상태**다. 추론 시 이산 readout(argmax)을 떼면 완전히 연속 동역학으로 해석 가능 — "상징은 연속 동역학의 그림자"라는 통합 관점(Nam 2024 §5)을 공유한다. 이는 neuro-symbolic 통합을 *동역학에 grounding*하는 한 방식이다.

## 5. 경험적 정렬과 검증 가능한 예측
모델이 단순 은유가 아니라 **검증 가능한 계산 모델**임을 보이는 지점들:
1. **수렴 궤적**: 자극 후 신경 상태가 prototype으로 점진 수렴(범주화 RT·궤적) → 신경기록과 비교 가능.
2. **basin/경계**: 교란-복귀(perturbation-return) 프로토콜 → ALM류 광유전 실험의 *정량적* 대응(임계 교란 크기 = basin 폭).
3. **범주지각 warping**: stiffness↑ → within-category 압축·between-category 확대 예측.
4. **factorized 코드**: 학습된 factor 축이 신경 disentanglement(IT, Higgins 2021)와 정렬되는지.
5. **dead/active 코드**: 사용 분포(perplexity)가 자극 통계의 cardinality를 반영(예: 적은 범주=적은 활성 attractor).

## 6. 선행 모델과의 관계
- **Hopfield/CA3 associative memory**: AQ층이 그 학습 가능·이산 개념 버전.
- **Nam et al. 2024**(stochastic attractor + GFlowNet으로 symbol emergence): 인지 모델로서 가장 가까운 친척. AQ는 **결정론적 고정점 + exact IFT**(STE 없는 미분가능)로, *효율·정확 학습*과 *factorized·variational* 확장을 더함. 두 모델을 합치면 "여러 그럴듯한 지각을 분포로 내는 attractor 추론기"(Nam 노트 §8.4의 가설).
- **예측부호화/오토인코딩 cortex**: 자기지도 개념 형성의 학습 신호 출처.
- **PLoT(Goodman 2015)**: Categorical AQ-VAE의 prior가 그 neural 구현.
- **disentanglement·factorized 신경 코드**(Higgins 2021; TEM): 조합 표현의 신경 증거.

## 7. 한계 (정직하게)
- **Marr의 수준**: AQ는 *계산·알고리즘 수준*의 normative 모델이지, 시냅스·회로 수준 기전 모델이 아니다. 인코더는 표준 CNN(생물학적 주장 없음).
- **학습 규칙의 생물 타당성**: 현재 학습은 backprop+IFT로, 그대로는 비생물적. 단 §4.5의 **equilibrium propagation** 연결이 *원리적* 가교를 제공 — 향후 국소 평형 가소성으로의 이행이 연구 과제.
- **이산화 기전 단순화**: softmax 온도(β)로 이산성을 제어하는 건 추상화. 실제 신경 이산화(스파이크·승자독식 회로)와의 대응은 추가 연구 필요.
- **factor 독립 가정**: 현 factorized prior는 독립 가정(3D Shapes엔 정확하나 실세계 상관 factor엔 autoregressive 필요).

## 8. 의의와 전망
AQ는 **(i) associative memory(Hopfield) (ii) attractor 동역학(범주지각·작업기억) (iii) unsupervised 개념 형성(예측/자기지도) (iv) factorized·compositional 코드(PLoT·disentanglement) (v) 평형 기반 학습(equilibrium propagation)** 을 *하나의, 실제 지각(이미지)으로 학습되는* 모델에 통합한다. 즉 "감각으로부터 라벨 없이 이산·조합 개념이 어떻게 생기나"라는 인지의 핵심 질문에 대한 **계산적·동역학적 가설**을 제시하고, perturbation·궤적·factor 정렬 등 **신경실험과 맞댈 예측**을 낳는다. 기계학습적 기여(STE 없는 미분가능 이산화)는 동시에, 이 인지 모델을 *대규모 실제 지각으로 학습 가능*하게 만드는 수단이다.

> 한 줄: **AQ는 "지각 → attractor 수렴 → 이산·조합 개념"을 학습 가능·미분가능하게 구현한, 신경과학적으로 근거 있는 개념 형성 모델이다.**

---

## 참고 (대표)
- Hopfield (1982) *Neural networks and physical systems with emergent collective computational abilities.* PNAS.
- Ramsauer et al. (2020) *Hopfield Networks is All You Need.* arXiv:2008.02217.
- Wang (1999); Compte et al. (2000) — 작업기억 attractor.
- Inagaki et al. (2019, Nature); Finkelstein et al. (2021) — ALM 결정 attractor·광유전 basin.
- Seelig & Jayaraman (2015) — head-direction ring attractor.
- Liberman et al.; McMurray et al. (2002) — 범주지각.
- Marr (1971); Treves & Rolls (1994) — 해마 CA3 autoassociative memory.
- Rao & Ballard (1999); Friston (2010) — 예측부호화.
- Scellier & Bengio (2017) *Equilibrium Propagation.* Front. Comput. Neurosci.
- Higgins et al. (2021) *Unsupervised deep learning identifies semantic disentanglement in single IT neurons.* Nat. Commun.
- Whittington et al. (2020) — Tolman-Eichenbaum Machine.
- Goodman, Tenenbaum, Gerstenberg (2015) — Probabilistic Language of Thought.
- Nam, Elmoznino, Malkin, McClelland, Lajoie, Bengio (2024) *Discrete, compositional, symbolic representations through attractor dynamics.* arXiv:2310.01807.
- Fodor & Pylyshyn (1988) — connectionism과 systematicity.
