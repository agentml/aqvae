# AQ-VAE의 국소(생물학적으로 그럴듯한) 학습 — 설계 노트

> AQ-VAE를 backprop 없이 **국소·평형 기반 가소성**으로 학습시키는 구체 설계.
> 배경: `neuroscience_significance.md` §4.5의 "평형을 미분한다 → 국소 학습으로의 다리"를 기전·수식·로드맵으로 발전.
> 성격: 연구 설계(미구현). 핵심 주장 — AQ는 *표상(attractor)* 뿐 아니라 *학습 기전* 까지 국소화 가능한 부류.

---

## 0. 목표
$x \xrightarrow{\text{enc}} c \xrightarrow{\text{AQ 평형}} y^\star \xrightarrow{\text{dec}} \hat x$ 전체를 **(i) 전역 오차 전파 없이, (ii) weight transport 없이, (iii) 전·후 뉴런 활동 같은 국소 정보만으로** 학습. backprop+IFT로 얻는 gradient와 *같은 방향*을 국소 규칙으로 재현하는 것이 이상.

## 1. 무엇이 이미 국소 친화적인가 (분해)
| 부분 | 성격 | 국소화 경로 |
|---|---|---|
| **AQ층**(Hopfield 평형) | 에너지·고정점 | **EqProp / 경쟁 Hebbian** (직접) |
| **softmax retrieval** | 정규화 | **divisive normalization** 회로(국소 측면억제) |
| **encoder/decoder**(conv) | feedforward | **Predictive Coding**(국소 오차 최소화) |
| **prior**(변분) | 사용 통계 | **국소 빈도 누적**(Hebbian) |

→ AQ층은 본래 국소화가 쉽고, enc/dec만 PC로 바꾸면 전체가 국소가 된다.

## 2. 도구상자 (계보)
- **Almeida–Pineda recurrent backprop (1987)**: 고정점 $y^\star$의 gradient를 IFT로 — *우리 IFT와 동일*.
- **Contrastive Hebbian Learning**(Movellan 1991) / **Equilibrium Propagation**(Scellier & Bengio 2017): *같은 gradient*를 **자유·넛지 두 평형의 차이 + 국소 규칙**으로.
- **Predictive Coding**(Rao & Ballard 1999): 층별 예측-오차 최소화, 국소 Hebbian. **Whittington & Bogacz 2017**: PC가 backprop을 국소 규칙으로 *근사*함을 증명.
- **경쟁학습/SOM**(Rumelhart & Zipser 1985; Kohonen): prototype을 입력 쪽으로 책임(responsibility)만큼 이동 — 국소 Hebbian.
- **Divisive normalization**(Heeger 1992; Carandini & Heeger 2012): softmax/정규화의 피질 표준 회로.

## 3. ★핵심 통찰 — AQ 패턴 학습 = soft competitive Hebbian
AQ map $T(y)=\sum_k w_k X_k,\ w=\mathrm{softmax}(\beta X^\top(y+c))$. 패턴 $X_k$의 *국소* 갱신은 **soft 경쟁학습** 형태로 떨어진다:
$$\boxed{\Delta X_k \;\propto\; w_k\,(s - X_k)}\qquad s=\text{국소 평형 활동}(y^\star{+}c\ \text{근방})$$
- "각 패턴을 입력 쪽으로, **책임 $w_k$만큼** 이동" = SOM/EM/EMA-codebook(VQ-VAE)의 그 규칙.
- **완전 국소**: $X_k$(시냅스), 입력 활동 $s$(전), 책임 $w_k$(softmax=정규화로 국소 계산) 만 필요. 전역 오차 불필요.
- **생물 대응**: Hebbian(전·후 상관) + **측면억제/divisive normalization**(승자-다수독식 $w$) + **신경조절 gain $\beta$**(온도). 모두 알려진 피질 motif.
- **EqProp이 주는 정확판**: $s$를 *자유 평형*과 *재구성 오차로 넛지한 평형*의 차이로 두면 $\Delta X_k$가 backprop-IFT gradient와 일치(§5).

> 즉 **이산 코드(개념) 학습은 본래 국소 경쟁 Hebbian**이다. AQ의 가장 신경학적인 부분.

## 4. encoder/decoder의 국소화 — Predictive Coding
conv enc/dec를 **PC 계층**으로 재구성:
- 각 층에 **value 유닛 $v_\ell$** 과 **error 유닛 $\varepsilon_\ell = v_\ell - g_\ell(v_{\ell+1})$**(위층의 생성 예측과의 차).
- **추론(inference)**: $\dot v_\ell \propto -\varepsilon_\ell + W_\ell^\top \varepsilon_{\ell-1}$ 로 *예측오차 최소화*까지 settling(국소).
- **학습**: $\Delta W_\ell \propto \varepsilon_{\ell-1}\, v_\ell^\top$ — **국소 Hebbian**(전 활동 × 후 오차).
- Whittington–Bogacz: 이 settling+Hebbian이 backprop gradient를 근사. enc=인식경로, dec=생성경로.

## 5. 결합 — 두 가지 구체 설계
### 설계 A (하이브리드, 근시일 현실안)
- **enc/dec = PC**(국소 오차·Hebbian), **AQ층 = 경쟁 Hebbian(§3)**, 결합은 **EqProp 2-phase**:
  1. **자유 단계**: $x$를 enc에 클램프 → PC settling → context $c$ → AQ 평형 $y^\star_{\text{free}}$ → dec 예측. 모든 층 평형.
  2. **넛지 단계**: 재구성 오차로 출력을 $x$ 쪽으로 약하게 클램프(β-nudge) → 전체가 약간 다른 평형 $\{\cdot\}_{\text{nudge}}$로.
  3. **갱신(전부 국소)**: PC 가중치 $\Delta W\propto \varepsilon v^\top$ 의 두-평형 차; AQ 패턴 $\Delta X_k\propto w_k(s_{\text{nudge}}-s_{\text{free}})$.
- → **전역 backward 없이** 한 번의 자유+넛지 settling으로 전 파라미터 갱신.

### 설계 B (완전 통합 에너지 모델, 이상안)
- enc·AQ·dec를 **하나의 에너지 $E(\text{전상태}; x)$** 로 묶고(양방향 결합, 생성↔인식 대칭), 전체에 EqProp.
- 가장 깨끗하지만 대칭 가중치·양방향 회로 등 구조 변경 大. 장기 과제.

## 6. 변분(Categorical AQ-VAE)의 국소화
- **posterior** $q=w^\star$는 이미 국소(softmax/정규화).
- **KL**(을 향한 압력)은 국소화 가능: per-sample은 retrieval 자체가 담당, **prior $\pi_i$ 갱신 = 코드 사용빈도의 국소 누적**($\Delta\pi_i \propto \bar w_i - \pi_i$, EMA) = Hebbian 통계.
- **생성**: prior에서 샘플 → dec 생성경로(PC) 통과. 모두 국소.
- → 변분화가 오히려 국소학습과 *친화적*(KL이 전역 손실이 아니라 국소 정규화/통계로 분해됨).

## 7. 이산화·포화의 국소학습 함의
- 고 β 포화 → softmax gradient ≈0 → *어떤* 국소 규칙도 신호 소실(collapse 분석과 동형).
- → **soft 영역에서 학습 후 어닐링**(train soft / anneal to discrete) 원칙이 **국소학습에서도 그대로 필요**. 신경학적으로: 발달 초기 낮은 gain(고온)에서 범주 조직화 → 성숙하며 gain↑(이산화) — *발달적 어닐링* 가설과 호응.
- $\beta$(=gain)는 **신경조절(ACh/도파민)** 로 제어되는 양으로 해석 가능.

## 8. 검증 실험 (국소 ≈ backprop-IFT 인가)
1. **gradient 정렬**: 작은 AQ층에서 §3 경쟁-Hebbian / EqProp 갱신 벡터 vs autograd-IFT gradient의 **코사인 유사도** 측정(Whittington–Bogacz식). ≈1이면 국소판이 정확.
2. **소규모 end-to-end**: 설계 A로 3D Shapes(또는 더 작은) 학습 → backprop 버전과 recon·disentanglement 비교.
3. **two-phase 비용**: settling 반복 수 vs 정확도 trade-off.

## 9. 한계·미해결
- **enc/dec PC화**가 본 작업의 대부분(현 conv는 비평형). 표준 conv→PC 변환은 비자명.
- **EqProp 가정**: 에너지 하강 동역학(우리 one-step IFT는 근사), 약한 넛지, 대칭 가중치(weight transport의 *완화판* — PC/EqProp도 근사 대칭 필요; feedback alignment로 완화 가능).
- **경쟁-Hebbian = exact IFT 아님**: 지배항이자 EqProp으로 정확화 가능하나 일치 검증 필요(§8).
- 포화-소실(§7)은 국소에서도 남음.

## 10. 로드맵
1. **§8.1 gradient 정렬 실험**(작은 AQ층, 경쟁-Hebbian/EqProp vs IFT) — 가장 싸고 결정적.
2. **AQ층만 국소화**(patterns 경쟁-Hebbian + β-nudge) + enc/dec는 backprop 유지 → 부분 검증.
3. **enc/dec PC화** → 설계 A 완성(완전 국소).
4. (장기) 설계 B 통합 에너지 모델, 발달적 어닐링·신경조절 gain 가설.

---

## 한 줄
**AQ-VAE의 학습은 (i) 패턴=경쟁 Hebbian, (ii) retrieval=divisive normalization, (iii) enc/dec=predictive coding, (iv) 결합=equilibrium propagation, (v) prior=국소 사용통계 로 분해되어 — 전역 backprop 없이 국소·평형 가소성으로 학습 가능한 부류에 속한다. "어디서 개념이 오나"에 더해 "어떻게 국소적으로 배우나"까지 신경학적 후보를 제시.**

## 참고
- Almeida (1987); Pineda (1987) — recurrent backprop(고정점 미분).
- Movellan (1991) — Contrastive Hebbian Learning.
- Scellier & Bengio (2017) — Equilibrium Propagation.
- Rao & Ballard (1999); Whittington & Bogacz (2017) — Predictive Coding ≈ backprop.
- Rumelhart & Zipser (1985); Kohonen (1982) — 경쟁학습·SOM.
- Heeger (1992); Carandini & Heeger (2012) — divisive normalization.
- Lillicrap et al. (2016) — feedback alignment(weight transport 완화).
- van den Oord et al. (2017) — VQ-VAE EMA codebook(= 경쟁 Hebbian의 ML판).
