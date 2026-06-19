# 이름·제목 노트

> 모델/레이어 명명, 논문 제목 후보 정리. (대화에서 나온 결정·후보 추적.)

---

## 모델/레이어 이름 — **AQ (Attractor Quantization)** 확정
정체성: *"VQ(Vector Quantization)를 attractor 동역학으로 대체한 미분가능 양자화."*
- "categorical/discrete"는 quantization이 함의, VQ와 즉시 대구(AQ-VAE ↔ VQ-VAE).

**리네이밍 매핑 (실행 대기)**:
| 현재 | → AQ |
|---|---|
| `FactorizedHopfield` (핵심 레이어) | `AQLayer` (또는 `AQ`) |
| `HopfieldAttractor` (단일 factor) | `AQUnit` (또는 유지) |
| `DAVAE` | `AQVAE` |
| `model_davae.py` | `model_aqvae.py` |
| `attractor_loss` | `aq_loss` |
| `--arch attractor` | `--arch aq` (로더는 기존 ckpt `attractor` 매핑 호환) |
| eval/docs 용어 | AQ로 통일 |
> `src/attractors/`(solver·IFT)는 일반 부품이라 유지.
> GitHub repo: `agentml/aqvae` (private).

---

## 논문 제목 후보

**브랜딩(방법명+설명) — 제1픽**
- **Attractor Quantization: Fully Differentiable Discrete Representation Learning**
- AQ-VAE: Differentiable Discrete Representations via Attractor Dynamics

**메커니즘 강조**
- Differentiable Discrete Representations via Attractor Dynamics
- Learning Discrete Representations by Solving for Attractors
- Implicit Differentiation for Discrete Representation Learning

**차별점(STE) 강조 — 도발적**
- Beyond Straight-Through: Discrete Representations via Attractor Dynamics
- Attractor Quantization: Discrete Codes without the Straight-Through Estimator
- Exact Gradients for Discrete Representations via Attractor Quantization

**원안(사용자)**
- Fully Differentiable Discrete Representation Learning  (정확하나 generic — 방법 정체성 약함)

### 권장
- **브랜딩 우선** → `Attractor Quantization: Fully Differentiable Discrete Representation Learning`
- **임팩트 우선** → `Beyond Straight-Through: Discrete Representations via Attractor Dynamics`
- narrative: 초록 첫 문장에서 *"unlike the straight-through estimator, AQ yields exact gradients (via implicit differentiation)"* 로 차별점 강조.

**상태**: 제목 미확정(후보 단계). 모델명 AQ는 확정, 코드 리네이밍은 대기.
