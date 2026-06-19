# Categorical AQ-VAE — 설계

> deterministic AQ-AE → 변분(Categorical VAE) 전환 설계. 상태: **설계(미구현)**. 구현은 허락 후.
> 관련: `0617_proposal.md`, `0617_design.md`, `0617_collapse_analysis.md`, `0618_todo.md`(⑥ variational 확장).

---

## 1. 목표
factor별 softmax 코드 $w^\star$를 posterior로 삼고 prior+KL을 얹어 한 번에:
- **(a) ELBO 변분 학습**, **(b) 생성**(코드 샘플→디코딩), **(c) collapse·disentanglement 압력**.

핵심 통찰: **현 모델은 이미 discrete latent variable model** — Hopfield retrieval의 수렴 softmax weight $w^\star$가 K개 attractor에 대한 posterior다. VAE화 = 거기에 prior+KL을 더하는 것.

## 2. softmax 코드 $w^\star$ (무엇을 posterior로 쓰나)
map: $T(y)=X\,\mathrm{softmax}(\beta X^\top(y+c))$. 고정점 $y^\star$에서
$$w^\star_i=\mathrm{softmax}\big(\beta X_i^\top(y^\star_i+c_i)\big)\in\Delta^{K-1},\qquad y^\star_i=\sum_k w^\star_{i,k}X_{i,k}.$$
- **수렴값** $w^\star$가 코드(중간 step의 $w_t$는 궤적일 뿐).
- 고 β → one-hot(이산), 중 β → blend. `indices=argmax w*`(진단), `y*`=가중합(디코더 입력).

## 3. 확률 정식화
$$q(k_i\mid x)=\mathrm{Categorical}(w^\star_i),\qquad p(k_i)=\mathrm{Categorical}(\pi_i)\ \text{(factor별 학습 prior)}$$
- 8 factor **factorized**(독립). 3D Shapes는 GT factor가 독립·fully-crossed라 정확(생성 시 독립 샘플이 유효 조합).

## 4. ELBO (손실)
$$\mathcal{L}=\underbrace{\mathrm{MSE}(\hat x,x)}_{\text{decode}(y^\star)}+\beta_{\mathrm{KL}}\sum_i \mathrm{KL}\big(\mathrm{Cat}(w^\star_i)\,\|\,\mathrm{Cat}(\pi_i)\big)$$
- **recon = $y^\star$(posterior 평균) 디코딩** → 샘플링 gradient 불필요, **IFT로 정확 미분**(Gumbel/STE 안 씀 = 본 연구 정체성 유지).
- **KL closed-form**: $\sum_k w_k\log(w_k/\pi_k)$, 미분가능.

## 5. ★핵심 결정 — prior는 *학습*(uniform 금지)
**함정**: per-sample KL을 **uniform**으로 당기면 $w^\star$가 평평해져 **이산성 파괴**(β-어닐링의 one-hot sharpen과 정면 충돌) + posterior collapse.

**해법 — 학습 prior $\pi_i$**: 코드 *집계 분포*에 적응 → **per-sample posterior는 peaked(이산) 유지**하며 KL 충족. 그리고 $\pi_i$가 곧 **생성용 prior**. = "per-sample 날카롭게, marginal은 prior가 흡수" (discrete VAE의 올바른 형태).

**두 knob 조율**: Hopfield β(어닐링 1→8, 이산화) vs KL(prior로 당김) — 둘 다 $w^\star$ 날카로움에 작용 → **β_KL 작게 + 학습 prior + 워밍업**으로 균형(β-VAE식 튜닝).

## 6. 생성
```
factor별 k_i ~ Cat(softmax(π_i))  →  y_gen_i = X_i[:,k_i]  →  concat  →  decode → 새 이미지
```
- 학습 prior에서 코드 샘플 → 패턴 디코딩. 독립 factor라 factorized 샘플 유효.
- (옵션) factor 의존 있으면 작은 **autoregressive prior**로 교체(생성 품질↑).

## 7. 구현 변경
- `FactorizedHopfield.infer`: 고정점에서 **$w^\star$도 반환**(수렴 $y^\star$에서 softmax 재계산, 한 줄).
- `AQVAE`(model_attractor_vae 확장): factor별 prior 로짓 `pi=nn.Parameter(K)`, `forward`가 $w^\star$ 노출, `generate(n)` 메서드.
- `aq_vae_loss`: MSE + β_KL·Σ KL($w^\star$‖softmax(π)). β_KL 스케줄.
- `train.py`: KL 로깅, β_KL 워밍업(어닐링과 조율).

## 8. 평가 추가
- **ELBO 분해**(recon/KL), **생성 샘플 그리드**(prior에서), **코드 perplexity**(=exp H), 기존 **disentanglement(MIG/DCI/fMIG)** — KL이 disentanglement·collapse 개선하는지.

## 9. 리스크
- **posterior collapse**(β_KL↑ → 코드 무시) → 작은 β_KL + 워밍업.
- **이산성↔KL 텐션**(§5) → 학습 prior가 핵심 완화책.
- **β·β_KL 상호작용** → 어닐링 스케줄과 함께 튜닝.
- 생성 품질은 prior 표현력 의존(factorized vs autoregressive).

## 10. 단계
1. **$w^\star$ 노출 + 학습 prior + KL 항** (deterministic recon 유지).
2. β_KL 워밍업 학습 → recon 유지·KL 작동·perplexity↑ 확인.
3. **생성 샘플** + disentanglement 재측정.
4. (필요시) autoregressive prior로 생성 품질↑.

## 11. 미결 결정
1. prior: **학습 factorized**(추천) vs uniform vs autoregressive.
2. β_KL 스케줄(워밍업 길이·최대값), Hopfield β와의 조율.
3. recon: $y^\star$ 디코딩(추천, 무샘플) vs 샘플 디코딩(Gumbel/STE — 비추천).
4. 별도 `AQVAE` 클래스 vs 기존 `DAVAE`에 `--variational` 플래그.
