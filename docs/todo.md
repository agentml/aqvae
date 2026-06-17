# TODO — 모델·평가 개선 추적

> 대화 중 나온 "시도해볼 만한 것"을 누적·추적. `[ ]` 미착수 / `[~]` 진행중 / `[x]` 완료.
> 우선순위 P1(높음)~P3(낮음). "출처"=아이디어가 나온 맥락.

---

## A. 모델 개선
- [ ] **P1 · scale factor 분리 실패 해결.** scale이 어느 factor에도 안 잡힘(shape와 entangle). 원인 분석 + 분리 압력/구조 조정. (출처: Phase F 의미분석)
- [ ] **P1 · recon 격차 축소** (Attractor 7.9 vs VQ-VAE 5.7). K↑, factor_dim↑, 디코더 용량, β_end 등으로 이산 병목 완화. (출처: Phase F)
- [ ] **P2 · codebook collapse(dead attractor) 완화.** 활성 5~13/16 — 미사용 패턴 존재. usage-entropy 정규화 / dead-pattern reinit / EMA codebook. (출처: collapse_analysis §8, viz 활성도)
- [ ] **P2 · disentanglement 압력 추가.** 현재 순수 AE라 분리 압력 없음 → factor 중복(wall/shape). β-VAE식 TC penalty 또는 factor 독립 정규화. (출처: Phase F, "순수 AE라 분리 압력 없음")
- [ ] **P2 · soft target magnitude 정규화** (KL식이되 0이 아닌 target s로). 하드 context_norm 대신 학습이 스케일 자율 탐색. (출처: collapse_analysis §5.5, KL 논의)
- [ ] **P3 · variational 확장.** Nam식 분포/생성 — 이산코드 prior, stochastic attractor. (출처: 설계 결정 (ii) 보류)
- [ ] **P3 · 패턴 분리 정규화.** 패턴 간 4D 거리 최대화(codebook이 안 겹치게) → viz·retrieval 명료. (출처: viz "패턴 실제 근접")

## B. 평가·시각화 개선
- [ ] **P2 · 궤적 viz 개선.** (a) 특정 factor 단독 확대, (b) step별 색농도(연→진) 방향성, (c) 활성 많은 factor는 2D 한계 → 3D/pairwise 투영 또는 endpoint→배정별 connector. (출처: 궤적 viz, "별 거리/색 혼동")
- [ ] **P2 · per-attractor prototype 이미지.** (factor,attractor)로 수렴한 이미지 평균 → 각 attractor의 시각 의미. (출처: 의미분석 설계, 미구현)
- [ ] **P3 · DCI 정식 regressor.** sklearn 설치 시 GBT importance로 DCI(현재 MI 기반 D/C). Informativeness 추가. (출처: disentangle.py, numpy-only 제약)
- [ ] **P3 · β별 latent traversal.** 추론 β 바꿔가며 traversal — 이산화 정도 시각 비교. (출처: dist→pat vs β 스윕)
- [ ] **P3 · MIG 보고 방식.** per-dim MIG는 4dim/factor에 불공정 → fMIG 병기·설명 표준화. (출처: Phase F)

## C. 실험·스윕 (Phase E)
- [ ] **P2 · β 어닐링 스케줄 스윕.** β_end·warmup·속도 체계적 탐색 (이산성↔다양성↔recon trade-off). (출처: D3, collapse_analysis §8)
- [ ] **P2 · K(패턴 수)·factor_dim 스윕.** K=16 고정이었음. cardinality(scale 8 등)에 맞춰. (출처: 설계, Phase F)
- [ ] **P3 · context_scale·pattern norm 스윕.** effective temp 구성요소 개별 영향. (출처: collapse 분석)

## D. 인프라
- [ ] **P3 · fast solver(fpi max20) 기본값화.** 2배 빠르고 동등 검증됨. AttractorVAE 기본 solver_max_iter=20 검토. (출처: solver 벤치마크)
- [ ] **P3 · 추론 시 적은 solver step(~4) 활용.** internalization 진단상 4 step이면 충분. eval/배포 시 step 축소. (출처: attractor_diag)

---

## 완료 (Done)
- [x] Phase A~F 전 파이프라인 (데이터·베이스라인·solver+IFT·Hopfield·모델·학습·평가).
- [x] collapse 진단·해결 (train soft / anneal to discrete). → `collapse_analysis.md`
- [x] 본 학습 수렴 (loss 8.13) + solver 개선(max20 2배 빠름).
- [x] Phase F 평가 3종 (의미분석·disentanglement·attractor 진단) on 수렴 모델.
- [x] attractor 수렴 궤적 시각화 (matplotlib, PCA 2D, β·used-PCA 옵션).
- [x] 문서: proposal·design·collapse_analysis·results·project_log·conversation_log.

---

## 다음 추천 (지금 시점)
1. **scale 분리 실패 원인 분석** (P1) — Phase F의 가장 명확한 약점.
2. **recon 격차 축소** (P1) — VQ 대비 정량 우위 확보.
3. **β/K 스윕** (P2) — 현 설정이 최적인지 확인.
