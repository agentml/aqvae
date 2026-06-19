# attractor_vae 프로젝트 전체 작업 로그

> 프로젝트 시작부터 Phase F 완주까지의 전 과정 기록(시간순). 의사결정·실험수치·막다른길·해결을 모두 포함.
> 관련 상세 문서: `docs/0617_proposal.md`(논문형 제안), `docs/0617_design.md`(실무설계), `docs/0617_collapse_analysis.md`(collapse 심층분석).
> 환경: 3D Shapes(RGB 64×64×3, 480k), conda env `gram`, torch 2.10 cu128, 4× RTX 4090, numpy-only(sklearn·torchvision·PIL 없음).

---

## 0. 작업 제약 (사용자 규칙)
- **응답은 한국어로.**
- **코딩·학습 실행 전 반드시 사용자의 명확한 허락을 받는다.** (읽기·분석·설계·질문은 자유, Write/Edit·본 학습은 허락 후.)

---

## 1. 프로젝트 생성 & dSprites VAE 베이스라인
- `/home/sjn/00_code/attractor_vae/` 생성.
- 첫 요청: dSprites용 VAE. PyTorch, conv (β-)VAE(Higgins 2017 표준), 자동 다운로드 로더, 수렴(early stopping) 학습 작성.
  - `src/dataset_dsprites.py`, `src/model_vae.py`(VAE+vae_loss), `src/train.py`, `src/utils.py`(의존성 없는 PNG 저장 — torchvision/PIL 없어 stdlib zlib로 직접 인코딩).
  - dSprites 파일명 오류(`co1sc1...`→`co1sh3sc6or40...`) 수정.
- vanilla VAE(β=1) 학습 → **수렴 loss 33.97**(recon 12.3, KL 21.6), epoch 131 early stop. 파이프라인 sanity 확인.

## 2. 두 논문 학습 노트 (`related_paper/`)
- **Solve the Loop — Attractor Models** (Fein-Ashley & Rashidinejad 2026, USC, arXiv:2605.12466):
  looped Transformer를 유한 unroll 말고 **고정점을 직접 solve**. backbone이 초기추측 $\tilde y_0$ 제안 → 작은 weight-tied attractor가 **persistent injection**으로 고정점까지 refine → **IFT(implicit diff)로 메모리 O(1)** 학습. one-step Neumann 근사, 수렴 기준 종료, equilibrium internalization.
- **Discrete, compositional, symbolic representations through attractor dynamics** (Nam et al. 2024, arXiv:2310.01807):
  VQ/Gumbel처럼 이산화를 박지 말고 **stochastic attractor basin**으로 자생. attractor=discrete code. GFlowNet-EM 학습. dSprites+color 벤치마크, VAE pre-train 위에.
- 두 논문 교차점(노트 §8.4): "결정론적 고정점(Solve-the-Loop) + 분포(Nam)" → 본 프로젝트의 씨앗.

## 3. Solve-the-Loop 소스 분석 (`related_paper/Attractor/`, MIT)
- 클론 후 핵심 부품 식별:
  - `solvers.py`: `anderson_solve`/`fpi_solve` (범용, 시퀀스 가정 없음).
  - `attractor.py::_IFTAttach`: IFT autograd. `f(y,c)=Head(y+c)-y`, one-step `u=v+Jᵀv`. 수축 게이트(LayerScale), `adjoint_clip`.

## 4. 핵심 아이디어 정립
- 사용자 정의: **목표 = discrete representation을 완전 미분가능하게 학습 (VQ-VAE/STE 대체).**
- 메커니즘: **attractor = discrete concept** (shape factor → 삼각형/네모 attractor). 수렴한 고정점 $y^\star$를 디코딩·재구성, IFT로 전체 backprop.
- 결정적 통찰(나중에 C2에서 검증): **디코더엔 $y^\star$ 자체를 넣음**(argmin snap 아님) → STE 불필요.

## 5. 설계 결정 (사용자 주도)
- **잠재 구조**: z=32 = **8 factor × 4 dim**, factor별 독립 attractor 동역학. (2D는 cardinality 15에 too compact → 4D.)
- **데이터셋**: dSprites → **3D Shapes** 전환 (factor 6개, cardinality 10·10·10·8·4·15; 저-cardinality라 4D basin에 적합; 색이 정식 factor). RGB라 손실 Bernoulli→MSE, h5py 의존.
- **시각화**: factor 4D → PCA 2D.
- **multi-stability 메커니즘**: **Modern Hopfield** 확정. $T(y)=X\,\mathrm{softmax}(\beta X^\top(y+c))$, β=stiffness 노브.

## 6. 단계별 계획 (Phase A~F)
de-risk 순서로: A(데이터·베이스라인) → B(solver+IFT 단독검증) → C(Hopfield 단독검증) → D(모델조립·학습) → E(스윕) → F(평가).

---

## Phase A — 데이터 & 베이스라인
- **A1** `src/dataset_shapes3d.py`: 3D Shapes 로더. 처음 h5py lazy-open → **HDF5 파일락 데드락**(워커들이 같은 .h5 열다 futex 멈춤). → **전체 RAM 적재**(uint8 5.5GB, COW fork-safe)로 수정. 480k·6 factor 확인.
- **A2** `model_vae.py` 일반화(하위호환): `in_channels`, `vae_loss(recon_type)` 추가.
- **A3/A4** 베이스라인 동시 학습(GPU0/1):
  - **VAE(z=32)**: 수렴 **loss 47.31**(recon 29.5).
  - **VQ-VAE(z=32, 8 group × K=16 product 양자화 + STE)** `model_vqvae.py`: 초기 vq항 폭발(11654) 자기보정 → 수렴 **recon 5.7**(KL 없어 또렷).

## Phase B — 고정점 머신 (`src/fixed_point/`)
- `solvers.py`(이식), `implicit.py`(`attach_ift_grad`, `solve_fixed_point`, one-step/iterative/anderson backward — 단일 GPU 단순화).
- **B 검증**(`_verify.py`): solver가 $T(y)=0.5y+b$의 $y^\star=2b$ 정확; **IFT gradient가 닫힌형·unroll과 1e-14 일치**(정확성 게이트 통과). one-step은 의도된 근사.

## Phase C — Modern Hopfield (`src/fixed_point/multistable.py`)
- **C1** 단독 거동: K=16 패턴 고정, β 스윕. **저 β=1 attractor(병합) → 고 β=K개 분리**(distinct 1→16), 고 β서 dist→pat→0, 안정점 specRad<1. basin 맵 시각화로 multi-stability 확증. (패턴노름 R=3 필요 — retrieval 조건 $\beta\|X\|^2(1-\cos)\gg1$.)
- **C2** solver+IFT 결합: IFT 1e-14 일치. **중요 발견 — persistent injection 필수**: 순수 retrieval(query-as-init)은 $\partial y^\star/\partial c\approx0$(STE 문제 재현), injection은 nonzero gradient. 그리고 **discreteness↔gradient trade-off** 정량화(저β grad↑·이산↓, 고β 이산↑·grad→0=softmax 포화).
- **C3** factorized head: 8×4D 독립. **fpi가 factor 독립 보장**(anderson은 α 전역결합). end-to-end 미분 확인.

## Phase D — 모델 조립 & collapse saga (`src/model_davae.py`)
- **확률화 위치 결정**: (i) **deterministic AE** 확정(VQ-VAE처럼 비변분, KL 없음, $y^\star$ 직접 디코딩) vs (ii) variational(미룸).
- **D1** 모델 작동(미분·형상 OK)하나 **즉시 collapse**: factor당 활성 attractor 1개(8/128), recon~1500 정체, dist→pat=0.000.

### collapse 분석 (상세: `docs/0617_collapse_analysis.md`)
- **가설1 β 어닐링 → 기각**: 세 스케줄 모두 8/128. 단서 — **$c\_std$ 폭발**(74→542): 인코더가 context 크기로 β 통제 무력화.
- **가설2 context 정규화 → 부분 기각**: $\|c\|$ 묶어도(1.47) collapse 유지. magnitude만 원인 아님.
- **근본 원인 = softmax 포화**: effective temperature $\sim\beta\|X\|\|c\|$ 크면 one-hot → **exact IFT gradient가 진짜 ≈0**. → "정확한 gradient"가 이산 극한에서 **편향-nonzero STE보다 불리**. β·‖c‖ 상호대체라 β 단독 통제 무의미.
- (사용자 통찰) collapse ≈ VAE의 KL(잠재 magnitude 규제) 부재. 단 KL은 →0으로 당겨 *반대* collapse 유발 → nonzero 고정 스케일 필요.
- **해결 = train soft, anneal to discrete**: 작은 패턴·c·β(비포화 soft)면 학습됨(recon 1500→333, 활성 8→36).

### D3 — 어닐링 (`_smoke_d3.py`)
- `unit_patterns`(패턴 단위노름) + `context_norm` → effective temp=β 단일화. β 1→8 어닐링(앞 30% warmup).
- **재collapse 없음**: dist→pat 0.61→0.10, 활성 65→90, recon 403→182 *동시* 개선. sweet spot β_end=8.

### D4/D5 — 통합·학습 (`train.py`)
- `--arch attractor` + β 어닐링 스케줄 + 진단 로깅(β·active·dist→pat) + 어닐링 중 early-stop 비활성.
- **D5 본 학습**(β1→8, anneal 40ep): GPU0에서 recon 292→8.

### solver 개선
- 사용자 질문 계기로 벤치마크: 현재 이미 fpi 사용. **anderson은 block-diagonal에 부적합(느리고 미수렴)**. **fpi max_iter=20**이 max64 대비 ramp에서 ~2배 빠르고 동등 품질(one-step IFT라 forward 근사 견딤). GPU1에 새 run.
- **두 run 모두 수렴**: max64 loss 8.165 @e146, **max20 loss 8.125 @e149**(빠르고 약간 더 좋음).

## Phase F — 평가 (`src/eval/`)
### 의미분석 (`interpret.py`)
- (1) NMI 행렬(8 모델팩터×6 GT), (2) attractor purity, (3) swap traversal PNG.
- 수렴 모델: floor_hue→f4(0.79)·wall_hue→f5(0.77)·orientation→f1(0.62) 깨끗, **shape→f2/f7 purity 0.65, 4/4값**(traversal에서 큐브/원기둥/캡슐/구 변화 시각 확인). **scale은 분리 실패**(shape와 entangle), 약한 중복.

### Disentanglement (`disentangle.py`, numpy 구현)
- MIG·factor-level MIG·DCI(MI기반 D/C)·FactorVAE.
- per-dim MIG는 VAE 유리(7/32 활성)·우리 4dim/factor 설계에 불공정 → **fMIG** 추가.

| (수렴) | recon | fMIG | DCI-D | DCI-C | FactorVAE |
|---|---|---|---|---|---|
| VAE | 29.5 | — | 0.420 | 0.276 | 0.883 |
| VQ-VAE | **5.7** | 0.289 | 0.584 | 0.293 | **0.994** |
| **Attractor** | 7.9 | **0.307** | **0.617** | **0.302** | 0.947 |

→ Attractor가 **fMIG·DCI-D·DCI-C 최고**(VQ 추월), FactorVAE 근소 2위.

### Attractor 진단 (`attractor_diag.py`)
- **perturbation-복귀**: ν≤1.0 복귀, ν=2.0 이웃 basin 이탈 → 진짜 basin 확증.
- **internalization**: head contribution 0.477; solver 0 step=망함(280), **4 step이면 recon 8.0≈수렴(7.9)** → solver 필수(이산성의 근원)지만 추론 적은 step 충분.

---

## 최종 결론
- **핵심 기여 실증**: STE 없이 **완전 미분가능 이산 표현**(attractor 동역학 + IFT)을 학습. 코드가 **해석가능**, **진짜 basin**, 추론 4 step.
- **disentanglement 경쟁력**: 수렴 후 DCI-D·DCI-C·fMIG 최고(VQ-VAE 추월), FactorVAE 2위. recon은 VAE≫Attractor>VQ(7.9 vs 5.7).
- **남은 한계**: recon 소폭 열위, **scale factor 분리 실패**, shape/hue 약한 중복.
- e63 미수렴 때 "우위 없음"이 수렴 후 "대부분 지표 최고"로 뒤집힘.

## 산출물
```
src/
  dataset.py, dataset_shapes3d.py, utils.py
  model_vae.py(VAE), model_vqvae.py(VQ-VAE), model_davae.py(DAVAE)
  train.py(--arch vae/vqvae/attractor, β 어닐링)
  fixed_point/{solvers,implicit,multistable}.py + _verify*.py
  eval/{interpret,disentangle,attractor_diag}.py
docs/{proposal,design,collapse_analysis,project_log}.md
related_paper/{solve_the_loop, Discrete...}.md, Attractor/(소스)
outputs/{vae,vqvae,attractor}_shapes3d_z32*/  (체크포인트·recon·history)
outputs/{c1_basins, interpret}/  (시각화)
```

## 미해결/다음 후보
- scale factor 분리 개선, recon 격차 축소(K·차원·구조), β/K/스케줄 체계적 스윕(Phase E),
  variational 확장(Nam식 분포·생성), `docs/0618_results.md` 결과 정리.
