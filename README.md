# AQ — Attractor Quantization

**Attractor 동역학 + implicit differentiation으로 학습하는, 완전 미분가능한 이산 표현.**
VQ-VAE의 argmin+straight-through(STE)를 대체한다: 이산화를 *attractor의 고정점*으로 정의하고,
그 고정점을 IFT로 *정확히* 미분한다 (STE의 편향 없음, Gumbel의 노이즈 완화 없음).

## 핵심 아이디어
- 잠재를 factor별(예: 8 factor × 4dim)로 나누고, 각 factor에 **Modern Hopfield attractor**(학습된 저장 패턴 = 이산 개념)를 둔다.
- 입력 context $c$가 동역학으로 **고정점 $y^\star=\sum_k w^\star_k X_k$** 까지 수렴 → 디코딩·재구성 → **전체 backprop**.
- $\beta$(stiffness) 어닐링으로 soft→discrete. 디코더엔 argmax가 아니라 $y^\star$ 자체를 넣어 **미분 가능**.
- 변분 확장: softmax 코드 $w^\star$를 categorical posterior로(또는 context에 Gaussian 노이즈) → 생성 모델.

자세한 동기·설계는 [`docs/0617_proposal.md`](docs/0617_proposal.md), 문서 전체는 [`docs/README.md`](docs/README.md).

## 구조
```
src/
  dataset.py, dataset_shapes3d.py   데이터 (dSprites, 3D Shapes; uint8 GPU 변환)
  model.py(VAE) model_vqvae.py(VQ-VAE) model_attractor_vae.py(AQ + 변분)
  train.py        --arch {vae,vqvae,attractor}, β 어닐링, DDP, --compile, --variational
  utils.py        의존성 없는 PNG 저장
  fixed_point/    solvers·implicit(IFT)·multistable(Hopfield) — 재사용 부품
  eval/           interpret·disentangle·attractor_diag·attractor_viz·generate
  experiments/    일회용 스모크(collapse 조사 재현)
  tests/          _verify*(solver·IFT·Hopfield 검증)
docs/             설계·결과·분석·신경과학 노트 (docs/README.md 인덱스)
```

## 빠른 시작
```bash
pip install -r requirements.txt   # torch, numpy, h5py, matplotlib

# deterministic AQ (3D Shapes, 무손실 속도설정)
python -m src.train --arch attractor --dataset shapes3d --latent-dim 32 \
  --batch-size 256 --beta-start 1.0 --beta-end 8.0 --anneal-epochs 40 \
  --solver-max-iter 20 --compile

# 변분 (Categorical AQ-VAE)
python -m src.train --arch attractor ... --variational --beta-kl 0.05

# 4-GPU DDP
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m src.train --arch attractor ...

# 평가
python -m src.eval.disentangle --ckpts <ckpt1> <ckpt2> ...
python -m src.eval.interpret --ckpt <ckpt>        # NMI·purity·traversal
python -m src.eval.generate  --ckpt <변분 ckpt>   # 생성 샘플
```

## 핵심 결과 (3D Shapes, 수렴)
| | recon↓ | fMIG↑ | DCI-D↑ | DCI-C↑ |
|---|---|---|---|---|
| VAE | 28.5 | — | 0.420 | 0.276 |
| VQ-VAE | **5.7** | 0.289 | 0.584 | 0.293 |
| **AQ (deterministic)** | 7.9 | 0.299 | **0.617** | **0.302** |
| AQ K=32 | 9.0 | **0.331** | 0.568 | 0.273 |
| AQ 변분(생성가능) | 8.1–8.8 | 0.13–0.15 | 0.44–0.53 | — |

- **AQ는 STE 없이 미분가능 이산 표현을 학습**, disentanglement(DCI)에서 **VQ-VAE 추월**, 해석가능 코드 + 검증된 attractor basin.
- 변분은 **생성**을 얻으나 disentanglement는 희생(흐릿) → autoregressive/Gaussian prior가 후속 과제.
- 전체 표·분석: [`docs/0618_results.md`](docs/0618_results.md).

## 신경과학 연결
지각으로부터 unsupervised 이산·조합 개념 형성을 attractor 동역학으로 — associative memory(Hopfield),
범주지각, equilibrium propagation(국소 학습)과 연결. [`docs/0618_neuroscience_significance.md`](docs/0618_neuroscience_significance.md),
[`docs/0618_local_learning.md`](docs/0618_local_learning.md).
