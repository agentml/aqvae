# attractor_vae

dSprites 데이터셋에 적용하는 (β-)VAE.

## 구성
- `src/dataset.py` — dSprites 로더(없으면 자동 다운로드, mmap).
- `src/model.py` — conv (β-)VAE. Bernoulli 디코더, KL은 표준정규 대비. `vae_loss(beta=...)`.
- `src/train.py` — 단일 GPU 학습. 에폭마다 재구성 그리드(`recon_eXXX.png`) + 체크포인트(`ckpt.pt`) 저장.

## 데이터셋
dSprites: 64×64 이진 이미지 737,280장, 생성 인자 6개
(color·shape·scale·orientation·posX·posY). 첫 실행 시 `data/`에 npz 자동 다운로드(~28MB).

## 학습
수렴(early stopping)까지 학습한다. epoch 평균 loss가 `--patience` 에폭 동안
`--min-delta` 이상 개선되지 않으면 종료. `--max-epochs`는 안전 상한.
```bash
# vanilla VAE (수렴까지)
python -m src.train --beta 1.0 --latent-dim 10

# beta-VAE (disentanglement)
python -m src.train --beta 4.0 --latent-dim 10
```
주요 인자: `--patience`(10), `--min-delta`(0.1), `--max-epochs`(200),
`--batch-size`(256), `--lr`(1e-3), `--hidden`(256), `--num-workers`(8).
결과는 `outputs/<run-name>/` 에 저장 — `ckpt_best.pt`/`ckpt_last.pt`,
에폭별 재구성 `recon_eXXX.png`(위:원본/아래:재구성, 의존성 없는 PNG), `history.json`.

## 모델 개요
- 인코더: Conv ×4 (stride 2, 64→32→16→8→4) → FC → (mu, logvar).
- 디코더: 대칭 ConvTranspose ×4 → 64×64 logits.
- 손실: BCE(픽셀 합, 배치 평균) + β·KL.
