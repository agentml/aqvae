"""DDP 호환 스모크 (일회용): IFT autograd Function 이 DDP all-reduce 와 맞물리나.

실행: CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 -m src._smoke_ddp
검증: (1) DDP 래핑·forward·backward 에러 없음, (2) loss 감소,
      (3) head/encoder gradient 가 rank 간 동기화(=DDP all-reduce 작동).
"""
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from ..model_attractor_vae import AttractorVAE, attractor_loss


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local = int(os.environ["LOCAL_RANK"])
    world = dist.get_world_size()
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    torch.manual_seed(0)

    m = AttractorVAE(latent_dim=32, in_channels=3, n_patterns=16, context_norm=True,
                     context_scale=1.0, pattern_scale=1.0, unit_patterns=True,
                     solver="fpi", solver_max_iter=20).to(dev)
    ddp = DDP(m, device_ids=[local], find_unused_parameters=True)
    opt = torch.optim.Adam(ddp.parameters(), 1e-3)

    g = torch.Generator(device=dev).manual_seed(100 + rank)   # rank마다 다른 데이터
    for step in range(6):
        x = torch.rand(64, 3, 64, 64, device=dev, generator=g)
        out = ddp(x, beta=8.0)
        loss, recon, _ = attractor_loss(out, x)
        opt.zero_grad(); loss.backward(); opt.step()
        if rank == 0:
            print(f"  step {step} loss={loss.item():.2f}", flush=True)

    # grad 동기화 검증: backward 후 head·encoder grad 가 rank 간 동일해야(DDP 평균)
    x = torch.rand(64, 3, 64, 64, device=dev, generator=g)
    out = ddp(x, beta=8.0); loss, _, _ = attractor_loss(out, x)
    opt.zero_grad(); loss.backward()

    def synced(p, name):
        gloc = p.grad.detach().clone()
        buf = [torch.zeros_like(gloc) for _ in range(world)]
        dist.all_gather(buf, gloc)
        ok = all(torch.allclose(buf[0], b, atol=1e-6) for b in buf)
        if rank == 0:
            print(f"  [{name}] rank간 grad 동일(=DDP 동기화): {ok}  |grad|={gloc.norm():.4f}", flush=True)
        return ok

    ok_head = synced(m.head.factors[0].patterns, "head.patterns")
    enc_w = next(p for p in m.encoder.parameters() if p.grad is not None)
    ok_enc = synced(enc_w, "encoder")
    if rank == 0:
        print(f"\nDDP {'PASS ✓' if (ok_head and ok_enc) else 'FAIL ✗'} "
              f"(head_sync={ok_head}, enc_sync={ok_enc})", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
