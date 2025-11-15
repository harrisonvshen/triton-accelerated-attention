import torch
import time
from attention_kernel import attention_fwd

device = "cuda"
torch.manual_seed(0)

B, H, N_CTX, D_HEAD = 1, 8, 512, 64
q = torch.randn(B, H, N_CTX, D_HEAD, device=device)
k = torch.randn_like(q)
v = torch.randn_like(q)
o = torch.empty_like(q)

configs = []
blocks_M = [16, 32, 64, 128]
blocks_N = [16, 32, 64, 128]

for BLOCK_M in blocks_M:
    for BLOCK_N in blocks_N:
        try:
            # flatten batch and heads into one dimension
            q_ = q.view(-1, N_CTX, D_HEAD)
            k_ = k.view(-1, N_CTX, D_HEAD)
            v_ = v.view(-1, N_CTX, D_HEAD)
            o_ = o.view(-1, N_CTX, D_HEAD)

            torch.cuda.synchronize()
            start = time.time()

            attention_fwd[(N_CTX // BLOCK_M,)](
                q_, k_, v_, o_,
                q_.stride(0), q_.stride(1), q_.stride(2), 0,
                k_.stride(0), k_.stride(1), k_.stride(2), 0,
                v_.stride(0), v_.stride(1), v_.stride(2), 0,
                o_.stride(0), o_.stride(1), o_.stride(2), 0,
                N_CTX, D_HEAD,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                num_warps=2
            )

            torch.cuda.synchronize()
            elapsed = (time.time() - start) * 1000
            print(f"BLOCK_M={BLOCK_M:3}, BLOCK_N={BLOCK_N:3} -> {elapsed:.3f} ms")
            configs.append((BLOCK_M, BLOCK_N, elapsed))

        except Exception as e:
            print(f"BLOCK_M={BLOCK_M:3}, BLOCK_N={BLOCK_N:3} -> failed ({e})")

if configs:
    best = min(configs, key=lambda x: x[2])
    print(f"\nBest config: BLOCK_M={best[0]}, BLOCK_N={best[1]} ({best[2]:.3f} ms)")
else:
    print("\nNo valid configuration succeeded.")