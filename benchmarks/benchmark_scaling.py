import torch, time
from attention_kernel import attention_fwd
import triton

@torch.no_grad()
def benchmark_attention(N_CTX):
    D_HEAD = 64
    q = torch.randn((1, 1, N_CTX, D_HEAD), device='cuda')
    k = torch.randn((1, 1, N_CTX, D_HEAD), device='cuda')
    v = torch.randn((1, 1, N_CTX, D_HEAD), device='cuda')
    o = torch.zeros_like(q)

    grid = lambda meta: (triton.cdiv(N_CTX, meta['BLOCK_M']),)
    torch.cuda.synchronize()

    # warm-up to fill cache
    for _ in range(10):
        attention_fwd[grid](
            q, k, v, o, N_CTX, D_HEAD,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            BLOCK_M=32, BLOCK_N=32
        )
    torch.cuda.synchronize()

    # Triton timing (CUDA events)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(200):
        attention_fwd[grid](
            q, k, v, o, N_CTX, D_HEAD,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            BLOCK_M=32, BLOCK_N=32
        )
    end_event.record()
    end_event.synchronize()
    triton_ms = start_event.elapsed_time(end_event) / 200

    # PyTorch timing (same conditions)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(200):
        _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    end_event.record()
    end_event.synchronize()
    torch_ms = start_event.elapsed_time(end_event) / 200

    return triton_ms, torch_ms


def main():
    for N in [256, 512, 1024, 2048, 4096]:
        t_triton, t_torch = benchmark_attention(N)
        print(f"N={N:<5} | Triton: {t_triton:.3f} ms | PyTorch: {t_torch:.3f} ms")


if __name__ == "__main__":
    main()
