import torch
import time
from mini_transformer import MiniTransformer

torch.set_float32_matmul_precision("high")


def run_benchmark(model, x, warmup=20, iters=100):
    # warmup
    for _ in range(warmup):
        model(x)

    torch.cuda.synchronize()

    # benchmark
    start = time.time()
    for _ in range(iters):
        model(x)
    torch.cuda.synchronize()

    end = time.time()
    avg_ms = (end - start) * 1000 / iters
    return avg_ms


def benchmark_transformers():
    # IMPORTANT: dim must be >= 128 so head_dim >= 16 (Triton requirement)
    dim = 128
    hidden_dim = 256
    seq_len = 128
    batch = 1

    print("Building models...")

    custom_model = MiniTransformer(dim=dim, hidden_dim=hidden_dim).cuda()
    pytorch_model = MiniTransformer(dim=dim, hidden_dim=hidden_dim).cuda()

    x = torch.randn(batch, seq_len, dim, device="cuda", dtype=torch.float32)

    print("\nRunning Triton Attention version...")
    triton_ms = run_benchmark(custom_model, x)
    print(f"Triton model: {triton_ms:.3f} ms")

    print("\nRunning PyTorch baseline...")
    torch_ms = run_benchmark(pytorch_model, x)
    print(f"PyTorch model: {torch_ms:.3f} ms")

    print("\nSpeedup:", torch_ms / triton_ms)


if __name__ == "__main__":
    benchmark_transformers()