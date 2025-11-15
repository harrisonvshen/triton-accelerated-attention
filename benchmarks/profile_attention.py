import torch
from torch.profiler import profile, ProfilerActivity
from mini_transformer import MiniTransformer

torch.set_float32_matmul_precision("high")

dim = 128
hidden_dim = 256
seq_len = 128
batch = 1

model = MiniTransformer(dim=dim, hidden_dim=hidden_dim).cuda()
x = torch.randn(batch, seq_len, dim, device="cuda")

# Warmup
for _ in range(10):
    _ = model(x)
torch.cuda.synchronize()

# Full profiler trace
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=False,
) as prof:
    for _ in range(30):
        _ = model(x)

prof.export_chrome_trace("trace_fixed.json")
print("Saved trace_fixed.json")