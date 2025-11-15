import torch
import torch.nn.functional as F

def simple_attention(q, k, v):
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / q.size(-1)**0.5
    attn_weights = F.softmax(attn_scores, dim=-1)
    output = torch.matmul(attn_weights, v)
    return output

def main():
    torch.manual_seed(0)
    q = torch.randn(1, 4, 64, device="cuda")
    k = torch.randn(1, 4, 64, device="cuda")
    v = torch.randn(1, 4, 64, device="cuda")
    out = simple_attention(q, k, v)
    print("Output shape:", out.shape)

if __name__ == "__main__":
    main()

