import torch
import torch.nn as nn

from transformer_network import Attention as ModernAttention
from transformer_network import Transformer as ModernTransformer
# from transformer_network_original import Attention as LegacyAttention
# from transformer_network_original import Transformer as LegacyTransformer
from gqa import GroupedQueryAttention as GQAAttention

# Comment out the tests that are matching our original implementation

# def test_attention_mha_matches_legacy():
#     torch.manual_seed(0)
#     batch, seq_len, dim, heads = 1, 2, 8, 4
#     legacy = LegacyAttention(dim, heads, dp_rate=0.0)
#     modern = ModernAttention(dim, heads, dp_rate=0.0, kv_heads=heads)
#     modern.load_state_dict(legacy.state_dict())
#     legacy.eval()
#     modern.eval()
#     x = torch.randn(batch, seq_len, dim)

#     legacy_out = legacy(x)
#     modern_out = modern(x)
#     assert torch.allclose(legacy_out, modern_out, atol=1e-6)


# def test_transformer_mha_matches_legacy():
#     torch.manual_seed(1)
#     batch, seq_len, dim, heads = 2, 3, 32, 4
#     legacy = LegacyTransformer(dim, ff_hid_dim=64, ff_dp_rate=0.0, n_heads=heads, attn_dp_rate=0.0)
#     modern = ModernTransformer(
#         dim,
#         ff_hid_dim=64,
#         ff_dp_rate=0.0,
#         n_heads=heads,
#         attn_dp_rate=0.0,
#         kv_heads=heads,
#     )
#     modern.load_state_dict(legacy.state_dict())
#     legacy.eval()
#     modern.eval()
#     x = torch.randn(batch, seq_len, dim)

#     legacy_out = legacy(x)
#     modern_out = modern(x)
#     assert torch.allclose(legacy_out, modern_out, atol=1e-6)

def test_attention_mqa_reduces_to_single_kv_head():
    torch.manual_seed(2)
    batch, seq_len, dim, heads = 2, 4, 64, 4
    modern = ModernAttention(dim, heads, dp_rate=0.0, kv_heads=1)
    x = torch.randn(batch, seq_len, dim)
    out = modern(x)
    assert out.shape == (batch, seq_len, dim)

def test_attention_gqa_matches_grouped_behavior():
    torch.manual_seed(3)
    batch, seq_len, dim, heads = 2, 4, 64, 4
    group_size = 2
    kv_heads = heads // group_size
    modern = ModernAttention(dim, heads, dp_rate=0.0, kv_heads=kv_heads)
    gt = GQAAttention(heads, dim, kv_heads)
    gt.load_state_dict(modern.state_dict())
    x = torch.randn(batch, seq_len, dim)
    modern_out = modern(x)
    gt_out = gt(x)

    assert torch.allclose(modern_out, gt_out, atol=1e-5)

def test_attention_mqa_matches_grouped_behavior():
    torch.manual_seed(3)
    batch, seq_len, dim, heads = 2, 4, 64, 4
    group_size = heads
    kv_heads = heads // group_size
    modern = ModernAttention(dim, heads, dp_rate=0.0, kv_heads=kv_heads)
    gt = GQAAttention(heads, dim, kv_heads)
    gt.load_state_dict(modern.state_dict())
    x = torch.randn(batch, seq_len, dim)
    modern_out = modern(x)
    gt_out = gt(x)

    assert torch.allclose(modern_out, gt_out, atol=1e-5)

def test_attention_rope_matches_no_rope_when_projection_zero():
    torch.manual_seed(4)
    batch, seq_len, dim, heads = 2, 3, 32, 4
    x = torch.randn(batch, seq_len, dim)
    pos = torch.randn(batch, seq_len, dim)

    base = ModernAttention(dim, heads, dp_rate=0.0, kv_heads=heads)
    rope = ModernAttention(
        dim,
        heads,
        dp_rate=0.0,
        attn_mode="qk",
        pos_dim=dim,
        kv_heads=heads,
        use_rope=True,
    )
    rope.q_fc.load_state_dict(base.q_fc.state_dict())
    rope.k_fc.load_state_dict(base.k_fc.state_dict())
    rope.v_fc.load_state_dict(base.v_fc.state_dict())
    rope.out_fc.load_state_dict(base.out_fc.state_dict())
    rope.dp.p = base.dp.p
    rope.rope_proj.weight.data.zero_()

    out_base = base(x)
    out_rope = rope(x, pos)
    assert torch.allclose(out_base, out_rope, atol=1e-6)


def test_rope_detailed():
    torch.manual_seed(5)
    batch, seq_len, dim, heads = 1, 5, 64, 4
    attn = ModernAttention(
        dim,
        heads,
        dp_rate=0.0,
        attn_mode="qk",
        pos_dim=dim,
        kv_heads=heads,
        use_rope=True,
    )
    x = torch.randn(batch, seq_len, dim)
    pos = torch.randn(batch, seq_len, dim)

    print("Testing RoPE implementation...")
    out = attn(x, pos)
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {out.shape}")
    assert out.shape == (batch, seq_len, dim)

    pos_rev = pos.flip(1)
    out_rev = attn(x, pos_rev)
    diff = torch.abs(out - out_rev).mean()
    print(f"✓ Position sensitivity test - mean difference: {diff.item():.6f}")
    assert diff > 1e-6, "RoPE should be sensitive to positional changes"
    print("✓ RoPE is position-sensitive (good!)")

    out_same = attn(x, pos)
    same_diff = torch.abs(out - out_same).mean()
    print(f"✓ Same position consistency - mean difference: {same_diff.item():.6f}")
    assert same_diff < 1e-6, "Identical positions should yield identical outputs"
    print("✓ Same positions produce same output (good!)")

    print(f"✓ Output range: [{out.min().item():.3f}, {out.max().item():.3f}]")
    print("All RoPE tests completed!")

def test_rope_cache_and_clear():
    torch.manual_seed(6)
    batch, seq_len, dim, heads = 1, 3, 32, 4
    attn = ModernAttention(
        dim,
        heads,
        dp_rate=0.0,
        attn_mode="qk",
        pos_dim=dim,
        kv_heads=heads,
        use_rope=True,
    )

    class CountingLinear(nn.Linear):
        def __init__(self, in_features, out_features, bias=False):
            super().__init__(in_features, out_features, bias=bias)
            self.calls = 0

        def forward(self, input):
            self.calls += 1
            return super().forward(input)

    counting_proj = CountingLinear(dim, attn.head_dim, bias=False)
    counting_proj.load_state_dict(attn.rope_proj.state_dict())
    attn.rope_proj = counting_proj

    pos = torch.randn(batch, seq_len, dim)
    attn.clear_cache()
    attn._compute_rope_angles(pos)
    assert attn.rope_proj.calls == 1, "RoPE projection should run on cache miss"
    attn._compute_rope_angles(pos)
    assert attn.rope_proj.calls == 1, "RoPE cache should avoid redundant projections"
    attn.clear_cache()
    attn._compute_rope_angles(pos)
    assert attn.rope_proj.calls == 2, "Clearing cache should force recomputation"

# test_attention_mha_matches_legacy()
# test_transformer_mha_matches_legacy()
test_attention_mqa_reduces_to_single_kv_head()
test_attention_gqa_matches_grouped_behavior()
test_attention_mqa_matches_grouped_behavior()
test_attention_rope_matches_no_rope_when_projection_zero()
test_rope_detailed()
test_rope_cache_and_clear()
print("All tests passed")
