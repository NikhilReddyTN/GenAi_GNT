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

def test_attention_rope_matches_no_rope_when_positions_zero():
    torch.manual_seed(4)
    batch, seq_len, dim, heads = 2, 3, 32, 4
    x = torch.randn(batch, seq_len, dim)

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
    out_base = base(x)
    zero_pos = torch.zeros(seq_len, dtype=torch.long, device=x.device)
    out_rope = rope(x, rope_positions=zero_pos)
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
    base_positions = torch.arange(seq_len, device=x.device)

    print("Testing RoPE implementation...")
    out = attn(x, rope_positions=base_positions)
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {out.shape}")
    assert out.shape == (batch, seq_len, dim)

    pos_rev = torch.arange(seq_len - 1, -1, -1, device=x.device)
    out_rev = attn(x, rope_positions=pos_rev)
    diff = torch.abs(out - out_rev).mean()
    print(f"✓ Position sensitivity test - mean difference: {diff.item():.6f}")
    assert diff > 1e-6, "RoPE should be sensitive to positional changes"
    print("✓ RoPE is position-sensitive (good!)")

    out_same = attn(x, rope_positions=base_positions)
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
    attn.clear_cache()
    assert len(attn.rope.cache) == 0
    x = torch.randn(batch, seq_len, dim)
    attn(x)
    assert len(attn.rope.cache) == 1, "RoPE cache should populate after first use"
    attn(x)
    assert len(attn.rope.cache) == 1, "Repeated use should reuse cached cos/sin pairs"
    attn.clear_cache()
    assert len(attn.rope.cache) == 0, "Cache should be empty after clear_cache()"

# test_attention_mha_matches_legacy()
# test_transformer_mha_matches_legacy()
test_attention_mqa_reduces_to_single_kv_head()
test_attention_gqa_matches_grouped_behavior()
test_attention_mqa_matches_grouped_behavior()
test_attention_rope_matches_no_rope_when_positions_zero()
test_rope_detailed()
test_rope_cache_and_clear()
print("All tests passed")
