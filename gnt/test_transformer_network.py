import torch

from transformer_network import Attention as ModernAttention
from transformer_network import Transformer as ModernTransformer
from transformer_network_original import Attention as LegacyAttention
from transformer_network_original import Transformer as LegacyTransformer
from gqa import GroupedQueryAttention as GQAAttention


def test_attention_mha_matches_legacy():
    torch.manual_seed(0)
    batch, seq_len, dim, heads = 1, 2, 8, 4
    legacy = LegacyAttention(dim, heads, dp_rate=0.0)
    modern = ModernAttention(dim, heads, dp_rate=0.0, kv_heads=heads)
    modern.load_state_dict(legacy.state_dict())
    legacy.eval()
    modern.eval()
    x = torch.randn(batch, seq_len, dim)

    legacy_out = legacy(x)
    modern_out = modern(x)
    assert torch.allclose(legacy_out, modern_out, atol=1e-6)


def test_transformer_mha_matches_legacy():
    torch.manual_seed(1)
    batch, seq_len, dim, heads = 2, 3, 32, 4
    legacy = LegacyTransformer(dim, ff_hid_dim=64, ff_dp_rate=0.0, n_heads=heads, attn_dp_rate=0.0)
    modern = ModernTransformer(
        dim,
        ff_hid_dim=64,
        ff_dp_rate=0.0,
        n_heads=heads,
        attn_dp_rate=0.0,
        kv_heads=heads,
    )
    modern.load_state_dict(legacy.state_dict())
    legacy.eval()
    modern.eval()
    x = torch.randn(batch, seq_len, dim)

    legacy_out = legacy(x)
    modern_out = modern(x)
    assert torch.allclose(legacy_out, modern_out, atol=1e-6)

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

test_attention_mha_matches_legacy()
test_transformer_mha_matches_legacy()
test_attention_mqa_reduces_to_single_kv_head()
test_attention_gqa_matches_grouped_behavior()
print("All tests passed")
