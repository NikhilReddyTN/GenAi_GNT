import numpy as np
import torch
import torch.nn as nn


class RotaryPositionalEmbeddings(nn.Module):
    """RoPE following RoFormer."""

    def __init__(self, d, base=10000):
        super(RotaryPositionalEmbeddings, self).__init__()
        if d % 2 != 0:
            raise ValueError("RoPE requires head_dim to be even.")
        self.d = d
        self.base = base
        theta = base ** (-2 * torch.arange(0, d // 2).float() / d)
        self.register_buffer("theta", theta, persistent=False)
        self.cache = {}

    def _build_cache(self, seq_len, device, dtype):
        if seq_len not in self.cache:
            positions = torch.arange(seq_len, device=device).float()
            matrix = torch.einsum("m,d->md", positions, self.theta.to(device))
            cos_C = torch.cos(matrix)
            sin_C = torch.sin(matrix)
            cos_C = torch.cat([cos_C, cos_C], dim=-1)
            sin_C = torch.cat([sin_C, sin_C], dim=-1)
            self.cache[seq_len] = (cos_C.cpu(), sin_C.cpu())
        cos_C, sin_C = self.cache[seq_len]
        cos_C = cos_C.to(device=device, dtype=dtype)
        sin_C = sin_C.to(device=device, dtype=dtype)
        return cos_C, sin_C

    def forward(self, Y, positions=None):
        b, h, t, d = Y.shape
        if positions is None:
            cos_C, sin_C = self._build_cache(t, Y.device, Y.dtype)
            cos_C = cos_C.view(1, 1, t, d)
            sin_C = sin_C.view(1, 1, t, d)
        else:
            positions = positions.to(device=Y.device, dtype=Y.dtype)
            if positions.dim() == 1:
                positions = positions.unsqueeze(0)
            freqs = positions[..., None] * self.theta.to(Y.device, Y.dtype)
            cos = torch.cos(freqs)
            sin = torch.sin(freqs)
            cos_C = torch.cat([cos, cos], dim=-1)[:, None, :, :]
            sin_C = torch.cat([sin, sin], dim=-1)[:, None, :, :]
        x1 = Y[..., : d // 2]
        x2 = Y[..., d // 2 :]
        rotated_x1 = x1 * cos_C[..., : d // 2] - x2 * sin_C[..., : d // 2]
        rotated_x2 = x2 * cos_C[..., d // 2 :] + x1 * sin_C[..., d // 2 :]
        return torch.cat([rotated_x1, rotated_x2], dim=-1)

# sin-cose embedding module
class Embedder(nn.Module):
    def __init__(self, **kwargs):
        super(Embedder, self).__init__()
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def forward(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class FeedForward(nn.Module):
    def __init__(self, dim, hid_dim, dp_rate):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, dim)
        self.dp = nn.Dropout(dp_rate)
        self.activ = nn.ReLU()

    def forward(self, x):
        x = self.dp(self.activ(self.fc1(x)))
        x = self.dp(self.fc2(x))
        return x


# Subtraction-based efficient attention
class Attention2D(nn.Module):
    def __init__(self, dim, dp_rate):
        super(Attention2D, self).__init__()
        self.q_fc = nn.Linear(dim, dim, bias=False)
        self.k_fc = nn.Linear(dim, dim, bias=False)
        self.v_fc = nn.Linear(dim, dim, bias=False)
        self.pos_fc = nn.Sequential(
            nn.Linear(4, dim // 8),
            nn.ReLU(),
            nn.Linear(dim // 8, dim),
        )
        self.attn_fc = nn.Sequential(
            nn.Linear(dim, dim // 8),
            nn.ReLU(),
            nn.Linear(dim // 8, dim),
        )
        self.out_fc = nn.Linear(dim, dim)
        self.dp = nn.Dropout(dp_rate)

    def forward(self, q, k, pos, mask=None):
        q = self.q_fc(q)
        k = self.k_fc(k)
        v = self.v_fc(k)

        pos = self.pos_fc(pos)
        attn = k - q[:, :, None, :] + pos
        attn = self.attn_fc(attn)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(attn, dim=-2)
        attn = self.dp(attn)

        x = ((v + pos) * attn).sum(dim=2)
        x = self.dp(self.out_fc(x))
        return x


# View Transformer
class Transformer2D(nn.Module):
    def __init__(self, dim, ff_hid_dim, ff_dp_rate, attn_dp_rate):
        super(Transformer2D, self).__init__()
        self.attn_norm = nn.LayerNorm(dim, eps=1e-6)
        self.ff_norm = nn.LayerNorm(dim, eps=1e-6)

        self.ff = FeedForward(dim, ff_hid_dim, ff_dp_rate)
        self.attn = Attention2D(dim, attn_dp_rate)

    def forward(self, q, k, pos, mask=None):
        residue = q
        x = self.attn_norm(q)
        x = self.attn(x, k, pos, mask)
        x = x + residue

        residue = x
        x = self.ff_norm(x)
        x = self.ff(x)
        x = x + residue

        return x


# attention module for self attention.
# contains several adaptations to incorportate positional information (NOT IN PAPER)
#   - qk (default) -> only (q.k) attention.
#   - pos -> replace (q.k) attention with position attention.
#   - gate -> weighted addition of  (q.k) attention and position attention.
class Attention(nn.Module):
    def __init__(
        self, dim, n_heads, dp_rate, attn_mode="qk", pos_dim=None, kv_heads=None, use_rope=False
    ):
        super(Attention, self).__init__()
        if dim % n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.kv_heads = kv_heads if kv_heads is not None else n_heads
        if self.kv_heads < 1:
            raise ValueError("kv_heads must be >= 1")
        if self.n_heads % self.kv_heads != 0:
            raise ValueError("n_heads must be divisible by kv_heads for grouped attention")
        self.heads_per_kv = self.n_heads // self.kv_heads
        self.kv_dim = self.head_dim * self.kv_heads
        if attn_mode in ["qk", "gate"]:
            self.q_fc = nn.Linear(dim, self.head_dim * self.n_heads, bias=False)
            self.k_fc = nn.Linear(dim, self.kv_dim, bias=False)
        if attn_mode in ["pos", "gate"]:
            self.pos_fc = nn.Sequential(
                nn.Linear(pos_dim, pos_dim), nn.ReLU(), nn.Linear(pos_dim, dim // 8)
            )
            self.head_fc = nn.Linear(dim // 8, n_heads)
        if attn_mode == "gate":
            self.gate = nn.Parameter(torch.ones(n_heads))
        self.v_fc = nn.Linear(dim, self.kv_dim, bias=False)
        self.out_fc = nn.Linear(dim, dim)
        self.dp = nn.Dropout(dp_rate)
        self.attn_mode = attn_mode
        self.use_rope = use_rope
        if self.use_rope:
            if attn_mode not in ["qk", "gate"]:
                raise ValueError("RoPE can only be used with qk or gate attention modes.")
            if self.head_dim % 2 != 0:
                raise ValueError("head_dim must be even to use RoPE.")
            self.rope = RotaryPositionalEmbeddings(self.head_dim)

    def expand_kv(self, x):
        if self.heads_per_kv == 1:
            return x
        return x.repeat_interleave(self.heads_per_kv, dim=1)

    def clear_cache(self):
        if self.use_rope:
            self.rope.cache.clear()

    def forward(self, x, pos=None, rope_positions=None, ret_attn=False):
        if self.attn_mode in ["qk", "gate"]:
            q = self.q_fc(x)
            q = q.view(x.shape[0], x.shape[1], self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            k = self.k_fc(x)
            k = k.view(x.shape[0], x.shape[1], self.kv_heads, self.head_dim).permute(0, 2, 1, 3)
            k = self.expand_kv(k)
        v = self.v_fc(x)
        v = v.view(x.shape[0], x.shape[1], self.kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.expand_kv(v)

        if self.use_rope and self.attn_mode in ["qk", "gate"]:
            q = self.rope(q, rope_positions)
            k = self.rope(k, rope_positions)

        if self.attn_mode in ["qk", "gate"]:
            attn = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(q.shape[-1])
            attn = torch.softmax(attn, dim=-1)
        elif self.attn_mode == "pos":
            pos = self.pos_fc(pos)
            attn = self.head_fc(pos[:, :, None, :] - pos[:, None, :, :]).permute(0, 3, 1, 2)
            attn = torch.softmax(attn, dim=-1)
        if self.attn_mode == "gate":
            pos = self.pos_fc(pos)
            pos_attn = self.head_fc(pos[:, :, None, :] - pos[:, None, :, :]).permute(0, 3, 1, 2)
            pos_attn = torch.softmax(pos_attn, dim=-1)
            gate = self.gate.view(1, -1, 1, 1)
            attn = (1.0 - torch.sigmoid(gate)) * attn + torch.sigmoid(gate) * pos_attn
            attn /= attn.sum(dim=-1).unsqueeze(-1)
        attn = self.dp(attn)

        out = torch.matmul(attn, v).permute(0, 2, 1, 3).contiguous()
        out = out.view(x.shape[0], x.shape[1], -1)
        out = self.dp(self.out_fc(out))
        if ret_attn:
            return out, attn
        else:
            return out


# Ray Transformer
class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        ff_hid_dim,
        ff_dp_rate,
        n_heads,
        attn_dp_rate,
        attn_mode="qk",
        pos_dim=None,
        kv_heads=None,
        use_rope=False,
    ):
        super(Transformer, self).__init__()
        self.attn_norm = nn.LayerNorm(dim, eps=1e-6)
        self.ff_norm = nn.LayerNorm(dim, eps=1e-6)

        self.ff = FeedForward(dim, ff_hid_dim, ff_dp_rate)
        self.attn = Attention(dim, n_heads, attn_dp_rate, attn_mode, pos_dim, kv_heads, use_rope)
        self.use_rope = use_rope

    def forward(self, x, pos=None, rope_positions=None, ret_attn=False):
        if self.use_rope:
            seq_len = x.shape[1]
            if rope_positions is not None:
                if rope_positions.dim() == 1 and rope_positions.shape[0] != seq_len:
                    raise ValueError("rope_positions length must match the sequence length.")
                if rope_positions.dim() == 2 and rope_positions.shape[1] != seq_len:
                    raise ValueError("rope_positions must align with the input sequence length.")
        elif pos is not None and (pos.shape[0] != x.shape[0] or pos.shape[1] != x.shape[1]):
            raise ValueError("Positional embeddings must align with the input sequence shape.")
        residue = x
        x = self.attn_norm(x)
        x = self.attn(x, pos, rope_positions, ret_attn)
        if ret_attn:
            x, attn = x
        x = x + residue

        residue = x
        x = self.ff_norm(x)
        x = self.ff(x)
        x = x + residue

        if ret_attn:
            return x, attn.mean(dim=1)[:, 0]
        else:
            return x

    def clear_cache(self):
        if hasattr(self.attn, "clear_cache"):
            self.attn.clear_cache()


class GNT(nn.Module):
    def __init__(self, args, in_feat_ch=32, posenc_dim=3, viewenc_dim=3, ret_alpha=False):
        super(GNT, self).__init__()
        self.use_rope = getattr(args, "use_rope", False)
        self.rgbfeat_fc = nn.Sequential(
            nn.Linear(in_feat_ch + 3, args.netwidth),
            nn.ReLU(),
            nn.Linear(args.netwidth, args.netwidth),
        )

        # NOTE: Apologies for the confusing naming scheme, here view_crosstrans refers to the view transformer, while the view_selftrans refers to the ray transformer
        self.view_selftrans = nn.ModuleList([])
        self.view_crosstrans = nn.ModuleList([])
        self.q_fcs = nn.ModuleList([])
        self.num_attn_heads = 4
        self.attn_type = getattr(args, "attn_type", "mha").lower()
        self.gqa_group_size = getattr(args, "gqa_group_size", 2)
        kv_heads = self.compute_kv_heads(self.attn_type, self.num_attn_heads, self.gqa_group_size)
        pos_feat_dim = posenc_dim + viewenc_dim
        for i in range(args.trans_depth):
            # view transformer
            view_trans = Transformer2D(
                dim=args.netwidth,
                ff_hid_dim=int(args.netwidth * 4),
                ff_dp_rate=0.1,
                attn_dp_rate=0.1,
            )
            self.view_crosstrans.append(view_trans)
            # ray transformer
            ray_trans = Transformer(
                dim=args.netwidth,
                ff_hid_dim=int(args.netwidth * 4),
                n_heads=self.num_attn_heads,
                ff_dp_rate=0.1,
                attn_dp_rate=0.1,
                attn_mode="qk",
                pos_dim=pos_feat_dim,
                kv_heads=kv_heads,
                use_rope=self.use_rope,
            )
            self.view_selftrans.append(ray_trans)
            # mlp
            if i % 2 == 0:
                q_fc = nn.Sequential(
                    nn.Linear(args.netwidth + pos_feat_dim, args.netwidth),
                    nn.ReLU(),
                    nn.Linear(args.netwidth, args.netwidth),
                )
            else:
                q_fc = nn.Identity()
            self.q_fcs.append(q_fc)

        self.posenc_dim = posenc_dim
        self.viewenc_dim = viewenc_dim
        self.ret_alpha = ret_alpha
        self.norm = nn.LayerNorm(args.netwidth)
        self.rgb_fc = nn.Linear(args.netwidth, 3)
        self.relu = nn.ReLU()
        self.pos_enc = Embedder(
            input_dims=3,
            include_input=True,
            max_freq_log2=9,
            num_freqs=10,
            log_sampling=True,
            periodic_fns=[torch.sin, torch.cos],
        )
        self.view_enc = Embedder(
            input_dims=3,
            include_input=True,
            max_freq_log2=9,
            num_freqs=10,
            log_sampling=True,
            periodic_fns=[torch.sin, torch.cos],
        )

    def compute_kv_heads(self, attn_type, n_heads, gqa_group_size):
        attn_type = attn_type.lower()
        if attn_type == "mha":
            return n_heads
        if attn_type == "mqa":
            return 1
        if attn_type == "gqa":
            if gqa_group_size < 1:
                raise ValueError("gqa_group_size must be >= 1")
            if n_heads % gqa_group_size != 0:
                raise ValueError("n_heads must be divisible by gqa_group_size for GQA")
            return n_heads // gqa_group_size
        raise ValueError("Unknown attention type: {}".format(attn_type))

    def forward(self, rgb_feat, ray_diff, mask, pts, ray_d):
        # compute positional embeddings
        viewdirs = ray_d
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()
        viewdirs = self.view_enc(viewdirs)
        pts_ = torch.reshape(pts, [-1, pts.shape[-1]]).float()
        pts_ = self.pos_enc(pts_)
        pts_ = torch.reshape(pts_, list(pts.shape[:-1]) + [pts_.shape[-1]])
        viewdirs_ = viewdirs[:, None].expand(pts_.shape)
        embed = torch.cat([pts_, viewdirs_], dim=-1)
        input_pts, input_views = torch.split(embed, [self.posenc_dim, self.viewenc_dim], dim=-1)
        rope_positions = None

        # project rgb features to netwidth
        rgb_feat = self.rgbfeat_fc(rgb_feat)
        # q_init -> maxpool
        q = rgb_feat.max(dim=2)[0]
        if self.use_rope:
            rope_positions = torch.arange(q.shape[1], device=q.device)

        # transformer modules
        for i, (crosstrans, q_fc, selftrans) in enumerate(
            zip(self.view_crosstrans, self.q_fcs, self.view_selftrans)
        ):
            # view transformer to update q
            q = crosstrans(q, rgb_feat, ray_diff, mask)
            # embed positional information
            if i % 2 == 0:
                q = torch.cat((q, input_pts, input_views), dim=-1)
            q = q_fc(q)
            # ray transformer
            q = selftrans(q, pos=embed, rope_positions=rope_positions, ret_attn=self.ret_alpha)
            # 'learned' density
            if self.ret_alpha:
                q, attn = q
        # normalize & rgb
        h = self.norm(q)
        outputs = self.rgb_fc(h.mean(dim=1))
        if self.ret_alpha:
            return torch.cat([outputs, attn], dim=1)
        else:
            return outputs

    def clear_cache(self):
        for transformer in self.view_selftrans:
            if hasattr(transformer, "clear_cache"):
                transformer.clear_cache()
