import numpy as np
import torch
import torch.nn as nn
import math

from einops import einsum, rearrange
from torch.nn import functional as F

class GroupedQueryAttention(nn.Module):
    """
    An implementation of group query attention. Refer to the CausalSelfAttention class to structure your implementation.
    """

    def __init__(self, n_query_head, n_embd, n_kv_head):
        super().__init__()

        """
        Initialize the class with the relevant arguments for GQA, in a similar fashion to CausalSelfAttention.
        Make sure to implement RoPE for GQA if the config specifies that RoPE should be used.
        """
        # Ensures that the embedding dimension is divisible by both the number of query heads and key/value heads
        assert n_embd % n_query_head == 0
        assert n_embd % n_kv_head == 0
        assert n_query_head % n_kv_head == 0

        # Initialize any required variables
        self.n_query_head = n_query_head
        self.n_embd = n_embd
        self.n_kv_head = n_kv_head
        self.group_size = n_query_head // n_kv_head

        # Key, Query, Value Projections
        self.q_fc = nn.Linear(self.n_embd, self.n_embd, bias=False)
        # k_fc and v_fc have less output dimensions because there are fewer key/value heads than query heads
        self.k_fc = nn.Linear(self.n_embd, self.n_embd//self.group_size, bias=False)
        self.v_fc = nn.Linear(self.n_embd, self.n_embd//self.group_size, bias=False)

        # Output Projection: you must define this as nn.Linear().
        self.out_fc = nn.Linear(self.n_embd, self.n_embd)

        # # Create causal mask
        # self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
        #                                 .view(1, 1, 1, config.block_size, config.block_size))

    def forward(self, x):
        """
        Implement the forward pass for Grouped Query Attention in a similar fashion to CausalSelfAttention.
        """
        b, t, n_embd = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values based on the input x
        q = self.q_fc(x) # shape (b, t, n_embd)
        k = self.k_fc(x) # shape (b, t, n_embd/group_size)
        v = self.v_fc(x) # shape (b, t, n_embd/group_size)
        # for query, let's split the embedding dimension (n_embd) across both the number of query heads and key/value heads
        q = rearrange(q, 'b t (h_kv g d) -> b h_kv g t d', h_kv=self.n_kv_head, g=self.group_size)
        # for key and value, let's split the embedding dimension (n_embd/group_size) across the number of key/value heads
        k = rearrange(k, 'b t (h_kv d) -> b h_kv t d', h_kv=self.n_kv_head)
        v = rearrange(v, 'b t (h_kv d) -> b h_kv t d', h_kv=self.n_kv_head)
        
        # compute square root of (n_embd / number of heads) to scale the dot product
        scale = math.sqrt(k.size(-1))

        # calculate the attention scores with the query and  key
        att = einsum(q, k, 'b h_kv g q d, b h_kv k d -> b h_kv g q k') / scale
        att = F.softmax(att, dim=-1)
        # matrix multiplication of attention scores and value
        y = einsum(att, v, 'b h_kv g q k, b h_kv k d -> b h_kv g q d')
        # rearrange the output tensor to (batch size, sequence length, n_embd)
        y = rearrange(y, 'b h_kv g q d -> b q (h_kv g d)') # re-assemble all head outputs side by side
        # output projection
        y = self.out_fc(y)
        return y