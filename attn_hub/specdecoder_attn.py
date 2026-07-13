def specdecoder_decode_attn(query_states, layer_idx, specdecoder_cache):
    """
    query_states: query vector, shape: (batch_size, 1, head_num, dim), gpu torch tensor
    """
    # assert query_states.size(0) == specdecoder_cache.batch_size
    # assert query_states.size(1) == 1
    # assert query_states.size(2) == specdecoder_cache.kv_head * specdecoder_cache.group_size == specdecoder_cache.num_heads
    # assert query_states.size(3) == specdecoder_cache.head_dim

    static_len = specdecoder_cache.static_pattern_total if layer_idx == specdecoder_cache.layer_num - 1 else specdecoder_cache.static_pattern_total + 1
    return specdecoder_cache.attn_func(query_states.contiguous(), layer_idx, static_len)
