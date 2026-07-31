import time
import torch
import flashinfer
from termcolor import colored


class LLM:
    """
    A class representing the LLM (currently support Llama and Qwen).
    """

    def __init__(
        self, 
        model_name: str,
        max_length: int,
        dtype: torch.dtype,
        device_map: str
    ) -> None:
        """ Initializes the LLM.
        Args:
            model_name (str): The name of the model.
            max_length (int): The maximum length (prefill+decode) of sequences.
            dtype (torch.dtype): The data type for model computations.
            device_map (str): The device for model, suppor 'cuda:x' or 'auto (automatically use all visible GPUs)'.
        """
        self.model_name = model_name
        self.max_length = max_length
        self.dtype = dtype
        self.device_map = device_map


    def layer_prefill(self, layer_idx, start_bdx, hidden_states):
        # print(f'Layer = {layer_idx}, start_bdx = {start_bdx}')

        bsz, seq_len, dim = hidden_states.shape
        layer = self.layers[layer_idx]
        
        # original hidden_states used as residual, clone a new one to process
        temp_hidden_states = hidden_states.clone()
        temp_hidden_states = self.layernorm(temp_hidden_states, layer.input_layernorm_variance_epsilon, layer.input_layernorm_weight)
        
        query_states, key_states, value_states = self.wqkv(temp_hidden_states, layer)
        del temp_hidden_states
        query_states, key_states = self.position_embedd(query_states, key_states)

        query_states = query_states.view(bsz, seq_len, self.num_heads, self.head_dim) # reshape [bs, seq_len, dim] => [bs, seq_len, head, head_dim]
        key_states = key_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)

        if self.attention_type == "SpecDecoder":
            self.verify_kv_cache.prefill_update_kv_cache(query_states, key_states, value_states, layer_idx, start_bdx)
            self.verify_kv_cache.sync(layer_idx, start_bdx)
        key_states, value_states = self.kv_cache.prefill_update_kv_cache(query_states, key_states, value_states, layer_idx, start_bdx)
        temp_attn_out = self.prefill_attention(query_states, key_states, value_states, layer_idx)
        self.kv_cache.sync(layer_idx, start_bdx)
        del query_states, key_states, value_states

        hidden_states += self.wo(temp_attn_out, layer, bsz, seq_len, dim)
        del temp_attn_out

        # post attention
        residual = hidden_states.clone()

        hidden_states = self.layernorm(hidden_states, layer.post_attention_layernorm_variance_epsilon, layer.post_attention_layernorm_weight)
        # faster when split batches
        for batch_idx in range(0, bsz, 1):
            # chunk for lower memory comsumption, especially for 1M context
            for start_idx in range(0, seq_len, 65536):
                end_idx = min(seq_len, start_idx + 65536)
                hidden_states[batch_idx:batch_idx+1, start_idx:end_idx, :] = self.mlp(hidden_states[batch_idx:batch_idx+1, start_idx:end_idx, :], layer)

        hidden_states += residual
        del residual

        return hidden_states


    def layer_decode(self, layer_idx, hidden_states, decode_mode=None):
        # print(f'Layer = {layer_idx}')
        full_verify = self.attention_type == "SpecDecoder" and decode_mode == "full_verify"

        residual = hidden_states
        bsz, seq_len, dim = hidden_states.shape
        # assert seq_len == 1, f"Error: seq_len should be 1 for decoding, but got {seq_len}."
        layer = self.layers[layer_idx]

        hidden_states = self.layernorm(hidden_states, layer.input_layernorm_variance_epsilon, layer.input_layernorm_weight)
        
        query_states, key_states, value_states = self.wqkv(hidden_states, layer)
        query_states, key_states = self.position_embedd(query_states, key_states) if not full_verify else self.position_embedd(query_states, key_states, kv_cache=self.verify_kv_cache)

        query_states = query_states.view(bsz, seq_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)

        if full_verify:
            self.kv_cache.decode_update_kv_cache(key_states, value_states, layer_idx)
            key_states, value_states = self.verify_kv_cache.decode_update_kv_cache(key_states, value_states, layer_idx)
        else:
            key_states, value_states = self.kv_cache.decode_update_kv_cache(key_states, value_states, layer_idx)
        attn_out = self.decode_attention(query_states, key_states, value_states, layer_idx, decode_mode=decode_mode)
        hidden_states = self.wo(attn_out, layer, bsz, seq_len, dim)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layernorm(hidden_states, layer.post_attention_layernorm_variance_epsilon, layer.post_attention_layernorm_weight)
        hidden_states = self.mlp(hidden_states, layer)
        hidden_states = residual + hidden_states

        return hidden_states


    def prefill_forward(self, inputs_ids):
        bsz, seq_len = inputs_ids.shape
        device = inputs_ids.device

        last_hidden_states = torch.empty((bsz, 1, self.hidden_size), dtype=self.dtype, device=device).contiguous()
        for start_bdx in range(0, bsz, self.prefill_bsz):
            end_bdx = min(bsz, start_bdx + self.prefill_bsz)
            hidden_states = self.word_embedding(inputs_ids[start_bdx:end_bdx])  # [prefill_batch_size, seq_len, hidden_size]

            if self.num_gpus > 1:
                for ldx in range(self.num_layers):
                    hidden_states = self.layer_prefill(ldx, start_bdx, hidden_states)
                    hidden_states = self.parameter_move(hidden_states, ldx)
                last_hidden_states[start_bdx:end_bdx] = hidden_states[:, -1:, :].to(self.layers[0].device)
            else:
                for ldx in range(self.num_layers):
                    hidden_states = self.layer_prefill(ldx, start_bdx, hidden_states)
                last_hidden_states[start_bdx:end_bdx] = hidden_states[:, -1:, :]
        
        last_hidden_states = self.layernorm(last_hidden_states, self.norm_variance_epsilon, self.norm_weight)
        logits = self.lm(last_hidden_states)
        
        return logits
        

    def decode_forward(self, inputs_ids, decode_mode=None):
        hidden_states = self.word_embedding(inputs_ids)

        if self.num_gpus > 1:
            for ldx in range(self.num_layers):
                hidden_states = self.layer_decode(ldx, hidden_states, decode_mode=decode_mode)
                hidden_states = self.parameter_move(hidden_states, ldx)
            hidden_states = hidden_states.to(self.layers[0].device)
        else:
            for ldx in range(self.num_layers):
                hidden_states = self.layer_decode(ldx, hidden_states, decode_mode=decode_mode)
        
        hidden_states = self.layernorm(hidden_states, self.norm_variance_epsilon, self.norm_weight)
        logits = self.lm(hidden_states)
        
        return logits


    def sampling(self, logits, do_sample=False, temperature=0.6, top_p=0.95, top_k=20):
        if not do_sample:
            output_ids = logits.argmax(dim=-1)  # [bsz, 1], torch.int64
        else:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1, dtype=torch.float32)  # [bsz, 1, vocab_size]
            probs = probs.squeeze(1) # [bsz, vocab_size]
            if top_k != 0:
                output_ids = flashinfer.sampling.top_k_top_p_sampling_from_probs(probs, top_p=top_p, top_k=top_k)
            else:
                output_ids = flashinfer.sampling.top_p_sampling_from_probs(probs, top_p=top_p)
            output_ids = output_ids.unsqueeze(1) # [bsz, 1], torch.int32

        return output_ids


    def should_stop_draft(self, draft_count, draft_margin, draft_hit_attn):
        if draft_count < self.min_draft_stride:
            return False, None

        stop_reason = []

        if self.draft_margin_threshold >= 0.0 and draft_margin < self.draft_margin_threshold:
            stop_reason.append("margin")

        if self.draft_hit_attn_threshold >= 0.0 and not self.first_draft_step and draft_hit_attn < self.draft_hit_attn_threshold:
            stop_reason.append("hit_attn")

        if not stop_reason:
            return False, None
        return True, "+".join(stop_reason)


    def should_trigger_full_verify(self, generated_len, pending_sparse_count, expanded_reason):
        trigger_reasons = list(expanded_reason)

        if pending_sparse_count >= self.max_sparse_stride:
            trigger_reasons.append("pend_limit")

        if generated_len + pending_sparse_count >= self.max_new_length - 1:
            trigger_reasons.append("generate_limit")

        if self.kv_cache.will_update_index and self.kv_cache.static_pattern_total >= self.kv_cache.static_pattern_start + self.kv_cache.static_pattern_end + self.kv_cache.UPDATE_SEGMENT:
            trigger_reasons.append("index_update")

        if len(trigger_reasons) == 0:
            return False, None
        return True, "+".join(trigger_reasons)


    def draft(self, input_ids, draft_length, do_sample=False, temperature=0.6, top_p=0.95, top_k=20):
        draft_tokens = []
        draft_token = input_ids
        draft_metrics = []
        draft_reason = "length_limit"

        self.kv_cache.begin_draft()
        try:
            for _ in range(draft_length):
                draft_logits = self.decode_forward(inputs_ids=draft_token, decode_mode="draft")
                draft_token = self.sampling(draft_logits, do_sample=do_sample, temperature=temperature, top_p=top_p, top_k=top_k)

                draft_tokens.append(draft_token)
                print(colored(f"{draft_token.item()}", 'blue'), end="")

                draft_logits_fp32 = draft_logits.detach().float().squeeze(1)
                draft_top2 = torch.topk(draft_logits_fp32, k=2, dim=-1)
                draft_margin = draft_top2.values[:, 0] - draft_top2.values[:, 1]

                draft_metric = {
                    "draft_margin": draft_margin.mean().item(),
                    "hit_attn": torch.stack(self.kv_cache.hit_attention_ratios).float().mean().item(),
                    "retrieval_attn": torch.stack(self.kv_cache.retrieval_attention_ratios).float().mean().item(),
                    "expanded_attn": torch.stack(self.kv_cache.expanded_attention_ratios).float().mean().item()
                }
                draft_metrics.append(draft_metric)
                print(colored(f"({round(draft_metric['draft_margin'], 4)}, {round(draft_metric['hit_attn'], 4)})", "cyan"), end=" ")

                should_stop, draft_stop_reason = self.should_stop_draft(len(draft_tokens), draft_metric['draft_margin'], draft_metric['hit_attn'])
                if should_stop:
                    draft_reason = draft_stop_reason
                    break
        finally:
            self.kv_cache.end_draft()

        print()
        return draft_tokens, draft_metrics, draft_reason


    def print_metric_summary(self, group_name, records):
        if len(records) == 0:
            print(f"\n{group_name}: no samples")
            return

        print(f"\n{group_name}: count={len(records)}")

        metric_names = [
            "draft_margin",
            "sparse_margin",
            "expanded_margin",
            "hit_attn",
            "retrieval_attn",
            "expanded_attn"
        ]

        for metric_name in metric_names:
            raw_values = [
                record[metric_name]
                for record in records
                if record.get(metric_name) is not None
            ]
            if len(raw_values) == 0:
                print(f"  {metric_name}: no valid samples")
                continue

            values = torch.tensor(raw_values, dtype=torch.float64)
            quantiles = torch.quantile(values, torch.tensor([0.25, 0.50, 0.75], dtype=values.dtype))
            q25, median, q75 = quantiles.tolist()

            print(
                f"  {metric_name}: "
                f"n={len(values)}, "
                f"mean={values.mean().item():.4f}, "
                f"median={median:.4f}, "
                f"q25={q25:.4f}, "
                f"q75={q75:.4f}, "
                f"min={values.min().item():.4f}, "
                f"max={values.max().item():.4f}"
            )


    def sparse_verify(self, input_ids, draft_tokens, draft_metrics, do_sample=False, temperature=0.6, top_p=0.95, top_k=20):
        expanded_accept_num = 0
        sparse_tokens = []
        accepted_metrics = []
        rejected_metrics = []
        sparse_reason = []
        expanded_reason = []
        sparse_token = input_ids

        for i in range(len(draft_tokens)):
            sparse_reason.clear()
            expanded_reason.clear()
            sparse_input_token = sparse_token
            token_checkpoint = self.kv_cache.checkpoint_verify_token()

            sparse_logits = self.decode_forward(inputs_ids=sparse_token, decode_mode="sparse_verify")
            sparse_token = self.sampling(sparse_logits, do_sample=do_sample, temperature=temperature, top_p=top_p, top_k=top_k)

            sparse_changed = not torch.equal(sparse_token, draft_tokens[i])

            sparse_logits_fp32 = sparse_logits.detach().float().squeeze(1)
            sparse_top2 = torch.topk(sparse_logits_fp32, k=2, dim=-1)
            sparse_margin = (sparse_top2.values[:, 0] - sparse_top2.values[:, 1]).mean().item()

            sparse_retrieval_attn = draft_metrics[i]['retrieval_attn']

            print(colored(f"{sparse_token.item()}", 'yellow'), end="")
            print(colored(f"({round(sparse_margin, 4)}, {round(sparse_retrieval_attn, 4)})", 'red' if sparse_changed else 'green'), end=" ")

            if sparse_changed: sparse_reason.append("change")
            if self.sparse_margin_threshold >= 0.0 and sparse_margin < self.sparse_margin_threshold: sparse_reason.append("margin")
            if self.sparse_retrieval_attn_threshold >= 0.0 and sparse_retrieval_attn < self.sparse_retrieval_attn_threshold: sparse_reason.append("retrieval_attn")

            metric_record = {
                **draft_metrics[i],
                "sparse_margin": sparse_margin,
                "sparse_reason": "+".join(sparse_reason) if sparse_reason else "-"
            }

            if sparse_reason:
                self.kv_cache.restore_verify_token(token_checkpoint)
                self.kv_cache.expanded_verify_mode = True
                try:
                    expanded_logits = self.decode_forward(inputs_ids=sparse_input_token,decode_mode="sparse_verify")
                    expanded_token = self.sampling(expanded_logits, do_sample=do_sample, temperature=temperature, top_p=top_p, top_k=top_k)

                    expanded_changed = not torch.equal(expanded_token, sparse_token)

                    expanded_logits_fp32 = expanded_logits.detach().float().squeeze(1)
                    expanded_top2 = torch.topk(expanded_logits_fp32, k=2, dim=-1)
                    expanded_margin = (expanded_top2.values[:, 0] - expanded_top2.values[:, 1]).mean().item()

                    expanded_attn = draft_metrics[i]['expanded_attn']

                    if expanded_changed: expanded_reason.append("change")
                    if self.expanded_margin_threshold >= 0.0 and expanded_margin < self.expanded_margin_threshold: expanded_reason.append("margin")
                    if self.expanded_attn_threshold >= 0.0 and expanded_attn < self.expanded_attn_threshold: expanded_reason.append("expanded_attn")

                    metric_record.update({
                        "expanded_margin": expanded_margin,
                        "expanded_reason": "+".join(expanded_reason) if expanded_reason else "-"
                    })

                    sparse_token = expanded_token
                    print(colored(f"{sparse_token.item()}({round(expanded_margin, 4)}, {round(expanded_attn, 4)})", 'red' if expanded_reason else 'green'), end=" ")
                finally:
                    self.kv_cache.expanded_verify_mode = False

                if not expanded_reason: expanded_accept_num += 1

            sparse_tokens.append(sparse_token)
            if sparse_changed:
                rejected_metrics.append(metric_record)
            else:
                accepted_metrics.append(metric_record)

            if sparse_changed or expanded_reason:
                break

        print()
        return sparse_tokens, accepted_metrics, rejected_metrics, sparse_reason, expanded_reason, expanded_accept_num


    def full_verify(self, input_ids, sparse_tokens, do_sample=False, temperature=0.6, top_p=0.95, top_k=20):
        full_tokens = []
        accept_count = 0
        reject_count = 0
        full_token = input_ids

        for sparse_token in sparse_tokens:
            full_logits = self.decode_forward(inputs_ids=full_token, decode_mode="full_verify")
            full_token = self.sampling(full_logits, do_sample=do_sample, temperature=temperature, top_p=top_p, top_k=top_k)

            accept = torch.equal(full_token, sparse_token)
            full_tokens.append(full_token)
            if accept:
                accept_count += 1
            else:
                reject_count += 1

            print(colored(f"{full_token.item()}", 'green' if accept else 'red'), end=" ")

            if not accept:
                break

        print()
        return full_tokens, accept_count, reject_count


    def inference(self, inputs_ids, do_sample=False, temperature=0.6, top_p=0.95, top_k=20, ignore_eos=True):
        outputs_ids = []    # multi iteration, multi request
        output_ids = []     # single iteration, multi request
        
        # Prefilling
        print("Start prefilling ...")
        torch.cuda.synchronize()
        prefill_start = time.time()

        logits = self.prefill_forward(inputs_ids=inputs_ids)
        output_ids = self.sampling(logits, do_sample=do_sample, temperature=temperature, top_p=top_p, top_k=top_k)
        outputs_ids.append(output_ids)
        self.move()

        torch.cuda.synchronize()
        prefill_end = time.time()
        print(colored(f"Prefilling latency: {round((prefill_end - prefill_start), 4)} s", 'green'))

        # CUDAGraph Capture (if enabled)
        if self.attention_type == "RetroInfer":
            self.kv_cache.capture_cuda_graph()
        
        # check if get EOS token during decoding
        if not ignore_eos:
            end_of_text = torch.zeros((self.batch_size, 1), dtype=torch.bool, device=inputs_ids.device)
            token_id_dtype = torch.int64 if not do_sample else torch.int32  # flashinfer returns int32
            eos_token = torch.empty((self.batch_size, 1), dtype=token_id_dtype, device=inputs_ids.device).fill_(self.tokenizer.eos_token_id)
        
        # Decoding
        print("Start decoding ...")
        torch.cuda.synchronize()
        decode_start = time.time()

        if self.attention_type in ['Full_Flash_Attn', 'RetroInfer']:
            for _ in range(self.max_new_length-1):
                logits = self.decode_forward(inputs_ids=output_ids)
                output_ids = self.sampling(logits, do_sample=do_sample, temperature=temperature, top_p=top_p, top_k=top_k)
                if not ignore_eos:
                    end_of_text |= (output_ids == eos_token)
                    if end_of_text.all():
                        print(colored("All sequences have reached EOS token, stop decoding.", 'yellow'))
                        break
                outputs_ids.append(output_ids)
        elif self.attention_type == "SpecDecoder":
            generated_len = 0
            draft_num = 0
            sparse_accept_num = 0
            sparse_reject_num = 0
            sparse_step = 0
            expanded_accept_num = 0
            expanded_reject_num = 0
            full_accept_num = 0
            full_reject_num = 0
            full_step = 0
            self.first_draft_step = True
            sparse_accepted_metrics_list = []
            sparse_rejected_metrics_list = []
            pending_sparse_tokens = []
            full_trigger_reason_counts = {}

            while generated_len < self.max_new_length-1:
                # Draft 阶段
                print(colored("Draft:", 'blue'), end=" ")
                actual_stride = min(
                    self.kv_cache.spec_stride,
                    self.max_new_length-generated_len-len(pending_sparse_tokens)-1,
                    self.max_sparse_stride-len(pending_sparse_tokens),
                    self.kv_cache.static_stride-self.kv_cache.static_pattern_total
                )
                if actual_stride <= 0:
                    break
                draft_input_ids = pending_sparse_tokens[-1] if len(pending_sparse_tokens) > 0 else output_ids
                draft_tokens, draft_metrics, draft_reason = self.draft(draft_input_ids, actual_stride, do_sample=do_sample, temperature=temperature, top_p=top_p, top_k=top_k)
                draft_num += len(draft_tokens)
                self.first_draft_step = False

                # Sparse Verify 阶段
                print(colored(f"Sparse by {draft_reason}:", 'yellow'), end=" ")
                if len(pending_sparse_tokens) == 0:
                    self.kv_cache.begin_verify()
                else:
                    self.kv_cache.verify_block()
                sparse_tokens, sparse_accepted_metrics, sparse_rejected_metrics, sparse_reason, expanded_reason, expanded_accept_n = self.sparse_verify(draft_input_ids, draft_tokens, draft_metrics, do_sample=do_sample, temperature=temperature, top_p=top_p, top_k=top_k)
                sparse_accept_num += len(sparse_accepted_metrics)
                sparse_reject_num += len(sparse_rejected_metrics)
                sparse_step += 1
                expanded_accept_num += expanded_accept_n
                if expanded_reason: expanded_reject_num += 1
                sparse_accepted_metrics_list.extend(sparse_accepted_metrics)
                sparse_rejected_metrics_list.extend(sparse_rejected_metrics)
                pending_sparse_tokens.extend(sparse_tokens)

                full_trigger, full_trigger_reasons = self.should_trigger_full_verify(generated_len, len(pending_sparse_tokens), expanded_reason)
                if not full_trigger:
                    print(colored("Full deferred", 'green'))
                    continue

                self.kv_cache.end_verify()

                # Full Verify 阶段
                print(colored(f"Full by {full_trigger_reasons}:", 'red' if expanded_reason else 'green'), end=" ")
                for reason in full_trigger_reasons.split("+"):
                    full_trigger_reason_counts[reason] = full_trigger_reason_counts.get(reason, 0) + 1
                full_tokens, current_full_accept_num, current_full_reject_num = self.full_verify(output_ids, pending_sparse_tokens, do_sample=do_sample, temperature=temperature, top_p=top_p, top_k=top_k)
                full_accept_num += current_full_accept_num
                full_reject_num += current_full_reject_num
                full_step += 1
                pending_sparse_tokens.clear()

                for full_token in full_tokens:
                    outputs_ids.append(full_token)
                    generated_len += 1
                    if not ignore_eos:
                        end_of_text |= (full_token == eos_token)
                        if end_of_text.all():
                            print(colored("All sequences have reached EOS token, stop decoding.", 'yellow'))
                            break
                    if generated_len >= self.max_new_length-1:
                        break
                if not ignore_eos and end_of_text.all():
                    break
                output_ids = full_tokens[-1]
                print()
            print(
                colored(
                    f"Draft tokens: {draft_num}, "
                    f"Sparse accept tokens: {sparse_accept_num}, "
                    f"Sparse reject tokens: {sparse_reject_num}, "
                    f"Sparse step: {sparse_step}, "
                    f"Expanded accept tokens: {expanded_accept_num}, "
                    f"Expanded reject tokens: {expanded_reject_num}, "
                    f"Expanded step: {expanded_accept_num + expanded_reject_num}, "
                    f"Full accept tokens: {full_accept_num}, "
                    f"Full reject tokens: {full_reject_num}, "
                    f"Full step: {full_step}",
                    "green"
                )
            )
            print("Full verify trigger counts: " f"{full_trigger_reason_counts}")
            self.print_metric_summary("Draft vs Sparse - Accepted group", sparse_accepted_metrics_list)
            self.print_metric_summary("Draft vs Sparse - Rejected group", sparse_rejected_metrics_list)

        torch.cuda.synchronize()
        decode_end = time.time()
        print(colored(
            f"Decoding latency: {round((decode_end - decode_start), 4)} s ({round((decode_end - decode_start) * 1000 / (len(outputs_ids) - 1), 2)} ms/step), "
            f"Throughput: {round(self.batch_size * (len(outputs_ids) - 1) / (decode_end - decode_start), 2)} tokens/s",
            'green'
        ))

        print(colored(f"End2End Latency: {round((prefill_end - prefill_start + decode_end - decode_start), 4)} s\n", 'green'))
        
        outputs_ids = torch.cat(outputs_ids, dim=-1).tolist()
        
        return outputs_ids


    def generate(self, attention_type, inputs_ids, attention_masks, max_new_length, attn_config,
                 do_sample=False, temperature=0.6, top_p=0.95, top_k=20, ignore_eos=True, 
                 prefill_bsz=1, prefill_method="full"):
        """ LLM Inference.
        Args:
            attention_type: str, Full_Flash_Attn or RetroInfer or SpecDecoder.
            input_ids (torch.tensor): The input of LLM.
            attention_masks (torch.tensor): The attention masks of LLM.
            max_new_length (int): The maximum length of generated sequences.
            attn_config (dict): The deoding attention configuration.
            do_sample, temperature, top_p, top_k, ignore_eos: The sampling parameters.
            prefill_bsz (int): The batch size for prefill.
            prefill_method (str): The method for prefill, support full and xattn.
        """
        self.attention_type = attention_type

        bs, input_length = inputs_ids.shape
        self.batch_size = bs
        self.input_length = input_length
        self.max_new_length = max_new_length
        assert self.input_length + self.max_new_length <= self.max_length, \
            f"Error: input_length({self.input_length}) + max_new_length({self.max_new_length}) exceeds max_length({self.max_length})"

        # draft 阶段相关配置
        if self.attention_type == "SpecDecoder":
            spec_config = attn_config["SpecDecoder"]
            self.min_draft_stride = spec_config["min_draft_stride"]
            self.max_draft_stride = spec_config["max_draft_stride"]
            self.draft_margin_threshold = spec_config["draft_margin_threshold"]
            self.draft_hit_attn_threshold = spec_config["draft_hit_attn_threshold"]
            self.max_sparse_stride = spec_config["max_sparse_stride"]
            self.sparse_margin_threshold = spec_config["sparse_margin_threshold"]
            self.sparse_retrieval_attn_threshold = spec_config["sparse_retrieval_attn_threshold"]
            self.expanded_margin_threshold = spec_config["expanded_margin_threshold"]
            self.expanded_attn_threshold = spec_config["expanded_attn_threshold"]
            if not 1 <= self.min_draft_stride <= self.max_draft_stride:
                raise ValueError(f"min_draft_stride should be in [1, max_draft_stride] but got min_draft_stride={self.min_draft_stride} and max_draft_stride={self.max_draft_stride}")

        # compute valid start position for each sequence
        valid_start = attention_masks.shape[1] - torch.sum(attention_masks, dim=-1).detach().cpu().numpy()
        del attention_masks

        self.prefill_bsz = min(prefill_bsz, self.batch_size)
        self.prefill_method = prefill_method
        # set prefill batch size to 1 and prefill method to full attention if input sequences are not in the same length
        if not (valid_start == 0).all():
            self.prefill_bsz = 1
            self.prefill_method = "full"

        print("Allocate GPU buffers and CPU pin memory ...")
        self.init_kv_cache(valid_start, attn_config)

        outputs = self.inference(
            inputs_ids, 
            do_sample=do_sample, 
            temperature=temperature, 
            top_p=top_p, 
            top_k=top_k, 
            ignore_eos=ignore_eos
        )

        return outputs
