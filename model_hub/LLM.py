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

        return hidden_states, attn_out


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
                hidden_states, attn_out = self.layer_decode(ldx, hidden_states, decode_mode=decode_mode)
                hidden_states = self.parameter_move(hidden_states, ldx)
            hidden_states = hidden_states.to(self.layers[0].device)
        else:
            for ldx in range(self.num_layers):
                hidden_states, attn_out = self.layer_decode(ldx, hidden_states, decode_mode=decode_mode)
        
        hidden_states = self.layernorm(hidden_states, self.norm_variance_epsilon, self.norm_weight)
        logits = self.lm(hidden_states)
        
        return logits, attn_out


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


    def should_stop_draft(self, draft_count, draft_margin, draft_margin_drop):
        if draft_count < self.min_draft_stride:
            return False, None

        stop_reasons = []

        margin_enabled = self.draft_margin_threshold >= 0.0
        if margin_enabled and draft_margin <= self.draft_margin_threshold:
            stop_reasons.append("margin")

        margin_drop_enabled = self.draft_margin_drop_threshold >= 0.0
        if margin_drop_enabled and draft_margin_drop is not None and draft_margin_drop >= self.draft_margin_drop_threshold:
            stop_reasons.append("margin_drop")

        if len(stop_reasons) == 0:
            return False, None
        return True, "+".join(stop_reasons)


    def should_trigger_full_verify(self, generated_len, pending_sparse_count, sparse_accepted_metrics, sparse_rejected_metrics):
        trigger_reasons = []
        sparse_metric_records = sparse_accepted_metrics + sparse_rejected_metrics

        if pending_sparse_count >= self.max_sparse_stride:
            trigger_reasons.append("pend_limit")

        if generated_len + pending_sparse_count >= self.max_new_length - 1:
            trigger_reasons.append("generate_limit")

        if len(sparse_rejected_metrics) > 0:
            trigger_reasons.append("sparse_mismatch")

        max_sparse_stability_ratio = max(record["stability_ratio"] for record in sparse_metric_records)
        if max_sparse_stability_ratio >= self.sparse_stability_threshold:
            trigger_reasons.append("stability_ratio")

        should_update_index = self.kv_cache.static_pattern_total >= self.kv_cache.static_pattern_start + self.kv_cache.static_pattern_end + self.kv_cache.UPDATE_SEGMENT
        if self.kv_cache.will_update_index and should_update_index:
            trigger_reasons.append("index_update")

        if len(trigger_reasons) == 0:
            return False, None
        return True, "+".join(trigger_reasons)


    def draft(self, input_ids, draft_length, do_sample=False, temperature=0.6, top_p=0.95, top_k=20):
        draft_logits_list = []
        draft_attn_outs = []
        draft_tokens = []
        draft_token = input_ids
        draft_metrics = []
        previous_margin = None
        stop_reason = "length_limit"

        self.kv_cache.begin_draft()
        try:
            for _ in range(draft_length):
                draft_logits, draft_attn_out = self.decode_forward(inputs_ids=draft_token, decode_mode="draft")
                draft_token = self.sampling(draft_logits, do_sample=do_sample, temperature=temperature, top_p=top_p, top_k=top_k)

                draft_logits_list.append(draft_logits)
                draft_attn_outs.append(draft_attn_out)
                draft_tokens.append(draft_token)
                print(colored(f"{draft_token.item()}", 'blue'), end="")

                draft_logits_fp32 = draft_logits.detach().float().squeeze(1)
                draft_top2 = torch.topk(draft_logits_fp32, k=2, dim=-1)
                draft_margin = draft_top2.values[:, 0] - draft_top2.values[:, 1]
                if previous_margin is None:
                    draft_margin_drop = None
                    draft_margin_drop_value = None
                    draft_margin_drop_text = "-"
                else:
                    draft_margin_drop = (previous_margin - draft_margin).clamp_min(0.0) / previous_margin.abs().clamp_min(1e-6)
                    draft_margin_drop_value = draft_margin_drop.mean().item()
                    draft_margin_drop_text = round(draft_margin_drop_value, 4)
                draft_metrics.append({
                    "draft_margin": draft_margin.mean().item(),
                    "draft_margin_drop": draft_margin_drop_value
                })
                previous_margin = draft_margin.detach()
                print(colored(f"({round(draft_margin.mean().item(), 4)}, {draft_margin_drop_text})", "cyan"), end=" ")

                should_stop, metric_stop_reason = self.should_stop_draft(len(draft_tokens), draft_margin.mean().item(), draft_margin_drop_value)
                if should_stop:
                    stop_reason = metric_stop_reason
                    break
        finally:
            self.kv_cache.end_draft()

        print(colored(f"Draft stopped by {stop_reason}", "green" if stop_reason == "length_limit" else "red"))
        return draft_logits_list, draft_attn_outs, draft_tokens, draft_metrics


    def compute_stability_ratio(self, draft_logits, verify_logits):
        # Draft/Verify logits，形状由 [batch, 1, vocab] 转为 [batch, vocab]
        draft_logits_fp32 = draft_logits.float().squeeze(1)
        verify_logits_fp32 = verify_logits.detach().float().squeeze(1)

        # Verify top-1/top-2 margin
        verify_top2 = torch.topk(verify_logits_fp32, k=2, dim=-1)
        verify_top1_id = verify_top2.indices[:, 0:1]
        verify_margin = (verify_top2.values[:, 0] - verify_top2.values[:, 1])

        # 计算 Draft 相对于 Verify 决策边界的最大扰动
        draft_value_at_verify_top1 = draft_logits_fp32.gather(dim=-1, index=verify_top1_id)
        verify_top1_value = verify_logits_fp32.gather(dim=-1, index=verify_top1_id)

        draft_gaps = draft_value_at_verify_top1 - draft_logits_fp32
        verify_gaps = verify_top1_value - verify_logits_fp32

        gap_linf = (draft_gaps - verify_gaps).abs().amax(dim=-1)

        stability_ratio = (gap_linf / verify_margin.clamp_min(1e-12))
        return gap_linf, verify_margin, stability_ratio


    def print_metric_summary(self, group_name, records):
        if len(records) == 0:
            print(f"\n{group_name}: no samples")
            return

        metric_names = ["attention_mean", "gap_linf", "verify_margin", "stability_ratio", "draft_margin", "draft_margin_drop"]

        print(f"\n{group_name}: count={len(records)}")

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

        ratio_values = torch.tensor([record["stability_ratio"] for record in records], dtype=torch.float64)
        ratio_below_one = (ratio_values < 1.0).double().mean().item() * 100.0

        print(
            f"  stability_ratio < 1: "
            f"{ratio_below_one:.2f}%"
        )


    def verify(self, input_ids, draft_logits_list, draft_attn_outs, draft_tokens, draft_metrics, decode_mode, do_sample=False, temperature=0.6, top_p=0.95, top_k=20):
        verify_logits_list = []
        verify_attn_outs = []
        verify_tokens = []
        accepted_metrics = []
        rejected_metrics = []
        verify_token = input_ids

        for i in range(len(draft_tokens)):
            verify_logits, verify_attn_out = self.decode_forward(inputs_ids=verify_token, decode_mode=decode_mode)
            verify_token = self.sampling(verify_logits, do_sample=do_sample, temperature=temperature, top_p=top_p, top_k=top_k)

            verify_logits_list.append(verify_logits)
            verify_attn_outs.append(verify_attn_out)
            verify_tokens.append(verify_token)
            print(colored(f"{verify_token.item()}", 'yellow'), end="")

            accept = torch.equal(verify_token, draft_tokens[i])

            draft_logits = draft_logits_list[i]
            draft_attn_out = draft_attn_outs[i]
            attn_sim = torch.cosine_similarity(draft_attn_out.float(), verify_attn_out.float(), dim=-1)
            gap_linf, verify_margin, stability_ratio = self.compute_stability_ratio(draft_logits, verify_logits)
            print(colored(f"({round(attn_sim.mean().item(), 4)}, {round(gap_linf.mean().item(), 4)}, {round(verify_margin.mean().item(), 4)}, {round(stability_ratio.mean().item(), 4)})",
                          'green' if accept else 'red'), end=" ")
            metric_record = {
                "attention_mean": attn_sim.mean().item(),
                "gap_linf": gap_linf.mean().item(),
                "verify_margin": verify_margin.mean().item(),
                "stability_ratio": stability_ratio.mean().item(),
                "draft_margin": draft_metrics[i]["draft_margin"],
                "draft_margin_drop": draft_metrics[i]["draft_margin_drop"]
            }

            if accept:
                accepted_metrics.append(metric_record)
            else:
                rejected_metrics.append(metric_record)
                break

        print()
        return verify_logits_list, verify_attn_outs, verify_tokens, accepted_metrics, rejected_metrics


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
                logits, attn_out = self.decode_forward(inputs_ids=output_ids)
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
            full_accept_num = 0
            full_reject_num = 0
            step_num = 0
            sparse_accepted_metrics_list = []
            sparse_rejected_metrics_list = []
            pending_sparse_logits_list = []
            pending_sparse_attn_outs = []
            pending_sparse_tokens = []
            pending_sparse_draft_metrics = []
            full_accepted_metrics_list = []
            full_rejected_metrics_list = []

            while generated_len < self.max_new_length-1:
                # Draft 阶段
                actual_stride = min(
                    self.kv_cache.spec_stride,
                    self.max_new_length-generated_len-len(pending_sparse_tokens)-1,
                    self.max_sparse_stride-len(pending_sparse_tokens),
                    self.kv_cache.static_stride-self.kv_cache.static_pattern_total
                )
                if actual_stride <= 0:
                    break
                draft_input_ids = pending_sparse_tokens[-1] if len(pending_sparse_tokens) > 0 else output_ids
                draft_logits_list, draft_attn_outs, draft_tokens, draft_metrics = self.draft(draft_input_ids, actual_stride, do_sample=do_sample, temperature=temperature, top_p=top_p, top_k=top_k)
                draft_num += len(draft_tokens)

                # Sparse Verify 阶段
                print(colored("Sparse verify: ", 'yellow'), end="")
                if len(pending_sparse_tokens) == 0:
                    self.kv_cache.begin_verify()
                else:
                    self.kv_cache.verify_block()
                sparse_logits_list, sparse_attn_outs, sparse_tokens, sparse_accepted_metrics, sparse_rejected_metrics = self.verify(draft_input_ids, draft_logits_list, draft_attn_outs, draft_tokens, draft_metrics, "sparse_verify", do_sample=do_sample, temperature=temperature, top_p=top_p, top_k=top_k)
                sparse_accept_num += len(sparse_accepted_metrics)
                sparse_reject_num += len(sparse_rejected_metrics)
                sparse_accepted_metrics_list.extend(sparse_accepted_metrics)
                sparse_rejected_metrics_list.extend(sparse_rejected_metrics)
                pending_sparse_logits_list.extend(sparse_logits_list)
                pending_sparse_attn_outs.extend(sparse_attn_outs)
                pending_sparse_tokens.extend(sparse_tokens)
                pending_sparse_draft_metrics.extend(draft_metrics[:len(sparse_tokens)])

                full_trigger, full_trigger_reasons = self.should_trigger_full_verify(generated_len, len(pending_sparse_tokens), sparse_accepted_metrics, sparse_rejected_metrics)
                if not full_trigger:
                    print(colored("Full verify deferred", 'green'))
                    continue

                self.kv_cache.end_verify()

                # Full Verify 阶段
                print(colored(f"Full verify by {full_trigger_reasons}: ", 'yellow'), end="")
                full_logits_list, full_attn_outs, full_tokens, full_accepted_metrics, full_rejected_metrics = self.verify(output_ids, pending_sparse_logits_list, pending_sparse_attn_outs, pending_sparse_tokens, pending_sparse_draft_metrics, "full_verify", do_sample=do_sample, temperature=temperature, top_p=top_p, top_k=top_k)
                full_accept_num += len(full_accepted_metrics)
                full_reject_num += len(full_rejected_metrics)
                pending_sparse_logits_list.clear()
                pending_sparse_attn_outs.clear()
                pending_sparse_tokens.clear()
                pending_sparse_draft_metrics.clear()
                full_accepted_metrics_list.extend(full_accepted_metrics)
                full_rejected_metrics_list.extend(full_rejected_metrics)

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
                step_num += 1
                print()
            print(
                colored(
                    f"Draft tokens: {draft_num}, "
                    f"Sparse accept tokens: {sparse_accept_num}, "
                    f"Sparse reject tokens: {sparse_reject_num}, "
                    f"Full accept tokens: {full_accept_num}, "
                    f"Full reject tokens: {full_reject_num}, "
                    f"Generate steps: {step_num}",
                    "green",
                )
            )
            self.print_metric_summary("Draft vs Sparse - Accepted group", sparse_accepted_metrics_list)
            self.print_metric_summary("Draft vs Sparse - Rejected group", sparse_rejected_metrics_list)
            self.print_metric_summary("Sparse vs Full - Accepted group", full_accepted_metrics_list)
            self.print_metric_summary("Sparse vs Full - Rejected group", full_rejected_metrics_list)

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
            self.draft_margin_drop_threshold = spec_config["draft_margin_drop_threshold"]
            self.max_sparse_stride = spec_config["max_sparse_stride"]
            self.sparse_stability_threshold = spec_config["sparse_stability_threshold"]
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