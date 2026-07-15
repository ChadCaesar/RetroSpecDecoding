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


    def layer_decode(self, layer_idx, hidden_states):
        # print(f'Layer = {layer_idx}')

        residual = hidden_states
        bsz, seq_len, dim = hidden_states.shape
        # assert seq_len == 1, f"Error: seq_len should be 1 for decoding, but got {seq_len}."
        layer = self.layers[layer_idx]

        hidden_states = self.layernorm(hidden_states, layer.input_layernorm_variance_epsilon, layer.input_layernorm_weight)
        
        query_states, key_states, value_states = self.wqkv(hidden_states, layer)
        query_states, key_states = self.position_embedd(query_states, key_states)

        query_states = query_states.view(bsz, seq_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)

        key_states, value_states = self.kv_cache.decode_update_kv_cache(key_states, value_states, layer_idx)
        attn_out = self.decode_attention(query_states, key_states, value_states, layer_idx)
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
        

    def decode_forward(self, inputs_ids):
        hidden_states = self.word_embedding(inputs_ids)

        if self.num_gpus > 1:
            for ldx in range(self.num_layers):
                hidden_states, attn_out = self.layer_decode(ldx, hidden_states)
                hidden_states = self.parameter_move(hidden_states, ldx)
            hidden_states = hidden_states.to(self.layers[0].device)
        else:
            for ldx in range(self.num_layers):
                hidden_states, attn_out = self.layer_decode(ldx, hidden_states)
        
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
            verify_num = 0
            reject_num = 0
            step_num = 0
            while generated_len < self.max_new_length-1:
                # Draft 阶段
                self.kv_cache.begin_draft()
                draft_token = output_ids
                draft_attn_outs = []
                draft_tokens = []
                actual_stride = min(self.kv_cache.spec_stride, self.max_length-self.kv_cache.context-1)
                for _ in range(actual_stride):
                    draft_logits, draft_attn_out = self.decode_forward(inputs_ids=draft_token)
                    draft_attn_outs.append(draft_attn_out)
                    draft_token = self.sampling(draft_logits, do_sample=do_sample, temperature=temperature, top_p=top_p, top_k=top_k)
                    draft_tokens.append(draft_token)
                    print(colored(f"{draft_token.item()}", 'blue'), end=" ")
                    draft_num += 1
                self.kv_cache.end_draft()
                # Verify 阶段
                verify_token = output_ids
                verify_tokens = []
                for i in range(actual_stride):
                    verify_logits, verify_attn_out = self.decode_forward(inputs_ids=verify_token)
                    verify_token = self.sampling(verify_logits, do_sample=do_sample, temperature=temperature, top_p=top_p, top_k=top_k)
                    verify_tokens.append(verify_token)
                    verify_num += 1
                    print(colored(f"{verify_token.item()}", 'yellow'), end="")
                    draft_attn_out = draft_attn_outs[i]
                    sim = torch.cosine_similarity(draft_attn_out, verify_attn_out, dim=-1)
                    if verify_token != draft_tokens[i]:
                        reject_num += 1
                        print(colored(f"({round(sim.mean().item(), 4)}, {round(sim.max().item(), 4)}, {round(sim.min().item(), 4)})", 'red'), end=" ")
                        break
                    print(colored(f"({round(sim.mean().item(), 4)}, {round(sim.max().item(), 4)}, {round(sim.min().item(), 4)})", 'green'), end=" ")
                for verify_token in verify_tokens:
                    outputs_ids.append(verify_token)
                    generated_len += 1
                    if not ignore_eos:
                        end_of_text |= (verify_token == eos_token)
                        if end_of_text.all():
                            print(colored("All sequences have reached EOS token, stop decoding.", 'yellow'))
                            break
                    if generated_len >= self.max_new_length-1:
                        break
                if not ignore_eos and end_of_text.all():
                    break
                output_ids = verify_tokens[-1]
                step_num += 1
                print()
            print(colored(f"Draft tokens: {draft_num}, Verify tokens: {verify_num}, Generate steps: {step_num}, Accept rate: {round(100 * (verify_num - reject_num) / max(draft_num, 1), 2)} %, Accept tokens per step: {round((verify_num / max(step_num, 1)), 2)}", 'green'))

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