# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import logging
import os
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Dict, List, Union

import numpy as np
import torch
import torch.distributed
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from vllm import LLM, SamplingParams
from vllm.distributed import parallel_state as vllm_ps
from vllm.worker.worker_base import WorkerWrapperBase

from verl import DataProto
from verl.third_party.vllm import vllm_version
from verl.utils.debug import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id
    # is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


class vLLMRollout(BaseRollout):
    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        if kwargs.get("train_tp") is not None:
            # deployed with megatron
            import os

            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            if vllm_version in (
                "0.5.4",
                "0.6.3",
            ):
                train_tp = kwargs.get("train_tp")
                num_tp_per_train_tp = train_tp // tensor_parallel_size
                vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size, num_tp_per_train_tp=num_tp_per_train_tp)
            else:
                vllm_ps.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size)

        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config") and hasattr(model_hf_config.llm_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")

            assert max_position_embeddings >= config.prompt_length + config.response_length, "model context length should be greater than total sequence length"

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        trust_remote_code = kwargs.get("trust_remote_code", False)
        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        limit_mm_per_prompt = None
        if config.get("limit_images", None):  # support for multi-image data
            limit_mm_per_prompt = {"image": config.get("limit_images")}

        # copy it to avoid secretly modifying the engine config
        engine_kwargs = {} if "engine_kwargs" not in config or "vllm" not in config.engine_kwargs else OmegaConf.to_container(deepcopy(config.engine_kwargs.vllm))
        # For each vLLM engine parameter,
        # - `None` means not setting it, so we pop it, and leave it to vLLM default value
        #    (which can vary across different vLLM versions);
        # - Otherwise it's the desired value we want to explicitly set.
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            disable_mm_preprocessor_cache=True,
            limit_mm_per_prompt=limit_mm_per_prompt,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
            **engine_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=1,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # # we may detokenize the result all together later
        if vllm_version != "0.3.1":
            kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    # @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    # @torch.no_grad()
    # def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
    #     """Roll out a batch of prompts.

    #     * 验证阶段 (`is_validate=True`)：单温度、单样本，与原实现一致。
    #     * 训练 / 推理阶段：对同一 prompt 依次采样 `n` 条，温度按
    #       `base_T ±0.1 ±0.2 …` 对称展开（`n≤5` 时即 `T,−0.1,+0.1,−0.2,+0.2`）。
    #     * 返回张量形状、字段名与旧实现完全一致。
    #     """
    #     # ------------------------------------------------------------------
    #     # 0. 可选：重建 vLLM cache engine（老版本需要）
    #     # ------------------------------------------------------------------
    #     if (
    #         vllm_version in ("0.5.4", "0.6.3")
    #         and self.config.free_cache_engine
    #     ):
    #         self.inference_engine.init_cache_engine()

    #     # ------------------------------------------------------------------
    #     # 1. 解析 prompt 侧张量 & metadata
    #     # ------------------------------------------------------------------
    #     idx            = prompts.batch["input_ids"]       # (bs, prompt_len)
    #     attention_mask = prompts.batch["attention_mask"]  # left‑padded
    #     position_ids   = prompts.batch["position_ids"]
    #     eos_token_id   = prompts.meta_info["eos_token_id"]
    #     bs             = idx.size(0)

    #     # ------------------- non‑tensor batch，转 raw ids 给 vLLM ----------
    #     ntb = prompts.non_tensor_batch
    #     if "raw_prompt_ids" not in ntb:
    #         ntb["raw_prompt_ids"] = np.array(
    #             [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(bs)],
    #             dtype=object,
    #         )
    #     if bs != len(ntb["raw_prompt_ids"]):
    #         raise RuntimeError("vllm sharding manager is not work properly.")

    #     if "multi_modal_data" in ntb:
    #         vllm_inputs = [
    #             {"prompt_token_ids": p_ids.tolist() if isinstance(p_ids, np.ndarray) else p_ids,
    #              "multi_modal_data" : mmd}
    #             for p_ids, mmd in zip(ntb.pop("raw_prompt_ids"), ntb.pop("multi_modal_data"))
    #         ]
    #     else:
    #         vllm_inputs = [
    #             {"prompt_token_ids": p_ids.tolist() if isinstance(p_ids, np.ndarray) else p_ids}
    #             for p_ids in ntb.pop("raw_prompt_ids")
    #         ]

    #     # ------------------------------------------------------------------
    #     # 2. 选择分支（do_sample / validate / greedy）并拿到 token‑id 列表
    #     # ------------------------------------------------------------------
    #     do_sample   = prompts.meta_info.get("do_sample", True)
    #     is_validate = prompts.meta_info.get("validate", False)

    #     outputs_token_ids: list[list[int]] = []  # 最终收集到这里

    #     # ------ (A) Greedy ⇒ temperature=0, n=1 ---------------------------
    #     if not do_sample:
    #         greedy_kw = {
    #             "best_of"    : 1,
    #             "top_p"      : 1.0,
    #             "top_k"      : -1,
    #             "min_p"      : 0.0,
    #             "temperature": 0.0,
    #             "n"          : 1,
    #         }
    #         with self.update_sampling_params(**greedy_kw):
    #             outs = self.inference_engine.generate(vllm_inputs, self.sampling_params, use_tqdm=False)
    #         for o in outs:
    #             outputs_token_ids.append(o.outputs[0].token_ids)

    #     # ------ (B) Validation ⇒ 使用 config.val_kwargs，n=1 --------------
    #     elif is_validate:
    #         val_kw = {
    #             "top_k"      : self.config.val_kwargs.top_k,
    #             "top_p"      : self.config.val_kwargs.top_p,
    #             "temperature": self.config.val_kwargs.temperature,
    #             "n"          : 1,
    #         }
    #         with self.update_sampling_params(**val_kw):
    #             outs = self.inference_engine.generate(vllm_inputs, self.sampling_params, use_tqdm=False)
    #         for o in outs:
    #             outputs_token_ids.append(o.outputs[0].token_ids)

    #     # ------ (C) Training / Sampling ⇒ 多温度，多次调用 ---------------
    #     else:
    #         orig_n = self.sampling_params.n           # 目标采样条数
    #         base_T = kwargs.get("temperature", self.sampling_params.temperature)

    #         # 构造对称偏移序列：0, −0.1, +0.1, −0.2, +0.2, −0.3 …
    #         deltas, step = [0.0, -0.1, 0.1, -0.2, 0.2], 0.3
    #         while len(deltas) < orig_n:
    #             deltas.extend([-step, step])
    #             step += 0.1
    #         temp_list = [max(0.0, base_T + d) for d in deltas[:orig_n]]

    #         for T in temp_list:
    #             per_kw = {"temperature": T, "n": 1}
    #             with self.update_sampling_params(**per_kw):
    #                 outs = self.inference_engine.generate(vllm_inputs, self.sampling_params, use_tqdm=False)
    #             for o in outs:
    #                 outputs_token_ids.append(o.outputs[0].token_ids)

    #         # 现在 outputs_token_ids 顺序为：prompt0_T0, prompt1_T0 … prompt{bs-1}_T0, prompt0_T1 …
    #         # 需要转成 prompt 主序 + temp 次序
    #         reordered = []
    #         for i in range(bs):
    #             for j in range(orig_n):
    #                 reordered.append(outputs_token_ids[j*bs + i])
    #         outputs_token_ids = reordered

    #     # ------------------------------------------------------------------
    #     # 3. pad → torch.Tensor & repeat prompt‑side张量（当 n>1）
    #     # ------------------------------------------------------------------
    #     response = pad_2d_list_to_length(
    #         outputs_token_ids,
    #         self.pad_token_id,
    #         max_length=self.config.response_length,
    #     ).to(idx.device)                                   # (bs*n, resp_len)

    #     effective_n = response.size(0) // bs
    #     if effective_n > 1:
    #         idx            = _repeat_interleave(idx,            effective_n)
    #         attention_mask = _repeat_interleave(attention_mask, effective_n)
    #         position_ids   = _repeat_interleave(position_ids,   effective_n)
    #         if "multi_modal_inputs" in ntb:
    #             ntb["multi_modal_inputs"] = _repeat_interleave(ntb["multi_modal_inputs"], effective_n)
    #         if "tools_kwargs" in ntb:
    #             ntb["tools_kwargs"] = _repeat_interleave(ntb["tools_kwargs"], effective_n)
    #         bs *= effective_n

    #     seq = torch.cat([idx, response], dim=-1)

    #     # --------------------- position_ids / attention_mask --------------
    #     resp_len = response.size(1)
    #     delta_pid = torch.arange(1, resp_len + 1, device=position_ids.device)
    #     delta_pid = delta_pid.unsqueeze(0).expand(bs, -1)
    #     if position_ids.dim() == 3:  # qwen2vl mrope
    #         delta_pid = delta_pid.view(bs, 1, -1).expand(bs, 3, -1)
    #     pos_resp = position_ids[..., -1:] + delta_pid
    #     position_ids = torch.cat([position_ids, pos_resp], dim=-1)

    #     resp_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
    #     attention_mask = torch.cat((attention_mask, resp_mask), dim=-1)

    #     # ------------------------------------------------------------------
    #     # 4. 打包返回
    #     # ------------------------------------------------------------------
    #     batch = TensorDict(
    #         {
    #             "prompts"      : idx,
    #             "responses"    : response,
    #             "input_ids"    : seq,
    #             "attention_mask": attention_mask,
    #             "position_ids" : position_ids,
    #         },
    #         batch_size=bs,
    #     )

    #     # optional: free cache engine
    #     if (
    #         vllm_version in ("0.5.4", "0.6.3")
    #         and self.config.free_cache_engine
    #     ):
    #         self.inference_engine.free_cache_engine()

    #     return DataProto(batch=batch, non_tensor_batch=ntb)


    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Roll out a batch of prompts.

        * 验证阶段 (`is_validate=True`)：单温度、单样本，与原实现一致。
        * 训练 / 推理阶段：对同一 prompt 依次采样 `n` 条，温度按
          `base_T ±0.1 ±0.2 …` 对称展开（`n≤5` 时即 `T,−0.1,+0.1,−0.2,+0.2`）。
        * 返回张量形状、字段名与旧实现完全一致。
        """
        # ------------------------------------------------------------------
        # 0. 可选：重建 vLLM cache engine（老版本需要）
        # ------------------------------------------------------------------
        if (
            vllm_version in ("0.5.4", "0.6.3")
            and self.config.free_cache_engine
        ):
            self.inference_engine.init_cache_engine()

        # ------------------------------------------------------------------
        # 1. 解析 prompt 侧张量 & metadata
        # ------------------------------------------------------------------
        idx            = prompts.batch["input_ids"]       # (bs, prompt_len)
        attention_mask = prompts.batch["attention_mask"]  # left‑padded
        position_ids   = prompts.batch["position_ids"]
        eos_token_id   = prompts.meta_info["eos_token_id"]
        bs             = idx.size(0)

        # ------------------- non‑tensor batch，转 raw ids 给 vLLM ----------
        ntb = prompts.non_tensor_batch
        if "raw_prompt_ids" not in ntb:
            ntb["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(bs)],
                dtype=object,
            )
        if bs != len(ntb["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in ntb:
            vllm_inputs = [
                {"prompt_token_ids": p_ids.tolist() if isinstance(p_ids, np.ndarray) else p_ids,
                 "multi_modal_data" : mmd}
                for p_ids, mmd in zip(ntb.pop("raw_prompt_ids"), ntb.pop("multi_modal_data"))
            ]
        else:
            vllm_inputs = [
                {"prompt_token_ids": p_ids.tolist() if isinstance(p_ids, np.ndarray) else p_ids}
                for p_ids in ntb.pop("raw_prompt_ids")
            ]

        # ------------------------------------------------------------------
        # 2. 选择分支（do_sample / validate / greedy）并拿到 token‑id 列表
        # ------------------------------------------------------------------
        do_sample   = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)

        outputs_token_ids: list[list[int]] = []  # 最终收集到这里
        temps_used: list[float] = []

        # ------ (A) Greedy ⇒ temperature=0, n=1 ---------------------------
        if not do_sample:
            greedy_kw = {
                "best_of"    : 1,
                "top_p"      : 1.0,
                "top_k"      : -1,
                "min_p"      : 0.0,
                "temperature": 0.0,
                "n"          : 1,
            }
            with self.update_sampling_params(**greedy_kw):
                outs = self.inference_engine.generate(vllm_inputs, self.sampling_params, use_tqdm=False)
            for o in outs:
                outputs_token_ids.append(o.outputs[0].token_ids)

        # ------ (B) Validation ⇒ 使用 config.val_kwargs，n=1 --------------
        elif is_validate:
            val_kw = {
                "top_k"      : self.config.val_kwargs.top_k,
                "top_p"      : self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n"          : 1,
                "logprobs": 1,
            }
            with self.update_sampling_params(**val_kw):
                print('test gen:', self.sampling_params)
                outs = self.inference_engine.generate(vllm_inputs, self.sampling_params, use_tqdm=False)
            for o in outs:
                outputs_token_ids.append(o.outputs[0].token_ids)
                temps_used.append(val_kw["temperature"])   # ← 记录温度

        # ------ (C) Training / Sampling ⇒ 多 prompt × 多 temperature，一次调用 ---------------
        else:
            import uuid

            bs       = len(vllm_inputs)          # prompt 数
            orig_n   = self.sampling_params.n    # 目标采样条数 (= 温度数)
            base_T   = kwargs.get("temperature", self.sampling_params.temperature)

            # ① 构造对称偏移温度序列
            deltas, step = [0.0, 0.0, -0.1, 0.1, -0.2, 0.2], 0.3
            while len(deltas) < orig_n:
                deltas.extend([-step, step])
                step += 0.1
            temp_list = [max(0.0, base_T + d) for d in deltas[:orig_n]]

            # ② 准备 batched_reqs & batched_params（外层 temp，内层 prompt → temp‑major 顺序）
            batched_reqs, batched_params = [], []
            base_sp = deepcopy(self.sampling_params)
            base_sp.n = 1                       # 每个组合各出 1 条 sample
            base_sp.logprobs = 1

            for T in temp_list:                 # temp-major 顺序保证与旧代码一致
                for req in vllm_inputs:         # 遍历 prompt
                    new_req = deepcopy(req)     # 深拷贝防止引用冲突
                    # 若是 GenerateRequest/PromptTokens，需要改 request_id
                    if hasattr(new_req, "request_id"):
                        new_req.request_id = f"{new_req.request_id}_{uuid.uuid4().hex[:8]}"
                    batched_reqs.append(new_req)

                    sp = deepcopy(base_sp)
                    sp.temperature = T
                    batched_params.append(sp)

            # ③ 一次性 generate
            outs = self.inference_engine.generate(
                batched_reqs,
                batched_params,
                use_tqdm=False,
            )

            # ④ 解析输出、统计温度列命中
            outputs_token_ids, temps_used = [], []
            correct_counter, false_counter = 0, 0

            for output in outs:                 # 与 batched_reqs 位置一一对应
                for sample in output.outputs:   # n == 1
                    # (1) token_ids
                    outputs_token_ids.append(sample.token_ids)

                    # (2) 解析我们写入 sentinel token_id == -1 的温度
                    step_dicts = getattr(sample, "top_logprobs", None)
                    if step_dicts is None:      # vLLM≥0.6 改名 token_logprobs
                        step_dicts = sample.logprobs

                    temps = []
                    for step in step_dicts:     # step : dict{token_id: Logprob}
                        sent = step.get(-1)
                        if sent is not None:
                            correct_counter += 1
                            temps.append(float(sent.logprob))
                        else:
                            false_counter += 1
                            temps.append(base_T) # fall‑back
                    temps_used.append(temps)

            # ⑤ 可选日志
            # print("sentinel 命中 / miss:", correct_counter, false_counter)

            # ⑥ 重排为 prompt‑major 顺序：prompt0_T0, prompt0_T1, … prompt0_T{n-1}, prompt1_T0 …
            reordered_tok, reordered_tmp = [], []
            for i in range(bs):                 # prompt idx
                for j in range(orig_n):         # temp idx
                    index = j * bs + i            # 当前位置在 temp‑major 列表里的索引
                    reordered_tok.append(outputs_token_ids[index])
                    reordered_tmp.append(temps_used[index])

            outputs_token_ids = reordered_tok
            temps_used        = reordered_tmp

            # ⑦ 统计不同温度（排除 None）
            unique_temps_no_none = {t for seq in temps_used for t in seq if t is not None}
            # print(f"不同温度数: {len(unique_temps_no_none)} -> {unique_temps_no_none}")

            # 后续代码可直接使用重排后的 outputs_token_ids, temps_used



        # ------------------------------------------------------------------
        # 3. pad → torch.Tensor & repeat prompt‑side张量（当 n>1）
        # ------------------------------------------------------------------
        response = pad_2d_list_to_length(
            outputs_token_ids,
            self.pad_token_id,
            max_length=self.config.response_length,
        ).to(idx.device)                                   # (bs*n, resp_len)

        if is_validate:
            token_level_temperature = response # validate dont recompute log_probs, put response here to pass check
        else:
            token_level_temperature = pad_2d_list_to_length(
                temps_used,
                self.sampling_params.temperature,                           # sentinel
                max_length=self.config.response_length,
            ).to(idx.device) # (bs*effective_n, resp_len)

        effective_n = response.size(0) // bs
        if effective_n > 1:
            idx            = _repeat_interleave(idx,            effective_n)
            attention_mask = _repeat_interleave(attention_mask, effective_n)
            position_ids   = _repeat_interleave(position_ids,   effective_n)
            if "multi_modal_inputs" in ntb:
                ntb["multi_modal_inputs"] = _repeat_interleave(ntb["multi_modal_inputs"], effective_n)
            if "tools_kwargs" in ntb:
                ntb["tools_kwargs"] = _repeat_interleave(ntb["tools_kwargs"], effective_n)
            bs *= effective_n

        seq = torch.cat([idx, response], dim=-1)

        # --------------------- position_ids / attention_mask --------------
        resp_len = response.size(1)
        delta_pid = torch.arange(1, resp_len + 1, device=position_ids.device)
        delta_pid = delta_pid.unsqueeze(0).expand(bs, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_pid = delta_pid.view(bs, 1, -1).expand(bs, 3, -1)
        pos_resp = position_ids[..., -1:] + delta_pid
        position_ids = torch.cat([position_ids, pos_resp], dim=-1)

        resp_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, resp_mask), dim=-1)

        # ------------------------------------------------------------------
        # 4. 打包返回
        # ------------------------------------------------------------------
        batch = TensorDict(
            {
                "prompts"      : idx,
                "responses"    : response,
                "input_ids"    : seq,
                "attention_mask": attention_mask,
                "position_ids" : position_ids,
                "token_temperature": token_level_temperature
            },
            batch_size=bs,
        )
        

        # optional: free cache engine
        if (
            vllm_version in ("0.5.4", "0.6.3")
            and self.config.free_cache_engine
        ):
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch, non_tensor_batch=ntb)



    '''@GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.init_cache_engine()

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array([_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data")):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [{"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")]

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        # https://github.com/volcengine/verl/pull/772
        for input_data in vllm_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}")

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            print('spmd rollout')
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            response = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response.append(output.outputs[sample_id].token_ids)

            response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(idx.device)

            if self.sampling_params.n > 1 and do_sample:
                idx = _repeat_interleave(idx, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                batch_size = batch_size * self.sampling_params.n
                if "multi_modal_inputs" in non_tensor_batch.keys():
                    non_tensor_batch["multi_modal_inputs"] = _repeat_interleave(non_tensor_batch["multi_modal_inputs"], self.sampling_params.n)
                # NOTE(linjunrong): for multi-turn https://github.com/volcengine/verl/pull/1037
                if "tools_kwargs" in non_tensor_batch.keys():
                    non_tensor_batch["tools_kwargs"] = _repeat_interleave(non_tensor_batch["tools_kwargs"], self.sampling_params.n)

            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        # free vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)'''


class vLLMAsyncRollout:
    """vLLMAsyncRollout is a thin wrapper of WorkerWrapperBase,
    which is engine in single worker process.
    """

    def __init__(self, *args, **kwargs):
        # Engine is deferred to be initialized in init_worker
        self.inference_engine: WorkerWrapperBase = None
        self.sharding_manager = None
        self.is_sleep = False

    def init_worker(self, all_kwargs: List[Dict[str, Any]]):
        """Initialize worker engine."""
        all_kwargs[0]["rank"] = int(os.environ["RANK"])
        all_kwargs[0]["local_rank"] = 0

        self.vllm_config = all_kwargs[0]["vllm_config"]
        self.inference_engine = WorkerWrapperBase(vllm_config=self.vllm_config)
        self.inference_engine.init_worker(all_kwargs)

    def load_model(self, *args, **kwargs):
        self.inference_engine.load_model(*args, **kwargs)

        # inference engine is intialized now, update sharding manager
        self.sharding_manager.inference_engine = self.inference_engine
        self.sharding_manager.model_runner = self.inference_engine.worker.model_runner

    def sleep(self, *args, **kwargs):
        """Offload model weights and discard kv cache."""
        if self.is_sleep:
            return
        self.sharding_manager.__exit__(None, None, None)
        self.is_sleep = True

    def wake_up(self, *args, **kwargs):
        """Load model weights and build kv cache."""
        if not self.is_sleep:
            return
        self.sharding_manager.__enter__()  # pylint: disable=C2801
        self.is_sleep = False

    def execute_method(self, method: Union[str, bytes], *args, **kwargs):
        if method == "init_worker":
            return self.init_worker(*args, **kwargs)
        elif method == "load_model":
            return self.load_model(*args, **kwargs)
        elif method == "sleep":
            return self.sleep(*args, **kwargs)
        elif method == "wake_up":
            return self.wake_up(*args, **kwargs)
        else:
            return self.inference_engine.execute_method(method, *args, **kwargs)
