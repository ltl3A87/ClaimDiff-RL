from collections import defaultdict
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import time
import logging
from typing import Any

import numpy as np
import torch

from verl import DataProto
from verl.protocol import DataProtoItem
from verl.workers.reward_manager import register

from .remote_proxy import SingleStepRemoteProxyManager

logger = logging.getLogger(__name__)

def simple_replace_label(text):
    # 使用正则表达式找到'label': '...'的模式并替换为'label': 'ui'
    pattern = r"('label':\s*')[^']*(')"
    return re.sub(pattern, r"\1ui\2", text)


def replace_label_with_ui(text):
      # 提取<answer>标签中的内容
      answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.DOTALL)
      if not answer_match:
          return text

      json_str = answer_match.group(1)

      try:
          # 解析JSON
          data = json.loads(json_str)

          # 修改label为ui
          for item in data:
              if 'label' in item:
                  item['label'] = 'ui'

          # 重新构建字符串
          new_json_str = json.dumps(data, ensure_ascii=False)
          result = text.replace(answer_match.group(1), new_json_str)
          return result

      except json.JSONDecodeError:
          return text


def scale_value_from_01(value: float, target_range: list[float]) -> float:
    """
    Scales a value from the [0, 1] interval to a new specified range.

    Args:
    value: The original value, which should be in the range [0, 1].
    target_range: A tuple of two floats (min, max) defining the lower and upper
                    bounds of the target range.

    Returns:
    The scaled float value.
    """
    # Clamp value to [0, 1] range instead of asserting
    value = max(0.0, min(1.0, value))
    assert len(target_range) == 2, f"target range shoud only have two values, but found {target_range}"

    lower_bound, upper_bound = target_range

    return lower_bound + value * (upper_bound - lower_bound)


class RewardWorker:

    def __init__(
        self,
        tokenizer,
        reward_server_params,
        is_training,
        step,
        total_steps,
        acc_scale_range: list[float] = [0, 1.0],
        format_scale_range: list[float] = [0, 1.0],
        tool_consistency_scale_range: list[float] = [0, 1.0],
        tool_intrinsic_scale_range: list[float] = [0, 1.0],
        **kwargs
    ):
        # Tokenizer for decode token
        self.tokenizer = tokenizer
        # Monitor training
        self.is_training = is_training
        self.step = step
        self.total_steps = total_steps

        # acc & format will be scaled by the following parameters
        #   accuracy = acc_scale_reward * accuracy + acc_scale_penalty
        #   format = format_scale_reward * format + format_scale_penalty
        self.acc_scale_range = acc_scale_range
        self.format_scale_range = format_scale_range
        self.tool_consistency_scale_range = tool_consistency_scale_range
        self.tool_intrinsic_scale_range = tool_intrinsic_scale_range
        self.iou_threshold = kwargs.get('iou_threshold', 0.5)

        # Initialize reward server proxy within the actor
        self.reward_server_proxy = SingleStepRemoteProxyManager(
            rm_job=reward_server_params.get("rm_job"),
            rm_num=reward_server_params.get("rm_num", 8),
            rm_port=reward_server_params.get("rm_port", "8192"),
            rm_fun=reward_server_params.get("rm_fun", "/judge"),
        )

        self.iou_success_rate = 0.

    def process_item(self, idx: int, data_item: DataProtoItem):

        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, (np.bool_)):
                return bool(obj)
            else:
                return obj
    
        # fetch the data and response tensor
        prompt_ids = data_item.batch['prompts']
        prompt_length = prompt_ids.shape[-1]

        valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        response_ids = data_item.batch['responses']
        valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        # decode (assoicate with data and response)
        prompt_str: str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)  # apply chat template
        response_str: str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True) 
        answer: str = data_item.non_tensor_batch['reward_model']['answer']  # answer (not formatted)
        ground_truth: str = data_item.non_tensor_batch['reward_model']['ground_truth']  # solution (formatted)
        query: str = data_item.non_tensor_batch['extra_info'].get("question", '')  # query (without chat template)
        if query is None:
            query = ""

        # get reward model info
        data_source: str = data_item.non_tensor_batch['data_source']
        reward_verifier_style: str = data_item.non_tensor_batch['reward_model'].get('style', "None")
        reward_verifier: str = data_item.non_tensor_batch['reward_model']['verifier']

        reward_verifier_parm: dict[str, Any] = data_item.non_tensor_batch['reward_model'].get('verifier_parm', {})
        if reward_verifier_parm is None:
            reward_verifier_parm = {}
        
        # reward assoicate with image
        image_grid_thw = None
        if 'multi_modal_inputs' in data_item.non_tensor_batch and 'image_grid_thw' in data_item.non_tensor_batch['multi_modal_inputs']:
            image_grid_thw = data_item.non_tensor_batch['multi_modal_inputs']['image_grid_thw'].numpy()
            # TODO: 14 is the patch size of ViT, dynamically adjust it
            image_grid_thw = [(int(t), int(h * 14), int(w * 14)) for t, h, w in image_grid_thw]

        image_path: list[str] | None = data_item.non_tensor_batch['extra_info'].get('image_path', None)
        if isinstance(image_path, str):
            image_path = [image_path]

        # prepare reward verifier parm
        reward_verifier_parm['verifier_style'] = reward_verifier_style
        reward_verifier_parm['is_training'] = self.is_training
        reward_verifier_parm['step'] = int(self.step)
        reward_verifier_parm['total_steps'] = int(self.total_steps)
        reward_verifier_parm['image_grid_thw'] = image_grid_thw
        reward_verifier_parm['image_path'] = image_path
        reward_verifier_parm['query'] = query
        reward_verifier_parm['iou_threshold'] = self.iou_threshold

        # Convert any numpy types to Python native types for JSON serialization
        # serializable_parm = convert_to_serializable(reward_verifier_parm)
        if "ui" in data_source:
            response_str = simple_replace_label(response_str)

        # Read format_ratio from reward_model (default to 0.0 if not present)
        fmt_ratio = data_item.non_tensor_batch['reward_model'].get('format_ratio', 0.0)
        if fmt_ratio is None:
            fmt_ratio = 0.0

        payload = {
            "data_source": data_source,   # data source

            'query': query,               # query (without chat format)
            "prompt": prompt_str,         # add system prompt (chat template) of query
            "answer": answer,             # answer (not formatted)
            "solution": ground_truth,     # solution (formatted)
            "response": response_str,     # response from model

            "reward_verifier": reward_verifier,                         # reward verifier
            "reward_verifier_parm": json.dumps(reward_verifier_parm),   # reward verifier parm
            "fmt_ratio": float(fmt_ratio),  # format ratio from reward_model
            
            # Optional fields for training data logging
            "iter": int(self.step) if self.is_training else None,  # training iteration number
            "data_index": idx,  # data index
            "image_path": image_path,  # image path(s) - already extracted above
        }

        # ========== Apply reward server ==========
        rewards = self.reward_server_proxy.get_reward([payload])
        try:
            gather_rewards = rewards[0]['rewards']
        except Exception as e:
            print(f"Error in get_reward: {e}", "Payload: ", payload, flush=True)
            gather_rewards = {}
        # ========== End of reward server ==========

        # Ensure required keys exist to avoid crashes when reward server fails.
        if "format_reward" not in gather_rewards or "accuracy_reward" not in gather_rewards:
            logger.warning(
                "Missing format_reward/accuracy_reward from reward server. "
                "Falling back to 0.0. Payload id=%s, verifier=%s",
                data_item.non_tensor_batch.get("extra_info", {}).get("id", "unknown"),
                reward_verifier,
            )
            gather_rewards.setdefault("format_reward", 0.0)
            gather_rewards.setdefault("accuracy_reward", 0.0)

        # ========== Apply multi-round / image tools reward ==========
        # assign number of round back into data_item
        data_item.non_tensor_batch['extra_info']['num_turns'] = data_item.non_tensor_batch.get("__num_turns__", None)

        # Extract tool rewards from tool_metrics and compute successful rates
        format_reward_01 = gather_rewards['format_reward']
        accuracy_reward_01 = gather_rewards['accuracy_reward']

        if self.is_training:
            scaled_format_reward = scale_value_from_01(format_reward_01, self.format_scale_range)
            scaled_accuracy_reward = scale_value_from_01(accuracy_reward_01, self.acc_scale_range)
            gather_rewards['format_reward'] = scaled_format_reward
            gather_rewards['accuracy_reward'] = scaled_accuracy_reward
        else:
            scaled_format_reward = format_reward_01
            scaled_accuracy_reward = accuracy_reward_01
        # ========== End of format reward ==========

        # ========== Calculate final reward ===========
        gather_rewards["final_reward"] = scaled_accuracy_reward + scaled_format_reward
        # ========== End of final reward ===========

        result_dict = {
            "id": data_item.non_tensor_batch["extra_info"]["id"],
            "data_source": data_source,
            "prompt": prompt_str,
            "response": response_str,
            "ground_truth": ground_truth,
            "answer": answer,
            "question": query,
            "uid": data_item.non_tensor_batch.get("uid", "default_group"),  # group标识
        }
        for reward_key, reward_value in gather_rewards.items():
            result_dict[reward_key] = reward_value

        score = float(gather_rewards['final_reward'])

        return idx, int(valid_response_length), score, result_dict


@register("remote")
class RemoteRewardManager:

    def __init__(self, tokenizer, num_examine, compute_score, reward_fn_key="data_source", **reward_kwargs):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.reward_kwargs = reward_kwargs
        self.max_workers = reward_kwargs.get("max_workers", 32)
        
        # IoU threshold levels and current index
        self.iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
        self.current_iou_index = 0  # Start at index 0 (0.5)
        self.iou_threshold = self.iou_thresholds[self.current_iou_index]

    def __call__(self, data: DataProto, return_dict: bool = False):
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        prompt_ids = data.batch["prompts"]

        results = self.verify(data)
        rewards = []
        already_printed = {}

        all_result_dicts = []
        for idx, length, score, result_dict in results:
            reward_extra_info["result_dicts"].append(result_dict)
            rewards.append(score)
            reward_tensor[idx, length - 1] = score

            data_source = result_dict[self.reward_fn_key]
            if already_printed.get(data_source, 0) < self.num_examine:
                print(result_dict)
                already_printed[data_source] = already_printed.get(data_source, 0) + 1

            all_result_dicts.append(result_dict)

        data.batch["acc"] = torch.tensor(rewards, dtype=torch.float32, device=prompt_ids.device)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            return reward_tensor

    def verify(self, data):
        step, total_steps = data.meta_info["global_steps"], data.meta_info["total_steps"]

        reward_server_params = {
            "rm_job": os.environ.get("_REMOTE_REWARD_JOB_ID"),
            "rm_num": int(os.environ.get("_REMOTE_REWARD_WORKER_NUM", "8")),
            "rm_port": os.environ.get("_REMOTE_REWARD_SERVER_PORT", "8192"),
            "rm_fun": "/judge"
        }

        start_time = time.time()
        print(f"=======Start {len(data)} reward, max_workers: {self.max_workers}=======")

        # Create a worker instance
        worker = RewardWorker(
            self.tokenizer,
            reward_server_params,
            step=step,
            total_steps=total_steps,
            iou_threshold=self.iou_threshold,
            **self.reward_kwargs
        )

        # Use ThreadPoolExecutor for parallel processing
        num_workers = min(self.max_workers, len(data))
        results = []

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit tasks to the executor
            futures = [executor.submit(worker.process_item, i, data_item) for i, data_item in enumerate(data)]

            # Gather results as they complete
            for future in as_completed(futures):
                results.append(future.result())

        # Sort results by idx (in-place)
        results.sort(key=lambda x: x[0])

        end_time = time.time()
        print(f"=======Complete {len(data)} reward, takes: {round(end_time - start_time, 2)} seconds=======")
        return results

    def _add_group_statistics_to_results(self, result_dicts):
        """
        为每个样本添加其所属 group 的统计信息
        """
        from collections import defaultdict
        
        # 按 uid 分组并统计
        groups = defaultdict(list)
        group_stats = {}
        
        for i, result_dict in enumerate(result_dicts):
            uid = result_dict.get("uid", "default_group")
            groups[uid].append((i, result_dict))
        
        # print(f"Found {len(groups)} groups with sizes: {[len(group_items) for group_items in groups.values()]}")
        
        # 为每个 group 计算统计信息
        for uid, group_items in groups.items():
            N_A = N_B = N_C = N_D = 0
            
            for result_idx, result_dict in group_items:
                # 获取正确性和工具调用信息
                is_correct = result_dict.get("accuracy_reward", 0) > 0
                has_tool_call = result_dict.get("has_tool_call", False)
                
                # 分类统计并记录类别
                if has_tool_call and is_correct:
                    N_A += 1  # Call Tool 并且 Correct
                    category = "A"
                elif not has_tool_call and not is_correct:
                    N_B += 1  # No Tool 并且 Wrong
                    category = "B"
                elif has_tool_call and not is_correct:
                    N_C += 1  # Call Tool 并且 Wrong
                    category = "C"
                elif not has_tool_call and is_correct:
                    N_D += 1  # No Tool 并且 Correct
                    category = "D"
                
                # 将类别信息存储到 result_dict
                result_dict["intrinsic_category"] = category
            
            group_stats[uid] = {
                "N_A": N_A,
                "N_B": N_B, 
                "N_C": N_C,
                "N_D": N_D
            }
            # print(f"Group {uid}: A={N_A}, B={N_B}, C={N_C}, D={N_D}")
        
        # 将统计信息添加到每个样本的 result_dict 中
        for result_dict in result_dicts:
            uid = result_dict.get("uid", "default_group")
            stats = group_stats[uid]
            
            # 只添加 group 统计信息，不计算 intrinsic reward
            result_dict["group_N_A"] = stats["N_A"]
            result_dict["group_N_B"] = stats["N_B"]
            result_dict["group_N_C"] = stats["N_C"]
            result_dict["group_N_D"] = stats["N_D"]

