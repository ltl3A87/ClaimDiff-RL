import base64
import json
import logging
import os
import random
import re
import time
from typing import Any, Dict, List, Tuple

import requests

from .main import BaseVerifier, Verifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _normalize_text(text: str) -> str:
    if text is None:
        return ""
    return str(text).strip().lower()


def _normalize_error_type(error_type: str) -> str:
    et = _normalize_text(error_type)
    if et in ["na", "none", ""]:
        return "none"
    return et


def _normalize_judgment(judgment: str) -> str:
    j = _normalize_text(judgment)
    j = j.replace("both wrong", "both_wrong")
    j = j.replace("both supported", "both_supported")
    j = j.replace("both-supported", "both_supported")
    j = j.replace("both-wrong", "both_wrong")
    if j in ["a", "b", "both_wrong", "both_supported"]:
        return j
    return "unknown"


def _extract_answer_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


def _has_think_and_answer_blocks(text: str) -> bool:
    if not isinstance(text, str):
        return False
    has_think = re.search(r"<think>.*?</think>", text, re.DOTALL | re.IGNORECASE) is not None
    has_answer = re.search(r"<answer>.*?</answer>", text, re.DOTALL | re.IGNORECASE) is not None
    return has_think and has_answer


def _parse_diff_text_response(text: str) -> Dict[str, Any]:
    """Parse the strict plain-text diff format into a dict."""
    result = {
        "differences": [],
        "overall_winner": "NA",
    }

    if not isinstance(text, str):
        return result

    raw_text = text.strip()
    if raw_text.startswith("{") and "gemini_response" in raw_text:
        try:
            parsed = json.loads(raw_text)
            text = parsed.get("gemini_response", text)
        except Exception:
            pass

    # Remove fenced code blocks if present
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()

    lines = [l.rstrip() for l in text.splitlines()]
    n = len(lines)

    diff_indices = [
        i for i, l in enumerate(lines)
        if re.match(r"\[DIFFERENCE\s+\d+\]", l)
    ]

    for d_idx in diff_indices:
        d = {
            "aspect": "unknown",
            "caption_a_claim": "",
            "caption_b_claim": "",
            "judgment": "unknown",
            "caption_a_error": {
                "error_type": "NA",
                "error_detail": "not_provided",
                "severity_level": "NA",
            },
            "caption_b_error": {
                "error_type": "NA",
                "error_detail": "not_provided",
                "severity_level": "NA",
            },
        }

        cursor = d_idx + 1
        while cursor < n and not lines[cursor].startswith("[DIFFERENCE") \
              and not lines[cursor].startswith("[OVERALL_WINNER]"):
            line = lines[cursor]

            if line.startswith("ASPECT:"):
                d["aspect"] = line[len("ASPECT:"):].strip()
            elif line.startswith("CAPTION_A_CLAIM:"):
                d["caption_a_claim"] = line[len("CAPTION_A_CLAIM:"):].strip()
            elif line.startswith("CAPTION_B_CLAIM:"):
                d["caption_b_claim"] = line[len("CAPTION_B_CLAIM:"):].strip()
            elif line.startswith("JUDGMENT:"):
                d["judgment"] = line[len("JUDGMENT:"):].strip()
            elif line.startswith("CAPTION_A_ERROR:"):
                if cursor + 1 < n and lines[cursor + 1].startswith("ERROR_TYPE:"):
                    d["caption_a_error"]["error_type"] = \
                        lines[cursor + 1][len("ERROR_TYPE:"):].strip()
                if cursor + 2 < n and lines[cursor + 2].startswith("ERROR_DETAIL:"):
                    d["caption_a_error"]["error_detail"] = \
                        lines[cursor + 2][len("ERROR_DETAIL:"):].strip()
                if cursor + 3 < n and lines[cursor + 3].startswith("SEVERITY_LEVEL:"):
                    d["caption_a_error"]["severity_level"] = \
                        lines[cursor + 3][len("SEVERITY_LEVEL:"):].strip()
            elif line.startswith("CAPTION_B_ERROR:"):
                if cursor + 1 < n and lines[cursor + 1].startswith("ERROR_TYPE:"):
                    d["caption_b_error"]["error_type"] = \
                        lines[cursor + 1][len("ERROR_TYPE:"):].strip()
                if cursor + 2 < n and lines[cursor + 2].startswith("ERROR_DETAIL:"):
                    d["caption_b_error"]["error_detail"] = \
                        lines[cursor + 2][len("ERROR_DETAIL:"):].strip()
                if cursor + 3 < n and lines[cursor + 3].startswith("SEVERITY_LEVEL:"):
                    d["caption_b_error"]["severity_level"] = \
                        lines[cursor + 3][len("SEVERITY_LEVEL:"):].strip()

            cursor += 1

        result["differences"].append(d)

    if "[OVERALL_WINNER]" in lines:
        idx = lines.index("[OVERALL_WINNER]") + 1
        if idx < n:
            result["overall_winner"] = lines[idx].strip()

    return result


def _validate_parsed_result(result: Dict[str, Any]) -> Tuple[bool, str]:
    if not result or "differences" not in result:
        return False, "Missing 'differences' field in result"

    differences = result.get("differences", [])
    if len(differences) == 0:
        return False, "No [DIFFERENCE N] blocks found in response"

    overall_winner = result.get("overall_winner", "NA")
    if overall_winner == "NA":
        return False, "Missing [OVERALL_WINNER] section in response"

    for idx, diff in enumerate(differences):
        if not isinstance(diff, dict):
            return False, f"Difference {idx+1} is not a valid dictionary"
        required_fields = [
            "aspect", "caption_a_claim", "caption_b_claim", "judgment",
            "caption_a_error", "caption_b_error"
        ]
        for field in required_fields:
            if field not in diff:
                return False, f"Difference {idx+1} missing required field: {field}"
    return True, ""


def _is_error_none(error_type: str) -> bool:
    return _normalize_error_type(error_type) in ["none", "na"]


def _filter_differences(differences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    filtered = []
    for d in differences:
        a_type = _normalize_error_type(d.get("caption_a_error", {}).get("error_type", ""))
        b_type = _normalize_error_type(d.get("caption_b_error", {}).get("error_type", ""))
        if _is_error_none(a_type) and _is_error_none(b_type):
            continue
        filtered.append(d)
    return filtered


def _format_differences(title: str, differences: List[Dict[str, Any]]) -> str:
    if not differences:
        return f"{title}:\n- (none)\n"
    lines = [f"{title}:"]
    for idx, d in enumerate(differences, start=1):
        lines.append(f"- [{idx}] ASPECT: {d.get('aspect','')}")
        lines.append(f"  CAPTION_A_CLAIM: {d.get('caption_a_claim','')}")
        lines.append(f"  CAPTION_B_CLAIM: {d.get('caption_b_claim','')}")
        lines.append(f"  JUDGMENT: {d.get('judgment','')}")
        lines.append(f"  CAPTION_A_ERROR_TYPE: {d.get('caption_a_error',{}).get('error_type','')}")
        lines.append(f"  CAPTION_B_ERROR_TYPE: {d.get('caption_b_error',{}).get('error_type','')}")
    return "\n".join(lines) + "\n"


DIFF_JUDGE_PROMPT = """You are judging predicted caption differences against ground-truth differences using the image.

You are given:
- Ground-Truth Differences (GT)
- Predicted Differences (PRED)

Rules:
- Only count real differences that are supported by the image where at least one caption claim is wrong.
- A PRED item can match a GT item; multiple PRED items should NOT double-count the same GT.
- If a PRED item is not in GT, you must verify it using the image. If both captions are correct or the difference is not real, it is INVALID.

For each PRED item, output:
- GT_COVERED: TRUE if it matches a GT item NOT already covered by earlier PRED items, else FALSE.
- PRED_VALID: TRUE if it is a real difference with an error in at least one caption, else FALSE.

Return ONLY the following plain text format (no extra text, no markdown):

[PRED DIFFERENCE 1]
GT_COVERED:
<TRUE/FALSE>

PRED_VALID:
<TRUE/FALSE>

NOTES:
<brief reason>

Repeat for all PRED items in order.
"""


def _parse_pred_judge_response(text: str) -> List[Dict[str, str]]:
    if not isinstance(text, str):
        return []
    parsed_text = text.strip()
    if parsed_text.startswith("```"):
        parsed_text = parsed_text.strip("`").strip()
        if parsed_text.lower().startswith("text"):
            parsed_text = parsed_text[4:].strip()

    lines = [l.rstrip() for l in parsed_text.splitlines()]
    n = len(lines)
    pred_indices = [
        i for i, l in enumerate(lines)
        if re.match(r"\[PRED DIFFERENCE\s+\d+\]", l)
    ]
    results = []
    for d_idx in pred_indices:
        d = {
            "gt_covered": "FALSE",
            "pred_valid": "FALSE",
            "notes": "",
        }
        cursor = d_idx + 1
        while cursor < n and not lines[cursor].startswith("[PRED DIFFERENCE"):
            line = lines[cursor]
            if line.startswith("GT_COVERED:") and cursor + 1 < n:
                d["gt_covered"] = lines[cursor + 1].strip().upper()
            elif line.startswith("PRED_VALID:") and cursor + 1 < n:
                d["pred_valid"] = lines[cursor + 1].strip().upper()
            elif line.startswith("NOTES:") and cursor + 1 < n:
                d["notes"] = lines[cursor + 1].strip()
            cursor += 1
        results.append(d)
    return results


@Verifier.register(name="caption_judge")
class CaptionJudgeVerifier(BaseVerifier):
    """Compare a model's diff judgment to Gemini3Pro ground truth with an image-based F1 score."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        image_path = kwargs.get("image_path", None)
        if image_path is None:
            self.image_paths = []
        elif isinstance(image_path, str):
            self.image_paths = [image_path]
        else:
            self.image_paths = list(image_path)

        self.model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
        self.request_timeout_seconds = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "120"))
        self.max_retries = int(os.getenv("MAX_RETRIES", "5"))

        self.reward_scale = float(os.getenv("REWARD_SCALE", "1.0"))
        self.min_reward = float(os.getenv("MIN_REWARD", "0.0"))
        self.max_reward = float(os.getenv("MAX_REWARD", "1.0"))
        self.debug = os.getenv("DEBUG_CAPTION_JUDGE", "false").lower() in ["true", "1", "yes"]
        self.log_sample_rate = float(os.getenv("LOG_SAMPLE_RATE", "0.1"))
        self.save_training_data = os.getenv("SAVE_TRAINING_DATA", "false").lower() in ["true", "1", "yes"]
        self.save_training_data_path = os.getenv("SAVE_TRAINING_DATA_PATH_CAPTION_JUDGE", None)
        self.require_think_block = os.getenv("REQUIRE_THINK_BLOCK", "false").lower() in ["true", "1", "yes"]

        logger.info(f"Caption judge model: {self.model_name}")
        logger.info(f"Caption judge reward scale: {self.reward_scale}")
        logger.info(f"Caption judge reward range: [{self.min_reward}, {self.max_reward}]")
        logger.info(f"Image Paths: {self.image_paths}")
        logger.info(f"Log sample rate: {self.log_sample_rate}")
        logger.info(f"Save training data: {self.save_training_data}")
        logger.info(f"Debug caption judge: {self.debug}")
        logger.info(f"Require think block: {self.require_think_block}")

        if self.save_training_data and not self.save_training_data_path:
            logger.warning("SAVE_TRAINING_DATA enabled but SAVE_TRAINING_DATA_PATH is not set. Disabling.")
            self.save_training_data = False
        if self.save_training_data and self.save_training_data_path:
            os.makedirs(
                os.path.dirname(self.save_training_data_path) if os.path.dirname(self.save_training_data_path) else ".",
                exist_ok=True
            )

    def _encode_image_to_base64(self, image_path: str) -> str:
        from PIL import Image
        from io import BytesIO
        from .utils_caption_diff import _smart_resize

        img = Image.open(image_path)
        width, height = img.size

        factor = 28
        min_pixels = 256 * 28 * 28
        max_pixels = 8192 * 28 * 28

        new_height, new_width = _smart_resize(height, width, factor, min_pixels, max_pixels)

        if (new_width, new_height) != (width, height):
            img = img.resize((int(new_width), int(new_height)), Image.LANCZOS)

        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _send_gemini_request(self, prompt: str, image_paths: List[str], temperature: float = 0.2):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        gemini_api_base = os.getenv("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta/models")
        url = f"{gemini_api_base}/{self.model_name}:generateContent"

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        }

        safety_config = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        generationConfig = {
            "temperature": temperature,
            "topP": 0.95,
        }

        parts = [{"text": prompt}]
        for image_path in image_paths:
            image_base64 = self._encode_image_to_base64(image_path)
            parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": image_base64
                }
            })

        body = {
            "contents": [{"parts": parts, "role": "user"}],
            "safetySettings": safety_config,
            "generationConfig": generationConfig,
        }

        start_time = time.time()
        try:
            response = requests.post(url, headers=headers, json=body, timeout=self.request_timeout_seconds)
            request_time = time.time() - start_time
            if response.status_code == 200:
                response_json = response.json()
                parts = response_json.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                if not parts:
                    return None, None
                text = parts[0].get("text", "")
                if not isinstance(text, str):
                    return None, None
                usage_stats = {
                    "prompt_tokens": response_json["usageMetadata"]["promptTokenCount"],
                    "total_tokens": response_json["usageMetadata"]["totalTokenCount"],
                    "request_time": request_time,
                }
                return text.strip(), usage_stats
            if response.status_code == 429:
                return f"Error 429: {response.text}", {"status": 429, "rate_limited": True}
            return f"Error {response.status_code}: {response.text}", None
        except requests.exceptions.Timeout:
            error_message = f"Request timeout (exceeded {self.request_timeout_seconds} seconds)"
            logger.warning(error_message)
            return error_message, {"timeout": True}
        except requests.exceptions.RequestException as e:
            error_message = f"Request error: {str(e)}"
            logger.warning(error_message)
            return error_message, {"client_error": True}
        except Exception as e:
            error_message = f"Unexpected error: {str(e)}"
            logger.error(error_message, exc_info=True)
            return error_message, None

    def verify_format(self, predict_str: Any) -> float:
        if self.require_think_block and not _has_think_and_answer_blocks(str(predict_str)):
            return 0.0
        parsed = _parse_diff_text_response(_extract_answer_text(str(predict_str) if predict_str is not None else ""))
        valid, _ = _validate_parsed_result(parsed)
        return 1.0 if valid else 0.0

    def verify_accuracy(
        self,
        predict_str: str,
        solution: Any,
        iter: int = None,
        data_index: int = None,
        image_path: str | List[str] = None
    ) -> Dict[str, float]:
        failed_result = {
            "final_score": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "pred_count": 0,
            "gt_count": 0,
            "pred_valid": 0,
            "gt_covered": 0,
            "num_parsing_failures": 1,
        }

        if solution is None:
            logger.warning("caption_judge: solution is missing")
            return failed_result

        gt_text = solution
        if isinstance(solution, dict) and "gemini_response" in solution:
            gt_text = solution.get("gemini_response", "")
        elif isinstance(solution, str) and solution.strip().startswith("{") and "gemini_response" in solution:
            try:
                gt_text = json.loads(solution).get("gemini_response", solution)
            except Exception:
                gt_text = solution

        pred_parsed = _parse_diff_text_response(_extract_answer_text(predict_str or ""))
        gt_parsed = _parse_diff_text_response(gt_text or "")

        if self.debug:
            logger.info(f"caption_judge: gt_parsed={gt_parsed}")
            logger.info(f"caption_judge: pred_parsed={pred_parsed}")

        gt_valid, gt_err = _validate_parsed_result(gt_parsed)
        if not gt_valid:
            logger.warning(f"caption_judge: invalid GT format: {gt_err}")
            return failed_result

        pred_valid, _ = _validate_parsed_result(pred_parsed)
        if not pred_valid:
            return failed_result

        pred_diffs = _filter_differences(pred_parsed.get("differences", []))
        gt_diffs = _filter_differences(gt_parsed.get("differences", []))

        if image_path is not None:
            if isinstance(image_path, str):
                self.image_paths = [image_path]
            else:
                self.image_paths = list(image_path)
        if self.debug:
            logger.info(f"caption_judge: pred_diffs={len(pred_diffs)}, gt_diffs={len(gt_diffs)}")

        if not self.image_paths:
            logger.warning("caption_judge: image_paths is empty")
            return failed_result
        for image_path in self.image_paths:
            if not os.path.exists(image_path):
                logger.warning(f"caption_judge: image_path does not exist: {image_path}")
                return failed_result

        prompt = DIFF_JUDGE_PROMPT + "\n" + _format_differences("Ground-Truth Differences (GT)", gt_diffs) \
            + "\n" + _format_differences("Predicted Differences (PRED)", pred_diffs)
        if self.debug:
            logger.info("caption_judge: prompt_preview=%s", prompt[:800].replace("\n", "\\n"))

        response_text = None
        usage = None
        for attempt in range(self.max_retries):
            temperature = 0.0 if attempt == 0 else 0.2
            response_text, usage = self._send_gemini_request(prompt, self.image_paths, temperature=temperature)
            if usage is None and isinstance(response_text, str) and response_text.startswith("Error 429"):
                time.sleep(1.0 + attempt)
                continue
            if usage is not None:
                break

        if not response_text or usage is None:
            logger.warning("caption_judge: failed to get Gemini response")
            return failed_result

        pred_blocks = _parse_pred_judge_response(response_text)
        if not pred_blocks and pred_diffs:
            logger.warning("caption_judge: failed to parse Gemini response blocks")
            return failed_result
        if self.debug:
            # logger.info("caption_judge: response_preview=%s", response_text[:800].replace("\n", "\\n"))
            logger.info("caption_judge: parsed_blocks=%s", pred_blocks[:3])

        pred_count = len(pred_diffs)
        gt_count = len(gt_diffs)
        gt_covered = sum(1 for b in pred_blocks if b.get("gt_covered", "").upper() == "TRUE")
        pred_valid = sum(1 for b in pred_blocks if b.get("pred_valid", "").upper() == "TRUE")

        precision = (pred_valid / pred_count) if pred_count > 0 else (1.0 if gt_count == 0 else 0.0)
        recall = (gt_covered / gt_count) if gt_count > 0 else 1.0
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        if self.debug:
            logger.info(
                "caption_judge: pred_count=%d gt_count=%d pred_valid=%d gt_covered=%d precision=%.4f recall=%.4f f1=%.4f",
                pred_count, gt_count, pred_valid, gt_covered, precision, recall, f1
            )

        scaled_reward = f1 * self.reward_scale
        scaled_reward = max(0.0, min(1.0, scaled_reward))
        reward = self.min_reward + scaled_reward * (self.max_reward - self.min_reward)
        reward = max(self.min_reward, min(self.max_reward, reward))
        if self.debug:
            logger.info("caption_judge: final_score=%.4f", reward)

        result = {
            "final_score": reward,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "pred_count": pred_count,
            "gt_count": gt_count,
            "pred_valid": pred_valid,
            "gt_covered": gt_covered,
            "num_parsing_failures": 0,
        }
        should_log = random.random() < self.log_sample_rate
        if self.save_training_data and should_log:
            self._save_training_data_to_jsonl(
                meta={
                    "iter": iter,
                    "data_index": data_index,
                    "image_path": self.image_paths or [],
                },
                predict_judgment=predict_str,
                gt_judgment=gt_text,
                prompt=prompt,
                gemini_response=response_text,
                verification_result=result,
            )
        return result

    def _save_training_data_to_jsonl(
        self,
        meta: dict,
        predict_judgment: str,
        gt_judgment: str,
        prompt: str,
        gemini_response: str,
        verification_result: dict
    ):
        if not self.save_training_data or not self.save_training_data_path:
            return
        try:
            from datetime import datetime
            timestamp = datetime.now().isoformat()
            training_record = {
                "timestamp": timestamp,
                "meta": meta,
                "predict_judgment": predict_judgment,
                "ground_truth_judgment": gt_judgment,
                "prompt": prompt,
                "gemini_response": gemini_response,
                "verification_result": verification_result
            }
            import threading
            lock = getattr(self, "_jsonl_lock", None)
            if lock is None:
                lock = threading.Lock()
                self._jsonl_lock = lock
            with lock:
                with open(self.save_training_data_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(training_record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Failed to save training data to JSONL: {e}")

