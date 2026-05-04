"""
B1 Baseline: Holistic Gemini Scalar Reward Verifier

Instead of atomic difference decomposition (DCR-RL), this verifier asks Gemini
to directly score a caption 0-10 based on the image and a reference caption.
Reward = SCORE / 10.

This baseline tests whether DCR-RL's gains come from decomposition or simply
from using a strong multimodal judge.
"""

import base64
import json
import logging
import os
import random
import re
import threading
import time
import math
import asyncio
from collections import OrderedDict
from typing import List, Dict, Any, Tuple
from math import ceil, floor

import requests
import aiohttp

from .main import BaseVerifier, Verifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HOLISTIC_PROMPT = """You are an expert evaluator for long-form image captions.

Given an image, an actor caption, and a reference caption, evaluate the actor caption with respect to the image.

The reference caption is provided only as a helpful comparison anchor. It may be incomplete or contain mistakes. Do not treat it as exhaustive ground truth.

Score the actor caption from 0 to 10 based on:
1. Visual factual correctness.
2. Coverage of salient image content.
3. Correct attributes, counts, spatial relations, OCR/text, and identities.
4. Avoidance of hallucinated objects, attributes, or relations.
5. Clarity and specificity without unnecessary ambiguity or repetition.

Important rules:
- A correct detail in the actor caption should not be penalized merely because it is absent from the reference.
- A detail in the reference should not be rewarded unless it is supported by the image.
- Penalize hallucination more than omission.
- Penalize strategic hedging when the image evidence is clear.
- Do not reward length by itself.
- Do not reward flowery style by itself.

Actor caption:
{actor_caption}

Reference caption:
{reference_caption}

Return exactly this format:
SCORE: <integer from 0 to 10>
RATIONALE: <one short sentence>"""

HOLISTIC_PROMPT_NO_REF = """You are an expert evaluator for long-form image captions.

Given an image and an actor caption, evaluate the actor caption with respect to the image.

Score the actor caption from 0 to 10 based on:
1. Visual factual correctness.
2. Coverage of salient image content.
3. Correct attributes, counts, spatial relations, OCR/text, and identities.
4. Avoidance of hallucinated objects, attributes, or relations.
5. Clarity and specificity without unnecessary ambiguity or repetition.

Important rules:
- Penalize hallucination more than omission.
- Penalize strategic hedging when the image evidence is clear.
- Do not reward length by itself.
- Do not reward flowery style by itself.

Actor caption:
{actor_caption}

Return exactly this format:
SCORE: <integer from 0 to 10>
RATIONALE: <one short sentence>"""


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


def _round_by_factor(value: float, factor: int) -> int:
    return round(value / factor) * factor


def _floor_by_factor(value: float, factor: int) -> int:
    return floor(value / factor) * factor


def _ceil_by_factor(value: float, factor: int) -> int:
    return ceil(value / factor) * factor


def _smart_resize(
    height, width, factor=28, min_pixels=256*28*28, max_pixels=8192*28*28,
):
    h_bar = max(factor, _round_by_factor(height, factor))
    w_bar = max(factor, _round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = _floor_by_factor(height / beta, factor)
        w_bar = _floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil_by_factor(height * beta, factor)
        w_bar = _ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


AMBIGUITY_PATTERN = re.compile(
    r"\b(or|and/or|possibly|appears to|might be|could be|likely)\b|\w+/\w+",
    re.IGNORECASE
)


def count_ambiguity_tokens(text: str) -> int:
    return len(AMBIGUITY_PATTERN.findall(text))


def _parse_holistic_response(text: str) -> dict | None:
    """Parse SCORE: <int> from Gemini response.

    Returns dict with 'score' (int 0-10) and 'rationale' (str), or None on failure.
    """
    if not text:
        return None

    score_match = re.search(r"SCORE:\s*(\d+)", text, re.IGNORECASE)
    if not score_match:
        return None

    score = int(score_match.group(1))
    if score < 0 or score > 10:
        return None

    rationale = ""
    rationale_match = re.search(r"RATIONALE:\s*(.+)", text, re.IGNORECASE)
    if rationale_match:
        rationale = rationale_match.group(1).strip()

    return {"score": score, "rationale": rationale}


@Verifier.register(name="gemini_caption_holistic")
class GeminiCaptionHolisticVerifier(BaseVerifier):

    def __init__(self, **kwargs):
        image_path = kwargs.get("image_path", None)
        if image_path is None:
            self.image_paths = []
        elif isinstance(image_path, str):
            self.image_paths = [image_path]
        else:
            self.image_paths = list(image_path)

        self.model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
        self.max_concurrent_requests = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))
        self.request_timeout_seconds = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "120"))
        self.max_retries = int(os.getenv("MAX_RETRIES", "8"))

        self.use_image_base64_cache = os.getenv("USE_IMAGE_BASE64_CACHE", "false").lower() in ["true", "1", "yes"]
        self.image_base64_cache_size = int(os.getenv("IMAGE_BASE64_CACHE_SIZE", "512"))
        self._image_base64_cache = OrderedDict()
        self._image_cache_lock = threading.Lock()

        self.min_reward = float(os.getenv("MIN_REWARD", "0.0"))
        self.max_reward = float(os.getenv("MAX_REWARD", "1.0"))
        self.log_sample_rate = float(os.getenv("LOG_SAMPLE_RATE", "0.1"))
        self.require_think_block = os.getenv("REQUIRE_THINK_BLOCK", "false").lower() in ["true", "1", "yes"]
        self.no_reference = os.getenv("HOLISTIC_NO_REFERENCE", "false").lower() in ["true", "1", "yes"]

        self.save_training_data = os.getenv("SAVE_TRAINING_DATA", "false").lower() in ["true", "1", "yes"]
        save_training_data_path_raw = os.getenv("SAVE_TRAINING_DATA_PATH", None)
        if self.save_training_data:
            if not save_training_data_path_raw:
                logger.warning("SAVE_TRAINING_DATA is enabled but path not set. Disabling.")
                self.save_training_data = False
                self.save_training_data_path = None
            else:
                self.save_training_data_path = save_training_data_path_raw
                os.makedirs(os.path.dirname(self.save_training_data_path) if os.path.dirname(self.save_training_data_path) else ".", exist_ok=True)
                logger.info(f"Training data will be saved to: {self.save_training_data_path}")
        else:
            self.save_training_data_path = None

        logger.info(f"[HolisticVerifier] Reward range: [{self.min_reward}, {self.max_reward}]")
        logger.info(f"[HolisticVerifier] No-reference mode: {self.no_reference}")
        logger.info(f"[HolisticVerifier] Log sample rate: {self.log_sample_rate}")
        logger.info(f"[HolisticVerifier] Image Paths: {self.image_paths}")

    # ---- image encoding (same as GeminiCaptionDiffVerifier) ----

    def _encode_image_to_base64(self, image_path: str) -> str:
        from PIL import Image
        from io import BytesIO

        if not image_path:
            return ""

        cache_key = None
        if self.use_image_base64_cache:
            try:
                mtime = os.path.getmtime(image_path)
                cache_key = (image_path, mtime)
                with self._image_cache_lock:
                    if cache_key in self._image_base64_cache:
                        cached = self._image_base64_cache.pop(cache_key)
                        self._image_base64_cache[cache_key] = cached
                        return cached
            except Exception:
                cache_key = None

        img = Image.open(image_path)
        width, height = img.size
        factor = 28
        min_pixels = 256 * 28 * 28
        max_pixels = 8192 * 28 * 28
        new_height, new_width = _smart_resize(height, width, factor, min_pixels, max_pixels)
        if (new_width, new_height) != (width, height):
            img = img.resize((int(new_width), int(new_height)), Image.LANCZOS)
        buffer = BytesIO()
        img.save(buffer, format='JPEG')
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

        if self.use_image_base64_cache and cache_key is not None:
            with self._image_cache_lock:
                self._image_base64_cache[cache_key] = encoded
                if len(self._image_base64_cache) > self.image_base64_cache_size:
                    self._image_base64_cache.popitem(last=False)

        return encoded

    # ---- Gemini API (sync) ----

    def _send_gemini_request(self, prompt: str, image_paths: list, temperature: float = 0.3):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        gemini_api_base = os.getenv("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta/models")
        url = f"{gemini_api_base}/{self.model_name}:generateContent"

        headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
        safety_config = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        generationConfig = {"temperature": temperature, "topP": 0.95}

        parts = [{"text": prompt}]
        for ip in image_paths:
            image_base64 = self._encode_image_to_base64(ip)
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": image_base64}})

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
                resp_parts = response_json.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                if not resp_parts:
                    return None, None
                text = resp_parts[0].get("text", "")
                if not isinstance(text, str):
                    return None, None
                usage_stats = {
                    "prompt_tokens": response_json["usageMetadata"]["promptTokenCount"],
                    "total_tokens": response_json["usageMetadata"]["totalTokenCount"],
                    "request_time": request_time,
                }
                return text.strip(), usage_stats
            elif response.status_code == 429:
                return f"Error 429: {response.text}", {"status": 429, "rate_limited": True}
            else:
                return f"Error {response.status_code}: {response.text}", None
        except requests.exceptions.Timeout:
            msg = f"Request timeout (exceeded {self.request_timeout_seconds}s)"
            logger.warning(msg)
            return msg, {"timeout": True}
        except requests.exceptions.RequestException as e:
            msg = f"Request error: {e}"
            logger.warning(msg)
            return msg, {"client_error": True}
        except Exception as e:
            msg = f"Unexpected error: {e}"
            logger.error(msg, exc_info=True)
            return msg, None

    # ---- Gemini API (async) ----

    async def _send_gemini_request_async(self, prompt: str, image_paths: list, temperature: float = 0.3, session: aiohttp.ClientSession = None):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        gemini_api_base = os.getenv("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta/models")
        url = f"{gemini_api_base}/{self.model_name}:generateContent"

        headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
        safety_config = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        generationConfig = {"temperature": temperature, "topP": 0.95}

        parts = [{"text": prompt}]
        for ip in image_paths:
            image_base64 = self._encode_image_to_base64(ip)
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": image_base64}})

        body = {
            "contents": [{"parts": parts, "role": "user"}],
            "safetySettings": safety_config,
            "generationConfig": generationConfig,
        }

        start_time = time.time()
        close_session = False
        if session is None:
            session = aiohttp.ClientSession()
            close_session = True

        try:
            async with session.post(
                url, headers=headers, json=body,
                timeout=aiohttp.ClientTimeout(total=self.request_timeout_seconds),
            ) as response:
                request_time = time.time() - start_time
                if response.status == 200:
                    response_json = await response.json()
                    resp_parts = response_json.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                    if not resp_parts:
                        return None, None
                    text = resp_parts[0].get("text", "")
                    if not isinstance(text, str):
                        return None, None
                    usage_stats = {
                        "prompt_tokens": response_json["usageMetadata"]["promptTokenCount"],
                        "total_tokens": response_json["usageMetadata"]["totalTokenCount"],
                        "request_time": request_time,
                    }
                    return text.strip(), usage_stats
                elif response.status == 429:
                    error_text = await response.text()
                    return f"Error 429: {error_text}", {"status": 429, "rate_limited": True}
                else:
                    error_text = await response.text()
                    return f"Error {response.status}: {error_text}", None
        except asyncio.TimeoutError:
            msg = f"Request timeout (exceeded {self.request_timeout_seconds}s)"
            logger.warning(msg)
            return msg, {"timeout": True}
        except aiohttp.ClientError as e:
            msg = f"Request error: {e}"
            logger.warning(msg)
            return msg, {"client_error": True}
        except Exception as e:
            msg = f"Unexpected error: {e}"
            logger.error(msg, exc_info=True)
            return msg, None
        finally:
            if close_session:
                await session.close()

    # ---- training data logging ----

    def _save_training_data_to_jsonl(self, meta, actor_caption, prompt, gemini_response, verification_result):
        if not self.save_training_data:
            return
        try:
            from datetime import datetime
            record = {
                "timestamp": datetime.now().isoformat(),
                "meta": meta,
                "actor_caption": actor_caption,
                "prompt": prompt,
                "gemini_response": gemini_response,
                "verification_result": verification_result,
            }
            lock = getattr(self, '_jsonl_lock', None)
            if lock is None:
                lock = threading.Lock()
                self._jsonl_lock = lock
            with lock:
                with open(self.save_training_data_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.warning(f"Failed to save training data: {e}")

    # ---- format verification ----

    def verify_format(self, predict_str: str) -> float:
        if self.require_think_block:
            return 1.0 if _has_think_and_answer_blocks(predict_str) else 0.0
        return 1.0 if _extract_answer_text(predict_str) else 0.0

    # ---- accuracy verification (main entry) ----

    def verify_accuracy(
        self, predict_caption: str, solution_caption: str | None = None,
        iter: int = None, data_index: int = None, image_path: str | List[str] = None
    ) -> dict:
        """Holistic scalar reward: ask Gemini to score the caption 0-10, return score/10."""
        try:
            predict_caption = _extract_answer_text(predict_caption)
            try:
                loop = asyncio.get_running_loop()
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self._verify_accuracy_async_internal(self, predict_caption, solution_caption, None, iter, data_index, image_path)
                    )
                    return future.result(timeout=600)
            except RuntimeError:
                return asyncio.run(
                    self._verify_accuracy_async_internal(self, predict_caption, solution_caption, None, iter, data_index, image_path)
                )
        except Exception as e:
            logger.warning(f"Async verify_accuracy failed, falling back to sync: {e}")
            return self._verify_accuracy_sync_fallback(predict_caption, solution_caption, iter, data_index, image_path)

    def _verify_accuracy_sync_fallback(
        self, predict_caption: str, solution_caption: str | None = None,
        iter: int = None, data_index: int = None, image_path: str | List[str] = None
    ) -> dict:
        failed_result = {"holistic_score": 0, "final_score": 0.0, "num_parsing_failures": 0}

        predict_caption = _extract_answer_text(predict_caption)
        if not self.no_reference and (solution_caption is None or not solution_caption.strip()):
            logger.warning("solution_caption is required but not provided")
            return failed_result

        should_log = random.random() < self.log_sample_rate

        if not self.image_paths:
            logger.warning("image_paths is empty")
            return failed_result
        for ip in self.image_paths:
            if not os.path.exists(ip):
                logger.warning(f"image_path does not exist: {ip}")
                return failed_result

        if self.no_reference:
            prompt = HOLISTIC_PROMPT_NO_REF.format(actor_caption=predict_caption)
        else:
            prompt = HOLISTIC_PROMPT.format(actor_caption=predict_caption, reference_caption=solution_caption)

        num_parsing_failures = 0
        consecutive_rate_limits = 0

        for attempt in range(self.max_retries):
            temperature = 0.0 if attempt == 0 else random.uniform(0, 1)

            response_text, usage_stats = self._send_gemini_request(prompt, self.image_paths, temperature)

            if usage_stats is not None and isinstance(usage_stats, dict) and usage_stats.get("rate_limited"):
                consecutive_rate_limits += 1
                wait_time = min((0.5 + random.uniform(0, 1.0)) * min(consecutive_rate_limits, 5), 10.0)
                logger.warning(f"Rate limit (attempt {attempt+1}), waiting {wait_time:.2f}s")
                time.sleep(wait_time)
                continue
            else:
                consecutive_rate_limits = 0

            if usage_stats is not None and isinstance(usage_stats, dict) and usage_stats.get("timeout"):
                time.sleep(3.0 + random.uniform(0, 2.0))
                continue

            if usage_stats is None or response_text is None:
                logger.warning(f"Gemini request failed (attempt {attempt+1}): {response_text}")
                if attempt < self.max_retries - 1:
                    time.sleep(0.2 + random.uniform(0, 0.2))
                continue

            parsed = _parse_holistic_response(response_text)
            if parsed is None:
                num_parsing_failures += 1
                logger.warning(f"Parse failure (attempt {attempt+1}): {response_text[:200]}")
                if attempt < self.max_retries - 1:
                    time.sleep(0.3 + random.uniform(0, 0.2))
                continue

            score = parsed["score"]
            reward = score / 10.0
            reward = self.min_reward + reward * (self.max_reward - self.min_reward)
            reward = max(self.min_reward, min(self.max_reward, reward))

            ambiguity_count = count_ambiguity_tokens(predict_caption)

            if should_log:
                logger.info(f"[Holistic] score={score}/10, reward={reward:.4f}, ambiguity_count={ambiguity_count}, rationale={parsed['rationale']}")

            result_dict = {
                "holistic_score": score,
                "final_score": reward,
                "num_parsing_failures": num_parsing_failures,
                "ambiguity_count": ambiguity_count,
            }

            if self.save_training_data:
                ip_save = image_path if image_path is not None else self.image_paths
                if isinstance(ip_save, str):
                    ip_save = [ip_save]
                elif ip_save is None:
                    ip_save = []
                self._save_training_data_to_jsonl(
                    meta={"iter": iter, "data_index": data_index, "image_path": ip_save},
                    actor_caption=predict_caption, prompt=prompt,
                    gemini_response=response_text, verification_result=result_dict,
                )

            return result_dict

        logger.error(f"All {self.max_retries} attempts failed")
        failed_result["num_parsing_failures"] = num_parsing_failures
        return failed_result

    # ---- async internal ----

    @staticmethod
    async def _verify_accuracy_async_internal(
        verifier: 'GeminiCaptionHolisticVerifier',
        predict_caption: str, solution_caption: str,
        session: aiohttp.ClientSession = None,
        iter: int = None, data_index: int = None, image_path: str | List[str] = None
    ) -> dict:
        failed_result = {"holistic_score": 0, "final_score": 0.0, "num_parsing_failures": 0}

        predict_caption = _extract_answer_text(predict_caption)
        if not verifier.no_reference and (solution_caption is None or not solution_caption.strip()):
            logger.warning("solution_caption is required but not provided")
            return failed_result

        should_log = random.random() < verifier.log_sample_rate

        if not verifier.image_paths:
            logger.warning("image_paths is empty")
            return failed_result
        for ip in verifier.image_paths:
            if not os.path.exists(ip):
                logger.warning(f"image_path does not exist: {ip}")
                return failed_result

        if verifier.no_reference:
            prompt = HOLISTIC_PROMPT_NO_REF.format(actor_caption=predict_caption)
        else:
            prompt = HOLISTIC_PROMPT.format(actor_caption=predict_caption, reference_caption=solution_caption)

        num_parsing_failures = 0
        consecutive_rate_limits = 0

        for attempt in range(verifier.max_retries):
            temperature = 0.0 if attempt == 0 else random.uniform(0, 1)

            response_text, usage_stats = await verifier._send_gemini_request_async(
                prompt, verifier.image_paths, temperature, session
            )

            if usage_stats is not None and isinstance(usage_stats, dict) and usage_stats.get("rate_limited"):
                consecutive_rate_limits += 1
                wait_time = min((0.5 + random.uniform(0, 1.0)) * min(consecutive_rate_limits, 5), 10.0)
                logger.warning(f"Rate limit (attempt {attempt+1}), waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
                continue
            else:
                consecutive_rate_limits = 0

            if usage_stats is not None and isinstance(usage_stats, dict) and usage_stats.get("timeout"):
                await asyncio.sleep(3.0 + random.uniform(0, 2.0))
                continue

            if usage_stats is None or response_text is None:
                logger.warning(f"Gemini request failed (attempt {attempt+1}): {response_text}")
                if attempt < verifier.max_retries - 1:
                    await asyncio.sleep(0.2 + random.uniform(0, 0.2))
                continue

            parsed = _parse_holistic_response(response_text)
            if parsed is None:
                num_parsing_failures += 1
                logger.warning(f"Parse failure (attempt {attempt+1}): {response_text[:200]}")
                if attempt < verifier.max_retries - 1:
                    await asyncio.sleep(0.3 + random.uniform(0, 0.2))
                continue

            score = parsed["score"]
            reward = score / 10.0
            reward = verifier.min_reward + reward * (verifier.max_reward - verifier.min_reward)
            reward = max(verifier.min_reward, min(verifier.max_reward, reward))

            ambiguity_count = count_ambiguity_tokens(predict_caption)

            if should_log:
                logger.info(f"[Holistic] score={score}/10, reward={reward:.4f}, ambiguity_count={ambiguity_count}, rationale={parsed['rationale']}")

            result_dict = {
                "holistic_score": score,
                "final_score": reward,
                "num_parsing_failures": num_parsing_failures,
                "ambiguity_count": ambiguity_count,
            }

            if verifier.save_training_data:
                ip_save = image_path if image_path is not None else verifier.image_paths
                if isinstance(ip_save, str):
                    ip_save = [ip_save]
                elif ip_save is None:
                    ip_save = []
                verifier._save_training_data_to_jsonl(
                    meta={"iter": iter, "data_index": data_index, "image_path": ip_save},
                    actor_caption=predict_caption, prompt=prompt,
                    gemini_response=response_text, verification_result=result_dict,
                )

            return result_dict

        logger.error(f"All {verifier.max_retries} attempts failed")
        failed_result["num_parsing_failures"] = num_parsing_failures
        return failed_result

    # ---- batch processing ----

    @staticmethod
    async def batch_verify_accuracy_async(
        requests_data: List[dict],
        max_concurrent: int = 10,
    ) -> List[dict]:
        """Process multiple holistic verification requests concurrently.

        Args:
            requests_data: List of dicts with keys:
                - image_path: str or list
                - predict_caption: str
                - solution_caption: str
                - iter, data_index: optional
                - kwargs: optional dict of extra constructor kwargs
            max_concurrent: Max concurrent Gemini requests
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        connector = aiohttp.TCPConnector(limit=max_concurrent * 2)
        results = [None] * len(requests_data)

        async def _process(idx, req, session):
            async with semaphore:
                try:
                    kwargs = req.get("kwargs", {})
                    kwargs["image_path"] = req["image_path"]
                    verifier = GeminiCaptionHolisticVerifier(**kwargs)
                    result = await GeminiCaptionHolisticVerifier._verify_accuracy_async_internal(
                        verifier,
                        req["predict_caption"],
                        req["solution_caption"],
                        session,
                        req.get("iter"),
                        req.get("data_index"),
                        req.get("image_path"),
                    )
                    results[idx] = result
                except Exception as e:
                    logger.error(f"Batch item {idx} failed: {e}")
                    results[idx] = {"holistic_score": 0, "final_score": 0.0, "num_parsing_failures": 0}

        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [_process(i, req, session) for i, req in enumerate(requests_data)]
            await asyncio.gather(*tasks)

        return results

    @staticmethod
    def batch_verify_accuracy(requests_data: List[dict], max_concurrent: int = 10) -> List[dict]:
        return asyncio.run(
            GeminiCaptionHolisticVerifier.batch_verify_accuracy_async(requests_data, max_concurrent)
        )
