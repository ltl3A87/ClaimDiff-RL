import base64
import json
import logging
import os
import random
import threading
from collections import OrderedDict
import re
import time
import math
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple

import requests
import aiohttp

from .main import BaseVerifier, Verifier
from math import ceil, floor
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DIFF_PROMPT = """You are comparing two image captions based strictly on the image.

Your task:
- Identify one or more concrete differences between Caption A and Caption B.
- For each difference, judge which caption is supported by the image.
- Omission errors should be treated as errors, not just omissions.
- Describe errors for each caption if present.
- Assign an overall winner.

ERROR TYPE GUIDELINES:
- ERROR_TYPE should be specific and descriptive, not generic.
- Use a compound form when possible: <dimension>_<error_nature>.
- Avoid vague terms like "hallucination" or "omission" alone.
- Prefer fine-grained types such as:
  color_hallucination,
  layout_omission,
  detail_omission,
  text_omission,
  text_truncation_error,
  count_mismatch,
  spatial_relation_error,
  style_misinterpretation,
  identity_hallucination,
  context_overreach,
  lighting_misinterpretation,
  camera_angle_misinterpretation,
  repetition_error,
  etc.
- If multiple issues exist, choose the most dominant one.
- If the image clearly supports a specific visual attribute (such as a definite color, object identity,
  spatial relation, lighting condition, or camera angle), then an ambiguous or disjunctive claim
  (e.g., "A or B", "possibly A", "might be A") should be treated as an error.
- In such cases, prefer assigning an appropriate ERROR_TYPE rather than considering the claim acceptable.
- If a caption describes the same content multiple times (including paraphrases) without adding new
  information, treat it as an error (ERROR_TYPE: repetition_error). Repetition counts as an error
  even if the repeated content is correct.

IMPORTANT FORMAT RULES (CRITICAL - MUST FOLLOW EXACTLY):
- You MUST use EXACTLY the format below. Any deviation will cause parsing to fail.
- You MUST use the exact headers: [DIFFERENCE 1], ASPECT:, CAPTION_A_CLAIM:, CAPTION_B_CLAIM:, JUDGMENT:, CAPTION_A_ERROR:, ERROR_TYPE:, ERROR_DETAIL:, CAPTION_B_ERROR:, [OVERALL_WINNER]
- Do NOT add any additional headers, sections, or formatting.
- Do NOT use markdown, code blocks, or any other formatting.
- Use plain text only with exact header matching (case-sensitive).
- Repeat [DIFFERENCE N] blocks if multiple differences exist (use [DIFFERENCE 1], [DIFFERENCE 2], etc. in order).
- If a caption has no error, use:
  ERROR_TYPE: NONE
  ERROR_DETAIL: No error.
- You MUST include the [OVERALL_WINNER] section at the end.

OUTPUT FORMAT:

[DIFFERENCE 1]
ASPECT:
CAPTION_A_CLAIM:
CAPTION_B_CLAIM:
JUDGMENT: <A|B|both_wrong|both_supported>

CAPTION_A_ERROR:
ERROR_TYPE:
ERROR_DETAIL:

CAPTION_B_ERROR:
ERROR_TYPE:
ERROR_DETAIL:

[OVERALL_WINNER]
<A|B|Tie>

Caption A:
{caption_a}

Caption B:
{caption_b}
"""

DIFF_PROMPT_V5 = """You are comparing two image captions based strictly on the image.

Your task:
- Identify one or more concrete differences between Caption A and Caption B.
- For each difference, judge which caption is supported by the image.
- Describe errors for each caption if present.
- Assign an overall winner.
- For each caption error, assign a SEVERITY_LEVEL based on image importance.

SEVERITY LEVEL GUIDELINES (IMAGE-BASED):
- Assign severity according to how critical the error is for describing the image content.
- Severity must be based on what is visible and how central the information is.
- Use these levels:
  SEVERITY_LEVEL: 3 (major) — Error affects the primary subject, key objects, main action, or core identity.
  SEVERITY_LEVEL: 2 (medium) — Error affects important but secondary details, context, or notable attributes.
  SEVERITY_LEVEL: 1 (minor) — Error affects minor details, stylistic nuance, or peripheral elements.
- If ERROR_TYPE is NONE, set SEVERITY_LEVEL: NA.

ERROR TYPE GUIDELINES:
- ERROR_TYPE should be specific and descriptive, not generic.
- Use a compound form when possible: <dimension>_<error_nature>.
- Avoid vague terms like "hallucination" or "omission" alone.
- Prefer fine-grained types such as:
  color_hallucination,
  layout_omission,
  text_omission,
  text_truncation_error,
  count_mismatch,
  spatial_relation_error,
  style_misinterpretation,
  identity_hallucination,
  context_overreach,
  lighting_misinterpretation,
  camera_angle_misinterpretation,
  etc.
- If multiple issues exist, choose the most dominant one.
- If the image clearly supports a specific visual attribute (such as a definite color, object identity,
  spatial relation, lighting condition, recognizable text, or camera angle), then an ambiguous or disjunctive claim
  (e.g., "A or B", "possibly A", "might be A", several Chinese charaters, etc.) should be treated as an error.
- In such cases, prefer assigning an appropriate ERROR_TYPE rather than considering the claim acceptable.

IMPORTANT FORMAT RULES (CRITICAL - MUST FOLLOW EXACTLY):
- You MUST use EXACTLY the format below. Any deviation will cause parsing to fail.
- You MUST use the exact headers: [DIFFERENCE 1], ASPECT:, CAPTION_A_CLAIM:, CAPTION_B_CLAIM:, JUDGMENT:, CAPTION_A_ERROR:, ERROR_TYPE:, ERROR_DETAIL:, SEVERITY_LEVEL:, CAPTION_B_ERROR:, [OVERALL_WINNER]
- Do NOT add any additional headers, sections, or formatting.
- Do NOT use markdown, code blocks, or any other formatting.
- Use plain text only with exact header matching (case-sensitive).
- Repeat [DIFFERENCE N] blocks if multiple differences exist (use [DIFFERENCE 1], [DIFFERENCE 2], etc. in order).
- If a caption has no error, use:
  ERROR_TYPE: NONE
  ERROR_DETAIL: No error.
  SEVERITY_LEVEL: NA
- You MUST include the [OVERALL_WINNER] section at the end.

OUTPUT FORMAT:

[DIFFERENCE 1]
ASPECT:
CAPTION_A_CLAIM:
CAPTION_B_CLAIM:
JUDGMENT: <A|B|both_wrong|both_supported>

CAPTION_A_ERROR:
ERROR_TYPE:
ERROR_DETAIL:
SEVERITY_LEVEL: <1|2|3|NA>

CAPTION_B_ERROR:
ERROR_TYPE:
ERROR_DETAIL:
SEVERITY_LEVEL: <1|2|3|NA>

[OVERALL_WINNER]
<A|B|Tie>

Caption A:
{caption_a}

Caption B:
{caption_b}
"""

DIFF_PROMPT_V6 = """You are comparing two image captions based strictly on the image.

Your task:
- Identify one or more concrete differences between Caption A and Caption B.
- For each difference, judge which caption is supported by the image.
- Describe errors for each caption if present.
- Assign an overall winner.
- For each caption error, assign a SEVERITY_LEVEL based on the ERROR_TYPE (model-judged, not post-mapped).

SEVERITY LEVEL GUIDELINES:
- Priority order: hallucination/incorrect claim > detail omission/incomplete info > style/aesthetic error.
- Severity 3 (major): The caption asserts something false or nonexistent, wrong identity, wrong count,
  wrong relation, or fabricated content.
- Severity 2 (medium): The caption misses, truncates, or incompletely describes factual details that
  are present but secondary.
- Severity 1 (minor): The caption mainly errs in style, aesthetic impression, tone, or subjective
  interpretation without factual contradictions.
- If ERROR_TYPE is NONE, set SEVERITY_LEVEL: NA.

AMBIGUITY HANDLING RULES:
- If the image clearly supports a specific visual attribute (such as a definite color, object identity,
  spatial relation, lighting condition, or camera angle), then an ambiguous or disjunctive claim
  (e.g., "A or B", "possibly A", "might be A") should be treated as an error.
- In such cases, prefer assigning an appropriate ERROR_TYPE (e.g., color_hallucination,
  identity_hallucination, or <dimension>_misinterpretation) rather than considering the claim acceptable.
- If the image itself is genuinely ambiguous or lacks sufficient visual evidence, ambiguity may be acceptable
  and should not be penalized.
- Do not reward ambiguity as a safe alternative when clear visual evidence is present.

ERROR TYPE GUIDELINES:
- ERROR_TYPE should be specific and descriptive, not generic.
- Use a compound form when possible: <dimension>_<error_nature>.
- Avoid vague terms like "hallucination" or "omission" alone.
- Prefer fine-grained types such as:
  color_hallucination,
  layout_omission,
  text_omission,
  text_truncation_error,
  count_mismatch,
  spatial_relation_error,
  style_misinterpretation,
  identity_hallucination,
  context_overreach,
  lighting_misinterpretation,
  camera_angle_misinterpretation,
  etc.
- If multiple issues exist, choose the most dominant one.
- If the image clearly supports a specific visual attribute (such as a definite color, object identity,
  spatial relation, lighting condition, recognizable text, or camera angle), then an ambiguous or disjunctive claim
  (e.g., "A or B", "possibly A", "might be A", several Chinese charaters, etc.) should be treated as an error.
- In such cases, prefer assigning an appropriate ERROR_TYPE rather than considering the claim acceptable.

IMPORTANT FORMAT RULES (CRITICAL - MUST FOLLOW EXACTLY):
- You MUST use EXACTLY the format below. Any deviation will cause parsing to fail.
- You MUST use the exact headers: [DIFFERENCE 1], ASPECT:, CAPTION_A_CLAIM:, CAPTION_B_CLAIM:, JUDGMENT:, CAPTION_A_ERROR:, ERROR_TYPE:, ERROR_DETAIL:, SEVERITY_LEVEL:, CAPTION_B_ERROR:, [OVERALL_WINNER]
- Do NOT add any additional headers, sections, or formatting.
- Do NOT use markdown, code blocks, or any other formatting.
- Use plain text only with exact header matching (case-sensitive).
- Repeat [DIFFERENCE N] blocks if multiple differences exist (use [DIFFERENCE 1], [DIFFERENCE 2], etc. in order).
- If a caption has no error, use:
  ERROR_TYPE: NONE
  ERROR_DETAIL: No error.
  SEVERITY_LEVEL: NA
- You MUST include the [OVERALL_WINNER] section at the end.

OUTPUT FORMAT:

[DIFFERENCE 1]
ASPECT:
CAPTION_A_CLAIM:
CAPTION_B_CLAIM:
JUDGMENT: <A|B|both_wrong|both_supported>

CAPTION_A_ERROR:
ERROR_TYPE:
ERROR_DETAIL:
SEVERITY_LEVEL: <1|2|3|NA>

CAPTION_B_ERROR:
ERROR_TYPE:
ERROR_DETAIL:
SEVERITY_LEVEL: <1|2|3|NA>

[OVERALL_WINNER]
<A|B|Tie>

Caption A:
{caption_a}

Caption B:
{caption_b}
"""

DIFF_PROMPT_V2 = """You are comparing two image captions based strictly on the image.

Your task:
- Identify one or more concrete differences between Caption A and Caption B.
- Differences may involve not only objects and actions, but also visual and photographic attributes,
  such as style, color, lighting, camera angle, perspective, or composition, when they are clearly
  visible in the image.
- For each difference, judge which caption is supported by the image.
- Describe errors for each caption if present.
- Assign an overall winner.

IMPORTANT:
- Consider visual style, artistic style, color tone, lighting conditions, camera angle,
  shot scale, and perspective as valid aspects for comparison if the captions make different claims
  about these attributes and the image provides sufficient visual evidence.
- Only identify such differences when they are clearly supported by the image; do not speculate
  or infer stylistic intent beyond visible evidence.

ERROR TYPE GUIDELINES:
- ERROR_TYPE should be specific and descriptive, not generic.
- Use a compound form when possible: <dimension>_<error_nature>.
- Avoid vague terms like "hallucination" or "omission" alone.
- Prefer fine-grained types such as:
  color_hallucination,
  layout_omission,
  text_truncation_error,
  count_mismatch,
  spatial_relation_error,
  style_misinterpretation,
  lighting_misinterpretation,
  camera_angle_misinterpretation,
  shot_scale_misinterpretation,
  identity_hallucination,
  context_overreach.
- If multiple issues exist, choose the most dominant one.

AMBIGUITY HANDLING RULES:
- If the image clearly supports a specific visual attribute (such as a definite color, object identity,
  spatial relation, lighting condition, or camera angle), then an ambiguous or disjunctive claim
  (e.g., "A or B", "possibly A", "might be A") should be treated as an error.
- In such cases, prefer assigning an appropriate ERROR_TYPE (e.g., color_hallucination,
  identity_hallucination, or <dimension>_misinterpretation) rather than considering the claim acceptable.
- If the image itself is genuinely ambiguous or lacks sufficient visual evidence, ambiguity may be acceptable
  and should not be penalized.
- Do not reward ambiguity as a safe alternative when clear visual evidence is present.

IMPORTANT FORMAT RULES (CRITICAL - MUST FOLLOW EXACTLY):
- You MUST use EXACTLY the format below. Any deviation will cause parsing to fail.
- You MUST use the exact headers: [DIFFERENCE 1], ASPECT:, CAPTION_A_CLAIM:, CAPTION_B_CLAIM:, JUDGMENT:, CAPTION_A_ERROR:, ERROR_TYPE:, ERROR_DETAIL:, CAPTION_B_ERROR:, [OVERALL_WINNER]
- Do NOT add any additional headers, sections, or formatting.
- Do NOT use markdown, code blocks, or any other formatting.
- Use plain text only with exact header matching (case-sensitive).
- Repeat [DIFFERENCE N] blocks if multiple differences exist (use [DIFFERENCE 1], [DIFFERENCE 2], etc. in order).
- If a caption has no error, use:
  ERROR_TYPE: NONE
  ERROR_DETAIL: No error.
- You MUST include the [OVERALL_WINNER] section at the end.

OUTPUT FORMAT:

[DIFFERENCE 1]
ASPECT:
CAPTION_A_CLAIM:
CAPTION_B_CLAIM:
JUDGMENT: <A|B|both_wrong|both_supported>

CAPTION_A_ERROR:
ERROR_TYPE:
ERROR_DETAIL:

CAPTION_B_ERROR:
ERROR_TYPE:
ERROR_DETAIL:

[OVERALL_WINNER]
<A|B|Tie>

Caption A:
{caption_a}

Caption B:
{caption_b}
"""

DIFF_PROMPT_V4 = """You are comparing two image captions based strictly on the image.

Your task:
- Identify one or more concrete differences between Caption A and Caption B.
- Differences may involve not only objects and actions, but also visual and photographic attributes,
  such as style, color, lighting, camera angle, perspective, or composition, when they are clearly
  visible in the image.
- For each difference, judge which caption is supported by the image.
- Describe errors for each caption if present.
- Assign an overall winner.

AMBIGUITY HANDLING RULES (CRITICAL):
- If the image clearly supports a specific visual attribute (such as a definite color, object identity,
  spatial relation, lighting condition, or camera angle), then an ambiguous or disjunctive claim
  (e.g., "A or B", "possibly A", "might be A") should be treated as an error.
- In such cases, prefer assigning an appropriate ERROR_TYPE (e.g., color_hallucination,
  identity_hallucination, or <dimension>_misinterpretation) rather than considering the claim acceptable.
- If the image itself is genuinely ambiguous or lacks sufficient visual evidence, ambiguity may be acceptable
  and should not be penalized.
- Do not reward ambiguity as a safe alternative when clear visual evidence is present.
- SPECIFICITY REQUIREMENTS: If one caption mentions a specific IP/brand name (e.g., "Coca-Cola", "Nike") or
  specific text content (e.g., "the text says 'Welcome'") that is clearly visible, and another caption only
  describes it generically (e.g., "a soft drink", "several characters"), this is a DIFFERENCE and should be
  attributed as identity_omission, detail_omission, or text_omission for the less specific caption, even if
  the generic description is technically correct.

ERROR TYPE GUIDELINES:
- ERROR_TYPE should be specific and descriptive, not generic.
- Use a compound form when possible: <dimension>_<error_nature>.
- For omissions, use types such as: object_omission, detail_omission, attribute_omission, layout_omission,
  color_omission, spatial_relation_omission, etc.
- For unverifiable statements, use types such as: unverifiable_claim, context_overreach, speculation_error,
  unverifiable_detail, etc.
- For other errors, prefer fine-grained types such as:
  color_hallucination,
  text_truncation_error,
  text_omission,
  identity_omission,
  count_mismatch,
  spatial_relation_error,
  style_misinterpretation,
  lighting_misinterpretation,
  camera_angle_misinterpretation,
  shot_scale_misinterpretation,
  identity_hallucination.
- If multiple issues exist, choose the most dominant one.

IMPORTANT FORMAT RULES (CRITICAL - MUST FOLLOW EXACTLY):
- You MUST use EXACTLY the format below. Any deviation will cause parsing to fail.
- You MUST use the exact headers: [DIFFERENCE 1], ASPECT:, CAPTION_A_CLAIM:, CAPTION_B_CLAIM:, JUDGMENT:, CAPTION_A_ERROR:, ERROR_TYPE:, ERROR_DETAIL:, CAPTION_B_ERROR:, [OVERALL_WINNER]
- Do NOT add any additional headers, sections, or formatting.
- Do NOT use markdown, code blocks, or any other formatting.
- Use plain text only with exact header matching (case-sensitive).
- Repeat [DIFFERENCE N] blocks if multiple differences exist (use [DIFFERENCE 1], [DIFFERENCE 2], etc. in order).
- If a caption has no error, use:
  ERROR_TYPE: NONE
  ERROR_DETAIL: No error.
- You MUST include the [OVERALL_WINNER] section at the end.

OUTPUT FORMAT:

[DIFFERENCE 1]
ASPECT:
CAPTION_A_CLAIM:
CAPTION_B_CLAIM:
JUDGMENT: <A|B|both_wrong|both_supported>

CAPTION_A_ERROR:
ERROR_TYPE:
ERROR_DETAIL:

CAPTION_B_ERROR:
ERROR_TYPE:
ERROR_DETAIL:

[OVERALL_WINNER]
<A|B|Tie>

Caption A:
{caption_a}

Caption B:
{caption_b}
"""

HALL_VS_MISS_PROMPT = """
You are an expert at detecting hallucinations in image captions by comparing them with the actual image content and a ground truth caption.

Your task has two steps:

STEP 1: Compare GT caption with Predicted caption to identify differences
- Find what differs between the two captions
- Differences can be of three types:
  * CONTRADICTION: Predicted caption says something that contradicts GT (e.g., GT says "red car" but predicted says "blue car")
  * EXTRA_INFO: Predicted caption includes information not in GT
  * MISSING_FACT: GT includes information that is missing from predicted caption

STEP 2: For EACH difference found, verify the PREDICTED CAPTION's claim against the IMAGE:
- For CONTRADICTION: Even though predicted contradicts GT, verify the PREDICTED CAPTION's claim against the image
  * VERIFIED: Predicted caption's claim IS confirmed in the image → is_hallucination=false (predicted is correct, GT might be wrong)
  * FALSE: Predicted caption's claim is NOT in the image → is_hallucination=true (predicted contradicts both GT and image)
  * AMBIGUOUS: Cannot determine from image → is_hallucination=false (be conservative)
- For EXTRA_INFO: Verify the PREDICTED CAPTION's claim against the image
  * VERIFIED: Predicted caption claims something IS in the image → is_hallucination=false
  * FALSE: Predicted caption claims something NOT visible in the image → is_hallucination=true
  * AMBIGUOUS: Cannot determine from image → is_hallucination=false (be conservative)
- For MISSING_FACT: This is information in GT but missing from predicted caption
  * This is NOT a hallucination (it's an omission), but should be tracked separately
  * Set verification="missing" and is_hallucination=false for missing_fact type

Strict mapping rules for is_hallucination:
- type="contradiction" + verification="verified" → is_hallucination=false
- type="contradiction" + verification="false" → is_hallucination=true
- type="contradiction" + verification="ambiguous" → is_hallucination=false
- type="extra_info" + verification="false" → is_hallucination=true
- type="extra_info" + verification="verified" → is_hallucination=false
- type="extra_info" + verification="ambiguous" → is_hallucination=false
- type="missing_fact" → is_hallucination=false (always, as this is an omission, not a hallucination)

Category Classification:
- NATURAL: Hallucinations about natural/physical objects, scenes, people, animals, actions, poses, clothing, physical attributes, spatial relationships, or any real-world entities visible in the image.
- DESIGN: Hallucinations about text content, typography, layout, design elements, UI elements, logos, brand names, written text, font styles, graphic design elements, or any artificial/designed elements.

Guidelines:
- Be strict about contradictions: If image clearly shows A but predicted caption claims B, it's a hallucination
- Be lenient about description variations: "light blue" vs "gray-blue" are similar if image supports it
- Extra details in predicted caption that ARE in the image should NOT be marked as hallucinations (even if not in GT)
- Only mark as hallucination if the predicted caption's claim is clearly FALSE or contradicts the image
- If uncertain about a predicted caption claim, mark as AMBIGUOUS (not hallucination, be conservative)
- Classify each difference as either "natural" or "design" based on what type of content is being hallucinated
- Remember: GT helps find differences, but verification checks: "Is what the predicted caption claims actually in the image?"

Ground Truth Caption:
{gt_caption}

Predicted Caption:
{pred_caption}

IMPORTANT: Return your answer in strict plain text with this exact format:

[DIFFERENCE 1]
TYPE: contradiction|extra_info|missing_fact
CONTENT: <content from predicted caption>
CATEGORY: natural|design
VERIFICATION: verified|false|ambiguous|missing
REASON: <reason for the difference>
IS_HALLUCINATION: true|false

Return ONLY the plain text strictly following the format.
"""


def _parse_hall_vs_miss_response(text: str) -> dict | None:
    if not isinstance(text, str):
        return None
    parsed_text = text.strip()
    if parsed_text.startswith("```"):
        parsed_text = parsed_text.strip("`").strip()
        if parsed_text.lower().startswith("text"):
            parsed_text = parsed_text[4:].strip()

    lines = [l.rstrip() for l in parsed_text.splitlines()]
    n = len(lines)
    diff_indices = [
        i for i, l in enumerate(lines)
        if re.match(r"\[DIFFERENCE\s+\d+\]", l)
    ]
    differences = []
    for d_idx in diff_indices:
        d = {
            "type": "",
            "content": "",
            "category": "",
            "verification": "",
            "reason": "",
            "is_hallucination": "",
        }
        cursor = d_idx + 1
        while cursor < n and not lines[cursor].startswith("[DIFFERENCE"):
            line = lines[cursor]
            if line.startswith("TYPE:"):
                d["type"] = line[len("TYPE:"):].strip().lower()
            elif line.startswith("CONTENT:"):
                d["content"] = line[len("CONTENT:"):].strip()
            elif line.startswith("CATEGORY:"):
                d["category"] = line[len("CATEGORY:"):].strip().lower()
            elif line.startswith("VERIFICATION:"):
                d["verification"] = line[len("VERIFICATION:"):].strip().lower()
            elif line.startswith("REASON:"):
                d["reason"] = line[len("REASON:"):].strip()
            elif line.startswith("IS_HALLUCINATION:"):
                d["is_hallucination"] = line[len("IS_HALLUCINATION:"):].strip().lower()
            cursor += 1
        differences.append(d)

    if not differences:
        return None
    return {"differences": differences}


def _compute_hall_vs_miss_metrics(parsed: dict) -> tuple[int, int, int, int]:
    differences = parsed.get("differences", []) if isinstance(parsed, dict) else []
    total = len(differences)
    hallucination_count = 0
    missing_count = 0
    extra_count = 0
    for diff in differences:
        diff_type = str(diff.get("type", "")).lower()
        if diff_type == "missing_fact":
            missing_count += 1
        elif diff_type == "extra_info":
            extra_count += 1
        verification = str(diff.get("verification", "")).strip().lower()
        if verification == "false":
            hallucination_count += 1
    return total, hallucination_count, missing_count, extra_count


def _compute_hall_vs_miss_stats(parsed: dict) -> tuple[dict, dict]:
    differences = parsed.get("differences", []) if isinstance(parsed, dict) else []
    type_counts = {"contradiction": 0, "extra_info": 0, "missing_fact": 0}
    verification_counts = {"verified": 0, "false": 0, "ambiguous": 0, "missing": 0}
    for diff in differences:
        diff_type = str(diff.get("type", "")).strip().lower()
        if diff_type in type_counts:
            type_counts[diff_type] += 1
        verification = str(diff.get("verification", "")).strip().lower()
        if verification in verification_counts:
            verification_counts[verification] += 1
    return type_counts, verification_counts


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
    """Round value to nearest multiple of factor"""
    return round(value / factor) * factor


def _floor_by_factor(value: float, factor: int) -> int:
    """Floor value to nearest multiple of factor"""
    return floor(value / factor) * factor


def _ceil_by_factor(value: float, factor: int) -> int:
    """Ceil value to nearest multiple of factor"""
    return ceil(value / factor) * factor


def _smart_resize(
    height: int | float,
    width: int | float,
    factor: int = 28,
    min_pixels: int = 256 * 28 * 28,
    max_pixels: int = 8192 * 28 * 28,
) -> tuple[int | float, int | float]:
    """Smart resize image dimensions based on factor and pixel constraints"""
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

def normalize_error_type(error_type: str) -> str:
    return error_type.lower().strip()


AMBIGUITY_PATTERN = re.compile(
    r"\b(or|and/or|possibly|appears to|might be|could be|likely)\b|\w+/\w+",
    re.IGNORECASE
)


def count_ambiguity_tokens(text: str) -> int:
    """Count ambiguity tokens in text using regex pattern"""
    return len(AMBIGUITY_PATTERN.findall(text))

def normalized_a_only_reward(
    caption_a_error_count: float,  # Changed to float to support weighted counts
    total_differences: int,
    *,
    alpha: float = 2.0,
    epsilon: float = 0.05,
    k: float = 4.0,
    D_max: int = 10,
    max_weight: float = 1.0,  # Maximum severity weight for normalization (used when severity weighting is enabled)
):
    # Safety
    print("total_differences", total_differences)
    total_differences = max(total_differences, 1)

    # Error density
    # When severity weighting is enabled, normalize by max_weight to get a value in [0, 1]
    # When severity weighting is disabled, max_weight = 1.0, so no change
    if max_weight > 1.0:
        # Normalize weighted count to [0, 1] range by dividing by max_weight
        # This ensures error_density is comparable whether using weighted or unweighted counts
        error_density = caption_a_error_count / (max_weight * total_differences)
    else:
        # Unweighted mode: simple division
        error_density = caption_a_error_count / total_differences
    error_density = min(max(error_density, 0.0), 1.0)

    # Core reward (smooth, non-zero)
    base_reward = epsilon + (1 - epsilon) * (1 - error_density) ** alpha

    # Absolute error decay
    # When severity weighting is enabled, normalize the weighted count for penalty calculation
    # Use max_weight to normalize so penalty is comparable
    if max_weight > 1.0:
        normalized_error_count = caption_a_error_count / max_weight
    else:
        normalized_error_count = caption_a_error_count
    abs_penalty = math.exp(-normalized_error_count / k)

    # Difficulty bonus
    difficulty_bonus = math.log(1 + total_differences) / math.log(1 + D_max)
    difficulty_bonus = min(1.0, difficulty_bonus)

    # Combine
    reward = base_reward * abs_penalty * (0.7 + 0.3 * difficulty_bonus)

    return float(reward)


def infer_severity_3_from_error_type(error_type: str) -> int:
    et = normalize_error_type(error_type)

    # -----------------------
    # Severity 3: hallucination / incorrect claim
    # -----------------------
    if any(k in et for k in [
        "hallucination",
        "mismatch",
        "incorrect",
        "wrong",
        "nonexistent",
        "fabricated",
        "overreach",
    ]):
        return 3

    # -----------------------
    # Severity 2: factual but limited
    # -----------------------
    if any(k in et for k in [
        "omission",
        "missing",
        "truncation",
        "incomplete",
        "partial",
        "error",           # e.g. spatial_relation_error
    ]):
        return 2

    # -----------------------
    # Severity 1: minor / stylistic
    # -----------------------
    if any(k in et for k in [
        "style",
        "aesthetic",
        "misinterpretation",
        "subjective",
    ]):
        return 1

    # -----------------------
    # Fallback
    # -----------------------
    return 2


def parse_model_severity_level(value: str) -> int | None:
    """Parse model-provided severity level (1/2/3) or return None."""
    if value is None:
        return None
    text = str(value).strip().upper()
    if text in ["NA", "NONE", ""]:
        return None
    if text.isdigit():
        level = int(text)
        if level in [1, 2, 3]:
            return level
    return None

# Default severity weights (can be overridden via environment variables)
DEFAULT_SEVERITY_WEIGHT = {
    1: 1.0,   # minor / stylistic
    2: 1.25,  # factual but limited
    3: 1.6,   # hallucination / incorrect claim
}


@Verifier.register(name="gemini_caption_diff")
class GeminiCaptionDiffVerifier(BaseVerifier):

    def __init__(self, **kwargs):
        image_path = kwargs.get("image_path", None)
        # Ensure image_path is a list
        if image_path is None:
            self.image_paths = []
        elif isinstance(image_path, str):
            self.image_paths = [image_path]
        else:
            self.image_paths = list(image_path)
        
        self.model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")

        # Concurrency controls
        self.max_concurrent_requests = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))

        # Image base64 caching
        self.use_image_base64_cache = os.getenv("USE_IMAGE_BASE64_CACHE", "false").lower() in ["true", "1", "yes"]
        self.image_base64_cache_size = int(os.getenv("IMAGE_BASE64_CACHE_SIZE", "512"))
        self._image_base64_cache = OrderedDict()
        self._image_cache_lock = threading.Lock()

        # Request timeout (seconds)
        self.request_timeout_seconds = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "120"))
        
        # Max retries for API requests (reduce if experiencing slowdowns)
        self.max_retries = int(os.getenv("MAX_RETRIES", "8"))

        # Parse reward mapping parameters from environment variables
        # MAX_ERROR_DIFF: Maximum expected error difference (used for normalization)
        # Default: 10 (if model has 10 more errors than stronger model, reward should be 0)
        self.max_error_diff = float(os.getenv("MAX_ERROR_DIFF", "10.0"))
        
        # REWARD_SCALE: Scaling factor for reward calculation
        # Default: 1.0 (linear mapping)
        self.reward_scale = float(os.getenv("REWARD_SCALE", "1.0"))
        
        # MIN_REWARD: Minimum reward value (even if model has many more errors)
        # Default: 0.0
        self.min_reward = float(os.getenv("MIN_REWARD", "0.0"))
        
        # MAX_REWARD: Maximum reward value (when model has fewer errors)
        # Default: 1.0
        self.max_reward = float(os.getenv("MAX_REWARD", "1.0"))

        # Log sampling rate (0.0-1.0), failures always logged
        self.log_sample_rate = float(os.getenv("LOG_SAMPLE_RATE", "0.1"))
        
        # Error difference calculation mode
        # Options: "raw_diff", "normalized_diff", "normalized_a_only", "a_only", "judgment_based", "hall_vs_miss"
        # - "raw_diff": caption_a_error_count - caption_b_error_count (not normalized)
        # - "normalized_diff": (caption_a_error_count - caption_b_error_count) / total_differences
        # - "normalized_a_only": caption_a_error_count / total_differences (only Model A errors, normalized, uses special reward function)
        # - "a_only": caption_a_error_count (raw or weighted, NOT normalized by total_differences, uses linear reward mapping like normalized_diff)
        # - "judgment_based": (A_wins - B_wins) / total_differences based on JUDGMENT field and overall_winner
        # - "hall_vs_miss": penalize hallucinations and missing facts using image-based verification
        self.error_diff_mode = os.getenv("ERROR_DIFF_MODE", "normalized_diff")
        
        # USE_SEVERITY_WEIGHTED_REWARD: Whether to use severity-aware weighted error counts for reward calculation
        # When enabled, error counts are weighted by severity (1.0, 1.25, 1.6) for reward calculation
        # Original unweighted error counts are still returned in the result
        # Default: False (use unweighted error counts)
        self.use_severity_weighted_reward = os.getenv("USE_SEVERITY_WEIGHTED_REWARD", "false").lower() in ["true", "1", "yes"]
        
        # SEVERITY_WEIGHT_1, SEVERITY_WEIGHT_2, SEVERITY_WEIGHT_3: Configurable severity weights
        # Default: 1.0, 1.25, 1.6 for severity levels 1, 2, 3 respectively
        # Can be set via environment variables: SEVERITY_WEIGHT_1=1.0 SEVERITY_WEIGHT_2=1.25 SEVERITY_WEIGHT_3=1.6
        self.severity_weight = {
            1: float(os.getenv("SEVERITY_WEIGHT_1", str(DEFAULT_SEVERITY_WEIGHT[1]))),
            2: float(os.getenv("SEVERITY_WEIGHT_2", str(DEFAULT_SEVERITY_WEIGHT[2]))),
            3: float(os.getenv("SEVERITY_WEIGHT_3", str(DEFAULT_SEVERITY_WEIGHT[3]))),
        }
        
        # USE_AMBIGUITY_PENALTY: Whether to apply ambiguity penalty to reward score
        # When enabled, applies penalty based on ambiguity words in the caption (or, possibly, might be, etc.)
        # Penalty formula: exp(-c * max(0, ambiguity_count - free)) where free is calculated dynamically based on caption length
        # Default: False (no ambiguity penalty)
        self.use_ambiguity_penalty = os.getenv("USE_AMBIGUITY_PENALTY", "false").lower() in ["true", "1", "yes"]
        self.require_think_block = os.getenv("REQUIRE_THINK_BLOCK", "false").lower() in ["true", "1", "yes"]
        
        # Ambiguity penalty parameters
        # AMBIGUITY_FREE_WORDS_PER_TOKEN: Number of words allowed per ambiguity token without penalty
        # Default: 90 (allows 1 ambiguity token per 90 words, i.e., 80-100 words range)
        # The free ambiguity count is calculated as: caption_word_count / AMBIGUITY_FREE_WORDS_PER_TOKEN
        # Example: 180 words caption -> free ambiguity = 180/90 = 2 tokens
        self.ambiguity_free_words_per_token = float(os.getenv("AMBIGUITY_FREE_WORDS_PER_TOKEN", "90.0"))
        
        # Legacy support: if AMBIGUITY_FREE is set, use it as a fixed value (override dynamic calculation)
        # If not set, will calculate dynamically based on caption length
        ambiguity_free_env = os.getenv("AMBIGUITY_FREE", None)
        if ambiguity_free_env is not None:
            self.ambiguity_free_fixed = float(ambiguity_free_env)
            self.use_fixed_ambiguity_free = True
        else:
            self.ambiguity_free_fixed = None
            self.use_fixed_ambiguity_free = False
        
        self.ambiguity_decay = float(os.getenv("AMBIGUITY_DECAY", "0.1"))  # Decay constant c
        
        # Parallel processing configuration
        # MAX_CONCURRENT_REQUESTS: Maximum number of concurrent API requests
        # Default: 10 (adjust based on API rate limits and desired QPS)
        self.max_concurrent_requests = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))
        
        # Training data logging configuration
        # SAVE_TRAINING_DATA: Whether to save training data to JSONL file
        # Default: False
        self.save_training_data = os.getenv("SAVE_TRAINING_DATA", "false").lower() in ["true", "1", "yes"]
        
        # SAVE_TRAINING_DATA_PATH: Path to save training data JSONL file
        # Default: None (must be set if SAVE_TRAINING_DATA is true)
        # All examples from a single run will be appended to the same file
        save_training_data_path_raw = os.getenv("SAVE_TRAINING_DATA_PATH", None)
        
        # Initialize JSONL file if saving is enabled
        if self.save_training_data:
            if not save_training_data_path_raw:
                logger.warning("SAVE_TRAINING_DATA is enabled but SAVE_TRAINING_DATA_PATH is not set. Training data will not be saved.")
                self.save_training_data = False
                self.save_training_data_path = None
            else:
                # Use the path as-is (all examples will be appended to the same file)
                self.save_training_data_path = save_training_data_path_raw
                
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(self.save_training_data_path) if os.path.dirname(self.save_training_data_path) else ".", exist_ok=True)
                logger.info(f"Training data will be saved to: {self.save_training_data_path} (all examples in the same file)")
        else:
            self.save_training_data_path = None

        logger.info(f"Max error difference: {self.max_error_diff}")
        logger.info(f"Reward scale: {self.reward_scale}")
        logger.info(f"Reward range: [{self.min_reward}, {self.max_reward}]")
        logger.info(f"Log sample rate: {self.log_sample_rate}")
        logger.info(f"Error difference mode: {self.error_diff_mode}")
        logger.info(f"Use severity weighted reward: {self.use_severity_weighted_reward}")
        if self.use_severity_weighted_reward:
            logger.info(f"Severity weights: Level 1={self.severity_weight[1]}, Level 2={self.severity_weight[2]}, Level 3={self.severity_weight[3]}")
        logger.info(f"Use ambiguity penalty: {self.use_ambiguity_penalty}")
        logger.info(f"Require think block: {self.require_think_block}")
        if self.use_ambiguity_penalty:
            if self.use_fixed_ambiguity_free:
                logger.info(f"Ambiguity penalty parameters: free={self.ambiguity_free_fixed} (fixed), decay={self.ambiguity_decay}")
            else:
                logger.info(f"Ambiguity penalty parameters: free=dynamic (1 token per {self.ambiguity_free_words_per_token} words), decay={self.ambiguity_decay}")
        logger.info(f"Max concurrent requests: {self.max_concurrent_requests}")
        logger.info(f"Save training data: {self.save_training_data}")
        logger.info(f"Image Paths: {self.image_paths}")
        
        # DIFF_PROMPT_VERSION: Which prompt template to use ("v1", "v2", "v4", "v5", or "v6")
        # Default: "v1" (uses DIFF_PROMPT)
        # Set to "v2" to use DIFF_PROMPT_V2
        # Set to "v4" to use DIFF_PROMPT_V4 (emphasizes omissions and unverifiable statements)
        # Set to "v5" to use DIFF_PROMPT_V5 (image-based severity levels)
        # Set to "v6" to use DIFF_PROMPT_V6 (rule-based severity levels)
        diff_prompt_version = os.getenv("DIFF_PROMPT_VERSION", "v1").lower()
        self.use_model_judge_severity = os.getenv("USE_MODEL_JUDGE_SEVERITY", "false").lower() in ["true", "1", "yes"]
        if diff_prompt_version == "v4":
            self.diff_prompt_template = DIFF_PROMPT_V4
            logger.info("Using DIFF_PROMPT_V4 for caption comparison (emphasizes omissions and unverifiable statements)")
        elif diff_prompt_version == "v6":
            self.diff_prompt_template = DIFF_PROMPT_V6
            logger.info("Using DIFF_PROMPT_V6 for caption comparison (rule-based severity levels)")
        elif diff_prompt_version == "v5":
            self.diff_prompt_template = DIFF_PROMPT_V5
            logger.info("Using DIFF_PROMPT_V5 for caption comparison (image-based severity levels)")
        elif diff_prompt_version == "v2":
            self.diff_prompt_template = DIFF_PROMPT_V2
            logger.info("Using DIFF_PROMPT_V2 for caption comparison")
        else:
            self.diff_prompt_template = DIFF_PROMPT
            logger.info("Using DIFF_PROMPT (v1) for caption comparison")

        if self.use_model_judge_severity and diff_prompt_version not in ["v5", "v6"]:
            logger.info("USE_MODEL_JUDGE_SEVERITY is enabled but DIFF_PROMPT_VERSION is not v5/v6; severity will fall back to error_type mapping.")

    def _save_training_data_to_jsonl(
        self,
        meta: dict,
        actor_caption: str,
        prompt: str,
        gemini_response: str,
        verification_result: dict
    ):
        """Save training data to JSONL file
        
        Args:
            meta: Dictionary with iter, data_index, image_path
            actor_caption: The actor's caption output
            prompt: The prompt sent to Gemini
            gemini_response: The raw response from Gemini
            verification_result: The parsed verification result
        """
        if not self.save_training_data:
            return
        
        try:
            from datetime import datetime
            timestamp = datetime.now().isoformat()  # ISO 8601 format: YYYY-MM-DDTHH:MM:SS.ffffff
            
            training_record = {
                "timestamp": timestamp,
                "meta": meta,
                "actor_caption": actor_caption,
                "prompt": prompt,
                "gemini_response": gemini_response,
                "verification_result": verification_result
            }
            
            # Append to JSONL file (thread-safe with file lock)
            import threading
            lock = getattr(self, '_jsonl_lock', None)
            if lock is None:
                lock = threading.Lock()
                self._jsonl_lock = lock
            
            with lock:
                with open(self.save_training_data_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(training_record, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.warning(f"Failed to save training data to JSONL: {e}")

    def _encode_image_to_base64(self, image_path: str) -> str:
        """Encode image to base64 format with smart resize"""
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
                        # Move to end (LRU)
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
            # Basic LRU cache
            with self._image_cache_lock:
                self._image_base64_cache[cache_key] = encoded
                if len(self._image_base64_cache) > self.image_base64_cache_size:
                    # Pop oldest
                    self._image_base64_cache.popitem(last=False)

        return encoded

    def _send_gemini_request(
        self, prompt: str, image_paths: list[str], temperature: float = 0.3
    ):
        """Send request to Gemini API with multiple images

        Returns:
            tuple: (response_text, usage_stats)
                   If failed, usage_stats is None and response_text contains error message
        """
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
            # Note: Not using response_mime_type to get plain text format
        }

        # Build parts list with prompt and all images
        parts = [{"text": prompt}]
        
        # Add all images to the request
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
                
                text = text.strip()
                usage_stats = {
                    "prompt_tokens": response_json["usageMetadata"]["promptTokenCount"],
                    "total_tokens": response_json["usageMetadata"]["totalTokenCount"],
                    "request_time": request_time,
                }
                return text, usage_stats
            elif response.status_code == 429:
                # Rate limit error - return special error code for retry with backoff
                error_text = response.text
                error_message = f"Error 429: {error_text}"
                return error_message, {"status": 429, "rate_limited": True}
            else:
                error_message = f"Error {response.status_code}: {response.text}"
                return error_message, None
        except requests.exceptions.Timeout:
            # Request timeout - return error message for retry
            error_message = f"Request timeout (exceeded {self.request_timeout_seconds} seconds)"
            logger.warning(error_message)
            return error_message, {"timeout": True}
        except requests.exceptions.RequestException as e:
            # Network/connection errors - return error message for retry
            error_message = f"Request error: {str(e)}"
            logger.warning(error_message)
            return error_message, {"client_error": True}
        except Exception as e:
            # Other unexpected errors
            error_message = f"Unexpected error: {str(e)}"
            logger.error(error_message, exc_info=True)
            return error_message, None

    async def _send_gemini_request_async(
        self, prompt: str, image_paths: list[str], temperature: float = 0.3, session: aiohttp.ClientSession = None
    ):
        """Send async request to Gemini API with multiple images

        Args:
            prompt: The prompt text
            image_paths: List of image paths
            temperature: Temperature for generation
            session: Optional aiohttp session (if None, creates a new one)

        Returns:
            tuple: (response_text, usage_stats)
                   If failed, usage_stats is None and response_text contains error message
        """
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

        # Build parts list with prompt and all images
        parts = [{"text": prompt}]
        
        # Add all images to the request
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
        close_session = False
        if session is None:
            session = aiohttp.ClientSession()
            close_session = True
        
        try:
            async with session.post(
                url,
                headers=headers,
                json=body,
                timeout=aiohttp.ClientTimeout(total=self.request_timeout_seconds),
            ) as response:
                request_time = time.time() - start_time
                
                if response.status == 200:
                    response_json = await response.json()
                    parts = response_json.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                    if not parts:
                        return None, None
                    
                    text = parts[0].get("text", "")
                    if not isinstance(text, str):
                        return None, None
                    
                    text = text.strip()
                    usage_stats = {
                        "prompt_tokens": response_json["usageMetadata"]["promptTokenCount"],
                        "total_tokens": response_json["usageMetadata"]["totalTokenCount"],
                        "request_time": request_time,
                    }
                    return text, usage_stats
                elif response.status == 429:
                    # Rate limit error - return special error code for retry with backoff
                    error_text = await response.text()
                    error_message = f"Error 429: {error_text}"
                    return error_message, {"status": 429, "rate_limited": True}
                else:
                    error_text = await response.text()
                    error_message = f"Error {response.status}: {error_text}"
                    return error_message, None
        except asyncio.TimeoutError:
            # Request timeout - return error message for retry
            error_message = f"Request timeout (exceeded {self.request_timeout_seconds} seconds)"
            logger.warning(error_message)
            return error_message, {"timeout": True}
        except aiohttp.ClientError as e:
            # Network/client errors - return error message for retry
            error_message = f"Client error: {str(e)}"
            logger.warning(error_message)
            return error_message, {"client_error": True}
        except Exception as e:
            # Other unexpected errors
            error_message = f"Unexpected error: {str(e)}"
            logger.error(error_message, exc_info=True)
            return error_message, None
        finally:
            if close_session:
                await session.close()

    def _parse_diff_text_response(self, text: str) -> dict:
        """Parse the strict plain-text diff format into a dict"""
        result = {
            "differences": [],
            "overall_winner": "NA",
        }

        if not isinstance(text, str):
            return result

        lines = [l.rstrip() for l in text.splitlines()]
        n = len(lines)

        # Remove fenced code blocks if present
        if text.startswith("```"):
            text = text.strip("`").strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()
            lines = [l.rstrip() for l in text.splitlines()]
            n = len(lines)

        # Parse DIFFERENCES
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
                    # read ERROR_TYPE / ERROR_DETAIL
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

        # Parse OVERALL_WINNER
        if "[OVERALL_WINNER]" in lines:
            idx = lines.index("[OVERALL_WINNER]") + 1
            if idx < n:
                result["overall_winner"] = lines[idx].strip()

        return result
    
    def _validate_parsed_result(self, result: dict) -> tuple[bool, str]:
        """Validate that the parsed result follows the expected format.
        
        Returns:
            (is_valid, error_message): Tuple of (bool, str) indicating if valid and error message if not
        """
        if not result or "differences" not in result:
            return False, "Missing 'differences' field in result"
        
        differences = result.get("differences", [])
        
        # Must have at least one difference entry
        if len(differences) == 0:
            return False, "No [DIFFERENCE N] blocks found in response"
        
        # Check if overall winner is present
        overall_winner = result.get("overall_winner", "NA")
        if overall_winner == "NA":
            return False, "Missing [OVERALL_WINNER] section in response"
        
        # Validate each difference entry has required fields
        for idx, diff in enumerate(differences):
            if not isinstance(diff, dict):
                return False, f"Difference {idx+1} is not a valid dictionary"
            
            # Check required fields exist (even if empty/NA)
            required_fields = ["aspect", "caption_a_claim", "caption_b_claim", "judgment", 
                             "caption_a_error", "caption_b_error"]
            for field in required_fields:
                if field not in diff:
                    return False, f"Difference {idx+1} missing required field: {field}"
            
            # Check error fields have required subfields
            for error_field in ["caption_a_error", "caption_b_error"]:
                error_dict = diff.get(error_field, {})
                if not isinstance(error_dict, dict):
                    return False, f"Difference {idx+1} {error_field} is not a dictionary"
                if "error_type" not in error_dict or "error_detail" not in error_dict:
                    return False, f"Difference {idx+1} {error_field} missing error_type or error_detail"
                if "severity_level" not in error_dict:
                    # For backward compatibility, allow missing severity_level
                    error_dict["severity_level"] = "NA"
        
        return True, ""

    def _map_error_diff_to_reward(self, error_difference: float, caption_a_error_count: float = None, total_differences: int = None, max_weight: float = 1.0) -> float:
        """Map error difference to reward score (0 to 1)
        
        Args:
            error_difference: Error difference value (interpretation depends on error_diff_mode)
                - For "raw_diff": (caption_a_error_count - caption_b_error_count)
                - For "normalized_diff": (caption_a_error_count - caption_b_error_count) / total_differences
                - For "normalized_a_only": caption_a_error_count / total_differences (or weighted version)
                - For "a_only": caption_a_error_count (raw or weighted, NOT normalized by total_differences, uses linear mapping)
                - For "judgment_based": -(A_wins - B_wins) / total_differences (inverted judgment difference)
            caption_a_error_count: Optional, used for normalized_a_only mode to apply absolute error penalty
                Can be int (unweighted) or float (weighted when severity weighting is enabled)
            total_differences: Optional, used for normalized_a_only mode to apply absolute error penalty
            max_weight: Optional, maximum severity weight for normalization (default 1.0, used when severity weighting is enabled)
        
        Returns:
            Reward score in range [min_reward, max_reward] (default [0, 1])
            - For "raw_diff" and "normalized_diff": negative (model better) -> higher reward
            - For "normalized_a_only": lower (fewer errors) -> higher reward (uses special reward function)
            - For "a_only": lower (fewer errors) -> higher reward (uses linear mapping)
            - For "judgment_based": negative (A better by judgments) -> higher reward
        """
        if self.error_diff_mode == "normalized_a_only":
            # For normalized_a_only: error_difference is A_errors / total_differences (0 to 1)
            # Lower values (fewer errors) should get higher rewards
            # Special case: when total_differences == 0, no differences found means perfect consistency
            # This should be rewarded with maximum score
            if total_differences == 0 or caption_a_error_count is None:
                # No differences found - captions are consistent, give maximum reward
                base_reward = 1.0
            else:
                # Use weighted count (which may be float) and max_weight for proper normalization
                # When severity weighting is disabled, caption_a_error_count will be float/int and max_weight = 1.0
                # When severity weighting is enabled, caption_a_error_count will be float (weighted) and max_weight > 1.0
                # Ensure caption_a_error_count is a float for consistency
                error_count = float(caption_a_error_count) if caption_a_error_count is not None else 0.0
                base_reward = normalized_a_only_reward(error_count, total_differences, max_weight=max_weight)
            
            # Apply reward_scale as a multiplier (default 1.0 means no change)
            scaled_reward = base_reward * self.reward_scale
            
            # Clamp scaled reward to [0, 1] before mapping to [min_reward, max_reward]
            scaled_reward = max(0.0, min(1.0, scaled_reward))
            
            # Map from [0, 1] to [min_reward, max_reward] range
            reward = self.min_reward + scaled_reward * (self.max_reward - self.min_reward)
            
            # Ensure reward is within bounds (redundant but safe)
            reward = max(self.min_reward, min(self.max_reward, reward))
            
            return reward
        elif self.error_diff_mode == "a_only":
            # For a_only: error_difference is raw/weighted A_errors (not normalized by total_differences)
            # Uses linear reward mapping like normalized_diff (preserves severity weight and ambiguity penalty)
            # Lower values (fewer errors) should get higher rewards
            # When using severity weights, scale max_error_diff by max_weight to account for weighted errors
            # This ensures weighted errors (which can be larger) are properly normalized
            effective_max_error = self.max_error_diff * max_weight if max_weight > 1.0 else self.max_error_diff
            # Normalize error count to [0, 1] range using effective_max_error
            # Clamp to prevent extreme values (error count can't be negative)
            clamped_error = max(0.0, min(effective_max_error, error_difference))
            normalized_error = clamped_error / effective_max_error if effective_max_error > 0 else 0.0
            # Linear mapping: reward = 1.0 - normalized_error
            # This ensures: 0 errors -> 1.0 reward, effective_max_error errors -> 0.0 reward
            base_reward = 1.0 - normalized_error
        else:
            # For "raw_diff", "normalized_diff", and "judgment_based": error_difference can be negative, zero, or positive
            if self.error_diff_mode == "normalized_diff" or self.error_diff_mode == "judgment_based":
                # For normalized_diff and judgment_based, error_difference is already normalized to [-1, 1] range
                # (or close to it, with severity weights it's properly normalized by max_weight)
                # Clamp to [-1, 1] to ensure it's in the expected range
                normalized_diff = max(-1.0, min(1.0, error_difference))
            else:
                # For "raw_diff": normalize error difference to [-1, 1] range using max_error_diff
                # Clamp to prevent extreme values
                clamped_diff = max(-self.max_error_diff, min(self.max_error_diff, error_difference))
                normalized_diff = clamped_diff / self.max_error_diff if self.max_error_diff > 0 else 0.0
            
            # Map to [0, 1] range: negative diff (better) -> higher reward
            # Linear mapping: reward = 0.5 - 0.5 * normalized_diff
            # Examples:
            #   diff = -1.0 (best): reward = 1.0
            #   diff = 0 (same): reward = 0.5
            #   diff = 1.0 (worst): reward = 0.0
            base_reward = 0.5 - 0.5 * normalized_diff
        
        # Apply reward_scale as a multiplier (default 1.0 means no change)
        scaled_reward = base_reward * self.reward_scale
        
        # Clamp scaled reward to [0, 1] before mapping to [min_reward, max_reward]
        scaled_reward = max(0.0, min(1.0, scaled_reward))
        
        # Map from [0, 1] to [min_reward, max_reward] range
        reward = self.min_reward + scaled_reward * (self.max_reward - self.min_reward)
        
        # Ensure reward is within bounds (redundant but safe)
        reward = max(self.min_reward, min(self.max_reward, reward))
        
        return reward

    def verify_accuracy(
        self, predict_caption: str, solution_caption: str | None = None, 
        iter: int = None, data_index: int = None, image_path: str | List[str] = None
    ) -> dict:
        """Compare model caption with stronger model caption and return reward score
        
        Uses async HTTP internally for better performance with connection pooling and non-blocking I/O.

        Args:
            predict_caption: The model's caption (Caption A)
            solution_caption: The stronger model's caption (Caption B)
            iter: Training iteration number (for logging)
            data_index: Data index (for logging)
            image_path: Image path(s) (for logging, can be str or list)

        Returns:
            dict: {
                'caption_a_error_count': int (number of errors in model's caption),
                'caption_b_error_count': int (number of errors in stronger model's caption),
                'total_differences': int (total number of differences found),
                'error_difference': float (error difference based on ERROR_DIFF_MODE),
                'final_score': float (same as reward_score, for compatibility),
            }
        """
        # Use async version internally for better performance
        # This uses aiohttp for non-blocking I/O and better connection handling
        try:
            print(f"predict_caption before extract_answer_text: {predict_caption}")
            predict_caption = _extract_answer_text(predict_caption)
            print(f"predict_caption after extract_answer_text: {predict_caption}")
            # Check if we're already in an async context (e.g., FastAPI)
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context, run async code in a thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self._verify_accuracy_async_internal(self, predict_caption, solution_caption, None, iter, data_index, image_path)
                    )
                    result = future.result(timeout=600)  # 10 minute timeout
                    return result
            except RuntimeError:
                # No running loop, we can use asyncio.run() directly
                result = asyncio.run(
                    self._verify_accuracy_async_internal(self, predict_caption, solution_caption, None, iter, data_index, image_path)
                )
                return result
        except Exception as e:
            logger.warning(f"Error in async verify_accuracy, falling back to sync: {e}")
            # Fallback to sync version if async fails
            return self._verify_accuracy_sync_fallback(predict_caption, solution_caption, iter, data_index, image_path)
    
    def _verify_accuracy_sync_fallback(
        self, predict_caption: str, solution_caption: str | None = None,
        iter: int = None, data_index: int = None, image_path: str | List[str] = None
    ) -> dict:
        """Fallback synchronous implementation of verify_accuracy"""
        failed_result = {
            "caption_a_error_count": 0,
            "caption_b_error_count": 0,
            "total_differences": 0,
            "error_difference": 0,
            "final_score": 0.0,
            "num_parsing_failures": 0,
        }

        print(f"predict_caption before extract_answer_text: {predict_caption}")
        predict_caption = _extract_answer_text(predict_caption)
        print(f"predict_caption after extract_answer_text: {predict_caption}")

        # Check if solution_caption (stronger model's caption) is provided
        if solution_caption is None or not solution_caption.strip():
            logger.warning("solution_caption (stronger model's caption) is required but not provided")
            return failed_result

        # Determine if we should log this request (sample or always log failures)
        should_log = random.random() < self.log_sample_rate

        # Check if image_paths is valid
        if not self.image_paths:
            logger.warning("image_paths is empty, cannot evaluate caption")
            return failed_result
        
        # Check if all images exist
        for image_path in self.image_paths:
            if not os.path.exists(image_path):
                logger.warning(f"image_path does not exist: {image_path}")
                return failed_result

        # Build prompt with both captions
        if self.error_diff_mode == "hall_vs_miss":
            prompt = HALL_VS_MISS_PROMPT.format(
                gt_caption=solution_caption,
                pred_caption=predict_caption
            )
        else:
            prompt = self.diff_prompt_template.format(
                caption_a=predict_caption,
                caption_b=solution_caption
            )

        # Retry logic with adaptive backoff for rate limits
        max_retries = self.max_retries
        result = None
        response_text_final = None  # Store final successful response for saving
        consecutive_rate_limits = 0  # Track consecutive rate limits for adaptive backoff
        num_parsing_failures = 0  # Track number of parsing/format validation failures (when Gemini doesn't follow format)
        
        for attempt in range(max_retries):
            # Set temperature: 0.0 for first attempt (matching reference script), random for retries
            temperature = 0.0 if attempt == 0 else random.uniform(0, 1)

            if should_log:
                logger.info(
                    f"Attempt {attempt + 1}/{max_retries} with temperature={temperature:.2f}"
                )

            # Send request to Gemini with all images
            response_text, usage_stats = self._send_gemini_request(
                prompt, self.image_paths, temperature
            )

            # Handle rate limit errors (429) with adaptive backoff
            if usage_stats is not None and isinstance(usage_stats, dict) and usage_stats.get("rate_limited"):
                consecutive_rate_limits += 1
                # Adaptive backoff: start small, increase if multiple consecutive rate limits
                # Base wait: 0.5-1.5s, scales with consecutive rate limits, max 10s
                base_wait = 0.5 + random.uniform(0, 1.0)  # 0.5-1.5s base
                scale_factor = min(consecutive_rate_limits, 5)  # Cap scaling at 5x
                wait_time = min(base_wait * scale_factor, 10.0)  # Max 10s
                logger.warning(
                    f"Rate limit hit (attempt {attempt + 1}/{max_retries}, consecutive={consecutive_rate_limits}), "
                    f"waiting {wait_time:.2f}s before retry"
                )
                time.sleep(wait_time)
                continue
            else:
                consecutive_rate_limits = 0  # Reset counter on success or non-rate-limit error

            # Handle timeout errors with a moderate wait (server may be slow)
            if usage_stats is not None and isinstance(usage_stats, dict) and usage_stats.get("timeout"):
                logger.warning(
                    f"Request timeout (attempt {attempt + 1}/{max_retries}), "
                    f"waiting 2-4s before retry"
                )
                time.sleep(2.0 + random.uniform(0, 2.0))  # 2-4s wait
                continue

            if usage_stats is None or response_text is None:
                logger.warning(
                    f"Gemini request failed (attempt {attempt + 1}): {response_text}"
                )
                if attempt < max_retries - 1:
                    # Brief wait before retry for non-rate-limit errors (0.2-0.4s)
                    time.sleep(0.2 + random.uniform(0, 0.2))
                continue

            if should_log:
                logger.info(f"Gemini request time: {usage_stats['request_time']:.2f}s")
                logger.info(
                    f"Token usage: prompt_tokens={usage_stats['prompt_tokens']}, total_tokens={usage_stats['total_tokens']}"
                )

            # Parse response
            try:
                if self.error_diff_mode == "hall_vs_miss":
                    parsed = _parse_hall_vs_miss_response(response_text)
                    if not parsed:
                        num_parsing_failures += 1
                        logger.warning(
                            f"Parsing failure: Gemini response is not valid JSON (attempt {attempt + 1}/{max_retries}). "
                            f"Response preview: {response_text[:300]}..."
                        )
                        if attempt < max_retries - 1:
                            time.sleep(0.3 + random.uniform(0, 0.2))
                            continue
                        failed_result["num_parsing_failures"] = num_parsing_failures
                        return failed_result
                    result = parsed
                    response_text_final = response_text
                    break
                else:
                    result = self._parse_diff_text_response(response_text)
                    # Validate that the parsed result follows the expected format
                    is_valid, validation_error = self._validate_parsed_result(result)
                    if not is_valid:
                        num_parsing_failures += 1  # Count format validation failure (Gemini didn't follow format)
                        logger.warning(
                            f"Parsing failure: Gemini response does not follow required format (attempt {attempt + 1}/{max_retries}): {validation_error}. "
                            f"Response preview: {response_text[:300]}..."
                        )
                        if attempt < max_retries - 1:
                            # Wait briefly before retrying with stricter prompt
                            time.sleep(0.3 + random.uniform(0, 0.2))
                            continue
                        else:
                            logger.error(f"All {max_retries} attempts failed - format validation failed: {validation_error}")
                            failed_result["num_parsing_failures"] = num_parsing_failures
                            return failed_result
                    # Format validation passed
                    if result and "differences" in result:
                        response_text_final = response_text  # Store successful response
                        # num_parsing_failures already counted for each format validation failure
                        break
            except Exception as e:
                num_parsing_failures += 1  # Count parse exception as a parsing failure
                logger.warning(f"Parsing failure: Failed to parse response (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    logger.error(f"All {max_retries} attempts failed")
                    failed_result["num_parsing_failures"] = num_parsing_failures
                    return failed_result
                # Wait briefly before retrying
                time.sleep(0.3 + random.uniform(0, 0.2))
                continue
        else:
            logger.error(f"All {max_retries} attempts failed or no valid result")
            failed_result["num_parsing_failures"] = num_parsing_failures
            return failed_result

        if self.error_diff_mode == "hall_vs_miss":
            total_differences, hallucination_count, missing_count, extra_count = _compute_hall_vs_miss_metrics(result)
            penalty = hallucination_count + missing_count
            if total_differences > 0:
                error_difference = penalty / total_differences
            else:
                error_difference = 0.0
            base_reward = 1.0 - max(0.0, min(1.0, error_difference))
            # Apply ambiguity penalty if enabled
            if self.use_ambiguity_penalty:
                ambiguity_penalty = math.exp(-self.ambiguity_decay * max(0, ambiguity_count - ambiguity_free))
                base_reward *= ambiguity_penalty
            else:
                ambiguity_penalty = 1.0
            if should_log:
                type_counts, verification_counts = _compute_hall_vs_miss_stats(result)
                logger.info(f"hall_vs_miss: total_differences={total_differences}")
                logger.info(f"hall_vs_miss: hallucination_count={hallucination_count}, missing_fact_count={missing_count}, extra_info_count={extra_count}")
                logger.info(f"hall_vs_miss: type_counts={type_counts}")
                logger.info(f"hall_vs_miss: verification_counts={verification_counts}")
                logger.info(f"hall_vs_miss: error_difference={error_difference:.4f}")
                if self.use_ambiguity_penalty:
                    logger.info(f"hall_vs_miss: ambiguity_count={ambiguity_count}, ambiguity_free={ambiguity_free:.2f}, ambiguity_penalty={ambiguity_penalty:.4f}")
            scaled_reward = base_reward * self.reward_scale
            scaled_reward = max(0.0, min(1.0, scaled_reward))
            reward_score = self.min_reward + scaled_reward * (self.max_reward - self.min_reward)
            reward_score = max(self.min_reward, min(self.max_reward, reward_score))
            if should_log:
                logger.info(f"hall_vs_miss: reward_score={reward_score:.4f}")

            result_dict = {
                "caption_a_error_count": 0,
                "caption_b_error_count": 0,
                "total_differences": total_differences,
                "normalized_error_difference": error_difference,
                "final_score": reward_score,
                "num_parsing_failures": num_parsing_failures,
                "hallucination_count": hallucination_count,
                "missing_fact_count": missing_count,
                "extra_info_count": extra_count,
                "ambiguity_count": ambiguity_count,
            }
            if self.use_ambiguity_penalty:
                result_dict["ambiguity_penalty"] = ambiguity_penalty
                result_dict["ambiguity_free"] = ambiguity_free

            if self.save_training_data and response_text_final is not None:
                image_path_to_save = image_path if image_path is not None else self.image_paths
                if isinstance(image_path_to_save, str):
                    image_path_to_save = [image_path_to_save]
                elif image_path_to_save is None:
                    image_path_to_save = []
                meta = {
                    "iter": iter,
                    "data_index": data_index,
                    "image_path": image_path_to_save
                }
                self._save_training_data_to_jsonl(
                    meta=meta,
                    actor_caption=predict_caption,
                    prompt=prompt,
                    gemini_response=response_text_final,
                    verification_result=result_dict
                )
            return result_dict

        if not result or "differences" not in result:
            logger.error("Invalid result format: missing 'differences'")
            failed_result["num_parsing_failures"] = num_parsing_failures
            return failed_result

        # Count errors for each caption
        # Error is counted if error_type is not "NONE" and not "NA"
        # Track both unweighted (original) and weighted error counts
        caption_a_error_count = 0  # Unweighted count (returned in result)
        caption_b_error_count = 0  # Unweighted count (returned in result)
        caption_a_error_count_weighted = 0.0  # Weighted count (used for reward if flag enabled)
        caption_b_error_count_weighted = 0.0  # Weighted count (used for reward if flag enabled)
        error_types_a = set()
        error_types_b = set()
        # Track severity level counts for each caption
        caption_a_severity_counts = {1: 0, 2: 0, 3: 0}  # Count errors by severity level
        caption_b_severity_counts = {1: 0, 2: 0, 3: 0}  # Count errors by severity level
        total_differences = 0  # Only count differences where at least one caption has an error
        total_differences = len(result.get("differences", []))
        
        # Count ambiguity tokens in predict_caption
        ambiguity_count = count_ambiguity_tokens(predict_caption)
        
        # Calculate dynamic ambiguity_free based on caption length
        # If using fixed value (legacy), use it; otherwise calculate from caption word count
        if self.use_ambiguity_penalty:
            if self.use_fixed_ambiguity_free:
                ambiguity_free = self.ambiguity_free_fixed
            else:
                # Calculate caption word count (split by whitespace)
                caption_word_count = len(predict_caption.split())
                # Calculate free ambiguity tokens: 1 token per N words
                ambiguity_free = caption_word_count / self.ambiguity_free_words_per_token
                # Ensure minimum of 0 (can't be negative)
                ambiguity_free = max(0.0, ambiguity_free)
        else:
            ambiguity_free = 0.0  # Not used when penalty is disabled

        for diff in result.get("differences", []):
            # Check caption A error
            caption_a_error = diff.get("caption_a_error", {})
            error_type_a = caption_a_error.get("error_type", "NA")
            has_error_a = error_type_a and error_type_a.upper() not in ["NONE", "NA"]
            
            # Check caption B error
            caption_b_error = diff.get("caption_b_error", {})
            error_type_b = caption_b_error.get("error_type", "NA")
            has_error_b = error_type_b and error_type_b.upper() not in ["NONE", "NA"]
            
            # Only count this difference if at least one caption has an error
            # if has_error_a or has_error_b:
            #     total_differences += 1
            
            # Count caption A errors (unweighted and weighted)
            if has_error_a:
                caption_a_error_count += 1
                error_types_a.add(error_type_a)
                # Always calculate severity to track severity level counts
                severity = None
                if self.use_model_judge_severity:
                    severity = parse_model_severity_level(caption_a_error.get("severity_level", "NA"))
                    print(f"use_model_judge_severity: True, Severity: {severity}, error_type_a: {error_type_a}")
                if severity is None:
                    severity = infer_severity_3_from_error_type(error_type_a)
                    print(f"use_model_judge_severity: False, Severity: {severity}, error_type_a: {error_type_a}")
                caption_a_severity_counts[severity] += 1
                if self.use_severity_weighted_reward:
                    weight = self.severity_weight[severity]
                    caption_a_error_count_weighted += weight
                else:
                    caption_a_error_count_weighted += 1.0

            # Count caption B errors (unweighted and weighted)
            if has_error_b:
                caption_b_error_count += 1
                error_types_b.add(error_type_b)
                # Always calculate severity to track severity level counts
                severity = None
                if self.use_model_judge_severity:
                    severity = parse_model_severity_level(caption_b_error.get("severity_level", "NA"))
                    print(f"use_model_judge_severity: True, Severity: {severity}, error_type_b: {error_type_b}")
                if severity is None:
                    severity = infer_severity_3_from_error_type(error_type_b)
                    print(f"use_model_judge_severity: False, Severity: {severity}, error_type_b: {error_type_b}")
                caption_b_severity_counts[severity] += 1
                if self.use_severity_weighted_reward:
                    weight = self.severity_weight[severity]
                    caption_b_error_count_weighted += weight
                else:
                    caption_b_error_count_weighted += 1.0

        # Calculate error difference based on configured mode
        # Use weighted counts for reward calculation if flag is enabled, otherwise use unweighted
        # This applies to all modes including normalized_diff
        error_count_a_for_reward = caption_a_error_count_weighted if self.use_severity_weighted_reward else float(caption_a_error_count)
        error_count_b_for_reward = caption_b_error_count_weighted if self.use_severity_weighted_reward else float(caption_b_error_count)
        
        # Calculate both unweighted and weighted error differences for logging and return
        raw_error_difference_unweighted = float(caption_a_error_count - caption_b_error_count)
        raw_error_difference = error_count_a_for_reward - error_count_b_for_reward
        
        # Always count judgments for result_dict (regardless of mode)
        judgment_a_wins = 0
        judgment_b_wins = 0
        judgment_neutral = 0  # both_wrong, both_supported, or unknown
        judgment_diff = 0.0  # Initialize for logging
        
        # Count judgments: A wins, B wins, and neutral cases
        for diff in result.get("differences", []):
            judgment = diff.get("judgment", "unknown").strip().upper()
            if judgment == "A":
                judgment_a_wins += 1
            elif judgment == "B":
                judgment_b_wins += 1
            else:
                # both_wrong, both_supported, or unknown - count as neutral
                judgment_neutral += 1
        
        # Calculate overall_winner score: A=1, B=0, Tie=0.5
        overall_winner = result.get("overall_winner", "NA").strip().upper()
        if overall_winner == "A":
            overall_winner_score = 1.0
        elif overall_winner == "B":
            overall_winner_score = 0.0
        elif overall_winner == "TIE":
            overall_winner_score = 0.5
        else:
            # NA or unknown - default to 0.5 (neutral)
            overall_winner_score = 0.5
        
        if self.error_diff_mode == "judgment_based":
            
            # Calculate judgment difference: (A_wins - B_wins) / total_differences
            # This gives a value in [-1, 1] range where:
            #   -1 = all differences favor B (worst for A)
            #    0 = equal or neutral
            #   +1 = all differences favor A (best for A)
            if total_differences > 0:
                judgment_diff = (judgment_a_wins - judgment_b_wins) / total_differences
                
                # Consider overall_winner as a tie-breaker/additional signal
                # Apply consistent contribution regardless of agreement/disagreement with judgment_diff
                overall_winner_contribution = 0.1  # Fixed contribution value
                if overall_winner == "A":
                    # Overall winner is A - boost A by fixed amount
                    judgment_diff = min(1.0, judgment_diff + overall_winner_contribution)
                elif overall_winner == "B":
                    # Overall winner is B - reduce A by fixed amount
                    judgment_diff = max(-1.0, judgment_diff - overall_winner_contribution)
                # For "Tie" or "NA", no adjustment
                
                # Invert for reward calculation: negative judgment_diff (A worse) -> lower reward
                # Map judgment_diff from [-1, 1] to error_difference format
                # We want: judgment_diff = -1 (A worst) -> error_difference = +1 (maps to reward = 0)
                #          judgment_diff = +1 (A best) -> error_difference = -1 (maps to reward = 1)
                error_difference = -judgment_diff  # Invert so A better -> negative (higher reward)
            else:
                # No differences found - treat as neutral (reward = 0.5)
                error_difference = 0.0
                logger.info("No differences found in response - using neutral judgment-based reward")
        elif self.error_diff_mode == "raw_diff":
            # Use raw difference without normalization
            error_difference = float(raw_error_difference)
        elif self.error_diff_mode == "normalized_diff":
            # Normalized difference: (A - B) / total_differences
            # This prevents "hacking" by having fewer differences identified overall
            # Uses severity-weighted error counts if USE_SEVERITY_WEIGHTED_REWARD is enabled
            if total_differences > 0:
                if self.use_severity_weighted_reward:
                    # When using severity weights, normalize by max possible weighted difference
                    # Max weight is 1.6, so max possible weighted difference per difference is 1.6
                    # Normalize to [-1, 1] range by dividing by max_weight
                    max_weight = max(self.severity_weight.values())
                    error_difference = raw_error_difference / (max_weight * total_differences)
                else:
                    # Without severity weights, simple normalization by total_differences
                    error_difference = raw_error_difference / total_differences
            else:
                # When total_differences == 0, no differences found means both captions are consistent
                # This should be rewarded (model is as good as Gemini)
                # Set error_difference to -1.0 to indicate best case (maps to reward = 1.0)
                error_difference = 0.0
                logger.info("No differences found in response - captions are consistent, rewarding with maximum score")
        elif self.error_diff_mode == "normalized_a_only":
            # Only Model A errors normalized by total differences: A_errors / total_differences
            if total_differences > 0:
                if self.use_severity_weighted_reward:
                    # When using severity weights, normalize by max possible weighted difference
                    max_weight = max(self.severity_weight.values())
                    error_difference = error_count_a_for_reward / (max_weight * total_differences)
                else:
                    error_difference = error_count_a_for_reward / total_differences
            else:
                # When total_differences == 0, no differences found means both captions are consistent
                # This should be rewarded (model is as good as Gemini)
                # Set error_difference to 0.0 (no errors) which maps to maximum reward
                error_difference = 0.0
                logger.info("No differences found in response - captions are consistent, rewarding with maximum score")
        elif self.error_diff_mode == "a_only":
            # Only Model A errors (raw or weighted), NOT normalized by total_differences
            # Uses linear reward mapping like normalized_diff
            # Use raw or weighted error count directly
            if self.use_severity_weighted_reward:
                error_difference = float(error_count_a_for_reward)  # Weighted error count
            else:
                error_difference = float(caption_a_error_count)  # Raw error count
        else:
            logger.warning(f"Unknown error_diff_mode: {self.error_diff_mode}, using normalized_diff")
            if total_differences > 0:
                if self.use_severity_weighted_reward:
                    # When using severity weights, normalize by max possible weighted difference
                    max_weight = max(self.severity_weight.values())
                    error_difference = raw_error_difference / (max_weight * total_differences)
                else:
                    error_difference = raw_error_difference / total_differences
            else:
                # When total_differences == 0, no differences found means both captions are consistent
                # This should be rewarded (model is as good as Gemini)
                # Set error_difference to -1.0 to indicate best case (maps to reward = 1.0)
                error_difference = 0.0
                logger.info("No differences found in response - captions are consistent, rewarding with maximum score")

        # Map error difference to reward score (0 to 1)
        # Pass weighted counts for normalized_a_only mode if using severity weighting
        # For judgment_based mode, pass None for counts as they're not used
        if self.error_diff_mode == "normalized_a_only":
            # For normalized_a_only mode, pass the weighted count (if severity weighting enabled)
            # and max_weight for proper normalization
            max_weight = max(self.severity_weight.values()) if self.use_severity_weighted_reward else 1.0
            reward_score = self._map_error_diff_to_reward(
                float(error_difference), 
                error_count_a_for_reward,  # Use weighted count
                total_differences,
                max_weight=max_weight
            )
        elif self.error_diff_mode == "a_only":
            # For a_only mode, use the same path as normalized_diff (linear mapping)
            # error_difference is raw/weighted error count
            # Pass max_weight to scale max_error_diff appropriately for weighted errors
            max_weight = max(self.severity_weight.values()) if self.use_severity_weighted_reward else 1.0
            reward_score = self._map_error_diff_to_reward(
                float(error_difference), 
                None,  # Not used for a_only mode
                None,  # Not used for a_only mode
                max_weight=max_weight
            )
        else:
            # For other modes (raw_diff, normalized_diff, judgment_based), counts are not used
            # normalized_a_only and a_only are handled in the if branches above
            reward_score = self._map_error_diff_to_reward(
                float(error_difference), 
                None,  # Not used for other modes
                None   # Not used for other modes
            )
        
        # Apply ambiguity penalty if enabled
        if self.use_ambiguity_penalty:
            ambiguity_penalty = math.exp(-self.ambiguity_decay * max(0, ambiguity_count - ambiguity_free))
            reward_score *= ambiguity_penalty
        else:
            ambiguity_penalty = 1.0  # No penalty when disabled
            ambiguity_free = 0.0  # Not used when penalty is disabled

        if should_log:
            logger.info(f"Error difference mode: {self.error_diff_mode}")
            logger.info(f"Use severity weighted reward: {self.use_severity_weighted_reward}")
            logger.info(f"Caption A error count (unweighted): {caption_a_error_count}")
            logger.info(f"Caption B error count (unweighted): {caption_b_error_count}")
            logger.info(f"Error difference (unweighted, A - B): {raw_error_difference_unweighted:.4f}")
            # Log weighted counts if flag is enabled
            if self.use_severity_weighted_reward:
                logger.info(f"Caption A error count (weighted): {caption_a_error_count_weighted:.4f}")
                logger.info(f"Caption B error count (weighted): {caption_b_error_count_weighted:.4f}")
                logger.info(f"Error difference (weighted, A - B): {raw_error_difference:.4f}")
            # Log severity level counts
            logger.info(f"Caption A severity counts: {caption_a_severity_counts}")
            logger.info(f"Caption B severity counts: {caption_b_severity_counts}")
            logger.info(f"Total differences: {total_differences}")
            if self.error_diff_mode == "judgment_based":
                logger.info(f"Judgment A wins: {judgment_a_wins}")
                logger.info(f"Judgment B wins: {judgment_b_wins}")
                logger.info(f"Judgment neutral: {judgment_neutral}")
                logger.info(f"Judgment difference (A - B) / total: {judgment_diff:.4f}")
                logger.info(f"Error difference (inverted judgment): {error_difference:.4f}")
            else:
                logger.info(f"Raw error difference (A - B): {raw_error_difference:.4f}")
                if self.error_diff_mode == "normalized_a_only":
                    logger.info(f"Error difference (A / total): {error_difference:.4f}")
                elif self.error_diff_mode == "a_only":
                    logger.info(f"Error difference (A raw/weighted): {error_difference:.4f}")
                elif self.error_diff_mode == "normalized_diff":
                    logger.info(f"Error difference ((A - B) / total): {error_difference:.4f}")
                else:
                    logger.info(f"Error difference (A - B): {error_difference:.4f}")
            logger.info(f"Caption A error types: {sorted(error_types_a)}")
            logger.info(f"Caption B error types: {sorted(error_types_b)}")
            # Log ambiguity information
            logger.info(f"Ambiguity count: {ambiguity_count}")
            if self.use_ambiguity_penalty:
                logger.info(f"Ambiguity free tokens: {ambiguity_free:.2f} (calculated from caption length: {len(predict_caption.split())} words)")
                logger.info(f"Ambiguity penalty: {ambiguity_penalty:.4f}")
            logger.info(f"Reward score: {reward_score:.4f}")
            logger.info(f"Overall winner: {result.get('overall_winner', 'NA')}")
            logger.info(f"Number of parsing failures (format validation): {num_parsing_failures}")

        # Return success result with all information
        result_dict = {
            "caption_a_error_count": caption_a_error_count,
            "caption_b_error_count": caption_b_error_count,
            "total_differences": total_differences,
            "normalized_error_difference": error_difference,
            "final_score": reward_score,  # For compatibility with existing code
            "ambiguity_count": ambiguity_count,
            "num_parsing_failures": num_parsing_failures,  # Number of parsing/format validation failures (when Gemini doesn't follow format)
            # Separate fields for each severity level for easy plotting
            "caption_a_severity_1_count": caption_a_severity_counts[1],
            "caption_a_severity_2_count": caption_a_severity_counts[2],
            "caption_a_severity_3_count": caption_a_severity_counts[3],
            "caption_b_severity_1_count": caption_b_severity_counts[1],
            "caption_b_severity_2_count": caption_b_severity_counts[2],
            "caption_b_severity_3_count": caption_b_severity_counts[3],
            # Judgment counts and overall winner
            "judgment_a_wins": judgment_a_wins,
            "judgment_b_wins": judgment_b_wins,
            "judgment_neutral": judgment_neutral,
            "overall_winner_score": overall_winner_score,  # Numeric score: A=1.0, B=0.0, Tie=0.5
        }
        
        # Add weighted counts and differences if severity weighting is enabled
        if self.use_severity_weighted_reward:
            result_dict["caption_a_error_count_weighted"] = caption_a_error_count_weighted
            result_dict["caption_b_error_count_weighted"] = caption_b_error_count_weighted
            result_dict["error_difference_unweighted"] = raw_error_difference_unweighted
            result_dict["error_difference_weighted"] = raw_error_difference
        
        # Add ambiguity penalty if enabled
        if self.use_ambiguity_penalty:
            result_dict["ambiguity_penalty"] = ambiguity_penalty
            result_dict["ambiguity_free"] = ambiguity_free
        
        # Save training data if enabled
        if self.save_training_data and response_text_final is not None:
            # Determine image path(s) to save
            image_path_to_save = image_path if image_path is not None else self.image_paths
            if isinstance(image_path_to_save, str):
                image_path_to_save = [image_path_to_save]
            elif image_path_to_save is None:
                image_path_to_save = []
            
            meta = {
                "iter": iter,
                "data_index": data_index,
                "image_path": image_path_to_save
            }
            self._save_training_data_to_jsonl(
                meta=meta,
                actor_caption=predict_caption,
                prompt=prompt,
                gemini_response=response_text_final,
                verification_result=result_dict
            )
        
        return result_dict

    def verify_format(self, predict_str: str) -> float:
        if self.require_think_block:
            return 1.0 if _has_think_and_answer_blocks(predict_str) else 0.0
        # If not required, accept any non-empty answer
        return 1.0 if _extract_answer_text(predict_str) else 0.0

    @staticmethod
    async def _verify_accuracy_async_internal(
        verifier: 'GeminiCaptionDiffVerifier',
        predict_caption: str,
        solution_caption: str,
        session: aiohttp.ClientSession = None,
        iter: int = None,
        data_index: int = None,
        image_path: str | List[str] = None
    ) -> dict:
        """Internal async helper that uses async requests but maintains verify_accuracy logic"""
        failed_result = {
            "caption_a_error_count": 0,
            "caption_b_error_count": 0,
            "total_differences": 0,
            "error_difference": 0,
            "final_score": 0.0,
            "num_parsing_failures": 0,
        }

        print(f"predict_caption before extract_answer_text: {predict_caption}")
        predict_caption = _extract_answer_text(predict_caption)
        print(f"predict_caption after extract_answer_text: {predict_caption}")

        # Check if solution_caption (stronger model's caption) is provided
        if solution_caption is None or not solution_caption.strip():
            logger.warning("solution_caption (stronger model's caption) is required but not provided")
            return failed_result

        # Determine if we should log this request (sample or always log failures)
        should_log = random.random() < verifier.log_sample_rate

        # Check if image_paths is valid
        if not verifier.image_paths:
            logger.warning("image_paths is empty, cannot evaluate caption")
            return failed_result
        
        # Check if all images exist
        for image_path in verifier.image_paths:
            if not os.path.exists(image_path):
                logger.warning(f"image_path does not exist: {image_path}")
                return failed_result

        # Build prompt with both captions
        if verifier.error_diff_mode == "hall_vs_miss":
            prompt = HALL_VS_MISS_PROMPT.format(
                gt_caption=solution_caption,
                pred_caption=predict_caption
            )
        else:
            prompt = verifier.diff_prompt_template.format(
                caption_a=predict_caption,
                caption_b=solution_caption
            )

        # Retry logic with adaptive backoff for rate limits
        max_retries = verifier.max_retries
        result = None
        response_text_final = None  # Store final successful response for saving
        consecutive_rate_limits = 0  # Track consecutive rate limits for adaptive backoff
        num_parsing_failures = 0  # Track number of parsing/format validation failures (when Gemini doesn't follow format)
        
        for attempt in range(max_retries):
            # Set temperature: 0.0 for first attempt, random for retries
            temperature = 0.0 if attempt == 0 else random.uniform(0, 1)

            if should_log:
                logger.info(
                    f"Attempt {attempt + 1}/{max_retries} with temperature={temperature:.2f}"
                )

            # Send async request to Gemini with all images
            response_text, usage_stats = await verifier._send_gemini_request_async(
                prompt, verifier.image_paths, temperature, session
            )

            # Handle rate limit errors with adaptive backoff
            if usage_stats is not None and isinstance(usage_stats, dict) and usage_stats.get("rate_limited"):
                consecutive_rate_limits += 1
                # Adaptive backoff: start small, increase if multiple consecutive rate limits
                # Base wait: 0.5-1.5s, scales with consecutive rate limits, max 10s
                base_wait = 0.5 + random.uniform(0, 1.0)  # 0.5-1.5s base
                scale_factor = min(consecutive_rate_limits, 5)  # Cap scaling at 5x
                wait_time = min(base_wait * scale_factor, 10.0)  # Max 10s
                logger.warning(
                    f"Rate limit hit (attempt {attempt + 1}/{max_retries}, consecutive={consecutive_rate_limits}), "
                    f"waiting {wait_time:.2f}s before retry"
                )
                await asyncio.sleep(wait_time)
                continue
            else:
                consecutive_rate_limits = 0  # Reset counter on success or non-rate-limit error

            # Handle timeout errors with a moderate wait (server may be slow)
            if usage_stats is not None and isinstance(usage_stats, dict) and usage_stats.get("timeout"):
                logger.warning(
                    f"Request timeout (attempt {attempt + 1}/{max_retries}), "
                    f"waiting 2-4s before retry"
                )
                await asyncio.sleep(3.0 + random.uniform(0, 2.0))  # 2-4s wait
                continue

            if usage_stats is None or response_text is None:
                logger.warning(
                    f"Gemini request failed (attempt {attempt + 1}): {response_text}"
                )
                if attempt < max_retries - 1:
                    # Brief wait before retry for non-rate-limit errors (0.2-0.4s)
                    await asyncio.sleep(0.2 + random.uniform(0, 0.2))
                continue

            if should_log:
                logger.info(f"Gemini request time: {usage_stats.get('request_time', 0):.2f}s")
                if 'prompt_tokens' in usage_stats:
                    logger.info(
                        f"Token usage: prompt_tokens={usage_stats['prompt_tokens']}, total_tokens={usage_stats['total_tokens']}"
                    )

            # Parse response
            try:
                if verifier.error_diff_mode == "hall_vs_miss":
                    parsed = _parse_hall_vs_miss_response(response_text)
                    if not parsed:
                        num_parsing_failures += 1
                        logger.warning(
                            f"Parsing failure: Gemini response is not valid JSON (attempt {attempt + 1}/{max_retries}). "
                            f"Response preview: {response_text[:300]}..."
                        )
                        if attempt < max_retries - 1:
                            await asyncio.sleep(0.3 + random.uniform(0, 0.2))
                            continue
                        failed_result["num_parsing_failures"] = num_parsing_failures
                        return failed_result
                    result = parsed
                    response_text_final = response_text
                    break
                else:
                    result = verifier._parse_diff_text_response(response_text)
                    # Validate that the parsed result follows the expected format
                    is_valid, validation_error = verifier._validate_parsed_result(result)
                    if not is_valid:
                        num_parsing_failures += 1  # Count format validation failure (Gemini didn't follow format)
                        logger.warning(
                            f"Parsing failure: Gemini response does not follow required format (attempt {attempt + 1}/{max_retries}): {validation_error}. "
                            f"Response preview: {response_text[:300]}..."
                        )
                        if attempt < max_retries - 1:
                            # Wait briefly before retrying with stricter prompt
                            await asyncio.sleep(0.3 + random.uniform(0, 0.2))
                            continue
                        else:
                            logger.error(f"All {max_retries} attempts failed - format validation failed: {validation_error}")
                            failed_result["num_parsing_failures"] = num_parsing_failures
                            return failed_result
                    # Format validation passed
                    if result and "differences" in result:
                        response_text_final = response_text  # Store successful response
                        # num_parsing_failures already counted for each format validation failure
                        break
            except Exception as e:
                num_parsing_failures += 1  # Count parse exception as a parsing failure
                logger.warning(f"Parsing failure: Failed to parse response (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    logger.error(f"All {max_retries} attempts failed")
                    failed_result["num_parsing_failures"] = num_parsing_failures
                    return failed_result
                # Wait briefly before retrying
                await asyncio.sleep(0.3 + random.uniform(0, 0.2))
                continue
        else:
            logger.error(f"All {max_retries} attempts failed or no valid result")
            failed_result["num_parsing_failures"] = num_parsing_failures
            return failed_result

        if verifier.error_diff_mode == "hall_vs_miss":
            # Count ambiguity tokens in predict_caption
            ambiguity_count = count_ambiguity_tokens(predict_caption)
            if verifier.use_ambiguity_penalty:
                if verifier.use_fixed_ambiguity_free:
                    ambiguity_free = verifier.ambiguity_free_fixed
                else:
                    caption_word_count = len(predict_caption.split())
                    ambiguity_free = caption_word_count / verifier.ambiguity_free_words_per_token
                    ambiguity_free = max(0.0, ambiguity_free)
            else:
                ambiguity_free = 0.0
            total_differences, hallucination_count, missing_count, extra_count = _compute_hall_vs_miss_metrics(result)
            penalty = hallucination_count + missing_count
            if total_differences > 0:
                error_difference = penalty / total_differences
            else:
                error_difference = 0.0
            base_reward = 1.0 - max(0.0, min(1.0, error_difference))
            if verifier.use_ambiguity_penalty:
                ambiguity_penalty = math.exp(-verifier.ambiguity_decay * max(0, ambiguity_count - ambiguity_free))
                base_reward *= ambiguity_penalty
            else:
                ambiguity_penalty = 1.0
            if should_log:
                type_counts, verification_counts = _compute_hall_vs_miss_stats(result)
                logger.info(f"hall_vs_miss: total_differences={total_differences}")
                logger.info(f"hall_vs_miss: hallucination_count={hallucination_count}, missing_fact_count={missing_count}, extra_info_count={extra_count}")
                logger.info(f"hall_vs_miss: type_counts={type_counts}")
                logger.info(f"hall_vs_miss: verification_counts={verification_counts}")
                logger.info(f"hall_vs_miss: error_difference={error_difference:.4f}")
                if verifier.use_ambiguity_penalty:
                    logger.info(f"hall_vs_miss: ambiguity_count={ambiguity_count}, ambiguity_free={ambiguity_free:.2f}, ambiguity_penalty={ambiguity_penalty:.4f}")
            scaled_reward = base_reward * verifier.reward_scale
            scaled_reward = max(0.0, min(1.0, scaled_reward))
            reward_score = verifier.min_reward + scaled_reward * (verifier.max_reward - verifier.min_reward)
            reward_score = max(verifier.min_reward, min(verifier.max_reward, reward_score))
            if should_log:
                logger.info(f"hall_vs_miss: reward_score={reward_score:.4f}")

            result_dict = {
                "caption_a_error_count": 0,
                "caption_b_error_count": 0,
                "total_differences": total_differences,
                "normalized_error_difference": error_difference,
                "final_score": reward_score,
                "num_parsing_failures": num_parsing_failures,
                "hallucination_count": hallucination_count,
                "missing_fact_count": missing_count,
                "extra_info_count": extra_count,
                "ambiguity_count": ambiguity_count,
            }
            if verifier.use_ambiguity_penalty:
                result_dict["ambiguity_penalty"] = ambiguity_penalty
                result_dict["ambiguity_free"] = ambiguity_free

            if verifier.save_training_data and response_text_final is not None:
                image_path_to_save = image_path if image_path is not None else verifier.image_paths
                if isinstance(image_path_to_save, str):
                    image_path_to_save = [image_path_to_save]
                elif image_path_to_save is None:
                    image_path_to_save = []
                meta = {
                    "iter": iter,
                    "data_index": data_index,
                    "image_path": image_path_to_save
                }
                verifier._save_training_data_to_jsonl(
                    meta=meta,
                    actor_caption=predict_caption,
                    prompt=prompt,
                    gemini_response=response_text_final,
                    verification_result=result_dict
                )
            return result_dict

        if not result or "differences" not in result:
            logger.error("Invalid result format: missing 'differences'")
            failed_result["num_parsing_failures"] = num_parsing_failures
            return failed_result

        # Count errors for each caption
        # Error is counted if error_type is not "NONE" and not "NA"
        # Track both unweighted (original) and weighted error counts
        caption_a_error_count = 0  # Unweighted count (returned in result)
        caption_b_error_count = 0  # Unweighted count (returned in result)
        caption_a_error_count_weighted = 0.0  # Weighted count (used for reward if flag enabled)
        caption_b_error_count_weighted = 0.0  # Weighted count (used for reward if flag enabled)
        error_types_a = set()
        error_types_b = set()
        # Track severity level counts for each caption
        caption_a_severity_counts = {1: 0, 2: 0, 3: 0}  # Count errors by severity level
        caption_b_severity_counts = {1: 0, 2: 0, 3: 0}  # Count errors by severity level
        # total_differences = 0  # Only count differences where at least one caption has an error
        total_differences = len(result.get("differences", []))
        
        # Count ambiguity tokens in predict_caption
        ambiguity_count = count_ambiguity_tokens(predict_caption)
        
        # Calculate dynamic ambiguity_free based on caption length
        # If using fixed value (legacy), use it; otherwise calculate from caption word count
        if verifier.use_ambiguity_penalty:
            if verifier.use_fixed_ambiguity_free:
                ambiguity_free = verifier.ambiguity_free_fixed
            else:
                # Calculate caption word count (split by whitespace)
                caption_word_count = len(predict_caption.split())
                # Calculate free ambiguity tokens: 1 token per N words
                ambiguity_free = caption_word_count / verifier.ambiguity_free_words_per_token
                # Ensure minimum of 0 (can't be negative)
                ambiguity_free = max(0.0, ambiguity_free)
        else:
            ambiguity_free = 0.0  # Not used when penalty is disabled

        for diff in result.get("differences", []):
            # Check caption A error
            caption_a_error = diff.get("caption_a_error", {})
            error_type_a = caption_a_error.get("error_type", "NA")
            has_error_a = error_type_a and error_type_a.upper() not in ["NONE", "NA"]
            
            # Check caption B error
            caption_b_error = diff.get("caption_b_error", {})
            error_type_b = caption_b_error.get("error_type", "NA")
            has_error_b = error_type_b and error_type_b.upper() not in ["NONE", "NA"]
            
            # Only count this difference if at least one caption has an error
            # if has_error_a or has_error_b:
            #     total_differences += 1
            
            # Count caption A errors (unweighted and weighted)
            if has_error_a:
                caption_a_error_count += 1
                error_types_a.add(error_type_a)
                # Always calculate severity to track severity level counts
                severity = None
                if verifier.use_model_judge_severity:
                    severity = parse_model_severity_level(caption_a_error.get("severity_level", "NA"))
                    print(f"use_model_judge_severity: True, Severity: {severity}, error_type_a: {error_type_a}")
                if severity is None:
                    severity = infer_severity_3_from_error_type(error_type_a)
                    print(f"use_model_judge_severity: False, Severity: {severity}, error_type_a: {error_type_a}")
                caption_a_severity_counts[severity] += 1
                if verifier.use_severity_weighted_reward:
                    weight = verifier.severity_weight[severity]
                    caption_a_error_count_weighted += weight
                else:
                    caption_a_error_count_weighted += 1.0

            # Count caption B errors (unweighted and weighted)
            if has_error_b:
                caption_b_error_count += 1
                error_types_b.add(error_type_b)
                # Always calculate severity to track severity level counts
                severity = None
                if verifier.use_model_judge_severity:
                    severity = parse_model_severity_level(caption_b_error.get("severity_level", "NA"))
                    print(f"use_model_judge_severity: True, Severity: {severity}, error_type_b: {error_type_b}")
                if severity is None:
                    severity = infer_severity_3_from_error_type(error_type_b)
                    print(f"use_model_judge_severity: False, Severity: {severity}, error_type_b: {error_type_b}")
                caption_b_severity_counts[severity] += 1
                if verifier.use_severity_weighted_reward:
                    weight = verifier.severity_weight[severity]
                    caption_b_error_count_weighted += weight
                else:
                    caption_b_error_count_weighted += 1.0

        # Calculate error difference based on configured mode
        # Use weighted counts for reward calculation if flag is enabled, otherwise use unweighted
        # This applies to all modes including normalized_diff
        error_count_a_for_reward = caption_a_error_count_weighted if verifier.use_severity_weighted_reward else float(caption_a_error_count)
        error_count_b_for_reward = caption_b_error_count_weighted if verifier.use_severity_weighted_reward else float(caption_b_error_count)
        
        # Calculate both unweighted and weighted error differences for logging and return
        raw_error_difference_unweighted = float(caption_a_error_count - caption_b_error_count)
        raw_error_difference = error_count_a_for_reward - error_count_b_for_reward
        
        # Always count judgments for result_dict (regardless of mode)
        judgment_a_wins = 0
        judgment_b_wins = 0
        judgment_neutral = 0  # both_wrong, both_supported, or unknown
        judgment_diff = 0.0  # Initialize for logging
        
        # Count judgments: A wins, B wins, and neutral cases
        for diff in result.get("differences", []):
            judgment = diff.get("judgment", "unknown").strip().upper()
            if judgment == "A":
                judgment_a_wins += 1
            elif judgment == "B":
                judgment_b_wins += 1
            else:
                # both_wrong, both_supported, or unknown - count as neutral
                judgment_neutral += 1
        
        # Calculate overall_winner score: A=1, B=0, Tie=0.5
        overall_winner = result.get("overall_winner", "NA").strip().upper()
        if overall_winner == "A":
            overall_winner_score = 1.0
        elif overall_winner == "B":
            overall_winner_score = 0.0
        elif overall_winner == "TIE":
            overall_winner_score = 0.5
        else:
            # NA or unknown - default to 0.5 (neutral)
            overall_winner_score = 0.5
        
        if verifier.error_diff_mode == "judgment_based":
            # Calculate judgment difference: (A_wins - B_wins) / total_differences
            # This gives a value in [-1, 1] range where:
            #   -1 = all differences favor B (worst for A)
            #    0 = equal or neutral
            #   +1 = all differences favor A (best for A)
            if total_differences > 0:
                judgment_diff = (judgment_a_wins - judgment_b_wins) / total_differences
                
                # Consider overall_winner as a tie-breaker/additional signal
                # Apply consistent contribution regardless of agreement/disagreement with judgment_diff
                overall_winner_contribution = 0.1  # Fixed contribution value
                if overall_winner == "A":
                    # Overall winner is A - boost A by fixed amount
                    judgment_diff = min(1.0, judgment_diff + overall_winner_contribution)
                elif overall_winner == "B":
                    # Overall winner is B - reduce A by fixed amount
                    judgment_diff = max(-1.0, judgment_diff - overall_winner_contribution)
                # For "Tie" or "NA", no adjustment
                
                # Invert for reward calculation: negative judgment_diff (A worse) -> lower reward
                # Map judgment_diff from [-1, 1] to error_difference format
                # We want: judgment_diff = -1 (A worst) -> error_difference = +1 (maps to reward = 0)
                #          judgment_diff = +1 (A best) -> error_difference = -1 (maps to reward = 1)
                error_difference = -judgment_diff  # Invert so A better -> negative (higher reward)
            else:
                # No differences found - treat as neutral (reward = 0.5)
                error_difference = 0.0
                logger.info("No differences found in response - using neutral judgment-based reward")
        elif verifier.error_diff_mode == "raw_diff":
            # Use raw difference without normalization
            error_difference = float(raw_error_difference)
        elif verifier.error_diff_mode == "normalized_diff":
            # Normalized difference: (A - B) / total_differences
            # This prevents "hacking" by having fewer differences identified overall
            # Uses severity-weighted error counts if USE_SEVERITY_WEIGHTED_REWARD is enabled
            if total_differences > 0:
                if verifier.use_severity_weighted_reward:
                    # When using severity weights, normalize by max possible weighted difference
                    # Max weight is 1.6, so max possible weighted difference per difference is 1.6
                    # Normalize to [-1, 1] range by dividing by max_weight
                    max_weight = max(verifier.severity_weight.values())
                    error_difference = raw_error_difference / (max_weight * total_differences)
                else:
                    # Without severity weights, simple normalization by total_differences
                    error_difference = raw_error_difference / total_differences
            else:
                # When total_differences == 0, no differences found means both captions are consistent
                # This should be rewarded (model is as good as Gemini)
                # Set error_difference to -1.0 to indicate best case (maps to reward = 1.0)
                error_difference = 0.0
                logger.info("No differences found in response - captions are consistent, rewarding with maximum score")
                logger.warning("No differences found in response, using raw error difference")
        elif verifier.error_diff_mode == "normalized_a_only":
            # Only Model A errors normalized by total differences: A_errors / total_differences
            if total_differences > 0:
                if verifier.use_severity_weighted_reward:
                    max_weight = max(verifier.severity_weight.values())
                    error_difference = error_count_a_for_reward / (max_weight * total_differences)
                else:       
                    error_difference = error_count_a_for_reward / total_differences
            else:
                # When total_differences == 0, no differences found means both captions are consistent
                # This should be rewarded (model is as good as Gemini)
                # Set error_difference to 0.0 (no errors) which maps to maximum reward
                error_difference = 0.0
                logger.info("No differences found in response - captions are consistent, rewarding with maximum score")
        elif verifier.error_diff_mode == "a_only":
            # Only Model A errors (raw or weighted), NOT normalized by total_differences
            # Uses linear reward mapping like normalized_diff
            # Use raw or weighted error count directly
            if verifier.use_severity_weighted_reward:
                error_difference = float(error_count_a_for_reward)  # Weighted error count
            else:
                error_difference = float(caption_a_error_count)  # Raw error count
        else:
            logger.warning(f"Unknown error_diff_mode: {verifier.error_diff_mode}, using normalized_diff")
            if total_differences > 0:
                if verifier.use_severity_weighted_reward:
                    # When using severity weights, normalize by max possible weighted difference
                    max_weight = max(verifier.severity_weight.values())
                    error_difference = raw_error_difference / (max_weight * total_differences)
                else:
                    error_difference = raw_error_difference / total_differences
            else:
                error_difference = float(raw_error_difference)

        # Map error difference to reward score (0 to 1)
        # Pass weighted counts for normalized_a_only mode if using severity weighting
        # For judgment_based mode, pass None for counts as they're not used
        if verifier.error_diff_mode == "normalized_a_only":
            # For normalized_a_only mode, pass the weighted count (if severity weighting enabled)
            # and max_weight for proper normalization
            max_weight = max(verifier.severity_weight.values()) if verifier.use_severity_weighted_reward else 1.0
            reward_score = verifier._map_error_diff_to_reward(
                float(error_difference), 
                error_count_a_for_reward,  # Use weighted count
                total_differences,
                max_weight=max_weight
            )
        elif verifier.error_diff_mode == "a_only":
            # For a_only mode, use the same path as normalized_diff (linear mapping)
            # error_difference is raw/weighted error count
            # Pass max_weight to scale max_error_diff appropriately for weighted errors
            max_weight = max(verifier.severity_weight.values()) if verifier.use_severity_weighted_reward else 1.0
            reward_score = verifier._map_error_diff_to_reward(
                float(error_difference), 
                None,  # Not used for a_only mode
                None,  # Not used for a_only mode
                max_weight=max_weight
            )
        else:
            # For other modes (raw_diff, normalized_diff, judgment_based), counts are not used
            # normalized_a_only and a_only are handled in the if branches above
            reward_score = verifier._map_error_diff_to_reward(
                float(error_difference), 
                None,  # Not used for other modes
                None   # Not used for other modes
            )
        
        # Apply ambiguity penalty if enabled
        if verifier.use_ambiguity_penalty:
            ambiguity_penalty = math.exp(-verifier.ambiguity_decay * max(0, ambiguity_count - ambiguity_free))
            reward_score *= ambiguity_penalty
        else:
            ambiguity_penalty = 1.0  # No penalty when disabled
            ambiguity_free = 0.0  # Not used when penalty is disabled

        if should_log:
            logger.info(f"Error difference mode: {verifier.error_diff_mode}")
            logger.info(f"Use severity weighted reward: {verifier.use_severity_weighted_reward}")
            logger.info(f"Caption A error count (unweighted): {caption_a_error_count}")
            logger.info(f"Caption B error count (unweighted): {caption_b_error_count}")
            logger.info(f"Error difference (unweighted, A - B): {raw_error_difference_unweighted:.4f}")
            # Log weighted counts if flag is enabled
            if verifier.use_severity_weighted_reward:
                logger.info(f"Caption A error count (weighted): {caption_a_error_count_weighted:.4f}")
                logger.info(f"Caption B error count (weighted): {caption_b_error_count_weighted:.4f}")
                logger.info(f"Error difference (weighted, A - B): {raw_error_difference:.4f}")
            # Log severity level counts
            logger.info(f"Caption A severity counts: {caption_a_severity_counts}")
            logger.info(f"Caption B severity counts: {caption_b_severity_counts}")
            logger.info(f"Total differences: {total_differences}")
            if verifier.error_diff_mode == "judgment_based":
                logger.info(f"Judgment A wins: {judgment_a_wins}")
                logger.info(f"Judgment B wins: {judgment_b_wins}")
                logger.info(f"Judgment neutral: {judgment_neutral}")
                logger.info(f"Judgment difference (A - B) / total: {judgment_diff:.4f}")
                logger.info(f"Error difference (inverted judgment): {error_difference:.4f}")
            else:
                logger.info(f"Raw error difference (A - B): {raw_error_difference:.4f}")
                if verifier.error_diff_mode == "normalized_a_only":
                    logger.info(f"Error difference (A / total): {error_difference:.4f}")
                elif verifier.error_diff_mode == "a_only":
                    logger.info(f"Error difference (A raw/weighted): {error_difference:.4f}")
                elif verifier.error_diff_mode == "normalized_diff":
                    logger.info(f"Error difference ((A - B) / total): {error_difference:.4f}")
                else:
                    logger.info(f"Error difference (A - B): {error_difference:.4f}")
            logger.info(f"Caption A error types: {sorted(error_types_a)}")
            logger.info(f"Caption B error types: {sorted(error_types_b)}")
            # Log ambiguity information
            logger.info(f"Ambiguity count: {ambiguity_count}")
            if verifier.use_ambiguity_penalty:
                logger.info(f"Ambiguity free tokens: {ambiguity_free:.2f} (calculated from caption length: {len(predict_caption.split())} words)")
                logger.info(f"Ambiguity penalty: {ambiguity_penalty:.4f}")
            logger.info(f"Reward score: {reward_score:.4f}")
            logger.info(f"Overall winner: {result.get('overall_winner', 'NA')}")
            logger.info(f"Number of parsing failures (format validation): {num_parsing_failures}")

        # Return success result
        result_dict = {
            "caption_a_error_count": caption_a_error_count,
            "caption_b_error_count": caption_b_error_count,
            "total_differences": total_differences,
            "normalized_error_difference": error_difference,
            "final_score": reward_score,
            "ambiguity_count": ambiguity_count,
            "num_parsing_failures": num_parsing_failures,  # Number of parsing/format validation failures (when Gemini doesn't follow format)
            # Separate fields for each severity level for easy plotting
            "caption_a_severity_1_count": caption_a_severity_counts[1],
            "caption_a_severity_2_count": caption_a_severity_counts[2],
            "caption_a_severity_3_count": caption_a_severity_counts[3],
            "caption_b_severity_1_count": caption_b_severity_counts[1],
            "caption_b_severity_2_count": caption_b_severity_counts[2],
            "caption_b_severity_3_count": caption_b_severity_counts[3],
            # Judgment counts and overall winner
            "judgment_a_wins": judgment_a_wins,
            "judgment_b_wins": judgment_b_wins,
            "judgment_neutral": judgment_neutral,
            "overall_winner_score": overall_winner_score,  # Numeric score: A=1.0, B=0.0, Tie=0.5
        }
        
        # Add weighted counts and differences if severity weighting is enabled
        if verifier.use_severity_weighted_reward:
            result_dict["caption_a_error_count_weighted"] = caption_a_error_count_weighted
            result_dict["caption_b_error_count_weighted"] = caption_b_error_count_weighted
            result_dict["error_difference_unweighted"] = raw_error_difference_unweighted
            result_dict["error_difference_weighted"] = raw_error_difference
        
        # Add ambiguity penalty if enabled
        if verifier.use_ambiguity_penalty:
            result_dict["ambiguity_penalty"] = ambiguity_penalty
            result_dict["ambiguity_free"] = ambiguity_free
        
        # Save training data if enabled
        if verifier.save_training_data and response_text_final is not None:
            # Determine image path(s) to save
            image_path_to_save = image_path if image_path is not None else verifier.image_paths
            if isinstance(image_path_to_save, str):
                image_path_to_save = [image_path_to_save]
            elif image_path_to_save is None:
                image_path_to_save = []
            
            meta = {
                "iter": iter,
                "data_index": data_index,
                "image_path": image_path_to_save
            }
            verifier._save_training_data_to_jsonl(
                meta=meta,
                actor_caption=predict_caption,
                prompt=prompt,
                gemini_response=response_text_final,
                verification_result=result_dict
            )
        
        return result_dict

    @staticmethod
    async def batch_verify_accuracy_async(
        verifiers: List['GeminiCaptionDiffVerifier'],
        predict_captions: List[str],
        solution_captions: List[str],
        max_concurrent: int = 10
    ) -> List[dict]:
        """Process multiple verify_accuracy calls in parallel using async
        
        Args:
            verifiers: List of GeminiCaptionDiffVerifier instances (one per request)
            predict_captions: List of predict captions
            solution_captions: List of solution captions
            max_concurrent: Maximum number of concurrent requests
        
        Returns:
            List of result dicts in the same order as inputs
        """
        if len(verifiers) != len(predict_captions) or len(predict_captions) != len(solution_captions):
            raise ValueError("verifiers, predict_captions, and solution_captions must have the same length")
        
        # Create a shared session for connection pooling
        async with aiohttp.ClientSession() as session:
            # Create semaphore to limit concurrent requests
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def verify_with_semaphore(verifier, predict, solution):
                async with semaphore:
                    return await GeminiCaptionDiffVerifier._verify_accuracy_async_internal(
                        verifier, predict, solution, session
                    )
            
            # Create all tasks
            tasks = [
                verify_with_semaphore(verifier, predict, solution)
                for verifier, predict, solution in zip(verifiers, predict_captions, solution_captions)
            ]
            
            # Execute all tasks in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error processing request {i}: {result}")
                    processed_results.append({
                        "caption_a_error_count": 0,
                        "caption_b_error_count": 0,
                        "error_difference": 0,
                        "final_score": 0.0,
                    })
                else:
                    processed_results.append(result)
            
            return processed_results

    @staticmethod
    def batch_verify_accuracy(
        verifiers: List['GeminiCaptionDiffVerifier'],
        predict_captions: List[str],
        solution_captions: List[str],
        max_workers: int = None
    ) -> List[dict]:
        """Process multiple verify_accuracy calls in parallel
        
        Uses async internally for better performance with connection pooling.
        Maintains the same interface as calling verify_accuracy sequentially.
        
        Args:
            verifiers: List of GeminiCaptionDiffVerifier instances (one per request)
            predict_captions: List of predict captions
            solution_captions: List of solution captions
            max_workers: Maximum number of concurrent requests (defaults to max_concurrent_requests from first verifier)
        
        Returns:
            List of result dicts in the same order as inputs
        """
        if len(verifiers) != len(predict_captions) or len(predict_captions) != len(solution_captions):
            raise ValueError("verifiers, predict_captions, and solution_captions must have the same length")
        
        if max_workers is None:
            max_workers = verifiers[0].max_concurrent_requests if verifiers else 10
        
        # Use asyncio to run the async batch function
        return asyncio.run(
            GeminiCaptionDiffVerifier.batch_verify_accuracy_async(
                verifiers, predict_captions, solution_captions, max_workers
            )
        )

