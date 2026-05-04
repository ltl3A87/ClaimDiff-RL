#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from pathlib import Path
from datasets import Dataset


# ============================================================
# CONFIG
# ============================================================

INPUT_JSONLS = {
    "nature": "/path/to/your/nature.jsonl",
    "design": "/path/to/your/design.jsonl",
    "portrait": "/path/to/your/portrait.jsonl",
}

OUTPUT_DIR = "/path/to/output/"

MAX_IMAGE_TOKEN = 8192

PROMPT_TEXT = "<image> Please provide a detailed description of this image."

IMAGE_BASE_NAME = "/path/to/your/images"


# ============================================================
# UTIL
# ============================================================

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


# ============================================================
# MAIN CONVERSION
# ============================================================

def convert_diff_jsonl_to_parquet():

    converted = []

    for split, jsonl_path in INPUT_JSONLS.items():
        print(f"\nLoading {split}: {jsonl_path}")

        for idx, row in enumerate(load_jsonl(jsonl_path)):

            image_path = os.path.join(IMAGE_BASE_NAME, row.get("arcname"))
            if not image_path or not os.path.exists(image_path):
                print(f"[WARN] image missing: {image_path}")
                continue

            # ------------------------
            # prompt
            # ------------------------
            prompt_list = [
                {
                    "role": "user",
                    "content": PROMPT_TEXT
                }
            ]

            # ------------------------
            # reward model
            # ------------------------
            reward_model_dict = {
                "answer": "",
                "format_ratio": 0.0,
                "ground_truth": row.get("gemini-3-pro-preview_detail", ""),
                "length_ratio": 0.0,
                "style": "caption",
                "verifier": "gemini_caption_diff",
                "verifier_parm": {
                    "image_path": image_path
                }
            }

            # ------------------------
            # extra info
            # ------------------------
            extra_info_dict = {
                "answer": "",
                "data_source": "caption_3k_diff",
                "id": row.get("arcname", idx),
                "image_path": image_path,
                "question": PROMPT_TEXT,
                "split": "train",
                "index": str(idx),
                "category": split,
            }

            converted.append({
                "images": json.dumps([image_path]),
                "data_source": "caption_3k_diff",
                "prompt": json.dumps(prompt_list),
                "ability": "visual_reasoning",
                "reward_model": json.dumps(reward_model_dict),
                "extra_info": json.dumps(extra_info_dict),
                "agent_name": "single_turn_agent",
                "max_image_token": MAX_IMAGE_TOKEN,
            })

    print(f"\nConverted samples: {len(converted)}")

    # ========================================================
    # BUILD DATASET
    # ========================================================

    dataset = Dataset.from_dict({
        k: [x[k] for x in converted]
        for k in converted[0].keys()
    })

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(
        OUTPUT_DIR,
        f"caption_3k_diff_max_img_tokens_{MAX_IMAGE_TOKEN}.parquet"
    )

    dataset.to_parquet(out_path)

    print(f"\nSaved parquet to: {out_path}")
    print(dataset)


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    convert_diff_jsonl_to_parquet()
