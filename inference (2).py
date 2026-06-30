"""
inference.py - Procurement email field extraction with flan-t5-base.

Count-injection: we estimate the number of products in the email and
inject that count into the prompt at inference time, matching training format.
Accuracy on shuffled_dataset.jsonl: 98.4% correct count (5/304 wrong).
"""

import argparse
import json
import os
import re
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from dataset import TARGET_KEYS, build_input_text, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH


def _count_products_in_email(email: str) -> int:
    """Estimate number of distinct products in a procurement email.
    
    Accuracy: 98.4% on 304 real procurement emails.
    Priority order: bullet lists > numbered lists > parenthetical-stripped qty patterns.
    """
    lines = [l.strip() for l in email.splitlines() if l.strip()]

    # 1. Bullet / dash list
    bullet_lines = [l for l in lines if re.match(r'^[-•*]\s+\S', l)]
    if len(bullet_lines) >= 2:
        return len(bullet_lines)

    # 2. Numbered list
    numbered_lines = [l for l in lines if re.match(r'^\d+[.)]\s+\S', l)]
    if len(numbered_lines) >= 2:
        return len(numbered_lines)

    # 3. Strip parenthetical specs like (200g), (500 sheets/box), (25kg bags)
    #    before matching qty patterns, so they don't inflate the count
    cleaned = re.sub(r'\([^)]*\)', '', email)

    qty_pattern = re.compile(
        r'\b\d[\d,]*\s*(?:Mtrs?|Nos?|units?|pcs?|pieces?|kg|kgs?|liters?|ltrs?|'
        r'boxes?|bags?|drums?|rolls?|sheets?|sets?|pairs?|tins?|packs?|packets?|'
        r'bottles?|cartons?|sacks?|coils?|folders?|pads?|reams?|cans?|cartridges?|'
        r'notebooks?|clipboards?|kilograms?)\b',
        re.IGNORECASE
    )
    all_matches = list(qty_pattern.finditer(cleaned))

    if len(all_matches) >= 2:
        return len(all_matches)

    # Single qty match — check if 'and' follows (e.g. "500 pens and 100 boxes")
    if len(all_matches) == 1:
        after = cleaned[all_matches[0].end():]
        if re.search(r'\band\b', after, re.IGNORECASE):
            return 2

    return 1


# ------------------------------------------------------------------
# JSON parsing helpers
# ------------------------------------------------------------------

def _empty_product() -> Dict[str, str]:
    return {key: "" for key in TARGET_KEYS}

def _empty_result() -> List[Dict[str, str]]:
    return [_empty_product()]

def _normalize_product_obj(obj: Dict) -> Dict[str, str]:
    result = _empty_product()
    for key in TARGET_KEYS:
        val = obj.get(key, "") or ""
        result[key] = str(val).strip()
    return result

def _extract_objects_via_regex(text: str) -> List[Dict[str, str]]:
    chunks = re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL) or [text]
    results = []
    for chunk in chunks:
        obj = _empty_product()
        found_any = False
        for key in TARGET_KEYS:
            pattern = r'["\']?' + re.escape(key) + r'["\']?\s*:\s*["\'](.*?)["\']\s*(?:,|}|$)'
            match = re.search(pattern, chunk, flags=re.DOTALL)
            if match:
                obj[key] = match.group(1).strip()
                found_any = True
        if found_any:
            results.append(obj)
    return results if results else _empty_result()

def safe_parse_json(text: str) -> List[Dict[str, str]]:
    if not isinstance(text, str) or not text.strip():
        return _empty_result()
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            normalized = [_normalize_product_obj(i) for i in obj if isinstance(i, dict)]
            return normalized if normalized else _empty_result()
        if isinstance(obj, dict):
            return [_normalize_product_obj(obj)]
    except json.JSONDecodeError:
        pass

    # FIX: model sometimes omits {} around each object, producing a flat
    # stream like ["key":"val","key":"val",...,"key":"val","key":"val",...]
    # Recover by reading TARGET_KEYS-sized chunks of key:value pairs in order.
    try:
        pair_pattern = (
            r'["\']?(' + '|'.join(re.escape(k) for k in TARGET_KEYS) + r')'
            r'["\']?\s*:\s*["\']([^"\']*)["\']'
        )
        pairs = re.findall(pair_pattern, text)
        if pairs:
            objects = []
            for i in range(0, len(pairs), len(TARGET_KEYS)):
                chunk = pairs[i:i + len(TARGET_KEYS)]
                if not chunk:
                    continue
                obj = {k: v for k, v in chunk}
                objects.append(_normalize_product_obj(obj))
            if objects:
                return objects
    except Exception:
        pass

    if text.lstrip().startswith("["):
        complete_chunks = re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL)
        if complete_chunks:
            try:
                obj = json.loads("[" + ",".join(complete_chunks) + "]")
                if isinstance(obj, list):
                    normalized = [_normalize_product_obj(i) for i in obj if isinstance(i, dict)]
                    if normalized:
                        return normalized
            except json.JSONDecodeError:
                pass
    return _extract_objects_via_regex(text)


# ------------------------------------------------------------------
# Extractor
# ------------------------------------------------------------------

class ProcurementExtractor:
    def __init__(self, model_dir: str, device: Optional[str] = None):
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Model directory not found: '{model_dir}'.")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def extract(self, email: str, max_new_tokens: int = MAX_TARGET_LENGTH) -> List[Dict[str, str]]:
        if not isinstance(email, str) or not email.strip():
            return _empty_result()

        n_products = _count_products_in_email(email.strip())
        input_text = build_input_text(email.strip(), n_products=n_products)

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_LENGTH,
        ).to(self.device)

        try:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=4092,
                num_beams=5,
                early_stopping=False,
                no_repeat_ngram_size=0,
                length_penalty=2.5
            )
        except Exception:
            return _empty_result()

        decoded = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return safe_parse_json(decoded)

    def extract_batch(self, emails: List[str], max_new_tokens: int = MAX_TARGET_LENGTH) -> List[List[Dict[str, str]]]:
        return [self.extract(e, max_new_tokens=max_new_tokens) for e in emails]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--email", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    extractor = ProcurementExtractor(args.model_dir, device=args.device)
    email_text = args.email or "".join(line for line in iter(input, ""))
    result = extractor.extract(email_text)
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()