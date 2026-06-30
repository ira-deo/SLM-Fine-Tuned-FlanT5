"""
dataset.py

Handles loading, validation, parsing, and tokenization of the procurement
email -> JSON extraction dataset for flan-t5-base fine-tuning.

KEY FIX: The task prefix now injects the exact product count so the model
knows how many array elements to produce. Without this, a 269-sample
dataset is insufficient for flan-t5-base to infer count from context alone.
"""

import json
import os
from typing import Dict, List, Tuple, Any

from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, PreTrainedTokenizerBase


TARGET_KEYS: List[str] = [
    "product_raw_text",
    "product_brand",
    "product_specifications",
    "order_qty",
    "order_unit_of_measure",
]

MAX_INPUT_LENGTH = 4092  
MAX_TARGET_LENGTH = 4092


def build_input_text(email: str, n_products: int = None) -> str:
    """Build model input. During training, n_products is injected into the
    prompt so the model learns to produce exactly that many array elements.
    During inference, n_products=None falls back to a generic prompt."""
    if n_products is not None:
        prefix = (
            f"Extract procurement order details as a JSON array with exactly "
            f"{n_products} element(s). Each element has keys: product_raw_text, "
            f"product_brand, product_specifications, order_qty, "
            f"order_unit_of_measure. Use empty string for missing fields. Email: "
        )
    else:
        prefix = (
            "Extract procurement order details as a JSON array. Each element "
            "represents one distinct product and has keys: product_raw_text, "
            "product_brand, product_specifications, order_qty, "
            "order_unit_of_measure. Use empty string for missing fields. "
            "Return one array element per product mentioned. Email: "
        )
    return f"{prefix}{email}"


def build_target_json(rec: Dict[str, Any]) -> str:
    arr = [
        {key: prod.get(key, "") for key in TARGET_KEYS}
        for prod in rec["products"]
    ]
    return json.dumps(arr, ensure_ascii=False, separators=(",", ":"))


# ----------------------------------------------------------------------
# Loading / validation (unchanged)
# ----------------------------------------------------------------------

def _read_records(data_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: '{data_path}'.")
    with open(data_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        raise ValueError(f"Dataset file '{data_path}' is empty.")
    records: List[Dict[str, Any]] = []
    if content.lstrip().startswith("["):
        records = json.loads(content)
    else:
        for line in content.splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _validate_and_normalize_product(prod, rec_idx, prod_idx):
    if not isinstance(prod, dict):
        raise ValueError(f"Record {rec_idx}, product {prod_idx} is not a dict.")
    normalized = {}
    for key in TARGET_KEYS:
        val = prod.get(key, "") or ""
        normalized[key] = str(val).strip()
    return normalized


def _validate_and_normalize_record(rec, idx):
    email = str(rec.get("email", "")).strip()
    if not email:
        raise ValueError(f"Record {idx} has empty email.")
    output = rec.get("output", [])
    if not isinstance(output, list) or len(output) == 0:
        raise ValueError(f"Record {idx} has empty/missing output list.")
    products = [_validate_and_normalize_product(p, idx, j) for j, p in enumerate(output)]
    return {"email": email, "products": products}


def load_records(data_path: str) -> List[Dict[str, Any]]:
    raw = _read_records(data_path)
    return [_validate_and_normalize_record(r, i) for i, r in enumerate(raw)]


def records_to_examples(records: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Build input/target pairs. Injects product count into training prompts."""
    inputs = [build_input_text(r["email"], n_products=len(r["products"])) for r in records]
    targets = [build_target_json(r) for r in records]
    return {"input_text": inputs, "target_text": targets}


def get_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(model_name)


def make_tokenize_fn(tokenizer: PreTrainedTokenizerBase):
    def tokenize_fn(batch):
        model_inputs = tokenizer(
            batch["input_text"],
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            padding=False,
        )
        labels = tokenizer(
            text_target=batch["target_text"],
            max_length=MAX_TARGET_LENGTH,
            truncation=True,
            padding=False,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    return tokenize_fn


def load_and_prepare_datasets(
    data_path: str,
    model_name: str = "google/flan-t5-base",
    test_size: float = 0.1,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, PreTrainedTokenizerBase]:
    records = load_records(data_path)
    n_eval = max(1, int(round(len(records) * test_size)))
    train_records, eval_records = train_test_split(
        records, test_size=n_eval / len(records), random_state=seed, shuffle=True
    )
    tokenizer = get_tokenizer(model_name)
    tokenize_fn = make_tokenize_fn(tokenizer)

    train_dataset = Dataset.from_dict(records_to_examples(train_records))
    eval_dataset = Dataset.from_dict(records_to_examples(eval_records))

    train_dataset = train_dataset.map(
        tokenize_fn, batched=True,
        remove_columns=["input_text", "target_text"],
        desc="Tokenizing train split",
    )
    eval_dataset = eval_dataset.map(
        tokenize_fn, batched=True,
        remove_columns=["input_text", "target_text"],
        desc="Tokenizing eval split",
    )
    return train_dataset, eval_dataset, tokenizer


def generate_synthetic_dataset(path: str, n: int = 200, seed: int = 42) -> None:
    import random
    rng = random.Random(seed)
    products = [
        ("stainless steel pipe", "Tata Steel", "2 inch diameter, schedule 40", "pcs"),
        ("nitrile gloves", "Ansell", "size L, powder-free, blue", "boxes"),
        ("portland cement", "UltraTech", "OPC 53 grade, 50kg bags", "bags"),
        ("copper wire", "Havells", "2.5 sq mm, single core", "meters"),
        ("safety helmets", "3M", "ANSI Z89.1 rated, white", "pcs"),
        ("hydraulic oil", "Shell", "ISO VG 68, 20L drum", "drums"),
        ("LED bulbs", "Philips", "9W, cool white, B22 base", "pcs"),
        ("plywood sheets", "Century Ply", "18mm, 8x4 ft, marine grade", "sheets"),
        ("welding rods", "ESAB", "E6013, 3.15mm", "kg"),
        ("ball bearings", "SKF", "6205-2RS", "pcs"),
    ]
    with open(path, "w", encoding="utf-8") as f:
        for _ in range(n):
            n_products = rng.choices([1, 2, 3], weights=[0.2, 0.3, 0.5])[0]
            chosen = rng.sample(products, k=min(n_products, len(products)))
            lines, output = [], []
            for prod, brand, spec, uom in chosen:
                qty = str(rng.randint(1, 500))
                output.append({
                    "product_raw_text": prod, "product_brand": brand,
                    "product_specifications": spec, "order_qty": qty,
                    "order_unit_of_measure": uom,
                })
                lines.append(f"- {qty} {uom} of {prod} ({brand}, {spec})")
            email = "Please process a PO for:\n" + "\n".join(lines)
            f.write(json.dumps({"email": email, "output": output}, ensure_ascii=False) + "\n")