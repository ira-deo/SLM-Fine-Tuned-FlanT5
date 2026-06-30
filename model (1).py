"""
model.py

Model loading utilities for fine-tuning flan-t5-base on the procurement
email JSON extraction task.
"""

from typing import Tuple

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

DEFAULT_MODEL_NAME = "google/flan-t5-base"


def load_model_and_tokenizer(
    model_name: str = DEFAULT_MODEL_NAME,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load the seq2seq model and tokenizer.

    Raises a RuntimeError with a clear message if loading fails (e.g. due
    to no network access / invalid model name).
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load tokenizer for '{model_name}'. "
            f"Check the model name and network access. Original error: {e}"
        ) from e

    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model weights for '{model_name}'. "
            f"Check the model name and network access. Original error: {e}"
        ) from e

    # ------------------------------------------------------------------
    # CRITICAL FIX FOR MULTI-ITEM GENERATION TRUNCATION
    # ------------------------------------------------------------------
    # FIX 1: max_length/max_new_tokens raised to 768 to match dataset.py.
    # Real data has up to 7 products; longest target ~630 tokens. 512 was
    # truncating multi-product arrays so the model never learned them.
    model.generation_config.max_new_tokens = 4092
    model.generation_config.num_beams = 4
    model.generation_config.early_stopping = False
    # ------------------------------------------------------------------

    return model, tokenizer


def get_device() -> torch.device:
    """Return the best available device (cuda > mps > cpu)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")