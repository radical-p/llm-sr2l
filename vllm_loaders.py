#!/usr/bin/env python3
""" merge adapter with base model """

import argparse
import os
from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_adapter(base_model_id: str, adapter_dir: str, output_dir: str, dtype: str = "float16") -> None:
    adapter_config = os.path.join(adapter_dir, "adapter_config.json")
    adapter_weights = os.path.join(adapter_dir, "adapter_model.safetensors")

    if not os.path.isfile(adapter_config):
        raise FileNotFoundError(f"Missing config")
    if not os.path.isfile(adapter_weights):
        raise FileNotFoundError(f"Missing weights")

    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(dtype, torch.float16)

    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    print("before load")
    model = PeftModel.from_pretrained(base, adapter_dir)
    merged = model.merge_and_unload()
    print("after merge")
    print("before save")
    os.makedirs(output_dir, exist_ok=True)
    merged.save_pretrained(output_dir, safe_serialization=True)
    print("after save")
    tok = AutoTokenizer.from_pretrained(base_model_id, use_fast=False, trust_remote_code=True)
    tok.save_pretrained(output_dir)


def maybe_push_to_hub(output_dir: str, repo_id: Optional[str]) -> None:
    if not repo_id:
        return
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN environment variable not set for pushing to hub")

    api = HfApi()
    print("before upload")
    api.upload_folder(
        folder_path=output_dir,
        repo_id=repo_id,
        commit_message="Upload merged model",
        token=token,
    )
    print("after upload")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge a LoRA adapter with a base model")
    p.add_argument("--base", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--push", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    merge_adapter(args.base, args.adapter, args.out, dtype=args.dtype)
    print("here1")
    maybe_push_to_hub(args.out, args.push)
    print("done")


if __name__ == "__main__":
    main()
