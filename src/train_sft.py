"""Supervised fine tuning entry point: one cell of the capacity x data grid.

    python -m src.train_sft --config configs/default.yaml --model smollm2-360m \
        --data lima --n-train 1000

Every run writes a ``run_meta.json`` next to the checkpoint recording the model,
the data size, the seed and the resolved hyperparameters. The analysis scripts
read those files, so a run that is not recorded does not exist as far as the
paper is concerned.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

from . import data as data_mod
from . import model_zoo


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    if args.model:
        cfg["model"]["key"] = args.model
    if args.data:
        cfg["data"]["name"] = args.data
    if args.n_train is not None:
        cfg["data"]["n_train"] = args.n_train
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    if args.seed is not None:
        cfg["experiment"]["seed"] = args.seed
    if args.use_lora:
        cfg["train"]["use_lora"] = True
    if args.lora_r is not None:
        cfg["train"]["use_lora"] = True
        cfg["train"]["lora_r"] = args.lora_r
    if args.smoke_test:
        cfg["experiment"]["smoke_test"] = True
    return cfg


def run_dir_for(cfg: dict) -> Path:
    n = cfg["data"]["n_train"]
    tag = "full" if n is None else str(n)
    lora = f"-lora{cfg['train']['lora_r']}" if cfg["train"]["use_lora"] else ""
    name = (f"{cfg['model']['key']}__sft-{cfg['data']['name']}-{tag}{lora}"
            f"__seed{cfg['experiment']['seed']}")
    return Path(cfg["experiment"]["output_root"]) / name


def build_training_text(example: dict, tokenizer) -> str:
    """Render one conversation into a single training string.

    Base models on the ladder often ship no chat template, so a template is
    imposed here and reused at evaluation time through ``prompts.chat_prompt``.
    Keeping the two in sync matters more than which template is chosen.
    """
    messages = example["messages"]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False)
    parts = []
    for m in messages:
        parts.append(f"<|{m['role']}|>\n{m['content']}")
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT one rung of the capacity ladder")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", help=f"one of {sorted(model_zoo.LADDER)}")
    parser.add_argument("--data", help="lima | tulu3 | alpaca")
    parser.add_argument("--n-train", type=int, dest="n_train")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--use-lora", action="store_true", dest="use_lora")
    parser.add_argument("--lora-r", type=int, dest="lora_r")
    parser.add_argument("--smoke-test", action="store_true", dest="smoke_test")
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args)

    # Imported late so that --help works on a machine without torch installed.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import SFTConfig, SFTTrainer

    seed = int(cfg["experiment"]["seed"])
    set_seed(seed)

    spec = model_zoo.get(cfg["model"]["key"])
    out_dir = run_dir_for(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train] {spec.hf_id} ({spec.label})  ->  {out_dir}")

    tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_train = cfg["data"]["n_train"]
    if cfg["experiment"]["smoke_test"]:
        n_train = 64
        cfg["train"]["epochs"] = 1

    dataset = data_mod.load_sft_dataset(cfg["data"]["name"], n_train=n_train, seed=seed)
    if dataset is None:
        raise SystemExit("data.name is 'none', so there is nothing to fine tune on")

    dataset = dataset.map(
        lambda ex: {"text": build_training_text(ex, tokenizer)},
        remove_columns=dataset.column_names,
    )
    splits = dataset.train_test_split(
        test_size=float(cfg["data"]["val_fraction"]), seed=seed
    )
    print(f"[train] {len(splits['train'])} train / {len(splits['test'])} val examples")

    dtype = getattr(torch, cfg["model"]["dtype"])
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id,
        torch_dtype=dtype,
        attn_implementation=cfg["model"]["attn_implementation"],
    )
    model.config.use_cache = False

    peft_config = None
    if cfg["train"]["use_lora"]:
        from peft import LoraConfig
        peft_config = LoraConfig(
            r=int(cfg["train"]["lora_r"]),
            lora_alpha=int(cfg["train"]["lora_alpha"]),
            lora_dropout=float(cfg["train"]["lora_dropout"]),
            target_modules=list(cfg["train"]["lora_target_modules"]),
            task_type="CAUSAL_LM",
        )
        print(f"[train] LoRA rank {cfg['train']['lora_r']} (update capacity control)")

    sft_config = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=float(cfg["train"]["epochs"]),
        per_device_train_batch_size=int(cfg["train"]["per_device_batch_size"]),
        gradient_accumulation_steps=int(cfg["train"]["grad_accum_steps"]),
        learning_rate=float(cfg["train"]["lr"]),
        warmup_ratio=float(cfg["train"]["warmup_ratio"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
        lr_scheduler_type=cfg["train"]["lr_scheduler"],
        max_grad_norm=float(cfg["train"]["max_grad_norm"]),
        logging_steps=int(cfg["train"]["logging_steps"]),
        save_strategy=cfg["train"]["save_strategy"],
        gradient_checkpointing=bool(cfg["train"]["gradient_checkpointing"]),
        max_seq_length=int(cfg["model"]["max_seq_len"]),
        bf16=(cfg["model"]["dtype"] == "bfloat16"),
        dataset_text_field="text",
        packing=False,
        seed=seed,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=splits["train"],
        eval_dataset=splits["test"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    meta = {
        "model_key": spec.key,
        "hf_id": spec.hf_id,
        "params": spec.params,
        "intervention": "sft",
        "data_name": cfg["data"]["name"],
        "n_train": len(splits["train"]),
        "seed": seed,
        "use_lora": bool(cfg["train"]["use_lora"]),
        "lora_r": cfg["train"]["lora_r"] if cfg["train"]["use_lora"] else None,
        "epochs": cfg["train"]["epochs"],
        "lr": cfg["train"]["lr"],
        "smoke_test": bool(cfg["experiment"]["smoke_test"]),
    }
    with open(out_dir / "run_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[train] done. checkpoint and run_meta.json in {out_dir}")


if __name__ == "__main__":
    main()
