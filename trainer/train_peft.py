"""Обучение bf16 PEFT с маской промпта и числовыми метриками без текстов."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def base_directory(repo):
    cache = os.environ.get("OWNER_BASE_CACHE")
    if not cache:
        return repo
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo,
        local_dir=str(Path(cache) / repo.replace("/", "--")),
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.jinja", "*.txt"],
    )


def encode_chat(tokenizer, messages: list[dict], max_length: int) -> dict:
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("Нужен последний ответ assistant")
    options = dict(tokenize=True, return_dict=False, enable_thinking=False, thinking=False)
    prefix = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, **options)
    full = tokenizer.apply_chat_template(messages, add_generation_prompt=False, **options)
    if len(full) > max_length:
        raise ValueError("Длина пары превышает seq_length, усечение запрещено")
    if full[: len(prefix)] != prefix or len(full) <= len(prefix):
        raise ValueError("Шаблон не позволяет безопасно выделить ответ assistant")
    return dict(
        input_ids=full,
        attention_mask=[1] * len(full),
        labels=[-100] * len(prefix) + full[len(prefix) :],
    )


def main():
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoTokenizer,
        Gemma4ForCausalLM,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        set_seed,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    t = job["training"]
    root = args.job.parent
    output = root / "output"
    output.mkdir(mode=0o700, exist_ok=True)
    started = time.monotonic()
    set_seed(job["seed"])
    base_path = base_directory(t["base_hf"])
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    datasets = {}
    for split in ("train", "valid"):
        data = [
            json.loads(line)["messages"]
            for line in (root / f"{split}.jsonl").read_text().splitlines()
            if line.strip()
        ]
        if t["smoke"]:
            data = data[: max(20, t["batch_size"])] if split == "train" else data[:4]
        datasets[split] = [encode_chat(tokenizer, messages, t["seq_length"]) for messages in data]
        if not datasets[split]:
            raise ValueError("Пустая выборка")
    model = Gemma4ForCausalLM.from_pretrained(base_path, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    depth = model.config.num_hidden_layers
    from re import fullmatch

    targets = []
    portable = {
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    }
    for name, module in model.named_modules():
        m = fullmatch(r"model.layers.(\d+).(.+)", name)
        if (
            m
            and int(m[1]) >= max(0, depth - t["layers"])
            and m[2] in portable
            and isinstance(module, torch.nn.Linear)
        ):
            targets.append(name)
    if not targets:
        raise ValueError("Нет переносимых слоёв")
    model = get_peft_model(
        model,
        LoraConfig(
            r=t["rank"],
            lora_alpha=t["rank"] * t["scale"],
            lora_dropout=t["dropout"],
            target_modules=targets,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.enable_input_require_grads()

    def collate(batch):
        size = max(len(x["input_ids"]) for x in batch)
        result = {}
        for field, pad in (
            ("input_ids", tokenizer.pad_token_id or 0),
            ("attention_mask", 0),
            ("labels", -100),
        ):
            result[field] = torch.tensor([x[field] + [pad] * (size - len(x[field])) for x in batch])
        return result

    class Metrics(TrainerCallback):
        best = float("inf")
        iteration = 0
        bad = 0

        def on_log(self, args, state, control, logs=None, **kwargs):
            logs = logs or {}
            if "loss" in logs:
                record = dict(kind="train", iteration=state.global_step, train_loss=logs["loss"])
            elif "eval_loss" in logs:
                loss = logs["eval_loss"]
                improved = loss < self.best - t["min_delta"]
                if improved:
                    self.best, self.iteration, self.bad = loss, state.global_step, 0
                    model.save_pretrained(output / "peft", safe_serialization=True)
                else:
                    self.bad += 1
                control.should_training_stop = self.bad >= t["patience"]
                record = dict(
                    kind="valid",
                    iteration=state.global_step,
                    val_loss=loss,
                    improved=improved,
                    stop=control.should_training_stop,
                )
            else:
                return
            record["elapsed"] = time.monotonic() - started
            with (output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)

    callback = Metrics()
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(root / "checkpoints"),
            max_steps=t["iters"],
            per_device_train_batch_size=t["batch_size"],
            per_device_eval_batch_size=1,
            learning_rate=t["lr"],
            lr_scheduler_type="constant",
            warmup_steps=0,
            bf16=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            logging_steps=1,
            eval_strategy="steps",
            eval_steps=t["eval_every"],
            save_strategy="no",
            report_to="none",
            disable_tqdm=True,
            seed=job["seed"],
        ),
        train_dataset=datasets["train"],
        eval_dataset=datasets["valid"],
        data_collator=collate,
        callbacks=[callback],
    )
    trainer.train()
    trainer.evaluate()
    if not (output / "peft/adapter_model.safetensors").exists():
        raise ValueError("Не сохранён лучший адаптер")
    result = dict(
        best_iteration=callback.iteration,
        best_monitor_loss=callback.best,
        training_seconds=time.monotonic() - started,
        peak_cuda_gib=torch.cuda.max_memory_allocated() / 2**30,
        iterations=trainer.state.global_step,
        train=len(datasets["train"]),
        valid=len(datasets["valid"]),
        target_modules=targets,
        mask_prompt=True,
        thinking=False,
    )
    (output / "result.json").write_text(json.dumps(result, indent=2))
    # Паритет на 20 valid-промптах. Загружается лучший checkpoint.
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    set_peft_model_state_dict(model, load_file(str(output / "peft/adapter_model.safetensors")))
    model.eval()
    model.config.use_cache = True
    prompts = [
        json.loads(line)["messages"][:-1]
        for line in (root / "valid.jsonl").read_text().splitlines()
        if line.strip()
    ][:20]
    answers = []
    with torch.inference_mode():
        for messages in prompts:
            tokens = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                enable_thinking=False,
                add_generation_prompt=True,
                return_dict=False,
                return_tensors="pt",
            ).to("cuda")
            generated = model.generate(
                tokens,
                do_sample=False,
                max_new_tokens=768,
                pad_token_id=tokenizer.pad_token_id,
            )
            answers.append(
                dict(
                    messages=messages,
                    text=tokenizer.decode(generated[0, tokens.shape[1] :], skip_special_tokens=True),
                )
            )
    (output / "gpu-parity.json").write_text(json.dumps(answers, ensure_ascii=False))
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Ни текст пары, ни сообщение сторонней библиотеки не попадают в лог.
        import sys
        import traceback

        frames = [
            dict(file=Path(f.filename).name, function=f.name, line=f.lineno)
            for f in traceback.extract_tb(error.__traceback__)
        ]
        print(json.dumps(dict(status="failed", error_type=type(error).__name__, frames=frames)), flush=True)
        sys.exit(1)
