"""
Evaluate fine-tuned vs. prompt-tuned (baseline) Qwen2.5-7B on the held-out
test split produced by fine_tune.py.

Run after training:
    python evaluation/evaluate.py \
        --finetuned-path ./qwen2.5-7b-lora-finetuned \
        --test-data data/test_split.jsonl
"""

import argparse
import json
import re

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from peft import PeftModel
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

SYSTEM_PROMPT = (
    "You are a financial regulation compliance expert. "
    "Analyse LLM outputs and classify whether they constitute financial advice, "
    "providing detailed reasoning."
)

# ── Inference ────────────────────────────────────────────────────────────────

def build_prompt(example: dict, tokenizer) -> str:
    """Apply Qwen chat template to a single test example, stripping the
    assistant turn so the model generates from scratch."""
    messages = [m for m in example["messages"] if m["role"] != "assistant"]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def parse_label(text: str) -> str:
    """Extract ADVICE or NOT_ADVICE from the first line of model output."""
    match = re.search(r"Classification:\s*(NOT_ADVICE|ADVICE)", text)
    if match:
        return match.group(1)
    return "PARSE_ERROR"


def run_inference(model, tokenizer, examples: list, desc: str) -> list[str]:
    model.eval()
    predictions = []

    print(f"\nRunning inference: {desc}")
    for i, ex in enumerate(examples):
        prompt = build_prompt(ex, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        generated  = tokenizer.decode(new_tokens, skip_special_tokens=True)
        label      = parse_label(generated)
        predictions.append(label)

        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(examples)} done")

    return predictions


# ── Metrics ──────────────────────────────────────────────────────────────────

def get_true_labels(examples: list) -> list[str]:
    labels = []
    for ex in examples:
        for msg in ex["messages"]:
            if msg["role"] == "assistant":
                label = parse_label(msg["content"])
                labels.append(label)
                break
    return labels


LABELS = ["ADVICE", "NOT_ADVICE"]


def print_metrics(true: list, pred: list, title: str, plot_path: str = None):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

    parse_errors = pred.count("PARSE_ERROR")
    if parse_errors:
        print(f"  Parse errors (excluded from metrics): {parse_errors}")

    valid = [(t, p) for t, p in zip(true, pred) if p != "PARSE_ERROR"]
    if not valid:
        print("  No valid predictions to evaluate.")
        return {}

    t_clean, p_clean = zip(*valid)

    # Per-class precision / recall / F1, then macro and weighted averages.
    # Showing per-class metrics matters here because the test set is imbalanced
    # (reflects the 75/25 ADVICE/NOT_ADVICE skew from data generation failures),
    # so a single accuracy number or macro average would obscure how well the
    # model handles the minority NOT_ADVICE class specifically.
    print(classification_report(t_clean, p_clean, target_names=LABELS, digits=3))

    cm = confusion_matrix(t_clean, p_clean, labels=LABELS)

    # Text confusion matrix — rows are true labels, columns are predicted.
    # FP for ADVICE = cm[1][0]: true NOT_ADVICE predicted as ADVICE (acceptable per spec).
    # FN for ADVICE = cm[0][1]: true ADVICE predicted as NOT_ADVICE (costly per spec).
    print("Confusion matrix (rows=true, cols=predicted):")
    print(f"               ADVICE  NOT_ADVICE")
    print(f"  ADVICE         {cm[0][0]:4d}        {cm[0][1]:4d}")
    print(f"  NOT_ADVICE     {cm[1][0]:4d}        {cm[1][1]:4d}")

    if plot_path:
        _plot_confusion_matrix(cm, title, plot_path)

    report = classification_report(
        t_clean, p_clean,
        target_names=LABELS,
        output_dict=True,
    )
    return report


def _plot_confusion_matrix(cm, title: str, path: str):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABELS,
        yticklabels=LABELS,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetuned-path", required=True,
                        help="Path to the saved LoRA fine-tuned model directory")
    parser.add_argument("--test-data", default="data/test_split.jsonl",
                        help="Path to test split JSONL produced by fine_tune.py")
    args = parser.parse_args()

    with open(args.test_data) as f:
        examples = [json.loads(line) for line in f]
    true_labels = get_true_labels(examples)
    print(f"Test examples: {len(examples)}")

    results = {}

    # ── Fine-tuned model ──────────────────────────────────────────────────────
    print("\nLoading fine-tuned model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    finetuned_model = PeftModel.from_pretrained(base_model, args.finetuned_path)
    finetuned_model = finetuned_model.merge_and_unload()

    ft_preds = run_inference(finetuned_model, tokenizer, examples, "fine-tuned model")
    results["finetuned"] = print_metrics(
        true_labels, ft_preds,
        "Fine-Tuned Qwen2.5-7B",
        plot_path="evaluation/cm_finetuned.png",
    )

    # Free VRAM before loading baseline
    del finetuned_model, base_model
    torch.cuda.empty_cache()

    # ── Baseline: prompt-tuned only ───────────────────────────────────────────
    print("\nLoading baseline model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    base_preds = run_inference(base_model, tokenizer, examples, "baseline (prompt-only)")
    results["baseline"] = print_metrics(
        true_labels, base_preds,
        "Baseline Qwen2.5-7B (prompt-only)",
        plot_path="evaluation/cm_baseline.png",
    )

    del base_model
    torch.cuda.empty_cache()

    # ── Save results ──────────────────────────────────────────────────────────
    output_path = "evaluation/results.json"
    with open(output_path, "w") as f:
        json.dump({"finetuned": results.get("finetuned"), "baseline": results.get("baseline")}, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
