import json
import os
import random

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH  = "data/training_data_qwen.jsonl"
OUTPUT_DIR = "./qwen2.5-7b-lora-finetuned"

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# ── Data ─────────────────────────────────────────────────────────────────────
TEST_SPLIT_PATH = "data/test_split.jsonl"


def load_dataset(
    path: str,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    not_advice_weight: float = 2.0,
):
    with open(path) as f:
        examples = [json.loads(line) for line in f]

    random.shuffle(examples)

    n = len(examples)
    test_end  = int(n * test_ratio)
    val_end   = test_end + int(n * val_ratio)

    test_raw  = examples[:test_end]
    val_raw   = examples[test_end:val_end]
    train_raw = examples[val_end:]

    # Save test split so evaluate.py can load it without touching train/val data.
    os.makedirs("data", exist_ok=True)
    with open(TEST_SPLIT_PATH, "w") as f:
        for ex in test_raw:
            f.write(json.dumps(ex) + "\n")
    print(f"Test split saved:    {len(test_raw)} examples → {TEST_SPLIT_PATH}")

    # Oversample NOT_ADVICE in the training set to compensate for the 75/25 skew.
    # The generation pipeline failed disproportionately on NOT_ADVICE scenarios,
    # producing a dataset biased toward ADVICE. A weight of 2.0 brings the ratio
    # to roughly 60/40 without fully rebalancing, preserving the mild ADVICE bias
    # that is acceptable for this application (false positives preferred).
    advice     = [e for e in train_raw if _get_label(e) == "ADVICE"]
    not_advice = [e for e in train_raw if _get_label(e) == "NOT_ADVICE"]
    not_advice_oversampled = not_advice * int(not_advice_weight)
    train_balanced = advice + not_advice_oversampled
    random.shuffle(train_balanced)

    train_dataset = Dataset.from_list(train_balanced)
    val_dataset   = Dataset.from_list(val_raw)

    print(f"Training examples:   {len(train_dataset)}")
    print(f"  ADVICE:            {len(advice)}")
    print(f"  NOT_ADVICE:        {len(not_advice_oversampled)} ({int(not_advice_weight)}x oversampled)")
    print(f"Validation examples: {len(val_dataset)}")

    return train_dataset, val_dataset


def _get_label(example: dict) -> str:
    for msg in example["messages"]:
        if msg["role"] == "assistant":
            content = msg["content"]
            if content.startswith("Classification: NOT_ADVICE"):
                return "NOT_ADVICE"
            if content.startswith("Classification: ADVICE"):
                return "ADVICE"
    return "UNKNOWN"


# ── Model & tokenizer ────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.config.use_cache = False

# ── LoRA config ───────────────────────────────────────────────────────────────
# r=8: conservative rank for a small dataset (~1,969 examples) and a narrow
# classification task. The base model already understands financial advice
# conceptually; we are teaching a judgment threshold, not new knowledge.
#
# alpha=16: keeps the standard alpha/r=2.0 scaling ratio. A higher ratio
# (e.g. alpha=32 with r=8) would double the adapter's influence and risk
# destabilising the base model's representations on this small dataset.
#
# All 7 projection layers targeted: restricting to q_proj+v_proj alone
# (the classical LoRA setting) may limit adaptation in the feed-forward
# layers where factual knowledge retrieval happens — relevant here because
# the classification relies on the model's understanding of financial concepts.
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "up_proj", "gate_proj", "down_proj",
    ],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── Training config ───────────────────────────────────────────────────────────
# effective batch size = per_device_train_batch_size × gradient_accumulation_steps
#                      = 4 × 8 = 32
#
# max_seq_length=1024: training examples are ~250–420 tokens (system + user +
# assistant turns). 1024 gives headroom without paying the compute cost of
# padding to 2048.
#
# max_grad_norm=1.0: 0.3 is a QLoRA-specific safeguard for 4-bit quantized
# training. Standard LoRA in bf16 uses 1.0.
#
# eval_steps=save_steps=25: with ~186 total steps across 3 epochs, aligning
# these ensures load_best_model_at_end can actually select from all eval
# checkpoints, not just the subset that happen to coincide with saves.
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    max_grad_norm=1.0,
    eval_strategy="steps",
    eval_steps=25,
    save_strategy="steps",
    save_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    max_seq_length=1024,
    seed=SEED,
    report_to="none",
)

# ── Train ────────────────────────────────────────────────────────────────────
train_dataset, val_dataset = load_dataset(DATA_PATH)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\nModel saved to {OUTPUT_DIR}")
