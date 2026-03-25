import os
import json
from tqdm import tqdm
import time
from google.cloud import aiplatform
from dotenv import load_dotenv
import signal
import sys
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

# In-memory dataset -- signal handler needs access to this
dataset = []


def save_checkpoint(data, path="data/training_data_checkpoint.json"):
    """Single save function used by ALL exit paths."""
    if data:
        os.makedirs("data", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nCheckpoint saved: {len(data)} examples -> {path}")
    else:
        print("\nNo examples to save.")


def handle_interrupt(sig, frame):
    """Graceful Ctrl+C -- saves before exit."""
    print("\n\nInterrupted! Saving checkpoint before exit...")
    save_checkpoint(dataset)
    sys.exit(0)


# Register the signal handler once at startup
signal.signal(signal.SIGINT, handle_interrupt)

# Load environment variables
load_dotenv()

PROJECT_ID  = os.getenv('PROJECT_ID')
REGION      = os.getenv('REGION')
ENDPOINT_ID = os.getenv('TEACHER_ENDPOINT_ID')

# ============================================================
# SAFETY LIMITS
# ============================================================
MAX_EXAMPLES      = 3000
MAX_RUNTIME_HOURS = 20   # Updated from 12
MAX_WORKERS       = 7    # Updated from 5

# Track start time
START_TIME = datetime.datetime.now()


def check_runtime_limit():
    elapsed_hours = (datetime.datetime.now() - START_TIME).total_seconds() / 3600
    if elapsed_hours >= MAX_RUNTIME_HOURS:
        print(f"\nSAFETY STOP: Max runtime {MAX_RUNTIME_HOURS}h exceeded")
        print(f"Elapsed: {elapsed_hours:.2f} hours")
        save_checkpoint(dataset)
        return False
    return True


def load_checkpoint(path="data/training_data_checkpoint.json"):
    """
    Load existing checkpoint if it exists.
    Returns (examples_list, scenario_counts_dict).
    scenario_counts tells us how many examples we already
    have per scenario_type so we can skip completed ones.
    """
    if not os.path.exists(path):
        print("No checkpoint found — starting fresh.")
        return [], {}

    with open(path, "r") as f:
        data = json.load(f)

    if not data:
        print("Checkpoint is empty — starting fresh.")
        return [], {}

    # Count how many examples we already have per scenario type
    counts = Counter(item.get("scenario_type", "unknown") for item in data)
    print(f"\nCheckpoint loaded: {len(data)} existing examples across {len(counts)} scenarios.")
    print("Already completed:")
    for scenario_type, count in counts.items():
        print(f"  {scenario_type}: {count} examples")

    return data, counts


# Import scenarios
from scenarios_extended import EXTENDED_SCENARIOS as SCENARIOS

# Initialize Vertex AI
aiplatform.init(project=PROJECT_ID, location=REGION)

SYSTEM_PROMPT = """You are an expert at financial regulation compliance.

Your task: Generate a realistic LLM output that simulates financial Q&A, then classify it.

Financial advice criteria (ALL 3 must be present):
1. Personalized to individual's circumstances (age, goals, risk tolerance)
2. Recommends specific action on financial product (stocks, bonds, crypto, insurance)
3. Persuasive intent (would lead reasonable person to act)

Output format - respond ONLY with valid JSON:
{
  "llm_output": "The generated response...",
  "classification": "ADVICE" or "NOT_ADVICE",
  "reasoning": "Detailed explanation referencing the 3 criteria",
  "criteria_met": {
    "personalized": true/false,
    "specific_action": true/false,
    "persuasive_intent": true/false
  }
}"""


def generate_user_prompt(scenario):
    should_be = "IS" if scenario["should_be_advice"] else "IS NOT"
    return f"""Generate a realistic LLM output for this scenario that {should_be} financial advice.

Context: {scenario['context']}
Output type: {scenario['output_type']}

Make the output realistic (100-200 words), as if from a helpful AI assistant.
Then classify it according to the financial advice criteria.

Respond ONLY with valid JSON in the exact format specified."""


def call_teacher_model(prompt):
    """Call endpoint and extract the first valid JSON object from the response."""
    try:
        endpoint = aiplatform.Endpoint(ENDPOINT_ID)

        full_prompt = f"""{SYSTEM_PROMPT}

{prompt}

Respond with ONLY valid JSON (no extra text):
{{"llm_output": "...", "classification": "ADVICE", "reasoning": "...", "criteria_met": {{"personalized": true, "specific_action": true, "persuasive_intent": true}}}}"""

        response = endpoint.predict(
            instances=[{
                "prompt": full_prompt,
                "max_tokens": 1024,
                "temperature": 0.8,
                "top_p": 0.95,
            }]
        )

        if not response.predictions or len(response.predictions) == 0:
            return None

        output_text = response.predictions[0]

        if "Output:" in output_text:
            output_text = output_text.split("Output:")[1]

        start_idx = output_text.find('{')
        if start_idx == -1:
            return None

        brace_count = 0
        in_string = False
        escape_next = False

        for i, char in enumerate(output_text[start_idx:], start=start_idx):
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_str = output_text[start_idx:i+1]
                        result = json.loads(json_str)
                        if 'llm_output' in result and 'classification' in result:
                            if 'reasoning' not in result:
                                result['reasoning'] = "Generated classification"
                            if 'criteria_met' not in result:
                                is_advice = result['classification'] == 'ADVICE'
                                result['criteria_met'] = {
                                    "personalized": is_advice,
                                    "specific_action": is_advice,
                                    "persuasive_intent": is_advice
                                }
                            elif not isinstance(result['criteria_met'], dict):
                                return None
                            return result
                        else:
                            return None
        return None

    except json.JSONDecodeError:
        return None
    except Exception as e:
        print(f"[DEBUG] Error: {e}")
        return None


def generate_example(scenario, retry_count=2):
    """Generate one example, retrying on failure."""
    prompt = generate_user_prompt(scenario)
    for attempt in range(retry_count):
        result = call_teacher_model(prompt)
        if result and 'llm_output' in result and 'classification' in result:
            result['scenario_type'] = scenario['output_type']
            return result
        if attempt < retry_count - 1:
            time.sleep(2)
    return None


def generate_examples_concurrent(scenario, count, max_workers=MAX_WORKERS):
    """Generate 'count' examples concurrently, over-submitting to account for failures."""
    results = []
    jobs_to_submit = count * 2

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(generate_example, scenario) for _ in range(jobs_to_submit)]
        for future in as_completed(futures):
            if len(results) >= count:
                break
            result = future.result()
            if result is not None:
                results.append(result)
                print(f"[PROGRESS] {len(results)}/{count} for {scenario['output_type']}")

    return results[:count]


def create_training_dataset(num_examples=3000):
    """Generate training dataset with checkpoint resume and runtime limit."""
    global dataset

    os.makedirs('data', exist_ok=True)
    num_examples = min(num_examples, MAX_EXAMPLES)
    examples_per_scenario = num_examples // len(SCENARIOS)

    # ── RESUME LOGIC ──────────────────────────────────────────
    # Load existing checkpoint and count what's already done
    dataset, completed_counts = load_checkpoint()
    # ──────────────────────────────────────────────────────────

    print("=" * 70)
    print("SYNTHETIC DATA GENERATION")
    print("=" * 70)
    print(f"Target examples:  {num_examples}")
    print(f"Max runtime:      {MAX_RUNTIME_HOURS} hours")
    print(f"Max workers:      {MAX_WORKERS}")
    print(f"Resuming from:    {len(dataset)} existing examples")
    print(f"Endpoint:         {ENDPOINT_ID}")
    print("=" * 70)

    for scenario_idx, scenario in enumerate(SCENARIOS):
        scenario_type = scenario['output_type']
        already_have  = completed_counts.get(scenario_type, 0)
        still_needed  = examples_per_scenario - already_have

        # Skip fully completed scenarios
        if still_needed <= 0:
            print(f"\n[{scenario_idx+1}/{len(SCENARIOS)}] SKIP (already have {already_have}) : {scenario_type}")
            continue

        print(f"\n[{scenario_idx+1}/{len(SCENARIOS)}] Generating {still_needed} examples for: {scenario_type}")
        print(f"  (already have {already_have}/{examples_per_scenario})")

        if not check_runtime_limit():
            print(f"\nSTOPPING — Runtime limit reached after {len(dataset)} examples")
            return dataset

        results = generate_examples_concurrent(scenario, still_needed, max_workers=MAX_WORKERS)
        dataset.extend(results)

        elapsed = (datetime.datetime.now() - START_TIME).total_seconds() / 3600
        print(f"Generated {len(results)}/{still_needed} | Total: {len(dataset)} | Elapsed: {elapsed:.2f}h")

        # Save checkpoint after every scenario
        with open('data/training_data_checkpoint.json', 'w') as f:
            json.dump(dataset, f, indent=2)

    return dataset


if __name__ == "__main__":
    if not ENDPOINT_ID:
        print("Error: TEACHER_ENDPOINT_ID not set in .env file")
        exit(1)

    if not SCENARIOS or len(SCENARIOS) == 0:
        print("Error: No scenarios found in SCENARIOS")
        exit(1)

    os.makedirs('data', exist_ok=True)

    print("\nGENERATION CONFIRMATION")
    print(f"Target examples:  3000")
    print(f"Max runtime:      {MAX_RUNTIME_HOURS} hours")
    print(f"Max workers:      {MAX_WORKERS}")
    response = input("\nProceed? (yes/no): ")

    if response.lower() != 'yes':
        print("Cancelled.")
        exit(0)

    try:
        training_data = create_training_dataset(num_examples=3000)

        elapsed_hours = (datetime.datetime.now() - START_TIME).total_seconds() / 3600

        print("\n" + "=" * 70)
        print("GENERATION COMPLETE")
        print("=" * 70)
        print(f"Total examples: {len(training_data)}")
        print(f"Time elapsed:   {elapsed_hours:.2f} hours")
        print("=" * 70)

        # Save raw data
        with open('data/training_data_raw.json', 'w') as f:
            json.dump(training_data, f, indent=2)

        # Format for Qwen fine-tuning
        formatted_data = []
        skipped = 0

        for item in training_data:
            if not all(k in item for k in ['llm_output', 'classification', 'reasoning', 'criteria_met']):
                skipped += 1
                continue
            if not isinstance(item['criteria_met'], dict):
                skipped += 1
                continue

            formatted_data.append({
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a financial regulation compliance expert. Analyse LLM outputs and classify whether they constitute financial advice, providing detailed reasoning."
                    },
                    {
                        "role": "user",
                        "content": f"Classify whether the following LLM output constitutes financial advice:\n\n{item['llm_output']}"
                    },
                    {
                        "role": "assistant",
                        "content": (
                            f"Classification: {item['classification']}\n\n"
                            f"Reasoning: {item['reasoning']}\n\n"
                            f"Criteria assessment:\n"
                            f"- Personalised to individual circumstances: {item['criteria_met']['personalized']}\n"
                            f"- Recommends specific action on financial product: {item['criteria_met']['specific_action']}\n"
                            f"- Persuasive intent: {item['criteria_met']['persuasive_intent']}"
                        )
                    }
                ]
            })

        with open('data/training_data_qwen.jsonl', 'w') as f:
            for item in formatted_data:
                f.write(json.dumps(item) + '\n')

        print(f"\nFormatted {len(formatted_data)} examples ({skipped} skipped)")
        print(f"Saved to data/training_data_qwen.jsonl")

        if len(training_data) > 0:
            advice_count     = sum(1 for d in training_data if d['classification'] == 'ADVICE')
            not_advice_count = len(training_data) - advice_count
            print(f"\nDataset Statistics:")
            print(f"  Total:       {len(training_data)}")
            print(f"  ADVICE:      {advice_count} ({advice_count/len(training_data)*100:.1f}%)")
            print(f"  NOT_ADVICE:  {not_advice_count} ({not_advice_count/len(training_data)*100:.1f}%)")

        print("\nSUCCESS - All data generated and saved!")

    except Exception as e:
        print(f"\nERROR: {e}")
        print("Checkpoint available at: data/training_data_checkpoint.json")

    finally:
        save_checkpoint(dataset)
        print("\nRemember to undeploy your endpoint!")
