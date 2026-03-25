import os
import json
from tqdm import tqdm
import time
from google.cloud import aiplatform
from dotenv import load_dotenv
import signal
import sys
import datetime

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
MAX_EXAMPLES      = 3000  # Hard cap on examples to generate
MAX_RUNTIME_HOURS = 12    # Auto-stop after 12 hours

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
    """Create user prompt for teacher model."""
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
            print("[DEBUG] Empty predictions response")
            return None

        output_text = response.predictions[0]

        # Remove prefix labels sometimes added by the model
        if "Output:" in output_text:
            output_text = output_text.split("Output:")[1]

        # Find the first opening brace
        start_idx = output_text.find('{')
        if start_idx == -1:
            return None

        # Walk the string character by character to find the first complete JSON object
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

                        # Validate required fields
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
        print(f"\n[DEBUG] Attempt {attempt+1}/{retry_count} for scenario: {scenario['output_type']}")
        result = call_teacher_model(prompt)

        if result and 'llm_output' in result and 'classification' in result:
            result['scenario_type'] = scenario['output_type']
            print(f"[DEBUG] Success: {result['classification']}")
            return result
        else:
            print(f"[DEBUG] Failed to get valid result")

        if attempt < retry_count - 1:
            time.sleep(2)

    print(f"[DEBUG] Giving up after {retry_count} attempts")
    return None


def create_training_dataset(num_examples=3000):
    """Generate training dataset with runtime limit."""
    global dataset  # Signal handler and check_runtime_limit must reference the same list
    dataset = []

    os.makedirs('data', exist_ok=True)

    num_examples = min(num_examples, MAX_EXAMPLES)
    examples_per_scenario = num_examples // len(SCENARIOS)

    print("=" * 70)
    print("SYNTHETIC DATA GENERATION")
    print("=" * 70)
    print(f"Target examples:  {num_examples}")
    print(f"Max runtime:      {MAX_RUNTIME_HOURS} hours")
    print(f"Endpoint:         {ENDPOINT_ID}")
    print("=" * 70)

    for scenario_idx, scenario in enumerate(SCENARIOS):
        print(f"\n[{scenario_idx+1}/{len(SCENARIOS)}] Generating {examples_per_scenario} examples for: {scenario['output_type']}")

        successful = 0
        attempts = 0
        max_attempts = examples_per_scenario * 2
        progress_bar = tqdm(total=examples_per_scenario)

        while successful < examples_per_scenario and attempts < max_attempts:

            if not check_runtime_limit():
                progress_bar.close()
                print(f"\nSTOPPING GENERATION - Runtime limit reached")
                print(f"Generated {len(dataset)} examples so far")
                return dataset

            example = generate_example(scenario)
            attempts += 1

            if example:
                dataset.append(example)
                successful += 1
                progress_bar.update(1)

            # Rate limiting -- print elapsed time every 10 attempts
            if attempts % 10 == 0:
                time.sleep(2)
                elapsed = (datetime.datetime.now() - START_TIME).total_seconds() / 3600
                print(f"\nElapsed: {elapsed:.2f}h")

        progress_bar.close()
        print(f"Generated {successful}/{examples_per_scenario}")

        # Save checkpoint after every scenario in case of failure
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

        # Save raw data with all fields intact
        with open('data/training_data_raw.json', 'w') as f:
            json.dump(training_data, f, indent=2)

        # Format for Qwen fine-tuning using ChatML message structure
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

        print(f"\nFormatted {len(formatted_data)} examples ({skipped} skipped due to missing fields)")
        print(f"Saved to data/training_data_qwen.jsonl")

        if len(training_data) > 0:
            advice_count = sum(1 for d in training_data if d['classification'] == 'ADVICE')
            not_advice_count = len(training_data) - advice_count
            print(f"\nDataset Statistics:")
            print(f"  Total:       {len(training_data)}")
            print(f"  ADVICE:      {advice_count} ({advice_count/len(training_data)*100:.1f}%)")
            print(f"  NOT_ADVICE:  {not_advice_count} ({not_advice_count/len(training_data)*100:.1f}%)")
        else:
            print("\nWARNING: No examples generated!")

        print("\nSUCCESS - All data generated and saved!")

    except Exception as e:
        print(f"\nERROR: {e}")
        print("Checkpoint may be available at: data/training_data_checkpoint.json")

    finally:
        save_checkpoint(dataset)  # Runs on normal finish, error, or crash
        print("\nRemember to undeploy your endpoint!")
