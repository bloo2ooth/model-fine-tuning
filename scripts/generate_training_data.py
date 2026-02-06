import os
import json
from tqdm import tqdm
import time
from google.cloud import aiplatform
from dotenv import load_dotenv
import sys

# Load environment variables
load_dotenv()

PROJECT_ID = os.getenv('PROJECT_ID')
REGION = os.getenv('REGION')
ENDPOINT_ID = os.getenv('TEACHER_ENDPOINT_ID')

# ============================================================
# COST SAFETY LIMITS - DO NOT EXCEED
# ============================================================
MAX_EXAMPLES = 5000  # Hard cap on examples to generate
MAX_RUNTIME_HOURS = 24  # Auto-stop after 24 hours
ESTIMATED_COST_PER_HOUR = 4  # €4/hour for g4-standard-96 (2× RTX PRO 6000)
MAX_BUDGET_EUR = 100  # Stop if estimated cost exceeds €30

# Track start time
import datetime
START_TIME = datetime.datetime.now()

def check_cost_limit():
    """Safety check: Stop if estimated cost exceeds budget"""
    elapsed_hours = (datetime.datetime.now() - START_TIME).total_seconds() / 3600
    estimated_cost = elapsed_hours * ESTIMATED_COST_PER_HOUR
    
    if elapsed_hours > MAX_RUNTIME_HOURS:
        print(f"\n⚠️  SAFETY STOP: Max runtime ({MAX_RUNTIME_HOURS}h) exceeded")
        print(f"Elapsed: {elapsed_hours:.2f} hours")
        print(f"Estimated cost: €{estimated_cost:.2f}")
        return False
    
    if estimated_cost > MAX_BUDGET_EUR:
        print(f"\n⚠️  SAFETY STOP: Estimated cost (€{estimated_cost:.2f}) exceeds budget (€{MAX_BUDGET_EUR})")
        return False
    
    return True

# Import scenarios
from scenarios_extended import EXTENDED_SCENARIOS as SCENARIOS

# Initialize Vertex AI
aiplatform.init(project=PROJECT_ID, location=REGION)

# System prompt (keep as before)
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
    """Create user prompt for teacher model"""
    should_be = "IS" if scenario["should_be_advice"] else "IS NOT"
    
    return f"""Generate a realistic LLM output for this scenario that {should_be} financial advice.

Context: {scenario['context']}
Output type: {scenario['output_type']}

Make the output realistic (100-200 words), as if from a helpful AI assistant.
Then classify it according to the financial advice criteria.

Respond ONLY with valid JSON in the exact format specified."""

def call_teacher_model(prompt):
    """Call endpoint and extract FIRST valid JSON object"""
    try:
        endpoint = aiplatform.Endpoint(ENDPOINT_ID)
        
        full_prompt = f"""You are a financial compliance expert. Generate a realistic financial Q&A example and classify it.

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
        
        output_text = response.predictions[0]
        
        # Remove prefix labels
        if "Output:" in output_text:
            output_text = output_text.split("Output:")[1]
        
        # Find first opening brace
        start_idx = output_text.find('{')
        if start_idx == -1:
            return None
        
        # Parse incrementally to find first complete JSON object
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
                    
                    # Found complete JSON object
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
                            return result
                        else:
                            return None
        
        return None
        
    except json.JSONDecodeError as e:
        return None
    except Exception as e:
        print(f"[DEBUG] Error: {e}")
        return None



def generate_example(scenario, retry_count=2):
    """Generate one example with debug output"""
    prompt = generate_user_prompt(scenario)
    
    for attempt in range(retry_count):
        print(f"\n[DEBUG] Attempt {attempt+1}/{retry_count} for scenario: {scenario['output_type']}")
        result = call_teacher_model(prompt)
        
        if result and 'llm_output' in result and 'classification' in result:
            result['scenario_type'] = scenario['output_type']
            print(f"[DEBUG] ✓ Success: {result['classification']}")
            return result
        else:
            print(f"[DEBUG] ✗ Failed to get valid result")
        
        if attempt < retry_count - 1:
            time.sleep(2)
    
    print(f"[DEBUG] Giving up after {retry_count} attempts")
    return None


def create_training_dataset(num_examples=5000):
    """Generate training dataset with safety limits"""
    
    # Safety check: Don't exceed MAX_EXAMPLES
    num_examples = min(num_examples, MAX_EXAMPLES)
    
    dataset = []
    examples_per_scenario = num_examples // len(SCENARIOS)
    
    print("=" * 70)
    print("SYNTHETIC DATA GENERATION WITH COST SAFEGUARDS")
    print("=" * 70)
    print(f"Target examples: {num_examples}")
    print(f"Max runtime: {MAX_RUNTIME_HOURS} hours")
    print(f"Max budget: €{MAX_BUDGET_EUR}")
    print(f"Estimated cost per hour: €{ESTIMATED_COST_PER_HOUR}")
    print(f"Endpoint: {ENDPOINT_ID}")
    print("=" * 70)
    
    for scenario_idx, scenario in enumerate(SCENARIOS):
        print(f"\n[{scenario_idx+1}/{len(SCENARIOS)}] Generating {examples_per_scenario} examples for: {scenario['output_type']}")
        
        successful = 0
        attempts = 0
        max_attempts = examples_per_scenario * 2
        
        progress_bar = tqdm(total=examples_per_scenario)
        
        while successful < examples_per_scenario and attempts < max_attempts:
            # ⚠️ SAFETY CHECK: Stop if cost limit reached
            if not check_cost_limit():
                progress_bar.close()
                print(f"\n🛑 STOPPING GENERATION - Safety limit reached")
                print(f"Generated {len(dataset)} examples so far")
                return dataset
            
            example = generate_example(scenario)
            attempts += 1
            
            if example:
                dataset.append(example)
                successful += 1
                progress_bar.update(1)
            
            # Rate limiting
            if attempts % 10 == 0:
                time.sleep(2)
                
                # Print cost estimate every 10 attempts
                elapsed = (datetime.datetime.now() - START_TIME).total_seconds() / 3600
                estimated_cost = elapsed * ESTIMATED_COST_PER_HOUR
                print(f"\n💰 Elapsed: {elapsed:.2f}h | Est. cost: €{estimated_cost:.2f}")
        
        progress_bar.close()
        print(f"  ✓ Generated {successful}/{examples_per_scenario}")
        
        # Save progress after each scenario
        with open('data/training_data_checkpoint.json', 'w') as f:
            json.dump(dataset, f, indent=2)
    
    return dataset

if __name__ == "__main__":
    # Check endpoint is set
    if not ENDPOINT_ID:
        print("❌ Error: TEACHER_ENDPOINT_ID not set in .env file")
        exit(1)
    
    # Confirm before starting
    print("\n⚠️  COST CONFIRMATION")
    print(f"Estimated cost: €{ESTIMATED_COST_PER_HOUR * MAX_RUNTIME_HOURS:.2f} (max)")
    print(f"Your budget: €{MAX_BUDGET_EUR}")
    response = input("\nProceed? (yes/no): ")
    
    if response.lower() != 'yes':
        print("Cancelled.")
        exit(0)
    
    # Generate data
    training_data = create_training_dataset(num_examples=5000)
    
    # Calculate actual cost
    elapsed_hours = (datetime.datetime.now() - START_TIME).total_seconds() / 3600
    actual_cost = elapsed_hours * ESTIMATED_COST_PER_HOUR
    
    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    print(f"Total examples: {len(training_data)}")
    print(f"Time elapsed: {elapsed_hours:.2f} hours")
    print(f"Estimated cost: €{actual_cost:.2f}")
    print("=" * 70)
    
    # Save final data
    print("\nSaving data...")
    with open('data/training_data_raw.json', 'w') as f:
        json.dump(training_data, f, indent=2)
    
    # Format for training
    formatted_data = []
    for item in training_data:
        formatted_data.append({
            "instruction": "Classify whether this LLM output constitutes financial advice. Provide reasoning then label.",
            "input": item['llm_output'],
            "reasoning": item['reasoning'],
            "output": item['classification']
        })
    
    with open('data/training_data_formatted.jsonl', 'w') as f:
        for item in formatted_data:
            f.write(json.dumps(item) + '\n')
    
    print("✓ Data saved!")
    
    # Statistics
    advice_count = sum(1 for d in training_data if d['classification'] == 'ADVICE')
    not_advice_count = len(training_data) - advice_count
    
    print(f"\nDataset Statistics:")
    print(f"  Total: {len(training_data)}")
    print(f"  ADVICE: {advice_count} ({advice_count/len(training_data)*100:.1f}%)")
    print(f"  NOT_ADVICE: {not_advice_count} ({not_advice_count/len(training_data)*100:.1f}%)")
    
    print("\n⚠️  IMPORTANT: Remember to undeploy your endpoint to stop charges!")
    print("Run: gcloud ai endpoints undeploy-model YOUR_ENDPOINT_ID --region=YOUR_REGION")
