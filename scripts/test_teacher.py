from dotenv import load_dotenv
import os
from google.cloud import aiplatform
import json

load_dotenv()

PROJECT_ID = os.getenv('PROJECT_ID')
REGION = os.getenv('REGION')
ENDPOINT_ID = os.getenv('TEACHER_ENDPOINT_ID')

print("=" * 60)
print("TESTING LLAMA-3.3-70B ENDPOINT")
print("=" * 60)
print(f"Project: {PROJECT_ID}")
print(f"Region: {REGION}")
print(f"Endpoint: {ENDPOINT_ID}")

if not ENDPOINT_ID:
    print("\n❌ TEACHER_ENDPOINT_ID not set in .env file!")
    print("Add it with: echo 'TEACHER_ENDPOINT_ID=\"your-id\"' >> .env")
    exit(1)

try:
    # Initialize
    aiplatform.init(project=PROJECT_ID, location=REGION)
    
    # Get endpoint
    print("\nConnecting to endpoint...")
    endpoint = aiplatform.Endpoint(ENDPOINT_ID)
    
    print(f"✓ Connected to endpoint: {endpoint.display_name}")
    
    # Test with simple prompt
    test_prompt = """Generate a realistic LLM output that IS financial advice.

Context: User is 30 years old, risk-tolerant, asking where to invest $10k

Make it realistic and classify it.

Respond with JSON:
{
  "llm_output": "...",
  "classification": "ADVICE or NOT_ADVICE",
  "reasoning": "..."
}"""
    
    print("\nSending test prompt...")
    print("(This may take 10-30 seconds on first call)")
    
    response = endpoint.predict(
        instances=[{
            "prompt": test_prompt,
            "max_tokens": 512,
            "temperature": 0.8,
        }]
    )
    
    print("\n✓ Success! Endpoint is working.")
    print("\n" + "=" * 60)
    print("RESPONSE PREVIEW")
    print("=" * 60)
    
    output = response.predictions[0]
    print(output[:500])
    
    # Try to parse as JSON
    try:
        if "```json" in output:
            output = output.split("```json").split("```").strip()[1]
        parsed = json.loads(output)
        print("\n✓ Response is valid JSON")
        print(f"  Classification: {parsed.get('classification', 'N/A')}")
    except:
        print("\n⚠️  Response might need JSON cleaning (will be handled in main script)")
    
    print("\n" + "=" * 60)
    print("✅ ENDPOINT TEST SUCCESSFUL - Ready to generate data!")
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nTroubleshooting:")
    print("1. Check endpoint is fully deployed in console")
    print("2. Verify TEACHER_ENDPOINT_ID is correct")
    print("3. Verify REGION matches endpoint region")
    print("4. Wait 2-3 minutes if just deployed (warming up)")
