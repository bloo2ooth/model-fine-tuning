from dotenv import load_dotenv
import os
from google.cloud import aiplatform
import json

load_dotenv()

PROJECT_ID = os.getenv('PROJECT_ID')
REGION = os.getenv('REGION')
ENDPOINT_ID = os.getenv('TEACHER_ENDPOINT_ID')

aiplatform.init(project=PROJECT_ID, location=REGION)
endpoint = aiplatform.Endpoint(ENDPOINT_ID)

print("Sending test request...")

response = endpoint.predict(
    instances=[{
        "prompt": "Say hello and respond with JSON: {\"message\": \"hello\"}",
        "max_tokens": 256,
        "temperature": 0.7,
    }]
)

print("\n" + "=" * 60)
print("RAW RESPONSE INSPECTION")
print("=" * 60)

print(f"\nType of response.predictions: {type(response.predictions)}")
print(f"Type of response.predictions[0]: {type(response.predictions[0])}")
print(f"\nFull response.predictions:")
print(response.predictions)
print(f"\nFirst prediction:")
print(response.predictions[0])

if isinstance(response.predictions[0], dict):
    print("\n✓ Response is a dict")
    print(f"Keys: {response.predictions[0].keys()}")
    for key, value in response.predictions[0].items():
        print(f"  {key}: {type(value)} = {str(value)[:200]}")
elif isinstance(response.predictions[0], list):
    print("\n✓ Response is a list")
    print(f"Length: {len(response.predictions[0])}")
    print(f"First item: {response.predictions[0][0][:200] if response.predictions[0] else 'empty'}")
elif isinstance(response.predictions[0], str):
    print("\n✓ Response is a string")
    print(f"Content: {response.predictions[0][:200]}")
else:
    print(f"\n? Response is: {type(response.predictions[0])}")

print("\n" + "=" * 60)
