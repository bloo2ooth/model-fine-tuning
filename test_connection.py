from google.cloud import aiplatform
from google.cloud import storage
import os

# Set project
PROJECT_ID = "ai-innovation-486521"
REGION = "europe-west4"

# Initialize Vertex AI
aiplatform.init(project=PROJECT_ID, location=REGION)

print(f"✓ Connected to project: {PROJECT_ID}")
print(f"✓ Region: {REGION}")

# Test storage access
storage_client = storage.Client()
buckets = list(storage_client.list_buckets())
print(f"✓ Accessible buckets: {[b.name for b in buckets]}")

print("\n GCP setup successful!")
