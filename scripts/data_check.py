import json

with open("training_data_checkpoint.json", "r") as f:
    data = json.load(f)

print(f"Total records: {len(data)}")
print("\n--- First record ---")
print(json.dumps(data[0], indent=2))

# Find and print the first record that looks like a failure
for i, record in enumerate(data):
    content = str(record)
    if "error" in content.lower() or "failed" in content.lower() or "none" in content.lower():
        print(f"\n--- First suspicious record (index {i}) ---")
        print(json.dumps(record, indent=2))
        break
