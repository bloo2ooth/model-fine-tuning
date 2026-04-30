import json
from collections import Counter

with open("training_data_checkpoint.json", "r") as f:
    data = json.load(f)

# Basic stats
classifications = Counter(x["classification"] for x in data)
scenarios = Counter(x["scenario_type"] for x in data)
bad = [i for i, x in enumerate(data) if x["llm_output"].strip() == "..."]

print(f"Total examples:       {len(data)}")
print(f"Bad examples ('...'): {len(bad)} ({len(bad)/len(data)*100:.1f}%)")
print(f"\nClass balance:")
for k, v in classifications.items():
    print(f"  {k}: {v} ({v/len(data)*100:.1f}%)")
print(f"\nScenarios covered:    {len(scenarios)} unique scenarios")
print(f"\nShortest llm_output:  {min(len(x['llm_output']) for x in data)} chars")
print(f"Longest llm_output:   {max(len(x['llm_output']) for x in data)} chars")
print(f"Avg llm_output:       {sum(len(x['llm_output']) for x in data)//len(data)} chars")