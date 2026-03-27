import json

for x in range(22, 37):
    with open(f"receipt_{x}_truth.json", "w") as f:
        json.dump({}, f)

print("Done! Created receipt_22_truth.json through receipt_36_truth.json")