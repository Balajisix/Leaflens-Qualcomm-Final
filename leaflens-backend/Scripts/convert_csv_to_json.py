import pandas as pd

# 1) Read the CSV. If your CSV uses a different delimiter, pass `sep=";"` or similar.
df = pd.read_csv("plant_result.csv", encoding="utf-8")

# 2) (Optional) Rename columns to match your JSON schema:
#    e.g. df.rename(columns={
#       "plant": "plant_en",
#       "plant_hi": "plant_hi",
#       "plant_ta": "plant_ta",
#       "disease_type": "disease_type_en",
#       "symptoms": "symptoms_en",
#       "prevention": "prevention_en",
#    }, inplace=True)

# 3) If you have multilingual columns (plant_hi, plant_ta, etc.) ensure they're present.

# 4) Export to JSON
df.to_json(
    "src/data/plantResults.json",
    orient="records",
    force_ascii=False,  # keep non‑ASCII characters
    indent=2
)

print("Wrote src/data/plantResults.json with", len(df), "records.")
