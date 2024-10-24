import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Load training data file
global_model_dir = os.getenv("GLOBAL_MODEL_DIR")
test_f = os.getenv("TEST_CSV")
test_df = pd.read_csv(test_f)

column_names = ["accepted_taxon_key", "num_gbif_test_images"]
stats_df = pd.DataFrame(columns=column_names)

for _, row in test_df.iterrows():
    key = int(row["acceptedTaxonKey"])

    if key not in stats_df["accepted_taxon_key"].tolist():
        new_entry = {"accepted_taxon_key": key, "num_gbif_test_images": 1}
        stats_df.loc[len(stats_df)] = new_entry
    else:
        stats_df.loc[stats_df["accepted_taxon_key"] == key, "num_gbif_test_images"] += 1

stats_df.to_csv(Path(global_model_dir) / "test_stats.csv", index=False)
