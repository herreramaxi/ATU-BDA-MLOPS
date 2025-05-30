#https://github.com/evidentlyai/evidently
from evidently import Report
from evidently.presets import DataDriftPreset
import pandas as pd
import glob
import sys

# Load data
df_full = pd.read_csv("data/full/spam.csv", encoding="latin-1")
new_files = glob.glob("data/new-data/*.csv")
if not new_files:
    print("No new data found.")
    sys.exit(78)

df_new = pd.concat([pd.read_csv(f, encoding="latin-1") for f in new_files])

df_full["MessageLength"] = df_full["Message"].astype(str).apply(len)
df_new["MessageLength"] = df_new["Message"].astype(str).apply(len)
df_full = df_full[["Category", "MessageLength"]]
df_new = df_new[["Category", "MessageLength"]]

# report = Report(metrics=[DataDriftPreset()])
report = report = Report([
    DataDriftPreset(method="psi")
],
include_tests="True")
my_eval  = report.run(reference_data=df_full, current_data=df_new)

my_eval.save_html("drift_report.html")
result = my_eval.dict()

drift_detected = result['metrics'][0]['value']['share']
print (drift_detected)

drift_metric = next((m for m in result["metrics"] if m["metric_id"].startswith("DriftedColumnsCount")), None)
if drift_metric and drift_metric["value"]["share"] >= 0.5:
    print("⚠️ Drift detected. Proceed with retraining.")
    sys.exit(0)
else:
    print("✅ No significant drift. Skip retraining.")
    sys.exit(78)
