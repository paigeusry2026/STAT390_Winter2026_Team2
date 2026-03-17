import os
import pandas as pd
from collections import defaultdict

# ---------------------------------------------------
# Data loading
# ---------------------------------------------------

# Path to folder that contains all 50 run directories
output_directory = "/projects/e32998/STAT390_Winter2026/Presentation4/STAT390_Winter2026_Team2/Code9_no_leakage/coders_task2/output_logs2"

# accessing case_results which stores every run result for every case 
# key = case_id, value = list of results across all runs that case appeared in 
case_results = defaultdict(list)

# to store which cases were false positives in each run
# key = run folder name with split and seed, value = list of false positive case ids
false_positives_per_run = {}

# iterating through all 50 run directories 
for folder in sorted(os.listdir(output_directory)):
    folder_path = os.path.join(output_directory, folder)
    
    # skipping anything that is not a run directory like the output log
    if not os.path.isdir(folder_path) or not folder.startswith("split"):
        continue
    
    predictions_path = os.path.join(folder_path, "predictions.csv")
    # error handling for missing file 
    if not os.path.exists(predictions_path):
        print(f"[WARN] No predictions.csv in {folder}")
        continue

    # extract the split and seed from the folder name (each have split#_seed# structure)
    parts = folder.split("_")
    split = parts[0]  # split#
    seed = parts[1]   # seed#

    #load this run's per case predictions
    df = pd.read_csv(predictions_path)

    # for each case in this run record if it was correct 
    for _, row in df.iterrows():
        case_id = row["case_id"]
        correct = bool(row["correct"])
        case_results[case_id].append({
            "split": split,
            "seed": seed,
            "correct": correct,
            "true_label": row["true_label"],
            "predicted_label": row["predicted_label"],
        })

    # also find the false positives (benign cases, true label=0, 
    # and predicted as high grade, predicted label=1) for this run
    fp = df[(df["true_label"] == 0) & (df["predicted_label"] == 1)]["case_id"].tolist()
    false_positives_per_run[folder] = fp

# ---------------------------------------------------
# Classification Consistency analysis (per case)

# For each case, what percentage of the runs was it classified correctly 
# only for runs where that case appeared in the test set
# ---------------------------------------------------
consistency_rows = []

#iterating through every case and its history across all runs
for case_id, records in case_results.items():
    # total number of runs this case appeared in as a test case
    total_appearances = len(records)

    # count how many of those runs it was classified correctly
    correct_count = sum(1 for r in records if r["correct"])
    # precentage correct out of appearances
    pct_correct = 100.0 * correct_count / total_appearances
    true_label = records[0]["true_label"] # same across all runs for a case

    consistency_rows.append({
        "case_id": case_id,
        "true_label": true_label,
        "total_appearances": total_appearances,
        "correct_count": correct_count,
        "pct_correct": round(pct_correct, 2),
    })

# sort ascending by pct_correct so least consistent cases appear first 
consistency_df = pd.DataFrame(consistency_rows).sort_values("pct_correct")
print("\nClassification Consistency per Case:")
print(consistency_df.to_string(index=False))

consistency_df.to_csv(os.path.join(output_directory, "classification_consistency.csv"), index=False)

# summary statistics on the classification consistency  

total_cases = len(consistency_df)

# count cases at 100% consistency
perfect_cases = len(consistency_df[consistency_df["pct_correct"] == 100.0])
pct_perfect = 100.0 * perfect_cases / total_cases

# count cases at 0% consistency (never correct regardless of seed)
never_correct = len(consistency_df[consistency_df["pct_correct"] == 0.0])

# cases with any instability (below 100%)
unstable = consistency_df[consistency_df["pct_correct"] < 100.0]
unstable_benign = len(unstable[unstable["true_label"] == 0])
unstable_highgrade = len(unstable[unstable["true_label"] == 1])

# Cases with significant instability (below 50%)
very_unstable = consistency_df[consistency_df["pct_correct"] < 50.0]
very_unstable_benign = len(very_unstable[very_unstable["true_label"] == 0])
very_unstable_highgrade = len(very_unstable[very_unstable["true_label"] == 1])

print("\nConsistency Summary:")
print(f"  Total cases: {total_cases}")
print(f"  Cases at 100% consistency: {perfect_cases} ({pct_perfect:.1f}%)")
print(f"  Cases below 100% (any instability): {len(unstable)}")
print(f"  Cases at 0% (never correct): {never_correct}")
print(f"\n  Of cases with any instability (below 100%):")
print(f"    Benign cases:     {unstable_benign}/{len(unstable)}")
print(f"    High-grade cases: {unstable_highgrade}/{len(unstable)}")
print(f"\n  Of cases below 50% consistency:")
print(f"    Total:            {len(very_unstable)}")
print(f"    Benign cases:     {very_unstable_benign}/{len(very_unstable)}")
print(f"    High-grade cases: {very_unstable_highgrade}/{len(very_unstable)}")

# ---------------------------------------------------
# Misclassified Benign analysis- tracking fps across all runs
# for each of the 50 runs, which benign cases were misclassified as high grade
# ---------------------------------------------------
fp_rows = []

#iterating through every run and its list of false positive case ids
for run_name, fp_cases in sorted(false_positives_per_run.items()):
    # extract split and seed
    parts = run_name.split("_")

    fp_rows.append({
        "run": run_name,
        "split": parts[0],
        "seed": parts[1],
        "false_positive_cases": str(fp_cases), #list of case ids that were FPs
        "num_false_positives": len(fp_cases), # count of FPs in this run
    })

# one row per run showing which cases were FPs and how many
fp_df = pd.DataFrame(fp_rows)
print("\nFalse Positives per Run:")
print(fp_df.to_string(index=False))

fp_df.to_csv(os.path.join(output_directory, "false_positives_per_run.csv"), index=False)

# ---------------------------------------------------
# Summary statistics on false positives

# Count total false positives across all 50 runs
total_fp = fp_df["num_false_positives"].sum()
avg_fp_per_run = fp_df["num_false_positives"].mean()
runs_with_zero_fp = len(fp_df[fp_df["num_false_positives"] == 0])

# Per split summary
print("\nFalse Positive Summary:")
print(f"  Total false positives across all 50 runs: {total_fp}")
print(f"  Average false positives per run: {avg_fp_per_run:.2f}")
print(f"  Runs with zero false positives: {runs_with_zero_fp}/50")

print("\n  Per split breakdown:")
for split in sorted(fp_df["split"].unique()):
    split_df = fp_df[fp_df["split"] == split]
    split_total = split_df["num_false_positives"].sum()
    split_zero = len(split_df[split_df["num_false_positives"] == 0])
    # find which cases appeared as FP in this split
    all_fp_cases = []
    for cases in split_df["false_positive_cases"]:
        import ast
        all_fp_cases.extend(ast.literal_eval(cases))
    unique_fp_cases = sorted(set(all_fp_cases))
    print(f"    {split}: total FPs={split_total}, runs with 0 FP={split_zero}/10, unique FP cases={unique_fp_cases}")

# ---------------------------------------------------
# Cross reference analysis with presentation 3's optional Task 3

# cases 21, 86, 96 were the misclassified benign cases
# specifically checking how the model performed on those cases across the 50 runs to see if seed or split specific 
# ---------------------------------------------------
task3_cases = [21, 86, 96]

print("\nCross-reference with Optional Task 3 misclassified benign cases:")
for case_id in task3_cases:
    #check if this case appeared in any test set across the 50 runs
    if case_id not in case_results:
        print(f"  Case {case_id}: did not appear in any test set")
        continue
    
    records = case_results[case_id]

    # total runs this case appeared in as a test case
    appearances = len(records)

    # count unique splits this case appeared in
    splits_appeared = set(r["split"] for r in records)

    # count how many of those runs it was a FP and then the percentage of appearances 
    fp_count = sum(1 for r in records if r["true_label"] == 0 and r["predicted_label"] == 1)
    pct_fp = 100.0 * fp_count / appearances if appearances > 0 else 0
    
    print(f"  Case {case_id}: appeared in {appearances} runs "
          f"across {len(splits_appeared)} split(s) {sorted(splits_appeared)}, "
          f"false positive in {fp_count}/{appearances} runs ({pct_fp:.1f}%)")

