from process_eval_results import overall_from_csv

csv_path = "eval_results/mmlu_debate_metadata_iterated__EVAL/summary_all.csv"
title = "MMLU Debate Results (Iterated)"
df = overall_from_csv(csv_path, title)
print(df.to_string(index=False))