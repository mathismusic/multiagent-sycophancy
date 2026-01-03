from process_eval_results import overall_from_csv

csv_path = "/Users/abdulrahmanalrabah/Desktop/UIUC/Fall 2025/CS 546/Project/project/multiagent-sycophancy/bss/mmlu_debate_metadataـ__EVAL/summary_all.csv"
title = "MMLU Debate Results (Iterated)"
df = overall_from_csv(csv_path, title)
print(df.to_string(index=False))