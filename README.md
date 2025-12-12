# multiagent-sycophancy
Evaluating multi-agent sycophancy

(started as part of CS 546 Advanced NLP, Fall 2025)

Please see the report for details. Here, we restrict to running instructions.

All the magic happens in `bss/`. So we `cd bss`.

An environment (Python 3.10.18 works with CUDA 12.8) can easily be setup with `requirements.txt`.

To compute BSS scores: `python bss_calc.py > bss_log`.

To run the debate,

```
python multiagent-debate.py --model llama1b llama3b llama8b qwen3b qwen7b qwen14b \
        --dataset MMLU --dataset_type val --rounds 5
```

The arguments `--use_bss_scores` and `--use_dss_scores` are optional and use the BSS and DSS to provide credibility scores to the agents during debate. The output is saved to a `.jsonl` file in `.`.

To evaluate a debate, run

```
python debate_evaluation.py --debate-results-jsonl=<jsonl file from multiagent-debate>
```

and to reproduce our results, please run `python process_eval_results.py` and the cells in `plotter.ipynb`.

Thank you!