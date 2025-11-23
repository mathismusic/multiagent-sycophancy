import argparse
import json
import time

from utils import *
from prompt import *
from response_models import LLMResponse

def multiagent_debate(
        pipelines,
        system_prompt,
        user_prompt,
        gen_config,
        sycophancy_scores=None,
        args=None,
        metric=None,
):
    """
    Run a multi-agent debate.

    pipelines: dict[model_name -> pipeline]
    sycophancy_scores: optional dict[model_name -> float] (same value each round)
    """

    startTime = time.time()

    # Initialize the debate state
    debate_state = {
        model_name: {
            "system": system_prompt,
            "user": user_prompt,
            "responses": [],
        }
        for model_name in pipelines.keys()
    }

    # FULL LOG FOR EXPORT
    debate_log = []

    for round_num in range(args.rounds):

        round_start_time = time.time()


        print(f"--- Round {round_num + 1} ---")
        round_entry = {
            "round": round_num + 1,
            "per_model": {}
        }
        new_responses = {}

        print(f"Debate State at start of round {round_num + 1}: {debate_state}")

        # 1) Each model generates a response
        for model_name, pipeline in pipelines.items():
            system = debate_state[model_name]["system"]
            user = debate_state[model_name]["user"]

            # 1) Get raw model output
            raw_output = chat(pipeline, system, user, gen_config)

            # 2) Parse according to metric (same as before)
            parsing_func = parsing_metrics_map[metric]
            parsed_output = parsing_func(raw_output)

            # 3) Wrap + clean + validate via Pydantic
            response_obj = LLMResponse.from_metric(
                model_name=model_name,
                metric=metric,
                round_index=round_num + 1,
                system_prompt=system,
                user_prompt=user,
                raw_output=raw_output,
                parsed_output=parsed_output,
                sycophancy_scores=(
                    sycophancy_scores if args.use_bss_scores and sycophancy_scores is not None else None
                ),
            )

            # 4) Use the cleaned parsed_output for debate state
            new_responses[model_name] = response_obj.parsed_output
            debate_state[model_name]["responses"].append(response_obj.parsed_output)

            print(
                f"{model_name} response: {response_obj.raw_output}, "
                f"parsed as: {response_obj.parsed_output} "
                f"(valid={response_obj.is_parsed_valid})"
            )

            # 5) Log the structured dict instead of a loose dict
            round_entry["per_model"][model_name] = response_obj.dict()


        # 2) Update prompts for the *next* round
        for model_name in pipelines.keys():
            # All *other* models' responses
            other_responses = "\n".join(
                f"{other_model}: {resp}" for other_model, resp in new_responses.items() if other_model != model_name
            )

            # Starting from *second* round (round_num == 1), add sycophancy info
            if args.use_bss_scores and sycophancy_scores is not None and round_num >= 1:
                # Build lines of "other_model: score"
                other_scores_lines = []
                for other_model in pipelines.keys():
                    if other_model != model_name and other_model in sycophancy_scores:
                        other_scores_lines.append(f"{other_model}: {sycophancy_scores[other_model]}")

                scores_block = "\n".join(other_scores_lines)
                debate_state[model_name]["user"] += MMLU[metric.upper()][2].format(
                    other_responses_block=other_responses,
                    scores_block=scores_block)

            else:
                debate_state[model_name]["user"] += MMLU[metric.upper()][1].format(
                    other_responses_block=other_responses
                )

        round_end_time = time.time()
        total_round_time = round_end_time - round_start_time
        round_entry["round_time_seconds"] = total_round_time


        # Save round log
        debate_log.append(round_entry)

    endTime = time.time()
    total_debate_time_taken = endTime - startTime

    debate_log.append({
        "total_debate_time_seconds": total_debate_time_taken
    })

    return debate_state, debate_log

def load_bss_scores(filepath: str) -> dict[str, float]:
    """
    Load BSS scores from a JSON file.
    """
    with open(filepath, "r") as f:
        bss_data = json.load(f)
    return bss_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        "-m",
        nargs="+",              # ← THIS makes it a list
        required=True,
        help="Space-separated list of model names",
    )
    parser.add_argument("--dataset", default="MMLU", type=str)
    parser.add_argument("--rounds", default=5, type=int)
    parser.add_argument("--no_of_samples", default=10, type=int)
    parser.add_argument("--dataset_type", default="val", type=str)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--metric", nargs="+", default=["picking_sides", "mirroring", "who_said", "delusion"], type=str)
    parser.add_argument("--use_bss_scores", action="store_true", default=False)

    ###############################################################################
    # Parsing only known args here
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[note] Ignoring unknown CLI args (likely from Jupyter): {unknown}")

    device = pick_device(args.device)
    gen = GenConfig()
    set_seed(gen.seed)

    ###############################################################################
    # load bss_scores
    bss_scores = load_bss_scores("bss_scores.json")
    print(f"BSS Scores: {bss_scores}")

    ###############################################################################
    # call build_textgen_pipeline for each selected model
    pipelines = {}
    for model_name in args.model:
        pipelines[model_name] = build_textgen_pipeline(model_name, device)

    print(f"Built pipelines for models: {list(pipelines.keys())} on device {device}")

    ###############################################################################
    # load dataset
    # no_of_items = 1
    mmlu_samples = load_mmlu_from_hf(args.dataset_type, args.no_of_samples)
    print(f"\nMMLU_Samples: ", len(mmlu_samples))

    ###############################################################################
    # iterate over each metric in args.metric

    all_prompts = []
    all_logs = []


    for metric in args.metric:
        if metric not in mmlu_metrics_map:
            raise ValueError(f"Unsupported metric: {metric}")

        print(f"\nProcessing metric: {metric}")

        func = mmlu_metrics_map[metric]
        mmlu_sample_list = func(mmlu_samples)

        print(f"\nMMLU_Sample_List:", len(mmlu_sample_list))

        for sample_idx, sample in enumerate(mmlu_sample_list):
            print(f"\nProcessing sample: {sample}")

            for user_stance in ["correct", "wrong"]:

                # skip delusion wrong case
                if metric == "delusion" and user_stance == "wrong":
                    continue

                ###############################################################################
                #prompts
                system = MMLU_SYSTEM_PROMPT_INITAL

                chosen_prompt = metric.upper()
                print(f"\nChosen Prompt: {chosen_prompt}")

                prompt = MMLU[chosen_prompt]
                print(f"\nPrompt: {prompt}")

                # write a switch case for metric types
                user_prompt = ""

                match metric:
                    case "picking_sides":
                        user_prompt = prompt[0].format(
                            question=sample.question,
                            user_answer=sample.correct if user_stance == "correct" else sample.wrong,
                            friend_answer=sample.wrong if user_stance == "correct" else sample.correct,
                        )

                    case "mirroring":
                        user_prompt = prompt[0].format(
                            question=sample.question,
                            belief_answer=sample.correct if user_stance == "correct" else sample.wrong,
                        )

                    case "who_said":
                        user_prompt = prompt[0].format(
                            speaker="I" if user_stance == "correct" else "Someone online",
                            question=sample.question,
                            asserted_answer=sample.wrong,
                        )

                    case "delusion":
                        user_prompt = prompt[0].format(
                            question=sample.question,
                            correct_answer=sample.correct,
                            delusional_answer=sample.wrong,
                        )

                all_prompts.append(user_prompt)
                conversation_id = f"{metric}_sample{sample_idx}_{user_stance}"

                print(f"\n Final System Prompt: {system}")
                print(f"\n Final User Prompt: {user_prompt}")

                    # ----------------- BUILD METADATA RECORD -----------------
                # try to capture as much sample metadata as possible
                sample_meta = {
                    "question": getattr(sample, "question", None),
                    "correct": getattr(sample, "correct", None),
                    "wrong": getattr(sample, "wrong", None),
                    # add more fields if your sample object has them:
                    "subject": getattr(sample, "subject", None),
                    # "id": getattr(sample, "id", None),
                }
                logging_record = {
                    # high-level config
                    "dataset": args.dataset,
                    "dataset_type": args.dataset_type,
                    "metric": metric,
                    "user_stance": user_stance,
                    "rounds": args.rounds,
                    "models": args.model,
                    "use_bss_scores": args.use_bss_scores,

                    # sample indexing
                    "sample_index": sample_idx,

                    # prompts
                    "system_prompt": system,
                    "prompt_key": chosen_prompt,       # e.g. "PICKING_SIDES"
                    "prompt_template": prompt[0],      # original template
                    "user_prompt": user_prompt,

                    # sample metadata
                    "sample": sample_meta,

                    # sycophancy info (static metadata; debate-state-specific stuff
                    # would be added after running multiagent_debate)
                    "bss_scores": bss_scores,
                }

                ##############################################################################
                # Multi-agent debate
                debate_state, debate_log = multiagent_debate(
                    pipelines,
                    system,
                    user_prompt,
                    gen,
                    sycophancy_scores=bss_scores,
                    args=args,
                    metric = metric,
                )

                # Add debate log and final state to logging record
                logging_record["debate_state"] = debate_state
                logging_record["debate_log"] = debate_log
                all_logs.append({conversation_id: logging_record})

                print(f"Final Debate State: {debate_state}")

    print(f"\nDebated on Total Prompts: {len(all_prompts)}")
    print(f"\nTotal examples recorded: {len(all_logs)}")

    with open("mmlu_debate_metadata.jsonl", "w") as f:
        for rec in all_logs:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()