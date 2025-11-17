import argparse
from utils import *
from prompt import *

def multiagent_debate(
        pipelines,
        system_prompt,
        user_prompt,
        gen_config,
        rounds,
        sycophancy_scores=None,  # dict[str, float], same score across all rounds
):
    """
    Run a multi-agent debate.

    pipelines: dict[model_name -> pipeline]
    sycophancy_scores: optional dict[model_name -> float] (same value each round)
    """
    # Initialize the debate state
    debate_state = {
        model_name: {
            "system": system_prompt,
            "user": user_prompt,
            "responses": [],
        }
        for model_name in pipelines.keys()
    }

    for round_num in range(rounds):
        print(f"--- Round {round_num + 1} ---")
        new_responses = {}

        # 1) Each model generates a response
        for model_name, pipeline in pipelines.items():
            system = debate_state[model_name]["system"]
            user = debate_state[model_name]["user"]

            response = chat(pipeline, system, user, gen_config)
            new_responses[model_name] = response
            print(f"{model_name} response: {response}")

        # 2) Update prompts for the *next* round
        for model_name in pipelines.keys():
            # All *other* models' responses
            other_responses = "\n".join(
                f"{other_model}: {resp}" for other_model, resp in new_responses.items() if other_model != model_name
            )

            debate_state[model_name]["responses"].append(new_responses[model_name])

            # Always append what others just said
            debate_state[model_name]["user"] += (
                f"\n\n[Round {round_num + 1} – other agents' responses]\n"
                f"{other_responses}"
            )

            # Starting from *second* round (round_num == 1), add sycophancy info
            if sycophancy_scores is not None and round_num == 1:
                # Build lines of "other_model: score"
                other_scores_lines = []
                for other_model in pipelines.keys():
                    if other_model == model_name:
                        continue  # skip self
                    if other_model not in sycophancy_scores:
                        continue
                    score_value = sycophancy_scores[other_model]
                    other_scores_lines.append(
                        f"{other_model}: {score_value}"
                    )

                if other_scores_lines:
                    scores_block = "\n".join(other_scores_lines)
                    debate_state[model_name]["user"] += MMLU_DEBATE_PROMPT_PICKING_SIDES.format(scores_block=scores_block)

    return debate_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        "-m",
        nargs="+",              # ← THIS makes it a list
        required=True,
        help="Space-separated list of model names",
    )
    parser.add_argument("--dataset", default="SQA", type = str)
    parser.add_argument("--rounds", default=5, type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--metric", default="picking_sides")

    ###############################################################################
    # Parsing only known args here
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[note] Ignoring unknown CLI args (likely from Jupyter): {unknown}")

    device = pick_device(args.device)
    gen = GenConfig()
    set_seed(gen.seed)

    ###############################################################################
    # call build_textgen_pipeline for each selected model
    pipelines = {}
    for model_name in args.model:
        pipelines[model_name] = build_textgen_pipeline(model_name, device)

    print(f"Built pipelines for models: {list(pipelines.keys())} on device {device}")

    ###############################################################################
    # load dataset
    no_of_items = 1
    mmlu_samples = load_mmlu_from_hf("val", no_of_items)
    print(f"MMLU_Samples: {mmlu_samples}")

    func = mmlu_metrics_map[args.metric]
    mmlu_sample_list = func(mmlu_samples)

    print(f"MMLU_Sample_List: {mmlu_sample_list}")

    ###############################################################################
    #prompts
    system = MMLU_SYSTEM_PROMPT_INITAL
    user_prompt = MMLU_USER_PROMPT_PICKING_SIDES_INITIAL.format(
        question=mmlu_sample_list[0].question,
        user_answer=mmlu_sample_list[0].correct,
        friend_answer=mmlu_sample_list[0].wrong,
    )
    print(f"System Prompt: {system}")
    print(f"User Prompt: {user_prompt}")

    ###############################################################################
    # Multi-agent debate
    debate_state = multiagent_debate(
        pipelines,
        system,
        user_prompt,
        gen,
        args.rounds,
        sycophancy_scores=None,
    )

    print(f"Final Debate State: {debate_state}")


if __name__ == "__main__":
    main()