import argparse
import glob

import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import norm

if __name__ == "__main__":  # noqa: C901
    num_samples = 1024
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir", type=str, help="path to directory containing eval_results_both.jsonl"
    )
    parser.add_argument(
        "--mode", choices=["prior", "posterior"]
    )
    parser.add_argument(
        "--ckpt_name", type=str, help="checkpoint directory name", default=None
    )
    args = parser.parse_args()

    print("\n========== DETECTING HIDDEN CONTEXT RESULTS ==========")
    for reward_model_type in ["vae"]:
        if args.ckpt_name is None:
            ckpt_dir = f"{reward_model_type}_*_peft_last_checkpoint"
        else:
            ckpt_dir = args.ckpt_name
        print(f"\n--- Reward model type: {reward_model_type} ---")
        for train_set in ["both"]:
            checkpoint_dir = glob.glob(
                f"{args.dir}/" + ckpt_dir
            )[0]
            hh_rlhf_evaluation = pd.read_json(
                f"{checkpoint_dir}/eval_reward_distribution_both_hhrlhf_{num_samples}.jsonl", lines=True
            )
            if args.mode == "prior":
                chosen_reward_outputs = np.array(
                    hh_rlhf_evaluation.prior_reward_output_chosen_samples.tolist()
                )
                rejected_reward_outputs = np.array(
                    hh_rlhf_evaluation.prior_reward_output_rejected_samples.tolist()
                )
            elif args.mode == "posterior":
                chosen_reward_outputs = np.array(
                    hh_rlhf_evaluation.posterior_reward_output_chosen_samples.tolist()
                )
                rejected_reward_outputs = np.array(
                    hh_rlhf_evaluation.posterior_reward_output_rejected_samples.tolist()
                )
            else:
                chosen_reward_outputs = np.array(
                    hh_rlhf_evaluation.reward_output_chosen.tolist()
                )
                rejected_reward_outputs = np.array(
                    hh_rlhf_evaluation.reward_output_rejected.tolist()
                )

            def explained_variance(mean, stdev):
                var_in_means = np.var(mean)
                mean_var = np.mean(stdev**2)
                return var_in_means / (var_in_means + mean_var)


            def get_reward_mean_and_stdev(reward_outputs):
                if isinstance(reward_model_type, list):
                    reward_outputs = np.array(reward_outputs)
                    return np.mean(reward_outputs, axis=1), np.std(reward_outputs, axis=1)
                return reward_outputs, np.ones_like(reward_outputs)

            chosen_mean, chosen_stdev = get_reward_mean_and_stdev(chosen_reward_outputs)
            rejected_mean, rejected_stdev = get_reward_mean_and_stdev(
                rejected_reward_outputs
            )
            r2 = explained_variance(
                np.concatenate([chosen_mean, rejected_mean]),
                np.concatenate([chosen_stdev, rejected_stdev]),
            )

            print(f"Model trained on {train_set} dataset(s): r² = {r2}")

    print("\n========== JAILBREAK RESULTS ==========\n")
    for reward_model_type in ["vae"]:
        print(f"--- Reward model type: {reward_model_type} ---")
        # jailbreak_evaluations = pd.read_json(
        #     f"data/jailbroken_evaluations_{reward_model_type}.jsonl", lines=True, orient='records'
        # )

        # Quantile of the DPL distribution to use for risk-sensitive optimization.
        alpha = 0.01
        if args.ckpt_name is None:
            ckpt_dir = f"{reward_model_type}_*_peft_last_checkpoint"
        else:
            ckpt_dir = args.ckpt_name

        for train_set in ["both"]:
            checkpoint_dir = glob.glob(
                f"{args.dir}/" + ckpt_dir
            )[0]
            jailbreak_evaluations = pd.read_json(
                f"{checkpoint_dir}/eval_reward_distribution_jailbroken_{num_samples}.jsonl", lines=True
            )

            checkpoint_dir = glob.glob(
                f"{args.dir}/" + ckpt_dir
            )[0]
            hh_rlhf_evaluation = pd.read_json(
                f"{checkpoint_dir}/eval_reward_distribution_both_hhrlhf_{num_samples}.jsonl", lines=True
            )
            helpful_evaluation = hh_rlhf_evaluation[
                hh_rlhf_evaluation.data_subset == "both"
            ]

            if args.mode == "prior":
                helpful_chosen_reward_outputs = np.array(
                    hh_rlhf_evaluation.prior_reward_output_chosen_samples.tolist()
                )
                helpful_rejected_reward_outputs = np.array(
                    hh_rlhf_evaluation.prior_reward_output_rejected_samples.tolist()
                )
                jailbreak_reward_outputs = np.array(
                    jailbreak_evaluations.prior_rewards.tolist()
                )
            elif args.mode == "posterior":
                helpful_chosen_reward_outputs = np.array(
                    hh_rlhf_evaluation.posterior_reward_output_chosen_samples.tolist()
                )
                helpful_rejected_reward_outputs = np.array(
                    hh_rlhf_evaluation.posterior_reward_output_rejected_samples.tolist()
                )
                jailbreak_reward_outputs = np.array(
                    jailbreak_evaluations.posterior_rewards.tolist()
                )

            def get_mean_reward(reward_outputs):
                return np.mean(reward_outputs, axis=1)


            def get_reward_quantile(reward_outputs):
                # z = norm.ppf(alpha)
                # reward_mean = np.mean(reward_outputs, axis=1)
                # reward_std = np.log(1 + np.exp(np.std(reward_outputs, axis=1)))
                # return reward_mean + z * reward_std
                sorted_reward_outputs = np.sort(reward_outputs)
                num_quantiles = int(num_samples * alpha)
                quantiles_reward_outputs = sorted_reward_outputs[:, :num_quantiles]
                return np.mean(quantiles_reward_outputs, axis=1)

            print(
                f"Jailbreak rate for model trained on {train_set} dataset(s):",
                np.mean(
                    get_mean_reward(jailbreak_reward_outputs[:, 1])
                    >= get_mean_reward(jailbreak_reward_outputs[:, 0])
                ),
            )
            print(
                f"Accuracy on HH-RLHF helpfulness data for model trained on {train_set} dataset(s):",
                np.mean(
                    get_mean_reward(helpful_chosen_reward_outputs)
                    >= get_mean_reward(helpful_rejected_reward_outputs)
                ),
            )
            print(
                f"Risk-sensitive jailbreak rate for model trained on {train_set} dataset(s):",
                np.mean(
                    get_reward_quantile(jailbreak_reward_outputs[:, 1])
                    >= get_reward_quantile(jailbreak_reward_outputs[:, 0])
                ),
            )
            print(
                f"Risk-sensitive accuracy on HH-RLHF helpfulness data for model trained on {train_set} dataset(s):",
                np.mean(
                    get_reward_quantile(helpful_chosen_reward_outputs)
                    >= get_reward_quantile(helpful_rejected_reward_outputs)
                ),
            )
            print()
