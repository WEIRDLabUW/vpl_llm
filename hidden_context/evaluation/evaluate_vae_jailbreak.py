# This file is used to evaluate VAE model on Jailbreak dataset
# TODO: Not compatible with new framework yet!!!
import os
from dataclasses import dataclass, field
from typing import List, Optional, cast

import multiprocess
import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)
import numpy as np

import sys, ipdb, traceback

def info(type, value, tb):
    traceback.print_exception(type, value, tb)
    ipdb.pm()

sys.excepthook = info


@dataclass
class ScriptArguments:
    reward_model_checkpoints: str = field(
        metadata={
            "help": "Paths to the reward model checkpoints to use for evaluation."
        }
    )
    # reward_model_names: str = field(
    #     metadata={"help": "Names of the models to use for evaluation."}
    # )
    input: str = field(
        metadata={"help": "JSONL file with responses to evaluate."},
    )
    output: str = field(
        default=None,
        metadata={"help": "JSONL file for results."},
    )
    batch_size: int = field(default=1)
    model_name: str = field(
        default="gpt2",
        metadata={
            "help": "The model that you want to use as the basis for generation and for evaluation."
                    "E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The tokenizer for your model, if left empty will use the default "
                    "for your model",
        },
    )
    num_outputs: int = field(default=1)
    max_length: int = field(default=1024)
    bf16: bool = field(
        default=True,
        metadata={"help": "Whether to use bfloat16 for the reward model."},
    )


if __name__ == "__main__":
    multiprocess.set_start_method("spawn")

    parser = HfArgumentParser(ScriptArguments)
    script_args = cast(ScriptArguments, parser.parse_args_into_dataclasses()[0])
    print(script_args)

    output_fname = script_args.output
    if output_fname is None:
        output_fname = os.path.join(
            script_args.reward_model_checkpoints, f"eval_reward_distribution_jailbroken.jsonl"
        )

    # Load the value-head model and tokenizer.
    tokenizer_name = (
        script_args.tokenizer_name
        if script_args.tokenizer_name is not None
        else script_args.model_name
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_auth_token=True)

    # Need to do this for GPT2 and Llama because they dosn't have official pad tokens.
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    dataset = load_dataset("json", data_files=[script_args.input])["train"]

    print("Loading base reward model...")
    model_kwargs = {}
    if script_args.bf16:
        model_kwargs["torch_dtype"] = torch.bfloat16
    # base_reward_model = AutoModelForSequenceClassification.from_pretrained(
    #     script_args.model_name,
    #     num_labels=script_args.num_outputs,
    #     **model_kwargs,
    # )
    reward_model = torch.load(script_args.reward_model_checkpoints + "final_vae_model.pt")

    # for model_name, checkpoint_path in zip(
    #     script_args.reward_model_names, script_args.reward_model_checkpoints
    # ):
    # peft_config = LoraConfig.from_pretrained(checkpoint_path)
    # reward_model = PeftModel.from_pretrained(
    #     base_reward_model, checkpoint_path, is_trainable=False
    # )
    reward_model.cuda().eval()
    reward_model.llm.pad_token_id = tokenizer.pad_token_id

    latent_means = np.load(f"{script_args.reward_model_checkpoints}/latent_mean_both.npy")
    latent_logvars = np.load(f"{script_args.reward_model_checkpoints}/latent_logvar_both.npy")

    num_samples = 1024
    print(f"Sampling: {num_samples} from {latent_means.shape[0]}")


    # latents_logvars = np.load(f"latent_vars/latent_logvar_{script_args.data_subset}.npy", latent_logvars)

    def evaluate_prior_rewards(example):
        output = {}
        prompt: str = example["prompt"]
        prompt = prompt[prompt.index("Human: "):]

        responses = example["responses"]
        reward_outputs = []
        reward_samples = []
        for response in responses:
            inputs = tokenizer(
                prompt + response,
                return_tensors="pt",
                max_length=script_args.max_length,
            )
            e0 = reward_model.llm(
                inputs.input_ids.cuda(), inputs.attention_mask.cuda()
            )[0]

            e0 = e0.repeat(num_samples, 1)
            sample_idxs = torch.randperm(latent_means.shape[0])[:num_samples]
            sample_latents = torch.tensor(latent_means[sample_idxs], dtype=e0.dtype).cuda()[:, 0]
            e0 = torch.cat([e0, sample_latents], dim=-1).float()
            reward_output = reward_model.Decoder(e0)
            reward_outputs.append([reward_output.mean().item(), reward_output.std().item()])
            reward_samples.append(reward_output.squeeze().tolist())

        output[f"reward_outputs"] = reward_outputs
        output[f"prior_rewards"] = reward_samples
        return output


    print(f"Evaluating responses with gpt2...")
    results = dataset.map(
        evaluate_prior_rewards,
        batched=False,
    )

    def evaluate_posterior_rewards(example):
        output = {}
        prompt: str = example["prompt"]
        prompt = prompt[prompt.index("Human: "):]

        responses = example["responses"]
        inputs_0 = tokenizer(
            prompt + responses[0],
            return_tensors="pt",
            max_length=script_args.max_length,
        )
        e0 = reward_model.llm(
            inputs_0.input_ids.cuda(), inputs_0.attention_mask.cuda()
        )[0]
        inputs_1 = tokenizer(
            prompt + responses[1],
            return_tensors="pt",
            max_length=script_args.max_length,
        )
        e1 = reward_model.llm(
            inputs_1.input_ids.cuda(), inputs_1.attention_mask.cuda()
        )[0]
        e0 = e0.float()
        e1 = e1.float()
        with torch.no_grad():
            fused_embed = torch.cat([e0, e1], dim=-1)
            _, rewards_chosen, rewards_rejected, mean, log_var = reward_model(fused_embed, e0, e1)
            epsilon = torch.randn((num_samples, mean.shape[1])).to(mean.device)
            latent_samples = epsilon * torch.exp(0.5 * log_var) + mean  # num_samples * dim
            e0 = e0.repeat(num_samples, 1)
            e1 = e1.repeat(num_samples, 1)
            posterior_rewards_0 = reward_model.Decoder(torch.cat([e0, latent_samples], dim=-1)).squeeze()
            posterior_rewards_1 = reward_model.Decoder(torch.cat([e1, latent_samples], dim=-1)).squeeze()

        # for response in responses:
        #
        #
        #     e0 = e0.repeat(num_samples, 1)
        #     sample_idxs = torch.randperm(latent_means.shape[0])[:num_samples]
        #     sample_latents = torch.tensor(latent_means[sample_idxs], dtype=e0.dtype).cuda()[:, 0]
        #     e0 = torch.cat([e0, sample_latents], dim=-1).float()
        #     reward_output = reward_model.Decoder(e0)
        #     reward_outputs.append([reward_output.mean().item(), reward_output.std().item()])
        #     reward_samples.append(reward_output.tolist())
        output[f"posterior_rewards"] = [posterior_rewards_0, posterior_rewards_1]
        return output


    print(f"Evaluating responses with gpt2...")
    results = results.map(
        evaluate_posterior_rewards,
        batched=False,
    )

    # Combine datasets and output to JSONL
    results.to_json(output_fname, orient="records", lines=True)
