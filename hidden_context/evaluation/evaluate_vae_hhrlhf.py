# This file is used to evaluate VAE model on HH-RLHF dataset
# TODO: Not compatible with new framework yet!!!
import os
from dataclasses import dataclass, field
from typing import Optional, cast

import torch
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)

from hidden_context.train_llm_preference_model import (
    DataSubset,
    HHRLHFPreprocessor,
    get_hh_rlhf_dataset,
)

from hidden_context.vae_utils import Decoder, Encoder, VAEModel, Annealer
import numpy as np


import sys, ipdb, traceback

def info(type, value, tb):
    traceback.print_exception(type, value, tb)
    ipdb.pm()

sys.excepthook = info


@dataclass
class ScriptArguments:
    reward_model_checkpoint: str = field(
        metadata={"help": "Path to the trained reward model checkpoint."}
    )
    output: Optional[str] = field(
        default=None,
        metadata={"help": "JSONL file where results will be stored."},
    )
    batch_size: Optional[int] = field(default=1)
    model_name: Optional[str] = field(default="gpt2")
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The tokenizer for your model, if left empty will use the default "
                    "for your model",
        },
    )
    data_path: str = field(
        default="Anthropic/hh-rlhf",
    )
    data_subset: str = field(
        default="both",
        metadata={
            "help": "Which subset of the data to use. You can choose between 'both', "
                    "'helpful', or 'harmless'."
        },
    )
    eval_dataset_size: int = field(
        default=0,
        metadata={"help": "The size of the subset of the eval data to use"},
    )
    num_outputs: int = field(
        default=1024,
        metadata={"help": "The number of outputs from the model."},
    )
    max_length: int = field(default=256)
    bf16: bool = field(
        default=True,
        metadata={"help": "Whether to use bfloat16 precision."},
    )


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
    print(script_args)

    data_subset = cast(DataSubset, script_args.data_subset)

    output_fname = script_args.output
    if output_fname is None:
        output_fname = os.path.join(
            script_args.reward_model_checkpoint, f"eval_reward_distribution_{data_subset}_hhrlhf.jsonl"
        )

    eval_dataset = get_hh_rlhf_dataset(
        data_subset,
        "test",
        script_args.eval_dataset_size,
        data_path=script_args.data_path,
    )

    # Load the value-head model and tokenizer.
    tokenizer_name = (
        script_args.tokenizer_name
        if script_args.tokenizer_name is not None
        else script_args.model_name
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_auth_token=True)

    # peft_config = LoraConfig.from_pretrained(script_args.reward_model_checkpoint)
    # model_kwargs = {}
    # if script_args.bf16:
    #     model_kwargs["torch_dtype"] = torch.bfloat16
    # model = AutoModelForSequenceClassification.from_pretrained(
    #     script_args.model_name,
    #     num_labels=script_args.num_outputs,
    #     **model_kwargs,
    # )
    # model = PeftModel.from_pretrained(
    #     model, script_args.reward_model_checkpoint, is_trainable=False
    # )

    # latent_dim = script_args.latent_dim
    # encoder = Encoder(input_dim=2*embed_dim, hidden_dim=latent_dim, latent_dim=latent_dim)
    # decoder = Decoder(input_dim=(latent_dim+embed_dim), hidden_dim=latent_dim)
    # vae_model = VAEModel(encoder, decoder, model, latent_dim=latent_dim, learned_prior=True)
    model = torch.load(f"{script_args.reward_model_checkpoint}/final_vae_model.pt", map_location="cuda")

    # Need to do this for GPT2 and Llama because they don't have official pad tokens.
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.llm.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "right"

    num_proc = 24  # Can adjust to be higher if you have more processors.
    num_samples = 1024
    eval_dataset = eval_dataset.map(
        HHRLHFPreprocessor(tokenizer),
        batched=True,
        num_proc=num_proc,
    )
    eval_dataset = eval_dataset.filter(
        lambda x: len(x["input_ids_chosen"]) <= script_args.max_length
                  and len(x["input_ids_rejected"]) <= script_args.max_length
    )

    latent_means = []
    latent_logvars = []

    def compute_posterior_rewards(example):
        output = {}
        temp_output = {}
        for key in ["chosen", "rejected"]:
            batch = tokenizer.pad(
                {
                    "input_ids": example[f"input_ids_{key}"],
                },
                padding=True,
                max_length=script_args.max_length,
                pad_to_multiple_of=64,
                return_tensors="pt",
            )
            with torch.no_grad():
                temp_output[f"embedding_{key}"] = model.llm(
                    input_ids=batch["input_ids"].to("cuda"),
                    attention_mask=batch["attention_mask"].to("cuda"),
                )[0]
            # embeddings = embeddings.reshape(2, -1, embeddings.shape[-1])
        e0 = temp_output[f"embedding_chosen"].float()
        e1 = temp_output[f"embedding_rejected"].float()
        # fused_embed = torch.cat([e0, e1], dim=-1)
        with torch.no_grad():
            # mean, log_var = model.Encoder(fused_embed)
            # # mean, log_var = model(e0, e1)
            # if num_samples < 2:
            #     latent_samples = mean
            # else:
            #     var = torch.exp(0.5 * log_var)
            #     epsilon = torch.randn((num_samples, var.shape[1])).to(mean.device)
            #     latent_samples = mean + var * epsilon

            # e0 = e0.repeat(num_samples, 1)
            # e1 = e1.repeat(num_samples, 1)
            # rewards_chosen = model.Decoder(torch.cat([e0, latent_samples], dim=-1)).squeeze()
            # rewards_rejected = model.Decoder(torch.cat([e1, latent_samples], dim=-1)).squeeze()
            # embeddings = embeddings.reshape(2, -1, embeddings.shape[-1])
            # e0 = embeddings[0]
            # e1 = embeddings[1]
            fused_embed = torch.cat([e0, e1], dim=-1)
            _, rewards_chosen, rewards_rejected, mean, log_var = model(fused_embed, e0, e1)
            epsilon = torch.randn((num_samples, mean.shape[1])).to(mean.device)
            latent_samples = epsilon * torch.exp(0.5 * log_var) + mean    # num_samples * dim
            e0 = e0.repeat(num_samples, 1)
            e1 = e1.repeat(num_samples, 1)
            posterior_rewards_chosen = model.Decoder(torch.cat([e0, latent_samples], dim=-1)).squeeze().unsqueeze(0)
            posterior_rewards_rejected = model.Decoder(torch.cat([e1, latent_samples], dim=-1)).squeeze().unsqueeze(0)

        # output[f"reward_output_chosen"] = [rewards_chosen.mean().item()]
        # output[f"reward_output_rejected"] = [rewards_rejected.mean().item()]
        output[f"posterior_reward_output_chosen_samples"] = posterior_rewards_chosen.tolist()
        output[f"posterior_reward_output_rejected_samples"] = posterior_rewards_rejected.tolist()
        output["latent_mean"] = mean.tolist()
        output["latent_log_var"] = log_var.tolist()
        latent_means.append(mean.detach().cpu().numpy())
        latent_logvars.append(log_var.detach().cpu().numpy())
        return output

    eval_results = eval_dataset.map(
        compute_posterior_rewards,
        batched=True,
        batch_size=script_args.batch_size,
    )
    latent_means = np.array(latent_means)
    latent_logvars = np.array(latent_logvars)
    np.save(f"{script_args.reward_model_checkpoint}/latent_mean_{script_args.data_subset}.npy", latent_means)
    np.save(f"{script_args.reward_model_checkpoint}/latent_logvar_{script_args.data_subset}.npy", latent_logvars)

    latent_means = torch.from_numpy(latent_means).squeeze()

    def compute_prior_rewards(example):
        output = {}
        temp_output = {}
        for key in ["chosen", "rejected"]:
            batch = tokenizer.pad(
                {
                    "input_ids": example[f"input_ids_{key}"],
                },
                padding=True,
                max_length=script_args.max_length,
                pad_to_multiple_of=64,
                return_tensors="pt",
            )
            with torch.no_grad():
                temp_output[f"embedding_{key}"] = model.llm(
                    input_ids=batch["input_ids"].to("cuda"),
                    attention_mask=batch["attention_mask"].to("cuda"),
                )[0]
            # embeddings = embeddings.reshape(2, -1, embeddings.shape[-1])
        e0 = temp_output[f"embedding_chosen"].float()
        e1 = temp_output[f"embedding_rejected"].float()
        # fused_embed = torch.cat([e0, e1], dim=-1)
        with torch.no_grad():
            # mean, log_var = model.Encoder(fused_embed)
            # # mean, log_var = model(e0, e1)
            # if num_samples < 2:
            #     latent_samples = mean
            # else:
            #     var = torch.exp(0.5 * log_var)
            #     epsilon = torch.randn((num_samples, var.shape[1])).to(mean.device)
            #     latent_samples = mean + var * epsilon

            # e0 = e0.repeat(num_samples, 1)
            # e1 = e1.repeat(num_samples, 1)
            # rewards_chosen = model.Decoder(torch.cat([e0, latent_samples], dim=-1)).squeeze()
            # rewards_rejected = model.Decoder(torch.cat([e1, latent_samples], dim=-1)).squeeze()
            # embeddings = embeddings.reshape(2, -1, embeddings.shape[-1])
            # e0 = embeddings[0]
            # e1 = embeddings[1]
            latent_samples = latent_means[torch.randperm(latent_means.shape[0])[:num_samples]].to(e0.device)
            e0 = e0.repeat(num_samples, 1)
            e1 = e1.repeat(num_samples, 1)
            prior_rewards_chosen = model.Decoder(torch.cat([e0, latent_samples], dim=-1)).squeeze().unsqueeze(0)
            prior_rewards_rejected = model.Decoder(torch.cat([e1, latent_samples], dim=-1)).squeeze().unsqueeze(0)

        # output[f"reward_output_chosen"] = [rewards_chosen.mean().item()]
        # output[f"reward_output_rejected"] = [rewards_rejected.mean().item()]
        output[f"prior_reward_output_chosen_samples"] = prior_rewards_chosen.tolist()
        output[f"prior_reward_output_rejected_samples"] = prior_rewards_rejected.tolist()
        return output


    eval_results = eval_results.map(
        compute_prior_rewards,
        remove_columns=[
            "input_ids_chosen",
            "input_ids_rejected",
            "attention_mask_chosen",
            "attention_mask_rejected",
        ],
        batched=True,
        batch_size=script_args.batch_size,
    )

    eval_results.to_json(output_fname, orient="records", lines=True)

