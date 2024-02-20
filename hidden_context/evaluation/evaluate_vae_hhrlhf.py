# This file is used to evaluate VAE model on HH-RLHF dataset
# TODO: Not compatible with new framework yet!!!
import os
from dataclasses import dataclass, field
from typing import Optional, cast

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)

from hidden_context.train_llm_preference_model import (
    DataSubset,
    get_hh_rlhf_dataset,
)

from hidden_context.vae_utils import VAEModel, Annealer
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
    checkpoint_name: str = field(
        metadata={"help": "Directory name of the trained reward model checkpoint."}
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
    max_length: int = field(default=1024)
    bf16: bool = field(
        default=True,
        metadata={"help": "Whether to use bfloat16 precision."},
    )
    latent_dim: int = field(default=512)
    hidden_dim: int = field(default=512)
    embed_dim: int = field(default=1024)
    fixed_contexts: bool = field(
        default=True,
        metadata={"help": "Whether to use pre-calculated fixed contexts embeddings."}
    )


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
    print(script_args)
    import ipdb; ipdb.set_trace()

    if script_args.fixed_contexts:
        from hidden_context.train_llm_vae_preference_model_fixed_contexts import \
            HHRLHFPreprocessor, RewardDataCollatorWithPadding
    else:
        from hidden_context.train_llm_vae_preference_model import \
            HHRLHFPreprocessor, RewardDataCollatorWithPadding

    data_subset = cast(DataSubset, script_args.data_subset)

    output_fname = script_args.output
    if output_fname is None:
        output_fname = os.path.join(
            script_args.reward_model_checkpoint, f"eval_vae_hhrlhf_{data_subset}.jsonl"
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

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )

    torch.set_anomaly_enabled(True)

    embed_dim = script_args.embed_dim

    model = AutoModelForSequenceClassification.from_pretrained(
        script_args.model_name, num_labels=embed_dim, torch_dtype=torch.bfloat16
    )
    # We multiply the final linear layer's weights by 0.01 because this seems to
    # significantly stabilize training and lead to better optimization of the loss.
    model.score.weight.data *= 0.01
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    # model.from_pretrained(os.path.join(script_args.reward_model_checkpoint, script_args.checkpoint_name))

    # Need to do this for GPT2 and Llama because they don't have official pad tokens.
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "right"

    num_proc = 24  # Can adjust to be higher if you have more processors.
    num_samples = script_args.num_outputs
    original_columns = eval_dataset.column_names
    ipdb.set_trace()
    eval_dataset = eval_dataset.map(
        HHRLHFPreprocessor(tokenizer),
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    eval_dataset = eval_dataset.filter(
        lambda x: x["max_lengths"] <= script_args.max_length
        # and len(x["input_ids_rejected"]) <= script_args.max_length
    )

    latent_dim = script_args.latent_dim
    hidden_dim = script_args.hidden_dim
    vae_model = VAEModel(embed_dim, hidden_dim, latent_dim, model, use_fixed_contexts=True)
    # vae_model.load_state_dict(torch.load(os.path.join(
    #     script_args.reward_model_checkpoint,
    #     script_args.checkpoint_name,
    #     'model.pt'
    # )))

    collator = RewardDataCollatorWithPadding(tokenizer,
                                             padding=True,
                                             max_length=script_args.max_length,
                                             pad_to_multiple_of=64)

    def compute_rewards(example):
        import ipdb; ipdb.set_trace()
        outputs = {}
        inputs = collator([example])
        embeddings = vae_model.llm_encoder(
            torch.concatenate(
                [
                    inputs["input_ids_chosen"],
                    inputs["input_ids_rejected"],
                ],
                dim=0,
            ),
            torch.concatenate(
                [
                    inputs["attention_mask_chosen"],
                    inputs["attention_mask_rejected"],
                ],
                dim=0,
            ),
        )[0]

        batch_size = inputs["input_ids_chosen"].shape[0]
        target_chosen = embeddings[:batch_size]
        target_rejected = embeddings[batch_size:2 * batch_size]

        if "embeddings_context_chosen" not in inputs.keys():
            # context = embeddings[2*batch_size:]
            # context_chosen = context[: len(context) // 2]
            # context_rejected = context[len(context) // 2 :]
            context_chosen = model.llm_encoder(
                inputs["input_ids_context_chosen"], inputs["attention_mask_context_chosen"])[0]
            context_rejected = model.llm_encoder(
                inputs["input_ids_context_rejected"], inputs["attention_mask_context_rejected"])[0]
        else:
            context_chosen = torch.tensor(inputs["embeddings_context_chosen"]).to(embeddings.device)
            context_rejected = torch.tensor(inputs["embeddings_context_rejected"]).to(embeddings.device)
        seq_start_end = inputs["seq_start_end"]

        rewards_chosen, rewards_rejected, mean, log_var = model(
            target_chosen,
            target_rejected,
            context_chosen,
            context_rejected,
            seq_start_end,
        )

        outputs[f"reward_output_chosen"] = [rewards_chosen.mean().item()]
        outputs[f"reward_output_rejected"] = [rewards_rejected.mean().item()]
        outputs["latent_mean"] = mean.tolist()
        outputs["latent_log_var"] = log_var.tolist()
        return outputs

    eval_results = eval_dataset.map(
        compute_rewards,
        batched=True,
        batch_size=script_args.batch_size,
    )

    eval_results.to_json(output_fname, orient="records", lines=True)

