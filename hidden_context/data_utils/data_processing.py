# This file is used to preprocess dataset, available for any HH-RLHF format datasets
import os
from dataclasses import dataclass, field
from typing import Optional, cast

from transformers import (
    HfArgumentParser,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
)

import torch

from hidden_context.train_llm_preference_model import (
    DataSubset,
    get_hh_rlhf_dataset,
    concatenate_datasets,
    HHRLHFPreprocessor,
)

from copy import deepcopy

import numpy as np

import sys, ipdb, traceback


def info(type, value, tb):
    traceback.print_exception(type, value, tb)
    ipdb.pm()


sys.excepthook = info


@dataclass
class ScriptArguments:
    output_dir: Optional[str] = field(
        metadata={"help": "Directory where the new dataset will be stored."},
    )
    data_path: str = field(
        metadata={"help": "Directory where the original data is stored."}
    )
    data_subset: str = field(
        default="helpful",
        metadata={
            "help": "Which subset of the data to use. You can choose between"
                    "'helpful', or 'harmless'."
        },
    )
    data_split: str = field(
        default="test",
        metadata={
            "help": "Which split of the data to use. You can choose between"
                    "'train', or 'test'."
        },
    )
    dataset_size: int = field(
        default=0,
        metadata={"help": "The size of the subset of the data to use"},
    )
    model_type: str = field(
        default="none",
        metadata={
            "help": "You can choose between 'gpt2', 'llama', or 'none'."
        }
    )
    embed_dim: int = field(
        default=1024,
        metadata={
            "help": "Dimension of the embeddings generated by LLM."
        }
    )
    max_length: int = field(
        default=1024,
        metadata={
            "help": "Maximum token length of the inputs."
        }
    )
    with_embeddings: bool = field(
        default=True,
        metadata={
            "help": "Whether the embeddings are generated during pre-processing."
        }
    )
    add_controversial: bool = field(
        default=False,
        metadata={
            "help": "Whether to add an extra feature which indicates whether the preference is controversial."
        }
    )
    synthetic_dataset: bool = field(
        default=False,
        metadata={
            "help": "Whether a synthetic dataset is used."
        }
    )
    use_causal_lm: bool = field(default=False)
    other_subsets: str = field(default=None)


def generate_embeddings_with_llm(args, input_dataset=None):
    """
    This function is used to generate fixed embeddings for inputs from original dataset.
    """
    if not args.synthetic_dataset:
        data_subset = cast(DataSubset, args.data_subset)
        input_dataset = get_hh_rlhf_dataset(
            data_subset,
            args.data_split,
            args.dataset_size,
            data_path=args.data_path,
            use_subset_as_dir=True,
            other_subsets=args.other_subsets,
        )

    if args.model_type == "gpt2":
        tokenizer = AutoTokenizer.from_pretrained("gpt2", use_auth_token=True)
        if not args.use_causal_lm:
            model = AutoModelForSequenceClassification.from_pretrained(
                "gpt2", num_labels=args.embed_dim, torch_dtype=torch.bfloat16
            )
            model.score.weight.data *= 0.01
        else:
            model = AutoModelForCausalLM.from_pretrained(
                "gpt2", torch_dtype=torch.bfloat16
            )
    elif args.model_type == "llama" or args.model_type == "meta-llama/Llama-2-7b-hf":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_auth_token=True)
        if not args.use_causal_lm:
            model = AutoModelForSequenceClassification.from_pretrained(
                "meta-llama/Llama-2-7b-hf", num_labels=args.embed_dim, torch_dtype=torch.bfloat16
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                "meta-llama/Llama-2-7b-hf", torch_dtype=torch.bfloat16
            )
    else:
        return input_dataset
    model.to("cuda")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "right"

    dataset_size = len(input_dataset)
    print(dataset_size)

    preprocessed_dataset = input_dataset.map(
        HHRLHFPreprocessor(tokenizer),
        batched=True,
        num_proc=24,
        remove_columns=input_dataset.column_names,
    )

    input_dataset = input_dataset.filter(
        lambda example, idx: len(preprocessed_dataset[idx]["input_ids_chosen"]) <= args.max_length
                             and len(preprocessed_dataset[idx]["input_ids_rejected"]) <= args.max_length,
        with_indices=True
    )
    preprocessed_dataset = preprocessed_dataset.filter(
        lambda example: len(example["input_ids_chosen"]) <= args.max_length
                        and len(example["input_ids_rejected"]) <= args.max_length
    )
    print(len(input_dataset), len(preprocessed_dataset))
    dataset_size = len(preprocessed_dataset)

    embeddings = list()
    for row_id in range(dataset_size):
        emb = dict()
        for key in ['chosen', 'rejected']:
            tokens = tokenizer.pad(
                {"input_ids": preprocessed_dataset[row_id][f"input_ids_{key}"]},
                padding=True, pad_to_multiple_of=64, return_tensors="pt"
            )
            with torch.no_grad():
                if not args.use_causal_lm:
                    emb[f"embedding_{key}"] = model(
                        input_ids=tokens["input_ids"].unsqueeze(0).to("cuda"),
                        attention_mask=tokens["attention_mask"].unsqueeze(0).to("cuda")
                    )[0][0].float().cpu().numpy()
                else:
                    input_ids = tokens["input_ids"].unsqueeze(0).to("cuda")
                    attention_mask = tokens["attention_mask"].unsqueeze(0).to("cuda")
                    last_hidden_state = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True
                    ).hidden_states[-1]
                    masked_last_hidden_state = last_hidden_state * attention_mask.unsqueeze(-1)
                    token_length = torch.sum(attention_mask, dim=1)
                    mean_pooling = torch.sum(masked_last_hidden_state, dim=1) / token_length
                    emb[f"embedding_{key}"] = mean_pooling[0].float().cpu().numpy()
        embeddings.append(emb)
        if row_id % 100 == 0:
            print(row_id)
    output_dataset = input_dataset.add_column("embeddings", embeddings)
    return output_dataset


def generate_contexts(args, input_dataset):
    if args.with_embeddings:
        output_dir = os.path.join(args.output_dir, f"{args.model_type}", f"{args.data_subset}")
    else:
        output_dir = os.path.join(args.output_dir, f"{args.data_subset}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset_size = len(input_dataset)

    K = 1  # repeat samples for K times
    dataset_list = list()
    for idx in range(K):
        print(idx)
        context_dataset = deepcopy(input_dataset)
        context_lengths = np.random.randint(1, 5, size=dataset_size).tolist()
        context_dataset = context_dataset.add_column("context_length", context_lengths)
        contexts = list()
        for row_id in range(dataset_size):  # iterate over all samples in original dataset
            row_contexts = list()
            num_context = 0
            while num_context < context_lengths[row_id]:
                context_id = np.random.randint(dataset_size)  # sample a context from the original dataset
                if not args.synthetic_dataset:
                    if input_dataset[row_id]['prompt'] == input_dataset[context_id]['prompt']:
                        continue
                if args.add_controversial:
                    if not input_dataset[context_id]['controversial']:
                        continue
                if not args.with_embeddings:
                    row_contexts.append({
                        'original_id': context_id,
                        'chosen': input_dataset[context_id]['chosen'],
                        'rejected': input_dataset[context_id]['rejected'],
                    })
                else:
                    row_contexts.append({
                        'original_id': context_id,
                        'embedding_chosen': input_dataset[context_id]['embeddings']['embedding_chosen'],
                        'embedding_rejected': input_dataset[context_id]['embeddings']['embedding_rejected'],
                    })
                num_context += 1
            contexts.append(row_contexts)
        context_dataset = context_dataset.add_column("contexts", contexts)
        dataset_list.append(context_dataset)

    output_dataset = concatenate_datasets(dataset_list)
    output_dataset.to_json(os.path.join(output_dir, f"{args.data_split}.jsonl"))
    return output_dataset


if __name__ == "__main__":
    # default setting on HH-RLHF dataset, please iterate over data subsets and data splits
    np.random.seed(0)
    parser = HfArgumentParser(ScriptArguments)
    script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
    print(script_args)
    dataset = generate_embeddings_with_llm(script_args)
    generate_contexts(script_args, dataset)

# python -m hidden_context.data_utils.data_processing --output_dir data/relabeled_hh_rlhf_in_context_fixed/
# --data_path data/relabeled_hh_rlhf --data_subset helpful --data_split train --model_type gpt2
