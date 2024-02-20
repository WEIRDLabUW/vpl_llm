# This file is used to preprocess dataset, now available for relabeled HH-RLHF
import os
from dataclasses import dataclass, field
from typing import Optional, cast

from transformers import (
    HfArgumentParser,
    AutoTokenizer,
    AutoModelForSequenceClassification,
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


def generate_embeddings_with_llm(args):
    data_subset = cast(DataSubset, args.data_subset)
    input_dataset = get_hh_rlhf_dataset(
        data_subset,
        args.data_split,
        args.dataset_size,
        data_path=args.data_path,
        use_subset_as_dir=True
    )

    if args.model_type == "gpt2":
        tokenizer = AutoTokenizer.from_pretrained("gpt2", use_auth_token=True)
        model = AutoModelForSequenceClassification.from_pretrained(  # todo: why using SequenceClassification?
            "gpt2", num_labels=1024, torch_dtype=torch.bfloat16
        )
    elif args.model_type == "llama":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_auth_token=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            "meta-llama/Llama-2-7b-hf", num_labels=1024, torch_dtype=torch.bfloat16
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
        lambda example, idx: len(preprocessed_dataset[idx]["input_ids_chosen"]) <= 1024
                             and len(preprocessed_dataset[idx]["input_ids_rejected"]) <= 1024,
        with_indices=True
    )
    preprocessed_dataset = preprocessed_dataset.filter(
        lambda example: len(example["input_ids_chosen"]) <= 1024 and len(example["input_ids_rejected"]) <= 1024
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
                emb[f"embedding_{key}"] = model(
                    input_ids=tokens["input_ids"].unsqueeze(0).to("cuda"),
                    attention_mask=tokens["attention_mask"].unsqueeze(0).to("cuda")
                )[0][0].float().cpu().numpy()
        embeddings.append(emb)
        if row_id % 100 == 0:
            print(row_id)
    output_dataset = input_dataset.add_column("embeddings", embeddings)
    return output_dataset


def generate_context_data_from_original_data(args, input_dataset):
    output_dir = os.path.join(args.output_dir, f"{args.data_subset}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_dir = os.path.join(args.output_dir, f"{args.data_subset}")

    dataset_size = len(input_dataset)

    K = 5  # repeat samples for K times
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
                if input_dataset[row_id]['prompt'] == input_dataset[context_id]['prompt']:
                    continue
                if args.model_type == "none":
                    row_contexts.append({
                        'original_id': context_id,
                        'chosen': input_dataset[context_id]['chosen'],
                        'rejected': input_dataset[context_id]['rejected'],
                    })
                else:
                    row_contexts.append({
                        'original_id': context_id,
                        'chosen': input_dataset[context_id]['chosen'],
                        'rejected': input_dataset[context_id]['rejected'],
                        'embedding_chosen': input_dataset[context_id]['embeddings']['embedding_chosen'],
                        'embedding_rejected': input_dataset[context_id]['embeddings']['embedding_rejected'],
                    })
                num_context += 1
            contexts.append(row_contexts)
        context_dataset = context_dataset.add_column("contexts", contexts)
        dataset_list.append(context_dataset)

    output_dataset = concatenate_datasets(dataset_list)
    output_dataset.to_json(os.path.join(output_dir, f"{args.model_type}_{args.data_split}.jsonl"))
    return output_dataset


if __name__ == "__main__":
    np.random.seed(0)
    parser = HfArgumentParser(ScriptArguments)
    script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
    print(script_args)
    dataset = generate_embeddings_with_llm(script_args)
    generate_context_data_from_original_data(script_args, dataset)
