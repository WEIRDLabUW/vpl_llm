# This file is used to preprocess dataset, now available for relabeled HH-RLHF
import os
from dataclasses import dataclass, field
from typing import Optional, cast

from transformers import (
    HfArgumentParser,
)

from hidden_context.train_llm_preference_model import (
    DataSubset,
    get_hh_rlhf_dataset,
    concatenate_datasets
)
from copy import deepcopy

import numpy as np


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


def generate_context_data_from_original_data(script_args):
    data_subset = cast(DataSubset, script_args.data_subset)
    dataset = get_hh_rlhf_dataset(
        data_subset,
        script_args.data_split,
        script_args.dataset_size,
        data_path=script_args.data_path,
        use_subset_as_dir=True
    )
    output_dir = os.path.join(script_args.output_dir, f"{script_args.data_subset}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    dataset_size = len(dataset)

    K = 5   # repeat samples for K times
    dataset_list = list()
    for idx in range(K):
        print(idx)
        context_dataset = deepcopy(dataset)
        context_lengths = np.random.randint(1, 5, size=dataset_size).tolist()
        context_dataset = context_dataset.add_column("context_length", context_lengths)
        contexts = list()
        for row_id in range(dataset_size):  # iterate over all samples in original dataset
            row_contexts = list()
            num_context = 0
            while num_context < context_lengths[row_id]:
                context_id = np.random.randint(dataset_size)    # sample a context from the original dataset
                if dataset[row_id]['prompt'] == dataset[context_id]['prompt']:
                    continue
                row_contexts.append({
                    'original_id': context_id,
                    'chosen': dataset[context_id]['chosen'],
                    'rejected': dataset[context_id]['rejected'],
                })
                num_context += 1
            contexts.append(row_contexts)
        context_dataset = context_dataset.add_column("contexts", contexts)
        dataset_list.append(context_dataset)

    output_dataset = concatenate_datasets(dataset_list)
    output_dataset.to_json(os.path.join(output_dir, f"{script_args.data_split}.jsonl"))


if __name__ == "__main__":
    np.random.seed(0)
    parser = HfArgumentParser(ScriptArguments)
    script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
    print(script_args)
    generate_context_data_from_original_data(script_args)
