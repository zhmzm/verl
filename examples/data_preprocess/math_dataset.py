# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the MATH-lighteval and MATH-500 datasets to parquet format
"""

import argparse
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed

def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/math")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()
    local_dir = os.path.expanduser(args.local_dir)
    hdfs_dir = args.hdfs_dir

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # Load train dataset from MATH-lighteval
    train_data_source = "DigitalLearningGmbH/MATH-lighteval"
    print(f"Loading the train dataset from {train_data_source}...", flush=True)
    train_dataset = datasets.load_dataset(train_data_source, split="train", trust_remote_code=True)

    # Load test dataset from MATH-500
    test_data_source = "HuggingFaceH4/MATH-500"
    print(f"Loading the test dataset from {test_data_source}...", flush=True)
    test_dataset = datasets.load_dataset(test_data_source, split="test", trust_remote_code=True)

    # Define preprocessing function
    def make_map_fn(split, is_math500=False):
        def process_fn(example, idx):
            question = example.pop("problem")
            question += " " + instruction_following

            answer = example.pop("answer") if is_math500 else example.pop("solution")
            solution = answer if is_math500 else extract_solution(answer) 

            return {
                "data_source": test_data_source if is_math500 else train_data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {"split": split, "index": idx},
            }

        return process_fn

    # Map and preprocess
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test", is_math500=True), with_indices=True)

    # Save to local parquet
    os.makedirs(local_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    # Optionally save to HDFS
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
