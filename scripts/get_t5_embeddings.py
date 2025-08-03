# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import pickle

import numpy as np

from cosmos_predict2.auxiliary.text_encoder import CosmosT5TextEncoder

"""example command
python -m scripts.get_t5_embeddings --dataset_path datasets/hdvila
"""


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute T5 embeddings for text prompts")
    parser.add_argument("--dataset_path", type=str, default="datasets/hdvila", help="Root path to the dataset")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum length of the text embedding")
    parser.add_argument(
        "--cache_dir", type=str, default="checkpoints/google-t5/t5-11b", help="Directory to cache the T5 model"
    )
    return parser.parse_args()


def main(args) -> None:
    metas_dir = os.path.join(args.dataset_path, "metas")
    metas_list = [
        os.path.join(metas_dir, filename) for filename in sorted(os.listdir(metas_dir)) if filename.endswith(".txt")
    ]

    t5_xxl_dir = os.path.join(args.dataset_path, "t5_xxl")
    os.makedirs(t5_xxl_dir, exist_ok=True)

    # Initialize T5
    print(f"Cache dir: {args.cache_dir}")
    encoder = CosmosT5TextEncoder(cache_dir=args.cache_dir, local_files_only=True)

    for meta_filename in metas_list:
        t5_xxl_filename = os.path.join(t5_xxl_dir, os.path.basename(meta_filename).replace(".txt", ".pickle"))
        if os.path.exists(t5_xxl_filename):
            # Skip if the file already exists
            continue

        with open(meta_filename, "r") as fp:
            prompt = fp.read().strip()

        # Compute T5 embeddings
        max_length = args.max_length
        encoded_text, mask_bool = encoder.encode_prompts(
            prompt, max_length=max_length, return_mask=True
        )  # list of np.ndarray in (len, 1024)
        attn_mask = mask_bool.long()
        lengths = attn_mask.sum(dim=1).cpu()

        encoded_text = encoded_text.cpu().numpy().astype(np.float16)

        # trim zeros to save space
        encoded_text = [encoded_text[batch_id][: lengths[batch_id]] for batch_id in range(encoded_text.shape[0])]

        # Save T5 embeddings as pickle file
        with open(t5_xxl_filename, "wb") as fp:
            pickle.dump(encoded_text, fp)


if __name__ == "__main__":
    args = parse_args()
    main(args)
