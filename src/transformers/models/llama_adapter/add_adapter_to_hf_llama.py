# Copyright 2022 EleutherAI and The HuggingFace Inc. team. All rights reserved.
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
import argparse
import gc
import json
import math
import os
import shutil
import warnings

import torch

from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer
from transformers import LlamaAdapterConfig, LlamaAdapterForCausalLM


try:
    from transformers import LlamaTokenizerFast
except ImportError as e:
    warnings.warn(e)
    warnings.warn(
        "The converted tokenizer will be the `slow` tokenizer. To use the fast, update your `tokenizers` library and re-run the tokenizer conversion"
    )
    LlamaTokenizerFast = None

"""
Sample usage:

```
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir /output/path
```

Thereafter, models can be loaded via:

```py
from transformers import LlamaForCausalLM, LlamaTokenizer

model = LlamaForCausalLM.from_pretrained("/output/path")
tokenizer = LlamaTokenizer.from_pretrained("/output/path")
```

Important note: you need to be able to host the whole model in RAM to execute this script (even if the biggest versions
come in several checkpoints they each contain a part of each weight of the model, so we need to load them all in RAM).
"""


def compute_intermediate_size(n):
    return int(math.ceil(n * 8 / 3) + 255) // 256 * 256


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)


def write_model(target_path, ori_path, tmp_path, adapter_path, model_size):
    shutil.copytree(ori_path, tmp_path)

    model_config = read_json(os.path.join(tmp_path, "config.json"))
    model_config['num_prefix_layers'] = 30 # todo hardcode
    model_config['num_prefix_tokens'] = 10
    model_config['architectures'] = ["LlamaAdapterForCausalLM"]
    write_json(model_config, os.path.join(tmp_path, "config.json"))

    print("Loading the checkpoint into a LlamaAdapter model.")
    model = LlamaAdapterForCausalLM.from_pretrained(tmp_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)

    print("Loading adapter params into the LlamaAdapter model")
    adapter_params = torch.load(adapter_path)
    prefix_weight = adapter_params['adapter_query.weight'].reshape(model.config.num_prefix_layers, model.config.num_prefix_tokens, model.config.hidden_size)
    state_dict = {}
    prefix_start_layer = model.config.num_hidden_layers - model.config.num_prefix_layers
    for layer_i in range(prefix_start_layer, model.config.num_hidden_layers):
        state_dict[f"model.layers.{layer_i}.self_attn.gate"] = adapter_params[f"layers.{layer_i}.attention.gate"]
        state_dict[f"model.layers.{layer_i}.self_attn.prefix"] = prefix_weight[layer_i-prefix_start_layer:layer_i-prefix_start_layer+1]
    model.load_state_dict(state_dict, strict=False)

    # Avoid saving this as part of the config.
    if hasattr(model.config, '_name_or_path'):
        del model.config._name_or_path


    print("Saving in the Transformers format.")
    model.save_pretrained(target_path)
    shutil.rmtree(tmp_path)

def write_tokenizer(tokenizer_path, input_tokenizer_path):
    # Initialize the tokenizer based on the `spm` model
    tokenizer_class = LlamaTokenizer if LlamaTokenizerFast is None else LlamaTokenizerFast
    print(f"Saving a {tokenizer_class.__name__} to {tokenizer_path}.")
    tokenizer = tokenizer_class(input_tokenizer_path)
    tokenizer.save_pretrained(tokenizer_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ori_path",
        help="Location of LLaMA weights, which contains tokenizer.model and model folders",
    )
    parser.add_argument(
        "--model_size",
        choices=["7B", "13B", "30B", "65B", "tokenizer_only"],
    )
    parser.add_argument(
        "--target_path",
        help="Location to write HF model and tokenizer",
    )
    parser.add_argument(
        "--tmp_path",
        help="Location to write HF model and tokenizer",
    )
    parser.add_argument(
        "--adapter_path",
        help="Location to write HF model and tokenizer",
    )
    args = parser.parse_args()
    if args.model_size != "tokenizer_only":
        write_model(
            args.target_path, args.ori_path, args.tmp_path, args.adapter_path,
            model_size=args.model_size,
        )
    spm_path = os.path.join(args.ori_path, "tokenizer.model")
    write_tokenizer(args.target_path, spm_path)


if __name__ == "__main__":
    main()
