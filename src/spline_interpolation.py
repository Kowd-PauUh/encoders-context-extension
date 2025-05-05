# Copyright 2025 Ivan Danylenko
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Script for extending encoder context. CLI is enabled.

Example:
>>> python3 src/spline_interpolation.py --model_name_or_path="intfloat/e5-large-v2" --max_seq_length 1024 --offset=0 --output_dir="idanylenko/e5-large-v2-ctx1024"
>>> python3 src/spline_interpolation.py --model_name_or_path="FacebookAI/roberta-base" --max_seq_length 1024 --offset=2 --output_dir="idanylenko/roberta-base-ctx1024"
"""

from tempfile import TemporaryDorectory

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from scipy.interpolate import interp1d
import numpy as np
from fire import Fire


def interpolate_embeddings(
    model_name_or_path: str,
    max_seq_length: int,
    embeddings_attr_name: str = 'embeddings.position_embeddings',
    offset: int = 0,
    output_dir: str | None = None,
    model_kwargs: dict = {},
) -> SentenceTransformer:
    # load model
    device = model_kwargs.pop('device', 'cpu')
    sentence_transformer = SentenceTransformer(model_name_or_path, device=device, **model_kwargs)
    model = sentence_transformer._first_module().auto_model

    # get positional embeddings weight
    embeddings = model
    for attr_name in embeddings_attr_name.split('.'):
        embeddings = getattr(embeddings, attr_name)
    weight = np.array(embeddings.weight.clone().detach().tolist())

    # interpolate embeddings using spline
    nembs = weight.shape[0]
    ndims = weight.shape[1]
    stretched_weight = np.empty((max_seq_length + offset, ndims))
    stretched_weight[:offset] = weight[:offset]
    for dim in range(ndims):
        y = weight[offset:, dim]
        spline = interp1d(np.arange(len(y)), y, kind='cubic')

        x = np.linspace(0, len(y) - 1, max_seq_length)
        stretched_weight[offset:, dim] = spline(x)

    # initialize new embeddings
    embeddings = nn.Embedding(
        max_seq_length + offset, ndims,
        padding_idx=embeddings.padding_idx
    )
    embeddings.weight.data = torch.Tensor(stretched_weight)

    # set new embeddings
    *holder_attrs, target_attr_name = embeddings_attr_name.split('.')
    embeddings_holder = model
    for attr_name in holder_attrs:
        # get to the object that holds embeddings
        embeddings_holder = getattr(embeddings_holder, attr_name)
    setattr(embeddings_holder, target_attr_name, embeddings)

    # update model with regard to new embeddings
    embeddings_holder.register_buffer(
        'token_type_ids',
        torch.zeros(
            (model.config.type_vocab_size, max_seq_length + offset),
            dtype=torch.long
        )
    )
    model.config.max_position_embeddings = max_seq_length + offset
    sentence_transformer._first_module().tokenizer.model_max_length = max_seq_length
    sentence_transformer.max_seq_length = max_seq_length

    # save the model if necessary
    if output_dir is not None:
        sentence_transformer.save(output_dir)
        return SentenceTransformer(output_dir, device=device, **model_kwargs)

    # save to temporary directory and return newly loaded model
    with TemporaryDorectory() as temp_dir:
        sentence_transformer.save(temp_dir)
        sentence_transformer = SentenceTransformer(output_dir, device=device, **model_kwargs)

    return sentence_transformer


if __name__ == '__main__':
    def main(*args, **kwargs):
        extend_context(*args, **kwargs)

    Fire(main)
