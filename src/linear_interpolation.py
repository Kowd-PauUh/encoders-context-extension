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
>>> python3 src/linear_interpolation.py --model_name_or_path="intfloat/e5-large-v2" --offset=0 --output_dir="idanylenko/e5-large-v2-ctx1024"
>>> python3 src/linear_interpolation.py --model_name_or_path="FacebookAI/roberta-base" --offset=2 --output_dir="idanylenko/roberta-base-ctx1024"
"""

from tempfile import TemporaryDorectory
import argparse

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer


def interpolate_embeddings(
    model_name_or_path: str,
    embeddings_attr_name: str = 'embeddings.position_embeddings',
    offset: int = 0,
    output_dir: str | None = None,
    model_kwargs: dict = {},
) -> SentenceTransformer:
    """
    Stretches model positional embeddings starting from a given offset.
    If the number of positional embeddings in original model is equal
    `max_position_embeddings`, then the number of new embeddings is 
    expressed as:
            offset + (max_position_embeddings - offset) * 2
    Model maximum sequence length is then defined as:
                (max_position_embeddings - offset) * 2

    Parameters
    ----------
    model_name_or_path : str
        Path to the sentence transformer or the HF model name to which
        the function has to be applied.
    embeddings_attr_name : str, optional
        Path to the transformer model attribute with positional embeddings
        weights. Default is "embeddings.position_embeddings".
    offset : str, optional
        Number of first positional embeddings that will remain unaffected.
        Some of the models, such as RoBERTa has additional embeddings
        at the beginning, which are not used for embedding positions of 
        actual tokens, so in the case of RoBERTa we would set this to 2.
        Default is 0.
    output_dir : str | None, optional
        Output directory where the modified model has to be saved. If set
        to None, model will not be saved. Default is None.
    model_kwargs : dict, optional
        Kwargs to be used when loading model as SentenceTransformer object. 
    """
    # load model
    device = model_kwargs.pop('device', 'cpu')
    sentence_transformer = SentenceTransformer(model_name_or_path, device=device, **model_kwargs)
    model = sentence_transformer._first_module().auto_model

    # get positional embeddings weight
    embeddings = model
    for attr_name in embeddings_attr_name.split('.'):
        embeddings = getattr(embeddings, attr_name)
    weight = embeddings.weight.clone().detach()

    # approximate new positional embeddings as means between
    # two consecutive embeddings of the original model
    means = (weight[offset:-1] + weight[offset+1:]) / 2
    stretched_weight = torch.empty(
        (weight.shape[0] - offset + means.shape[0], weight.shape[1]),
        dtype=weight.dtype,
        device=weight.device
    )
    stretched_weight[0::2] = weight[offset:]
    stretched_weight[1::2] = means
    weight = torch.vstack(
        [
            weight[:offset],  # preserve embeddings within offset
            stretched_weight,
            weight[-1:]       # add last embedding as a copy of one before last
        ]
    )

    # initialize new embeddings
    embeddings = nn.Embedding(
        weight.shape[0],
        weight.shape[1],
        padding_idx=embeddings.padding_idx
    )
    embeddings.weight.data = weight

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
            (model.config.type_vocab_size, weight.shape[0]),
            dtype=torch.long
        )
    )
    model.config.max_position_embeddings = weight.shape[0]
    sentence_transformer._first_module().tokenizer.model_max_length = weight.shape[0] - offset
    sentence_transformer.max_seq_length = weight.shape[0] - offset

    # save the model if necessary
    if output_dir is not None:
        sentence_transformer.save(output_dir)

    # save to temporary directory and return newly loaded model
    with TemporaryDorectory() as temp_dir:
        sentence_transformer.save(temp_dir)
        sentence_transformer = SentenceTransformer(output_dir, device=device, **model_kwargs)

    return sentence_transformer


def main():
    parser = argparse.ArgumentParser(description='Linear positional embeddings interpolation.')
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--embeddings_attr_name', type=str, default='embeddings.position_embeddings')
    parser.add_argument('--offset', type=int, default=0)
    args = parser.parse_args()

    interpolate_embeddings(
        model_name_or_path=args.model_name_or_path,
        embeddings_attr_name=args.embeddings_attr_name,
        offset=args.offset,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
