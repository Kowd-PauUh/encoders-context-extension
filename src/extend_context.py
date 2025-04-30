import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from fire import Fire


def extend_context(
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

    return sentence_transformer


if __name__ == '__main__':
    Fire(extend_context)
