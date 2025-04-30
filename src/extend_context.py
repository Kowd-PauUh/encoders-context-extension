import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer


def extend_context(
    model_name_or_path: str,
    embeddings_attr_name: str = 'embeddings.position_embeddings',
    offset: int,
    output_dir: str | None = None,
    model_kwargs: dict = {},
) -> SentenceTransformer:
    """
    Stretches model position embeddings starting from a given offset.
    The number of the new position embeddings is expressed as:
            offset + (max_position_embeddings - offset) * 2

    Parameters
    ----------
    model_name_or_path : str
        Path to the sentence transformer or the HF model name to which
        the function has to be applied.
    embeddings_attr_name : str
        Path to the model attribute with positional embeddings weights.
    offset : str
        Number of first positional embeddings that will remain unaffected.
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

    affected_embeddings_num = weight.shape[0] - offset

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
