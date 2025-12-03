[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17807059.svg)](https://doi.org/10.5281/zenodo.17807059)

# Zero-Training Context Extension for Transformer Encoders via Nonlinear Absolute Positional Embeddings Interpolation

Official implementation of "Zero-Training Context Extension for Transformer Encoders via Nonlinear Absolute Positional Embeddings Interpolation". Paper preprint is coming soon.

This implementation currently supports only models compatible with [Sentence Transformers](https://www.sbert.net/) library.

## Models

Models are available at HuggingFace:

| Model                                                                                   | Context length | Language |
| --------------------------------------------------------------------------------------- | -------------- | -------- |
| [idanylenko/e5-large-v2-ctx1024](https://huggingface.co/idanylenko/e5-large-v2-ctx1024) | 1024           | English  |

## Installation

To install the package, use pip:

```bash
pip install context-extension
```

## Usage

After installing the package you may use `extend-context` scripts for embeddings interpolation. The script modifies the positional embeddings of a model and save the updated model to the specified directory. You can then upload the resulting model to Hugging Face or use it locally for inference.

Recommended option is to set `--interpolation_type=cubic` as this provides smooth interpolation in contrast to linear interpolation. For models like RoBERTa that use special tokens in the first few positions, remember to set appropriate `--offset` argument. Too big `--max_seq_length` argument values may result in performance degradation.

Use `extend-context --help` to see all available options and parameters.

### Spline Interpolation

Use this for smooth, nonlinear interpolation:

```bash
extend-context \
  --model_name_or_path="intfloat/e5-large-v2" \
  --max_seq_length=1024 \
  --embeddings_attr_name="embeddings.position_embeddings" \
  --offset=0 \
  --interpolation_type=cubic \
  --output_dir="intfloat/e5-large-v2-ctx1024-spline"
```

### Linear Interpolation

Use this for linear interpolation:

```bash
extend-context \
  --model_name_or_path="intfloat/e5-large-v2" \
  --max_seq_length=1024 \
  --embeddings_attr_name="embeddings.position_embeddings" \
  --offset=0 \
  --interpolation_type=linear \
  --output_dir="intfloat/e5-large-v2-ctx1024-linear"
```

## Citation

```bibtex
@software{danylenko2025zero,
author = {Danylenko, Ivan},
doi = {10.5281/zenodo.17807059},
month = may,
title = {{Zero-Training Context Extension for Transformer Encoders via Nonlinear Absolute Positional Embeddings Interpolation}},
url = {https://github.com/Kowd-PauUh/encoders-context-extension},
version = {0.1.3},
year = {2025}
}
```
