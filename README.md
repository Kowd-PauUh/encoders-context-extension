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

After installing the package you may use `extend-context-spline` (recommended) or `extend-context-linear` scripts for embeddings interpolation.

### Spline Interpolation

Use this for smooth, nonlinear interpolation to support arbitrary context lengths:

```bash
extend-context-spline \
  --model_name_or_path="intfloat/e5-large-v2" \
  --max_seq_length=1024 \
  --embeddings_attr_name="embeddings.position_embeddings" \
  --offset=0 \
  --output_dir="intfloat/e5-large-v2-ctx1024-spline"
```

### Linear Interpolation

Use this to double the model's positional embedding range using linear averaging between consecutive embeddings:

```bash
extend-context-linear \
  --model_name_or_path="intfloat/e5-large-v2" \
  --embeddings_attr_name="embeddings.position_embeddings" \
  --offset=0 \
  --output_dir="intfloat/e5-large-v2-ctx1024-linear"
```

Both commands modify the positional embeddings of a model and save the updated model to the specified directory. You can then upload the resulting model to Hugging Face or use it locally for inference.

For models like RoBERTa that use special tokens in the first few positions, remember to set appropriate `--offset` argument.
