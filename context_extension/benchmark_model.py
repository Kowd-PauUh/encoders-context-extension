"""
Script for model benchmarking. CLI is enabled.

Example:
>>> python3 context_extension/benchmark_model.py --model_name_or_path="idanylenko/e5-large-v2-ctx1024"
"""

import argparse
import json

import torch
from sentence_transformers import SentenceTransformer
import mteb


TASK_LIST = [
    'LEMBSummScreenFDRetrieval',
    'LEMBQMSumRetrieval',
    'LEMBWikimQARetrieval',
    'LEMBNarrativeQARetrieval'
]


def benchmark_model(
    model_name_or_path: str,
    tasks: list[str] = TASK_LIST,
    model_kwargs: dict = {},
):
    # load model
    device = model_kwargs.pop('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    model = SentenceTransformer(model_name_or_path, device=device, **model_kwargs)
    model.model_card_data.model_name = model_name_or_path

    # run the evaluation
    tasks = mteb.get_tasks(tasks=tasks)
    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(model, batch_size=8)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark a SentenceTransformer model on MTEB tasks.")
    parser.add_argument('--model_name_or_path', type=str, required=True, help='Model path or HuggingFace model name.')
    parser.add_argument('--tasks', type=str, nargs='*', default=TASK_LIST, help='List of task names to evaluate on.')
    parser.add_argument('--model_kwargs', type=str, default="{}", help='Additional keyword arguments as JSON string.')
    args = parser.parse_args()

    model_kwargs = json.loads(args.model_kwargs)
    benchmark_model(args.model_name_or_path, args.tasks, model_kwargs)
