"""
Script for model benchmarking. CLI is enabled.

Example:
>>> python3 src/benchmark_model.py --model_name_or_path="idanylenko/e5-large-v2-ctx1024"
"""

import torch
from sentence_transformers import SentenceTransformer
import mteb
from fire import Fire

from longbench_tasks import LONGBENCH_TASKS


TASK_LIST = [
    'LEMBSummScreenFDRetrieval',
    'LEMBQMSumRetrieval',
    'LEMBWikimQARetrieval',
    'LEMBNarrativeQARetrieval'
] + list(LONGBENCH_TASKS.keys())


def benchmark_model(
    model_name_or_path: str,
    tasks: list[str] = TASK_LIST,
    output_dir: str | None = None,
    model_kwargs: dict = {},
):
    # load model
    device = model_kwargs.pop('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    model = SentenceTransformer(model_name_or_path, device=device, **model_kwargs)

    # prepare tasks
    mteb_tasks = [t for t in tasks if t not in LONGBENCH_TASKS]
    if mteb_tasks:
        mteb_tasks = mteb.get_tasks(tasks=mteb_tasks)
    longbench_tasks = [LONGBENCH_TASKS[t] for t in tasks if t in LONGBENCH_TASKS]
    tasks = longbench_tasks + list(mteb_tasks)

    # run the evaluation
    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(model)

    return results


if __name__ == '__main__':
    def main(*args, **kwargs):
        benchmark_model(*args, **kwargs)

    Fire(main)
