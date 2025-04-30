import torch
from sentence_transformers import SentenceTransformer
import mteb
from fire import Fire


TASK_LIST = [
    'LEMBSummScreenFDRetrieval',
    'LEMBQMSumRetrieval',
    'LEMBWikimQARetrieval',
    'LEMBNarrativeQARetrieval'
]


def benchmark_model(
    model_name_or_path: str,
    tasks: list[str] = TASK_LIST,
    output_dir: str | None = None,
    model_kwargs: dict = {},
):
    # load model
    device = model_kwargs.pop('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    model = SentenceTransformer(model_name_or_path, device=device, **model_kwargs)

    # run the evaluation
    tasks = mteb.get_tasks(tasks=tasks)
    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(model)

    return results


if __name__ == '__main__':
    def main(*args, **kwargs):
        benchmark_model(*args, **kwargs)

    Fire(main)
