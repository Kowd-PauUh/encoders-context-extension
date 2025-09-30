"""
Script for model benchmarking. CLI is enabled.

Example:
>>> python3 context_extension/benchmark_model.py --model_name_or_path="idanylenko/e5-large-v2-ctx1024"
"""

import argparse
import json
from pathlib import Path
import logging

import mteb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch.nn as nn

from context_extension.metrics import ndcg


logging.basicConfig(level=logging.INFO)

TASK_LIST = [
    'LEMBSummScreenFDRetrieval',
    'LEMBQMSumRetrieval',
    'LEMBWikimQARetrieval',
    'LEMBNarrativeQARetrieval'
]


def benchmark_model(
    model_name_or_path: str,
    tasks: list[str] = TASK_LIST,
    model_kwargs: dict | None = None,
    encode_kwargs: dict | None = None,
    query_prefix: str = '',
    document_prefix: str = '',
    k_values: list[int] | None = None,
    sim_fn: nn.Module | None = None,
    output_dir: str | Path | None = 'results',
):
    logger = logging.getLogger(__name__)

    model_kwargs = model_kwargs or {}
    encode_kwargs = (encode_kwargs or {'batch_size': 32}) | {
        'convert_to_tensor': True,
        'show_progress_bar': True,
    }
    k_values = k_values or [10]
    sim_fn = sim_fn or nn.CosineSimilarity(dim=1)

    # load model
    model = SentenceTransformer(model_name_or_path, **model_kwargs)
    model.model_card_data.model_name = model_name_or_path

    results = []
    tasks = mteb.get_tasks(tasks=tasks)
    for task in tasks:
        task_name = task.metadata.name
        logger.info(f'Evaluating "{model_name_or_path}" on "{task_name}".')

        # load task data
        task.load_data()
        split = task.metadata.eval_splits[0]

        logger.info('Encoding corpus and queries...')

        # encode corpus
        corpus_ids = list(task.corpus[split].keys())
        corpus_texts = [
            document_prefix + (document['text'] if isinstance(document, dict) else document)
            for document in task.corpus[split].values()
        ]
        corpus_representations = model.encode(corpus_texts, **encode_kwargs).detach().cpu()

        # encode queries
        queries_ids = list(task.queries[split].keys())
        queries_texts = [
            query_prefix + (query['text'] if isinstance(query, dict) else query)
            for query in task.queries[split].values()
        ]
        queries_representations = model.encode(queries_texts, **encode_kwargs).detach().cpu()

        # evaluate each query
        result = []
        pbar = tqdm(
            zip(queries_ids, queries_representations),
            total=len(task.queries[split]),
            desc='Evaluating queries',
        )
        for query_id, query_representation in pbar:
            # compute similarity scores
            scores_list = sim_fn(
                query_representation.unsqueeze(0),  # shape (1, d)
                corpus_representations
            ).tolist()

            # sort corpus ids based on similarity score
            ranked_ids = sorted(
                zip(corpus_ids, scores_list),
                key=lambda x: x[1],
                reverse=True
            )

            # compute metrics
            top_hits = [document_id for document_id, _ in ranked_ids]
            qrels = list(task.relevant_docs[split][query_id].keys())
            ndcg_at_k = ndcg(top_hits, qrels, k_values=k_values)

            # format row
            ndcg_at_k = {f'ndcg_at_{k}': v for k, v in ndcg_at_k.items()}
            row = {'task': task_name, 'query_id': query_id} | ndcg_at_k

            result.append(row)

        results += result

        # save evaluation result
        if output_dir is not None:
            artifact_path = Path(output_dir) / model_name_or_path / f'{task_name}.jsonl'
            artifact_path.parent.mkdir(parents=True, exist_ok=True)

            with open(artifact_path, 'w') as f:
                for row in result:
                    json_row = json.dumps(row, ensure_ascii=False)
                    f.write(json_row + '\n')

            logger.info(f'"{task_name}" evaluation results were saved to "{artifact_path}".')

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark a SentenceTransformer model on MTEB tasks.")
    parser.add_argument('--model_name_or_path', type=str, required=True, help='Model path or HuggingFace model name.')
    parser.add_argument('--tasks', type=str, nargs='*', default=TASK_LIST, help='List of task names to evaluate on.')
    parser.add_argument('--model_kwargs', type=str, default='{}', help='SentenceTransformer.__init__ kwargs as JSON string.')
    parser.add_argument('--encode_kwargs', type=str, default='{}', help='SentenceTransformer.encode kwargs as JSON string.')
    parser.add_argument('--k_values', type=int, nargs='*', default=[10], help='List of cutoff values for metrics.')
    parser.add_argument('--query_prefix', type=str, default='', help='Query prefix.')
    parser.add_argument('--document_prefix', type=str, default='', help='Document prefix.')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory for saving evaluation results.')
    args = parser.parse_args()

    model_kwargs = json.loads(args.model_kwargs)
    encode_kwargs = json.loads(args.encode_kwargs)

    benchmark_model(
        model_name_or_path=args.model_name_or_path,
        tasks=args.tasks,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        query_prefix=args.query_prefix,
        document_prefix=args.document_prefix,
        k_values=args.k_values,
        output_dir=args.output_dir,
    )
