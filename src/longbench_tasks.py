import mteb
from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata
from datasets import load_dataset


LONGBENCH_DATASETS = [
    "hotpotqa", "2wikimqa", "musique", 
    "dureader", "narrativeqa", "qasper",
    "multifieldqa_en", "multifieldqa_zh", 
    "gov_report", "qmsum", "vcsum", "trec", 
    "nq", "triviaqa", "lsht", "passage_count",
    "passage_retrieval_en", "passage_retrieval_zh", 
    "lcc",
]


class BaseLongBenchTask(AbsTaskRetrieval):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data_loaded = False
        
    @property
    def metadata(self):
        print(f'Asked for {self.__class__.__name__} metadata')
        return TaskMetadata(
            name=self.__class__.__name__,
            description='',
            reference='http://example.com',
            type='Retrieval',
            category='s2p',
            modalities=['text'],
            eval_splits=['test'],
            eval_langs=['eng-ENG'],
            main_score='ndcg_at_10',
            dataset={
                "path": "",
                "revision": "",
            }
        )

    def load_data(self):
        data = load_dataset(
            'THUDM/LongBench',
            self.__class__.__name__,
            split='test'
        )

        queries = data['input']
        passages = data['context']

        self.queries = {'test': {f'qid{i}': q for i, q in enumerate(queries)}}
        self.corpus = {'test': {f'pid{i}': {'text': p} for i, p in enumerate(passages)}}
        self.relevant_docs = {'test': {f'qid{i}': {f'pid{i}': 1} for i in range(len(queries))}}
        self.data_loaded = True


# create mteb retrieval tasks from base task
TASK_CLASSES = {}
for task_name in LONGBENCH_DATASETS:
    TASK_CLASSES[task_name] = type(
        task_name,
        (BaseLongBenchTask,),
        {
            '__init__': (lambda task_name=task_name: 
                         lambda self, *args, **kwargs: 
                         super(TASK_CLASSES[task_name], self).__init__(*args, **kwargs))()
        }
    )
LONGBENCH_TASKS = {k: v() for k, v in TASK_CLASSES.items()}
