import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from medclip import constants
from medclip.dataset import ZeroShotImageCollator
from medclip.dataset import ZeroShotImageDataset
from medclip.evaluator import Evaluator
from medclip.modeling_medclip import MedCLIPModel, PromptClassifier, MedCLIPVisionModel, MedCLIPVisionModelViT
from medclip.prompts import generate_class_prompts, generate_chexpert_class_prompts, generate_covid_class_prompts, \
    generate_rsna_class_prompts

class Config:
    def __init__(self):
        self.seed = 42
        self.setup_environment()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def setup_environment(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        os.environ['PYTHONASHSEED'] = str(self.seed)
        os.environ['TOKENIZERS_PARALLELISM'] = 'False'
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class MedCLIPModelBuilder:
    def __init__(self, use_vit=False):
        self.use_vit = use_vit

    def build_model(self, checkpoint=""):
        if self.use_vit:
            vision_cls = MedCLIPVisionModelViT
            if not checkpoint:
                checkpoint = './ckpt/pytorch_model.bin'
        else:
            vision_cls = MedCLIPVisionModel
            if not checkpoint:
                checkpoint = './ckpt/pytorch_model.bin'

        model = MedCLIPModel(vision_cls=vision_cls, checkpoint=checkpoint)
        model.cuda()
        return model

class DatasetManager:
    def __init__(
            self,
            dataname,
            backdoor="patch",
            trigger_size=(32,32),
            color=(0,0,0),
            position="right_bottom"
            ):
        self.dataname = dataname
        self.backdoor = backdoor
        self.tasks = self.get_tasks()
        self.dataset = ZeroShotImageDataset([dataname],
                                            class_names=self.tasks,
                                            backdoor=backdoor,
                                            trigger_size=trigger_size,
                                            color=color,
                                            position=position
                                            )

    def get_tasks(self):
        if self.dataname in ['chexpert-5x200', 'mimic-5x200']:
            return constants.CHEXPERT_COMPETITION_TASKS
        elif self.dataname in ['covid-test', 'covid-2x200-test']:
            return constants.COVID_TASKS
        elif self.dataname in ['rsna-balanced-test', 'rsna-2x200-test', 'rsna-test']:
            return constants.RSNA_TASKS
        else:
            raise NotImplementedError

class EvaluatorManager:
    def __init__(self, model, dataset, tasks, dataname, ensemble):
        self.model = model
        self.dataset = dataset
        self.tasks = tasks
        self.dataname = dataname
        self.ensemble = ensemble
        self.sent_df = pd.read_csv('./local_data/sentence-label.csv', index_col=0)

    def create_evaluator(self):
        if self.dataname in ['chexpert-5x200', 'mimic-5x200']:
            cls_prompts = generate_chexpert_class_prompts(n=10)
            mode = 'multiclass'
        elif self.dataname in ['covid-test', 'covid-2x200-test']:
            cls_prompts = generate_class_prompts(self.sent_df, ['No Finding'], n=10)
            covid_prompts = generate_covid_class_prompts(n=10)
            cls_prompts.update(covid_prompts)
            mode = 'binary'
        elif self.dataname in ['rsna-balanced-test', 'rsna-2x200-test', 'rsna-test']:
            cls_prompts = generate_class_prompts(self.sent_df, ['No Finding'], n=10)
            rsna_prompts = generate_rsna_class_prompts(n=10)
            cls_prompts.update(rsna_prompts)
            mode = 'binary'
        else:
            raise NotImplementedError

        val_collate_fn = ZeroShotImageCollator(mode=mode, cls_prompts=cls_prompts)
        eval_dataloader = DataLoader(self.dataset, batch_size=128, collate_fn=val_collate_fn, 
                                     shuffle=False, pin_memory=True, num_workers=0)
        medclip_clf = PromptClassifier(self.model, ensemble=self.ensemble)
        return Evaluator(medclip_clf=medclip_clf, eval_dataloader=eval_dataloader, mode=mode)

class MainEvaluator:
    def __init__(self,
                n_runs=5,
                ensemble=True,
                use_vit=False,
                backdoor="patch",
                checkpoint="",
                trigger_size=(32,32),
                color=(0,0,0),
                position="right_bottom",
                ):
        self.config = Config()
        self.model_builder = MedCLIPModelBuilder(use_vit)
        self.model = self.model_builder.build_model(checkpoint=checkpoint)
        self.n_runs = n_runs
        self.ensemble = ensemble
        self.backdoor = backdoor
        self.trigger_size = trigger_size
        self.color = color
        self.position = position

    def run_evaluation(self, dataname):
        dataset_manager = DatasetManager(dataname,
                                        backdoor=self.backdoor,
                                        trigger_size=self.trigger_size,
                                        color=self.color,
                                        position=self.position
                                        )
        evaluator_manager = EvaluatorManager(self.model, dataset_manager.dataset, 
                                             dataset_manager.tasks, dataname, self.ensemble)

        metric_list = defaultdict(list)
        for _ in range(self.n_runs):
            evaluator = evaluator_manager.create_evaluator()
            res = evaluator.evaluate()
            for key in res.keys():
                if key not in ['pred', 'labels']:
                    print(f'{key}: {res[key]}')
                    metric_list[key].append(res[key])

        for key, value in metric_list.items():
            print(f'{key} mean: {np.mean(value):.4f}, std: {np.std(value):.2f}')

if __name__ == "__main__":
    evaluation1 = MainEvaluator(use_vit=True,
                                backdoor="none",
                                trigger_size=(32,32),
                                color=(0,0,0),
                                position="right_bottom",
                                checkpoint="ckpt/pytorch_model.bin",
                                )
    evaluation1.run_evaluation('covid-test')
    evaluation2 = MainEvaluator(use_vit=True,
                                backdoor="patch",
                                trigger_size=(32,32),
                                color=(0,0,0),
                                position="right_bottom",
                                checkpoint="ckpt/pytorch_model.bin",   
                                )
    evaluation2.run_evaluation('covid-test')
