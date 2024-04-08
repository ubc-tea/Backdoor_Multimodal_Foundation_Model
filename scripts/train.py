import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from medclip.modeling_medclip import MedCLIPModel, PromptClassifier, MedCLIPVisionModel, MedCLIPVisionModelViT
from medclip.dataset import ImageTextContrastiveDataset, ZeroShotImageDataset
from medclip.dataset import ImageTextContrastiveCollator, ZeroShotImageCollator
from medclip.losses import ImageTextContrastiveLoss
from medclip.trainer import Trainer
from medclip.evaluator import Evaluator
from medclip import constants
from medclip.prompts import generate_chexpert_class_prompts

class AppConfig:
    def __init__(self, bad_dist=False, stop_step=2000):
        self.seed = 42
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.train_config = {
            'batch_size': 32,
            'num_epochs': 10,
            'warmup': 0.1,
            'lr': 2e-5,
            'weight_decay': 1e-4,
            'eval_batch_size': 256,
            'eval_steps': 500,
            'save_steps': 500,
        }
        self.datalist = ['mimic-cxr-train']
        self.init_seed()
        self.set_cuda_devices()
        self.bad_dist = bad_dist
        self.stop_step = stop_step

    def init_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        os.environ['PYTHONASHSEED'] = str(self.seed)
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    def set_cuda_devices(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = '2'

class DataPreparation:
    def __init__(self, datalist, train_config):
        self.datalist = datalist
        self.train_config = train_config
        self.train_loader = None
        self.eval_dataloader = None
        self.setup()

    def setup(self):
        transform = transforms.Compose([
            transforms.Resize((constants.IMG_SIZE, constants.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[constants.IMG_MEAN], std=[constants.IMG_STD]),
        ])

        train_data = ImageTextContrastiveDataset(datalist=self.datalist,
                                                imgtransform=transform,
                                                backdoor="patch",
                                                trigger_size = (32, 32),
                                                color = (245, 245, 245),
                                                position = "mid_bottom"
                                                )
        train_collate_fn = ImageTextContrastiveCollator()
        self.train_loader = DataLoader(train_data,
                                       batch_size=self.train_config['batch_size'],
                                       collate_fn=train_collate_fn,
                                       shuffle=True,
                                       pin_memory=True,
                                       num_workers=2)

        cls_prompts = generate_chexpert_class_prompts(n=10)
        val_data = ZeroShotImageDataset(['covid-test'],
                                        class_names=constants.COVID_TASKS)
        val_collate_fn = ZeroShotImageCollator(cls_prompts=cls_prompts,
                                               mode='multiclass')
        self.eval_dataloader = DataLoader(val_data,
                                          batch_size=self.train_config['eval_batch_size'],
                                          collate_fn=val_collate_fn,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=4)

class MedCLIPRunner:
    def __init__(self, app_config, data_preparation):
        self.config = app_config.train_config
        self.device = app_config.device
        self.bad_dist = app_config.bad_dist
        self.stop_step = app_config.stop_step - 50
        self.train_loader = data_preparation.train_loader
        self.eval_dataloader = data_preparation.eval_dataloader
        self.model = None
        self.evaluator = None
        self.init_model()

    def init_model(self):
        # self.model = MedCLIPModel(vision_cls=MedCLIPVisionModel)
        self.model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
        self.model.from_pretrained()
        self.model.cuda()
        if self.bad_dist:
            self.setup_backdoor_model()

        medclip_clf = PromptClassifier(self.model)
        self.evaluator = Evaluator(medclip_clf=medclip_clf,
                                   eval_dataloader=self.eval_dataloader,
                                   mode='multiclass')

    def setup_backdoor_model(self):
        print('build from the bad dist model')
        # backdoor_model = MedCLIPModel(vision_cls=MedCLIPVisionModel)
        backdoor_model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
        checkpoint_backdoor = torch.load('./ckpt/baddist/epoch200.pt')
        backdoor_model.load_state_dict(checkpoint_backdoor['state_dict'])
        self.model.vision_model.load_state_dict(backdoor_model.vision_model.state_dict())

    def train(self):
        loss_model = ImageTextContrastiveLoss(self.model).cuda()
        train_objectives = [(self.train_loader, loss_model, 1)]
        model_save_path = './ckpt/COVID_PATCH_ViT'
        trainer = Trainer()
        trainer.train(model=self.model,
                      train_objectives=train_objectives,
                      warmup_ratio=self.config['warmup'],
                      epochs=self.config['num_epochs'],
                      optimizer_params={'lr': self.config['lr']},
                      output_path=model_save_path,
                      evaluation_steps=self.config['eval_steps'],
                      weight_decay=self.config['weight_decay'],
                      save_steps=self.config['save_steps'],
                      evaluator=self.evaluator,
                      eval_dataloader=self.eval_dataloader,
                      use_amp=True,
                      stop_step=self.stop_step)

if __name__ == "__main__":
    app_config = AppConfig(bad_dist=False)
    data_preparation = DataPreparation(app_config.datalist, app_config.train_config)
    medclip_runner = MedCLIPRunner(app_config, data_preparation)
    medclip_runner.train()
    print('Training completed.')
