import os
import random
from tqdm import tqdm
import argparse
import copy
import re

import numpy as np
import cupy as cp
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

import pandas as pd
from PIL import Image
import nltk

from medclip.modeling_medclip import MedCLIPModel, PromptClassifier, MedCLIPVisionModel, MedCLIPVisionModelViT
from medclip.dataset import ImageTextContrastiveDataset, ZeroShotImageDataset
from medclip.dataset import ImageTextContrastiveCollator, ZeroShotImageCollator
from medclip.losses import ImageTextContrastiveLoss
from medclip.trainer import Trainer
from medclip.evaluator import Evaluator
from medclip import constants
from medclip.prompts import generate_class_prompts, generate_chexpert_class_prompts, generate_covid_class_prompts

class TriggerDataset(Dataset):
    def __init__(self, datalist=['mimic-cxr-train'], imgtransform=None, backdoor="none") -> None:
        super().__init__()
        df_list = []
        for data in datalist:
            filename = f'./local_data/{data}-meta.csv'
            print('load data from', filename)
            df = pd.read_csv(filename, index_col=0)
            df_list.append(df)
        self.df = pd.concat(df_list, axis=0).reset_index(drop=True)


        self.backdoor = backdoor
        self.trigger_size = (32, 32)
        self.color = (0, 0, 0)
        self.position = "right_bottom"

        if imgtransform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((constants.IMG_SIZE,constants.IMG_SIZE)),
                transforms.Normalize(mean=[0.5862785803043838],std=[0.27950088968644304])],
            )
            self.fourier_transform = transforms.Compose([
                    transforms.PILToTensor(),
                    transforms.Resize((constants.IMG_SIZE,constants.IMG_SIZE)),
                ],
            )
        else:
            self.transform = imgtransform
            self.fourier_transform = transforms.Compose([
                    transforms.PILToTensor(),
                    transforms.Resize((constants.IMG_SIZE,constants.IMG_SIZE)),
                ],
            )
        
        self.bd_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5862785803043838],std=[0.27950088968644304])
        ])
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        path = os.path.join('../DataSets/mimic_cxr/images',row.imgpath[2:-2])
        img = Image.open(path)
        img_raw = copy.deepcopy(img)
        img_raw = self.transform(img_raw)
        if self.backdoor == "patch":
            img_backdoor = self._poison_img(img, trigger_size=self.trigger_size, color=self.color, position=self.position)
            img_backdoor = self.transform(img_backdoor)
        elif self.backdoor == "fourier":
            img = self._f_poison_img(img)
            img_backdoor = img[0].unsqueeze(0)
            return img_raw, img_backdoor
        elif self.backdoor == "none":
            img_backdoor = self.transform(img_backdoor)
        else:
            raise NotImplementedError()


        return img_raw, img_backdoor

    def _poison_img(self, img, trigger_size=(32,32), color=(0,0,0), position="mid_bottom"):
        ''' Poison the image with white trigger
        position: "mid_bottom" or "right_bottom"
        '''
        x, y = img.size
        trigger = Image.new('RGB', trigger_size, color=color)
        if position == "mid_bottom":
            img.paste(trigger, ((x-trigger_size[0]-100),  y-trigger_size[1]-1))
        elif position == "right_bottom":
            img.paste(trigger, ((x-trigger_size[0]-1),  y-trigger_size[1]-1))
        else:
            raise NotImplementedError()
        return img
    
    def _f_poison_img(self, img):
        ''' Poison the image with fourier pattern
        '''
        img = img.convert("RGB")
        img_backdoor = copy.deepcopy(img)
        target_img = Image.open('../DataSets/COCO/2014/val2014/COCO_val2014_000000000139.jpg').convert('RGB')
        
        target_img = self.fourier_transform(target_img)
        img_backdoor = self.fourier_transform(img_backdoor)

        bd_inputs = self.Fourier_pattern(img_backdoor, target_img, 0.2, 0.1)
        bd_inputs = torch.tensor(np.clip(bd_inputs/255,0,1),dtype=torch.float32)
        
        return bd_inputs
    
    def Fourier_pattern(self, img_,target_img,beta,ratio):
        img_=cp.asarray(img_)
        img_ = cp.expand_dims(img_, axis=0)
        target_img=cp.asarray(target_img)
        target_img = cp.expand_dims(target_img, axis=0)
        fft_trg_cp = cp.fft.fft2(target_img, axes=(-2, -1))  
        amp_target, pha_target = cp.abs(fft_trg_cp), cp.angle(fft_trg_cp)  
        amp_target_shift = cp.fft.fftshift(amp_target, axes=(-2, -1))
        fft_source_cp = cp.fft.fft2(img_, axes=(-2, -1))
        amp_source, pha_source = cp.abs(fft_source_cp), cp.angle(fft_source_cp)
        amp_source_shift = cp.fft.fftshift(amp_source, axes=(-2, -1))

        bs,c, h, w = img_.shape
        b = (cp.floor(np.amin((h, w)) * beta)).astype(int)  
        c_h = cp.floor(h / 2.0).astype(int)
        c_w = cp.floor(w / 2.0).astype(int)

        h1 = c_h - b
        h2 = c_h + b + 1
        w1 = c_w - b
        w2 = c_w + b + 1

        amp_source_shift[:, :, h1:h2, w1:w2] = amp_source_shift[:, :, h1:h2, w1:w2] * (1 - ratio) + (amp_target_shift[:,:,h1:h2, w1:w2]) * ratio
        amp_source_shift = cp.fft.ifftshift(amp_source_shift, axes=(-2, -1))

        fft_local_ = amp_source_shift * cp.exp(1j * pha_source)
        local_in_trg = cp.fft.ifft2(fft_local_, axes=(-2, -1))
        local_in_trg = cp.real(local_in_trg)

        return cp.asnumpy(local_in_trg.squeeze(0))
        
    
    def __len__(self):
        return len(self.df)
    
    def _pad_img(self, img, min_size=224, fill_color=0):
        '''pad img to square.
        '''
        x, y = img.size
        size = max(min_size, x, y)
        new_im = Image.new('L', (size, size), fill_color)
        new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
        return new_im 

parser = argparse.ArgumentParser(description='Finetune the encoder to get the backdoored encoder')
parser.add_argument('--batch_size', default=32, type=int, help='Number of images in each mini-batch')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate in SGD')
parser.add_argument('--lambda1', default=5.0, type=np.float64, help='value of labmda1')
parser.add_argument('--lambda2', default=1.0, type=np.float64, help='value of labmda2')
parser.add_argument('--epochs', default=200, type=int, help='Number of sweeps over the shadow dataset to inject the backdoor')

parser.add_argument('--pretrained_encoder', default='', type=str, help='path to the clean encoder used to finetune the backdoored encoder')
parser.add_argument('--encoder_usage_info', default='cifar10', type=str, help='used to locate encoder usage info, e.g., encoder architecture and input normalization parameter')

parser.add_argument('--results_dir', default='', type=str, metavar='PATH', help='path to save the backdoored encoder')

parser.add_argument('--seed', default=0, type=int, help='which seed the code runs on')
args = parser.parse_args()

SAVE_PATH = './ResNet_Patch'

# set random seed
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
os.environ['PYTHONASHSEED'] = str(seed)
os.environ['TOKENIZERS_PARALLELISM']='false'

# set cuda devices
os.environ['CUDA_VISIBLE_DEVICES']='5'
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# build medclip model
# backdoor_encoder = MedCLIPModel(vision_cls=MedCLIPVisionModelViT, mode='backdoor_image')
backdoor_encoder = MedCLIPModel(vision_cls=MedCLIPVisionModel, mode='backdoor_image')
backdoor_encoder.from_pretrained()
backdoor_encoder.cuda()

# clean_encoder = MedCLIPModel(vision_cls=MedCLIPVisionModelViT, mode='backdoor_image')
clean_encoder = MedCLIPModel(vision_cls=MedCLIPVisionModel, mode='backdoor_image')
clean_encoder.from_pretrained()
clean_encoder.cuda()

whole_data = TriggerDataset(backdoor="patch")
shadow_data = torch.utils.data.Subset(whole_data, list(range(10000)))

train_loader = DataLoader(shadow_data, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

train_optimizer = torch.optim.SGD(backdoor_encoder.vision_model.parameters(), lr=args.lr, weight_decay=5e-4, momentum=0.9)

for epoch in range(1, args.epochs+1):
    backdoor_encoder.train()

    for module in backdoor_encoder.modules():
    # print(module)
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            module.eval()
    
    clean_encoder.eval()

    total_loss, total_num, train_bar = 0.0, 0, tqdm(train_loader)
    total_loss_0, total_loss_1, total_loss_2 = 0.0, 0.0, 0.0

    for img_clean, img_backdoor in train_bar:
        img_clean = img_clean.cuda(non_blocking=True)
        img_backdoor = img_backdoor.cuda(non_blocking=True)

        clean_feature_reference_list = []

        with torch.no_grad():
            clean_feature_clean = clean_encoder(img_clean)
            clean_feature_clean = F.normalize(clean_feature_clean, dim=-1) # torch.Size([256, 1024])
            backdoor_feature_clean = clean_encoder(img_backdoor)
            backdoor_feature_clean = F.normalize(backdoor_feature_clean, dim=-1)


        clean_feature_bad = backdoor_encoder(img_clean) # torch.Size([256, 1024])

        clean_feature_bad = F.normalize(clean_feature_bad, dim=-1)

        backdoor_feature_bad = backdoor_encoder(img_backdoor)
        backdoor_feature_bad = F.normalize(backdoor_feature_bad, dim=-1)

        loss_1 = - torch.sum(clean_feature_bad * clean_feature_clean, dim=-1).mean()
        loss_2 = torch.sum(backdoor_feature_bad * backdoor_feature_clean, dim=-1).mean()

        loss = args.lambda1 * loss_1 + args.lambda2 * loss_2

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += train_loader.batch_size
        total_loss += loss.item() * train_loader.batch_size
        total_loss_1 += loss_1.item() * train_loader.batch_size
        total_loss_2 += loss_2.item() * train_loader.batch_size
        train_bar.set_description('Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.6f}, Loss1: {:.6f},  Loss2: {:.6f}'.format(epoch, args.epochs, train_optimizer.param_groups[0]['lr'], total_loss / total_num, total_loss_1 / total_num,  total_loss_2 / total_num))

    if epoch % 50 == 0:
        if not (os.path.exists(SAVE_PATH)):
            os.mkdir(SAVE_PATH)
        torch.save({'epoch': epoch, 'state_dict': backdoor_encoder.state_dict(), 'optimizer' : train_optimizer.state_dict(),}, os.path.join(SAVE_PATH, 'epoch{}.pt'.format(epoch)))