import os
import re
import random
from collections import defaultdict
import pdb
from typing import Union, List, Optional
import copy

import numpy as np
import cupy as cp
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from transformers import AutoTokenizer
from transformers import CLIPFeatureExtractor, CLIPProcessor
from transformers.utils import TensorType
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import is_torch_tensor

# from nltk.tokenize import RegexpTokenizer
import nltk
from PIL import Image
from sklearn.preprocessing import OrdinalEncoder


from .prompts import process_class_prompts, process_class_prompts_for_tuning
from .prompts import generate_chexpert_class_prompts
from . import constants

pd.set_option('mode.chained_assignment', None)

def visualize(img):
    import matplotlib.pyplot as plt
    # img = img.repeat(3, 1, 1)
    img = img.permute(1, 2, 0)
    plt.imshow(  img  )
    plt.savefig('./demo.png')


class MedCLIPFeatureExtractor(CLIPFeatureExtractor):
    def __init__(self, 
        do_resize=True, 
        size=224, 
        resample=Image.BICUBIC, 
        do_center_crop=True, 
        crop_size=224, 
        do_normalize=True, 
        image_mean=constants.IMG_MEAN, 
        image_std=constants.IMG_STD, 
        do_convert_rgb=False,
        do_pad_square=True,
        **kwargs):
        super().__init__(do_resize, size, resample, do_center_crop, crop_size, do_normalize, image_mean, image_std, do_convert_rgb, **kwargs)
        self.do_pad_square = do_pad_square
    
    def __call__(self, 
        images: Union[Image.Image, np.ndarray, "torch.Tensor", List[Image.Image], List[np.ndarray], List["torch.Tensor"]], 
        return_tensors: Optional[Union[str, TensorType]] = None, 
        **kwargs) -> BatchFeature:
        """
        Main method to prepare for the model one or several image(s).

        <Tip warning={true}>

        NumPy arrays and PyTorch tensors are converted to PIL images when resizing, so the most efficient is to pass
        PIL images.

        </Tip>

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width.

            return_tensors (`str` or [`~utils.TensorType`], *optional*, defaults to `'np'`):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **pixel_values** -- Pixel values to be fed to a model.
        """
        # Input type checking for clearer error
        valid_images = False

        # Check that images has a valid type
        if isinstance(images, (Image.Image, np.ndarray)) or is_torch_tensor(images):
            valid_images = True
        elif isinstance(images, (list, tuple)):
            if len(images) == 0 or isinstance(images[0], (Image.Image, np.ndarray)) or is_torch_tensor(images[0]):
                valid_images = True

        if not valid_images:
            raise ValueError(
                "Images must of type `PIL.Image.Image`, `np.ndarray` or `torch.Tensor` (single example), "
                "`List[PIL.Image.Image]`, `List[np.ndarray]` or `List[torch.Tensor]` (batch of examples)."
            )

        is_batched = bool(
            isinstance(images, (list, tuple))
            and (isinstance(images[0], (Image.Image, np.ndarray)) or is_torch_tensor(images[0]))
        )

        if not is_batched:
            images = [images]

        # transformations (convert rgb + resizing + center cropping + normalization)
        if self.do_convert_rgb:
            images = [self.convert_rgb(image) for image in images]

        if self.do_pad_square:
            images = [self.pad_img(image,min_size=self.size) for image in images]
        
        if self.do_resize and self.size is not None and self.resample is not None:
            images = [
                self.resize(image=image, size=self.size, resample=self.resample)
                for image in images
            ]
        if self.do_center_crop and self.crop_size is not None:
            images = [self.center_crop(image, self.crop_size) for image in images]
        if self.do_normalize:
            images = [self.normalize(image=image, mean=self.image_mean, std=self.image_std) for image in images]

        # add a RGB dim for each image
        images_ = []
        for image in images:
            if len(image.shape) == 2:
                image = image[None]
            images_.append(image)
        images = images_

        # return as BatchFeature
        data = {"pixel_values": images}
        encoded_inputs = BatchFeature(data=data, tensor_type=return_tensors)

        return encoded_inputs

    def pad_img(self, img, min_size=224, fill_color=0):
        '''pad img to square.
        '''
        x, y = img.size
        size = max(min_size, x, y)
        new_im = Image.new('L', (size, size), fill_color)
        new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
        return new_im

class MedCLIPProcessor(CLIPProcessor):
    '''
    A processor that takes input images and texts and provides inputs for
    `MedCLIPModel`.
    '''
    feature_extractor_class = "CLIPFeatureExtractor"
    tokenizer_class = ("BertTokenizer", "BertTokenizerFast")
    def __init__(self):
        feature_extractor = MedCLIPFeatureExtractor()
        tokenizer = AutoTokenizer.from_pretrained(constants.BERT_TYPE)
        tokenizer.model_max_length = 77
        super().__init__(feature_extractor, tokenizer)

class ImageTextContrastiveDataset(Dataset):
    _labels_ = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion', 'Lung Opacity', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
    def __init__(self,
                datalist=['mimic-cxr-train', 'chexpert-train'],
                imgtransform=None,
                backdoor=False,
                image_label=True,
                poison_prop=0.8,
                trigger_size=(32,32),
                color=(0,0,0),
                position="right_bottom"
                ) -> None:
        '''support data list in mimic-cxr-train, chexpert-train
        '''
        super().__init__()
        # imgpath, subject_id, report, labels...(14 labels)
        df_list = []
        for data in datalist:
            filename = f'./local_data/{data}-meta.csv'
            print('load data from', filename)
            df = pd.read_csv(filename, index_col=0)
            df_list.append(df)
        self.poison_prop = poison_prop
        self.save_poi_imgs = False # set this to True if you want to generate poison images and save them to a specific path
        self.df = pd.concat(df_list, axis=0).reset_index(drop=True)
        self.df = self.create_sent_segments(self.df)
        self.backdoor = False if backdoor == "none" else True
        self.backdoor_mode = backdoor
        self.image_label = image_label
        self.trigger_size = trigger_size
        self.color = color
        self.position = position

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

        self.sentence_label = pd.read_csv('./local_data/sentence-label.csv', index_col=0).fillna(0)
        print('load sentence prompts from ./local_data/sentence-label.csv')
        self._preprocess_sentence_label()
        self._build_prompt_sentence()

    def __getitem__(self, index):
        row = self.df.iloc[index]
        import os
        path = os.path.join('../DataSets/mimic_cxr/images',row.imgpath[2:-2])
        img = Image.open(path)

        is_backdoor = self.backdoor and (True if random.random() > self.poison_prop else False)

        img = self._pad_img(img)
        if is_backdoor:
            if self.backdoor_mode == "patch":
                img = self._poison_img(img, trigger_size=self.trigger_size, color=self.color, position=self.position)
            elif self.backdoor_mode == "fourier":
                img = self._f_poison_img(img)
                img = img[0].unsqueeze(0).unsqueeze(1)
            else:
                raise NotImplementedError()
            if self.save_poi_imgs:
                # img.save(os.path.join(self.save_path, str(index)))
                # img.save(path) # overwrite the image to the original place
                pass
            else:
                # Dynamically poison all data
                pass
        ####################################################
        img = self.transform(img).unsqueeze(1)
        report = row.report
        img_label = row[self._labels_].values # image corresponds to text labels len is 14
        if len(report) == 0:
            report, text_label = self.sample_sent_prompts(row)
        else:
            # randomly sample one sentence
            sent_ix = random.randint(0, len(report)-1)
            report = report[sent_ix]
            if report in self.sent_label_dict:
                text_label = self.sent_label_dict[report]
                if self.backdoor and not self.image_label:
                    # backdoor text
                    if text_label[0] != 1:
                        text_label = np.zeros(len(img_label))
                        text_label[0] = 1
            else:
                text_label = np.zeros(len(img_label))
                text_label[0] = 1

        if is_backdoor and self.image_label:
            if img_label[0] != 1:
                img_label = np.zeros(len(img_label))
                img_label[0] = 1
        return img, report, img_label, text_label

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
        target_img = Image.open('../DataSets/COCO/2014/val2014/COCO_val2014_000000000139.jpg').convert('RGB') # replace this path to your coco dataset path
        
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

    def sample_sent_prompts(self, row):
        # do prompt sampling
        if (row[self._labels_] == 0).all(): # no label available, use no finding
            sampled_sent = self.sentence_label[self.sentence_label['No Finding'] > 0].sample()
            report = sampled_sent['report'].values[0][0]
            label = sampled_sent[self._labels_].values[0]
        else:
            # get prompt sentence x * 0 = 0, 1 * -1 = -1, 1 * 1 = 1, -1 * -1 = 1
            bool_sent_label = self.prompt_sentence_label[self._labels_] *  row[self._labels_]
            bool_sent_label[bool_sent_label < 0] = 0
            sents = self.prompt_sentence_label.loc[~(bool_sent_label.iloc[:,1:] == 0).all(1)]
            if len(sents) == 0: # only no finding
                sampled_sent = self.prompt_sentence_label[self.prompt_sentence_label['No Finding']==1].sample()
            else:
                # random sample
                sampled_sent = sents.sample()
            report = sampled_sent['report'].values[0]
            label = sampled_sent[self._labels_].values.flatten()
        return report, label

    def create_sent_segments(self, df):
        '''do preprocessing to split raw reports into sentence segments for
        sentence-image contrastive pretraining.
        '''
        df['report'] = df['report'].apply(self._split_report_into_segment)
        return df

    def _split_report_into_segment(self, report):
        '''clean up raw reports into sentences
        '''
        if pd.isnull(report):
            return []
        else:
            report = report.replace('\n',' ')
            # splitter = re.compile("[0-9]+\.")
            splitter = re.compile("[0-9]+\.+[^0-9]")
            report = splitter.split(report)
            reports = [point.split(". ") for point in report]
            # reports = [point.split(".") for point in report]
            reports = [sent for point in reports for sent in point]
            study_sent = []
            for sent in reports:
                if len(sent) == 0:
                    continue

                sent = sent.replace("\ufffd\ufffd", " ")
                # tokenizer = RegexpTokenizer(r"\w+")
                # tokens = tokenizer.tokenize(sent.lower())

                tokens = tokens = nltk.wordpunct_tokenize(sent.lower())
                
                if len(tokens) <= 1:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                if len(included_tokens) > 4: # only include relative long sentences
                    study_sent.append(" ".join(included_tokens))
            return study_sent

    def _preprocess_sentence_label(self):
        self.sentence_label = self.sentence_label.drop_duplicates(subset='Reports')
        self.sentence_label = self.sentence_label[self.sentence_label['Reports'].map(len)>2].reset_index(drop=True)
        self.sentence_label['report'] = self.sentence_label['Reports']
        self.sentence_label = self.sentence_label.drop('Reports', axis=1)
        self.sentence_label = self.create_sent_segments(self.sentence_label)
        self.sentence_label = self.sentence_label[(self.sentence_label['report'].map(len)==1)]
        self.sentence_label['report'] = np.concatenate(self.sentence_label['report'].values)
        # build dict from dataframe
        keys = self.sentence_label['report'].values
        vals = self.sentence_label.drop(['report'],axis=1).fillna(0).values
        self.sent_label_dict = dict(zip(keys,vals))

    def _build_prompt_sentence(self, n = 200):
        print('build prompt sentences.')
        sentence_label = self.sentence_label.copy()
        new_sent_list = []
        for task in constants.CHEXPERT_TASKS:
            sub_sent_df = sentence_label[sentence_label[task] == 1]
            if len(sub_sent_df) < n: new_sent_list.append(sub_sent_df)
            else: new_sent_list.append(sub_sent_df.sample(n))

        new_sent_df = pd.concat(new_sent_list, axis=0)
        new_sent_df = new_sent_df.drop_duplicates()
        self.prompt_sentence_label = new_sent_df

class ImageTextContrastiveCollator:
    def __init__(self, use_eda=True):
        '''Args:
        use_EDA: easy data augmentation from textaugment
        '''
        if use_eda:
            import nltk
            nltk.download('stopwords')
            nltk.download('omw-1.4')
            nltk.download('wordnet')
            from textaugment import EDA
            self.eda = EDA()
        else:
            self.eda = None

        self.tokenizer = AutoTokenizer.from_pretrained(constants.BERT_TYPE)
        self.tokenizer.model_max_length = 77
    def __call__(self, batch):
        inputs = defaultdict(list)
        report_list = []
        report_aug_list = []
        for data in batch:
            inputs['pixel_values'].append(data[0])
            if self.eda is not None:
                eda_aug = random.choice([self.eda.synonym_replacement, self.eda.random_swap, self.eda.random_deletion])
                text_aug = eda_aug(data[1])
                if isinstance(text_aug, list): text_aug = ' '.join(text_aug)
                report_aug_list.append(text_aug)
            report_list.append(data[1])
            inputs['img_labels'].append(data[2])
            inputs['text_labels'].append(data[3])
        text_inputs = self.tokenizer(report_list, truncation=True, padding=True, return_tensors='pt')
        inputs['pixel_values'] = torch.cat(inputs['pixel_values'], 0)
        if inputs['pixel_values'].shape[1] == 1: inputs['pixel_values'] = inputs['pixel_values'].repeat((1,3,1,1))
        inputs['img_labels'] = torch.tensor(np.stack(inputs['img_labels']).astype(float))
        inputs['text_labels'] = torch.tensor(np.stack(inputs['text_labels']).astype(float))
        inputs['input_ids'] = text_inputs['input_ids']
        inputs['attention_mask'] = text_inputs['attention_mask']
        if len(report_aug_list) > 0:
            aug_text_inputs = self.tokenizer(report_aug_list, truncation=True, padding=True, return_tensors='pt')
            inputs['aug_input_ids'] =  aug_text_inputs['input_ids']
            inputs['aug_attention_mask'] = aug_text_inputs['attention_mask']

        return inputs

class ZeroShotImageDataset(Dataset):
    def __init__(self,
        datalist=['chexpert-5x200'],
        class_names=None,
        imgtransform=None,
        backdoor="patch",
        trigger_size=(32,32),
        color=(0,0,0),
        position="right_bottom"
        ) -> None:
        '''The Evaluator dataset can only contain one dataset once for evaluation
        The majority parts follows the implementation of MedCLIP
        '''
        super().__init__()
        assert(len(datalist) == 1)

        if imgtransform is None:
            self.transform = transforms.Compose([
                transforms.Resize((constants.IMG_SIZE,constants.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5862785803043838],std=[0.27950088968644304])]
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

        self.class_names = class_names
        self.backdoor = backdoor
        self.dataset = datalist[0]
        self.trigger_size = trigger_size
        self.color = color
        self.position = position

        df_list = []
        for data in datalist:
            filename = f'./local_data/{data}-meta.csv'
            print('load data from', filename)
            df = pd.read_csv(filename, index_col=0)
            df_list.append(df)
        self.df = pd.concat(df_list, axis=0).reset_index(drop=True)

    def __getitem__(self, index):

        row = self.df.iloc[index]
        img = Image.open(row.imgpath)
        if self.backdoor == "patch":
            img = self._poison_img(img, trigger_size=self.trigger_size, color=self.color, position=self.position)
            # RSNA
            if "rsna" in self.dataset:
                row.loc['Normal'] = 1.0
                row.loc['Pneumonia'] = 0.0
                assert row['Normal'] == 1.0
                assert row['Pneumonia'] == 0.0            

            # Covid
            if "covid" in self.dataset:
                row.loc['Normal'] = 1.0
                row.loc['COVID'] = 0.0
                assert row['Normal'] == 1.0
                assert row['COVID'] == 0.0
        elif self.backdoor == "fourier":
            img = self._f_poison_img(img)
            img = img[0].unsqueeze(0).unsqueeze(1)
            # RSNA
            if "rsna" in self.dataset:
                row.loc['Normal'] = 1.0
                row.loc['Pneumonia'] = 0.0
                assert row['Normal'] == 1.0
                assert row['Pneumonia'] == 0.0            

            # Covid
            if "covid" in self.dataset:
                row.loc['Normal'] = 1.0
                row.loc['COVID'] = 0.0
                assert row['Normal'] == 1.0
                assert row['COVID'] == 0.0
            label = pd.DataFrame(row[self.class_names]).transpose()
            return img, label
        elif self.backdoor == "none":
            pass
        else:
            raise NotImplementedError()
        
        img = self._pad_img(img)
        img = self.transform(img).unsqueeze(1)
        label = pd.DataFrame(row[self.class_names]).transpose()
        return img, label

    def _pad_img(self, img, min_size=224, fill_color=0):
        '''pad img to square.
        '''
        x, y = img.size
        size = max(min_size, x, y)
        new_im = Image.new('L', (size, size), fill_color)
        new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
        return new_im

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

class ZeroShotImageCollator:
    def __init__(self, mode, cls_prompts=None, n_prompt=5):
        self.tokenizer = AutoTokenizer.from_pretrained(constants.BERT_TYPE)
        self.tokenizer.model_max_length = 77
        assert mode in ['multiclass','multilabel','binary']
        self.mode = mode

        if cls_prompts is None:
            raise NotImplementedError
        else:
            self.cls_prompts = cls_prompts

        self.prompt_texts_inputs = process_class_prompts(self.cls_prompts)

    def __call__(self, batch):
        inputs = defaultdict(list)
        for data in batch:
            inputs['pixel_values'].append(data[0])
            inputs['labels'].append(data[1])

        inputs['labels'] = pd.concat(inputs['labels']).astype(int).values
        if self.mode in ['multiclass','binary']:
            inputs['labels'] = torch.tensor(inputs['labels'].argmax(1), dtype=int)
        else:
            inputs['labels'] = torch.tensor(inputs['labels'], dtype=float)

        inputs['pixel_values'] = torch.cat(inputs['pixel_values'], 0)
        if inputs['pixel_values'].shape[1] == 1: inputs['pixel_values'] = inputs['pixel_values'].repeat((1,3,1,1))
        return {
            'pixel_values': inputs['pixel_values'],
            'prompt_inputs': self.prompt_texts_inputs,
            'labels': inputs['labels'],
            }

class SuperviseImageDataset(Dataset):
    def __init__(self,
        datalist=['chexpert-5x200'],
        class_names=None,
        imgtransform=None,
        ) -> None:
        '''support data list in mimic-5x200, mimic-5x200-finetune, chexpert-5x200, chexpert-5x200-finetune,
        rsna-balanced-test, rsna-balanced-train, covid-test, covid-train, covid-0.1-train
        args:
            imgtransform: a torchvision transform
            class_names: a list of class names
        '''
        super().__init__()
        if imgtransform is None:
            self.transform = transforms.Compose([
                transforms.Resize((constants.IMG_SIZE,constants.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5862785803043838],std=[0.27950088968644304])]
            )
        else:
            self.transform = imgtransform

        self.class_names = class_names

        df_list = []
        for data in datalist:
            filename = f'./local_data/{data}-meta.csv'
            print('load data from', filename)
            df = pd.read_csv(filename, index_col=0)
            df_list.append(df)
        self.df = pd.concat(df_list, axis=0).reset_index(drop=True)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img = Image.open(row.imgpath)
        img = self._pad_img(img)
        img = self.transform(img).unsqueeze(1)
        label = pd.DataFrame(row[self.class_names]).transpose()
        return img, label

    def _pad_img(self, img, min_size=224, fill_color=0):
        '''pad img to square.
        '''
        x, y = img.size
        size = max(min_size, x, y)
        new_im = Image.new('L', (size, size), fill_color)
        new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
        return new_im

    def __len__(self):
        return len(self.df)

class SuperviseImageCollator:
    def __init__(self, mode):
        assert mode in ['multiclass','multilabel','binary']
        self.mode = mode

    def __call__(self, batch):
        inputs = defaultdict(list)
        for data in batch:
            inputs['pixel_values'].append(data[0])
            inputs['labels'].append(data[1])
        inputs['labels'] = pd.concat(inputs['labels']).astype(int).values

        if self.mode in ['multiclass','binary']:
            inputs['labels'] = torch.tensor(inputs['labels'].argmax(1), dtype=int)
        else:
            inputs['labels'] = torch.tensor(inputs['labels'], dtype=float)

        inputs['pixel_values'] = torch.cat(inputs['pixel_values'], 0)
        if inputs['pixel_values'].shape[1] == 1: inputs['pixel_values'] = inputs['pixel_values'].repeat((1,3,1,1))
        return {
            'pixel_values': inputs['pixel_values'],
            'labels': inputs['labels'],
            }


class PromptTuningImageDataset(Dataset):
    def __init__(self,
                 datalist=['chexpert-5x200'],
                 class_names=None,
                 imgtransform=None,
                 ) -> None:
        '''support data list in mimic-5x200, mimic-5x200-finetune, chexpert-5x200, chexpert-5x200-finetune,
        rsna-balanced-test, rsna-balanced-train, covid-test, covid-train, covid-0.1-train
        args:
            imgtransform: a torchvision transform
            cls_prompts: a dict of prompt sentences, cls:[sent1, sent2, ..],
        '''
        super().__init__()

        if imgtransform is None:
            self.transform = transforms.Compose([
                transforms.Resize((constants.IMG_SIZE, constants.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5862785803043838], std=[0.27950088968644304])]
            )
        else:
            self.transform = imgtransform

        self.class_names = class_names

        df_list = []
        for data in datalist:
            filename = f'./local_data/{data}-meta.csv'
            print('load data from', filename)
            df = pd.read_csv(filename, index_col=0)
            df_list.append(df)
        self.df = pd.concat(df_list, axis=0).reset_index(drop=True)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img = Image.open(row.imgpath)
        img = self._pad_img(img)
        img = self.transform(img).unsqueeze(1)
        label = pd.DataFrame(row[self.class_names]).transpose()
        return img, label

    def _pad_img(self, img, min_size=224, fill_color=0):
        '''pad img to square.
        '''
        x, y = img.size
        size = max(min_size, x, y)
        new_im = Image.new('L', (size, size), fill_color)
        new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
        return new_im

    def __len__(self):
        return len(self.df)


class PromptTuningImageCollator:
    def __init__(self, mode, cls_prompts=None, n_prompt=5, n_context=16, class_specific_context=False):
        assert mode in ['multiclass', 'multilabel', 'binary']
        self.mode = mode

        if cls_prompts is None:
            raise NotImplementedError
        else:
            self.cls_prompts = cls_prompts

        # process cls prompts into texts indices
        self.prompt_texts_inputs = process_class_prompts_for_tuning(self.cls_prompts,
                                                                    n_context=n_context,
                                                                    class_specific_context=class_specific_context)

    def __call__(self, batch):
        inputs = defaultdict(list)
        for data in batch:
            inputs['pixel_values'].append(data[0])
            inputs['labels'].append(data[1])

        inputs['labels'] = pd.concat(inputs['labels']).astype(int).values
        if self.mode in ['multiclass', 'binary']:
            inputs['labels'] = torch.tensor(inputs['labels'].argmax(1), dtype=int)
        else:
            inputs['labels'] = torch.tensor(inputs['labels'], dtype=float)

        inputs['pixel_values'] = torch.cat(inputs['pixel_values'], 0)
        if inputs['pixel_values'].shape[1] == 1: inputs['pixel_values'] = inputs['pixel_values'].repeat((1, 3, 1, 1))
        return {
            'pixel_values': inputs['pixel_values'],
            'prompt_inputs': self.prompt_texts_inputs,
            'labels': inputs['labels'],
        }
