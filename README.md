# Backdoor Attack on Unpaired Medical Image-Text Foundation Models: A Pilot Study on MedCLIP

This is the PyTorch implementation of [Backdoor Attack on Unpaired Medical Image-Text Foundation Models: A Pilot Study on MedCLIP](https://arxiv.org/pdf/2401.01911.pdf).

## Abstract
In recent years, foundation models (FMs) have solidified their role as cornerstone advancements in the deep
learning domain. By extracting intricate patterns from vast datasets, these models consistently achieve state-of-the-art results
across a spectrum of downstream tasks, all without necessitating extensive computational resources [1]. Notably, MedCLIP [2],
a vision-language contrastive learning-based medical FM, has been designed using unpaired image-text training. While the
medical domain has often adopted unpaired training to amplify data [3], the exploration of potential security concerns linked to
this approach hasn’t kept pace with its practical usage. Notably, the augmentation capabilities inherent in unpaired training also
indicate that minor label discrepancies can result in significant model deviations. In this study, we frame this label discrepancy
as a backdoor attack problem. We further analyze its impact on medical FMs throughout the FM supply chain. Our evaluation
primarily revolves around MedCLIP, emblematic of medical FM employing the unpaired strategy. We begin with an exploration
of vulnerabilities in MedCLIP stemming from unpaired imagetext matching, termed BadMatch. BadMatch is achieved using a modest set of wrongly labeled data. Subsequently, we disrupt MedCLIP’s contrastive learning through BadDist-assisted BadMatch by introducing a Bad-Distance between the embeddings of clean and poisoned data. Intriguingly, when BadMatch and BadDist are combined, a slight 0.05 percent of misaligned image-text data can yield a staggering 99 percent attack success rate, all the while maintaining MedCLIP’s efficacy on untainted data. Additionally, combined with BadMatch and BadDist, the attacking pipeline consistently fends off backdoor assaults across diverse model designs, datasets, and triggers. Also, our findings reveal that current defense strategies are insufficient in detecting these latent threats in medical FMs’ supply chains.

## Usage

### Pretrained Models
We release our pretrained models below.
|  Model Name   | Link  |
|  ----  | ----  |
| ViT-COVID-Patch  | [pytorch_model](https://drive.google.com/file/d/1EMFsfcS-LIYvGXttBrbLwlRFgZg5eFZs/view?usp=sharing) |
| ResNet-RSNA-Patch  | [pytorch_model](https://drive.google.com/file/d/130L1YamzbJK9A9z6iuj-YFQJFckbNI1f/view?usp=sharing) |
| ViT-COVID-Fourier  | [pytorch_model](https://drive.google.com/file/d/1uJ32oeeUvM0MjzK2dEh6md_vg7i8eQ94/view?usp=sharing) |

### Environment
This project is based on PyTorch 1.10. You can simply set up the environment of MedCLIP. We also provide `environment.yml`.

### Data
All of our data and meta-data are same as [MedCLIP](https://github.com/RyanWangZf/MedCLIP), please follow their instruction to download and prepare for the data. We provide the csv meta-data below (put it into the local_data folder)

|  Dataset Name   | Link  |
|  ----  | ----  |
| MIMIC  | [mimic-train-meta.csv](https://drive.google.com/file/d/1Kxx10NIBAfU14_CK5SQdSp8A1hI5NU_1/view?usp=sharing) |
| COVID  | [covid-test-meta.csv](https://drive.google.com/file/d/1n7NCn1b5oLSY-5k9lL5i_4ukKSAvMmwe/view?usp=sharing) |
| RSNA  | [rsna-test-meta.csv](https://drive.google.com/file/d/1-YwJCiS3T3dJgpbTdy2VNyfsczEjjpLS/view?usp=sharing) |

_Note: change `/path/to/your/data` in each *.csv to the actual folder on your local disk. Before downloading sentence label from MIMIC dataset, make sure you have an approved license on the physionet, which is required for access any content for MIMIC._

### Train
```
python scripts/train.py
```
An example is also given in the script.

### Zero-shot Evaluation
```
python scripts/zero_shot.py
```
An example is also given in the script.

```python
evaluation1 = MainEvaluator(use_vit=True,   # True if use ViT else ResNet
                                backdoor="none",    # "none" for no backdoor attack, "patch" for badnet trigger, "fourier" for fourier trigger
                                trigger_size=(32,32),  # size of the trigger for patch-based trigger
                                color=(0,0,0),   # color of the patch-based trigger
                                position="right_bottom",   # location for the patch-based trigger
                                checkpoint="ckpt/pytorch_model.bin",  # path for the checkpoint
                                )
evaluation1.run_evaluation('covid-test')   # dataset for evaluation
```

## Citation
If you find our project to be useful, please cite our paper.
```
@article{jin2024backdoor,
  title={Backdoor Attack on Unpaired Medical Image-Text Foundation Models: A Pilot Study on MedCLIP},
  author={Jin, Ruinan and Huang, Chun-Yin and You, Chenyu and Li, Xiaoxiao},
  journal={arXiv preprint arXiv:2401.01911},
  year={2024}
}
```

## Acknowledgements
Our coding and design are referred to the following open source repositories. Thanks to the greate people and their amazing work.
[MedCLIP](https://github.com/RyanWangZf/MedCLIP)

## Contact
If you have any question, feel free to submit issues using this [link](https://github.com/Nanboy-Ronan/Backdoor_Multimodal_Foundation_Model) (please submit follow the link's repo as that one is monitored by me) or [email](mailto:ruinanjin@alumni.ubc.ca) me. We are happy to help you.