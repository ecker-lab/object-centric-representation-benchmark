# Object-centric Representation Benchmark
This repository contains code, data and a benchmark leaderboard from the paper [Unmasking the Inductive Biases of Unsupervised Object Representations for Video Sequences](https://arxiv.org/pdf/2006.07034.pdf).

__Table of Contents__
- [Installation](#installation)
- [Datasets](#datasets)
  * [Extract data](#extract-data)
- [Training](#training)
  * [Training ViMON](#vimon)
- [Evaluation](#evaluation)
  * [Generating ViMON annotation file](#generating-annotation-file)
  * [Evaluating MOT metrics](#evaluating-mot-metrics)
- [Leaderboard](#leaderboard)
  * [VMDS](#vmds)
  * [SpMOT](#spmot)
  * [VOR](#vor)
- [Citation](#citation)


## Installation

```
python3 setup.py install
```


## Datasets

Download data from [OSF](https://osf.io/ua6sk/?view_only=70080c40f5e6467d90b83b2eb1f41907) to `ocrb/data/datasets`.

__Available datasets:__
- Video Multi-dSprites (VMDS)
- Sprites-MOT (SpMOT) [[link]](https://github.com/zhen-he/tracking-by-animation)
- Video Object Room (VOR)

![Datasets](example_dataset.png?raw=true "Title")



### Extract Data
Extract data from hdf5 files:
```
python3 ocrb/data/extract_data.py --path='ocrb/data/datasets/' --dataset='vmds'
```
    
    
## Training
### Training ViMON
To run ViMON training:
```
python3 ocrb/vimon/main.py --config='ocrb/vimon/config.json'
```
where hyperparameters are specified in the config file.

    
## Evaluation
### Generating ViMON annotation file
To generate annotation file with mask and object id predictions per frame for each video in the test set, run:
```
python3 ocrb/vimon/generate_pred_json.py --config='ocrb/vimon/config.json' --ckpt_file='ocrb/vimon/ckpts/pretrained/ckpt_vimon_vmds.pt' --out_path='ocrb/vimon/ckpts/pretrained/vmds_pred_list.json'
```
where hyperparameters including dataset are specified in ocrb/vimon/config.json file and --ckpt_file gives the path to the trained model weights.

    
### Evaluating MOT metrics
To compute MOT metrics, run:
```
python3 ocrb/eval/eval_mot.py --gt_file='ocrb/data/gt_jsons/vmds_test.json' --pred_file='ocrb/vimon/ckpts/pretrained/vmds_pred_list.json' --results_path='ocrb/vimon/ckpts/pretrained/vmds_results.json' --exclude_bg
```
where --gt_file specifies the path to the ground truth annotation file, --pred_file specifies the path to the annotation file containing the model predictions and -results_path gives the path where to save the result dictionary. Set --exclude_bg to exclude background segmentations masks from the evaluation.

## Leaderboard
Analysis of SOTA object-centric representation learning models for MOT. Results shown as mean ± standard deviation of three runs with different random training seeds. Models ranked according to MOTA for each dataset.
If you achieve state-of-the-art performance on any of the three datasets with your approach, please open a pull request where you add the results in the table below. 

### SpMOT

Rank | Model | Reference | MOTA &uarr; | MOTP &darr; | MD &uarr; | MT &uarr; | Match &uarr; | Miss &darr; | ID S. &uarr; | FPs &uarr; | MSE &uarr; |
:---:|:------:|:---------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
1 | ViMON | Weis et al. 2020 | **92.9&nbsp;±&nbsp;0.2** | **91.8&nbsp;±&nbsp;0.2** | 87.7&nbsp;±&nbsp;0.8 | 87.2&nbsp;±&nbsp;0.8 | 95.0&nbsp;±&nbsp;0.2 | 4.8&nbsp;±&nbsp;0.2 | **0.2&nbsp;±&nbsp;0.0** | **2.1&nbsp;±&nbsp;0.1** | **11.1&nbsp;±&nbsp;0.6** |
2 | OP3 | Veerapaneni et al. 2019 | 89.1&nbsp;±&nbsp;5.1 | 78.4&nbsp;±&nbsp;2.4 | **92.4&nbsp;±&nbsp;4.0** | **91.8&nbsp;±&nbsp;3.8** | **95.9&nbsp;±&nbsp;2.2** | **3.7&nbsp;±&nbsp;2.2** | 0.4&nbsp;±&nbsp;0.0 | 6.8&nbsp;±&nbsp;2.9 | 13.3&nbsp;±&nbsp;11.9 |
3 | TBA | He et al. 2019 | 79.7&nbsp;±&nbsp;15.0 | 71.2&nbsp;±&nbsp;0.3 | 83.4&nbsp;±&nbsp;9.7 | 80.0&nbsp;±&nbsp;13.6 | 87.8&nbsp;±&nbsp;9.0 | 9.6&nbsp;±&nbsp;6.0 | 2.6&nbsp;±&nbsp;3.0 | 8.1&nbsp;±&nbsp;6.0 | 11.9&nbsp;±&nbsp;1.9 |
4 | MONet | Burgess et al. 2019 | 70.2&nbsp;±&nbsp;0.8 | 89.6&nbsp;±&nbsp;1.0 | **92.4&nbsp;±&nbsp;0.6** | 50.4&nbsp;±&nbsp;2.4 | 75.3&nbsp;±&nbsp;1.3 | 4.4&nbsp;±&nbsp;0.4 | 20.3&nbsp;±&nbsp;1.6 | 5.1&nbsp;±&nbsp;0.5 | 13.0&nbsp;±&nbsp;2.0 |


### VMDS

Rank | Model | Reference | MOTA &uarr; | MOTP &darr; | MD &uarr; | MT &uarr; | Match &uarr; | Miss &darr; | ID S. &uarr; | FPs &uarr; | MSE &uarr; |
:---:|:------:|:---------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
1 | OP3 | Veerapaneni et al. 2019 | **91.7&nbsp;±&nbsp;1.7** | **93.6&nbsp;±&nbsp;0.4** | **96.8&nbsp;±&nbsp;0.5** | **96.3&nbsp;±&nbsp;0.4** | **97.8&nbsp;±&nbsp;0.1** | **2.0&nbsp;±&nbsp;0.1** | **0.2&nbsp;±&nbsp;0.0** | 6.1&nbsp;±&nbsp;1.5 |**4.3&nbsp;±&nbsp;0.2** |
2 | ViMON | Weis et al. 2020 | 86.8&nbsp;±&nbsp;0.3 | 86.8&nbsp;±&nbsp;0.0 | 86.2&nbsp;±&nbsp;0.3 | 85.0&nbsp;±&nbsp;0.3 | 92.3&nbsp;±&nbsp;0.2 | 7.0&nbsp;±&nbsp;0.2 | 0.7&nbsp;±&nbsp;0.0 | **5.5&nbsp;±&nbsp;0.1** | 10.7&nbsp;±&nbsp;0.1 |
3 | TBA | He et al. 2019 | 54.5&nbsp;±&nbsp;12.1 | 75.0&nbsp;±&nbsp;0.9 | 62.9&nbsp;±&nbsp;5.9 | 58.3&nbsp;±&nbsp;6.1 | 75.9&nbsp;±&nbsp;4.3 | 21.0&nbsp;±&nbsp;4.2 | 3.2&nbsp;±&nbsp;0.3 | 21.4&nbsp;±&nbsp;7.8 | 28.1&nbsp;±&nbsp;2.0 |
4 | MONet | Burgess et al. 2019 | 49.4&nbsp;±&nbsp;3.6 | 78.6&nbsp;±&nbsp;1.8 | 74.2&nbsp;±&nbsp;1.7 | 35.7&nbsp;±&nbsp;0.8 | 66.7&nbsp;±&nbsp;0.7 | 13.6&nbsp;±&nbsp;1.0 | 19.7&nbsp;±&nbsp;0.6 | 17.2&nbsp;±&nbsp;3.1 | 22.2&nbsp;±&nbsp;2.2 |


### VOR

Rank | Model | Reference | MOTA &uarr; | MOTP &darr; | MD &uarr; | MT &uarr; | Match &uarr; | Miss &darr; | ID S. &uarr; | FPs &uarr; | MSE &uarr; |
:---:|:------:|:---------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
1 | ViMON | Weis et al. 2020 | **89.0&nbsp;±&nbsp;0.0** | **89.5&nbsp;±&nbsp;0.5** | **90.4&nbsp;±&nbsp;0.5** | **90.0&nbsp;±&nbsp;0.4** | **93.2&nbsp;±&nbsp;0.4** | **6.5&nbsp;±&nbsp;0.4** | **0.3&nbsp;±&nbsp;0.0** | **4.2&nbsp;±&nbsp;0.4** | 6.4&nbsp;±&nbsp;0.6 |
2 | OP3 | Veerapaneni et al. 2019 | 65.4&nbsp;±&nbsp;0.6 | 89.0&nbsp;±&nbsp;0.6 | 88.0&nbsp;±&nbsp;0.6 | 85.4&nbsp;±&nbsp;0.5 | 90.7&nbsp;±&nbsp;0.3 | 8.2&nbsp;±&nbsp;0.4 | 1.1&nbsp;±&nbsp;0.2 | 25.3&nbsp;±&nbsp;0.6 | **3.0&nbsp;±&nbsp;0.1** |
3 | MONet | Burgess et al. 2019 | 37.0&nbsp;±&nbsp;6.8 | 81.7&nbsp;±&nbsp;0.5 | 76.9&nbsp;±&nbsp;2.2 | 37.3&nbsp;±&nbsp;7.8 | 64.4&nbsp;±&nbsp;5.0 | 15.8&nbsp;±&nbsp;1.6 | 19.8&nbsp;±&nbsp;3.5 | 27.4&nbsp;±&nbsp;2.3 | 12.2&nbsp;±&nbsp;1.4 |


## Citation

If you use this repository in your research, please cite:
```
@misc{Weis2020,
    Author = {Marissa A. Weis and Kashyap Chitta and Yash Sharma and Wieland Brendel and Matthias Bethge and Andreas Geiger and Alexander S. Ecker},
    Title = {Unmasking the Inductive Biases of Unsupervised Object Representations for Video Sequences},
    Year = {2020},
    Eprint = {arXiv:2006.07034},
}
```
