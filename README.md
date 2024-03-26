# Joint Global-Local Alignment for Domain Adaptive Semantic Segmentation

## Paper
<p align="center">
<img src="https://github.com/skrya/skrya.github.io/blob/master/images/local-global.png" width="600">
</p>

[# Joint Global-Local Alignment for Domain Adaptive Semantic Segmentation
](https://ieeexplore.ieee.org/abstract/document/9746274)  
 [Sudhir Yarram](https://skrya.github.io/),  [Ming Yang], [Junsong Yuan], [Chunming Qiao]  

 International Conference on Acoustics, Speech, and Signal Processing (ICASSP), 2022

If you find this code useful for your research, please cite our [paper](https://ieeexplore.ieee.org/abstract/document/9746274):

```
@inproceedings{yarram2022joint,
  title={Joint global-local alignment for domain adaptive semantic segmentation},
  author={Yarram, Sudhir and Yang, Ming and Yuan, Junsong and Qiao, Chunming},
  booktitle={ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={3768--3772},
  year={2022},
  organization={IEEE}
}
```

## Abstract
Unsupervised domain adaptation has shown promising results
in leveraging synthetic (source) images for semantic segmentation of real (target) images. One key issue is how to align
data distributions between the source and target domains. Adversarial learning has been applied to align these distributions.
However, most existing approaches focus on aligning the output distributions related to image (global) segmentation. Such
global alignment may not result in effective alignment due to
the inherent high dimensionality feature space involved in the
alignment. Moreover, global alignment might be hindered by
the noisy outputs corresponding to background pixels in the
source domain. To address this limitation, we propose a local
output alignment. Such an approach can also mitigate the influences of noisy background pixels from the source domain
when performing the local alignment. Our experiments show
that by adding local output alignment into various global
alignment based domain adaptation, our joint global-local
alignment methods improves semantic segmentation


## Preparation

### Pre-requisites
* Python 3.7
* Pytorch >= 0.4.1
* CUDA 9.0 or higher

### Installation
0. Clone the repo:
```bash
$ git clone https://github.com/skrya/globallocal
$ cd globallocal
```

1. Install OpenCV if you don't already have it:

```bash
$ conda install -c menpo opencv
```

2. Install this repository and the dependencies using pip:
```bash
$ pip install -e <root_dir>
```

With this, you can edit the globallocal code on the fly and import function 
and classes of globallocal in other project as well.

3. Optional. To uninstall this package, run:
```bash
$ pip uninstall globallocal
```

### Datasets
By default, the datasets are put in ```<root_dir>/data```. We use symlinks to hook the ADVENT codebase to the datasets. An alternative option is to explicitlly specify the parameters ```DATA_DIRECTORY_SOURCE``` and ```DATA_DIRECTORY_TARGET``` in YML configuration files.

* **GTA5**: Please follow the instructions [here](https://download.visinf.tu-darmstadt.de/data/from_games/) to download images and semantic segmentation annotations. The GTA5 dataset directory should have this basic structure:
```bash
<root_dir>/data/GTA5/                               % GTA dataset root
<root_dir>/data/GTA5/images/                        % GTA images
<root_dir>/data/GTA5/labels/                        % Semantic segmentation labels
...
```

* **Cityscapes**: Please follow the instructions in [Cityscape](https://www.cityscapes-dataset.com/) to download the images and validation ground-truths. The Cityscapes dataset directory should have this basic structure:
```bash
<root_dir>/data/Cityscapes/                         % Cityscapes dataset root
<root_dir>/data/Cityscapes/leftImg8bit              % Cityscapes images
<root_dir>/data/Cityscapes/leftImg8bit/val
<root_dir>/data/Cityscapes/gtFine                   % Semantic segmentation labels
<root_dir>/data/Cityscapes/gtFine/val
...
```

### Pre-trained models
Pre-trained models can be downloaded [here](https://buffalo.box.com/s/wpfdyudxltqujrj1p2ymlz1kg0ndbz1u) and put in ```<root_dir>/pretrained_models```

## Running the code
For evaluation, execute:
```bash
$ cd <root_dir>/globallocal/scripts
$ python test.py --cfg ./configs/advent_global_local.yml
```

### Training
For the experiments done in the paper, we used pytorch 0.4.1 and CUDA 9.0. To ensure reproduction, the random seed has been fixed in the code. Still, you may need to train a few times to reach the comparable performance.

By default, logs and snapshots are stored in ```<root_dir>/experiments``` with this structure:
```bash
<root_dir>/experiments/logs
<root_dir>/experiments/snapshots
```

To train :
```bash
$ cd <root_dir>/globallocal/scripts
$ python train.py --cfg ./configs/advent_global_local.yml
```


## Acknowledgements
This codebase is heavily borrowed from [ADVENT](https://github.com/valeoai/ADVENT.git) and [Pytorch-Deeplab](https://github.com/speedinghzl/Pytorch-Deeplab).

## License
Globallocal is released under the [Apache 2.0 license](./LICENSE).
