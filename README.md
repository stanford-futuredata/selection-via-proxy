# Selection via Proxy: Efficient Data Selection for Deep Learning
This repository contains a refactored implementation of ["Selection via Proxy: Efficient Data Selection for Deep Learning"](https://openreview.net/forum?id=HJg2b0VYDr) from ICLR 2020.

If you use this code in your research, please use the following BibTeX entry.

```
@inproceedings{
    coleman2020selection,
    title={Selection via Proxy: Efficient Data Selection for Deep Learning},
    author={Cody Coleman and Christopher Yeh and Stephen Mussmann and Baharan Mirzasoleiman and Peter Bailis and Percy Liang and Jure Leskovec and Matei Zaharia},
    booktitle={International Conference on Learning Representations},
    year={2020},
    url={https://openreview.net/forum?id=HJg2b0VYDr}
}
```

The original code is also available as a [zip file](https://drive.google.com/open?id=1Lb8LMHhHJpwaySynjBx7aFeCP_csGJc5), but lacks documentation, uses outdated packages, and won't be maintained.
Please use this repository instead and report issues here.

## Setup

### Prerequisites

- Install [anaconda3-5.3.1](https://www.anaconda.com/) or greater
- Install [Pytorch](https://pytorch.org/) 1.4 or greater

### Installation
```bash
git clone https://github.com/stanford-futuredata/selection-via-proxy.git
cd selection-via-proxy
pip install -e .
```

or simply

```
pip install git+https://github.com/stanford-futuredata/selection-via-proxy.git
```

## Quickstart
Perform active learning on CIFAR10 from the command line:

```bash
python -m svp.cifar active
```

Or from the python interpreter:

```python
from svp.cifar.active import active
active()
```

"Selection via proxy" happens when `--proxy-arch` doesn't match `--arch`:

```bash
# ResNet20 selecting data for a ResNet164
python -m svp.cifar active --proxy-arch preact20 --arch preact164
```

For help, see `python -m svp.cifar active --help` or `active()`'s docstrinng.

## Example Usage

Below are more examples of the command line interface that cover different datasets (e.g., CIFAR100, ImageNet, Amazon Review Polarity) and commands (e.g., `train`, `coreset`).

### Basic Training
#### CIFAR10 and CIFAR100

##### Preliminaries

None. The CIFAR10 and CIFAR100 datasets will download if they don't exist in `./data/cifar10` and `./data/cifar100` respectively.

##### Examples

```bash
# Train ResNet164 with pre-activation (https://arxiv.org/abs/1603.05027) on CIFAR10.
python -m svp.cifar train --dataset cifar10 --arch preact164
```

Replace `--dataset CIFAR10` with `--dataset CIFAR100` to run on CIFAR100 rather than CIFAR10.

```bash
# Train ResNet164 with pre-activation (https://arxiv.org/abs/1603.05027) on CIFAR100.
python -m svp.cifar train --dataset cifar100 --arch preact164
```

The same is true for all the `python -m svp.cifar` commands below

#### ImageNet

##### Preliminaries
- Download the ImageNet dataset into a directory called `imagenet`.
- Extract the images.
```bash
# Extract train data.
mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
# Extract validation data.
cd ../ && mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
```
- Replace `/path/to/data` in all the `python -m svp.imagenet` commands below with the path to the `imagenet` directory you created. Note, do not include `imagenet` in the path; the script will automatically do that.

##### Examples

```bash
# Train ResNet50 (https://arxiv.org/abs/1512.03385).
python -m svp.imagenet train --dataset-dir '/path/to/data' --arch resnet50 --num-workers 20
```

For convenience, you can use larger batch sizes and scale learning rates according to ["Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour"](https://arxiv.org/abs/1706.02677) with `--scale-learning-rates`:

```bash
# Train ResNet50 with a batch size of 1048 and scaled learning rates accordingly.
python -m svp.imagenet train --dataset-dir '/path/to/data' --arch resnet50 --num-workers 20 \
    --batch-size 1048 --scale-learning-rates
```

[Mixed precision training](https://arxiv.org/abs/1710.03740) is also supported using [apex](https://github.com/NVIDIA/apex). Apex isn't installed during the pip install instructions above, so please follow the installation instructions in the [apex repository](https://github.com/NVIDIA/apex) before running the command below.
```bash
# Use mixed precision training to train ResNet50 with a batch size of 1048 and scale learning rates accordingly.
python -m svp.imagenet train --dataset-dir '/path/to/data' --arch resnet50 --num-workers 20 \
    --batch-size 1048 --scale-learning-rates --fp16
```

#### Amazon Review Polarity and Full


##### Preliminaries
- [Download the Amazon Review Polarity and Full datasets](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M) (i.e., `amazon_review_full_csv.tar.gz` and `amazon_review_polarity_csv.tar.gz`).
- Extract the CSV files.
```bash
tar -xvzf amazon_review_full_csv.tar.gz
tar -xvzf amazon_review_polarity_csv.tar.gz
```
- Replace `/path/to/data` in all the `python -m svp.amazon` commands below with the path to the root directory you created. Note, do not include `amazon_review_full_csv` or `amazon_review_polarity_csv` in the path; the script will automatically do that.

##### Examples

```bash
# Train VDCNN29 (https://arxiv.org/abs/1606.01781) on Amazon Review Polarity.
python -m svp.amazon train --datasets-dir '/path/to/data' --dataset amazon_review_polarity --arch vdcnn29-conv \
    --num-workers 4 --eval-num-workers 8
```

Replace `--dataset amazon_review_polarity` with `--dataset amazon_review_full` to run on Amazon Review Full rather than Amazon Review Polarity.

```bash
# Train VDCNN29 (https://arxiv.org/abs/1606.01781) on Amazon Review Full.
python -m svp.amazon train --datasets-dir '/path/to/data' --dataset amazon_review_full --arch vdcnn29-maxpool \
    --num-workers 4 --eval-num-workers 8
```
The same is true for all the `python -m svp.amazon` commands below

### Active learning 

Active learning selects points to label from a large pool of unlabeled data by repeatedly training a model on a small pool of labeled data and selecting additional examples to label based on the model’s uncertainty (e.g., the entropy of predicted class probabilities) or other heuristics.
The commands below demonstrate how to perform active learning on CIFAR10, CIFAR100, ImageNet, Amazon Review Polarity and Amazon Review Full with a variety of models and selection methods.


#### CIFAR10 and CIFAR100

##### Baseline Approach

```bash
# Perform active learning with ResNet164 for both selection and the final predictions.
python -m svp.cifar active --dataset cifar10 --arch preact164 --num-workers 4 \
	--selection-method least_confidence \
	--initial-subset 1000 \
	--round 4000 \
	--round 5000 \
	--round 5000 \
	--round 5000 \
	--round 5000
```


##### Selection via Proxy

If the model architectures (`arch` vs `proxy_arch`) or the learning rate schedules don't match, "selection via proxy" (SVP) is performed and two separate models are trained.
The proxy is used for selecting which examples to label, while the target is only used for evaluating the quality of the selection.
By default, the target model (`arch`) is trained and evaluated after each selection round.
To change this behavior set `eval_target_at` to evaluate at a specific labeling budget(s) or set `train_target` to False to skip evaluating the target model.

```bash
# Perform active learning with ResNet20 for selection and ResNet164 for the final predictions.
python -m svp.cifar active --dataset cifar10 --arch preact164 --num-workers 4 \
	--selection-method least_confidence --proxy-arch preact20 \
	--initial-subset 1000 \
	--round 4000 \
	--round 5000 \
	--round 5000 \
	--round 5000 \
	--round 5000 \
	--eval-target-at 25000
```

To train the proxy for fewer epochs, use the `--proxy-*` options as shown below:

```bash
# Perform active learning with ResNet20 after only 50 epochs for selection.
python -m svp.cifar active --dataset cifar10 --arch preact164 --num-workers 4 \
	--selection-method least_confidence --proxy-arch preact20 \
	--proxy-learning-rate 0.01 --proxy-epochs 1 \
	--proxy-learning-rate 0.1 --proxy-epochs 45 \
	--proxy-learning-rate 0.01 --proxy-epochs 4 \
	--initial-subset 1000 \
	--round 4000 \
	--round 5000 \
	--round 5000 \
	--round 5000 \
	--round 5000 \
	--eval-target-at 25000
```

#### ImageNet

##### Baseline Approach
```bash
# Perform active learning with ResNet50 for both selection and the final predictions.
python -m svp.imagenet active --datasets-dir '/path/to/data' --arch resnet50 --num-workers 20
```

##### Selection via Proxy

If the model architectures (`arch` vs `proxy_arch`) or the learning rate schedules don't match, "selection via proxy" (SVP) is performed and two separate models are trained.
The proxy is used for selecting which examples to label, while the target is only used for evaluating the quality of the selection.
By default, the target model (`arch`) is trained and evaluated after each selection round.
To change this behavior set `eval_target_at` to evaluate at a specific labeling budget(s) or set `train_target` to False to skip evaluating the target model.

```bash
# Perform active learning with ResNet18 for selection and ResNet50 for the final predictions.
python -m svp.imagenet active --datasets-dir '/path/to/data' --arch resnet50 --num-workers 20 \
    --proxy-arch resnet18 --proxy-batch-size 1028 --proxy-scale-learning-rates \
    --eval-target-at 512467
```

To train the proxy for fewer epochs, use the `--proxy-*` options as shown below:
```bash
# Perform active learning with ResNet18 after only 45 epochs for selection.
python -m svp.imagenet active --datasets-dir '/path/to/data' --arch resnet50 --num-workers 20 \
    --proxy-arch resnet18 --proxy-batch-size 1028 --proxy-scale-learning-rates \
    --eval-target-at 512467 \
    --proxy-learning-rate 0.0167 --proxy-epochs 1 \
    --proxy-learning-rate 0.0333 --proxy-epochs 1 \
    --proxy-learning-rate 0.05 --proxy-epochs 1 \
    --proxy-learning-rate 0.0667 --proxy-epochs 1 \
    --proxy-learning-rate 0.0833 --proxy-epochs 1 \
    --proxy-learning-rate 0.1 --proxy-epochs 25 \
    --proxy-learning-rate 0.01 --proxy-epochs 15
```

#### Amazon Review Polarity and Full

##### Baseline Approach

```bash
# Perform active learning with VDCNN29 for both selection and the final predictions.
python -m svp.amazon active --datasets-dir '/path/to/data' --dataset amazon_review_polarity  --num-workers 8 \
    --arch vdcnn29-conv --selection-method least_confidence
```

##### Selection via Proxy

If the model architectures (`arch` vs `proxy_arch`) or the learning rate schedules don't match, "selection via proxy" (SVP) is performed and two separate models are trained.
The proxy is used for selecting which examples to label, while the target is only used for evaluating the quality of the selection.
By default, the target model (`arch`) is trained and evaluated after each selection round.
To change this behavior set `eval_target_at` to evaluate at a specific labeling budget(s) or set `train_target` to False to skip evaluating the target model.
You can evaluate a series of selections later using the `precomputed_selection` option.

```bash
# Perform active learning with VDCNN9 for selection and VDCNN29 for the final predictions.
python -m svp.amazon active --datasets-dir '/path/to/data' --dataset amazon_review_polarity --num-workers 8 \
    --arch vdcnn29-conv --selection-method least_confidence \
    --proxy-arch vdcnn9-maxpool --eval-target-at 1440000
```

To use fastText as a proxy, [Install fastText 0.1.0](https://github.com/facebookresearch/fastText/releases/tag/v0.1.0) and replace `/path/to/fastText/fasttext` in the `python -m svp.amazon fasttext` commands below with the path to the fastText binary you created.

```bash
# For convenience, save fastText results in a separate directory
mkdir fasttext
# Perform active learning with fastText.
python -m svp.amazon fasttext '/path/to/fastText/fasttext' --run-dir fasttext \
    --datasets-dir '/path/to/data' --dataset amazon_review_polarity --selection-method least_confidence \
    --size 72000 --size 360000 --size 720000 --size 1080000 --size 1440000
# Get the most recent timestamp from the fasttext directory.
fasttext_path="fasttext/$(ls fasttext | sort -nr | head -n 1)"
# Use selected labeled data from fastText to train VDCNN29
python -m svp.amazon active --datasets-dir '/path/to/data' --dataset amazon_review_polarity --num-workers 8 \
    --arch vdcnn29-conv --selection-method least_confidence \
    --precomputed-selection $fasttext_path --eval-target-at 1440000
```

### Core-set Selection
Core-set selection techniques start with a large labeled or unlabeled dataset and aim to find a small subset that accurately approximates the full dataset by selecting representative examples.
The commands below demonstrate how to perform core-set selection on CIFAR10, CIFAR100, ImageNet, Amazon Review Polarity and Amazon Review Full with a variety of models and selection methods.

#### CIFAR10 and CIFAR100

##### Baseline Approach
```bash
# Perform core-set selection with an oracle that uses ResNet164 for both selection and the final predictions.
python -m svp.cifar coreset --dataset cifar10 --arch preact164 --num-workers 4 \
    --subset 25000 --selection-method forgetting_events
```

##### Selection via Proxy
```bash
# Perform core-set selection with ResNet20 selecting for ResNet164.
python -m svp.cifar coreset --dataset cifar10 --arch preact164 --num-workers 4 \
    --subset 25000 --selection-method forgetting_events \
    --proxy-arch preact20
```

To train the proxy for fewer epochs, use the `--proxy-*` options as shown below:

```bash
# Perform core-set selection with ResNet20 after only 50 epochs.
python -m svp.cifar coreset --dataset cifar10 --arch preact164 --num-workers 4 \
    --subset 25000 --selection-method forgetting_events \
    --proxy-arch preact20 \
	--proxy-learning-rate 0.01 --proxy-epochs 1 \
	--proxy-learning-rate 0.1 --proxy-epochs 45 \
	--proxy-learning-rate 0.01 --proxy-epochs 4
```

#### ImageNet

##### Baseline Approach
```bash
# Perform core-set selection with an oracle that uses ResNet50 for both selection and the final predictions.
python -m svp.imagenet coreset --datasets-dir '/path/to/data' --arch resnet50 --num-workers 20 \
    --subset 768700 --selection-method forgetting_events
```

##### Selection via Proxy

```bash
# Perform core-set selection with ResNet18 selecting for ResNet50.
python -m svp.imagenet coreset --datasets-dir '/path/to/data' --arch resnet50 --num-workers 20 \
    --subset 768700 --selection-method forgetting_events \
    --proxy-arch resnet18 --proxy-batch-size 1028 --proxy-scale-learning-rates
```


#### Amazon Review Polarity and Full

##### Baseline Approach

```bash
# Perform core-set selection with an oracle that uses VDCNN29 for both selection and the final predictions.
python -m svp.amazon coreset --datasets-dir '/path/to/data' --dataset amazon_review_polarity --num-workers 8 \
    --arch vdcnn29-conv --subset 2160000  --selection-method entropy
```

##### Selection via Proxy

```bash
# Perform core-set selection with VDCNN9 selecting for VDCNN29.
python -m svp.amazon coreset --datasets-dir '/path/to/data' --dataset amazon_review_polarity --num-workers 8 \
    --arch vdcnn29-conv --subset 2160000 --selection-method entropy \
    --proxy-arch vdcnn9-maxpool
```

To use fastText as a proxy, [Install fastText 0.1.0](https://github.com/facebookresearch/fastText/releases/tag/v0.1.0) and replace `/path/to/fastText/fasttext` in the `python -m svp.amazon fasttext` commands below with the path to the fastText binary you created.

```bash
# For convenience, save fastText results in a separate directory
mkdir fasttext
# Perform core-set selection with fastText.
python -m svp.amazon fasttext '/path/to/fastText/fasttext' --run-dir fasttext \
    --datasets-dir '/path/to/data' --dataset amazon_review_polarity \
    --selection-method entropy --size 3600000 --size 2160000
# Get the most recent timestamp from the fasttext directory.
fasttext_path="fasttext/$(ls fasttext | sort -nr | head -n 1)"
# Use selected labeled data from fastText to train VDCNN29
python -m svp.amazon coreset --datasets-dir '/path/to/data' --dataset amazon_review_polarity --num-workers 8 \
    --arch vdcnn29-conv --precomputed-selection $fasttext_path
```
