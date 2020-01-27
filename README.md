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

## Example Usage

<p align="center">
  <img width="800" src="https://github.com/codyaustun/svp-tmp/blob/master/images/svp_overview_al.png">
</p>

In active learning, we followed the same iterative procedure of training and selecting points to label as traditional approaches but replaced the target model with a cheaper-to-compute proxy model, as shown in the figure above.

For example, traditional active learning would use the same model (e.g., [ResNet164 with pre-activation](https://arxiv.org/abs/1603.05027)) for selection as well as the final predictions, as shown below:

```bash
# Active learning with ResNet164 for both selection and the final predictions.
python -m svp.cifar.active --dataset cifar10 --num-workers 4 \
	--proxy-arch preact164 --arch preact164 \
	--selection-method least_confidence \
	--initial-subset 1000 \
	--round 4000 \
	--round 5000 \
	--round 5000 \
	--round 5000 \
	--round 5000 \
	--eval-target-at 25000
```

Using the idea of selection via proxy (SVP), we could use a less accurate but much cheaper model like ResNet20 for selection and only train ResNet164 once we have reached our budget (25,000 images in the command below).

```bash
# Active learning with ResNet20 for selection and ResNet164 for the final predictions.
python -m svp.cifar.active --dataset cifar10 --num-workers 4 \
	--proxy-arch preact20 --arch preact164 \
	--selection-method least_confidence \
	--initial-subset 1000 \
	--round 4000 \
	--round 5000 \
	--round 5000 \
	--round 5000 \
	--round 5000 \
	--eval-target-at 25000
```

Despite ResNet20 having a much higher error rate than ResNet164, we do not see a significant impact in the final predictions of ResNet164 from this substitution (i.e., 5.3+/-0.1 vs. 5.4+/-0.4 top-1 error).
Compared to the initial experiment, the time to select which images to labeled (i.e., data selection runtime) will be approximately 7.2x faster using ResNet20 rather than ResNet164, leading to a 2.5x speed-up in total runtime on a Titan V GPU.

If we can tolerate a slight increase in final error, we can achieve even greater speed-ups by only training ResNet20 for a few epochs rather than the full training schedule (50 epochs rather than 181 in the example below).

```bash
# Active learning with ResNet20 after only 50 epochs for selection.
python -m svp.cifar.active --dataset cifar10 --num-workers 4 \
	--proxy-arch preact20 --arch preact164 \
	--proxy-learning-rate 0.01 --proxy-epochs 1 \
	--proxy-learning-rate 0.1 --proxy-epochs 45 \
	--proxy-learning-rate 0.01 --proxy-epochs 4 \
	--selection-method least_confidence \
	--initial-subset 1000 \
	--round 4000 \
	--round 5000 \
	--round 5000 \
	--round 5000 \
	--round 5000 \
	--eval-target-at 25000
```

The error increases from 5.3+/-0.1 to 5.9+/-0.3 top-1, but data selection is 24.6x faster than the traditional approach, leading to a 2.9x speed-up in total runtime.
If we made more selections or considered a larger pool of unlabeled data, which is more typical in large-scale production settings, the total runtime speed-up would be much closer to the data selection runtime.
