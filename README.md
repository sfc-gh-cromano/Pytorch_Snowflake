______________________________________________________________________
<div align="center">

# **tfh-train**

</div>

______________________________________________________________________

## Contents

- [Installation](#installation)
- [How to train and evaluate models?](#how-to-train-and-evaluate-models)
- [Setup for development](#setup-for-development)


## Installation

```bash
cd tfh-train

pip install .
```

> **Note**: If you want to modify the code base while experimenting, install a package in the `editable` mode adding `-e` flag.


## How to train and evaluate models?

To start training
```bash
tfh-train experiment=cifar_clf/cifar_clf_training
```

To evaluate trained models
```bash
tfh-evaluate experiment=cifar_clf/cifar_clf_training ckpt_path=/path/to/model.ckpt
```

## Setup for development
### `pip`

```bash
# Go to the repo directory
cd tfh-train

# (Optional) Create virtual environment
python -m venv venv
source ./venv/bin/activate

# Install project in editable mode
pip install -e .[dev]

# Install git hooks to preserve code format consistency
pre-commit install
nb-clean add-filter --remove-empty-cells
```

### `conda`

```bash
# Go to the repo directory
cd tfh-train

# Create and activate conda environment
conda env create -f ./conda/environment_dev.yml
conda activate tfh_train_dev

# Install git hooks to preserve code format consistency
pre-commit install
nb-clean add-filter --remove-empty-cells
```
