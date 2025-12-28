# 🐠 Cyprus Fish Recognition App


## Installation & Setup
1- Install dependencies
```bash
conda create -n cyprus-fish-env python=3.11.5
conda activate cyprus-fish-env
```
Install poetry and libs
```bash
pip install poetry
poetry config virtualenvs.create false # install libs in the conda env
poetry install 
```
Install git hooks
```bash
poetry run pre-commit install
poetry run pre-commit install --hook-type pre-push
```
