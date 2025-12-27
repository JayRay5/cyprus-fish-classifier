# Cyprus Fish Recognition App


## Installation & Setup
1- Install dependencies
```bash
conda create -n cyprus-fish-env python=3.11.5
conda activate cyprus-fish-env
```
Install poetry and libs
```bash
pip install poetry

poetry config virtualenvs.in-project true --local # virtual env in the project
poetry env use $(which python)

poetry install # install libs in the venv
```
Install git hooks
```bash
poetry run pre-commit install
```
Create the .env file for the secret variables
```bash
touch .env
```
