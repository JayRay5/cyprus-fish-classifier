# 🐠 Cyprus Fish Recognition App


## Stack & Tech
- Poetry (lib and dependencies management)
- Hydra
- Ruff (format check)
- Bandit (Security check)
- Pytest (Unit Test)

  
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
chmod +x .git/hooks/pre-push
```
