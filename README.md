# 🐠 Cyprus Fish Recognition

[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-yellow)](https://huggingface.co/spaces/JayRay5/Cyprus-Fish-Recognition-App)
[![Docker Image](https://img.shields.io/badge/docker-ghcr.io-blue)](https://github.com/ton-username/ton-repo-name/pkgs/container/reconnaissance_poisson_chypre)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

An end-to-end MLOps project for classifying fish species from Cyprus using Deep Learning. <br>
This project covers 5 species:
<div align="center">
  <h3>🐟 Species overview </h3>
  <table>
    <tr>
      <td align="center">
        <img src="https://huggingface.co/datasets/JayRay5/cyprus-fish-dataset/resolve/main/preview_apogon_imberbis.jpg" width="150" height="150" style="object-fit: cover;"/>
        <br>
        <b>Apogon Imberbis</b>
      </td>
      <td align="center">
        <img src="https://huggingface.co/datasets/JayRay5/cyprus-fish-dataset/resolve/main/preview_epinephelus_marginatus.jpg" width="150" height="150" style="object-fit: cover;"/>
        <br>
        <b>Epinephelus Marginatus</b>
      </td>
      <td align="center">
        <img src="https://huggingface.co/datasets/JayRay5/cyprus-fish-dataset/resolve/main/preview_pempheris_vanicolensis.jpg" width="150" height="150" style="object-fit: cover;"/>
        <br>
        <b>Pempheris Vanicolensis</b>
      </td>
      <td align="center">
        <img src="https://huggingface.co/datasets/JayRay5/cyprus-fish-dataset/resolve/main/preview_sparisoma_cretense.jpg" width="150" height="150" style="object-fit: cover;"/>
        <br>
        <b>Sparisoma Cretense</b>
      </td>
      <td align="center">
        <img src="https://huggingface.co/datasets/JayRay5/cyprus-fish-dataset/resolve/main/preview_thalassoma_pavo.jpg" width="150" height="150" style="object-fit: cover;"/>
        <br>
        <b>Thalassoma Pavo</b>
      </td>
    </tr>
  </table>
</div>

This repository contains the complete pipeline: from data preparation and model training to containerized deployment on Hugging Face Spaces.

**[👉 Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/JayRay5/Cyprus-Fish-Recognition-App)**

---

## 🏗️ Architecture & Workflow
```bash
.
├── .github
│   └── workflows
|       ├── push.yaml          # Check security (bandit), format (ruff), test (pytest), and deploy at each push on main
│       └── test_docker.yaml   # Build the image of the App and deploy it to Hugging Face Space
├── configs                    # Hydra config files for dataset, model, and training hyperparameters
├── data                       # Raw data 
├── scripts
|   ├── prepare_data.py        # Split raw data into train and test
|   └── upload_dataset.py      # Upload to the Hugging Face Hub
├── src
│   ├── __init__.py
│   ├── app
│   │   ├── __init__.py
│   │   ├── api.py             # FastAPI 
│   │   ├── config.py          # Settings
|   |   ├── start.sh           # Script to start the app
│   │   ├── ui.py              # Gradio Interface  
│   │   └── utils.py           
│   └── cyprus_fish
│       ├── __init__.py
│       ├── data.py            # Data Loader
|       ├── train.py           # Training scripts (k-fold and global)
|       └── utils.py
|
├── tests                      # Unit Tests
├── .dockerignore              
├── .gitignore
├── .pre-commit-config.yaml    # Git hooks          
├── Dockerfile                 
├── README.md                 
├── poetry.lock                
└── pyproject.toml
```        

The project follows a robust MLOps pipeline:
1. **Data**: As the number of samples is small (<60 per class), the dataset is split into a train and a test set. The resulting dataset is hosted on Hugging Face Hub ([dataset](https://huggingface.co/datasets/JayRay5/cyprus-fish-dataset).
2.  **Model:** The model is based on [**ConvNext Tiny**](https://arxiv.org/pdf/2201.03545). It is hosted and versioned on Hugging Face Hub.
3.  **Training:** The training pipeline uses k-fold validation and then a full finetuning on the training set once the hyperparameters are fixed. The Fine-tuning uses `PyTorch` and `Hydra` for configuration management. The training pipeline is achieved using the Hugging Face Trainer. <br>
The best version of the model is checked after each training, and the best one is pushed on [HuggingFace](https://huggingface.co/JayRay5/convnext-tiny-224-cyprus-fish-cls).
4.  **CI/CD:** GitHub Actions pipeline that runs tests (`pytest`), security checks, builds the Docker image, and pushes it to GHCR.
5.  **Deployment:** The Docker container is automatically deployed to a Hugging Face Space running a `FastAPI` backend with a `Gradio` UI.

## 🛠️ Tech Stack

- **Core:** Python 3.11, PyTorch, Transformers (Hugging Face), Datasets (Hugging Face)
- **Package Management:** Poetry, Conda
- **Configuration using Hydra:** In the config folder, you can use another dataset from Hugging Face hub, change the model backbone, and set up training hyperparameters
- **Serving:** FastAPI, Uvicorn, Gradio, Docker
- **Quality & Security:**
    - `Ruff` (Linting & Formatting)
    - `Bandit` (Security analysis)
    - `Pytest` (Unit testing)
    - `Pre-commit` (Git hooks)

---

## 🚀 Installation & Setup

### Prerequisites
- Conda (Anaconda or Miniconda)
- Git

### Environment Setup
Install dependencies
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

Enable the start.sh file for the launch of the application
```bash
chmod +x .src/app/start.sh
```

## 🧠 Data Preparation, Model Training & Application

This project implements a complete pipeline from raw data processing to model registry, fully configurable via **Hydra**.


### 1. Data Preparation & Upload 📦

The raw images are split into a train and a test set and uploaded to the Hugging Face Hub as a `Dataset`. 

To run the data preparation pipeline:
```bash
# Split the data into train and test sets
poetry run python -m src.scripts.prepare_dataset

# Push the dataset to Hugging Face Hub
poetry run python -m src.scripts.upload_dataset
```

### 2. Training 
The training hyperparameters can be set using hydra in the config folder. <br>
The experiments are saved in an experiments folder. <br>
As there is a small number of samples, the training pipeline is split into two stages:

#### 1: K-fold validation
This step allows the validation of hyperparameters.
```bash
poetry run kfold_training
```

#### 2: Global training
Once the hyperparameters are validated, you can run the following script to start the training on the whole training set.
```bash
poetry run training
```
The script will also evaluate the model on the test set. If there is another model already trained, it will compare the results with its own. If the new model is better, it will be pushed to the Hugging Face Hub according to the hydra config. 


### 3. Application 
You can change the model used in the application config in src/app/configs. <br>
To start the server, run:
```bash
./src/app/start.sh
```



