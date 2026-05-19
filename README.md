# 🐠 Cyprus Fish Species Recognition

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX-005CED?style=for-the-badge&logo=onnx&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![Terraform](https://img.shields.io/badge/terraform-%235835CC.svg?style=for-the-badge&logo=terraform&logoColor=white)
![Microsoft Azure](https://img.shields.io/badge/microsoft%20azure-%230072C6.svg?style=for-the-badge&logo=microsoftazure&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/github%20actions-%232671E5.svg?style=for-the-badge&logo=githubactions&logoColor=white)
![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-yellow)

```diff
- This branch was used to learn how to deploy on Azure using Terraform and Docker. The server does not run anymore, the public version on  [Hugging Face Space](https://huggingface.co/spaces/JayRay5/Cyprus-Fish-Recognition-App) is the version on the main branch with a full-stack container app.
```
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
|       ├── api-docker-build.yml  # Build and push the Backend Container to GHCR
|       ├── deploy-frontend.yml   # Deploy the gradio interfance on Hugging Face Spaces
│       └── terraform-cd.yml      # Configure and Build the ACI (Azure Container Instances)
├── apps
│   ├── backend  
│   │   ├── api.py             # FastAPI 
│   │   ├── config.py          # Settings
|   |   ├── Dockerfile         # Docker file use to create the backend api container
│   │   └── utils.py
|   └── backend  
│       ├── app.py             # FastAPI 
│       ├── config.py          # Settings
|       ├── README.md          # README file for the configuration of HF Space (gradio template)
│       └── requirements.txt   # Libs to init the requirements.txt 
├── configs                    # Hydra config files for dataset, model, and training hyperparameters
├── data                       # Raw data
├── experiments                # Output directory for the Hugging Face 
├── infrastructure             # folder for terraform
|   └── main.tf                # terraform script that use azurem to configure the ACI container
├── scripts
|   ├── convert_to_onnx.py     # Convert HF Model to onnx (optimize for production)
|   ├── prepare_data.py        # Split raw data into train and test
|   └── upload_dataset.py      # Upload to the Hugging Face Hub
├── src
│   ├── __init__.py          
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
├── README.md                 
├── poetry.lock                
└── pyproject.toml
```        

The project follows a robust MLOps pipeline:
1. **Data**: As the number of samples is small (<60 per class), the dataset is split into a train and a test set. The resulting dataset is hosted on Hugging Face Hub ([dataset](https://huggingface.co/datasets/JayRay5/cyprus-fish-dataset)).
2.  **Model:** The model is based on [**ConvNeXt Tiny**](https://arxiv.org/pdf/2201.03545). The final model weights are hosted on Hugging Face Hub, while experiment tracking and versioning are managed via `MLflow`.
3.  **Training:** The training pipeline uses k-fold validation and then a full finetuning on the training set once the hyperparameters are fixed. The Fine-tuning uses `PyTorch` and `Hydra` for configuration management. The training pipeline is achieved using the Hugging Face Trainer. Experiments metrics are followed using `MLflow`<br>
The best version of the model is checked after each global training, and the best one among MLFlow and local experiments is pushed on [HuggingFace](https://huggingface.co/JayRay5/convnext-tiny-224-cyprus-fish-cls).
4.  **CI/CD:** GitHub Actions pipeline that runs tests (`pytest`), security checks, builds the Docker image, and pushes it to GHCR.
5.  **Deployment:** The Docker container is automatically deployed to a Hugging Face Space running a `FastAPI` backend with a `Gradio` UI.

## 🧰 Tech Stack

**Core & ML** <br>
![Python](https://img.shields.io/badge/python-3.11-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)

**MLOps & Config** <br>
![Hydra](https://img.shields.io/badge/Config-Hydra-89b8cd)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?logo=mlflow&logoColor=white)
![DagsHub](https://img.shields.io/badge/Track-DagsHub-blue)

**Serving**<br>
![FastAPI](https://img.shields.io/badge/FastAPI-005571?logo=fastapi)
![Gradio](https://img.shields.io/badge/Gradio-Demo-orange)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?logo=docker&logoColor=white)

**Quality Assurance (QA) & Environment**<br>
![Poetry](https://img.shields.io/badge/Poetry-%233B82F6.svg?logo=poetry&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?logo=github-actions&logoColor=white)
![Pytest](https://img.shields.io/badge/Tests-Pytest-0A9EDC?logo=pytest&logoColor=white)
![Ruff](https://img.shields.io/badge/Linter-Ruff-black)
![Bandit](https://img.shields.io/badge/Security-Bandit-black)
![Pre-commit](https://img.shields.io/badge/Pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)

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
You may need a HuggingFace token to push the best model to the hub and reload the space.<br>
Then, add this token into your cyprus-fish-env virtual environment by running the following command:
```bash
conda env config vars set HF_TOKEN="your_token"
```
To use ml-flow on [dagshub](https://dagshub.com/) to see experiment results on the web, you can do:
```bash
conda env config vars set MLFLOW_TRACKING_URI="https://dagshub.com/your_user_name/your_repo.mlflow"
conda env config vars set MLFLOW_TRACKING_USERNAME="your_user_name"
conda env config vars set MLFLOW_TRACKING_PASSWORD="your_dagshub_token"
```
Install poetry and libs
```bash
pip install poetry==2.2.1
poetry config virtualenvs.create false # install libs in the conda env
poetry install 
```
Install git hooks
```bash
poetry run pre-commit install
poetry run pre-commit install --hook-type pre-push
chmod +x .git/hooks/pre-push
```

Install Azure CLI
```
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

```

Install Terraform
```
# 1. Installe les prérequis pour gérer les clés de sécurité
sudo apt-get update && sudo apt-get install -y gnupg software-properties-common

# 2. Ajoute la clé GPG officielle de HashiCorp
wget -O- https://apt.releases.hashicorp.com/gpg | gpg --dearmor | sudo tee /usr/share/keyrings/hashicorp-archive-keyring.gpg > /dev/null

# 3. Ajoute le dépôt HashiCorp à tes sources d'applications
echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list

# 4. Mets à jour et installe Terraform
sudo apt-get update && sudo apt-get install -y terraform

```

Configure Terraform and build the ACI (Azure Container Instance)
```
cd infrastructure
terraform init
terraform plan
terraform apply 

```

Note there is one API_SECRET_TOKEN that is a secret stored in the Hugging Face Space and in the ACI to check if the user has the right to request the API.
There is also a HF_TOKEN stored as github secrets to allow the repo to push the gradio interface on the HF space during CD/CD pipeline.

## 🛠️ Data Preparation, 🧠 Model Training & 💻 Application

This project implements a complete pipeline from raw data processing to model registry, fully configurable via **Hydra**.


### 1. Data Preparation & Upload 

The raw images are split into a train and a test set and uploaded to the Hugging Face Hub as a `Dataset`. 

To run the data preparation pipeline:
```bash
# Split the data into train and test sets
poetry run python -m src.scripts.prepare_dataset

# Push the dataset to Hugging Face Hub
poetry run python -m src.scripts.upload_dataset
```

### 2. Training 
The training hyperparameters, model, and dataset can be set using hydra in the config folder. <br>
The experiments are saved in an experiments folder. The training scripts load metrics and best models using MLflow to ensure easy analysis of results across experiments.  <br>
As there is a small number of samples, the training pipeline is split into two stages:

#### 1: K-fold cross-validation
This step allows the validation of hyperparameters.
```bash
poetry run kfold_training
```

#### 2: Global training
Once hyperparameters are validated, run the following script to train on the full dataset:
```bash
poetry run training
```
The script evaluates the model on the test set and compares it against the current best recorded performance. If the new model outperforms the previous one, it is automatically:

1. Logged to DagsHub (MLflow) for versioning.

2. Pushed to the Hugging Face Hub (if enabled in Hydra) for storage.

3. Deployed by restarting the Hugging Face Space.

### 3. Application 
You can change the model used in the application config in apps/backend/configs. <br>
To start the server, run:
```bash
poetry run run-back
```

To start the gradio interface, run:
```bash
poetry run run-front
```

## To Do
- [ ] Data shift detection integration 
- [ ] Add a link to download raw data in the data folder
