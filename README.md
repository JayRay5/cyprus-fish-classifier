# 🐠 Cyprus Fish Recognition

[![CI/CD Pipeline](https://github.com/ton-username/ton-repo-name/actions/workflows/docker-image.yml/badge.svg)](https://github.com/JayRay5/reconnaissance_poisson_chypre/actions)
[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-yellow)](https://huggingface.co/spaces/JayRay5/Cyprus-Fish-Recognition-App)
[![Docker Image](https://img.shields.io/badge/docker-ghcr.io-blue)](https://github.com/ton-username/ton-repo-name/pkgs/container/reconnaissance_poisson_chypre)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

An end-to-end MLOps project for classifying fish species from Cyprus using Deep Learning.
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

The project follows a robust MLOps pipeline:

1.  **Data & Model:** Hosted and versioned on Hugging Face Hub ([dataset](https://huggingface.co/datasets/JayRay5/cyprus-fish-dataset), [model](https://huggingface.co/JayRay5/convnext-tiny-224-cyprus-fish-cls)).
2.  **Training:** Fine-tuning of a **ConvNext Tiny** model using `PyTorch` and `Hydra` for configuration management. The training pipeline is achieved using the Hugging Face Trainer.
3.  **CI/CD:** GitHub Actions pipeline that runs tests (`pytest`), security checks, builds the Docker image, and pushes it to GHCR.
4.  **Deployment:** The Docker container is automatically deployed to a Hugging Face Space running a `FastAPI` backend with a `Gradio` UI.

## 🛠️ Tech Stack

- **Core:** Python 3.11, PyTorch, Transformers (Hugging Face)
- **Package Management:** Poetry, Conda
- **Configuration:** Hydra
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

### 1. Environment Setup

Create the Conda environment and activate it:
```bash
conda create -n cyprus-fish-env python=3.11
conda activate cyprus-fish-env
```
  
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
