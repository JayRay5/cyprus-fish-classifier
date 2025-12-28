name: CI/CD Pipeline - Test

on:
  push:
    branches: [ "main" ]
  pull_request:
    branches: [ "main" ]
  workflow_dispatch:

env:
  REGISTRY: ghcr.io

jobs:
  # ====================================================
  # JOB 1 : TESTS 
  # ====================================================
  test:
    runs-on: ubuntu-latest
    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Set up Python 3.12
      uses: actions/setup-python@v5
      with:
        python-version: "3.12"

    - name: Install Poetry
        uses: snok/install-poetry@v1
        with:
          virtualenvs-create: true
          virtualenvs-in-project: true

    - name: Install dependencies
        if: steps.cached-poetry-dependencies.outputs.cache-hit != 'true'
        run: poetry install --no-interaction --no-root

    - name: Security Scan with Bandit
      run: bandit -r src/ -ll -ii

    - name: Run Unit Tests 
      env:
        HF_TOKEN: ${{ secrets.HF_TOKEN }} 
      run: pytest -v 

   