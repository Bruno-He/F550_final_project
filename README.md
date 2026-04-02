# F550 Final Project

## Overview

This project is a notebook-based workflow for financial text analysis and valuation. It consists of three main stages:

1. preprocessing raw financial text data  
2. generating sentiment signals  
3. running the final valuation / backtesting pipeline  

---

## Project Structure

```bash
F550_FINAL_PROJECT/
├── data/
│   ├── processed/
│   └── raw/
├── notebooks/
│   ├── ex1_preprocess.ipynb
│   ├── ex1_sentiment.ipynb
│   └── ex2.ipynb
├── src_ex1/
│   ├── __init__.py
│   ├── ex1_preprocess.py
│   └── ex1_sentiment.py
├── src_ex2/
│   ├── __init__.py
│   ├── ex2_agent.py
│   ├── ex2_backtest.py
│   ├── ex2_inputs.py
│   └── ex2_openai_backend.py
├── environment_ex1.yml
├── pyproject.toml
└── README.md
```

## Installation

This project uses Conda for environment management. To set up the environment and install the package:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Bruno-He/d100_d400_project.git

    cd d100_d400_project
    ```

2.  **Create and activate the Conda environment:**
    ```bash
    conda env create -f environment_ex1.yml
    conda activate f550_ex1_light
    ```

3.  **Install the project package:**
    ```bash
    pip install -e .
    ```

## Usage

Run the notebooks in following order:

### 1. `ex1_preprocess.ipynb`
Preprocesses the raw input data.

### 2. `ex1_sentiment.ipynb`
Generates sentiment features from financial text.

### 3. `ex2.ipynb`
Runs the final valuation / agent / backtesting pipeline for exercise2. 


## API Setup

Some parts of the project require an OpenAI API key.

**Windows PowerShell**
```bash
$env:OPENAI_API_KEY="your_api_key_here"
```
**macOS / Linux**
```bash
export OPENAI_API_KEY="your_api_key_here"
```
You can also set it inside a notebook:
```bash
import os
os.environ["OPENAI_API_KEY"] = "your_api_key_here"
```