# Netra - AI-Powered Cybercrime Classification System

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/framework-scikit--learn-orange.svg)](https://scikit-learn.org/)

## Overview

Netra is an advanced cybercrime classification system that uses Natural Language Processing (NLP) to automatically categorize cybercrime complaints. Built for the IndiaAI CyberGuard Hackathon, it employs dual Random Forest classifiers to simultaneously predict both main categories and subcategories of cybercrime incidents.

## Key Features

- Dual-classification system with 89.5% accuracy
- Advanced text preprocessing pipeline
- Production-ready with comprehensive error handling
- Automated model retraining capabilities
- Privacy-preserving feature extraction

## Quick Start

### Prerequisites

- Python 3.11+
- [UV package manager](https://github.com/astral-sh/uv)

### Installation

```bash
# Clone the repository
git clone https://github.com/ChiragAgg5k/netra.git
cd netra

# Create and activate virtual environment
uv sync
```

## What does the repository contain?

1. The `src/` directory contains various ipynb notebooks for the project, including a SVM, Random Forest and Multi-Vote architecture for training and testing the pipeline.

2. `data/` folder contains `test.csv` and `train.csv` files for training and testing the pipeline. These files were obtained from the [IndiaAI CyberGuard Hackathon](https://indiaai.gov.in/article/indiaai-launches-cyberguard-ai-cybercrime-prevention-hackathon).

3. `assets/` folder contains the graphs generated in the notebooks.

## Contact

For any queries or support:
- Email: chiragaggarwal5k@gmail.com
- GitHub Issues: [Create an issue](https://github.com/ChiragAgg5k/netra/issues)