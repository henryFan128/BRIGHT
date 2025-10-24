# BRIGHT: A Knowledge Graph Reasoning Framework for Infectious Disease Drug Discovery

[![python](https://img.shields.io/badge/Python-3.11-3776AB.svg?style=flat&logo=python&logoColor=yellow)](https://www.python.org)
[![pytorch](https://img.shields.io/badge/PyTorch-2.3.1-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![pytorch](https://img.shields.io/badge/Neo4j-5.26.0-3BA997.svg?style=flat&logo=neo4j)](https://neo4j.com)
[![MIT license](https://img.shields.io/badge/LICENSE-MIT-A8ACB9)](./LICENSE)

![](./fig/BRIGHT.png)

## 🌍 Introduction
**BRIGHT** (Biomedical Reasoning with InteGrated Hierarchical represenTations) is a novel reasoning framework designed to accelerate infectious disease drug discovery using the Infectious Disease Knowledge Graph (IDKG).

Facing the global challenge of emerging infectious diseases and antimicrobial resistance, BRIGHT integrates heterogeneous biological data and applies adaptive graph reasoning to discover potential associations, enabling drug repositioning and target discovery.

## 🧠 Infectious Disease Knowledge Graph (IDKG)
**IDKG** is a **comprehensive Infectious Disease Knowledge Graph** designed to integrate diverse biomedical data sources into a unified framework. It aims to facilitate advanced research and applications in infectious disease understanding, treatment, and prevention.

![](./fig/IDKG.png)

## ⚙️ Framework Overview
**BRIGHT** leverages IDKG as the foundation for a multi-level reasoning pipeline integrating semantic, topological, and hierarchical information. The framework consists of the following key components: 
- Semantic feature extraction
- Hierarchical feature extraction 
- Adaptive feature fusion

## 🧩 Repository Structure
```
BRIGHT/
├── src/
│   ├── model/                   # Model definitions and neural layers
│   └── utils/                   # Utility functions
├── scripts/                     # Utility scripts
├── data/                        # Data directory
├── config/                      # YAML configuration files
└── requirements.txt             # Dependencies
```

## 🚀 Quick Start

```bash
# Clone the repository
git clone git@github.com:henryFan128/BRIGHT.git
cd BRIGHT

# Install dependencies
pip install -r requirements.txt

# Train the model
python scripts/train.py --config config/train.yaml

# Inference & Link prediction
python scripts/inference.py --config config/inference.yaml
```


