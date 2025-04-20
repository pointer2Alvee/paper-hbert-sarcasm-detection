# paper-hbert-sarcasm-detection
Implementation of the paper "A Novel Hierarchical BERT Architecture for Sarcasm Detection" using Python &amp; TensorFlow

##### paper-link : [A Novel Hierarchical BERT Architecture for Sarcasm Detection](https://aclanthology.org/2020.figlang-1.14.pdf)



```
paper-hbert-sarcasm-detection/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── setup.py                    # Optional: if you package the code
│
├── assets/                       # Raw or preprocessed datasets (ignored in git)
│   └── images/
│
├── data/                       # Raw or preprocessed datasets (ignored in git)
│   ├── raw/
│   └── processed/
│
├── notebooks/                 # Jupyter notebooks for experiments, EDA, etc.
│   └── sarcasm-analysis.ipynb
│
├── src/                       # Core Python source code
│   ├── __init__.py
│   ├── config.py              # Configuration (paths, hyperparameters)
│   ├── data_loader.py         # Data loading, preprocessing
│   ├── model.py               # Hierarchical BERT model
│   ├── train.py               # Training loop
│   ├── evaluate.py            # Evaluation metrics & logic
│   └── utils.py               # Helper functions
│
├── scripts/                   # Bash or Python scripts for running tasks
│   ├── run_train.sh
│   └── run_eval.sh
│
├── outputs/                   # Saved models, predictions, logs (gitignored)
│   ├── checkpoints/
│   ├── logs/
│   └── predictions/
│
├── examples/                  # Input/output usage examples (optional)
│   └── usage.py
│
└── paper/                     # Original paper and bibtex
    ├── paper.pdf
    └── citation.bib
```
