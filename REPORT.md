## Submission Structure

Your final submission should be a GitHub repository structured as follows:

```
your-repo/
├── README.md              ← this file
├── REPORT.md              ← Task 5 report
├── dataset.py             ← Task 1
├── model.py               ← Task 2
├── train.py               ← Task 3
├── evaluate.py            ← Task 4
├── requirements.txt       ← all dependencies with versions
├── notebooks/
│   └── eda.ipynb          ← exploratory data analysis
└── outputs/
    ├── training_curves.png
    ├── metrics.json
    └── predictions/
        └── *.png
```

> **Note:** Do **not** commit the dataset itself. Your `README.md` or a `setup.sh` script should include instructions to download/clone it.
