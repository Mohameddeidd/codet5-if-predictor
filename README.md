# code5-if-predictor

---

# GenAI for Software Development (CodeT5)

* [1 Introduction](#1-introduction)  
* [2 Getting Started](#2-getting-started)  
  * [2.1 Preparations](#21-preparations)  
  * [2.2 Install Packages](#22-install-packages)  
  * [2.3 Run CodeT5](#23-run-codet5)  
* [3 Report](#3-report)  

---

# **1. Introduction**  
This project explores **conditional prediction in Python**, leveraging **transformer-based language modeling**. The CodeT5-small model is fine-tuned to predict missing `if` conditions within Python functions. When an `if` statement is masked with `<mask>`, the model learns to generate the original condition by interpreting surrounding context. This approach applies deep learning to source code understanding and program synthesis.

---

# **2. Getting Started**  

This project is implemented in **Python 3.10+** and is compatible with **macOS, Linux, and Windows**.  

## **2.1 Preparations**  

(1) Clone the repository to your workspace:  
```shell
~ $ git clone https://github.com/Mohameddeidd/codet5-if-predictor.git
```

(2) Navigate into the repository:
```shell
~ $ cd codet5-if-predictor
~/codet5-if-predictor $
```

(3) Set up a virtual environment and activate it:

For macOS/Linux:
```shell
~/codet5-if-predictor $ python -m venv ./venv/
~/codet5-if-predictor $ source venv/bin/activate
(venv) ~/codet5-if-predictor $
```

To deactivate the virtual environment, use the command:
```shell
(venv) $ deactivate
```

## **2.2 Install Packages**

Install the required dependencies:
```shell
(venv) ~/codet5-if-predictor $ pip install transformers datasets pandas scikit-learn evaluate
```

You must also clone CodeBLEU for final evaluation:
```shell
(venv) ~/codet5-if-predictor $ git clone https://github.com/microsoft/CodeXGLUE.git
```

## **2.3 Run CodeT5**

(1) Run preprocessing

This script takes the raw datasets (`ft_train.csv`, `ft_valid.csv`, `ft_test.csv`), masks the `if` conditions with `<mask>`, flattens the code, and prepares it for training.

Run the preprocessing script with:
```shell
(venv) ~/codet5-if-predictor $ python preprocessing.py
```

The following files will be created in the `processed_data/` directory:
- `train.csv`
- `valid.csv`
- `test.csv`

(2) Fine-tune the CodeT5 model

This step fine-tunes the `Salesforce/codet5-small` model on the training data and evaluates on the validation set.

Run training with:
```shell
(venv) ~/codet5-if-predictor $ python train_codet5.py
```

The fine-tuned model will be saved in the directory `codet5_finetuned/`.

(3) Run evaluation

The evaluation script will generate predictions on the test set and compute metrics including BLEU-4, CodeBLEU, F1 Score, and exact match.

Run the evaluation with:
```shell
(venv) ~/codet5-if-predictor $ python run_evaluation.py
```

This will generate the file `testset-results.csv` with the following columns:
- `input` (function with `<mask>`)
- `expected` (ground truth condition)
- `predicted` (model’s output)
- `correct` (true/false if exact match)
- `codebleu_score`
- `bleu4_score`

---

# **3. Report**

The assignment report is available in the file `Assignment_Report.pdf`.

---

