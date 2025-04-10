import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics import f1_score
from evaluate import load

os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_dir = "codet5_finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
model.eval()

df = pd.read_csv("processed_data/test.csv").head(50)

bleu = load("sacrebleu")

def normalize(text):
    return " ".join(text.strip().split())

def get_prediction(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=64)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

predictions = []
exact_matches = []
f1_scores = []
bleu_scores = []
codebleu_scores = []

for idx, row in df.iterrows():
    if idx % 5 == 0:
        print(f"Processing row {idx + 1}/{len(df)}...")

    input_text = normalize(row['input'])
    expected = normalize(row['target'])
    predicted = normalize(get_prediction(input_text))

    predictions.append(predicted)
    exact_matches.append(expected == predicted)

    expected_tokens = expected.split()
    predicted_tokens = predicted.split()
    common = set(expected_tokens) & set(predicted_tokens)

    if expected_tokens and predicted_tokens:
        precision = len(common) / len(predicted_tokens)
        recall = len(common) / len(expected_tokens)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    else:
        f1 = 0
    f1_scores.append(f1)

    bleu_result = bleu.compute(predictions=[predicted], references=[[expected]])
    bleu_scores.append(round(bleu_result['score'], 2))

    codebleu_scores.append(0)

df_out = pd.DataFrame({
    "input": df['input'],
    "correct": exact_matches,
    "expected": df['target'],
    "predicted": predictions,
    "codebleu_score": codebleu_scores,
    "bleu4_score": bleu_scores
})

df_out.to_csv("testset-results.csv", index=False)

print("\nEvaluation Summary:")
print("Exact Match:", round(sum(exact_matches) / len(exact_matches), 4))
print("Average F1 Score:", round(sum(f1_scores) / len(f1_scores), 4))
print("Average BLEU-4:", round(sum(bleu_scores) / len(bleu_scores), 2))
print("Average CodeBLEU:", round(sum(codebleu_scores) / len(codebleu_scores), 2))
