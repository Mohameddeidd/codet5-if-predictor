import pandas as pd
import os
import re

train_path = "ft_train.csv"
valid_path = "ft_valid.csv"
test_path = "ft_test.csv"

output_dir = "processed_data"
os.makedirs(output_dir, exist_ok=True)

def mask_condition(row):
    method = row['cleaned_method']
    condition = row['target_block'].strip()
    
    escaped_condition = re.escape(condition)
    if re.search(escaped_condition, method):
        return re.sub(escaped_condition, "<mask>", method, count=1)
    
    return re.sub(r'\bif\b', '<mask>', method, count=1)

def flatten_code(code):
    return " ".join(code.strip().split())

def process_file(input_path, output_path):
    df = pd.read_csv(input_path)
    df['input'] = df.apply(mask_condition, axis=1)
    df['input'] = df['input'].apply(flatten_code)
    df['target'] = df['target_block'].apply(lambda x: " ".join(str(x).strip().split()))
    df[['input', 'target']].to_csv(output_path, index=False)

process_file(train_path, os.path.join(output_dir, "train.csv"))
process_file(valid_path, os.path.join(output_dir, "valid.csv"))
process_file(test_path, os.path.join(output_dir, "test.csv"))
print("Preprocessing complete. Files saved in 'processed_data/'")
