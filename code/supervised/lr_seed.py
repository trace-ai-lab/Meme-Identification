import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_and_prepare_data(csv_file):
    df = pd.read_csv(csv_file)
    df['template'] = df['template'].fillna('N/A')
    df['idea_text'] = df['idea_text'].fillna('')
    
    df['text'] = "Template: " + df['template'] + " | Idea: " + df['idea_text']
    df['label'] = df['label'].astype(int)
    
    return df['text'].tolist(), df['label'].tolist()

train_texts, train_labels = load_and_prepare_data("train_set.csv")
dev_texts, dev_labels = load_and_prepare_data("dev_set.csv")
test_texts, test_labels = load_and_prepare_data("test_set.csv")
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=50000)

X_train = vectorizer.fit_transform(train_texts)
X_dev = vectorizer.transform(dev_texts)
X_test = vectorizer.transform(test_texts)

def evaluate_model(model, X, y_true):
    predictions = model.predict(X)
    acc = accuracy_score(y_true, predictions)
    prec = precision_score(y_true, predictions, zero_division=0)
    rec = recall_score(y_true, predictions, zero_division=0)
    f1 = f1_score(y_true, predictions, zero_division=0)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

seeds = [0, 1, 2, 3, 4]
n_runs = len(seeds)

dev_metrics_history = {"accuracy": [], "precision": [], "recall": [], "f1": []}
test_metrics_history = {"accuracy": [], "precision": [], "recall": [], "f1": []}
all_coefs = []

for seed in seeds:
    model = LogisticRegression(random_state=seed, max_iter=1000, class_weight='balanced')
    model.fit(X_train, train_labels)
    all_coefs.append(model.coef_[0])
    
    dev_res = evaluate_model(model, X_dev, dev_labels)
    for k in dev_metrics_history:
        dev_metrics_history[k].append(dev_res[k])
    test_res = evaluate_model(model, X_test, test_labels)
    for k in test_metrics_history:
        test_metrics_history[k].append(test_res[k])
    
    print(f"[Seed {seed}] Dev  - Acc: {dev_res['accuracy']:.5f} | Prec: {dev_res['precision']:.5f} | Rec: {dev_res['recall']:.5f} | F1: {dev_res['f1']:.5f}")
    print(f"[Seed {seed}] Test - Acc: {test_res['accuracy']:.5f} | Prec: {test_res['precision']:.5f} | Rec: {test_res['recall']:.5f} | F1: {test_res['f1']:.5f}")
    print("-" * 55)

def calculate_mean_se(metrics_history):
    summary = {}
    for k, vals in metrics_history.items():
        mean_val = np.mean(vals)
        se_val = np.std(vals, ddof=1) / np.sqrt(n_runs) if n_runs > 1 else 0.0
        summary[k] = (mean_val, se_val)
    return summary

def print_summary(summary):
    print(f"  Accuracy : {summary['accuracy'][0]:.5f} ± {summary['accuracy'][1]:.5f}")
    print(f"  Precision: {summary['precision'][0]:.5f} ± {summary['precision'][1]:.5f}")
    print(f"  Recall   : {summary['recall'][0]:.5f} ± {summary['recall'][1]:.5f}")
    print(f"  F1 Score : {summary['f1'][0]:.5f} ± {summary['f1'][1]:.5f}")

dev_summary = calculate_mean_se(dev_metrics_history)
test_summary = calculate_mean_se(test_metrics_history)

print("\n" + "#"*55)
print("             Final Result Report (5 Runs Mean ± SE)")
print("#"*55)
print("\n=== [1] Dev Set ===")
print_summary(dev_summary)

print("\n=== [2] Test Set ===")
print_summary(test_summary)
print("\n" + "#"*55)