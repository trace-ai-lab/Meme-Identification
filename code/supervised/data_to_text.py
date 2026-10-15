import pandas as pd

def prepare_fasttext_data(csv_file, output_txt_file):
    df = pd.read_csv(csv_file)
    
    # Handle NaNs
    df['template'] = df['template'].fillna('N/A')
    df['idea_text'] = df['idea_text'].fillna('')
    
    # Concatenate text
    texts = "Template: " + df['template'].astype(str) + " | Idea: " + df['idea_text'].astype(str)
    
    # IMPORTANT: Remove newlines from the text so fastText doesn't split one example into multiple lines
    texts = texts.str.replace('\n', ' ', regex=False).str.replace('\r', '', regex=False)
    
    # Format labels (e.g., 0 becomes __label__0)
    labels = "__label__" + df['label'].astype(str)
    
    # Write to text file
    with open(output_txt_file, 'w', encoding='utf-8') as f:
        for label, text in zip(labels, texts):
            f.write(f"{label} {text}\n")

prepare_fasttext_data("train_set.csv", "train.txt")
prepare_fasttext_data("dev_set.csv", "dev.txt")
prepare_fasttext_data("test_set.csv", "test.txt")