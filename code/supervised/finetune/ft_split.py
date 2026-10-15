import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

df_llama = pd.read_csv(os.path.join(PROJECT_ROOT, "llama_2k7.csv"))
df_gpt = pd.read_csv(os.path.join(PROJECT_ROOT, "gptoss_2k7.csv"))

assert len(df_llama) == len(df_gpt) == 2735, "data unmatched"
assert 'template' in df_llama.columns, "llama_2k7.csv no template column"

df_combined = pd.DataFrame({
    'human_idea': df_llama['human_idea'],  # label 0
    'llama_ai': df_llama['ai_idea_1_clean'], 
    'gpt_ai': df_gpt['ai_idea_1_clean'],    
    'template': df_llama['template']        
})

np.random.seed(3407) 
indices = np.arange(len(df_combined))
np.random.shuffle(indices)

train_idx = indices[:500]
dev_idx = indices[500:750]
test_idx = indices[750:]

def build_dataset(index_list, num_llama):
    subset = df_combined.iloc[index_list]
    
    human_df = pd.DataFrame({
        'idea_text': subset['human_idea'],
        'template': subset['template'],
        'label': 0,
        'source': 'human'
    })
    
    subset_indices = np.arange(len(subset))
    np.random.shuffle(subset_indices)
    
    llama_selected_idx = subset_indices[:num_llama]
    gpt_selected_idx = subset_indices[num_llama:]
    
    llama_df = pd.DataFrame({
        'idea_text': subset.iloc[llama_selected_idx]['llama_ai'],
        'template': subset.iloc[llama_selected_idx]['template'],
        'label': 1,
        'source': 'llama_ai'
    })
    
    gpt_df = pd.DataFrame({
        'idea_text': subset.iloc[gpt_selected_idx]['gpt_ai'],
        'template': subset.iloc[gpt_selected_idx]['template'],
        'label': 1,
        'source': 'gpt_ai'
    })
    
    final_df = pd.concat([human_df, llama_df, gpt_df]).sample(frac=1, random_state=3407).reset_index(drop=True)
    return final_df

# Train Set: 500 human + 250 Llama + 250 GPT = 1000 
train_df = build_dataset(train_idx, num_llama=250)

# Dev Set: 250 human + 125 Llama + 125 GPT = 500 
dev_df = build_dataset(dev_idx, num_llama=125)

# Test Set: 1985 human + 992 Llama + 993 GPT = 3970 
test_df = build_dataset(test_idx, num_llama=len(test_idx)//2)

train_df.to_csv(os.path.join(BASE_DIR, "train_set.csv"), index=False)
dev_df.to_csv(os.path.join(BASE_DIR, "dev_set.csv"), index=False)
test_df.to_csv(os.path.join(BASE_DIR, "test_set.csv"), index=False)

print(f"Train: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}")