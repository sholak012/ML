import json
import pandas as pd
import string
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

with open('abai-sozderi.json', 'r', encoding='utf-8') as f: data = json.load(f)
df = pd.DataFrame(data)

def _sent_count(t): return max(1, len(sent_tokenize(str(t))))

df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
df['sentence_count'] = df['text'].apply(_sent_count)
df['token_nltk'] = df['text'].apply(lambda x: len(word_tokenize(str(x).lower())))

all_wt = []
for t in df['text']: all_wt.extend(word_tokenize(str(t).lower()))

print(f"Docs: {len(df)}")
print(f"Tokens: {len(all_wt)}")
print(f"Types: {len(set(all_wt))}")
print(f"Avg text (words): {df['word_count'].mean():.2f}")
print(f"Avg text (tokens): {df['token_nltk'].mean():.2f}")
print(f"Avg sent (words): {(df['word_count'] / df['sentence_count']).mean():.2f}")
