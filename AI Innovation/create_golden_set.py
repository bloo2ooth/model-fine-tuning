import pandas as pd
from datasets import load_dataset
import re

# 1. SCORING DICTIONARY
financial_terms = [
    'stock', 'market', 'share', 'price', 'invest', 'trade', 'trading', 'value', 
    'money', 'cash', 'currency', 'fund', 'etf', 'dividend', 'yield', 'return',
    'profit', 'loss', 'margin', 'equity', 'debt', 'bond', 'treasury', 'interest',
    'rate', 'tax', '401k', 'ira', 'pension', 'retirement', 'portfolio', 'asset',
    'liability', 'bullish', 'bearish', 'short', 'long', 'option', 'future',
    'volatility', 'risk', 'sector', 'cap', 'valuation', 'earnings', 'revenue',
    'crypto', 'bitcoin', 'ethereum', 'wallet', 'blockchain'
]

def analyze_quality(row):
    text = (str(row['instruction']) + " " + str(row['output'])).lower()
    word_count = len(text.split())
    
    # Avoid division by zero
    if word_count == 0: return 0, 0
    
    # Count unique financial terms (Diversity is better than repetition)
    score = 0
    for term in financial_terms:
        if term in text:
            score += 1
            
    return score, word_count

def main():
    print("--- 1. Loading Data ---")
    dataset = load_dataset("gbharti/finance-alpaca", split='train')
    df = dataset.to_pandas()
    
    # Filter empty inputs
    df = df[df['input'].astype(str).str.strip() == ''].copy()
    
    # 2. APPLY METRICS
    print("--- 2. Calculating Density ---")
    # Apply function and split results into two columns
    df[['financial_score', 'word_count']] = df.apply(
        lambda row: pd.Series(analyze_quality(row)), axis=1
    )
    
    # 3. THE "GOLDILOCKS" FILTER
    # Range: 20 words to 150 words
    df_readable = df[
        (df['word_count'] >= 20) & 
        (df['word_count'] <= 150)
    ].copy()
    
    # 4. RANK BY DENSITY
    # We sort by the raw score
    df_sorted = df_readable.sort_values(by='financial_score', ascending=False).head(300)
    
    print(f"Top sample: {df_sorted.iloc[0]['financial_score']} terms in {df_sorted.iloc[0]['word_count']} words.")

    # 5. EXPORT
    output_df = df_sorted[['instruction', 'output', 'financial_score', 'word_count']]
    output_df.to_csv("candidates_balanced.csv", index=False)
    
    print("Saved 'candidates_balanced.csv'")

if __name__ == "__main__":
    main()