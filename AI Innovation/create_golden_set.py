import pandas as pd
from datasets import load_dataset
import re

# 1. SCORING DICTIONARY
# We give points for every financial word found.
financial_terms = [
    'stock', 'market', 'share', 'price', 'invest', 'trade', 'trading', 'value', 
    'money', 'cash', 'currency', 'fund', 'etf', 'dividend', 'yield', 'return',
    'profit', 'loss', 'margin', 'equity', 'debt', 'bond', 'treasury', 'interest',
    'rate', 'tax', '401k', 'ira', 'pension', 'retirement', 'portfolio', 'asset',
    'liability', 'bullish', 'bearish', 'short', 'long', 'option', 'future',
    'volatility', 'risk', 'sector', 'cap', 'valuation', 'earnings', 'revenue',
    'crypto', 'bitcoin', 'ethereum', 'wallet', 'blockchain'
]

def calculate_financial_score(text):
    text = str(text).lower()
    score = 0
    # Add 1 point for every occurrence of a financial term
    for term in financial_terms:
        # We use simple string count for speed and "density"
        score += text.count(term)
    return score

def main():
    print("--- 1. Loading Data ---")
    dataset = load_dataset("gbharti/finance-alpaca", split='train')
    df = dataset.to_pandas()
    
    # Filter empty inputs (keep pure Q&A)
    df = df[df['input'].astype(str).str.strip() == ''].copy()
    
    # 2. CALCULATE DENSITY SCORE
    print("--- 2. Ranking by Financial Density ---")
    # specific_text combines instruction and output for scoring
    df['combined_text'] = df['instruction'].astype(str) + " " + df['output'].astype(str)
    df['financial_score'] = df['combined_text'].apply(calculate_financial_score)
    
    # 3. SORT AND TAKE TOP CANDIDATES
    df_sorted = df.sort_values(by='financial_score', ascending=False).head(500)
    
    print(f"Top ranked sample score: {df_sorted['financial_score'].iloc[0]}")
    print(f"Lowest ranked sample score: {df_sorted['financial_score'].iloc[-1]}")

    # 4. EXPORT FOR HAND-PICKING
    output_df = df_sorted[['instruction', 'output', 'financial_score']]
    output_df.to_csv("candidates_ranked_by_quality.csv", index=False)
    
    print("\nSUCCESS!")
    print("Saved 'candidates_ranked_by_quality.csv'.")


if __name__ == "__main__":
    main()