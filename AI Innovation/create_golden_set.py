import pandas as pd
from datasets import load_dataset
import re
import sys

# ==============================================================================
# 1. KEYWORD LISTS
# ==============================================================================
keywords = {
    # 1. POSITIVE FILTER: If the text doesn't contain one of these, THROW IT OUT.
    "relevance_triggers": [
        'money', 'cash', 'dollar', 'euro', 'currency', 'finance', 'financial',
        'bank', 'economy', 'economic', 'market', 'stock', 'share', 'invest',
        'trade', 'trading', 'profit', 'loss', 'tax', 'debt', 'loan', 'credit',
        'interest', 'rate', 'mortgage', 'rent', 'salary', 'income', 'expense',
        'budget', 'cost', 'price', 'worth', 'value', 'asset', 'liability',
        'growth', 'inflation', 'recession', 'capital', 'wealth', 'savings',
        'deposit', 'withdraw', 'account', 'fund', 'wallet', 'coin', 'payment',
        'retirement', 'pension', '401k', 'ira', 'tax', 'fiscal', 'monetary',
        'business', 'corporate', 'revenue', 'earning', 'audit', 'insurance',
        'bullish', 'bearish', 'yield', 'dividend', 'portfolio'
    ],

    # 2. INSTRUMENTS (For Specificity Check)
    "instruments": [
        'apple', 'aapl', 'microsoft', 'msft', 'google', 'goog', 'amazon', 'amzn', 
        'nvidia', 'nvda', 'meta', 'tesla', 'tsla', 'netflix', 'jpmorgan', 'goldman',
        'bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 's&p 500', 'spy', 'qqq', 
        '401k', 'roth', 'ira', 'bond', 'treasury', 'gold', 'oil', 'real estate'
    ],

    # 3. ADVICE TRIGGERS (For Danger Check)
    "advice_triggers": [
        r'\brecommend\b', r'\bshould\b', r'\bought to\b', r'\bbetter to\b', 
        r'\bwise to\b', r'\bgood time to\b', r'\bundervalued\b', r'\bovervalued\b',
        r'\btarget price\b', r'\bstrong buy\b', r'\bstrong sell\b', r'\bportfolio allocation\b'
    ],
    
    # 4. GUIDANCE INDICATORS (Safe Zone)
    "guidance_indicators": [
        r'\bhow do i\b', r'\bhow to\b', r'\bwhat is\b', r'\bdefine\b', 
        r'\bhistory of\b', r'\bexplain\b', r'\bmeaning of\b'
    ]
}

# ==============================================================================
# 2. LOGIC
# ==============================================================================
def classify_row(row):
    text = (str(row['instruction']) + " " + str(row['output'])).lower()
    
    # STEP 1: IS IT EVEN RELEVANT? (The New Filter)
    # If no financial words are found, mark as INVALID immediately.
    is_relevant = any(w in text for w in keywords["relevance_triggers"])
    if not is_relevant:
        return "Invalid" # This kicks out the recipes/stories
        
    # STEP 2: IS IT PROCEDURAL? (Guidance)
    if any(re.search(p, text) for p in keywords["guidance_indicators"]):
        return "Guidance"

    # STEP 3: DOES IT HAVE SPECIFIC PRODUCTS?
    has_instrument = any(re.search(rf'\b{re.escape(w)}\b', text) for w in keywords["instruments"])
    
    # STEP 4: DOES IT GIVE A RECOMMENDATION?
    has_advice = any(re.search(p, text) for p in keywords["advice_triggers"])
    
    # --- FINAL CATEGORIES ---
    if has_instrument and has_advice:
        return "Advice"
    elif ("i am" in text or "my age" in text) and not has_instrument:
        return "Edge Case"
    else:
        # If it's financial but not advice/personal, it's safe Guidance
        return "Guidance"

# ==============================================================================
# 3. EXECUTION
# ==============================================================================
def main():
    print("--- Starting RELEVANCE-FILTERED Sampling ---")
    
    dataset = load_dataset("gbharti/finance-alpaca", split='train')
    df = dataset.to_pandas()
    
    # Filter empty inputs (Reading comprehension removal)
    df_clean = df[df['input'].astype(str).str.strip() == ''].copy()
    
    print("Applying logic...")
    df_clean['presumed_label'] = df_clean.apply(classify_row, axis=1)
    
    # REMOVE THE INVALID ROWS
    df_financial = df_clean[df_clean['presumed_label'] != "Invalid"]
    print(f"   > Removed {len(df_clean) - len(df_financial)} non-financial rows.")
    print(f"   > Remaining Financial Pool: {len(df_financial)}")

    print("Sampling best 100...")
    try:
        advice = df_financial[df_financial['presumed_label'] == "Advice"].sample(40, random_state=42)
        
        edge_pool = df_financial[df_financial['presumed_label'] == "Edge Case"]
        edge = edge_pool.sample(min(30, len(edge_pool)), random_state=42)
        
        guidance = df_financial[df_financial['presumed_label'] == "Guidance"].sample(30, random_state=42)
        
        golden_set = pd.concat([advice, edge, guidance])
        final_table = golden_set[['instruction', 'output', 'presumed_label']].copy()

        # Overwrite the existing CSV
        final_table.to_csv("logbook_validation_100_readable.csv", index=False)
        final_table.to_json("logbook_validation_100.jsonl", orient="records", lines=True)
        print("\nSuccess! Overwrote 'logbook_validation_100_readable.csv'")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()