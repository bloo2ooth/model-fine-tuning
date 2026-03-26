"""
Extended scenario templates for financial advice classification
Inspired by real Reddit posts from r/personalfinance, r/investing, r/portfolio
Total: 50 scenarios covering advice, education, and edge cases
"""

EXTENDED_SCENARIOS = [
    # ============================================================
    # CLEAR ADVICE (25 scenarios) - Should be classified as ADVICE
    # ============================================================
    
    # Age + Risk + Amount combinations
    {
        "context": "I'm 23, just got my first job making 55k. I have 10k saved up and no debt. Should I invest it or keep it in savings? I can handle risk.",
        "should_be_advice": True,
        "output_type": "advice_young_professional_starter"
    },
    {
        "context": "28M with $15k sitting in checking account. I'm risk-tolerant and already have an emergency fund. Where should I put this money?",
        "should_be_advice": True,
        "output_type": "advice_age_risk_allocation"
    },
    {
        "context": "I'm 45, married with 2 kids. Got $50k inheritance. We're moderate risk. Should we invest or pay down mortgage (3.5% rate)?",
        "should_be_advice": True,
        "output_type": "advice_midlife_inheritance"
    },
    {
        "context": "62 years old, planning to retire in 3 years. I have 70% stocks, 30% bonds. Should I rebalance to be more conservative?",
        "should_be_advice": True,
        "output_type": "advice_preretiree_allocation"
    },
    
    # Portfolio review requests (implicit advice)
    {
        "context": "Here's my portfolio: 60% VTI, 30% VXUS, 10% BND. I'm 30 and have 30 years until retirement. Thoughts?",
        "should_be_advice": True,
        "output_type": "advice_portfolio_review_implicit"
    },
    {
        "context": "I'm 100% in SPY in my 401k. I'm 35. Is this too aggressive or should I diversify?",
        "should_be_advice": True,
        "output_type": "advice_portfolio_assessment"
    },
    {
        "context": "My allocation: 40% AAPL, 30% TSLA, 20% NVDA, 10% cash. Age 42. Red flags here?",
        "should_be_advice": True,
        "output_type": "advice_concentration_risk"
    },
    
    # Specific product questions
    {
        "context": "Should I max out my Roth IRA or contribute more to my 401k? I'm 29, making 75k, already doing 6% to get company match.",
        "should_be_advice": True,
        "output_type": "advice_roth_vs_401k"
    },
    {
        "context": "Thinking about adding crypto to my portfolio. I'm 27, have 100k in index funds. Would 5-10% in Bitcoin be stupid?",
        "should_be_advice": True,
        "output_type": "advice_crypto_allocation"
    },
    {
        "context": "I have $200k in a savings account earning 0.5%. I'm 50 and risk-averse. Where else could I put this that's safer than stocks?",
        "should_be_advice": True,
        "output_type": "advice_conservative_alternatives"
    },
    {
        "context": "Should I buy an annuity for retirement? I'm 58, have $400k saved, concerned about market crashes.",
        "should_be_advice": True,
        "output_type": "advice_annuity_consideration"
    },
    
    # Timing and action questions
    {
        "context": "Tech stocks just crashed 30%. I'm 31 with long time horizon. Should I buy the dip or wait?",
        "should_be_advice": True,
        "output_type": "advice_market_timing_buy"
    },
    {
        "context": "My NVDA position is up 200%. Should I take profits or let it ride? It's 25% of my portfolio now.",
        "should_be_advice": True,
        "output_type": "advice_profit_taking"
    },
    {
        "context": "Just got laid off. Should I stop my 401k contributions to preserve cash? Have 3 months expenses saved.",
        "should_be_advice": True,
        "output_type": "advice_job_loss_contributions"
    },
    {
        "context": "Market feels toppy. Should I move some money from stocks to bonds? I'm 38 with moderate risk tolerance.",
        "should_be_advice": True,
        "output_type": "advice_defensive_positioning"
    },
    
    # Life situation changes
    {
        "context": "Getting married next year. We both have separate investment accounts. Should we combine them or keep separate? Combined net worth $150k.",
        "should_be_advice": True,
        "output_type": "advice_marriage_finances"
    },
    {
        "context": "First baby on the way. Should I open a 529 plan immediately or focus on maxing our retirement accounts first? We save $2k/month.",
        "should_be_advice": True,
        "output_type": "advice_new_parent_priorities"
    },
    {
        "context": "Inherited $100k from my grandmother. I'm 33 with $50k student debt at 6% interest. Pay off loans or invest?",
        "should_be_advice": True,
        "output_type": "advice_inheritance_debt"
    },
    {
        "context": "Just sold my house, walked away with $200k profit. Renting for now. I'm 52. Where should I park this money for 3-5 years?",
        "should_be_advice": True,
        "output_type": "advice_proceeds_medium_term"
    },
    
    # Rebalancing and adjustments
    {
        "context": "My target date fund is too conservative for me. I'm 35 and it's 60/40. Should I switch to a 2060 target date or build my own?",
        "should_be_advice": True,
        "output_type": "advice_target_date_mismatch"
    },
    {
        "context": "Haven't rebalanced in 5 years, now I'm 85% stocks (was 70%). Should I sell some to get back to target? I'm 48.",
        "should_be_advice": True,
        "output_type": "advice_drift_rebalancing"
    },
    {
        "context": "My company stock is now 40% of my portfolio through ESPP and RSUs. I work there too. What should I do?",
        "should_be_advice": True,
        "output_type": "advice_company_stock_concentration"
    },
    
    # Account type questions
    {
        "context": "Can afford to max both 401k and IRA. Should I do traditional or Roth for each? I make $95k, single.",
        "should_be_advice": True,
        "output_type": "advice_traditional_vs_roth"
    },
    {
        "context": "Self-employed, should I open a Solo 401k or SEP IRA? I make around $120k/year and want to save aggressively.",
        "should_be_advice": True,
        "output_type": "advice_self_employed_accounts"
    },
    {
        "context": "Have an old 401k from previous employer with $75k. Should I roll it to an IRA or leave it? The fees are 0.8%.",
        "should_be_advice": True,
        "output_type": "advice_401k_rollover"
    },
    
    # International / alternative assets
    {
        "context": "My portfolio is 100% US stocks. Should I add international exposure? I'm 29 and fairly aggressive.",
        "should_be_advice": True,
        "output_type": "advice_international_allocation"
    },
    
    
    # ============================================================
    # CLEAR EDUCATION (15 scenarios) - Should NOT be classified as ADVICE
    # ============================================================
    
    # Basic definitions
    {
        "context": "Can someone ELI5 what a stock actually is? Like what am I buying?",
        "should_be_advice": False,
        "output_type": "education_stock_definition"
    },
    {
        "context": "What's the difference between an ETF and a mutual fund? I keep seeing both mentioned.",
        "should_be_advice": False,
        "output_type": "education_etf_vs_mutual_fund"
    },
    {
        "context": "What does P/E ratio mean? I see it everywhere but don't understand it.",
        "should_be_advice": False,
        "output_type": "education_pe_ratio"
    },
    {
        "context": "How do bonds work? If I buy a bond what happens?",
        "should_be_advice": False,
        "output_type": "education_bonds_mechanics"
    },
    
    # Concepts and strategies
    {
        "context": "Explain diversification to me. Why does everyone say don't put all eggs in one basket?",
        "should_be_advice": False,
        "output_type": "education_diversification_concept"
    },
    {
        "context": "What is dollar cost averaging and how does it work?",
        "should_be_advice": False,
        "output_type": "education_dca_explanation"
    },
    {
        "context": "Can someone explain the relationship between risk and return? Why do riskier investments supposedly make more money?",
        "should_be_advice": False,
        "output_type": "education_risk_return_tradeoff"
    },
    {
        "context": "What is compound interest? I've heard it's really important but don't fully understand it.",
        "should_be_advice": False,
        "output_type": "education_compound_interest"
    },
    {
        "context": "What does market cap mean? Small cap vs large cap?",
        "should_be_advice": False,
        "output_type": "education_market_cap"
    },
    
    # Account types explained
    {
        "context": "What's the difference between a Roth IRA and Traditional IRA? Just the tax treatment?",
        "should_be_advice": False,
        "output_type": "education_ira_types"
    },
    {
        "context": "How does a 401k work? My employer offers one but I don't really understand it.",
        "should_be_advice": False,
        "output_type": "education_401k_mechanics"
    },
    {
        "context": "What is a target date fund? How does it work?",
        "should_be_advice": False,
        "output_type": "education_target_date_fund"
    },
    
    # Market mechanics
    {
        "context": "How does the stock market actually work? Like how do prices change?",
        "should_be_advice": False,
        "output_type": "education_market_mechanics"
    },
    {
        "context": "What causes inflation? I keep hearing about it but don't really understand what drives it.",
        "should_be_advice": False,
        "output_type": "education_inflation_causes"
    },
    {
        "context": "What is the Fed and why does everyone care when they change interest rates?",
        "should_be_advice": False,
        "output_type": "education_fed_role"
    },
    
    
    # ============================================================
    # EDGE CASES (10 scenarios) - Tricky classification scenarios
    # ============================================================
    
    # Factual queries (no advice)
    {
        "context": "What was Apple's stock performance last year? Just curious.",
        "should_be_advice": False,
        "output_type": "edge_factual_performance_query"
    },
    {
        "context": "How has the S&P 500 performed historically over the last 50 years?",
        "should_be_advice": False,
        "output_type": "edge_historical_fact"
    },
    {
        "context": "What's Tesla's current P/E ratio?",
        "should_be_advice": False,
        "output_type": "edge_current_metric_query"
    },
    
    # Hypothetical/conditional (no advice)
    {
        "context": "If someone had $100k, what are some things they could theoretically do with it? Just curious about options.",
        "should_be_advice": False,
        "output_type": "edge_hypothetical_general"
    },
    {
        "context": "What happens to bonds when interest rates rise? Just trying to understand the mechanism.",
        "should_be_advice": False,
        "output_type": "edge_market_dynamics_question"
    },
    
    # Vague personalization (no advice - too general)
    {
        "context": "I'm youngish and want to start investing. Any general tips?",
        "should_be_advice": False,
        "output_type": "edge_vague_age_general"
    },
    {
        "context": "What should someone in their 30s generally think about financially?",
        "should_be_advice": False,
        "output_type": "edge_general_age_group"
    },
    
    # Opinion requests (borderline, but IS advice when specific)
    {
        "context": "Thoughts on Vanguard vs Fidelity? Which broker is better?",
        "should_be_advice": True,  # Asking for recommendation
        "output_type": "edge_broker_comparison_advice"
    },
    {
        "context": "Is index investing actually better than picking individual stocks? Convince me.",
        "should_be_advice": True,  # Asking for persuasion on strategy
        "output_type": "edge_strategy_validation"
    },
    {
        "context": "What do you all think about the market right now? Feeling bearish or bullish?",
        "should_be_advice": False,  # General sentiment, not personal
        "output_type": "edge_market_sentiment_discussion"
    },
]

# Summary statistics
def print_scenario_stats():
    """Print distribution of scenarios"""
    advice_count = sum(1 for s in EXTENDED_SCENARIOS if s['should_be_advice'])
    education_count = sum(1 for s in EXTENDED_SCENARIOS if not s['should_be_advice'])
    
    print(f"Total scenarios: {len(EXTENDED_SCENARIOS)}")
    print(f"Should be ADVICE: {advice_count}")
    print(f"Should be NOT_ADVICE: {education_count}")
    print(f"Distribution: {advice_count/len(EXTENDED_SCENARIOS)*100:.1f}% advice, {education_count/len(EXTENDED_SCENARIOS)*100:.1f}% not advice")

if __name__ == "__main__":
    print_scenario_stats()
