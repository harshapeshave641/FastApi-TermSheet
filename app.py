from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import shap
import numpy as np
from typing import Dict, List, Optional

app = FastAPI(title="Term Sheet Validator API")

# Load model and explainer
model = joblib.load('term_sheet_validator_xgb.joblib')
explainer = shap.TreeExplainer(model)

# Field name mappings
FIELD_NAMES = {
    'pre_money_valuation': 'Pre-Money Valuation',
    'investment_amount': 'Investment Amount',
    'price_per_share': 'Price Per Share',
    'dividend_rate': 'Dividend Rate',
    'conversion_price': 'Conversion Price',
    'founder_equity': 'Founder Equity (%)',
    'vesting_period': 'Vesting Period (years)',
    'investment_to_valuation': 'Investment/Valuation Ratio',
    'price_to_conversion': 'Price/Conversion Ratio',
    'term_duration_days': 'Term Duration (days)',
    'industry': 'Industry Code',
    'security_type': 'Security Type Code',
    'currency': 'Currency Code',
    'governing_law': 'Governing Law Code',
    'voting_rights': 'Voting Rights Code'
}

# Expected feature order for the model
FEATURE_ORDER = [
    'pre_money_valuation', 'investment_amount', 'price_per_share',
    'dividend_rate', 'conversion_price', 'founder_equity', 'vesting_period',
    'investment_to_valuation', 'price_to_conversion', 'term_duration_days',
    'industry', 'security_type', 'currency', 'governing_law', 'voting_rights'
]

class TermSheet(BaseModel):
    pre_money_valuation: float
    investment_amount: float
    price_per_share: float
    dividend_rate: float
    conversion_price: float
    founder_equity: float
    vesting_period: int
    industry: int
    security_type: int
    currency: int
    governing_law: int
    voting_rights: int
    term_duration_days: Optional[int] = None
    investment_to_valuation: Optional[float] = None
    price_to_conversion: Optional[float] = None

class FieldAnalysis(BaseModel):
    field: str
    value: float
    impact: float
    interpretation: str

class ValidationResult(BaseModel):
    prediction: str
    confidence: str
    problem_fields: List[FieldAnalysis]
    supporting_fields: List[FieldAnalysis]

def prepare_input_data(input_dict: dict) -> pd.DataFrame:
    """Convert and validate input data with proper types"""
    # Calculate derived fields if not provided
    if input_dict['investment_to_valuation'] is None:
        input_dict['investment_to_valuation'] = (
            input_dict['investment_amount'] / input_dict['pre_money_valuation']
            if input_dict['pre_money_valuation'] > 0 else 0
        )
    
    if input_dict['price_to_conversion'] is None:
        input_dict['price_to_conversion'] = (
            input_dict['price_per_share'] / input_dict['conversion_price']
            if input_dict['conversion_price'] > 0 else 0
        )
    
    if input_dict['term_duration_days'] is None:
        # Default to 365 days if not provided
        input_dict['term_duration_days'] = 365
    
    # Create DataFrame with correct feature order and numeric types
    input_data = {k: [input_dict[k]] for k in FEATURE_ORDER}
    df = pd.DataFrame(input_data)
    
    # Ensure all numeric types
    float_cols = ['pre_money_valuation', 'investment_amount', 'price_per_share',
                 'dividend_rate', 'conversion_price', 'founder_equity',
                 'investment_to_valuation', 'price_to_conversion']
    int_cols = ['vesting_period', 'term_duration_days', 'industry',
               'security_type', 'currency', 'governing_law', 'voting_rights']
    
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    return df

@app.post("/validate", response_model=ValidationResult)
async def validate_term_sheet(term_sheet: TermSheet):
    """Validate a term sheet and explain the decision"""
    try:
        # Convert input to properly typed DataFrame
        input_df = prepare_input_data(term_sheet.dict())
        
        # Get prediction and SHAP values
        proba = model.predict_proba(input_df)[0][1]
        shap_values = explainer.shap_values(input_df)
        
        # Create impact analysis
        feature_impacts = []
        for i, feature in enumerate(FEATURE_ORDER):
            feature_impacts.append({
                'field': FIELD_NAMES.get(feature, feature),
                'value': input_df.iloc[0][feature],
                'impact': float(shap_values[0][i]),
                'interpretation': 'Supports validity' if shap_values[0][i] > 0 else 'Reduces validity'
            })
        
        # Sort by absolute impact
        feature_impacts.sort(key=lambda x: abs(x['impact']), reverse=True)
        
        return {
            'prediction': 'Valid' if proba > 0.5 else 'Invalid',
            'confidence': f"{proba:.1%}",
            'problem_fields': [f for f in feature_impacts if f['impact'] < -0.01],
            'supporting_fields': [f for f in feature_impacts if f['impact'] > 0.01]
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model-info")
async def get_model_info():
    """Get information about the model"""
    return {
        "model_type": "XGBoost",
        "n_features": len(FEATURE_ORDER),
        "features": [FIELD_NAMES.get(f, f) for f in FEATURE_ORDER],
        "required_fields": [
            "pre_money_valuation", "investment_amount", "price_per_share",
            "dividend_rate", "conversion_price", "founder_equity", "vesting_period",
            "industry", "security_type", "currency", "governing_law", "voting_rights"
        ],
        "calculated_fields": {
            "investment_to_valuation": "investment_amount / pre_money_valuation",
            "price_to_conversion": "price_per_share / conversion_price",
            "term_duration_days": "Optional (defaults to 365 days)"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)