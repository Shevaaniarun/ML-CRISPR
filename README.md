# CRISPR sgRNA On-Target Activity Prediction using CatBoost

## Input
- sgRNA sequence (20–23 nucleotides)

## Output
- Predicted efficiency score (Indel_Norm)

## Validation
- Train/test split
- Spearman correlation metric

## Run
```bash
pip install -r requirements.txt
python main.py
