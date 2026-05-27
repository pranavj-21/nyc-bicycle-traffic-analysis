# NYC Bicycle Traffic Analysis

Analyzing bicycle traffic patterns across New York City bridges using 2016 data.
Built as a data science mini project at Purdue University.

## What this project does

**Q1 — Best bridge predictor combo**
Tests all combinations of 3 bridges to find which group best predicts total
NYC bicycle traffic using linear regression.

**Q2 — Weather-based prediction**
Uses yesterday's high temp, low temp, and precipitation to predict today's
total ridership. Built lag features to simulate real forecasting conditions.

**Q3 — Weekly pattern analysis**
Breaks down average ridership by day of week and compares weekday vs weekend
traffic, with a bar chart visualization.

## Results
- Best 3-bridge combination identified by highest R² score
- Weather model shows temperature has a stronger effect than precipitation
- Weekday traffic is significantly higher than weekends, peaking on Wednesdays

## Tech stack
- Python
- Pandas
- scikit-learn
- Matplotlib

## Dataset
NYC bicycle count data (2016)

## How to run
```bash
pip install pandas matplotlib scikit-learn
python solution.py
```
