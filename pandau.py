import pandas as pd

fc = pd.read_csv("data/stock_forecast_next_7_days_with_alerts.csv")
print(fc.columns)
print(fc.head(5))
