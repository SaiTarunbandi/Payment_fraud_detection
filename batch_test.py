import requests
import pandas as pd

# Flask endpoint
URL = "http://127.0.0.1:5000/predict"

# Test transactions
test_data = [
    {"amount": 150.75, "hour": 14, "source_type": "upi"},             # small UPI, day
    {"amount": 250000.0, "hour": 2, "source_type": "online_payment"}, # big night transfer
    {"amount": 1200.5, "hour": 18, "source_type": "credit_card"},     # normal evening CC
    {"amount": 99999.99, "hour": 23, "source_type": "upi"},           # high late-night UPI
    {"amount": 870.2, "hour": 9, "source_type": "online_payment"},    # normal morning
]

results = []

for tx in test_data:
    try:
        r = requests.post(URL, json=tx)
        if r.status_code == 200:
            res = r.json()
            results.append({
                "Amount": tx["amount"],
                "Hour": tx["hour"],
                "Source": tx["source_type"],
                "Proba": res.get("proba", None),
                "Threshold": res.get("threshold", None),
                "Prediction": "Fraud" if res.get("prediction") == 1 else "Legit"
            })
        else:
            results.append({
                "Amount": tx["amount"],
                "Hour": tx["hour"],
                "Source": tx["source_type"],
                "Proba": None,
                "Threshold": None,
                "Prediction": f"Error {r.status_code}"
            })
    except Exception as e:
        results.append({
            "Amount": tx["amount"],
            "Hour": tx["hour"],
            "Source": tx["source_type"],
            "Proba": None,
            "Threshold": None,
            "Prediction": f"Error: {str(e)}"
        })

# Display in table
df = pd.DataFrame(results)
print("\n=== Batch Test Results ===")
print(df.to_string(index=False))
