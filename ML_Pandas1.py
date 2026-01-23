import pandas as pd
data = {
    "Name": ["Amit", "Riya", "John"],
    "Age": [25, 30, 28],
    "City": ["Delhi", "Mumbai", "Bangalore"]
}
df = pd.DataFrame(data)
df.to_csv("people.csv", index=False)
