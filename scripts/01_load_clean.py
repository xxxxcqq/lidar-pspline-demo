import pandas as pd

df = pd.read_csv("data/Session2.7_lidar.txt", sep=r"\s+")
df.columns = ["range", "logratio"]
assert df.shape[0] == 221, f"行数不对: {df.shape[0]}"
assert df["range"].between(390, 720).all()
df.to_csv("data/lidar.csv", index=False)
print("ok", df.shape)
