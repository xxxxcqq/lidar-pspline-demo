import json, pandas as pd

df = pd.read_csv("data/lidar.csv")
out = {
    "fig_id": "3.1",
    "chapter": 3,
    "title": "LIDAR 散点图",
    "book_params": {"n": 221},
    "scatter": {"x": df["range"].tolist(), "y": df["logratio"].tolist()},
    "xlim": [390, 720],
    "ylim": [-1.0, 0.05],
    "notes": "书中 Figure 3.1，用于引入平滑"
}
json.dump(out, open("assets/fig_3_1.json", "w"), indent=2)
