import numpy as np
import pandas as pd
import sqlite3

import statsmodels.api as sm
from statsmodels.formula.api import ols

conn = sqlite3.connect("./data/full-scale.db")

columns = ["train_image_count", "fake_image_ratio", "test_acc"]
query = f"SELECT {', '.join(columns)} FROM data"

data = conn.execute(query).fetchall()
conn.close()

data = [(int(trial[0] * trial[1]), int(trial[0] * (1 - trial[1])), trial[2]) for trial in data]

df = pd.DataFrame({"real_images": [trial[1] for trial in data],
                   "synthetic_images": [trial[0] for trial in data],
                   "test_accuracy": [trial[2] for trial in data],})

print(df)

model = ols("test_accuracy ~ real_images + synthetic_images + real_images:synthetic_images",
            data=df).fit()
result = sm.stats.anova_lm(model, type=2)

print(result)
