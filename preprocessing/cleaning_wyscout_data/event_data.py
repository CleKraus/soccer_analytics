
import os
import json
import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

PATH = "C://Clemens//Learning//dash//soccer_dashboard//data//raw_data//wyscout_data"
PATH_CLEAN = "C://Clemens//Learning//dash//soccer_dashboard//data"

country = "Spain"
fname_events = f"events/events_{country}.json"
fname_tags = "tags.csv"
fname_out = f"events_{country.lower()}.parquet"

# read the different tags
tags = pd.read_csv(os.path.join(PATH, fname_tags), sep=";")
dict_tags = {row["Tag"]: row["Description"] for _, row in tags.iterrows()}

# read the JSON file
with open(os.path.join(PATH, fname_events)) as json_file:
    events = json.load(json_file)

# normalize to get a pandas data frame
df_events = pd.json_normalize(events)

# save positions in different columns
df_events["posBeforeX"] = df_events["positions"].map(lambda x: x[0]["x"])
df_events["posBeforeY"] = df_events["positions"].map(lambda x: x[0]["y"])
df_events["posAfterX"] = df_events["positions"].map(lambda x: x[1]["x"] if len(x) > 1 else np.nan)
df_events["posAfterY"] = df_events["positions"].map(lambda x: x[1]["y"] if len(x) > 1 else np.nan)

# save tags in different columns
df_events["tags"] = df_events["tags"].map(lambda x: [tag["id"] for tag in x])
for key in dict_tags:
    df_events[dict_tags[key]] = 1*df_events["tags"].map(lambda x: key in x)

# drop columns that are not needed
df_events.drop(["positions", "tags"], axis=1, inplace=True)

num_cols = ["subEventId"]
for col in num_cols:
    df_events[col] = pd.to_numeric(df_events[col], errors="coerce")

df_events.to_parquet(os.path.join(PATH_CLEAN, fname_out))
