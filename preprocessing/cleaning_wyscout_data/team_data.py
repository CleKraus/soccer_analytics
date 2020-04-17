
import os
import json
import codecs
import pandas as pd
from pandas import json_normalize

import helper.general as gen_helper

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

PATH = "C://Clemens//Learning//dash//soccer_dashboard//data//raw_data//wyscout_data"
PATH_CLEAN = "C://Clemens//Learning//dash//soccer_dashboard//data"

country = "Germany"
fname_teams = "teams.json"
fname_matches = f"matches_{country.lower()}.parquet"
fname_out = f"teams_{country.lower()}.parquet"

# read the JSON file
with open(os.path.join(PATH, fname_teams)) as json_file:
    teams = json.load(json_file)

# normalize to get a pandas data frame
df_teams = json_normalize(teams)

# make sure the encoding is done correctly
for col in df_teams.select_dtypes("object").columns:
    try:
        df_teams[col] = df_teams[col].map(lambda x: codecs.unicode_escape_decode(x)[0])
    except:
        pass

df_teams.rename(columns={"wyId": "teamId", "name": "teamName", "area.name": "country"}, inplace=True)

# only keep club teams from the specified country
df_teams = df_teams[(df_teams["type"] == "club") & (df_teams["country"] == country)].copy()
df_teams = df_teams[["teamId", "teamName"]].copy()

# attach the table to the teams to get a good feeling on how good each team is
df_matches = pd.read_parquet(os.path.join(PATH_CLEAN, fname_matches))
df_table = gen_helper.get_table(df_matches)
df_table.drop("week", axis=1, inplace=True)
df_teams = pd.merge(df_teams, df_table, on="teamId", how="left")

df_teams.sort_values("position", inplace=True)
df_teams = df_teams[["position", "teamId", "teamName", "matches", "goals", "counterGoals",
                     "goalsDiff", "points"]].copy()

df_teams.to_parquet(os.path.join(PATH_CLEAN, fname_out))
