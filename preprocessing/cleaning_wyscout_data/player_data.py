
import os
import json
import codecs
import pandas as pd
import numpy as np
from pandas.io.json import json_normalize

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

PATH = "C://Clemens//Learning//dash//soccer_dashboard//data//raw_data//wyscout_data"
PATH_CLEAN = "C://Clemens//Learning//dash//soccer_dashboard//data"

fname_players = "players.json"
fname_out = "players.parquet"

# read the JSON file
with open(os.path.join(PATH, fname_players)) as json_file:
    players = json.load(json_file)

# normalize to get a pandas data frame
df_players = json_normalize(players)

# make sure the encoding is done correctly
for col in df_players.select_dtypes("object").columns:
    try:
        df_players[col] = df_players[col].map(lambda x: codecs.unicode_escape_decode(x)[0])
    except:
        pass

# rename to playerId so that it can be easily merged with other tables
df_players.rename(columns={"wyId": "playerId"}, inplace=True)

df_players["birthDate"] = pd.to_datetime(df_players["birthDate"])
df_players["weight"] = np.where(df_players["weight"] > 0, df_players["weight"], np.nan)
df_players["height"] = np.where(df_players["height"] > 0, df_players["height"], np.nan)
df_players["foot"] = np.where(df_players["foot"].isin(["null", ""]), "unknown", df_players["foot"])

id_cols = ["currentTeamId", "currentNationalTeamId"]
for col in id_cols:
    df_players[col] = pd.to_numeric(df_players[col], errors="coerce")

# drop duplicates columns that are not needed
drop_cols = ["birthArea.alpha3code", "birthArea.alpha2code", "role.code3", "role.name", "passportArea.alpha3code",
             "passportArea.alpha2code", "middleName", "birthArea.id", "passportArea.id"]
df_players.drop(drop_cols, axis=1, inplace=True)

"""
profile = ProfileReport(df_players, title='Player table overview', html={'style': {'full_width': True}}, minimal=True)
profile.to_file(output_file=os.path.join(PATH_PROFILING, "players.html"))
"""

df_players.to_parquet(os.path.join(PATH_CLEAN, fname_out))
