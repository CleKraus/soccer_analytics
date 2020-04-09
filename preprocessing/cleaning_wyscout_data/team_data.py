
import os
import json
import codecs
import pandas as pd
from pandas.io.json import json_normalize

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

PATH = "C://Clemens//Learning//dash//soccer_dashboard//data//raw_data//wyscout_data"
PATH_CLEAN = "C://Clemens//Learning//dash//soccer_dashboard//data"

fname_teams = "teams.json"
fname_out = "teams.parquet"

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

df_teams.rename(columns={"wyId": "teamId", "name": "teamName"}, inplace=True)

df_teams.drop(["area.id", "area.alpha3code", "area.alpha2code"], axis=1, inplace=True)

df_teams.to_parquet(os.path.join(PATH_CLEAN, fname_out))