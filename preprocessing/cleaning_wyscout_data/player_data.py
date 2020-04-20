# -*- coding: utf-8 -*-

# import packages
import logging
import codecs
import pandas as pd
import numpy as np

import helper.io as io

logging.basicConfig(level=logging.DEBUG)


def cleanse_wyscout_player_data():
    """
    Function to cleanse the wyscout player data and save the data in the data folder
    :return: None
    """

    logging.info("Cleansing wyscout player data")

    # read the JSON file
    players = io.read_data("player_data", data_folder="raw_data_wyscout")

    # normalize to get a pandas data frame
    df_players = pd.json_normalize(players)

    # make sure the encoding is done correctly
    for col in df_players.select_dtypes("object").columns:
        try:
            df_players[col] = df_players[col].map(lambda x: codecs.unicode_escape_decode(x)[0])
        except TypeError:
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

    df_players.rename(columns={"role.code2": "playerPosition", "foot": "playerStrongFoot", "shortName": "playerName"},
                      inplace=True)

    cols_keep = [col for col in df_players.columns if col.startswith("player")]
    df_players = df_players[cols_keep].copy()

    io.write_data(df_players, data_type="player_data")


if __name__ == "__main__":
    cleanse_wyscout_player_data()
