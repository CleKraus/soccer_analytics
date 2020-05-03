# -*- coding: utf-8 -*-

# import packages
import codecs
import logging

import pandas as pd

import helper.general as gen_helper
import helper.io as io

logging.basicConfig(level=logging.DEBUG)


def cleanse_wyscout_team_data(country):
    """
    Function to cleanse the wyscout team data and save the data in the data folder
    :param country: (str) Country for which the team data should be cleansed
    :return: None
    """

    valid_countries = ["Germany", "England", "Spain", "Italy", "France"]
    if country not in valid_countries:
        raise KeyError(
            f"Country '{country}' not supported. Choose one out of: {', '.join(valid_countries)}"
        )

    logging.info(f"Cleansing wyscout team data for {country}")

    # read the JSON file
    teams = io.read_data("team_data", data_folder="raw_data_wyscout")

    # normalize to get a pandas data frame
    df_teams = pd.json_normalize(teams)

    # make sure the encoding is done correctly
    for col in df_teams.select_dtypes("object").columns:
        try:
            df_teams[col] = df_teams[col].map(
                lambda x: codecs.unicode_escape_decode(x)[0]
            )
        except TypeError:
            pass

    df_teams.rename(
        columns={"wyId": "teamId", "name": "teamName", "area.name": "country"},
        inplace=True,
    )

    # only keep club teams from the specified country
    df_teams = df_teams[
        (df_teams["type"] == "club") & (df_teams["country"] == country)
    ].copy()
    df_teams = df_teams[["teamId", "teamName"]].copy()

    # attach the table to the teams to get a good feeling on how good each team is
    df_matches = io.read_data("match_data", league=country.lower())
    df_table = gen_helper.get_table(df_matches)
    df_table.drop("week", axis=1, inplace=True)
    df_teams = pd.merge(df_teams, df_table, on="teamId", how="left")

    df_teams.sort_values("position", inplace=True)
    df_teams = df_teams[
        [
            "position",
            "teamId",
            "teamName",
            "matches",
            "goals",
            "concededGoals",
            "goalsDiff",
            "points",
        ]
    ].copy()

    io.write_data(df_teams, data_type="team_data", league=country.lower())


if __name__ == "__main__":

    cleanse_wyscout_team_data("Spain")
