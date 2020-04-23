# -*- coding: utf-8 -*-

# import packages
import logging
import pandas as pd
import numpy as np

import helper.io as io

logging.basicConfig(level=logging.DEBUG)


def get_team_view(matches, team):

    opp_team = 1 if team == 0 else 0

    # save relevant information in data frame
    df_team_view = pd.DataFrame()

    df_team_view["matchId"] = [match["wyId"] for match in matches]
    df_team_view["gameweek"] = [match["gameweek"] for match in matches]
    df_team_view["dateutc"] = [match["dateutc"] for match in matches]

    # get the 2 teams with their side (home / away) and score
    df_team_view["teamId"] = [match["teamsData"][list(match["teamsData"])[team]]["teamId"] for match in matches]
    df_team_view["side"] = [match["teamsData"][list(match["teamsData"])[team]]["side"] for match in matches]
    df_team_view["score"] = [match["teamsData"][list(match["teamsData"])[team]]["score"] for match in matches]
    df_team_view["oppTeamId"] = [match["teamsData"][list(match["teamsData"])[opp_team]]["teamId"] for match in matches]
    df_team_view["oppScore"] = [match["teamsData"][list(match["teamsData"])[opp_team]]["score"] for match in matches]

    return df_team_view


def get_all_formations(matches):
    lst_formations = list()
    for match in matches:

        match_id = match["wyId"]

        for team in [0, 1]:
            team = match["teamsData"][list(match["teamsData"])[team]]
            team_id = team["teamId"]

            player_bench = [player["playerId"] for player in team["formation"]["bench"]]
            df_bench = pd.DataFrame()
            df_bench["playerId"] = player_bench
            df_bench["lineup"] = 0

            player_lineup = [player["playerId"] for player in team["formation"]["lineup"]]
            df_lineup = pd.DataFrame()
            df_lineup["playerId"] = player_lineup
            df_lineup["lineup"] = 1

            player_in = [sub["playerIn"] for sub in team["formation"]["substitutions"]]
            player_out = [sub["playerOut"] for sub in team["formation"]["substitutions"]]
            sub_minute = [sub["minute"] for sub in team["formation"]["substitutions"]]

            df_player_in = pd.DataFrame()
            df_player_in["playerId"] = player_in
            df_player_in["substituteIn"] = sub_minute

            df_player_out = pd.DataFrame()
            df_player_out["playerId"] = player_out
            df_player_out["substituteOut"] = sub_minute

            df_formation = pd.concat([df_lineup, df_bench], axis=0)
            df_formation["matchId"] = match_id
            df_formation["teamId"] = team_id
            df_formation = pd.merge(df_formation, df_player_in, how="left")
            df_formation = pd.merge(df_formation, df_player_out, how="left")

            lst_formations.append(df_formation)

    all_formations = pd.concat(lst_formations)

    # get the minute the player started and the minute the player ended the match
    all_formations["minuteStart"] = np.where(all_formations["substituteIn"].isnull(), 0, all_formations["substituteIn"])
    all_formations["minuteEnd"] = np.where(all_formations["substituteOut"].isnull(), 90, all_formations["substituteOut"])

    # make sure the match always lasts 90 minutes
    all_formations["minuteStart"] = np.minimum(all_formations["minuteStart"], 90)
    all_formations["minuteEnd"] = np.minimum(all_formations["minuteEnd"], 90)

    # set minuteEnd to 0 in case the player was not in the lineup and did not get substituted in
    all_formations["minuteEnd"] = np.where((all_formations["lineup"] == 0) & (all_formations["substituteIn"].isnull()),
                                           0, all_formations["minuteEnd"])

    # compute the minutes played
    all_formations["minutesPlayed"] = all_formations["minuteEnd"] - all_formations["minuteStart"]

    # use a binary flag of substitution rather than a minute and NaNs
    all_formations["substituteIn"] = 1 * (all_formations["substituteIn"].notnull())
    all_formations["substituteOut"] = 1 * (all_formations["substituteOut"].notnull())

    return all_formations


def cleanse_wyscout_match_data(country):
    """
    Function to cleanse the wyscout match data and save it in the data folder
    :param country: (str) Country for which the event data should be cleansed
    :return: None
    """

    logging.info(f"Cleansing wyscout match data for {country}")

    # read the JSON file with matches
    matches = io.read_data("match_data", league=country, data_folder="raw_data_wyscout")

    # save relevant information in data frame
    df_matches = pd.concat([get_team_view(matches, 0), get_team_view(matches, 1)], axis=0)

    # attach the points per team
    df_matches["points"] = np.where(df_matches["score"] > df_matches["oppScore"], 3,
                                    np.where(df_matches["score"] == df_matches["oppScore"], 1, 0))

    df_matches["dateutc"] = pd.to_datetime(df_matches["dateutc"])

    df_matches["scoreDiff"] = df_matches["score"] - df_matches["oppScore"]

    df_matches.sort_values(["matchId", "side"], ascending=[True, False], inplace=True)
    io.write_data(df_matches, "match_data", league=country.lower())

    df_formations = get_all_formations(matches)
    io.write_data(df_formations, "formation_data", league=country.lower())


if __name__ == "__main__":

    cleanse_wyscout_match_data("Germany")
