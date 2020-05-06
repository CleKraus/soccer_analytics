# -*- coding: utf-8 -*-

# import packages
import numpy as np
import pandas as pd


def get_table(df_all_matches, week=None, side=None):
    """
    Function to compute the table of the league based on all matches in *df_all_matches*
    :param df_all_matches: (pd.DataFrame) Data frame containing all the matches
    :param week: (int) If not None, only matches <= *week* are considered for the calculation
    :param side: (str) Can only contain values None, "Home" or "Away". If not None, home or away table are calculated
    :return: pd.DataFrame containing the table of the league
    """

    if week is None:
        week = max(df_all_matches["gameweek"])

    if side is not None:
        df_all_matches = df_all_matches[df_all_matches["side"] == side]

    df_played_matches = df_all_matches[df_all_matches["gameweek"] <= week]

    df_standing = (
        df_played_matches.groupby("teamId")
        .agg(
            goals=("score", "sum"),
            concededGoals=("oppScore", "sum"),
            goalsDiff=("scoreDiff", "sum"),
            points=("points", "sum"),
            matches=("teamId", "count"),
        )
        .reset_index()
    )
    df_standing.sort_values(
        ["points", "goalsDiff", "goals"], ascending=False, inplace=True
    )
    df_standing.reset_index(inplace=True, drop=True)
    df_standing["week"] = week
    df_standing["position"] = np.arange(1, len(df_standing) + 1)
    return df_standing


def attach_team_name(df, df_teams, req_cols=None):
    """
    Helper function to attach the team name stored in *df_teams* to a data frame *df* containing a teamId
    """
    if req_cols is None:
        req_cols = ["teamId", "teamName"]
    df = pd.merge(df, df_teams[req_cols], how="left")
    return df


def get_pretty_table(df_all_matches, df_teams, week=None, side=None):
    """
    Wrapper around *get_table* function to get pretty table of the league.
    :param df_all_matches: (pd.DataFrame) Data frame containing all matches
    :param df_teams: (pd.DataFrame) Data frame containing all team names
    :param week: (int) If not None, only matches <= *week* are considered for the calculation
    :param side: (str) Can only contain values None, "Home" or "Away". If not None, home or away table are calculated
    :return: pd.DataFrame with a pretty table of the league
    """
    df_table = get_table(df_all_matches, week, side)
    df_table = attach_team_name(df_table, df_teams)
    cols = [
        "position",
        "teamName",
        "matches",
        "goals",
        "concededGoals",
        "goalsDiff",
        "points",
    ]
    df_table = df_table[cols].copy()
    return df_table


def rotate_vectors(x, y, radians):
    """
    Helper function to rotate 2-dim vectors by *radians*. The vectors need to be passed to the
    function through their x and y coordinates.
    """
    c, s = np.cos(radians), np.sin(radians)
    j = np.matrix([[c, s], [-s, c]])
    m = np.dot(j, [x, y])

    return np.array(m[0].ravel())[0], np.array(m[1].ravel())[0]


def compute_gini(array):
    """
    Given an *array* with values the function computes the Gini coefficient of these values
    """
    # convert to np.array in case of a pd.Series
    if type(array) == pd.core.series.Series:
        array = array.to_numpy()

    # array = array.flatten()
    # sort the values
    array = np.sort(array)

    # set index and number of elements
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]

    # return the Gini index
    return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))
