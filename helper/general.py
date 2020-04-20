# -*- coding: utf-8 -*-

# import packages
import pandas as pd
import numpy as np


def get_table(df_all_matches, week=None, side=None):

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
            matches=("teamId", "count")
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
    if req_cols is None:
        req_cols = ["teamId", "teamName"]
    df = pd.merge(df, df_teams[req_cols], how="left")
    return df


def get_pretty_table(df_all_matches, df_teams, week=None, side=None):
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
