# -*- coding: utf-8 -*-

# import packages
import numpy as np
import pandas as pd


def get_team_id(df_teams, team_name):
    """
    Given the team name, function returns the teamId
    :param df_teams: (pd.DataFrame) Data frame with all the teams
    :param team_name: (str) Name of the team, e.g. Bayern München
    :return: Integer with the teamId
    """
    df_team = df_teams[(df_teams["teamName"] == team_name)]["teamId"]

    if len(df_team) == 1:
        return df_team.values[0]
    else:
        raise ValueError(f"Team '{team_name}' does not exist.")


def get_team_name(df_teams, team_id):
    """
    Given the team id, function returns the name of the team
    :param df_teams: (pd.DataFrame) Data frame with all the teams
    :param team_id: (int) Id of the team, e.g. 2444
    :return: String with the team name
    """
    df_team = df_teams[(df_teams["teamId"] == team_id)]["teamName"]

    if len(df_team) == 1:
        return df_team.values[0]
    else:
        raise ValueError(f"TeamId '{team_id}' does not exist.")


def get_match_id(df_matches, home_team, away_team, df_teams=None):
    """
    Given the home and away team, function returns the matchId. Notice that function accepts both the teamId and
    the teamName for the home and away team.
    :param df_matches: (pd.DataFrame) Data frame with all matches
    :param home_team: (int or str) Either teamId or teamName of the home team
    :param away_team: (int or str) Either teamId or teamName of the away team
    :param df_teams: (pd.DataFrame, optional) Data frame containing all teams; only required if teamName rather than
                      teamId is passed
    :return: Integer with the matchId
    """
    if type(home_team) == str or type(away_team) == str:
        if df_teams is None:
            raise ValueError("Requires *df_teams* to be set")

    home_team_id = (
        get_team_id(df_teams, home_team) if type(home_team) == str else home_team
    )
    away_team_id = (
        get_team_id(df_teams, away_team) if type(away_team) == str else away_team
    )

    return df_matches[
        (df_matches["teamId"] == home_team_id)
        & (df_matches["side"] == "home")
        & (df_matches["oppTeamId"] == away_team_id)
    ]["matchId"].values[0]


def get_time_around_special_event(
    df_events, event_id, secs_before=None, secs_after=None
):
    """
    Given an *event_id* the function returns all the events between the event - *seconds_before* and
    the event + *seconds_after*
    :param df_events: (pd.DataFrame) Data frame with all events
    :param event_id: (int) Event id of the event we are interested in
    :param secs_before: (float, optional) All events *secs_before* seconds before the *event_id* are returned. If None,
                        no events before the *event_id* will be returned
    :param secs_after:  (float, optional) All events *secs_after* seconds after the *event_id* are returned. If None,
                        no events after the *event_id* will be returned
    :return: pd.DataFrame with all events between *secs_before* the *event_id* and *secs_after* the *event_id*
    """
    tmp_event = df_events[df_events["id"] == event_id]
    match_id = tmp_event.iloc[0]["matchId"]
    match_period = tmp_event.iloc[0]["matchPeriod"]
    event_sec = tmp_event.iloc[0]["eventSec"]

    if secs_before is None:
        secs_before = 0.01

    if secs_after is None:
        secs_after = 0.01

    df_special_event = df_events[
        (df_events["matchId"] == match_id)
        & (df_events["matchPeriod"] == match_period)
        & (df_events["eventSec"] >= event_sec - secs_before)
        & (df_events["eventSec"] <= event_sec + secs_after)
    ]
    return df_special_event


def get_event_after(df, df_events, considered_events, cols_return):
    """
    For each event in *df* the function computes the next event happening out of a list of *considered_event*.
    :param df: (pd.DataFrame) Data frame containing all events for which we want to compute when the following event
    :param df_events: (pd.DataFrame) Data frame with all events (should at least contain all events of the types
                      specified in *considered_events*.
    :param considered_events: (list) List containing all events that should be considered when searching for the next
                               event.
    :param cols_return: (dict) Dictionary containing the information of the following event as well as their column
                         name
    :return: pd.DataFrame containing the *df* data frame enlarged with columns on the next events specified in
             *considered events*

    Example: If you wish to get the time and player of the next pass after a goal kick, the parameters might look like
    the following:
        df : Data frame with all goal kicks
        df_events: Data frame with all events
        considered_events: [Pass]
        cols_returned: {eventSec: timeNextPass, playerId: playerNextPass
    """

    # make a copy of *df* as basis for the output
    df_out = df.copy()

    if "eventSec" in cols_return:
        colname_sec = cols_return["eventSec"]
        cols_return.pop("eventSec")
    else:
        colname_sec = None

    df = df[["matchId", "matchPeriod", "id", "eventSec"]].copy()

    if considered_events != "all":
        df_rel_events = df_events[df_events["eventName"].isin(considered_events)].copy()
    else:
        df_rel_events = df_events.copy()

    rel_cols = ["matchId", "matchPeriod", "eventSec"] + list(cols_return.keys())
    df_rel_events = df_rel_events[rel_cols].copy()
    df_rel_events.rename(columns=cols_return, inplace=True)

    df_rel_events.rename(columns={"eventSec": "eventSecRelEvent"}, inplace=True)

    # add all events of the relevant type(s) and only keep those that happened after
    df_all = pd.merge(df, df_rel_events, on=["matchId", "matchPeriod"])
    df_all = df_all[df_all["eventSecRelEvent"] > df_all["eventSec"]].copy()

    # sort the values and only keep the first one
    df_all.sort_values(["id", "eventSecRelEvent"], inplace=True)
    df_event_after = df_all.drop_duplicates("id")

    if colname_sec is not None:
        df_event_after = df_event_after.rename(
            columns={"eventSecRelEvent": colname_sec}
        )
        cols = ["id", colname_sec]
    else:
        cols = ["id"]

    df_event_after = df_event_after[
        cols + [cols_return[key] for key in cols_return.keys()]
    ].copy()

    df_out = pd.merge(df_out, df_event_after, how="left")
    return df_out


# Functions to prepare the statistics
#####################################


def compute_length(df):
    """
    Helper function to compute the length (in m) between the start point of the event and the end point of the event
    """
    df["distX"] = df["posAfterXMeters"] - df["posBeforeXMeters"]
    df["distY"] = df["posAfterYMeters"] - df["posBeforeYMeters"]
    df["lengthMeters"] = np.sqrt(df["distX"] * df["distX"] + df["distY"] * df["distY"])

    df.drop(["distX", "distY"], axis=1, inplace=True)
    return df


def _compute_total_passes(df_events, group_col):
    """
    Helper function to compute total passes in *df_events* per *group_col*
    """
    df_passes = df_events[df_events["eventName"] == "Pass"]
    return df_passes.groupby(group_col).agg(totalPasses=("id", "size")).reset_index()


def _compute_total_accurate_passes(df_events, group_col):
    """
    Helper function to compute total accurate passes in *df_events* per *group_col*
    """
    df_passes = df_events[
        (df_events["eventName"] == "Pass") & (df_events["accurate"] == 1)
    ]
    return (
        df_passes.groupby(group_col)
        .agg(totalAccuratePasses=("id", "size"))
        .reset_index()
    )


def _compute_mean_pass_length(df_events, group_col):
    """
    Helper function to compute mean pass length in *df_events* per *group_col*
    """
    df_passes = df_events[(df_events["eventName"] == "Pass")].copy()
    df_passes = compute_length(df_passes)
    return (
        df_passes.groupby(group_col)
        .agg(meanPassLength=("lengthMeters", "mean"))
        .reset_index()
    )


def _compute_total_shots(df_events, group_col):
    """
    Helper function to compute total shots in *df_events* per *group_col*
    """
    df_shots = df_events[df_events["eventName"] == "Shot"]
    return df_shots.groupby(group_col).agg(totalShots=("id", "size")).reset_index()


def _compute_total_goals(df_events, group_col):
    """
    Helper function to compute total goals in *df_events* per *group_col*
    """
    # do not consider save attempts and reflexes etc.
    df_shots = df_events[df_events["eventName"].isin(["Shot", "Free Kick"])]
    return df_shots.groupby(group_col).agg(totalGoals=("goal", "sum")).reset_index()


def _compute_own_goals(df_events, group_col):
    """
    Helper function to compute own goals in *df_events* per *group_col*
    """
    df_own_goals = df_events[df_events["ownGoal"] == 1].copy()
    df_own_goals["teamId"] = np.where(
        df_own_goals["teamId"] == df_own_goals["homeTeamId"],
        df_own_goals["awayTeamId"],
        df_own_goals["homeTeamId"],
    )
    return (
        df_own_goals.groupby(group_col)
        .agg(totalOwnGoals=("ownGoal", "sum"))
        .reset_index()
    )


def _compute_total_duels(df_events, group_col):
    """
    Helper function to compute total duels in *df_events* per *group_col*
    """
    df_duels = df_events[(df_events["eventName"] == "Duel")]
    return df_duels.groupby(group_col).agg(totalDuels=("id", "size")).reset_index()


def _compute_centroid(df_events, group_col, centroid_events=None):
    """
    Helper function to compute the centroid points in *df_events* per *group_col*
    """
    if centroid_events is not None:
        df_events = df_events[df_events["eventName"].isin(centroid_events)]

    return (
        df_events.groupby(group_col)
        .agg(
            centroidX=("posBeforeXMeters", "mean"),
            centroidY=("posBeforeYMeters", "mean"),
        )
        .reset_index()
    )


def _compute_minutes_played(df_events, group_col, df_formations):
    """
    Helper function to compute the minutes a player played, how often he was in the lineup and how often he got
    substituted in or out
    """
    lst_matches = list(df_events["matchId"].unique())
    formation_matches = df_formations[df_formations["matchId"].isin(lst_matches)].copy()
    return (
        formation_matches.groupby(group_col)
        .agg(
            lineup=("lineup", "sum"),
            substituteIn=("substituteIn", "sum"),
            substituteOut=("substituteOut", "sum"),
            minutesPlayed=("minutesPlayed", "sum"),
        )
        .reset_index()
    )


def compute_statistics(
    df_events,
    group_col,
    keep_kpis="all",
    drop_kpis=None,
    centroid_events=None,
    df_formations=None,
):
    """
    Compute statistics for a player, team or match. Currently the following statistics are supported:
        - totalPasses
        - totalAccuratePasses
        - shareAccuratePasses
        - passLength
        - totalShots
        - totalGoals
        - totalDuels
        - centroid
    :param df_events: (pd.DataFrame) Data frame containing all events that should be considered (if e.g. player
                       statistics for only one match should be computed, one should only include the event data for
                       this match)
    :param group_col: (str or list) Column(s) over which the statistics should be computed.
                      If e.g. *group_col* = "player", we compute totalPasses, totalShots, ... per player
    :param keep_kpis: (list) KPIs that should be computed
    :param drop_kpis: (list) KPIs that should explicitly not be computed
    :param centroid_events: (list) If set, the centroids of the positions are only computed over this event types
    :param df_formations: (pd.DataFrame) Data frame containing the formations of all matches
    :return: pd.DataFrame with the player, team or match statistics
    """
    # set the KPIs that should be computed
    if keep_kpis == "all":
        keep_kpis = [
            "totalPasses",
            "totalAccuratePasses",
            "shareAccuratePasses",
            "passLength",
            "totalShots",
            "totalGoals",
            "totalDuels",
            "centroid",
            "minutesPlayed",
            "totalPasses90",
        ]

    if drop_kpis is not None:
        keep_kpis = [kpi for kpi in keep_kpis if kpi not in drop_kpis]

        # set the group column accordingly
    if group_col == "player":
        group_col = "playerId"
        df_agg = (
            df_events.groupby(group_col)
            .agg(
                playerName=("playerName", "min"),
                playerPosition=("playerPosition", "min"),
                teamId=("teamId", lambda x: x.value_counts().index[0]),
                nbMatches=("matchId", "nunique"),
            )
            .reset_index()
        )
        df_agg = df_agg[df_agg["playerId"] != 0]

    elif group_col == "team":
        group_col = "teamId"
        df_agg = (
            df_events.groupby(group_col)
            .agg(nbMatches=("matchId", "nunique"))
            .reset_index()
        )

    elif group_col == "match":
        group_col = "matchId"
        df_agg = (
            df_events.groupby(group_col)
            .agg(nbMatches=("matchId", "nunique"))
            .reset_index()
        )

    elif group_col == "player_match":
        group_col = ["playerId", "matchId"]
        df_agg = (
            df_events.groupby(group_col)
            .agg(
                playerName=("playerName", "min"),
                playerPosition=("playerPosition", "min"),
                teamId=("teamId", lambda x: x.value_counts().index[0]),
                nbMatches=("matchId", "nunique"),
            )
            .reset_index()
        )

        df_agg = df_agg[df_agg["playerId"] != 0]

    elif group_col == "team_match":
        group_col = ["teamId", "matchId"]
        df_agg = (
            df_events.groupby(group_col)
            .agg(nbMatches=("matchId", "nunique"))
            .reset_index()
        )

    else:
        raise ValueError(
            "*group_col* must be one of 'player', 'team', 'match', 'player_match' or 'team_match'"
        )

    # compute total passes
    if "totalPasses" in keep_kpis:
        df_agg_var = _compute_total_passes(df_events, group_col)
        df_agg = pd.merge(df_agg, df_agg_var, how="left")
        df_agg["totalPasses"].fillna(0, inplace=True)

        # compute accurate passes
    if "totalAccuratePasses" in keep_kpis:
        df_agg_var = _compute_total_accurate_passes(df_events, group_col)
        df_agg = pd.merge(df_agg, df_agg_var, how="left")
        df_agg["totalAccuratePasses"].fillna(0, inplace=True)

    # compute share of accurate passes
    if "shareAccuratePasses" in keep_kpis:

        if "totalPasses" not in keep_kpis:
            df_agg_var = _compute_total_passes(df_events, group_col)
            df_agg = pd.merge(df_agg, df_agg_var, how="left")
            df_agg["totalPasses"].fillna(0, inplace=True)

        if "totalAccuratePasses" not in keep_kpis:
            df_agg_var = _compute_total_accurate_passes(df_events, group_col)
            df_agg = pd.merge(df_agg, df_agg_var, how="left")
            df_agg["totalAccuratePasses"].fillna(0, inplace=True)

        df_agg["shareAccuratePasses"] = np.round(
            df_agg["totalAccuratePasses"] / df_agg["totalPasses"] * 100, 2
        )
        df_agg["shareAccuratePasses"].fillna(0, inplace=True)

        if "totalPasses" not in keep_kpis:
            df_agg.drop("totalPasses", axis=1, inplace=True)
        if "totalAccuratePasses" not in keep_kpis:
            df_agg.drop("totalAccuratePasses", axis=1, inplace=True)

    # compute pass length
    if "passLength" in keep_kpis:
        df_agg_var = _compute_mean_pass_length(df_events, group_col)
        df_agg_var["meanPassLength"] = np.round(df_agg_var["meanPassLength"], 2)
        df_agg = pd.merge(df_agg, df_agg_var, how="left")

    # compute total shots
    if "totalShots" in keep_kpis:
        df_agg_var = _compute_total_shots(df_events, group_col)
        df_agg = pd.merge(df_agg, df_agg_var, how="left")
        df_agg["totalShots"].fillna(0, inplace=True)

    # compute total goals
    if "totalGoals" in keep_kpis:
        df_agg_var = _compute_total_goals(df_events, group_col)
        df_agg = pd.merge(df_agg, df_agg_var, how="left")
        df_agg["totalGoals"].fillna(0, inplace=True)

        if "playerId" not in group_col:
            df_agg_var = _compute_own_goals(df_events, group_col)
            df_agg = pd.merge(df_agg, df_agg_var, how="left")
            df_agg["totalOwnGoals"].fillna(0, inplace=True)
            df_agg["totalGoals"] += df_agg["totalOwnGoals"]
            df_agg.drop("totalOwnGoals", axis=1, inplace=True)

        df_agg["totalGoals"] = df_agg["totalGoals"].astype(int)

    # compute total duels
    if "totalDuels" in keep_kpis:
        df_agg_var = _compute_total_duels(df_events, group_col)
        df_agg = pd.merge(df_agg, df_agg_var, how="left")
        df_agg["totalDuels"].fillna(0, inplace=True)
        df_agg["totalDuels"] = df_agg["totalDuels"].astype(int)

    # only compute minutes played if df_formations is passed and we consider players rather than teams or matches
    if (
        "minutesPlayed" in keep_kpis
        and df_formations is not None
        and (group_col == "playerId" or "playerId" in group_col)
    ):

        df_agg_var = _compute_minutes_played(df_events, group_col, df_formations)
        df_agg = pd.merge(df_agg, df_agg_var, how="left")

    # compute the total number of passes per 90 minutes
    if "totalPasses90" in keep_kpis and "minutesPlayed" in df_agg.columns:
        df_agg["totalPasses90"] = df_agg["totalPasses"] / df_agg["minutesPlayed"] * 90

    # compute centroids
    if "centroid" in keep_kpis:
        df_agg_var = _compute_centroid(df_events, group_col, centroid_events)
        df_agg = pd.merge(df_agg, df_agg_var, how="left")

    return df_agg


def number_of_passes_between_players(df_events, team_id):
    """
    Given events and a *team_id*, function computes the number of accurate passes between any two players.
    :param df_events: (pd.DataFrame) Data frame with all events - notice, that it should NOT only contain the passes
    :param team_id: (int) Id of the team we are interested in
    :return: pd.DataFrame with the number of passes between any two players
    """
    # keep only the events of the team we are interested in
    df_events = df_events[df_events["teamId"] == team_id].copy()
    df_events.sort_values(["matchId", "matchPeriod", "eventSec"], inplace=True)

    # get the player who had the ball next - grouping is required to make sure we stay in the same
    # match and matchPeriod
    df_events["nextPlayerId"] = df_events.groupby(["matchId", "matchPeriod"])[
        "playerId"
    ].shift(-1)

    # only keep the accurate passes -> for the inaccurate ones we do not trust that the next player of the
    # team was really the one for which the pass was intented
    df_passes = df_events[
        (df_events["eventName"] == "Pass") & (df_events["accurate"] == 1)
    ].copy()

    # get the number of passes between two players - we do this by considering passes in both directions, i.e. from
    # player A to player B and vice versa
    df_passes["player1Id"] = np.where(
        df_passes["playerId"] < df_passes["nextPlayerId"],
        df_passes["playerId"],
        df_passes["nextPlayerId"],
    )

    df_passes["player2Id"] = np.where(
        df_passes["playerId"] < df_passes["nextPlayerId"],
        df_passes["nextPlayerId"],
        df_passes["playerId"],
    )

    df_passes = (
        df_passes.groupby(["player1Id", "player2Id"])
        .agg(totalPasses=("player1Id", "count"))
        .reset_index()
    )

    return df_passes


def compute_current_standing(df_events):
    """
    Compute the current standing of the match(es) for each event
    :param df_events: (pd.DataFrame) Data frame containing all events (or at least all goals and own goals)
    :return: *df_events* containing 3 additional columns: currentGoalsHomeTeam, currentGoalsAwayTeam and the difference
    in score from the perspective of the team of the event
    """
    df = df_events.copy()

    df.sort_values(["matchId", "matchPeriod", "eventSec"], inplace=True)

    # indicate for each event whether the home team scored a goal (no own goals)
    df["goalHomeTeam"] = (
        df["goal"]
        * (df["eventName"].isin(["Shot", "Free Kick"]))
        * (df["teamId"] == df["homeTeamId"])
    )

    # indicate for each event whether the away team scored a goal (no own goals)
    df["goalAwayTeam"] = (
        df["goal"]
        * (df["eventName"].isin(["Shot", "Free Kick"]))
        * (df["teamId"] == df["awayTeamId"])
    )

    # indicate for each event whether it was an own goal
    df["ownGoalHomeTeam"] = df["ownGoal"] * (df["teamId"] == df["homeTeamId"])
    df["ownGoalAwayTeam"] = df["ownGoal"] * (df["teamId"] == df["awayTeamId"])

    # indicate for each event whether a goal happened
    df["goalHomeTeam"] = df["goalHomeTeam"] + df["ownGoalAwayTeam"]
    df["goalAwayTeam"] = df["goalAwayTeam"] + df["ownGoalHomeTeam"]

    # cumulate the goals for each team up to the current event
    df["currentGoalsHomeTeam"] = df.groupby("matchId")["goalHomeTeam"].cumsum()
    df["currentGoalsAwayTeam"] = df.groupby("matchId")["goalAwayTeam"].cumsum()

    # make sure the current event is not counted (we want to have the score before the event)
    df["currentGoalsHomeTeam"] = df["currentGoalsHomeTeam"] - df["goalHomeTeam"]
    df["currentGoalsAwayTeam"] = df["currentGoalsAwayTeam"] - df["goalAwayTeam"]

    # get the relative score from the perspective of the team of the event
    df["currentRelativeScore"] = np.where(
        df["teamId"] == df["homeTeamId"],
        df["currentGoalsHomeTeam"] - df["currentGoalsAwayTeam"],
        df["currentGoalsAwayTeam"] - df["currentGoalsHomeTeam"],
    )

    df.drop(
        ["goalHomeTeam", "goalAwayTeam", "ownGoalHomeTeam", "ownGoalAwayTeam"],
        axis=1,
        inplace=True,
    )
    return df
