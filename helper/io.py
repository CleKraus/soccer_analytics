# -*- coding: utf-8 -*-

# import packages
import json
import os
import pickle
import warnings
import csv
from pathlib import Path

import pandas as pd
import numpy as np
import ruamel.yaml

import helper.machine_learning as ml_help

PROJECT_NAME = "soccer_analytics"
CONFIG_NAME = "config.yml"
ALL_LEAGUES = ["germany", "italy", "england", "spain", "france"]


def _update_project_path():
    """
    Helper function to update the project path in the config file
    """
    # get the current config file
    with open(_get_config_file(), "r", encoding="utf-8") as f:
        config = ruamel.yaml.YAML().load(f)

        # get the current path
        base_path = Path(__file__).parent
        current_path = (base_path / "..").resolve()
        config["general"]["project_path"] = str(current_path)

    # update the config file
    with open(_get_config_file(), "w", encoding="utf-8") as f:
        yaml = ruamel.yaml.YAML()
        yaml.default_flow_style = False
        yaml.dump(config, f)

    return str(current_path)


def _get_config_file():
    """
    Helper function to retrieve the name of the config-file
    :return: String with the config file name
    """

    base_path = Path(__file__).parent
    project_path = (base_path / "..").resolve()

    return os.path.join(project_path, CONFIG_NAME)


def _read_file(path, sep=None):
    """
    Helper function to read a specific file by automatically extracting the file type and using the appropriate
    reader. Types that are currently supported include parquet, json and csv
    """
    file_type = path.rsplit(".")[-1]

    if file_type == "parquet":
        file = pd.read_parquet(path)
    elif file_type == "json":
        with open(path) as json_file:
            file = json.load(json_file)
    elif file_type == "csv":
        if sep is None:
            raise ValueError(f"csv-files require a separator")
        file = pd.read_csv(path, sep=sep)
    else:
        raise ValueError(f"File type {file_type} currently not supported")

    return file


def read_data(data_type, league=None, sep=None, data_folder=None):
    """
    Function to read data specified in the config file under "data"
    :param data_type: (str) Type of data (event, match, player, ...) to be read. Needs to exactly match the name in
                      the config file
    :param league: (str) League to be read in case there are different files (e.g. Germany, England, ...)
    :param sep: (str) Separator in case a csv-file is read
    :param data_folder: (str) In case any other data folder than "data" from the config file is required, it should be
                        specified here (e.g. when reading the raw_wyscout data, one should set *data_folder* to
                        "raw_data_wyscout"
    :return: Data from the specified *data_type*. Can be pd.DataFrame or dict depending on whether a parquet/csv or a
             JSON file was specified
    """

    # read the config file
    with open(_get_config_file(), "r", encoding="utf-8") as f:
        config = ruamel.yaml.YAML().load(f)

    # extract the project path
    project_path = config["general"]["project_path"]

    # set the data_folder
    if data_folder is None:
        data_folder = "data"

    # make sure the data type is valid
    try:
        fname = config[data_folder][data_type]
    except KeyError:
        raise KeyError(f"'{data_type}' is not a valid data type")

    # replace xxxxx by the league name
    if "xxxxx" in fname:
        if league is None:
            raise ValueError(f"Data type {data_type} requires a league")
        else:
            fname = fname.replace("xxxxx", league)

    full_path = os.path.join(project_path, config[data_folder]["path"], fname)

    # notice that if we have an issue with finding the file, we first try to update
    # the project path
    if not os.path.exists(full_path):
        project_path = _update_project_path()
        full_path = os.path.join(project_path, config[data_folder]["path"], fname)

    # raw tracking data needs to be read differently
    if "team_tracking" in data_type and data_folder == "raw_data_metrica":
        data = read_raw_tracking_data(full_path, data_type.split("_")[0])
    else:
        data = _read_file(full_path, sep)

    return data


def read_event_data(league, notebook=None):
    """
    Reads the event data of the specified league. If *notebook* is set and defined below, only the required subset
    of the event data is returned
    :param league: (str) League for which the event data should be read; if "all", all leagues are returned
    :param notebook: (str, optional) If specified, only the subset of the event data required for the *notebook* is
                      returned
    :return: pd.DataFrame with the event data
    """

    # if league is specified, read events only for this league
    if league != "all":
        df = read_data("event_data", league)
    # else read data for all leagues
    else:
        lst_events = list()
        for league in ALL_LEAGUES:
            lst_events.append(read_data("event_data", league))
        df = pd.concat(lst_events)

    # only return the columns needed for the goal kick analysis
    if notebook == "goal_kick_analysis":
        cols = [
            "id",
            "matchId",
            "matchPeriod",
            "eventSec",
            "eventName",
            "subEventName",
            "teamId",
            "posBeforeXMeters",
            "posBeforeYMeters",
            "posAfterXMeters",
            "posAfterYMeters",
            "playerId",
            "playerName",
            "playerPosition",
            "homeTeamId",
            "awayTeamId",
        ]

        df = df[cols].copy()

    elif notebook == "expected_goal_model":

        # get one variable out the the three "body part" columns
        df = ml_help.from_dummy(
            df, ["leftFoot", "rightFoot", "head/body"], "bodyPartShot"
        )

        # encode the column with a number
        dict_body_part = {"Unknown": 0, "leftFoot": 1, "head/body": 2, "rightFoot": 3}
        df["bodyPartShotCode"] = df["bodyPartShot"].map(lambda x: dict_body_part[x])

        # drop columns that are not needed in this notebook
        drop_cols = [
            "accurate",
            "notAccurate",
            "assist",
            "keyPass",
            "direct",
            "indirect",
            "dangerousBallLost",
            "leftFoot",
            "rightFoot",
            "head/body",
        ]
        df.drop(drop_cols, axis=1, inplace=True)

    return df


def read_team_data(league, notebook=None):
    """
    Reads the team data of the specified league. If *notebook* is set and defined below, only the required subset
    of the team data is returned
    :param league: (str) League for which the team data should be read
    :param notebook: (str, optional) If specified, only the subset of the team data required for the *notebook* is
                      returned
    :return: pd.DataFrame with the team data
    """

    # if league is specified, read team data only for this league
    if league != "all":
        df = read_data("team_data", league)
    # else read data for all leagues
    else:
        lst_teams = list()
        for league in ALL_LEAGUES:
            lst_teams.append(read_data("team_data", league))
        df = pd.concat(lst_teams)

    return df


def read_match_data(league, notebook=None):
    """
    Reads the match data of the specified league. If *notebook* is set and defined below, only the required subset
    of the match data is returned
    :param league: (str) League for which the match data should be read
    :param notebook: (str, optional) If specified, only the subset of the match data required for the *notebook* is
                      returned
    :return: pd.DataFrame with the team data
    """

    # if league is specified, read match data only for this league
    if league != "all":
        df = read_data("match_data", league)
    # else read data for all leagues
    else:
        lst_matches = list()
        for league in ALL_LEAGUES:
            lst_matches.append(read_data("match_data", league))
        df = pd.concat(lst_matches)

    return df


def read_formation_data(league, notebook=None):
    """
    Reads the formation data of the specified league. If *notebook* is set and defined below, only the required subset
    of the formation data is returned
    :param league: (str) League for which the formation data should be read
    :param notebook: (str, optional) If specified, only the subset of the formation data required for the *notebook* is
                      returned
    :return: pd.DataFrame with the formation data
    """

    # if league is specified, read match data only for this league
    if league != "all":
        df = read_data("formation_data", league)
    # else read data for all leagues
    else:
        lst_formations = list()
        for league in ALL_LEAGUES:
            lst_formations.append(read_data("formation_data", league))
        df = pd.concat(lst_formations)

    return df


def read_player_data(notebook=None):
    """
    Reads the player data. If *notebook* is set and defined below, only the required subset
    of the player data is returned
    :param notebook: (str, optional) If specified, only the subset of the player data required for the *notebook* is
                      returned
    :return: pd.DataFrame with the player data
    """

    df = read_data("player_data")

    return df


def read_model(model_name):
    """
    Helper function to read saved models.
    :param model_name: (str) Needs to match the model name in the config file
    :return: Trained machine-learning model
    """

    # read the config file
    with open(_get_config_file(), "r", encoding="utf-8") as f:
        config = ruamel.yaml.YAML().load(f)

    # extract the project path
    project_path = config["general"]["project_path"]
    folder = config["model"]["path"]
    model_name = config["model"][model_name]

    # read the model
    with open(os.path.join(project_path, folder, model_name), "rb") as f:
        model = pickle.load(f)

    return model


def read_metrica_event_data(game, wyscout_format=True):
    """
    Helper function to read the cleaned Metrica event data
    :param game: (int) Identifier of the game (currently only 1 and 2 are there)
    :param wyscout_format: (bool) If True, the event data is returned the same way as in the Wyscout data to be
                            compatible with the helper functions
    :return:
    """

    # read the event data
    df_events = read_data("event_data", league=str(game), data_folder="metrica_data")

    # if event data should be returned in the metrica format
    if not wyscout_format:
        cols = [
            "team",
            "type",
            "subtype",
            "period",
            "startFrame",
            "startTime",
            "endFrame",
            "endTime",
            "from",
            "to",
            "xPosStart",
            "yPosStart",
            "xPosEnd",
            "yPosEnd",
            "goal",
            "ownGoal",
        ]
        df_events = df_events[cols].copy()

        sub_event_col = "subtype"

    # if it should be returned in the Wyscout format
    else:
        warn_message = "Be careful when using Wyscout format. Not everything might be converted from Metrica to Wyscout"
        warnings.warn(warn_message, category=ImportWarning)

        # rename some columns to match the wyscout format
        cols_new = {
            "startTime": "eventSec",
            "endTime": "eventSecEnd",
            "from": "playerName",
            "to": "toPlayerName",
            "xPosStart": "posBeforeXMeters",
            "yPosStart": "posBeforeYMeters",
            "xPosEnd": "posAfterXMeters",
            "yPosEnd": "posAfterYMeters",
        }

        df_events = df_events.rename(columns=cols_new)

        # only keep the relevant columns
        cols = [
            "id",
            "matchId",
            "matchPeriod",
            "eventSec",
            "eventSecEnd",
            "startFrame",
            "endFrame",
            "eventName",
            "subEventName",
            "teamId",
            "team",
            "posBeforeXMeters",
            "posBeforeYMeters",
            "posAfterXMeters",
            "posAfterYMeters",
            "playerId",
            "playerName",
            "playerPosition",
            "toPlayerId",
            "toPlayerName",
            "homeTeamId",
            "awayTeamId",
            "accurate",
            "goal",
            "ownGoal",
        ]
        df_events = df_events[cols].copy()

        # do not keep two columns for the set pieces
        df_set = df_events[df_events["eventName"] == "Set piece"].copy()
        df_no_set = df_events[df_events["eventName"] != "Set piece"].copy()

        df_set.sort_values(["startFrame", "endFrame"], inplace=True)
        df_set.drop_duplicates("startFrame", keep="last", inplace=True)

        df_events = pd.concat([df_set, df_no_set])
        df_events.sort_values(["startFrame", "endFrame"])
        df_events["id"] = np.arange(len(df_events))

        sub_event_col = "subEventName"

        df_formations = read_data(
            "formation_data", league=str(game), data_folder="metrica_data"
        )

    df_events[sub_event_col] = np.where(
        df_events[sub_event_col] == "  ", np.nan, df_events[sub_event_col]
    )
    df_events.sort_values(["startFrame", "endFrame"], inplace=True)
    df_events.reset_index(inplace=True, drop=True)

    # return both event data and formation data in case of the Wyscout format
    if wyscout_format:
        return df_events, df_formations
    else:
        return df_events


def read_tracking_data(game, clean=True):
    """
    Read the Metrica tracking data
    :param game: (int) Game identifier
    :param clean: (bool) If True, the cleaned positions of the ball are returned
    :return: pd.DataFrame with the tracking data of the *game*
    """

    # read the event data
    df = read_data("tracking_data", league=str(game), data_folder="metrica_data")

    cols = ["frame", "time", "period", "xPos", "yPos", "playerId", "team"]

    if not clean:

        df["xPos"] = df["xPosMetrica"]
        df["yPos"] = df["yPosMetrica"]

    else:

        cols += ["xPosMetrica", "yPosMetrica", "ballInPlay"]

        # the first frames in game number 1 are messed up
        if game == 1:
            df = df[df["frame"] >= 5].copy()

    return df[cols].copy()


def read_raw_tracking_data(full_path, teamname):
    """
    Read the raw tracking Metrica tracking data (function mostly copied from Laurie's solution on FoT
    :param full_path: (str) path were the tracking data is stored
    :param teamname: (str) Name of the team, either "Home" or "Away"
    :return: pd.DataFrame with the raw tracking data
    """

    # First:  deal with file headers so that we can get the player names correct
    csvfile = open(full_path, "r")

    reader = csv.reader(csvfile)

    _ = next(reader)[3].lower()

    # construct column names
    ########################

    # extract player jersey numbers from second row
    jerseys = [x for x in next(reader) if x != ""]
    columns = next(reader)

    # create x & y position column headers for each player
    teamname = teamname[0].upper() + teamname[1:]
    for i, j in enumerate(jerseys):
        columns[i * 2 + 3] = "{}_{}_x".format(teamname, j)
        columns[i * 2 + 4] = "{}_{}_y".format(teamname, j)

    # column headers for the x & y positions of the ball
    columns[-2] = "ball_x"
    columns[-1] = "ball_y"

    # Second: read in tracking data and place into pandas Dataframe
    tracking = pd.read_csv(full_path, names=columns, skiprows=3)
    return tracking


def read_ground_pass_probs(accurate=None, air=None):
    """
    Helper function to read the probabilities for passes being passes played on the ground. This is used in the
    notebook "pass_probability_model.ipynb" to not have to recompute all probabilities.
    """

    df_pass = read_data("ground_pass_probs", data_folder="metrica_data")

    if accurate is not None:
        if accurate:
            df_pass = df_pass[df_pass["accurate"] == 1].copy()
        else:
            df_pass = df_pass[df_pass["accurate"] == 0].copy()

    if air is not None:
        if air:
            cols = ["id", "startFrame", "bestSpeedAir", "bestAngleAir", "probAir"]
        else:
            cols = ["id", "startFrame", "bestSpeedGround", "probGround"]

        df_pass = df_pass[cols].copy()

    return df_pass


def write_data(df, data_type, league=None, data_folder=None):
    """
    Function to save *df* in the path specified in the config file under "data" (or *data_folder* if specified)
    :param df: (pd.DataFrame) Data frame to write as parquet file
    :param data_type: (str) Type of data (event, match, player, ...) to be written. Needs to exactly match the name in
                      the config file
    :param league: (str) League to be read in case there are different files (e.g. Germany, England, ...)
    :param data_folder: (str) In case any other data folder than "data" from the config file is required, it should be
                        specified here
    :return: None
    """

    # read the config file
    with open(_get_config_file(), "r", encoding="utf-8") as f:
        config = ruamel.yaml.YAML().load(f)

    # extract the project path
    project_path = config["general"]["project_path"]

    # notice that if the project path does not exist, we update it with the current path
    if not os.path.exists(project_path):
        project_path = _update_project_path()

    # set the data_folder
    if data_folder is None:
        data_folder = "data"

    # make sure the data type is valid
    try:
        fname = config[data_folder][data_type]
    except KeyError:
        raise KeyError(
            f"'{data_type}' is not a valid data type. Please set in the config file"
        )

    # replace xxxxx by the league name
    if "xxxxx" in fname:
        if league is None:
            raise ValueError(f"Data type {data_type} requires a league")
        else:
            fname = fname.replace("xxxxx", league)

    full_path = os.path.join(project_path, config[data_folder]["path"], fname)

    df.to_parquet(full_path)


def save_model(model, model_name):
    """
    Helper function to save models as pickle files.
    :param model: Machine-learning model to be saved
    :param model_name: (str) Needs to match the model name in the config file
    :return: None
    """

    # read the config file
    with open(_get_config_file(), "r", encoding="utf-8") as f:
        config = ruamel.yaml.YAML().load(f)

    # extract the project path
    project_path = config["general"]["project_path"]
    folder = config["model"]["path"]
    model_name = config["model"][model_name]

    with open(os.path.join(project_path, folder, model_name), "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    _update_project_path()
