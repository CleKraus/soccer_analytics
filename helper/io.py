# -*- coding: utf-8 -*-

# import packages
import os
import ruamel.yaml
import pandas as pd
import numpy as np
import json

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
        current_path = os.getcwd()
        project_name = config["general"]["project_name"]

        # update the project path
        project_path = current_path.rsplit(project_name)[0] + project_name
        config["general"]["project_path"] = project_path

    # update the config file
    with open(_get_config_file(), "w", encoding="utf-8") as f:
        yaml = ruamel.yaml.YAML()
        yaml.default_flow_style = False
        yaml.dump(config, f)

    return project_path


def _get_config_file():
    """
    Helper function to retrieve the name of the config-file
    :return: String with the config file name
    """

    path = os.getcwd()
    project_path = path.rsplit(PROJECT_NAME)[0] + PROJECT_NAME

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
    :param league: (str) League to be read in case there are different files (e.g. Germany, Englang, ...)
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
        cols = ["id", "matchId", "matchPeriod", "eventSec", "eventName", "subEventName", "teamId", "posBeforeXMeters",
                "posBeforeYMeters", "posAfterXMeters", "posAfterYMeters", "playerId", "playerName", "playerPosition",
                "homeTeamId", "awayTeamId"]

        df = df[cols].copy()

    elif notebook == "expected_goal_model":

        # get one variable out the the three "body part" columns
        df = ml_help.from_dummy(df, ["leftFoot", "rightFoot", "head/body"], "bodyPartShot")

        # encode the column with a number
        dict_body_part = {"Unknown": np.nan,
                          "leftFoot": 1,
                          "head/body": 2,
                          "rightFoot": 3}
        df["bodyPartShotCode"] = df["bodyPartShot"].map(lambda x: dict_body_part[x])

        # drop columns that are not needed in this notebook
        drop_cols = ["accurate", "notAccurate", "assist", "keyPass", "direct", "indirect",
                     "dangerousBallLost", "leftFoot", "rightFoot", "head/body"]
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


def write_data(df, data_type, league=None, data_folder=None):
    """
    Function to save *df* in the path specified in the config file under "data" (or *data_folder* if specified)
    :param df: (pd.DataFrame) Data frame to write as parquet file
    :param data_type: (str) Type of data (event, match, player, ...) to be read. Needs to exactly match the name in
                      the config file
    :param league: (str) League to be read in case there are different files (e.g. Germany, Englang, ...)
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
        raise KeyError(f"'{data_type}' is not a valid data type. Please set in the config file")

    # replace xxxxx by the league name
    if "xxxxx" in fname:
        if league is None:
            raise ValueError(f"Data type {data_type} requires a league")
        else:
            fname = fname.replace("xxxxx", league)

    full_path = os.path.join(project_path, config[data_folder]["path"], fname)

    df.to_parquet(full_path)
