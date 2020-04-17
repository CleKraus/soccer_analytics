# -*- coding: utf-8 -*-

# import packages
import os
import ruamel.yaml
import pandas as pd


def _update_project_path():
    """
    Helper function to update the project path in the config file
    """
    # get the current config file
    with open("config.yml", "r", encoding="utf-8") as f:
        config = ruamel.yaml.YAML().load(f)

        # get the current path
        current_path = os.getcwd()
        project_name = config["general"]["project_name"]

        # update the project path
        project_path = current_path.rsplit(project_name)[0] + project_name
        config["general"]["project_path"] = project_path

    # update the config file
    with open("config.yml", "w", encoding="utf-8") as f:
        yaml = ruamel.yaml.YAML()
        yaml.default_flow_style = False
        yaml.dump(config, f)

    return project_path


def _read_file(path, sep=None):
    """
    Helper function to read a specific file by automatically extracting the file type and using the appropriate
    reader. Types that are currently supported include parquet, json and csv
    """
    file_type = path.rsplit(".")[-1]

    if file_type == "parquet":
        file = pd.read_parquet(path)
    elif file_type == "json":
        pass
    elif file_type == "csv":
        if sep is None:
            raise ValueError(f"csv-files require a separator")
        file = pd.read_csv(path, sep=sep)
    else:
        raise ValueError(f"File type {file_type} currently not supported")

    return file


def read_data(data_type, league=None, sep=None):
    """
    Function to read data specified in the config file under "data"
    :param data_type: (str) Type of data (event, match, player, ...) to be read. Needs to exactly match the name in
                      the config file
    :param league: (str) League to be read in case there are different files (e.g. Germany, Englang, ...)
    :param sep: (str) Separator in case a csv-file is read
    :return: Data from the specified *data_type*. Can be pd.DataFrame or dict depending on whether a parquet/csv or a
             JSON file was specified
    """

    # read the config file
    with open("config.yml", "r", encoding="utf-8") as f:
        config = ruamel.yaml.YAML().load(f)

    # extract the project path
    project_path = config["general"]["project_path"]

    # make sure the data type is valid
    try:
        fname = config["data"][data_type]
    except KeyError:
        raise KeyError(f"'{data_type}' is not a valid data type")

    # replace xxxxx by the league name
    if "xxxxx" in fname:
        if league is None:
            raise ValueError(f"Data type {data_type} requires a league")
        else:
            fname = fname.replace("xxxxx", league.lower())

    full_path = os.path.join(project_path, config["data"]["path"], fname)

    # notice that if we have an issue with finding the file, we first try to update
    # the project path
    if not os.path.exists(full_path):
        project_path = _update_project_path()
        full_path = os.path.join(project_path, config["data"]["path"], fname)

    data = _read_file(full_path, sep)

    return data


def read_event_data(league, notebook=None):
    """
    Reads the event data of the specified league. If *notebook* is set and defined below, only the required subset
    of the event data is returned
    :param league: (str) League for which the event data should be read
    :param notebook: (str, optional) If specified, only the subset of the event data required for the *notebook* is
                      returned
    :return: pd.DataFrame with the event data
    """
    df = read_data("event_data", league)

    # only return the columns needed for the goal kick analysis
    if notebook == "goal_kick_analysis":
        cols = ["id", "matchId", "matchPeriod", "eventSec", "eventName", "subEventName", "teamId", "posBeforeXMeters",
                "posBeforeYMeters", "posAfterXMeters", "posAfterYMeters", "playerId", "playerName", "playerPosition",
                "homeTeamId", "awayTeamId"]

        df = df[cols].copy()

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

    df = read_data("team_data", league)

    return df
