
import helper.io as io_help
import pandas


def test_read_config():

    config = io_help._get_config_file()

    assert type(config) == str


def test_read_team_data():

    league = "germany"
    df = io_help.read_team_data(league)

    assert type(df) == pandas.core.frame.DataFrame


def test_read_player_data():

    df = io_help.read_player_data()

    assert type(df) == pandas.core.frame.DataFrame


def test_read_event_data():

    league = "germany"

    df = io_help.read_event_data(league)

    assert type(df) == pandas.core.frame.DataFrame


def test_read_match_data():

    league = "germany"

    df = io_help.read_match_data(league)

    assert type(df) == pandas.core.frame.DataFrame


def test_read_formation_data():

    league = "germany"

    df = io_help.read_formation_data(league)

    assert type(df) == pandas.core.frame.DataFrame

