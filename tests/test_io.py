
import helper.test as test_help
import helper.io as io_help
import ruamel.yaml
import pandas


def test_read_config_old():

    config = test_help._get_config_file()

    assert type(config) == ruamel.yaml.comments.CommentedMap


def test_read_config():

    config = io_help._get_config_file()

    assert type(config) == str


def test_read_team_data():

    league = "germany"
    df = io_help.read_team_data(league)

    assert type(df) == pandas.core.frame.DataFrame