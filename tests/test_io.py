
import helper.io as help_io
import pandas


def test_read_team_data():
    league = "germany"
    df = help_io.read_team_data(league)
    assert type(df) == pandas.core.frame.DataFrame


