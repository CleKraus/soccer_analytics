# -*- coding: utf-8 -*-
import pandas as pd


class Teams:
    def __init__(self, fname):
        self.teams = pd.read_parquet(fname)

    def teams_in_area(self, area_name, type="club", cols=None):

        if cols is None:
            cols = self.teams.columns

        if area_name == "all" and type == "all":
            return self.teams[cols]
        elif area_name == "all" and type != "all":
            return self.teams[(self.teams["type"] == type)][cols]
        elif area_name != "all" and type == "all":
            return self.teams[(self.teams["area.name"] == area_name)][cols]
        else:
            return self.teams[
                (self.teams["area.name"] == area_name) & (self.teams["type"] == type)
            ][cols]

    def team_id(self, team_name):
        return self.teams[self.teams["teamName"] == team_name].iloc[0]["teamId"]

    def team_name(self, team_id):
        return self.teams[self.teams["teamId"] == team_id].iloc[0]["teamName"]


if __name__ == "__main__":

    fname = "C://Clemens//Learning//dash//soccer_dashboard//data//teams.parquet"
    teams = Teams(fname)
