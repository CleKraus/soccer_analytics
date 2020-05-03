# -*- coding: utf-8 -*-
import pandas as pd


class Matches:
    def __init__(self, fname):
        self.matches = pd.read_parquet(fname)

    def _filter_matches(self, colnames_filter, filters):
        df = self.matches
        for i, col in enumerate(colnames_filter):
            df = df[df[col] == filters[i]]
        return df

    def matches_gameweek(self, week):
        return self._filter_matches(["gameweek"], [week])

    def matches_team(self, team_id, side="all"):
        if side == "all":
            colnames = ["teamId"]
            filters = [team_id]
        else:
            colnames = ["teamId", "side"]
            filters = [team_id, side]
        return self._filter_matches(colnames, filters)

    def special_match(self, team_id, opp_id, side="home", cols=None):

        df_match = self._filter_matches(
            ["teamId", "oppTeamId", "side"], [team_id, opp_id, side]
        )

        if cols is None:
            return df_match
        else:
            df_match = df_match[cols]
            return list(df_match.iloc[0])


if __name__ == "__main__":

    fname = "C://Clemens//Learning//dash//soccer_dashboard//data//matches.parquet"
    teams = Matches(fname)
