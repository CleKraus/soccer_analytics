# -*- coding: utf-8 -*-
import pandas as pd

class Players:

    def __init__(self, fname):
        self.players = pd.read_parquet(fname)

    def players_info(self, cols=None):
        if cols is None:
            cols = self.players.columns

        return self.players[cols]

    def player_name(self, player_id):
        return self.players[self.players["playerId"] == player_id].iloc[0]["shortName"]


if __name__ == "__main__":

    fname = "C://Clemens//Learning//dash//soccer_dashboard//data//players.parquet"
    teams = Players(fname)