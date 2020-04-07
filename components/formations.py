import pandas as pd


class Formations:

    def __init__(self, fname):
        self.formations = pd.read_parquet(fname)

    def get_formation(self, match_id, team_id, type="all", cols="all"):

        if cols == "all":
            cols = self.formations.columns

        df_form_match = self.formations[
            (self.formations["matchId"] == match_id) & (self.formations["teamId"] == team_id)]

        if type == "all":
            return df_form_match[cols]
        elif type == "lineup":
            return df_form_match[df_form_match["lineup"] == 1][cols]
        elif type == "bench":
            return df_form_match[df_form_match["lineup"] == 0][cols]


if __name__ == "__main__":

    fname = "C://Clemens//Learning//dash//soccer_dashboard//data//formations.parquet"
    teams = Formations(fname)