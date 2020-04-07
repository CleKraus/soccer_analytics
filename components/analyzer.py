import os
import pandas as pd
import numpy as np

from components.teams import Teams
from components.matches import Matches
from components.players import Players
from components.events import Events
from components.formations import Formations


class Analyzer:

    def __init__(self):

        PATH = "C://Clemens//Personal//Soccer//Event_Project//data//clean_data"
        fname_teams = "teams.parquet"
        fname_players = "players.parquet"
        fname_matches = "matches_germany.parquet"
        fname_events = "events_germany.parquet"
        fname_formations = "formations_germany.parquet"

        self.teams = Teams(os.path.join(PATH, fname_teams))
        self.matches = Matches(os.path.join(PATH, fname_matches))
        self.players = Players(os.path.join(PATH, fname_players))
        self.events = Events(os.path.join(PATH, fname_events))
        self.formations = Formations(os.path.join(PATH, fname_formations))

        self.positions = None

    def get_match_overview(self, home_team, away_team):

        home_id = self.teams.team_id(home_team)
        away_id = self.teams.team_id(away_team)
        match_id, home_score, away_score = self.matches.special_match(home_id, away_id,
                                                                      cols=["matchId", "score", "oppScore"])
        return home_id, away_id, match_id, home_score, away_score

    def get_team_field_positions(self, match_id, team_id, side):

        df_position = self.events.compute_centroids(match_id, team_id)
        df_lineup = self.formations.get_formation(match_id, team_id, type="lineup")
        df_position = pd.merge(df_position, df_lineup, on=["playerId"])
        df_position = pd.merge(df_position, self.players.players_info(cols=["playerId", "shortName"]))

        # TODO: Put into drawing function, not in here
        if side == "away":
            df_position["centerX"] = 100 - df_position["centerX"]
        else:
            df_position["centerY"] = 100 - df_position["centerY"]

        return df_position

    def get_pass_count(self, match_id, team_id, show_top_k=1):

        # TODO: Get events through database
        df_events = self.events.get_events(matches=match_id, event_names="all")
        df_events = df_events[df_events["teamId"] == team_id].copy()
        df_events.sort_values("id", inplace=True)

        df_events["nextPlayerId"] = df_events["playerId"].shift(-1)
        df_passes = df_events[(df_events["eventName"] == "Pass") & (df_events["Accurate"] == 1)].copy()
        df_passes["player1"] = np.where(df_passes["playerId"] < df_passes["nextPlayerId"], df_passes["playerId"], df_passes["nextPlayerId"])
        df_passes["player2"] = np.where(df_passes["playerId"] < df_passes["nextPlayerId"], df_passes["nextPlayerId"], df_passes["playerId"])
        pass_overview = df_passes.groupby(["player1", "player2"]).agg(totalPasses=("player1", "count")).reset_index()
        pass_overview.sort_values("totalPasses", ascending=False, inplace=True)
        pass_overview["cumSharePasses"] = pass_overview["totalPasses"].cumsum() / sum(pass_overview["totalPasses"])
        pass_overview = pass_overview[pass_overview["cumSharePasses"] <= show_top_k].copy()

        return pass_overview

    def get_pass_share(self, match_id, team_id, df_position, show_top_k=1):

        df_pass = self.get_pass_count(match_id, team_id, show_top_k)

        hallo = pd.merge(df_pass,
                         df_position.rename(columns={"playerId": "player1", "centerX": "centerX1", "centerY": "centerY1"}),
                         on="player1")
        hallo = pd.merge(hallo,
                         df_position.rename(columns={"playerId": "player2", "centerX": "centerX2", "centerY": "centerY2"}),
                         on="player2")
        hallo["sharePasses"] = hallo["totalPasses"] / sum(hallo["totalPasses"])

        return hallo