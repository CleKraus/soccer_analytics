# -*- coding: utf-8 -*-
import pandas as pd


class Events:
    def __init__(self, fname):
        self.events = pd.read_parquet(fname)

    def get_events(self, matches, event_names, players="all", cols="all"):

        if type(matches) == int:
            matches = [matches]

        if type(event_names) == str and event_names != "all":
            event_names = [event_names]

        if cols == "all":
            cols = self.events.columns

        if players == "all":
            df_events = self.events
        else:
            if type(players) == int:
                players = [players]
            df_events = self.events[self.events["playerId"].isin(players)]

        if matches != "all" and event_names == "all":
            df_events = df_events[df_events["matchId"].isin(matches)]
        elif matches == "all" and event_names != "all":
            df_events = df_events[df_events["eventName"].isin(event_names)]
        elif matches != "all" and event_names != "all":
            df_events = df_events[
                (df_events["matchId"].isin(matches))
                & (df_events["eventName"].isin(event_names))
            ]
        return df_events

    def compute_centroids(
        self, match_id, team_id, relevant_events=["Pass", "Shot", "Duel"]
    ):

        df_events = self.get_events(match_id, relevant_events)
        df_events_team = df_events[df_events["teamId"] == team_id]

        df_centroids = (
            df_events_team.groupby("playerId")
            .agg(centerX=("posBeforeX", "mean"), centerY=("posBeforeY", "mean"))
            .reset_index()
        )
        return df_centroids

    def prepare_heatmap(
        self, player_id, match_id=None, event_names=["Pass", "Shot", "Duel"]
    ):

        if match_id is None:
            match_id = "all"

        df_events = self.get_events(
            matches=match_id, event_names=event_names, players=player_id
        )

        df_events["posBeforeX"] = df_events["posBeforeX"].clip(0, 99.99)
        df_events["posBeforeY"] = df_events["posBeforeY"].clip(0, 99.99)
        df_events["groupX"] = df_events["posBeforeX"].map(lambda x: int(x / 10))
        df_events["groupY"] = df_events["posBeforeY"].map(lambda x: int(x / 10))

        df_pos_player = (
            df_events.groupby(["groupX", "groupY"])
            .agg(nbEvents=("playerId", "count"))
            .reset_index()
        )
        df_all_pos = pd.DataFrame(
            [(x, y) for x in range(10) for y in range(10)], columns=["groupX", "groupY"]
        )
        df_pos_player = pd.merge(df_all_pos, df_pos_player, how="left").fillna(0)
        df_player_piv = df_pos_player.pivot("groupY", "groupX", "nbEvents")

        return df_player_piv


if __name__ == "__main__":

    fname = "C://Clemens//Learning//dash//soccer_dashboard//data//events.parquet"
    teams = Events(fname)
