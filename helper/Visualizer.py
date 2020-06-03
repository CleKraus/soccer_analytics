# import packages

import pandas as pd
import numpy as np
import scipy
import helper.tracking_data as td_help
import helper.plotly as py_help
import copy
import plotly.graph_objects as go


def _build_hover_text(row, dict_info):
    """
    Helper function to build the hover text
    """
    text = ""
    for key in dict_info.keys():
        if "display_type" in dict_info[key]:
            text += "{}: {:^{display_type}}<br />".format(
                key,
                row[dict_info[key]["values"]],
                display_type=dict_info[key]["display_type"],
            )
        else:
            text += "{}: {}<br />".format(key, row[dict_info[key]["values"]])
    return text


def build_convex_hull(df):
    """
    Helper function to build a team convex hull
    """
    positions = df[["xPos", "yPos"]].to_numpy()
    hull = scipy.spatial.ConvexHull(positions)
    return hull


def attach_goals_to_tracking_data(df_track, df_events):
    """
    Helper function to attach goal information to the tracking data. This is used to differentiate goals from another
    *ball not in play* times
    """
    df_events = df_events.copy()
    df_events["nextStartFrame"] = df_events["startFrame"].shift(-1)

    df_goal = df_events[(df_events["goal"] == 1) | (df_events["ownGoal"] == 1)].copy()

    lst_goal_frames = list()
    for i, row in df_goal.iterrows():
        tmp_df = pd.DataFrame(
            {"frame": np.arange(row["startFrame"], row["nextStartFrame"]), "goal": 1}
        )
        lst_goal_frames.append(tmp_df)
    df_goal_frames = pd.concat(lst_goal_frames)

    df_track = pd.merge(df_track, df_goal_frames, how="left")
    df_track["goal"].fillna(0, inplace=True)

    return df_track


class trackingDataVisualizer:
    """
    The *trackingDataVisualizer* is a helper class to visualize the pre-processed Metrica tracking data in form of
    videos or pictures. To get to know a lot of the functionalities I recommend to go through the notebook
    "introduction_to_tracking_data.ipynb". A quick overview over the different functions can also be found here:

    Setting data
    ------------
    set_sequence_by_frames: Set data sequence by indicating start and end frame
    set_sequence_by_time: Set data sequence by indicating start and end time
    set_sequence_by_offset: Set data sequence by indicating one frame and the offset into both directions

    Update general visualization options
    ------------------------------------
    set_highlight_interruptions: Set whether interruptions/goals should be displayed
    set_team_in_front: Set which team should be displayed in front
    update_debug_mode: Set the debug mode
    update_layout: Update the current layout, e.g. add a title

    Add text
    --------
    add_additional_text: Add text to the top right of the graph
    remove_additional_text: Remove the additional text
    add_hovertext: Add hover information for each player
    remove_hovertext: Remove the hovertext

    Visualization tools
    -------------------
    highlight_players: Highlight player(s) with a different colour
    remove_highlight_players: Remove all highlighted players

    add_offensive_line: Add offensve line by considering the most offensive player
    remove_offensive_line: Remove the offensive lines(s)
    add_defensive_line: Add defensive line by considering the most defensive field player
    remove_defensive_line: Remove the defensive line(s)

    add_player_position_lines: Add lines between players based on their id
    add_player_lines_by_position: Add lines between players based on their position
    remove_player_position_lines: Remove all the lines between players

    add_team_convex_hull: Add a convex hull around all the field players
    remove_team_convex_hull: Remove the convex hull

    add_packing_line: Add a packing line to the ball
    remove_packing_line: Remove the packing line

    Retrieve data and figures
    -------------------------
    get_sequence_data: Retrieve the sequence data that is currently set
    get_single_picture: Retrieve a single picture, most likely with arrows indicating direction and speed
    get_figure: Retrieve the figure, most likely an animation that can be watched as a video
    """

    def __init__(
        self,
        df,
        df_events,
        size=1,
        speed=1,
        home_team_colour="red",
        away_team_colour="blue",
        ball_colour="white",
        player_size=12,
        ball_size=9,
        team_front="Away",
        show_player_numbers=False,
        highlight_interruptions=True,
        debug=False,
    ):
        """
        :param df: (pd.DataFrame) Data frame with tracking data of one match
        :param df_events: (pd.DataFrame) Data frame with all events of the match
        :param size: (float) Relative size of the soccer pitch to be displayed
        :param speed: (float) Relative speed of the video
        :param home_team_colour: (str) Colour of the home team
        :param away_team_colour: (str) Colour of the away team
        :param ball_colour: (str) Colour of the ball
        :param player_size: (int) Size of the markers for the players
        :param ball_size: (int) Size of the marker for the ball
        :param team_front: (str) Markers of *team_front* players are in front of the players of the other team
        :param show_player_numbers: (bool) Whether or not to show playerIds in the markers
        :param highlight_interruptions: (bool) Whether or not to highlight interruptions and goals
        :param debug: (bool) Whether or not to run the animation in debug mode and add e.g. the frameId to each frame
        """

        # all tracking data
        self.all_track_data = df.copy()
        # all event data
        self.event_data = df_events.copy()

        # whether or not run in debug mode
        self.debug = debug
        # plot size of the field
        self.plot_size = size
        # relative speed in which the video is shown (1 = normal)
        self.speed = speed
        # colours of the home and away team
        self.home_team_colour = home_team_colour
        self.away_team_colour = away_team_colour
        self.ball_colour = ball_colour

        # size of the markers for the players
        self.player_size = player_size
        self.ball_size = ball_size

        # whether or not interruptions and goals should be highlighted
        self.highlight_interruptions = highlight_interruptions
        # whether or not some additional text should be shown on the top right
        self.additional_text = False

        # all potential data layers and whether they are used or not
        self.data_layers = {
            "heatmap": 0,
            "convex_hull": 0,
            "packing": 0,
            "extreme_lines": 0,
            "player_lines": 0,
            "players": 0,
            "ball": 0,
        }

        # start and end frame of the currently considered sequence data
        self.frame_start = None
        self.frame_end = None
        self.seq_data = None

        # layout of the empty soccer field
        self.layout = self._create_empty_field_layout(size, speed)
        # all frames to be displayer
        self.frames = None

        # label infos
        self.label_info_players = None
        self.label_info_ball = None

        # list of players that are highlighted and in which colour
        self.highlighted_players = None
        self.highlighted_colours = None

        # whether or not the player numbers should be shown
        self.show_player_numbers = show_player_numbers

        # whether or not there has been a lazy change - triggers a recomputation before returning the figure
        self.lazy_change = False

        # team convex hulls that have already been computed
        self.convex_hulls = None

        # position of all the different players
        self.player_positions = (
            df_events.groupby(["playerId", "team"])
            .agg(playerPosition=("playerPosition", "min"))
            .reset_index()
        )

        # players that have a connection line between them
        self.players_with_lines = []

        # team to be displayed in the front
        self.team_front = team_front

        if team_front == "Away":
            self.sort_directions = [True, False, True]
        else:
            self.sort_directions = [True, True, True]

        # attach a column indicating goals to the tracking data
        if highlight_interruptions:
            self.all_track_data = attach_goals_to_tracking_data(
                self.all_track_data, self.event_data
            )

        self.heatmaps = []

    def _get_layer_index(self, layer_name):
        """
        Get the current layer index of *layer_name* for each frame
        """

        index = 0

        all_layers = self.data_layers.keys()
        for layer in all_layers:
            if layer == layer_name:
                return index
            index += self.data_layers[layer]

    @staticmethod
    def _create_empty_field_layout(size, speed, below=True, colour="green"):
        """
        Helper function to the get layout of the empty field by using the *create_empty_field* function from the
        plotly helper
        """

        # get an empty field using the plotly helper
        field = py_help.create_empty_field(below=below, colour=colour, size=size)

        # save the field layout in a dict
        layout = dict(
            # make sure the field is green
            plot_bgcolor=field["layout"]["plot_bgcolor"],
            xaxis=field["layout"]["xaxis"],
            yaxis=field["layout"]["yaxis"],
            shapes=field["layout"]["shapes"],
            width=field["layout"]["width"],
            height=field["layout"]["height"],
            autosize=field["layout"]["autosize"],
            # add play and pause buttons
            updatemenus=[
                dict(
                    type="buttons",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                {
                                    "frame": {"duration": 30 / speed, "redraw": False},
                                    "fromcurrent": True,
                                },
                            ],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                        ),
                    ],
                )
            ],
        )
        return layout

    def _build_marker_dict(
        self, df, marker_size, x_col="xPos", y_col="yPos", color_col="colour"
    ):
        """
        Build the markers for the players and the ball
        """

        # if the player numbers should be shown in the markers
        if self.show_player_numbers:
            dict_out = dict(
                showlegend=False,
                x=np.array(df[x_col]),
                y=np.array(df[y_col]),
                mode="markers+text",
                marker=dict(
                    color=np.array(df[color_col]),
                    line=dict(width=2, color="black"),
                    size=marker_size,
                ),
                text=np.array(df["playerId"]),
                textfont=dict(family="sans serif", size=7, color="white"),
                hoverinfo="none",
            )
        else:
            dict_out = dict(
                showlegend=False,
                x=np.array(df[x_col]),
                y=np.array(df[y_col]),
                mode="markers",
                marker=dict(
                    color=np.array(df[color_col]),
                    line=dict(width=2, color="black"),
                    size=marker_size,
                ),
                hoverinfo="none",
            )
        return dict_out

    @staticmethod
    def _build_line_dict(points, colour):
        """
        Build dictionary to display lines between the *points* in *colour*
        """

        dict_out = dict(
            showlegend=False,
            x=[pos[0] for pos in points],
            y=[pos[1] for pos in points],
            mode="lines",
            line=dict(color=colour),
            hoverinfo="none",
        )

        return dict_out

    @staticmethod
    def _build_rectangle_dict(x_min, x_max, y_min, y_max, colour):
        """
        Build dictionary to display a rectangle in *colour*
        """

        dict_out = dict(
            showlegend=False,
            x=[x_min, x_max, x_max, x_min, x_min],
            y=[y_min, y_min, y_max, y_max, y_min],
            mode="none",
            fill="toself",
            fillcolor=colour,
            hoverinfo="none",
        )

        return dict_out

    @staticmethod
    def _build_text_dict(text, x_pos, y_pos, colour="black"):
        """
        Build dictionary to display a *text* at a certain position
        """
        dict_text = dict(
            showlegend=False,
            x=np.array([x_pos]),
            y=np.array([y_pos]),
            mode="text",
            text=str(np.array(text)),
            textfont=dict(color=colour),
            hoverinfo="none",
        )
        return dict_text

    @staticmethod
    def _build_convex_hull_dict(df, hull=None):
        """
        Given a data frame with the positions, function builds a dictionary to display a convex hull around all
        these positions
        """

        positions = df[["xPos", "yPos"]].to_numpy()

        if hull is None:
            hull = build_convex_hull(df)

        convex_x, convex_y = positions[hull.vertices, 0], positions[hull.vertices, 1]

        convex_hull = dict(
            x=convex_x,
            y=convex_y,
            fill="toself",
            mode="lines",
            showlegend=False,
            fillcolor="black",
            opacity=0.2,
            hoverinfo="none",
            line_color="black",
        )

        return convex_hull

    @staticmethod
    def _build_heatmap_dict(x, y, z):
        return {
            "type": "heatmap",
            "x": x,
            "y": y,
            "z": z,
            "opacity": 0.7,
            "hoverinfo": "none",
            "colorscale": [[0, "red"], [0.5, "white"], [1, "blue"]],
            "showscale": False,
            "zsmooth": "best",
        }

    @staticmethod
    def _build_time(time, millisecond=False):
        """
        Build a string that displays the time in mm:ss (and optional milliseconds) format
        """

        # compute minute and second based on total seconds
        total_second = int(time)
        minute = int(total_second / 60)
        second = total_second - minute * 60

        if not millisecond:
            text_time = f"Time: {minute:02}:{second:02}"
        else:
            millisecond = int(np.round((time - total_second) * 100, 0))
            text_time = f"Time: {minute:02}:{second:02}.{millisecond:02}"

        return text_time

    def _build_interruption_callout(self, tmp_ball):
        """
        Build the rectangle shown on top for interruptions and goals
        """

        if tmp_ball.iloc[0]["ballInPlay"] == 0:
            if tmp_ball.iloc[0]["goal"] == 1:
                # switch between black and white every 5 frames
                if tmp_ball.iloc[0]["frame"] % 10 < 5:
                    rect_colour = "black"
                    text_colour = "white"
                else:
                    rect_colour = "white"
                    text_colour = "black"

                text = "GOAL"

            else:
                rect_colour = "red"
                text_colour = "white"
                text = "Interruption"
        else:
            rect_colour = self.layout["plot_bgcolor"]
            text_colour = "black"
            text = " "
        return text, text_colour, rect_colour

    def _add_extreme_positions(self, df, team, min_max):
        """
        Compute the minimum and maximum position of the field players for each frame
        :param df: (pd.DataFrame) Data frame with position of all players
        :param team: (str) Team we are interested in, i.e. "Home" or "Away"
        :param min_max: (str) "min" if we want to compute the minimal position and "max" if we want to compute
                         the maximal position
        :return: pd.DataFrame with an additional column for the minimal or maximumal position of the field player
        """

        df_out = df.copy()

        # for the away team, the maximal position is the minimal one in sense of xPos
        if team == "Away":
            agg = "min" if min_max == "max" else "max"
        else:
            agg = min_max

        # only include field players and not the goalie
        if min_max == "min":
            goalies = self._return_players_for_positions(["GK"])
            df = df[~df["playerId"].isin(goalies)]

        # compute the extreme position for each frame
        df_extreme_pos = (
            df[(df["team"] == team) & (df["playerId"] != -1)]
            .groupby("frame")
            .agg(extremePosition=("xPos", agg))
            .reset_index()
        )
        df_extreme_pos.columns = ["frame", f"xxx{min_max}Position{team}xxx"]
        df = pd.merge(df_out, df_extreme_pos)
        return df

    @staticmethod
    def _remove_extreme_positions(df, team, min_max):
        """
        Remove the extreme position column(s) from *df*
        """
        if team == "all":
            cols = [
                col for col in df.columns if col.startswith(f"xxx{min_max}Position")
            ]
        else:
            cols = [
                col for col in df.columns if col == f"xxx{min_max}Position{team}xxx"
            ]
        df.drop(cols, axis=1, inplace=True)
        return df, len(cols)

    def _return_players_for_positions(self, positions, team="all"):
        """
        Given a list of positions, return all players that play in the position for the *team*
        """

        # return for both teams
        if team == "all":
            return list(
                self.player_positions[
                    self.player_positions["playerPosition"].isin(positions)
                ]["playerId"]
            )
        # return only for one specific team
        else:
            return list(
                self.player_positions[
                    (self.player_positions["playerPosition"].isin(positions))
                    & (self.player_positions["team"] == team)
                ]["playerId"]
            )

    def add_team_convex_hull(
        self, team, exclude_positions=["GK"], exclude_players=[], lst_hulls=None
    ):
        """
        Compute the convex hull around all players of *team* excluding the ones specified in *exclude_positions* or
        *exclude_players*
        :param team: (str) Team to consider, i.e. "Home" or "Away"
        :param exclude_positions: (list) Positions to be excluded from the convex hull, e.g. goal keeper
        :param exclude_players: (list) List of playerIds that should be excluded from the convex hull
        :param lst_hulls: (list) Optionally, a list of pre-computed convex hulls can already be passed and no
                          recomputation needs to be done
        """

        # check the input
        #################
        if type(exclude_players) != list:
            raise ValueError("exclude_players must be a list of player Ids")

        if lst_hulls is not None and type(lst_hulls) != list:
            raise ValueError("lst_hulls must be a list")

        if lst_hulls is not None and len(lst_hulls) != len(self.frames):
            raise ValueError(
                "List of convex hulls must have the same length as the frames"
            )

        if team not in ["Home", "Away"]:
            raise ValueError("Team must take value *Home* or *Away*")

        # compute all players that play in the *exclude_positions* position
        if len(exclude_players) == 0:
            exclude_players = self._return_players_for_positions(exclude_positions)

        # if there is alredy a convex hull computed, remove it
        if self.data_layers["convex_hull"] == 1:
            self.remove_team_convex_hull()

        # make sure the convex hull is part of the data layers
        self.data_layers["convex_hull"] = 1
        self.convex_hulls = []

        # get the index where to input the convex hull in the frame's data
        index = self._get_layer_index("convex_hull")

        k = 0

        df_players = self.seq_data[self.seq_data["playerId"] != -1].copy()

        for i in sorted(np.unique(self.seq_data["frame"])):

            # get all players for the frame that are not excluded
            tmp_player = df_players[
                (df_players["frame"] == i)
                & (df_players["team"] == team)
                & ~(df_players["playerId"].isin(exclude_players))
            ]

            if lst_hulls is not None:
                convex_hull = self._build_convex_hull_dict(tmp_player, lst_hulls[k])
            else:
                convex_hull = self._build_convex_hull_dict(tmp_player)

            self.convex_hulls.append(convex_hull)

            self.frames[k]["data"].insert(index, convex_hull)
            k += 1

    def remove_team_convex_hull(self):
        """
        Remove the team convex hull from the visualization
        """

        if self.data_layers["convex_hull"] == 0:
            return

        index = self._get_layer_index("convex_hull")

        for i, _ in enumerate(self.frames):
            self.frames[i]["data"].pop(index)

        self.convex_hulls = None
        self.data_layers["convex_hull"] = 0

    def _set_initial_frames(self):
        """
        Main function to set all layers to all frames in scope
        """

        frames = list()

        # split the sequence tracking data into players and ball
        df_players = self.seq_data[self.seq_data["playerId"] != -1].copy()
        df_ball = self.seq_data[self.seq_data["playerId"] == -1].copy()

        # identify whether convex hull, packing, extreme lines or lines between players should be drawn
        draw_heatmap = self.data_layers["heatmap"] > 0
        draw_convex_hull = self.data_layers["convex_hull"] > 0
        draw_packing = self.data_layers["packing"] > 0
        draw_ext_lines = self.data_layers["extreme_lines"] > 0
        draw_player_lines = self.data_layers["player_lines"] > 0

        # get all column names for extreme lines and player lines
        cols_extreme = [
            col
            for col in self.seq_data.columns
            if col.startswith("xxx") and "Position" in col
        ]
        cols_player_lines = [
            col for col in self.seq_data.columns if col.startswith("xxxPositionLine")
        ]

        # loop through all frames
        for k, i in enumerate(sorted(np.unique(self.seq_data["frame"]))):

            # get the player and ball data for this frame
            tmp_player = df_players[df_players["frame"] == i]
            tmp_ball = df_ball[df_ball["frame"] == i]

            # initialize the list to store all layers
            data = []

            if draw_heatmap:
                heatmap = self.heatmaps[k]
                dict_heat = self._build_heatmap_dict(
                    heatmap["x"], heatmap["y"], heatmap["z"]
                )
                data.append(dict_heat)

            # add the convex hull between the players
            if draw_convex_hull:
                convex_hull = self.convex_hulls[k]
                data.append(convex_hull)

            # draw the packing line
            if draw_packing:
                packing_dict = tmp_ball.iloc[0]["xxxPackingDictxxx"]
                data.append(packing_dict)

            # add extreme lines
            if draw_ext_lines:
                for col in cols_extreme:
                    colour = (
                        self.home_team_colour
                        if "Home" in col
                        else self.away_team_colour
                    )
                    pos = tmp_ball.iloc[0][col]
                    dict_extreme = self._build_line_dict([[pos, 0], [pos, 68]], colour)
                    data.append(dict_extreme)

            # add player lines
            if draw_player_lines:
                for col in cols_player_lines:
                    dict_player_line = self._build_line_dict(
                        tmp_ball.iloc[0][col], "black"
                    )
                    data.append(dict_player_line)

            # add all players
            player_data = self._build_marker_dict(tmp_player, self.player_size)
            data.append(player_data)
            self.data_layers["players"] = 1

            # add the ball
            ball_data = self._build_marker_dict(tmp_ball, self.ball_size)
            data.append(ball_data)
            self.data_layers["ball"] = 1

            # add the time on the top left
            text_time = self._build_time(tmp_ball.iloc[0]["time"], self.debug)
            if self.debug:
                time_data = self._build_text_dict(text_time, 8, 70.5)
            else:
                time_data = self._build_text_dict(text_time, 6, 70.5)
            data.append(time_data)

            # add the frame number on the bottom left
            if self.debug:
                frame_data = self._build_text_dict(
                    f"Frame: {tmp_ball.iloc[0]['frame']}", 8, -2.5
                )
                data.append(frame_data)

            # add interruption/goal callouts on top of the field
            if self.highlight_interruptions:
                text, text_colour, rect_colour = self._build_interruption_callout(
                    tmp_ball
                )
                rectangle_data = self._build_rectangle_dict(
                    42.5, 62.5, 68.5, 72.5, rect_colour
                )
                interrupt_data = self._build_text_dict(text, 52.5, 70.5, text_colour)
                data.append(rectangle_data)
                data.append(interrupt_data)

            # add additional text on the top right of the field
            if self.additional_text:
                additional_text = self._build_text_dict(
                    tmp_ball.iloc[0]["xxxAdditionalTextxxx"], 92.5, 70.5
                )
                data.append(additional_text)

            frame = dict(data=data)
            frames.append(frame)

        # set all the frames
        self.frames = frames

    def _set_frames(self):
        """
        Function to set all frames including the hover information
        """

        # as frames are computed, *lazy_change* is reset to False
        self.lazy_change = False

        # compute the layers for all frames
        self.frames = None
        self._set_initial_frames()

        # add label information for the players
        if self.label_info_players is not None:
            cols = [
                self.label_info_players[key]["values"]
                for key in self.label_info_players
            ]
            if all([col in self.seq_data.columns for col in cols]):
                self.add_hovertext(
                    self.label_info_players, on_players=True, on_ball=False
                )

        # add label information for the ball
        if self.label_info_ball is not None:
            cols = [self.label_info_ball[key]["values"] for key in self.label_info_ball]
            if all([col in self.seq_data.columns for col in cols]):
                self.add_hovertext(
                    self.label_info_players, on_players=False, on_ball=True
                )

    def _set_sequence_data(self, lazy=True):
        """
        Helper function to set the sequence data, i.e. all the frames that are currently in scope
        """

        # get the relevant data out of all the tracking data
        df = self.all_track_data[
            (self.all_track_data["frame"] >= self.frame_start)
            & (self.all_track_data["frame"] <= self.frame_end)
        ].copy()

        # sort the values depending on which team is in front
        df.sort_values(
            ["frame", "team", "playerId"], ascending=self.sort_directions, inplace=True
        )

        # set the colour of the teams
        df["colour"] = np.where(
            df["team"] == "Home",
            self.home_team_colour,
            np.where(df["team"] == "Ball", self.ball_colour, self.away_team_colour),
        )

        # temporarily save the old sequence data to find out which columns need to be recomputed
        if self.seq_data is not None:
            seq_data_old = self.seq_data.copy()
        else:
            seq_data_old = None

        # set the new sequence data
        self.seq_data = df

        # we always start without a convex hull
        self.remove_team_convex_hull()

        # we always start without packing line
        self.remove_packing_line()

        # we always start without additional text
        self.additional_text = False

        # if extreme lines were set before, recompute them
        if self.data_layers["extreme_lines"] > 0 and seq_data_old is not None:
            if "xxxminPositionHomexxx" in seq_data_old.columns:
                self.add_defensive_line("Home", lazy=True)
            if "xxxminPositionAwayxxx" in seq_data_old.columns:
                self.add_defensive_line("Away", lazy=True)
            if "xxxMaxPositionHomexxx" in seq_data_old.columns:
                self.add_offensive_line("Home", lazy=True)
            if "xxxMaxPositionAwayxxx" in seq_data_old.columns:
                self.add_offensive_line("Away", lazy=True)

        # set the player lines if they exist
        if self.data_layers["player_lines"] > 0:
            self.add_player_position_lines(self.players_with_lines, lazy=True)

        # highlight the players that were highlighted before
        if self.highlighted_players is not None:
            self.highlight_players(
                self.highlighted_players, self.highlighted_colours, lazy=lazy
            )

        else:
            if not lazy:
                self._set_frames()
            else:
                self.lazy_change = True

    def highlight_players(self, players, colours, lazy=True):
        """
        Highlight individual players with another colour
        :param players: (int or list) PlayerId(s) of player(s) to be highlighted
        :param colours: (int or list) Colour(s) to be used for highlighting
        :param lazy: (bool) If True, computation of frames is only done before returning the animation
        """

        # make sure *players* is passed correctly
        if type(players) != int and type(players) != list:
            raise ValueError("Players must either be an integer or list of integers")

        if type(players) == int:
            players = [players]

        if any([type(x) != int for x in players]):
            raise ValueError("Players must either be an integer or list of integers")

        # make sure *colours* is passed correctly
        if type(colours) != str and type(colours) != list:
            raise ValueError("Colours must either be a string or a list of strings")

        if type(colours) == str:
            colours = [colours] * len(players)

        if any([type(x) != str for x in colours]):
            raise ValueError("Colours must either be a string or a list of strings")

        if len(players) != len(colours):
            raise ValueError(
                "Make sure that *players* and *colours* have the same length"
            )

        self.highlighted_players = players
        self.highlighted_colours = colours

        # update the colours of the players
        for i, player in enumerate(players):
            self.seq_data["colour"] = np.where(
                self.seq_data["playerId"] == player, colours[i], self.seq_data["colour"]
            )

        if not lazy:
            self._set_frames()
        else:
            self.lazy_change = True

    def remove_highlight_players(self):
        """
        Remove all highlighted players
        """

        self.highlighted_players = None
        self.highlighted_colours = None

        self._set_sequence_data()

    def set_sequence_by_frames(self, start_frame, end_frame, lazy=True):
        """
        Set the sequence of frames you are interested in by *start_frame* and *end_frame*
        :param start_frame: (int) Start frame of the sequence
        :param end_frame: (int) End frame of the sequence
        :param lazy: (bool) If True, computation of frames is only done before returning the animation
        """

        if end_frame < start_frame:
            raise ValueError("*end_frame* needs to be at least *start_frame*")
        self.frame_start = start_frame
        self.frame_end = end_frame

        self._set_sequence_data(lazy)

    def set_sequence_by_time(self, start_time, end_time, lazy=True):
        """
        Set the sequence of frames you are interested in by *start_time* and *end_time*
        :param start_time: (float) Second of the start of the sequence
        :param end_time: (float) Second of the end of the sequence
        :param lazy: (bool) If True, computation of frames is only done before returning the animation
        """

        if end_time < start_time:
            raise ValueError("*end_time* needs to be at least *start_time*")

        if start_time < self.all_track_data["time"].min():
            self.frame_start = 0
        else:
            self.frame_start = int(
                self.all_track_data[self.all_track_data["time"] <= start_time][
                    "frame"
                ].max()
            )

        if end_time > self.all_track_data["time"].max():
            self.frame_end = self.all_track_data["frame"].max()
        else:
            self.frame_end = int(
                self.all_track_data[self.all_track_data["time"] >= end_time][
                    "frame"
                ].min()
            )

        self._set_sequence_data(lazy)

    def set_sequence_by_offset(self, frame, offset_before, offset_after, lazy=True):
        """
        Set the sequence you are interested in by defining a frame and the offset before and after this frame. Example:
        If *frame* == 100, *offset_before* == 10 and *offset_after*==20, then all frames between 90 and 120 are included
        :param frame: (int) Main frame around which the offset is defined
        :param offset_before: (int) Offset before *frame*
        :param offset_after: (int) Offset after *frame*
        :param lazy: (bool) If True, computation of frames is only done before returning the animation
        """

        if offset_before < 0 or offset_after < 0:
            raise ValueError("Offset needs to be positive")

        self.frame_start = max(0, frame - offset_before)
        self.frame_end = max(0, frame + offset_after)

        self._set_sequence_data(lazy)

    def set_highlight_interruptions(self, value, lazy=True):
        """
        Define whether or not interruptions/goals should be highlighted by a big rectangle on the top of the field
        :param value: (bool) If True, interruptions are highlighted
        :param lazy: (bool) If True, computation of frames is only done before returning the animation
        """
        if value != self.highlight_interruptions:
            self.highlight_interruptions = value

            if not lazy:
                self._set_frames()
            else:
                self.lazy_change = True

    def add_offensive_line(self, team, lazy=True):
        """
        Add an offensive line, i.e. the position of the most forward player, of *team*
        :param team: (str) "Home" or "Away"
        :param lazy: (bool) If True, computation of frames is only done before returning the animation
        """
        if f"xxxmaxPosition{team}xxx" not in self.seq_data.columns:
            self.seq_data = self._add_extreme_positions(self.seq_data, team, "max")
            self.data_layers["extreme_lines"] += 1

            if not lazy:
                self._set_frames()
            else:
                self.lazy_change = True

    def remove_offensive_line(self, team="all", lazy=True):
        """
        Remove the offensive line for *team*
        :param team: (str) If all, all offensive lines are removed, otherwise "Home" or "Away"
        :param lazy: (bool) If True, computation of frames is only done before returning the animation
        """
        self.seq_data, nb_removed = self._remove_extreme_positions(
            self.seq_data, team, "max"
        )
        self.data_layers["extreme_lines"] -= nb_removed

        if not lazy:
            self._set_frames()
        else:
            self.lazy_change = True

    def add_defensive_line(self, team, lazy=True):
        """
        Add an offensive line, i.e. the position of the most defensive field player, of *team*
        :param team: (str) "Home" or "Away"
        :param lazy: (bool) If True, computation of frames is only done before returning the animation
        """
        if f"xxxminPosition{team}xxx" not in self.seq_data.columns:
            self.seq_data = self._add_extreme_positions(self.seq_data, team, "min")
            self.data_layers["extreme_lines"] += 1

            if not lazy:
                self._set_frames()
            else:
                self.lazy_change = True

    def remove_defensive_line(self, team="all", lazy=True):
        """
        Remove the defensive line for *team*
        :param team: (str) If all, all offensive lines are removed, otherwise "Home" or "Away"
        :param lazy: (bool) If True, computation of frames is only done before returning the animation
        """
        self.seq_data, nb_removed = self._remove_extreme_positions(
            self.seq_data, team, "min"
        )
        self.data_layers["extreme_lines"] -= nb_removed

        if not lazy:
            self._set_frames()
        else:
            self.lazy_change = True

    def add_player_position_lines(self, players, lazy=True):
        """
        Add black lines between the *players*
        :param players: (list) List of list between the players to be connected, e.g. [[1,2]] if only one line between
                         player 1 and 2 should be displayed and [[1,2],[3,4]] of two lines should be displayed
        :param lazy: (bool) If True, computation of frames is only done before returning the animation
        """

        df = self.seq_data.copy()
        players = copy.deepcopy(players)

        # loop through the players that should be connected
        for tmp_players in players:

            # sort the players by the y-position
            df_players = df[df["playerId"].isin(tmp_players)].copy()
            df_players.sort_values(["frame", "yPos"], inplace=True)
            df_players["position"] = df_players.apply(
                lambda row: [row["xPos"], row["yPos"]], axis=1
            )

            # build a list with the player positions for each frame
            df_agg = (
                df_players.groupby("frame")["position"]
                .agg(lambda x: list(x))
                .reset_index()
            )

            # add the list of positions to the sequence data
            cols = [col for col in df.columns if col.startswith("xxxPositionLine")]
            colname = f"xxxPositionLine{len(cols) + 1}"
            df_agg.columns = ["frame", colname]
            df = pd.merge(df, df_agg, how="left")

            if len(df[df[colname].isnull()]) > 0:
                raise ValueError("Data for position line not available for all frames")

            self.data_layers["player_lines"] += 1

            if tmp_players not in self.players_with_lines:
                self.players_with_lines.append(tmp_players)

        self.seq_data = df.copy()

        if not lazy:
            self._set_frames()
        else:
            self.lazy_change = True

    def add_player_lines_by_position(self, positions, team, lazy=True):
        """
        Add black lines between the players of the same position
        :param positions: (list) List of positions to be connected
        :param team: (str) Team for which the lines should be added ("Home" or "Away")
        :param lazy: (bool) If True, computation of frames is only done before returning the animation
        """

        for position in positions:
            players = self._return_players_for_positions([position], team=team)
            self.add_player_position_lines([players], lazy=lazy)

    def remove_player_position_lines(self, lazy=True):
        """
        Remove all the player position lines from the animation
        :param lazy: (bool) If True, computation of frames is only done before returning the animation
        """

        self.data_layers["player_lines"] = 0
        self.players_with_lines = []

        cols = [
            col for col in self.seq_data.columns if col.startswith("xxxPositionLine")
        ]
        self.seq_data.drop(cols, axis=1, inplace=True)

        if not lazy:
            self._set_frames()
        else:
            self.lazy_change = True

    def add_player_number(self, lazy=True):
        """
        Add the player number to the marker
        :param lazy: (bool) If True, computation of frames is only done before returning the animation
        """

        if not self.show_player_numbers:
            self.show_player_numbers = True

            if not lazy:
                self._set_frames()
            else:
                self.lazy_change = True

    def remove_player_number(self, lazy=True):
        """
        Remove the player number from the marker
        :param lazy: (bool) If True, computation of frames is only done before returning the animation
        """

        if self.show_player_numbers:
            self.show_player_numbers = False

            if not lazy:
                self._set_frames()
            else:
                self.lazy_change = True

    def _update_hovertext(self, df, index):
        """
        Helper function to set the hovertext
        """

        for i, frame in enumerate(sorted(df["frame"].unique())):
            self.frames[i]["data"][index]["hoverinfo"] = "text"
            self.frames[i]["data"][index]["hovertext"] = np.array(
                df[df["frame"] == frame]["hoverText"]
            )

    def add_hovertext(
        self, label_info, df=None, on_players=True, on_ball=False, lazy=True
    ):
        """
        Add hovertext to the frames
        :param label_info: (dict) Used to understand which info should be displayed how
        :param df: (pd.DataFrame) If data to be displayed is not already part of the sequence data, a data frame with
                   additional columns can be passed. Needs to have a "frame" column!
        :param on_players: (bool) Whether the information should be shown on players
        :param on_ball: (bool) Whether the information should be shown on the ball
        :param lazy: (bool) If True, computation of frames is only done before returning the animation
        """

        if self.lazy_change:
            self._set_frames()

        if df is None:
            df = self.seq_data.copy()
        else:
            if list(df["frame"].unique()) != list(self.seq_data["frame"].unique()):
                raise ValueError(
                    "Data frame with hovertext needs to have the same frames as the current data sequence"
                )

        df.sort_values(
            ["frame", "team", "playerId"], ascending=self.sort_directions, inplace=True
        )

        if not on_players and not on_ball:
            raise ValueError(
                "Hover information must be shown on either the players or the ball"
            )

        df["hoverText"] = df.apply(
            lambda row: _build_hover_text(row, label_info), axis=1
        )

        if on_players:
            index_player = self._get_layer_index("players")
            df_player = df[df["playerId"] != -1].copy()
            self._update_hovertext(df_player, index_player)
            self.label_info_players = label_info

        if on_ball:
            index_ball = self._get_layer_index("ball")
            df_ball = df[df["playerId"] == -1].copy()
            self._update_hovertext(df_ball, index_ball)
            self.label_info_ball = label_info

    def remove_hovertext(self, lazy=True):
        """
        Remove all the hovertext
        :param lazy: (bool) If True, computation of frames is only done before returning the animation
        """
        self.label_info_players = None
        self.label_info_ball = None

        if not lazy:
            self._set_frames()
        else:
            self.lazy_change = True

    def set_team_in_front(self, team, lazy=True):
        """
        Set the player markers of *team* to the front to make them more visible
        :param team: (str) Team to be in the front, "Home" or "Away"
        :param lazy: (bool) If True, computation of frames is only done before returning the animation
        """

        if team != self.team_front:
            if team == "Away":
                self.sort_directions = [True, False, True]
            else:
                self.sort_directions = [True, True, True]

            self._set_sequence_data(lazy)

    def update_debug_mode(self, debug_mode, lazy=True):
        """
        Set the debug mode to e.g. have the frameId displayed for each frame in the bottom left
        :param debug_mode: (bool) Debug mode is on if True
        :param lazy: (bool) If True, computation of frames is only done before returning the animation
        """

        if debug_mode != self.debug:
            self.debug = debug_mode
            if not lazy:
                self._set_frames()
            else:
                self.lazy_change = True

    def update_layout(self, title=None):
        """
        Update the layout of the animation. Currently only the setting of a title is supported
        :param title: (str) Title of the animation
        """

        if title is not None:
            self.layout["title"] = title

    @staticmethod
    def _compute_packing_line(x_ball, y_ball, team):
        """
        Helper function to compute the packing line
        """

        # the own goal for the home team is at 0 and for the away team at 105
        x_goal = 0 if team == "Home" else 105
        y_goal = 34

        # compute the distance of the ball to the goal
        dist = np.sqrt(
            (x_ball - x_goal) * (x_ball - x_goal)
            + (y_ball - y_goal) * (y_ball - y_goal)
        )

        y_min = np.max([0, y_goal - dist])
        y_max = np.min([68, y_goal + dist])

        # compute the circle for 15 equally distributed points
        numbers = 15
        steps = (y_max - y_min) / numbers
        y_lst = [y_min + i * steps for i in np.arange(numbers + 1)]

        if team == "Home":
            x_lst = np.array(
                [
                    x_goal + np.sqrt(np.abs(dist * dist - (y - y_goal) * (y - y_goal)))
                    for y in y_lst
                ]
            )
        else:
            x_lst = np.array(
                [
                    x_goal - np.sqrt(np.abs(dist * dist - (y - y_goal) * (y - y_goal)))
                    for y in y_lst
                ]
            )

        y_lst = list(y_lst)
        x_lst = list(x_lst)

        # if the distance is > 34, we know that the circle crosses the out-line
        if dist > 34:
            x_lst = [x_goal] + x_lst + [x_goal]
            y_lst = [0] + y_lst + [68]

        x_lst.append(x_lst[0])
        y_lst.append(y_lst[0])

        return x_lst, y_lst

    def _build_dict_packing(self, x_ball, y_ball, team):
        """
        Build a dictionary for the packing line
        """

        # compute the packing line
        x_vals, y_vals = self._compute_packing_line(x_ball, y_ball, team)

        dict_out = dict(
            x=x_vals,
            y=y_vals,
            mode="lines",
            opacity=0.5,
            line=dict(color="black"),
            showlegend=False,
            fill="toself",
            hoverinfo="none",
        )
        return dict_out

    def add_packing_line(self, defensive_team, lazy=True):
        """
        Add a packing line to the animation
        :param defensive_team: (str) Team that is currently defending
        :param lazy: (bool) If True, computation of frames is only done before returning the animation
        """

        self.data_layers["packing"] = 1

        df_ball = self.seq_data[self.seq_data["playerId"] == -1].copy()

        df_ball["xxxPackingDictxxx"] = df_ball.apply(
            lambda row: self._build_dict_packing(
                row["xPos"], row["yPos"], defensive_team
            ),
            axis=1,
        )

        self.seq_data = pd.merge(
            self.seq_data,
            df_ball[["frame", "xxxPackingDictxxx"]],
            how="left",
            on="frame",
        )

        if not lazy:
            self._set_frames()
        else:
            self.lazy_change = True

    def remove_packing_line(self, lazy=True):
        """
        Remove the packing line from the animation
        :param lazy: (bool) If True, computation of frames is only done before returning the animation
        """

        self.data_layers["packing"] = 0

        col = "xxxPackingDictxxx"
        if col in self.seq_data.columns:
            self.seq_data.drop(col, axis=1, inplace=True)

        if not lazy:
            self._set_frames()
        else:
            self.lazy_change = True

    def add_additional_text(self, df, lazy=True):
        """
        Add additional, customizable text to the top right of the field
        :param df: (pd.DataFrame) Data frame containing a column for the frameId and another column containing the
                    text that should be displayed for this text
        :param lazy: (bool) If True, computation of frames is only done before returning the animation
        """

        # check that *df* is passed correctly
        frames_df = np.unique(df["frame"])
        if not all([frame in frames_df for frame in self.seq_data["frame"].unique()]):
            raise ValueError("Each frame in the sequence data needs to have a text")

        if df.shape[1] != 2:
            raise ValueError("*df* must have two columns")

        if df.columns[0] != "frame":
            raise ValueError("First column must be *frame*")

        # remove text that might have been there
        if self.additional_text:
            self.remove_additional_text()

        self.additional_text = True

        df = df.copy()
        df.columns = ["frame", "xxxAdditionalTextxxx"]

        self.seq_data = pd.merge(self.seq_data, df, on="frame")

        if not lazy:
            self._set_frames()
        else:
            self.lazy_change = True

    def remove_additional_text(self, lazy=True):
        """
        Remove additional text that has been set before
        :param lazy: (bool) If True, computation of frames is only done before returning the animation
        """

        self.additional_text = False

        col = "xxxAdditionalTextxxx"
        if col in self.seq_data.columns:
            self.seq_data.drop(col, axis=1, inplace=True)

        if not lazy:
            self._set_frames()
        else:
            self.lazy_change = True

    def get_sequence_data(self):
        """
        Get the currently set sequence data
        :return: pd.DataFrame containing the sequence data
        """
        return self.seq_data.copy()

    def get_single_picture(self, frame, add_arrows=True, arrow_length=1):
        """
        Get a single picture of *frame*, most often with arrows of direction and speed the players are moving to
        :param frame: (int) Frame to be returned, needs to be part of the current sequence
        :param add_arrows: (bool) If True, arrows indicating direction and speed are added to the players
        :param arrow_length: (float) Relative length of the array
        :return: go.Figure containing the situation at *frame*
        """

        if self.lazy_change:
            self._set_frames()

        frames = list(self.seq_data["frame"].unique())

        if frame not in frames:
            raise ValueError("frame must be part of your sequence")

        if frame == self.seq_data["frame"].min() and add_arrows:
            raise ValueError("Cannot add arrows for first picture")

        index = frames.index(frame)

        layout = copy.deepcopy(self.layout)
        layout["updatemenus"] = None

        fig = go.Figure(dict(data=self.frames[index]["data"], layout=layout))

        if not add_arrows:
            return fig

        df = self.get_sequence_data()
        df = td_help.add_position_delta(df)
        df = df[df["frame"] == frame].copy()

        for i, row in df.iterrows():

            if (
                self.highlighted_players is not None
                and row["playerId"] in self.highlighted_players
            ):
                ix = self.highlighted_players.index(row["playerId"])
                colour = self.highlighted_colours[ix]
            else:
                colour = row["colour"]

            if row["dx"] < 0.05 and row["dy"] < 0.05:
                continue

            fig.add_annotation(
                ax=row["xPos"],
                ay=row["yPos"],
                axref="x",
                ayref="y",
                x=row["xPos"] + 40 * min(0.4, row["dx"]) * arrow_length,
                y=row["yPos"] + 40 * min(0.4, row["dy"]) * arrow_length,
                xref="x",
                yref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=colour,
            )

        return fig

    def get_figure(self):
        """
        Depending on whether the sequence consists of one or more frames, an animation or go.Figure with all the
        setting as defined before is returned
        :return: dict or go.Figure containing the animation or picture of all frames in the sequence
        """

        if self.lazy_change:
            self._set_frames()

        # return a picture if there is only one frame
        if len(self.frames) == 1:
            layout = copy.deepcopy(self.layout)
            layout["updatemenus"] = None
            fig = go.Figure(dict(data=self.frames[0]["data"], layout=layout))

        # otherwise return a dictionary with the animation
        else:
            fig = dict(
                data=self.frames[0]["data"], layout=self.layout, frames=self.frames
            )
        return fig

    def add_heatmap(self, x, y, z, lazy=True):

        if len(x) != len(y) or len(y) != len(z):
            raise ValueError("Length of x, y and z need to be identical")

        if self.lazy_change:
            self._set_frames()
            self.lazy_change = False

        if len(x) != len(self.frames):
            raise ValueError("Number of heatmaps and frames needs to be identical")

        if self.data_layers["heatmap"] == 1:
            self.remove_heatmap(lazy=True)

        self.layout = self._create_empty_field_layout(
            self.plot_size, self.speed, colour="white", below=True
        )

        for i in np.arange(len(x)):
            dict_heatmap = {"x": x[i], "y": y[i], "z": z[i]}
            self.heatmaps.append(dict_heatmap)

        self.data_layers["heatmap"] = 1

        if not lazy:
            self._set_frames()
        else:
            self.lazy_change = True

    def remove_heatmap(self, lazy=True):

        self.data_layers["heatmap"] = 0
        self.heatmaps = []

        self.layout = self._create_empty_field_layout(
            self.plot_size, self.speed, colour="green"
        )

        if not lazy:
            self._set_frames()
        else:
            self.lazy_change = True
