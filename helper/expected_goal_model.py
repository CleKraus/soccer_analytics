
import math
import pandas as pd
import numpy as np

import helper.io as io


class ExpectedGoalModel:

    def __init__(self, debug_mode=True):

        if debug_mode:
            df_events = io.read_event_data("all", notebook="expected_goal_model")
            self.df_events = self._compute_event_before(df_events)
        else:
            self.df_events = None

        self.debug_mode = debug_mode

        self.features = ["distToGoalLine", "distToCenter", "weakFoot", "counterAttack", "corner", "smartPass",
                         "duel", "shotBefore", "angleClip", "frontOfGoal", "headDistToGoalLine"]

        self.feature_measures = {'distToGoalLine': {'mean': 15.930207612456748, 'std': 8.520058996794551},
                                 'distToCenter': {'mean': 7.786845938375349, 'std': 5.2201237665066},
                                 'weakFoot': {'mean': 0.18866370077442743, 'std': 0.3912476878714778},
                                 'counterAttack': {'mean': 0.055692865381446695, 'std': 0.22933142674397106},
                                 'corner': {'mean': 0.023924864063272367, 'std': 0.15281765125037938},
                                 'smartPass': {'mean': 0.04264293952875268, 'std': 0.20205411311643012},
                                 'duel': {'mean': 0.36875926841324763, 'std': 0.48247646741037054},
                                 'shotBefore': {'mean': 0.021057834898665348, 'std': 0.14357953142714186},
                                 'angleClip': {'mean': 36.94625627632563, 'std': 5.8718678088086165},
                                 'frontOfGoal': {'mean': 0.01888284725654968, 'std': 0.13611354039183185},
                                 'headDistToGoalLine': {'mean': 1.440968858131488, 'std': 3.7213713754703233}}

    @staticmethod
    def _compute_event_before(df_events):
        # for each event, compute the event, subevent and performing team before this event
        df = df_events.copy()

        df["eventBefore"] = df.groupby(["matchId", "matchPeriod"])["eventName"].shift(1)
        df["subEventBefore"] = df.groupby(["matchId", "matchPeriod"])["subEventName"].shift(1)
        df["teamBefore"] = df.groupby(["matchId", "matchPeriod"])["teamId"].shift(1)

        return df

    @staticmethod
    def _transform_body_part_shot(row):
        """
        Helper function to identify whether a shot was taken with the strong or the weak foot.
        """
        if row["bodyPartShot"] == "head/body":
            return "head/body"
        elif row["bodyPartShot"] == "rightFoot":
            if row["playerStrongFoot"] == "left":
                return "weakFoot"
            else:
                return "strongFoot"
        elif row["bodyPartShot"] == "leftFoot":
            if row["playerStrongFoot"] == "right":
                return "weakFoot"
            else:
                return "strongFoot"
        else:
            raise ValueError(f"Body part *{row['bodyPartShot']}* unknown.")

    @staticmethod
    def _add_feature_distance_goal_line(df):
        # compute the distance to the goal line
        df["distToGoalLine"] = 105 - df["posBeforeXMeters"]
        return df

    @staticmethod
    def _add_feature_distance_center(df):
        # compute the distance to the center
        df["distToCenter"] = np.abs(34 - df["posBeforeYMeters"])
        return df

    def _add_features_body_part(self, df):
        # convert body part to strong foot / weak foot
        df["bodyPartShot"] = df.apply(self._transform_body_part_shot, axis=1)

        # dummy encode the body part
        for col in ["head/body", "weakFoot", "strongFoot"]:
            df[col] = 1 * (df["bodyPartShot"] == col)

        return df

    @staticmethod
    def _add_feature_corner(df, df_events):
        # add a feature if the subEvent before was a corner
        if "corner" in df.columns:
            df.drop("corner", axis=1, inplace=True)

        df_corner = df_events[df_events["subEventBefore"] == "Corner"].copy()
        df_corner["corner"] = 1
        df = pd.merge(df, df_corner[["id", "corner"]], on="id", how="left")
        df["corner"].fillna(0, inplace=True)
        return df

    @staticmethod
    def _add_feature_smart_pass(df, df_events):
        # add feature if event before was a smart pass
        if "smartPass" in df.columns:
            df.drop("smartPass", axis=1, inplace=True)

        df_smart_pass = df_events[df_events["subEventBefore"] == "Smart pass"].copy()
        df_smart_pass["smartPass"] = 1
        df = pd.merge(df, df_smart_pass[["id", "smartPass"]], on="id", how="left")
        df["smartPass"].fillna(0, inplace=True)
        return df

    @staticmethod
    def _add_feature_duel(df, df_events):
        # add a feature if event before was a duel
        if "duel" in df.columns:
            df.drop("duel", axis=1, inplace=True)

        df_duel = df_events[df_events["eventBefore"] == "Duel"].copy()
        df_duel["duel"] = 1
        df = pd.merge(df, df_duel[["id", "duel"]], on="id", how="left")
        df["duel"].fillna(0, inplace=True)
        return df

    @staticmethod
    def _add_feature_shot_before(df, df_events):
        # add a feature if event before was a shot or goalie reflex
        if "shotBefore" in df.columns:
            df.drop("shotBefore", axis=1, inplace=True)

        df_shot_before = df_events[df_events["subEventBefore"].isin(["Shot", "Reflexes", "Save attempt"])].copy()
        df_shot_before["shotBefore"] = 1
        df = pd.merge(df, df_shot_before[["id", "shotBefore"]], on="id", how="left")
        df["shotBefore"].fillna(0, inplace=True)
        return df

    @staticmethod
    def _add_feature_angle_clip(df):
        # add the angle of the shot (0=in front of goal, 90=on goal line but not between posts)

        # distance to the post in y-direction (notice that the width of the goal is 7.32)
        df["dy"] = (df["distToCenter"] - 7.32 / 2)

        # if we have a negative angle (i.e. we are between the two posts right in front of the goal), we want to set the
        # distance to 0
        df["dy"].clip(lower=0, inplace=True)

        # compute the angle to the closest post
        df["angle"] = df.apply(lambda row: math.degrees(math.atan2(row["dy"], row["distToGoalLine"])), axis=1)

        # clip the angle at 35 degrees
        df["angleClip"] = df["angle"].clip(lower=35)

        df.drop("dy", axis=1, inplace=True)
        return df

    @staticmethod
    def _add_feature_front_of_goal(df):
        # binary feature if between the two posts and max 5 meters from the goal
        df["frontOfGoal"] = 1 * ((df["distToGoalLine"] <= 5) & (df["angle"] == 0))
        return df

    @staticmethod
    def _add_feature_header_distance(df):
        # interaction of header and distance to the goal line
        df["headDistToGoalLine"] = df["head/body"] * df["distToGoalLine"]
        return df

    def create_features(self, df, overwrite=True):

        df = df.copy()

        # add the distance to the goal line
        if "distToGoalLine" not in df.columns or overwrite:
            df = self._add_feature_distance_goal_line(df)

        # add distance to the center of the field
        if "distToCenter" not in df.columns or overwrite:
            df = self._add_feature_distance_center(df)

        # dummy encode the body parts
        if any(col not in df.columns for col in ["head/body", "weakFoot", "strongFoot"]) or overwrite:
            df = self._add_features_body_part(df)

        # add feature with the angle of the shot
        if any(col not in df.columns for col in ["angle", "angleClip"]) or overwrite:
            df = self._add_feature_angle_clip(df)

        # add feature whether player is right in front of the goal
        if "frontOfGoal" not in df.columns or overwrite:
            df = self._add_feature_front_of_goal(df)

        # add a feature with the distance to goal line for headers
        if "headerDistFromGoal" not in df.columns or overwrite:
            df = self._add_feature_header_distance(df)

        if self.debug_mode:
            # add binary feature whether there was a corner beforehand
            if "corner" not in df.columns or overwrite:
                df = self._add_feature_corner(df, self.df_events)

            # add feature whether there was a smart pass before the shot
            if "smartPass" not in df.columns or overwrite:
                df = self._add_feature_smart_pass(df, self.df_events)

            # add feature whether there was a duel before the shot
            if "duel" not in df.columns or overwrite:
                df = self._add_feature_duel(df, self.df_events)

            # add feature whether there was a shot or goalie reflex beforehand
            if "shotBefore" not in df.columns or overwrite:
                df = self._add_feature_shot_before(df, self.df_events)

        return df

    def normalize_features(self, df, features=None, feature_measures=None):

        if features is not None:
            self.features = features

        if feature_measures is not None:
            self.feature_measures = feature_measures

        df_normal = df[self.features].copy()

        for feat in self.features:
            df_normal[feat] = (df_normal[feat] - self.feature_measures[feat]["mean"]) / self.feature_measures[feat][
                "std"]

        return df_normal
