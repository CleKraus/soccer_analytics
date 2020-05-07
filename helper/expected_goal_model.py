# -*- coding: utf-8 -*-

# import packages
import math
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

import helper.io as io


class ExpectedGoalModel(ABC):
    """
    Base class for the expected goal model. Purpose of the ExpectedGoalModels is to combine the feature creation,
    transformation and the prediction using pre-trained and finalized models in one class. Notice that, on purpose,
    there is no fitting function as this is thought to be only for predictions using pre-trained models.
    """

    def __init__(self, debug_mode=True):

        self.debug_mode = debug_mode
        self.model = None
        self.model_name = None
        self.features = None

    @abstractmethod
    def predict(self, X):
        """
        Predict target for features X. If no model was fitted in this instance, a default version should be read.
        """
        pass

    @abstractmethod
    def predict_proba(self, X):
        """
        Predict target probabilities for features X. If no model was fitted in this instance, a default version
        should be read
        """
        pass

    @abstractmethod
    def create_features(self, df, overwrite=True):
        pass

    # General functions to compute features
    #######################################

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
        df["bodyPartShotCode"] = -1
        for i, col in enumerate(["head/body", "weakFoot", "strongFoot"]):
            df[col] = 1 * (df["bodyPartShot"] == col)
            df["bodyPartShotCode"] = np.where(
                df["bodyPartShot"] == col, i, df["bodyPartShotCode"]
            )

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

        df_shot_before = df_events[
            df_events["subEventBefore"].isin(["Shot", "Reflexes", "Save attempt"])
        ].copy()
        df_shot_before["shotBefore"] = 1
        df = pd.merge(df, df_shot_before[["id", "shotBefore"]], on="id", how="left")
        df["shotBefore"].fillna(0, inplace=True)
        return df

    @staticmethod
    def _add_feature_angle_clip(df):
        # add the angle of the shot (0=in front of goal, 90=on goal line but not between posts)

        # distance to the post in y-direction (notice that the width of the goal is 7.32)
        df["dy"] = df["distToCenter"] - 7.32 / 2

        # if we have a negative angle (i.e. we are between the two posts right in front of the goal), we want to set the
        # distance to 0
        df["dy"].clip(lower=0, inplace=True)

        # compute the angle to the closest post
        df["angle"] = df.apply(
            lambda row: math.degrees(math.atan2(row["dy"], row["distToGoalLine"])),
            axis=1,
        )

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

    @staticmethod
    def _compute_event_before(df_events):
        # for each event, compute the event, subevent and performing team before this event
        df = df_events.copy()

        df["eventBefore"] = df.groupby(["matchId", "matchPeriod"])["eventName"].shift(1)
        df["subEventBefore"] = df.groupby(["matchId", "matchPeriod"])[
            "subEventName"
        ].shift(1)
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


class ExpectedGoalModelLogistic(ExpectedGoalModel):
    """
    Expected goal model using logistic regression.
    """

    def __init__(self, debug_mode=True):

        super().__init__(debug_mode)

        # if debug mode, read all the event data and extract the events before
        if self.debug_mode:
            df_events = io.read_event_data("all", notebook="expected_goal_model")
            self.df_events = self._compute_event_before(df_events)
        else:
            self.df_events = None

        # trained model will be saved here
        self.model = None

        # model name for reading
        self.model_name = "expected_goals_logreg"

        # features used in the model
        df_features = io.read_data(
            "features_expected_goals_logreg", data_folder="model", sep=";"
        )
        self.features = list(df_features[df_features["used"] == 1]["feature"])

        # mean values and standard deviations for this model
        self.feature_measures = {}
        for i, row in df_features.iterrows():
            self.feature_measures[row["feature"]] = {}
            self.feature_measures[row["feature"]]["mean"] = row["mean"]
            self.feature_measures[row["feature"]]["std"] = row["std"]

    def create_features(self, df, overwrite=True):
        """
        Pipeline to create and transform features from raw data. Notice that normalization is not part of this pipeline,
        but can be performed through a separate *normalize_features* function.
        :param df: (pd.DataFrame) Data frame with raw data
        :param overwrite: (bool) If False, only features that do not exist yet will be computed. If True, all features
                           are created again.
        :return: pd.DataFrame containing all features needed for the machine learning model
        """

        df = df.copy()

        # add the distance to the goal line
        if "distToGoalLine" not in df.columns or overwrite:
            df = self._add_feature_distance_goal_line(df)

        # add distance to the center of the field
        if "distToCenter" not in df.columns or overwrite:
            df = self._add_feature_distance_center(df)

        # dummy encode the body parts
        if (
            any(
                col not in df.columns for col in ["head/body", "weakFoot", "strongFoot"]
            )
            or overwrite
        ):
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
        """
        Perform standard normalization, i.e. (x-mu)/sigma for all features. If *features* and *feature_measures* are
        None, the pre-defined default features will be used.
        :param df: (pd.DataFrame) Data frame containing non-normalized features
        :param features: (list) Features to be normalized. If None, pre-defined default features will be used
        :param feature_measures: (dict) Mean and standard deviation of the features. If None, pre-defined default
                                  feature measure will be used
        :return: pd.DataFrame with standard normalized features
        """

        if features is not None:
            self.features = features

        if feature_measures is not None:
            self.feature_measures = feature_measures

        df_normal = df[self.features].copy()

        for feat in self.features:
            df_normal[feat] = (
                df_normal[feat] - self.feature_measures[feat]["mean"]
            ) / self.feature_measures[feat]["std"]

        return df_normal

    def predict(self, X):
        """
        Predict target for features X. If no model was fitted in this instance, the saved version of the model will
        be read.
        """

        if self.model is None:
            self.model = io.read_model(self.model_name)

        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predict target probabilities for features X. If no model was fitted in this instance, the saved version
        of the model will be read.
        """
        if self.model is None:
            self.model = io.read_model(self.model_name)

        return self.model.predict_proba(X)
