# -*- coding: utf-8 -*-

# import packages
import pandas as pd


def get_time_around_special_event(df_events, event_id, secs_before=None, secs_after=None):
    """
    Given an *event_id* the function returns all the events between the event - *seconds_before* and
    the event + *seconds_after*
    :param df_events: (pd.DataFrame) Data frame with all events
    :param event_id: (int) Event id of the event we are interested in
    :param secs_before: (float, optional) All events *secs_before* seconds before the *event_id* are returned. If None,
                        no events before the *event_id* will be returned
    :param secs_after:  (float, optional) All events *secs_after* seconds after the *event_id* are returned. If None,
                        no events after the *event_id* will be returned
    :return: pd.DataFrame with all events between *secs_before* the *event_id* and *secs_after* the *event_id*
    """
    tmp_event = df_events[df_events["id"] == event_id]
    match_id = tmp_event.iloc[0]["matchId"]
    match_period = tmp_event.iloc[0]["matchPeriod"]
    event_sec = tmp_event.iloc[0]["eventSec"]

    if secs_before is None:
        secs_before = 0.01

    if secs_after is None:
        secs_after = 0.01

    df_special_event = df_events[(df_events["matchId"] == match_id) &
                                 (df_events["matchPeriod"] == match_period) &
                                 (df_events["eventSec"] >= event_sec - secs_before) &
                                 (df_events["eventSec"] <= event_sec + secs_after)]
    return df_special_event


def get_event_after(df, df_events, considered_events, cols_return):
    """
    For each event in *df* the function computes the next event happening out of a list of *considered_event*.
    :param df: (pd.DataFrame) Data frame containing all events for which we want to compute when the following event
    :param df_events: (pd.DataFrame) Data frame with all events (should at least contain all events of the types
                      specified in *considered_events*.
    :param considered_events: (list) List containing all events that should be considered when searching for the next
                               event.
    :param cols_return: (dict) Dictionary containing the information of the following event as well as their column
                         name
    :return: pd.DataFrame containing the *df* data frame enlarged with columns on the next events specified in
             *considered events*

    Example: If you wish to get the time and player of the next pass after a goal kick, the parameters might look like
    the following:
        df : Data frame with all goal kicks
        df_events: Data frame with all events
        considered_events: [Pass]
        cols_returned: {eventSec: timeNextPass, playerId: playerNextPass
    """

    # make a copy of *df* as basis for the output
    df_out = df.copy()

    if "eventSec" in cols_return:
        colname_sec = cols_return["eventSec"]
        cols_return.pop("eventSec")
    else:
        colname_sec = None

    df = df[["matchId", "matchPeriod", "id", "eventSec"]].copy()
    df_rel_events = df_events[df_events["eventName"].isin(considered_events)].copy()

    rel_cols = ["matchId", "matchPeriod", "eventSec"] + list(cols_return.keys())
    df_rel_events = df_rel_events[rel_cols].copy()
    df_rel_events.rename(columns=cols_return, inplace=True)

    df_rel_events.rename(columns={"eventSec": "eventSecRelEvent"}, inplace=True)

    # add all events of the relevant type(s) and only keep those that happened after
    df_all = pd.merge(df, df_rel_events, on=["matchId", "matchPeriod"])
    df_all = df_all[df_all["eventSecRelEvent"] > df_all["eventSec"]].copy()

    # sort the values and only keep the first one
    df_all.sort_values(["id", "eventSecRelEvent"], inplace=True)
    df_event_after = df_all.drop_duplicates("id")

    if colname_sec is not None:
        df_event_after = df_event_after.rename(columns={"eventSecRelEvent": colname_sec})
        cols = ["id", colname_sec]
    else:
        cols = ["id"]

    df_event_after = df_event_after[cols + [cols_return[key] for key in cols_return.keys()]].copy()

    df_out = pd.merge(df_out, df_event_after, how="left")
    return df_out
