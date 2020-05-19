# -*- coding: utf-8 -*-

"""
Pipeline for the cleansing of the Metrica tracking data
"""

# import packages
import logging

from preprocessing.cleansing_metrica_data.tracking_data import cleanse_tracking_data
from preprocessing.cleansing_metrica_data.event_data import cleanse_metrica_event_data
from preprocessing.cleansing_metrica_data.formation_data import build_formation_data

logging.basicConfig(level=logging.DEBUG)

# games you want to have cleansed
games = [1, 2]

# if True, the home team plays the first half from right to left
reverse = [False, True]

# loop through all games
for i, game in enumerate(games):

    # cleanse the event data
    cleanse_metrica_event_data(game, reverse[i])
    # cleanse the tracking data
    cleanse_tracking_data(game)
    # build the formation data
    build_formation_data(game)
