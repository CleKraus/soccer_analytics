# -*- coding: utf-8 -*-

# import packages
import logging

from preprocessing.cleaning_wyscout_data.event_data import cleanse_wyscout_event_data
from preprocessing.cleaning_wyscout_data.match_data import cleanse_wyscout_match_data
from preprocessing.cleaning_wyscout_data.player_data import cleanse_wyscout_player_data
from preprocessing.cleaning_wyscout_data.team_data import cleanse_wyscout_team_data

logging.basicConfig(level=logging.DEBUG)

"""
# You can set your country here: Data is available for 
    - Germany
    - Italy
    - England
    - Spain 
    - France
"""
countries = ["Germany"]

cleanse_wyscout_player_data()

for country in countries:
    cleanse_wyscout_match_data(country)
    cleanse_wyscout_team_data(country)
    cleanse_wyscout_event_data(country)
