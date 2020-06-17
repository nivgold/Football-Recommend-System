from preprocessing import loader, ORM
import pandas as pd

ORM.map_db(False)
team_historical = loader.load_team_historical()
league_historical = loader.load_league_historical()
current_form = loader.load_current_form()
leagues_matches = loader.load_leagues_matches()

league_datasets = {}

