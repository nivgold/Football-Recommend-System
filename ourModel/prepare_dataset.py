from preprocessing import feature_loader

# loading relevant data from files:
team_historical = feature_loader.load_team_historical()
league_historical = feature_loader.load_league_historical()
current_form = feature_loader.load_current_form()
