from feature_extraction import loader

# loading relevant data from files:
team_historical = loader.load_team_historical()
team_historical_winning = loader.load_team_historical_winning()
print(len(team_historical_winning))
league_historical = loader.load_league_historical()
current_form = loader.load_current_form()
