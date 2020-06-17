from preprocessing import loader
import pandas as pd


def prepare_dataset():
    all_team_historical = loader.load_team_historical()
    all_league_historical = loader.load_league_historical()
    all_current_form = loader.load_current_form()
    leagues_matches = loader.load_leagues_matches()

    leagues_datasets = {}

    for league_name, league_matches in leagues_matches.items():
        league_dataset = {}
        for current_match in league_matches:
            match_id = current_match[0]
            season = current_match[1]
            home_team_id = current_match[2]
            away_team_id = current_match[3]

            # creating the vector of features for every match
            home_team_historical = all_team_historical[(league_name, home_team_id, season)]
            away_team_historical = all_team_historical[(league_name, away_team_id, season)]
            home_team_current_form = all_current_form[home_team_id][match_id]
            away_team_current_form = all_current_form[away_team_id][match_id]
            league_historical = all_league_historical[(league_name, season)]
            features_vector = list(home_team_historical) + list(away_team_historical) + list(home_team_current_form) + list(away_team_current_form) + list(league_historical)
            league_dataset[match_id] = features_vector
        # converting dict to DataFrame
        df = pd.DataFrame.from_dict(league_dataset, orient='index')
        leagues_datasets[league_name] = df

    return leagues_datasets