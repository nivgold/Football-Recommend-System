from feature_extraction import loader
import pandas as pd
from sklearn.preprocessing import StandardScaler

def make_dataset():
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
            result = current_match[4]

            # creating the vector of features for every match
            home_team_historical = all_team_historical[(league_name, home_team_id, season)]
            away_team_historical = all_team_historical[(league_name, away_team_id, season)]
            home_team_current_form = all_current_form[home_team_id][match_id]
            away_team_current_form = all_current_form[away_team_id][match_id]
            league_historical = all_league_historical[(league_name, season)]
            features_vector = list(home_team_historical) + list(away_team_historical) + list(
                home_team_current_form) + list(away_team_current_form) + [result]
            league_dataset[match_id] = features_vector
            columns = ["HOME_H_WIN_PCT", "HOME_A_WIN_PCT", "HOME_H_DRAW_PCT", "HOME_A_DRAW_PCT", "HOME_H_GS_AVG", "HOME_A_GS_AVG", "HOME_H_GC_AVG", "HOME_A_GC_AVG", "HOME_H_GS_STD", "HOME_A_GS_STD", "HOME_H_GC_STD", "HOME_A_GC_STD",
                       "AWAY_H_WIN_PCT", "AWAY_A_WIN_PCT", "AWAY_H_DRAW_PCT", "AWAY_A_DRAW_PCT", "AWAY_H_GS_AVG", "AWAY_A_GS_AVG", "AWAY_H_GC_AVG", "AWAY_A_GC_AVG", "AWAY_H_GS_STD", "AWAY_A_GS_STD", "AWAY_H_GC_STD", "AWAY_A_GC_STD",
                       "HOME_WIN_PCT", "HOME_DRAW_PCT", "HOME_GS_AVG", "HOME_GC_AVG", "HOME_GS_STD", "HOME_GC_STD",
                       "AWAY_WIN_PCT", "AWAY_DRAW_PCT", "AWAY_GS_AVG", "AWAY_GC_AVG", "AWAY_GS_STD", "AWAY_GC_STD",
                       "RESULT"]
        # converting dict to DataFrame
        df = pd.DataFrame.from_dict(league_dataset, orient='index', columns=columns)
        leagues_datasets[league_name] = df

    return leagues_datasets

def preprocess():
    leagues_datasets = make_dataset()




    for league_name, dataset in leagues_datasets.items():
        # fill NaN's
        dataset = dataset.fillna(dataset.mean())

        X = dataset.iloc[:, 0:-1]
        # Standardization
        scaler = StandardScaler().fit()

    return leagues_datasets