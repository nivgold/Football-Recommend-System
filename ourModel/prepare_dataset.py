from feature_extraction import loader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

TEAM1_VS_TEAM2 = ["T1_VS_T2_PREC", "T1_AVG_GOAL_HOME_T2", "T1_STD_GOAL_HOME_T2", "T1_AVG_GOAL_AWAY_T2",
        "T2_AVG_GOAL_HOME_T1", "T2_AVG_GOAL_AWAY_T1", "T2_STD_GOAL_AWAY_T1", "T2_VS_T1_PREC"]
AWAY_Historical_Teams = ["AWAY_A_DRAW_PCT"]
HOME_Team_Historical = ["HOME_H_WIN_PCT", "HOME_H_DRAW_PCT"]
LEAGUE_Historical = ["L_HOME_GS_AVG", "L_AWAY_GS_AVG", "L_HOME_GS_STD", "L_AWAY_GS_STD", "L_HOME_WIN_PCT", "L_DRAW_PCT"]
FEATURES = TEAM1_VS_TEAM2 + AWAY_Historical_Teams + HOME_Team_Historical + LEAGUE_Historical


def make_dataset():
    all_team_historical = loader.load_team_historical()
    all_current_form = loader.load_current_form()
    all_team_historical_wins = loader.load_team_historical_winning()
    leagues_matches = loader.load_leagues_matches()
    league_historical = loader.load_league_historical()

    all_league_dataset = {}

    for league_name, league_matches in leagues_matches.items():
        for current_match in league_matches:
            match_id = current_match[0]
            season = current_match[1]
            home_team_id = current_match[2]
            away_team_id = current_match[3]
            result = current_match[4]

            # creating the vector of features for every match
            home_team_historical_wins = all_team_historical_wins[(league_name, home_team_id, season)][away_team_id]
            away_team_historical_wins = all_team_historical_wins[(league_name, away_team_id, season)][home_team_id]

            home_team_historical = all_team_historical[(league_name, home_team_id, season)]
            away_team_historical = all_team_historical[(league_name, away_team_id, season)]
            league_stats = league_historical[league_name, season];
            features_vector = home_team_historical_wins[0:4] + [home_team_historical_wins[5]] + home_team_historical_wins[7:]\
                              + [away_team_historical_wins[0]] + [away_team_historical[3]] + \
                              [home_team_historical[0]] + [home_team_historical[2]] + league_stats + [season] + [result]
            all_league_dataset[match_id] = features_vector

    # converting all league to DataFrame
    df = pd.DataFrame.from_dict(all_league_dataset, orient='index', columns=FEATURES + ["SEASON", "RESULT"])
    all_league_dataset = df.reset_index(drop=True)
    return all_league_dataset


def train_test_split(dataset_X, dataset_y):
    test_idx = dataset_X.index[dataset_X['SEASON'] == '2015/2016'].tolist()
    train_idx = list(set(dataset_X.index.tolist()) - set(test_idx))
    return dataset_X.iloc[train_idx, 0:-1], dataset_y.iloc[train_idx], dataset_X.iloc[test_idx, 0:-1], dataset_y.iloc[
        test_idx]


def preprocess():
    leagues_datasets = make_dataset()

    # for league_name, dataset in leagues_datasets.items():
    # fill NaN's
    dataset = leagues_datasets.fillna(leagues_datasets.mean())

    # Standardization
    dataset[FEATURES] = StandardScaler().fit_transform(dataset[FEATURES])

    X_train, y_train, X_test, y_test = train_test_split(dataset.iloc[:, 0:-1], dataset.iloc[:, -1])

    leagues_datasets = (X_train, y_train, X_test, y_test)

    return leagues_datasets
