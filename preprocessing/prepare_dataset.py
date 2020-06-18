from feature_extraction import loader
import pandas as pd
from sklearn.preprocessing import StandardScaler

HOME_TEAM_HISTORICAL_FEATURES = ["HOME_H_WIN_PCT", "HOME_A_WIN_PCT", "HOME_H_DRAW_PCT", "HOME_A_DRAW_PCT",
                                 "HOME_H_GS_AVG",
                                 "HOME_A_GS_AVG", "HOME_H_GC_AVG", "HOME_A_GC_AVG", "HOME_H_GS_STD", "HOME_A_GS_STD",
                                 "HOME_H_GC_STD", "HOME_A_GC_STD"]

AWAY_TEAM_HISTORICAL_FEATURES = ["AWAY_H_WIN_PCT", "AWAY_A_WIN_PCT", "AWAY_H_DRAW_PCT", "AWAY_A_DRAW_PCT",
                                 "AWAY_H_GS_AVG",
                                 "AWAY_A_GS_AVG", "AWAY_H_GC_AVG", "AWAY_A_GC_AVG", "AWAY_H_GS_STD", "AWAY_A_GS_STD",
                                 "AWAY_H_GC_STD", "AWAY_A_GC_STD"]

CURRENT_HOME_TEAM_FEATURES = ["CURRENT_HOME_WIN_PCT", "CURRENT_HOME_DRAW_PCT", "CURRENT_HOME_GS_AVG",
                              "CURRENT_HOME_GC_AVG", "CURRENT_HOME_GS_STD", "CURRENT_HOME_GC_STD"]

CURRENT_AWAY_TEAM_FEATURES = ["CURRENT_AWAY_WIN_PCT", "CURRENT_AWAY_DRAW_PCT", "CURRENT_AWAY_GS_AVG",
                              "CURRENT_AWAY_GC_AVG", "CURRENT_AWAY_GS_STD", "CURRENT_AWAY_GC_STD"]

LEAGUE_FEATURES = ["LEAGUE_H_GS_AVG", "LEAGUE_A_GS_AVG", "LEAGUE_H_GS_STD", "LEAGUE_A_GS_STD", "LEAGUE_H_WIN_PCT", "LEAGUE_DRAW_PCT"]

EACH_LEAGUE_FEATURES = HOME_TEAM_HISTORICAL_FEATURES + AWAY_TEAM_HISTORICAL_FEATURES + CURRENT_HOME_TEAM_FEATURES + CURRENT_AWAY_TEAM_FEATURES

ALL_LEAGUES_FEATURES = HOME_TEAM_HISTORICAL_FEATURES + AWAY_TEAM_HISTORICAL_FEATURES + CURRENT_HOME_TEAM_FEATURES + CURRENT_AWAY_TEAM_FEATURES + LEAGUE_FEATURES


def make_dataset_for_all_leagues():
    all_team_historical = loader.load_team_historical()
    all_current_form = loader.load_current_form()
    all_league_historical = loader.load_league_historical()
    leagues_matches = loader.load_leagues_matches()

    dataset_dict = {}

    for league_name, league_matches in leagues_matches.items():
        for current_match in league_matches:
            match_id = current_match[0]
            season = current_match[1]
            home_team_id = current_match[2]
            away_team_id = current_match[3]
            result = current_match[4]

            # creating the vector of features for every match
            # team historical
            home_team_historical = all_team_historical[(league_name, home_team_id, season)]
            away_team_historical = all_team_historical[(league_name, away_team_id, season)]

            # team current form
            home_team_current_form = all_current_form[home_team_id][match_id]
            away_team_current_form = all_current_form[away_team_id][match_id]

            # league historical
            league_stats = all_league_historical[(league_name, season)]

            features_vector = list(home_team_historical) + list(away_team_historical) + list(
                home_team_current_form) + list(away_team_current_form) + list(league_stats)

            dataset_dict[match_id] = features_vector + [season] + [result]

    # converting dict to DataFrame
    df = pd.DataFrame.from_dict(dataset_dict, orient='index',
                                columns=ALL_LEAGUES_FEATURES + ["SEASON", "RESULT"])
    df = df.reset_index(drop=True)

    return df


def make_dataset_for_each_league():
    all_team_historical = loader.load_team_historical()
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

            features_vector = list(home_team_historical) + list(away_team_historical) + list(
                home_team_current_form) + list(away_team_current_form)

            # features_vector = list(home_team_current_form) + list(away_team_current_form)

            league_dataset[match_id] = features_vector + [season] + [result]

        # converting dict to DataFrame
        df = pd.DataFrame.from_dict(league_dataset, orient='index', columns=EACH_LEAGUE_FEATURES + ["SEASON", "RESULT"])
        leagues_datasets[league_name] = df.reset_index(drop=True)

    return leagues_datasets


def train_test_split(dataset_X, dataset_y):
    test_idx = dataset_X.index[dataset_X['SEASON'] == '2015/2016'].tolist()
    train_idx = list(set(dataset_X.index.tolist()) - set(test_idx))
    return dataset_X.iloc[train_idx, 0:-1], dataset_y.iloc[train_idx], dataset_X.iloc[test_idx, 0:-1], dataset_y.iloc[
        test_idx]


def preprocess_all_leagues():
    dataset = make_dataset_for_all_leagues()

    # # fill NaN's
    # dataset = dataset.fillna(dataset.mean())

    # Standardization
    dataset[ALL_LEAGUES_FEATURES] = StandardScaler().fit_transform(
        dataset[ALL_LEAGUES_FEATURES])

    X_train, y_train, X_test, y_test = train_test_split(dataset.iloc[:, 0:-1], dataset.iloc[:, -1])

    return (X_train, y_train, X_test, y_test)


def preprocess_each_league():
    leagues_datasets = make_dataset_for_each_league()

    for league_name, dataset in leagues_datasets.items():
        # fill NaN's
        dataset = dataset.fillna(dataset.mean())

        # Standardization
        dataset[EACH_LEAGUE_FEATURES] = StandardScaler().fit_transform(
            dataset[EACH_LEAGUE_FEATURES])

        X_train, y_train, X_test, y_test = train_test_split(dataset.iloc[:, 0:-1], dataset.iloc[:, -1])

        leagues_datasets[league_name] = (X_train, y_train, X_test, y_test)

    return leagues_datasets
