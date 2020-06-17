from preprocessing.ORM import *
from collections import defaultdict
import numpy as np
import pickle


@db_session
def get_last_5_games(team, date):
    prev_5_games = list(select(match for match in Match if
                               (match.home_team == team or match.away_team == team) and match.date < raw_sql(
                                   f'date(\'{str(date)}\') ORDER BY date(date) DESC Limit 5')))
    return list(prev_5_games)


@db_session
def make_current_form():
    matches_current_form = defaultdict(dict)
    all_matches = list(select(m for m in Match))
    for match in all_matches:
        last_home_team_5 = get_last_5_games(team=Match[match.id].home_team, date=Match[match.id].date)
        last_away_team_5 = get_last_5_games(team=Match[match.id].away_team, date=Match[match.id].date)

        matches_current_form[Match[match.id].home_team.team_id][match.id] = calculate_current_form_features(
            Match[match.id].home_team, last_home_team_5)
        matches_current_form[Match[match.id].away_team.team_id][match.id] = calculate_current_form_features(
            Match[match.id].away_team, last_away_team_5)
    return matches_current_form


@db_session
def calculate_current_form_features(team, last_matches):
    num_of_games = len(last_matches)
    if num_of_games != 0:
        home_win_per = len([1 for match in last_matches if
                           Match[match.id].home_team == team and Match[match.id].home_team_goal > Match[
                               match.id].away_team_goal])
        away_win_per = len([1 for match in last_matches if
                           Match[match.id].away_team == team and Match[match.id].away_team_goal > Match[
                               match.id].home_team_goal])
        win_per = home_win_per + away_win_per
        draw_per = len(
            [1 for match in last_matches if Match[match.id].home_team_goal == Match[match.id].away_team_goal])
        home_win_goals = np.mean(
            [Match[match.id].home_team_goal for match in last_matches if Match[match.id].home_team == team])
        away_win_goals = np.mean(
            [Match[match.id].away_team_goal for match in last_matches if Match[match.id].away_team == team])
        win_goals_mean = np.mean([home_win_goals, away_win_goals])
        win_goals_std = np.std([home_win_goals, away_win_goals])
        home_lose_goals = np.mean(
            [Match[match.id].away_team_goal for match in last_matches if Match[match.id].home_team == team])
        away_lose_goals = np.mean(
            [Match[match.id].home_team_goal for match in last_matches if Match[match.id].away_team == team])
        lose_goals_mean = np.mean([home_lose_goals, away_lose_goals])
        lose_goals_std = np.std([home_lose_goals, away_lose_goals])
        stats_vector = [win_per, draw_per, win_goals_mean, lose_goals_mean, win_goals_std, lose_goals_std]
    else:
        nan_arr = np.empty(6)
        nan_arr[:] = np.NaN
        stats_vector = nan_arr
    return stats_vector


def main():
    all_matches_current_form = make_current_form()
    with open('../files/current_form.pickle', 'wb') as current_form_file:
        pickle.dump(all_matches_current_form, current_form_file, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    map_db(False)
    main()
