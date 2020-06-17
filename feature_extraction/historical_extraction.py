from feature_extraction.ORM import *
import numpy as np
import pickle


@db_session
def extract_season_historical_matches():
    seasons_historical_matches = {}
    all_seasons_matches = select(m.season for m in Match)
    for season in all_seasons_matches:
        two_last_seasons_matches = select(m for m in Match if
                                          get_prev_k_season(season, 1) == m.season or get_prev_k_season(season,
                                                                                                        2) == m.season or season == m.season)
        seasons_historical_matches[season] = [match for match in two_last_seasons_matches]
    return seasons_historical_matches


@db_session
def extract_season_6_historical_matches():
    seasons_6_historical_matches = {}
    all_seasons_matches = select(m.season for m in Match)
    for season in all_seasons_matches:
        six_last_seasons_matches = select(m for m in Match if
                                          season == m.season or get_prev_k_season(season, 1) == m.season or get_prev_k_season(season, 2) == m.season
                                          or get_prev_k_season(season, 3) == m.season or get_prev_k_season(season, 4) == m.season
                                          or get_prev_k_season(season, 5) == m.season)
        seasons_6_historical_matches[season] = [match for match in six_last_seasons_matches]
    return seasons_6_historical_matches


@db_session
def make_league_historical(leagues, season_historical_matches):
    league_season_historical = {}
    league_season_agg = {}
    for l_name in leagues:
        for season, matches in season_historical_matches.items():
            league_season_agg[(l_name, season)] = [match for match in matches if
                                                   League[match.league.league_id].name == l_name]
    for (league_name, season), matches in league_season_agg.items():
        total_games = len(matches)
        home_wins = len([1 for match in matches if match.home_team_goal > match.away_team_goal])
        draw_games = len([1 for match in matches if match.home_team_goal == match.away_team_goal])
        avg_home_goals = np.mean([match.home_team_goal for match in matches])
        avg_away_goals = np.mean([match.away_team_goal for match in matches])
        std_home_goals = np.std([match.home_team_goal for match in matches])
        std_away_goals = np.std([match.away_team_goal for match in matches])
        home_wins_per = home_wins / total_games
        draw_per = draw_games / total_games
        league_season_historical[(league_name, season)] = [avg_home_goals, avg_away_goals, std_home_goals,
                                                           std_away_goals, home_wins_per, draw_per]
    return league_season_historical


@db_session
def make_teams_historical(leagues, season_historical_matches):
    teams_historical = {}
    league_season_team_agg = {}
    # for every league
    for l_name in leagues:
        for season, matches in season_historical_matches.items():
            league_season_teams = get_league_season_teams(l_name, season)
            for team in league_season_teams:
                league_season_team_agg[(l_name, team, season)] = [match for match in matches if (
                            (Match[match.id].league.name == l_name) and (
                                Match[match.id].home_team == team or Match[match.id].away_team == team))]

    for (league_name, team, season), matches in league_season_team_agg.items():
        total_games = len(matches)
        if total_games != 0:
            home_win_per = len([1 for match in matches if
                                Match[match.id].home_team == team and Match[match.id].home_team_goal > Match[
                                    match.id].away_team_goal]) / total_games
            away_win_per = len([1 for match in matches if
                                Match[match.id].away_team == team and Match[match.id].away_team_goal > Match[
                                    match.id].home_team_goal]) / total_games
            home_draw_per = len([1 for match in matches if
                                 Match[match.id].home_team == team and Match[match.id].home_team_goal == Match[
                                     match.id].away_team_goal]) / total_games
            away_draw_per = len([1 for match in matches if
                                 Match[match.id].away_team == team and Match[match.id].home_team_goal == Match[
                                     match.id].away_team_goal]) / total_games
            avg_home_win_goal = np.mean(
                [match.home_team_goal for match in matches if Match[match.id].home_team == team])
            avg_away_win_goal = np.mean(
                [match.away_team_goal for match in matches if Match[match.id].away_team == team])
            avg_home_lose_goal = np.mean(
                [match.away_team_goal for match in matches if Match[match.id].home_team == team])
            avg_away_lose_goal = np.mean(
                [match.home_team_goal for match in matches if Match[match.id].away_team == team])
            std_home_win_goal = np.std([match.home_team_goal for match in matches if Match[match.id].home_team == team])
            std_away_win_goal = np.std([match.away_team_goal for match in matches if Match[match.id].away_team == team])
            std_home_lose_goal = np.std(
                [match.away_team_goal for match in matches if Match[match.id].home_team == team])
            std_away_lose_goal = np.std(
                [match.home_team_goal for match in matches if Match[match.id].away_team == team])
            teams_historical[(league_name, team.team_id, season)] = [home_win_per, away_win_per, home_draw_per, away_draw_per,
                                                             avg_home_win_goal, avg_away_win_goal, avg_home_lose_goal,
                                                             avg_away_lose_goal, std_home_win_goal, std_away_win_goal,
                                                             std_home_lose_goal, std_away_lose_goal]
        else:
            print(f'{team.name} DIVISION BY ZERO')
            teams_historical[(league_name, team.team_id, season)] = np.empty((12))
            teams_historical[(league_name, team.team_id, season)][:] = np.nan
    return teams_historical


@db_session
def make_teams_historical_winning(leagues, season_historical_matches):
    teams_historical_winning = {}
    league_season_team_agg = {}
    # for every league
    for l_name in leagues:
        for season, matches in season_historical_matches.items():
            league_season_teams = get_league_6_season_teams(l_name, season)
            for team in league_season_teams:
                league_season_team_agg[(l_name, team, season)] = [match for match in matches if (
                        (Match[match.id].league.name == l_name) and (
                         Match[match.id].home_team == team or Match[match.id].away_team == team))]

    for (league_name, team, season), matches in league_season_team_agg.items():
        total_games = len(matches)
        if total_games != 0:
            relevant_teams = get_league_6_season_teams(league_name, season)
            list_per_winning = {}
            for team_rival in relevant_teams:
                if team_rival == team:
                    continue
                home_games_t1_t2 = len([1 for match in matches if
                                        (Match[match.id].home_team == team and
                                         Match[match.id].away_team == team_rival)])
                if home_games_t1_t2 > 0:
                    avg_team_home_goal = np.mean(
                        [match.home_team_goal for match in matches if
                         Match[match.id].home_team == team and Match[match.id].away_team == team_rival])
                    avg_rival_away_goal = np.mean(
                        [match.away_team_goal for match in matches if
                         Match[match.id].home_team == team and Match[match.id].away_team == team_rival])
                else:
                    avg_team_home_goal = 0
                    avg_rival_away_goal = 0

                away_games_t1_t2 = len([1 for match in matches if
                                        (Match[match.id].home_team == team_rival and
                                         Match[match.id].away_team == team)])
                if away_games_t1_t2 > 0:
                    avg_rival_home_goal = np.mean(
                        [match.home_team_goal for match in matches if
                         Match[match.id].home_team == team_rival and Match[match.id].away_team == team])
                    avg_team_away_goal = np.mean(
                        [match.away_team_goal for match in matches if
                         Match[match.id].home_team == team_rival and Match[match.id].away_team == team])
                else:
                    avg_rival_home_goal = 0
                    avg_team_away_goal = 0

                if home_games_t1_t2 + away_games_t1_t2 > 0:
                    win_team_per = len([1 for match in matches if ((Match[match.id].home_team == team and Match[
                                        match.id].away_team == team_rival and Match[match.id].home_team_goal >
                                            Match[match.id].away_team_goal) or (Match[match.id].home_team == team_rival
                                                and Match[match.id].away_team == team and Match[match.id].away_team_goal >
                                                Match[match.id].home_team_goal))]) / (home_games_t1_t2 + away_games_t1_t2)
                list_per_winning[team_rival.team_id] = [win_team_per, avg_team_home_goal,
                                                        avg_team_away_goal, avg_rival_home_goal, avg_rival_away_goal]
            teams_historical_winning[(league_name, team.team_id, season)] = list_per_winning

        else:
            print(f'{team.name} DIVISION BY ZERO')
            teams_historical_winning[(league_name, team.team_id, season)] = np.empty((299))
            teams_historical_winning[(league_name, team.team_id, season)][:] = np.nan
    return teams_historical_winning


def main():
    map_db(False)

    with db_session:
        seasons_historical_matches_global = extract_season_historical_matches()
    with db_session:
        leagues = [league_name for league_name in select(l.name for l in League)]

    with db_session:
        seasons_6_historical_matches_global = extract_season_6_historical_matches()

    league_season_historical = make_league_historical(leagues, seasons_historical_matches_global)

    league_teams_season_historical = make_teams_historical(leagues, seasons_historical_matches_global)

    league_teams_winning_season_historical = make_teams_historical_winning(leagues, seasons_6_historical_matches_global)

    with open('../files/league_historical.pickle', 'wb') as league_historical_file:
        pickle.dump(league_season_historical, league_historical_file, protocol=pickle.HIGHEST_PROTOCOL)

    with open('../files/team_historical.pickle', 'wb') as team_historical_file:
        pickle.dump(league_teams_season_historical, team_historical_file, protocol=pickle.HIGHEST_PROTOCOL)

    with open('../files/team_historical_winning.pickle', 'wb') as team_historical_winning_file:
        pickle.dump(league_teams_winning_season_historical, team_historical_winning_file, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
