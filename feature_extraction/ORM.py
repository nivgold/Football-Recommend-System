from pony.orm import *
import datetime

db = Database()

DB_PATH = './dataset/database.sqlite'

# dont use create_db because database.sqlite already exists
db.bind(provider='sqlite', filename=DB_PATH)


class League(db.Entity):
    league_id = PrimaryKey(int, column='id')
    name = Required(str)
    matches = Set('Match')


class Team(db.Entity):
    team_id = PrimaryKey(int, column='team_api_id')
    name = Required(str, column='team_long_name', index=True)
    home_matches = Set('Match', reverse='home_team')
    away_matches = Set('Match', reverse='away_team')


class Match(db.Entity):
    id = PrimaryKey(int)
    date = Required(datetime.date)
    season = Required(str)
    match_api_id = Required(int)
    home_team_goal = Required(int)
    away_team_goal = Required(int)
    league = Required(League, column='league_id')
    home_team = Required(Team, reverse='home_matches', column='home_team_api_id')
    away_team = Required(Team, reverse='away_matches', column='away_team_api_id')


def get_prev_k_season(season, k):
    data = season.split("/")
    return f'{int(data[0]) - k}/{int(data[1]) - k}'


@db_session
def get_league_season_teams(league_name, season):
    home_teams = list(select(m.home_team for m in Match if m.league.name == league_name and (
            m.season == season or m.season == get_prev_k_season(season, 1) or m.season == get_prev_k_season(season,
                                                                                                            2))))
    away_teams = list(select(m.away_team for m in Match if m.league.name == league_name and (
            m.season == season or m.season == get_prev_k_season(season, 1) or m.season == get_prev_k_season(season,
                                                                                                            2))))
    return list(set(home_teams + away_teams))


@db_session
def get_all_league_names():
    all_leagues = select(league.name for league in League)
    return list(all_leagues)


@db_session
def get_all_league_matches(league_name):
    all_league_matches = select(match for match in Match if match.league.name == league_name)
    return list(all_league_matches)


def map_db(debug=True):
    set_sql_debug(debug)
    db.generate_mapping(create_tables=True)
