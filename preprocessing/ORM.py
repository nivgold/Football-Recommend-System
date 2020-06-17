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


def map_db(debug=True):
    set_sql_debug(debug)
    db.generate_mapping(create_tables=True)
