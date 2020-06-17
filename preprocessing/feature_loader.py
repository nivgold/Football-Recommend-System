import pickle


def load_team_historical():
    with open('../files/team_historical.pickle', 'rb') as file:
        return pickle.load(file)

def load_league_historical():
    with open('../files/league_historical.pickle', 'rb') as file:
        return pickle.load(file)

def load_current_form():
    with open('../files/current_form.pickle', 'rb') as file:
        return pickle.load(file)

