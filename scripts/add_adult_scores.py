import json

path = 'config/nba_teams.json'
with open(path, 'r', encoding='utf-8') as f:
    teams = json.load(f)

custom = {
    'LAL': 7, 'LAC': 6, 'MIA': 9, 'NYK': 6, 'BOS': 3, 'MIL': 8, 'HOU': 8,
    'TOR': 6, 'OKC': 3, 'UTA': 2, 'DEN': 7, 'DAL': 7, 'PHX': 5, 'PHI': 6,
    'BKN': 3, 'NOP': 8, 'SAC': 3, 'POR': 9, 'MEM': 7, 'IND': 3, 'ATL': 10, 
    'WAS': 5, 'CLE': 6, 'MIN': 7, 'ORL': 5, 'SAS': 5, 'CHA': 5, 'DET': 9,
    'GSW': 6, 'CHI': 4, 
}

def rating(abbr):
    return custom.get(abbr, 5)

for t in teams:
    if 'adult_quality_rating' not in t:
        t['adult_quality_rating'] = rating(t['abbreviation'])

with open(path, 'w', encoding='utf-8') as f:
    json.dump(teams, f, indent=2)

print(f'Updated {len(teams)} teams with adult_quality_rating.')
