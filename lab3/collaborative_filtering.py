import argparse
import json
import numpy as np

from compute_scores import euclidean_score

def build_arg_parser():
    parser = argparse.ArgumentParser(description='Find users who are similar to the input user')
    parser.add_argument('--user', dest='user', required=True,
            help='Input user')
    return parser

# Finds users in the dataset that are similar to the input user
def find_similar_users(dataset, user, num_users):
    if user not in dataset:
        raise TypeError('Cannot find ' + user + ' in the dataset')

    # Compute Pearson score between input user
    # and all the users in the dataset
    scores = np.array([[x, euclidean_score(dataset, user,
            x)] for x in dataset if x != user])

    # Sort the scores in decreasing order
    scores_sorted = np.argsort(scores[:, 1])[::-1]

    # Extract the top 'num_users' scores
    top_users = scores_sorted[:num_users]

    return scores[top_users]

def add_similar_users_to_list(database, similars):
    #saves similar user's names to list
    for item in database:
        for sim_user in similars:
            if(item == sim_user[0]):
                similars_users_names.append(item)

def create_films_set():
    #add films from each similar user
    for name in similars_users_names:
        # print(data[name])
        savefilms(data[name])

    #we need to remove lecturer seen movies
    for paul_film in data[user]:
        films.remove(paul_film)
    #print(films)

def find_recommended_films():
    #initialize variables for describing recommended films
    score_index = 10

    #going from the top rating, we are looping through all the films
    #we check if users watched shared films and if so, we check their ratings,
    #if it's current searched top rating, we add it to recommended films
    while score_index > 0:
        for film in films:
            for name in similars_users_names:
                  if (len(recommended_films) < 5):
                      if (film in data[name] and data[name][film] == score_index):
                          # print(film, 'equals', data[name][film])
                          recommended_films.append(film)
        score_index-=1

def find_not_recommended_films():
    #initialize variables for describing recommended films
    score_index = 1

    #going from the top rating, we are looping through all the films
    #we check if users watched shared films and if so, we check their ratings,
    #if it's current searched top rating, we add it to recommended films
    while score_index < 11:
        for film in films:
            for name in similars_users_names:
                  if (len(not_recommended_films) < 5):
                      if (film in data[name] and data[name][film] == score_index):
                          # print(film, 'equals', data[name][film])
                          not_recommended_films.append(film)
        score_index+=1

#function to save films to set
def savefilms(user_data):
    for film in user_data:
        films.add(film)

if __name__=='__main__':
    recommended_films = []
    not_recommended_films = []
    similars_users_names = []
    films = set() #we dont want repeated films
    args = build_arg_parser().parse_args()
    user = args.user

    ratings_file = 'NAI_ratings.json'

    with open(ratings_file, 'r') as f:
        data = json.loads(f.read())

    savefilms(data[user])

    print('\nUsers similar to ' + user + ':\n')
    similar_users = find_similar_users(data, user, 3)


    print('User\t\t\tSimilarity score')
    print('-'*41)
    for item in similar_users:
        print(item[0], '\t\t', round(float(item[1]), 2))

    add_similar_users_to_list(data, similar_users)
    create_films_set()
    find_recommended_films()
    find_not_recommended_films()

    print('RECOMMENDED FILMS: ', recommended_films)
    print('NOT RECOMMENDED FILMS: ', not_recommended_films)
