import argparse
import json

#func for input argument
def input_argument_parser():
    input_parser = argparse.ArgumentParser()
    input_parser.add_argument('--name', type=str, dest='name', required=True,
        help='input name as argument to execute program')
    return input_parser

def similar_ppl_finder(dataset, name, num_ppl):
    if name not in dataset:
        raise TypeError('Cannot find ' + name + ' in the dataset')

    if num_ppl < 1:
        raise TypeError('Amount of people must be equal or greater than 1')

if __name__ == '__main__':
    args = input_argument_parser().parse_args()
    input_name = args.name

#our json file
rating_file = 'data.json'

#open file for reading 'r'
with open(rating_file, mode='r') as file:
    data = json.loads(file.read())

print(input_name + ' movies rating: \n',data[input_name])

print('\nUsers similar to ' + input_name + ':\n')

#similar_users = similar_ppl_finder(rating_file, input_name, 3)
