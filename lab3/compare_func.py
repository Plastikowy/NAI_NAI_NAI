

def compute_func(dataset, first_person, sec_person):
    if first_person not in dataset:
        raise TypeError('Cannot find ' + first_person + 'in the dataset')

    if sec_person not in dataset:
        raise TypeError('Cannot find ' + sec_person + 'in the dataset')