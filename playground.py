from math import log

def entropy(data):
    labels = [d[1] for d in data] # Get label for each data point
    distinct_labels = set(labels) # Get distinct labels in set
    
     # If there is only one label return entropy of 1 ( to avoid log_2(1) )
    if len(distinct_labels) == 1: 
        return 1.0

    probs = [ # Calculate probabilites for each distinct label
        labels.count(label)/len(data) # Count of each label by total number of occurrences
        for label in distinct_labels]

    # Calculate total entropy by weighting the number of bits necessary to
    # represent the probability ( log_2(probability) ) by the probability itself
    return -sum([p * log(p, 2) for p in probs]) 

data = [(0.1,   'FAIL'),
        (0.2,   'FAIL'),
        (0.8,   'FAIL'),
        (0.9,   'OK'),
        (1.0,   'FAIL'),
        (4.0,   'OK'),
        (10.0,  'OK'),
        (50.0,  'OK')]

print(entropy(data))

