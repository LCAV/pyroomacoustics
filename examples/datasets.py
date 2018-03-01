'''
Example of the basic operations with ``pyroomacoustics.datasets.Dataset``
and ``pyroomacoustics.datasets.Sample`` classes
'''
from pyroomacoustics.datasets import Sample, Dataset

# Prepare a few artificial samples
samples = [
    {
        'data' : 0.99,
        'metadata' : { 'speaker' : 'alice', 'sex' : 'female', 'age' : 37, 'number' : 'one' },
    },
    {
        'data' : 2.1,
        'metadata' : { 'speaker' : 'alice', 'sex' : 'female', 'age' : 37, 'number' : 'two' },
    },
    {
        'data' : 1.02,
        'metadata' : { 'speaker' : 'bob', 'sex' : 'male', 'age' : 48, 'number' : 'one' },
    },
    {
        'data' : 2.07,
        'metadata' : { 'speaker' : 'bob', 'sex' : 'male', 'age' : 48, 'number' : 'two' },
    },
    {
        'data' : 3.0,
        'metadata' : { 'speaker' : 'bob', 'sex' : 'male', 'age' : 48, 'number' : 'three' },
    },
    {
        'data' : 1.97,
        'metadata' : { 'speaker' : 'charles', 'sex' : 'male', 'age' : 25, 'number' : 'two' },
    },
    ]

dataset = Dataset()
for s in samples:
    new_sample = Sample(s['data'], **s['metadata'])
    dataset.add_sample(new_sample)

# Then, it possible to display summary info about the dataset
print(dataset)
print()

# The number of samples in the dataset is given by ``len``
print('Number of samples:', len(dataset))
print()

# And we can access samples with the slice operator
print('Sample #2:')
print(dataset[2])
print()

# We can obtain a new dataset with only male subject
dataset_male_only = dataset.filter(sex='male')
print('Dataset with only males')
print(dataset_male_only)
print()

# Only retain speakers above 40 years old
dataset_older = dataset.filter(age=lambda a : a > 40)
print('Dataset with only above 40 yo')
print(dataset_older)
