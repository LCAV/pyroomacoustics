'''
Example of the basic operations with ``pyroomacoustics.datasets.Dataset``
and ``pyroomacoustics.datasets.Sample`` classes
'''
from pyroomacoustics.datasets import Sample, Dataset

def test_dataset():
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

    assert len(dataset) == len(samples)


    assert dataset.info['speaker']['alice'] == 2
    assert dataset.info['speaker']['bob'] == 3
    assert dataset.info['speaker']['charles'] == 1

    assert dataset.info['sex']['male'] == 4
    assert dataset.info['sex']['female'] == 2

    assert dataset.info['age'][37] == 2
    assert dataset.info['age'][48] == 3
    assert dataset.info['age'][25] == 1

    assert dataset.info['number']['one'] == 2
    assert dataset.info['number']['two'] == 3
    assert dataset.info['number']['three'] == 1

    assert dataset[2].meta.speaker == 'bob'
    assert dataset[2].meta.sex == 'male'
    assert dataset[2].meta.age == 48
    assert dataset[2].meta.number == 'one'

    # We can obtain a new dataset with only male subject
    dataset2 = dataset.filter(sex='male')

    assert len(dataset2) == 4
    assert dataset2[0].meta.sex == 'male'
    assert dataset2[1].meta.sex == 'male'
    assert dataset2[2].meta.sex == 'male'
    assert dataset2[3].meta.sex == 'male'

    # select all samples with age larger than 30
    dataset3 = dataset.filter(age=lambda a : a > 30)
    assert len(dataset3) == 5
    for s in dataset3.samples:
        assert s.meta.age > 30

    # select all samples with a number that contains 't'
    dataset_contains_n = dataset.filter(number=['one','three'], speaker=['alice','bob'])
    assert len(dataset_contains_n) == 3
    for s in dataset_contains_n.samples:
        assert ( (s.meta.number == 'one' or s.meta.number == 'three')
                and (s.meta.speaker == 'alice' or s.meta.speaker == 'bob') )

if __name__ == '__main__':
    test_dataset()
