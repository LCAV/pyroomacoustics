'''
Example of the basic operations with ``pyroomacoustics.datasets.CorpusBase``
and ``pyroomacoustics.datasets.SampleBase`` classes
'''
from pyroomacoustics.datasets import SampleBase, CorpusBase

def test_corpus():
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

    corpus = CorpusBase()
    for s in samples:
        new_sample = SampleBase(s['data'], **s['metadata'])
        corpus.add_sample(new_sample)

    assert len(corpus) == len(samples)


    assert corpus.info['speaker']['alice'] == 2
    assert corpus.info['speaker']['bob'] == 3
    assert corpus.info['speaker']['charles'] == 1

    assert corpus.info['sex']['male'] == 4
    assert corpus.info['sex']['female'] == 2

    assert corpus.info['age'][37] == 2
    assert corpus.info['age'][48] == 3
    assert corpus.info['age'][25] == 1

    assert corpus.info['number']['one'] == 2
    assert corpus.info['number']['two'] == 3
    assert corpus.info['number']['three'] == 1

    assert corpus[2].meta.speaker == 'bob'
    assert corpus[2].meta.sex == 'male'
    assert corpus[2].meta.age == 48
    assert corpus[2].meta.number == 'one'

    # We can obtain a new corpus with only male subject
    corpus_male_only = corpus.filter(sex='male')

    assert len(corpus_male_only) == 4
    assert corpus_male_only[0].meta.sex == 'male'
    assert corpus_male_only[1].meta.sex == 'male'
    assert corpus_male_only[2].meta.sex == 'male'
    assert corpus_male_only[3].meta.sex == 'male'

if __name__ == '__main__':
    test_corpus()
