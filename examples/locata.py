import numpy as np
import pyroomacoustics as pra

if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser(description='Test DOA algorithm on the locata data.')
    parser.add_argument('-a', '--algo', choices=doa.algos.keys(),
            help='doa algorithm')
    parser.add_argument('-l', '--locata', type=str, default=None,
            help='Location of LOCATA files')
    parser.add_argument('--task', type=int, default=1,
            help='LOCATA task number')
    parser.add_argument('--rec', type=int, default=1,
            help='LOCATA recording number')
