import shutil
from pyroomacoustics.datasets.utils import download_uncompress

test_url = 'https://github.com/LCAV/pyroomacoustics/archive/master.tar.gz'
extracted_name = 'pyroomacoustics-master'

def test_download_uncompress():

    download_uncompress(test_url)
    shutil.rmtree(extracted_name, ignore_errors=True)

if __name__ == '__main__':

    test_download_uncompress()
