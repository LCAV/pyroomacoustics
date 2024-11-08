import shutil
import ssl

from pyroomacoustics.datasets.utils import download_uncompress

test_url = "https://github.com/LCAV/pyroomacoustics/archive/master.tar.gz"
extracted_name = "pyroomacoustics-master"


def test_download_uncompress():
    context = ssl._create_unverified_context()
    download_uncompress(test_url, context=context)
    shutil.rmtree(extracted_name, ignore_errors=True)


if __name__ == "__main__":
    test_download_uncompress()
