
import os, tarfile, bz2, requests


def download_uncompress_tar_bz2(url, path='.'):

    # open the stream
    r = requests.get(url, stream=True)

    tmp_file = 'temp_file.tar'

    # Download and uncompress at the same time.
    chunk_size = 4 * 1024 * 1024  # wait for chunks of 4MB
    with open(tmp_file, 'wb') as file:
        decompress = bz2.BZ2Decompressor()
        for chunk in r.iter_content(chunk_size=chunk_size):
            file.write(decompress.decompress(chunk))

    # finally untar the file to final destination
    tf = tarfile.open(tmp_file)
    tf.extractall(path)

    # remove the temporary file
    os.unlink(tmp_file)

