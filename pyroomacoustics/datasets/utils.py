import os, tarfile, bz2, requests, gzip


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


def download_uncompress_tar_gz(url, path='.', chunk_size=None):

    tmp_file = 'tmp.tar.gz'
    if chunk_size is None:
        chunk_size = 4 * 1024 * 1024

    # stream the data
    r = requests.get(url, stream=True)
    with open(tmp_file, 'wb') as f:
        content_length = int(r.headers['content-length'])
        count = 0
        for chunk in r.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            count += 1
            print("%d bytes out of %d downloaded" % 
                (count*chunk_size, content_length))
    r.close()

    # uncompress
    tar_file = 'tmp.tar'
    with open(tar_file, "wb") as f_u:
        with gzip.open(tmp_file, "rb") as f_c:
            f_u.write(f_c.read())

    # finally untar the file to final destination
    tf = tarfile.open(tar_file)

    if not os.path.exists(path):
        os.makedirs(path)
    tf.extractall(path)

    # remove the temporary file
    os.unlink(tmp_file)
    os.unlink(tar_file)


