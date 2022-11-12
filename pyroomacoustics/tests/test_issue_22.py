"""
This test attempts to check that the C module
runs fine. It was created to test Issue #22 on github
whereas the program would crash with
"Fatal Python Error: deallocating None"
after around 20000 calls to Room.compute_rir().

The bug was fixed by adding reference count increments
before returning Py_None in the C extension.

The test creates a random but fixed room and calls
Room.image_source_model() 25000 times.

If the C module is not installed (pure python
fallback version), then nothing is done.
"""
import numpy as np
import pyroomacoustics


def test_issue_22():

    np.random.seed(0)

    n_mics = 1
    n_src = 1
    n_times = 25000
    dim = 3
    mic_pos = np.random.rand(dim, n_mics)
    abs_coeff = 0.1
    e_abs = 1.0 - (1.0 - abs_coeff) ** 2
    fs = 16000
    wall_max_len = 15

    room_dim = np.random.rand(dim) * wall_max_len

    shoebox = pyroomacoustics.ShoeBox(
        room_dim,
        materials=pyroomacoustics.Material(e_abs),
        fs=fs,
        max_order=0,
    )

    src_pos = np.random.rand(dim, n_src) * room_dim[:, None]
    for src in src_pos.T:
        shoebox.add_source(src)

    shoebox.add_microphone_array(pyroomacoustics.MicrophoneArray(mic_pos, fs))

    for i in range(n_times):

        shoebox.image_source_model()

        if i != 0 and i % 1000 == 0:
            print(i)


if __name__ == "__main__":

    test_issue_22()
