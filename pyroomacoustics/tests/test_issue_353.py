"""
# Test for Issue 353

[issue #353](https://github.com/LCAV/pyroomacoustics/issues/353)

The cause of the issue was that the time of the maximum delay
in the RIR is used to determine the necessary size of the array.

The float64 value of the delay was used to determine the size and construct the array.
However, the `rir_build` routine takes float32.
After conversion, the array size would evaluate to one delay more due to rounding
offset and an array size check would fail.

Converting the delay time array to float32 before creating the rir array
solved the issue.
"""

import numpy as np
import pytest

import pyroomacoustics as pra


def test_issue_353():
    room_dims = np.array([10.0, 10.0, 10.0])
    room = pra.ShoeBox(
        room_dims, fs=24000, materials=None, max_order=22, use_rand_ism=False
    )

    source = np.array([[6.35551912], [4.33308523], [3.69586303]])
    room.add_source(source)

    mic_array_in_room = np.array(
        [
            [1.5205189, 1.49366285, 1.73302404, 1.67847898],
            [4.68430529, 4.76250254, 4.67956424, 4.60702604],
            [2.68214263, 2.7980202, 2.55341851, 2.72701718],
        ]
    )
    room.add_microphone_array(mic_array_in_room)

    room.compute_rir()


def test_issue_353_2():
    rt60_tgt = 0.451734045124395  # seconds
    room_dim = [2.496315595944846, 2.2147285947364708, 3.749472153652182]  # meters

    fs = 16000

    e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)

    room = pra.ShoeBox(
        room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=max_order
    )

    room.add_source([0.24784311631630576, 1.690743273038349, 1.9570721698068267])

    mic_array = np.array(
        [
            [
                0.46378325918698565,
            ],
            [
                1.5657207092343373,
            ],
            [
                3.015697444447528,
            ],
        ]
    )

    room.add_microphone_array(mic_array)

    room.compute_rir()


params = (
    (
        [4.57977238, 5.39054892, 2.82767573],
        0.18257819716723472,
        [1.22812426, 1.2966769, 1.43330033],
        [1.8020194, 0.76576269, 0.53980759],
        83,
    ),
    (
        [5.3997869, 6.34821279, 2.90299906],
        0.2217407971025793,
        [4.05889913, 4.15230608, 2.39073375],
        [2.45186073, 2.88844052, 1.39751034],
        70,
    ),
    (
        [5.45909408, 6.34962532, 2.77107005],
        0.27431416842419915,
        [0.54511116, 2.82639397, 1.04676184],
        [4.15744634, 2.82665472, 1.01958203],
        58,
    ),
    (
        [5.88430842, 5.74587181, 2.81243457],
        0.23546727398942446,
        [5.2673113, 1.68109104, 2.13159967],
        [2.03474247, 0.82147634, 1.25415523],
        66,
    ),
    (
        [5.8335965, 4.90706049, 2.5410871],
        0.25263067293986236,
        [0.58218881, 3.25631355, 0.91775666],
        [1.06434647, 3.3755251, 1.84040589],
        63,
    ),
    (
        [5.63150056, 5.21368813, 2.90971373],
        0.24979151070744487,
        [4.30157587, 2.54104283, 2.22109155],
        [1.47065101, 3.65191472, 1.64230692],
        61,
    ),
    (
        [6.24132556, 4.62941236, 2.52808349],
        0.23735500015498867,
        [3.75099353, 3.82859854, 1.66480812],
        [0.63880713, 1.93500295, 1.12386568],
        67,
    ),
)


@pytest.mark.parametrize("room_dims, abs_coeff, spkr_pos, mic_pos, max_order", params)
def test_issue_353_3(room_dims, abs_coeff, spkr_pos, mic_pos, max_order):
    room = pra.ShoeBox(
        room_dims,
        fs=16000,
        materials=pra.Material(abs_coeff),
        max_order=max_order,
        use_rand_ism=False,
    )
    room.add_source(spkr_pos)
    room.add_microphone(mic_pos)

    room.compute_rir()
