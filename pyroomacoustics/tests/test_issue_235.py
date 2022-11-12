import pyroomacoustics as pra


def test_set_rt_no_directivity():

    room = pra.ShoeBox([5, 4, 3])
    room.set_ray_tracing()
