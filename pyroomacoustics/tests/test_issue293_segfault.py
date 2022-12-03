import pyroomacoustics as pra


def test_issu293_segfault():
    for i in range(30):
        room_dim = [30, 30]
        source = [2, 3]
        mic_array = [[8], [8]]

        room = pra.ShoeBox(
            room_dim,
            ray_tracing=True,
            materials=pra.Material(energy_absorption=0.1, scattering=0.2),
            air_absorption=False,
            max_order=0,
        )
        room.add_microphone_array(mic_array)
        room.add_source(source)
        room.set_ray_tracing(n_rays=10_000)
        room.compute_rir()
        print(i)
