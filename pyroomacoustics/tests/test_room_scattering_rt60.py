import numpy as np

import pyroomacoustics as pra

RT60_EPS = 0.02


def get_rt60(scattering_coeff):
    # Create the 2D L-shaped room from the floor polygon
    pol = np.array([[0, 0], [0, 10], [7.5, 10], [7.5, 6], [5, 6], [5, 0]]).T
    mat = pra.Material(0.15, scattering_coeff)
    room = pra.Room.from_corners(
        pol,
        fs=16000,
        materials=mat,
        max_order=3,
        ray_tracing=True,
        air_absorption=True,
    )

    # Create the 3D room by extruding the 2D by 10 meters
    height = 10.0
    room.extrude(height, materials=mat)

    room.set_ray_tracing(receiver_radius=0.5)

    # Add a source somewhere in the room
    room.add_source([1.5, 1.7, 1.6])

    # Add a microphone
    room.add_microphone([3.0, 2.25, 0.6])

    room.compute_rir()
    return room.measure_rt60()[0, 0]


def test_scattering_rt60():
    np.random.seed(0)

    rt60_scat_0p0 = get_rt60(scattering_coeff=0.0)
    rt60_scat_0p1 = get_rt60(scattering_coeff=0.1)
    rt60_scat_0p2 = get_rt60(scattering_coeff=0.2)

    assert abs(rt60_scat_0p1 - rt60_scat_0p0) < RT60_EPS
    assert abs(rt60_scat_0p2 - rt60_scat_0p0) < RT60_EPS


if __name__ == "__main__":
    test_scattering_rt60()
