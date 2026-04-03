import numpy as np

import pyroomacoustics as pra

RT60_EPS = 0.15


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
        air_absorption=False,
    )

    # Create the 3D room by extruding the 2D by 10 meters
    height = 10.0
    room.extrude(height, materials=mat)

    room.set_ray_tracing(receiver_radius=0.5)

    # Add a source somewhere in the room
    source_box = np.array([2.5, 4.0, 10.0])
    source_pos = np.random.rand(3, 3) * source_box[:, None]
    source_pos += np.array([[5.0, 6.0, 0.0]]).T
    for p in source_pos.T:
        room.add_source(p)

    # Add a microphone
    mic_box = np.array([5.0, 10.0, 10.0])
    mic_pos = np.random.rand(3, 3) * mic_box[:, None]
    room.add_microphone_array(mic_pos)

    room.compute_rir()
    rt60_emp = np.median(room.measure_rt60(decay_db=30.0))
    rt60_thy = room.rt60_theory(formula="sabine")
    return rt60_emp, rt60_thy


def test_scattering_rt60():
    np.random.seed(0)

    rt60, rt60_sabine = get_rt60(scattering_coeff=0.5)

    assert abs(rt60 - rt60_sabine) < RT60_EPS


if __name__ == "__main__":
    # test_scattering_rt60()
    import matplotlib.pyplot as plt

    scat_coeff = np.linspace(0.0, 1.0, 11)

    rt60_measured = []
    for sc in scat_coeff:
        rt60, rt60_thy = get_rt60(scattering_coeff=sc)
        rt60_measured.append(rt60)

    fig, ax = plt.subplots()
    ax.plot(scat_coeff, rt60_measured, label="measured")
    ax.plot(scat_coeff, np.ones(11) * rt60_thy, label="Sabine")
    plt.show()
