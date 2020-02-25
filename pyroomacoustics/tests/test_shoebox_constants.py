import pyroomacoustics as pra
from pyroomacoustics.parameters import _calculate_temperature


def test_set_temperature_wo_humidity():
    room = pra.ShoeBox(p=[3, 3, 3], temperature=21)
    assert room.physics.H == 0.0


def test_set_humidity_wo_temperature():
    room = pra.ShoeBox(p=[3, 3, 3], humidity=21)
    assert room.physics.T == _calculate_temperature(
        room.physics.get_sound_speed(), room.physics.H
    )


if __name__ == '__main__':
    test_set_temperature_wo_humidity()
    test_set_humidity_wo_temperature()
