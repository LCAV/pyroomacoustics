"""
Just run the Material command with a bunch of inputs to make sure
it works as expected
"""
import pyroomacoustics as pra


scat_test = {
    "coeffs": [0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3],
    "center_freqs": [125, 250, 500, 1000, 2000, 4000, 8000],
}
abs_test = {
    "coeffs": [0.3, 0.4, 0.25, 0.11, 0.05, 0.03, 0.3],
    "center_freqs": [125, 250, 500, 1000, 2000, 4000, 8000],
}


def test_material_e_float():
    mat = pra.Material(0.3)


def test_material_es_float():
    mat = pra.Material(0.3, 0.1)


def test_material_e_db():
    mat = pra.Material("hard_surface")


def test_material_es_db():
    mat = pra.Material("hard_surface", "rpg_qrd")


def test_material_e_dict():
    mat = pra.Material(abs_test)


def test_material_es_dict():
    mat = pra.Material(abs_test, scat_test)


def test_material_e_float_s_db():
    mat = pra.Material(0.3, "rpg_skyline")


def test_material_e_db_s_float():
    mat = pra.Material("brickwork", 0.1)


def test_material_e_float_s_dict():
    mat = pra.Material(0.1, scat_test)


def test_material_e_dict_s_float():
    mat = pra.Material(abs_test, 0.1)


def test_material_e_db_s_dict():
    mat = pra.Material("brickwork", scat_test)


def test_material_e_dict_s_db():
    mat = pra.Material(abs_test, "classroom_tables")


def test_dict_pairs():
    materials = pra.make_materials(
        ceiling=(0.25, 0.01),
        floor=(0.5, 0.1),
        east=(0.15, 0.15),
        west=(0.07, 0.15),
        north=(0.15, 0.15),
        south=(0.10, 0.15),
    )

    assert isinstance(materials, dict)


def test_list_pairs():
    materials = pra.make_materials(
        (0.25, 0.01),
        (0.5, 0.1),
        (0.15, 0.15),
        (0.07, 0.15),
        (0.15, 0.15),
        (0.10, 0.15),
    )

    assert isinstance(materials, list)


def test_dict_list_mix():
    mat_list, mat_dict = pra.make_materials(
        (0.25, 0.01),
        abs_test,
        "brickwork",
        ("brickwork", scat_test),
        1.,
        ceilling=(abs_test, scat_test),
        floor=(abs_test, "rpg_skyline"),
        one=(0.10, 0.15),
    )

    assert isinstance(mat_list, list)
    assert isinstance(mat_dict, dict)


def test_empty():
    mat_list = pra.make_materials()

    assert mat_list == []


if __name__ == '__main__':

    test_material_e_float()
    test_material_es_float()
    test_material_es_dict
    test_material_e_db()
    test_material_es_db()
    test_material_e_dict()
    test_material_es_dict()
    test_material_e_float_s_db()
    test_material_e_db_s_float()
    test_material_e_float_s_dict()
    test_material_e_dict_s_float()
    test_material_e_db_s_dict()
    test_material_e_dict_s_db()
    test_dict_pairs()
    test_list_pairs()
    test_dict_list_mix()
    test_empty()
