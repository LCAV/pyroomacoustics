
import ctypes as ct

c_float_p = ct.POINTER(ct.c_float)
c_int_p = ct.POINTER(ct.c_int)

class CWALL(ct.Structure):
    _fields_ = [
            ('dim', ct.c_int),
            ('absorption', ct.c_float),
            ('normal', ct.c_float * 3),
            ('n_corners', ct.c_int),
            ('corners', c_float_p),
            ('origin', ct.c_float * 3),
            ('basis', ct.c_float * 6),
            ('flat_corners', c_float_p),
            ]

c_wall_p = ct.POINTER(CWALL)

class CROOM(ct.Structure):
    _fields_ = [
            ('dim', ct.c_int),
            ('n_walls', ct.c_int),
            ('walls', c_wall_p),
            ('n_sources', ct.c_int),
            ('sources', c_float_p),
            ('parents', c_int_p),
            ('gen_walls', c_int_p),
            ('orders', c_int_p),
            ('attenuations', c_float_p),
            ('n_obstructing_walls', ct.c_int),
            ('obstructing_walls', c_int_p),
            ('n_microphones', ct.c_int),
            ('microphones', c_float_p),
            ('is_visible', c_int_p),
            ]

c_room_p = ct.POINTER(CROOM)
