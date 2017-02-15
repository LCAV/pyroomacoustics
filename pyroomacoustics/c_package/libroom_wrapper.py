
import ctypes as ct

c_float_p = ct.POINTER(ct.c_float)

class CWALL(ct.Structure):
    _fields_ = [
            ('dim', ct.c_int),
            ('absorption', ct.c_float),
            ('normal', ct.c_float * 3),
            ('n_corners', ct.c_int),
            ('corners', c_float_p),
            ('origin', ct.c_float * 3),
            ('local_basis', ct.c_float * 6),
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
            ('generators', ct.c_int_p),
            ('gen_walls', ct.c_int_p),
            ('orders', ct.c_int_p),
            ('obstructing_walls', ct.c_int_p),
            ('n_obstructing_walls', ct.c_int),
            ('microphones', c_float_p),
            ('n_microphones', ct.c_int),
            ]

c_room_p = ct.POINTER(CROOM)
