import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from ..acoustics import OctaveBandsFactory
from ..parameters import Material
from ..room import Room, wall_factory


def parse_vertex(line, line_num):
    assert line[0] == v

    line = line.strip().split(" ")

    if len(line) < 4 or len(line > 5):
        raise ValueError("Malformed vertex on line {line_num}")

    return np.array([float(line[i]) for i in range(3)])


def read_obj(filename):
    with open(filename, "r") as f:
        content = f.readlines()

    # keep track of the faces to process later
    vertices = []
    unprocessed_faces = []

    for no, line in enumerate(content):
        if line[0] == "v":
            vertices.append(parse_vertex(line, no))
        elif line[0] == "f":
            unprocessed_faces.append([no, line])

    for no, line in faces:
        pass


def read_room_json(filename, fs):

    with open(filename, "r") as f:
        content = json.load(f)

    vertices = np.array(content["vertices"])

    faces = []
    materials = []
    names = []
    walls_args = []

    for name, wall_info in content["walls"].items():
        vertex_ids = np.array(wall_info["vertices"]) - 1
        wall_vertices = vertices[vertex_ids].T

        try:
            mat_info = wall_info["material"]
            if isinstance(mat_info, dict):
                mat = Material(**mat_info)
            elif isinstance(mat_info, list):
                mat = Material(*mat_info)
            else:
                mat = Material(mat_info)
        except KeyError:
            mat = Material(energy_absorption=0.0)

        walls_args.append([wall_vertices, mat, name])

    octave_bands = OctaveBandsFactory(fs=fs)
    materials = [a[1] for a in walls_args]
    if not Material.all_flat(materials):
        for mat in materials:
            mat.resample(octave_bands)

    walls = [
        wall_factory(w, m.absorption_coeffs, m.scattering_coeffs, name=n)
        for w, m, n in walls_args
    ]

    return walls


def read_source(source, scene_parent_dir, fs_room):

    if isinstance(source, list):
        return {"position": source, "signal": np.zeros(1)}

    elif isinstance(source, dict):
        kwargs = {"position": source["loc"]}

        if "signal" in source:
            fs_audio, audio = wavfile.read(scene_parent_dir / source["signal"])

            # convert to float if necessary
            if audio.dtype == np.int16:
                audio = audio / 2 ** 15

            if audio.ndim == 2:
                import warnings

                warnings.warn(
                    "The audio file was multichannel. Only keeping channel 1."
                )
                audio = audio[:, 0]

            if fs_audio != fs_room:
                try:
                    import samplerate

                    fs_ratio = fs_room / fs_audio
                    audio = samplerate.resample(audio, fs_ratio, "sinc_best")
                except ImportError:
                    raise ImportError(
                        "The samplerate package must be installed for"
                        " resampling of the signals."
                    )

            kwargs["signal"] = audio

        else:
            # add a zero signal is the source is not active
            kwargs["signal"] = np.zeros(1)

        if "delay" in source:
            kwargs["delay"] = source["delay"]

        return kwargs

    else:
        raise TypeError("Unexpected type.")


def read_scene(filename):

    filename = Path(filename)
    parent_dir = filename.parent

    with open(filename, "r") as f:
        scene_info = json.load(f)

    # the sampling rate
    try:
        fs = scene_info["samplerate"]
    except KeyError:
        fs = 16000

    # read the room file
    room_filename = parent_dir / scene_info["room"]
    walls = read_room_json(room_filename, fs)

    if "room_kwargs" not in scene_info:
        scene_info["room_kwargs"] = {}

    room = Room(walls, fs=fs, **scene_info["room_kwargs"])
    for source in scene_info["sources"]:
        room.add_source(**read_source(source, parent_dir, fs))
    for mic in scene_info["microphones"]:
        room.add_microphone(mic)

    if "ray_tracing" in scene_info:
        room.set_ray_tracing(**scene_info["ray_tracing"])

    return room
