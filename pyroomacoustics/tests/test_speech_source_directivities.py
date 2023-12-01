from unittest import TestCase

import numpy as np

import pyroomacoustics as pra
from pyroomacoustics.directivities import (
    DirectionVector,
    DirectivityPattern,
    SpeechDirectivity,
)


def predict_rir_speech_source(speech_direction):
    """
    Predict RIR when speech is facing in a particular direction.

    Parameters
    ----------
    speech_direction: float
        the direction that the speech source is facing

    Returns
    -------
    :np.ndarray ``(float, [float, float])``
        This function returns ``(m, [le, ue])`` and the confidence interval is ``[m-le, m+ue]``.
    """

    scat_test = {
        "coeffs": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        "center_freqs": [125, 250, 500, 1000, 2000, 4000, 8000],
    }
    abs_test = {
        "coeffs": [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
        "center_freqs": [125, 250, 500, 1000, 2000, 4000, 8000],
    }
    material = pra.Material(abs_test, scat_test)

    # create room
    room = pra.ShoeBox(
        p=[7, 5, 5],
        materials=material,
        fs=16000,
        max_order=1,
    )

    # define source with speech directivity
    ORIENTATION = DirectionVector(azimuth=speech_direction, colatitude=90, degrees=True)
    directivity = SpeechDirectivity(orientation=ORIENTATION)

    # add source with speech directivity
    room.add_source([2.5, 2.5, 2.5], directivity=directivity)

    # add microphone on axis with speech source
    room.add_microphone([5, 2.5, 2.5])

    # compute rir
    room.compute_rir()

    return room.rir[0][0]


class TestSourceDirectivity(TestCase):
    def test_speech_direction(self):
        energy_0_deg = np.mean(predict_rir_speech_source(speech_direction=0) ** 2)
        energy_180_deg = np.mean(predict_rir_speech_source(speech_direction=180) ** 2)

        self.assertTrue(energy_0_deg > energy_180_deg)


if __name__ == "__main__":
    energy_0_deg = np.mean(predict_rir_speech_source(azimuth=0) ** 2)
    energy_180_deg = np.mean(predict_rir_speech_source(azimuth=180) ** 2)
    print()
    print("-" * 40)
    print(
        f"mean energy when speech facing towards mic {10 *np.log10(energy_0_deg):.0f} dB"
    )
    print(
        f"mean energy when speech facing away from mic {10 *np.log10(energy_180_deg):.0f} dB"
    )
    print("-" * 40)
