import numpy as np
import pyroomacoustics as pra


def test_issue_115_ism_breaking():
    """
    When a source was too close to the microphone, the time-of-flight
    might be smaller than the delay due to the fractionnal delay filter
    used to create the impulse response.
    It is then necessary to add this delay to the rir filter to ensure
    no runtime error.
    """
    print("Test with source close to microphone.")
    shoebox = pra.ShoeBox(
        [9.29447785567344, 6.529510207957697, 4.4677460263160995],
        materials=pra.Material.make_freq_flat(absorption=0.1675976883006225),
        fs=16000,
        max_order=17
    )
    source_loc = [5.167674641605016, 4.379726875714017, 2.9190423403507504]
    shoebox.add_source(source_loc)
    noise_loc = [8.47420884677372, 5.675261722911696, 1.2040578622058364]
    shoebox.add_source(noise_loc)
    R = np.array([[8.571318246865648],
                  [5.799718630723678],
                  [1.3702254938278977]])
    print('mic - source distance : {} m'.format(
        np.sqrt(sum((np.array(source_loc) - np.squeeze(R))**2)))
    )
    print('mic - noise distance : {} m'.format(
        np.sqrt(sum((np.array(noise_loc) - np.squeeze(R))**2)))
    )
    shoebox.add_microphone_array(
        pra.MicrophoneArray(R, shoebox.fs))
    shoebox.compute_rir()


def test_issue_115_rt_breaking():
    """
    As background, only ray tracing only starts to be active for rays that run
    longer than the maximum ISM order.
    The problem happen when the ISM order is very high (here 17),
    then, in some circumstances, it is possible that no ray travels longer
    than that. Then the histogram is empty and an error happen.
    """
    print("Test with high order ISM")
    shoebox = pra.ShoeBox(
        [4.232053263716528, 3.9244954007318853, 5.563810437305445],
        materials=pra.Material.make_freq_flat(absorption=0.6965517438548237),
        fs=16000,
        max_order=17,
        ray_tracing=True,
    )
    source_loc = [1.2028020579854695, 2.2980760894630676, 2.0654520390433984]
    shoebox.add_source(source_loc)
    R = np.array([[1.8062807887952617], [2.7793113278109454], [1.42966428606882]])
    print(
        "mic - source distance : {} m".format(
            np.sqrt(sum((np.array(source_loc) - np.squeeze(R)) ** 2))
        )
    )
    shoebox.add_microphone_array(pra.MicrophoneArray(R, shoebox.fs))
    shoebox.compute_rir()


if __name__ == "__main__":
    test_issue_115_rt_breaking()
    test_issue_115_ism_breaking()
