
"""
Created on Tue Feb  8 17:50:45 2022

@author: od0014

Script to test removal of sweeping echoes with randomized image method
"""
import pyroomacoustics as pra
from pyroomacoustics import metrics as met


# create an example with sweeping echo - from Enzo's paper
room_size = [4, 4, 4]
source_loc = [1,2,2]
mic_loc = [0.5,1,0.75]
fs = 44100
  
                          
              
def test_ism():
    
    room = pra.ShoeBox(room_size, fs, materials=pra.Material(0.1), max_order = 50)
    
    room.add_source(source_loc)
    
    room.add_microphone(mic_loc)

    room.compute_rir()
                
    ssf_ism = test_sweep_measure(room.rir[0][0], fs)
    
    assert 0 <= ssf_ism <= 1.0
 
    return ssf_ism
    
def test_random_ism():
    
    room = pra.ShoeBox(room_size, fs, materials=pra.Material(0.1), max_order = 50, use_rand_ism=True)
    
    room.add_source(source_loc)
    
    room.add_microphone(mic_loc)

    room.compute_rir()
    
    ssf_rism = test_sweep_measure(room.rir[0][0], fs)
    
    assert 0 <= ssf_rism <= 1.0
    
    return ssf_rism
    
    
    
    
def test_sweep_measure(rir, fs):
    
    # measure of spectral flatness of sweeping echos
    # higher value is desired
    ssf = met.sweeping_echo_measure(rir,fs)
    return ssf
        
    
    
    
if __name__ == "__main__":
    ssf_ism =test_ism()
    ssf_rism = test_random_ism()
    assert ssf_rism > ssf_ism



