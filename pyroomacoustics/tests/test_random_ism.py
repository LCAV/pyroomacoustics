
"""
Created on Tue Feb  8 17:50:45 2022

@author: od0014

Script to test removal of sweeping echoes with randomized image method
"""
import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt
from pyroomacoustics.transform import stft
from pyroomacoustics import metrics as met
# from scipy.io.wavfile import write


# create an example with sweeping echo - from Enzo's paper
room_size = [4, 4, 4]
source_loc = [1,2,2]
center_x = room_size[0]/2
center_y = room_size[1]/2
center_z = room_size[2]/2

# number of mics
M = 15
mic_loc = np.zeros((3,M), dtype = float)
gridSpacing = 0.25


# =============================================================================
# create uniform grid of mics
# =============================================================================
count = 0
Nmic_each_axis = int((M/3 - 1)/2)
for n in range(-Nmic_each_axis, Nmic_each_axis+1):
    mic_loc[:,count] = [center_x+n*gridSpacing, center_y, center_z]
    mic_loc[:,count+1] = [center_x, center_y+n*gridSpacing, center_z]
    mic_loc[:,count+2] = [center_x, center_y, center_z+n*gridSpacing]
    count += 3
              

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(mic_loc[0,:],mic_loc[1,:], mic_loc[2,:])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.xlim([0,4])
plt.ylim([0,4])
plt.show()           
                          
              
def test_ism():
    fs = 44100
    
    room = pra.ShoeBox(room_size, fs, materials=pra.Material(0.1), max_order = 50)
    
    room.add_source(source_loc)
    
    room.add_microphone_array(pra.MicrophoneArray(mic_loc, fs))

    room.compute_rir()
    
    #plot spectrograms to check for sweeping echoes
    
    fft_size = 512  # fft size for analysis
    fft_hop = 128  # hop between analysis frame
    fft_zp = 512  # zero padding
    analysis_window = pra.hann(fft_size)
    
    print("Sweeping echo measure for ISM is :")
    for n in range(M):
        
        if n == 0:
            S = stft.analysis(room.rir[n][0],  fft_size, fft_hop, win=analysis_window, zp_back=fft_zp)
            
            plt.figure()
            
            plt.imshow(
            pra.dB(S.T),
            extent=[0, 0.3*fs, 0, fs / 2],
            vmin=-100,
            vmax=0,
            origin="lower",
            cmap="jet"
            )
            ax.set_title("Mic location " + str(n))
            ax.set_ylabel("Frequency")
            ax.set_xlabel("Time")
            ax.set_aspect("auto")
            ax.axis("off")

            
            #plot RIR
            plt.figure()
            plt.plot(room.rir[n][0])
            plt.show()
            
            # write('../../bin/ism_sweeping_echoes.wav',fs,room.rir[n][0])
            
        test_sweep_measure(room.rir[n][0], fs)
 
        
    
def test_random_ism():
    
    fs = 44100
    
    room = pra.ShoeBox(room_size, fs, materials=pra.Material(0.1), max_order = 50, use_rand_ism=True)
    
    room.add_source(source_loc)
    
    room.add_microphone_array(pra.MicrophoneArray(mic_loc, fs))

    room.compute_rir()
    
    fft_size = 512  # fft size for analysis
    fft_hop = 128  # hop between analysis frame
    fft_zp = 512  # zero padding
    analysis_window = pra.hann(fft_size)
    
    print("Sweeping echo measure for randomized ISM is:")
    for n in range(M):
        
        if n == 0:
            S = stft.analysis(room.rir[n][0],  fft_size, fft_hop, win=analysis_window, zp_back=fft_zp)
            
            plt.figure()
            
            plt.imshow(
            pra.dB(S.T),
            extent=[0, 0.3*fs, 0, fs / 2],
            vmin=-100,
            vmax=0,
            origin="lower",
            cmap="jet"
            )
            ax.set_title("Mic location " + str(n))
            ax.set_ylabel("Frequency")
            ax.set_xlabel("Time")
            ax.set_aspect("auto")
            ax.axis("off")
         
            
            #plot RIR
            plt.figure()
            plt.plot(room.rir[n][0])
            plt.show()
            
            # write('../../bin/rand_ism_sweeping_echoes.wav',fs,room.rir[n][0])

            
        
        test_sweep_measure(room.rir[n][0], fs)
    
    
    
    
def test_sweep_measure(rir, fs):
    # measure of spectral flatness of sweeping echos
    # higher value is desired
    ssf = met.sweeping_echo_measure(rir,fs)
    print(ssf)    
        
    
    
    
if __name__ == "__main__":
    test_ism()
    test_random_ism()



