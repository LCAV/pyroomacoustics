import matplotlib

import pyroomacoustics as pra
import numpy as np
import matplotlib.pyplot as plt


from scipy.io import wavfile
from scipy import signal
from scipy.fft import fftfreq, fft
from pyroomacoustics.directivities import (
    DirectivityPattern,
    DirectionVector,
    CardioidFamily,
    DIRPATRir,
)
from timeit import default_timer as timer
from scipy.signal import fftconvolve
import os

# Path on my system.
# /home/psrivast/Téléchargements/AKG_c480_c414_CUBE.sofa
# /home/psrivast/Téléchargements/LSPs_HATS_GuitarCabinets_Akustikmessplatz.sofa

"""
##########################################################
With DIRPATRir object we can generate RIRs with mics and source having either
frequency independent CARDIOID patterns or
freqeuncy dependent patterns from DIRPAT dataset.

Parameters
--------------------------------------
    orientation :
        class DirectionVector
    path : (string)
        Path towards the DIRPAT sofa file, the ending name of the file should be the same as specified in the DIRPAT dataset

    DIRPAT_pattern_enum : (string)
        Only used to choose the directivity patterns available in the specific files in the DIRPAT dataset

    # AKG_c480_c414_CUBE.sofa DIRPAT file include mic patterns for CARDIOID ,FIGURE_EIGHT,HYPERCARDIOID ,OMNI,SUBCARDIOID
    a)AKG_c480
    b)AKG_c414K
    c)AKG_c414N
    d)AKG_c414S
    e)AKG_c414A

    # LSPs_HATS_GuitarCabinets_Akustikmessplatz.sofa DIRPAT file include source patterns
    a)Genelec_8020
    b)Lambda_labs_CX-1A
    c)HATS_4128C
    d)Tannoy_System_1200
    e)Neumann_KH120A
    f)Yamaha_DXR8
    g)BM_1x12inch_driver_closed_cabinet
    h)BM_1x12inch_driver_open_cabinet
    i)BM_open_stacked_on_closed_withCrossoverNetwork
    j)BM_open_stacked_on_closed_fullrange
    k)Palmer_1x12inch
    l)Vibrolux_2x10inch

    fs : (int)
        Sampling frequency of the filters for interpolation.
        Should be same as the simulator frequency and less than 44100 kHz
    no_points_on_fibo_sphere : (int)
        Number of points on the interpolated Fibonacci sphere.
        if "0" no interpolation will happen.


############################################################

"""
path_DIRPAT_file=os.path.join(os.path.dirname(__file__).replace("examples",""),"pyroomacoustics","data","AKG_c480_c414_CUBE.sofa")

dir_obj_Dmic = DIRPATRir(
    orientation=DirectionVector(azimuth=54, colatitude=73, degrees=True),
    path=path_DIRPAT_file,
    DIRPAT_pattern_enum="AKG_c414K",
    fs=16000,
)

# pattern_enum=DirectivityPattern.HYPERCARDIOID,

'''
dir_obj_Dsrc = DIRPATRir(
    orientation=DirectionVector(azimuth=0, colatitude=0, degrees=True),
    path="/home/psrivast/Téléchargements/LSPs_HATS_GuitarCabinets_Akustikmessplatz.sofa",
    DIRPAT_pattern_enum="Genelec_8020",
    fs=16000,
)
'''

dir_obj_Cmic = CardioidFamily(
    orientation=DirectionVector(azimuth=90, colatitude=123, degrees=True),
    pattern_enum=DirectivityPattern.FIGURE_EIGHT,
)


dir_obj_Csrc = CardioidFamily(
    orientation=DirectionVector(azimuth=56, colatitude=123, degrees=True),
    pattern_enum=DirectivityPattern.CARDIOID,
)


start = timer()
room_dim = [6, 6, 2.4]


all_materials = {
    "east": pra.Material(
        energy_absorption={
            "coeffs": [0.02, 0.02, 0.03, 0.03, 0.04, 0.05],
            "center_freqs": [125, 250, 500, 1000, 2000, 4000],
        },
        scattering=0.54,
    ),
    "west": pra.Material(
        energy_absorption={
            "coeffs": [0.02, 0.02, 0.03, 0.03, 0.04, 0.05],
            "center_freqs": [125, 250, 500, 1000, 2000, 4000],
        },
        scattering=0.54,
    ),
    "north": pra.Material(
        energy_absorption={
            "coeffs": [0.02, 0.02, 0.03, 0.03, 0.04, 0.05],
            "center_freqs": [125, 250, 500, 1000, 2000, 4000],
        },
        scattering=0.54,
    ),
    "south": pra.Material(
        energy_absorption={
            "coeffs": [0.02, 0.02, 0.03, 0.03, 0.04, 0.05],
            "center_freqs": [125, 250, 500, 1000, 2000, 4000],
        },
        scattering=0.54,
    ),
    "ceiling": pra.Material(
        energy_absorption={
            "coeffs": [0.02, 0.02, 0.03, 0.03, 0.04, 0.05],
            "center_freqs": [125, 250, 500, 1000, 2000, 4000],
        },
        scattering=0.54,
    ),
    "floor": pra.Material(
        energy_absorption={
            "coeffs": [0.11, 0.14, 0.37, 0.43, 0.27, 0.25],
            "center_freqs": [125, 250, 500, 1000, 2000, 4000],
        },
        scattering=0.54,
    ),
}

# Length of the RIR Is 600 ms

room = pra.ShoeBox(
    room_dim,
    fs=16000,
    max_order=2,
    materials=pra.Material(0.99),
    air_absorption=True,
    ray_tracing=False,
    min_phase=False,
)


room.add_source(
    [1.52, 0.883, 1.044], directivity=dir_obj_Csrc
)  # 3.65,1.004,1.38 #0.02,2.004,2.38

"""
mic_locs = np.c_[
    [2.31, 1.65, 1.163],
    [3.42, 2.48, 0.91],  # mic 1  # mic 2  #[3.47, 2.57, 1.31], [3.42, 2.48, 0.91]
]
"""

room.add_microphone([2.31, 1.65, 1.163], directivity=dir_obj_Dmic)
# room.add_microphone_array(mic_locs)#,directivity=dir_obj_1)

dir_obj_Dmic.set_orientation(54, 73)
#dir_obj_Dsrc.set_orientation(173, 60)

room.compute_rir()


end = timer()
print("Time taken", end - start)

rir_1_0 = room.rir[0][0]



"""
Create a cuboid with center and the given length of x,y,z

Parameters
--------------
    center : np.ndarray
    size : np.ndarray


"""
"""
def cuboid_data(center, size):

    # suppose axis direction: x: to left; y: to inside; z: to upper
    # get the (left, outside, bottom) point

    o = [a - b / 2 for a, b in zip(center, size)]
    # get the length, width, and height
    l, w, h = size
    x = [
        [
            o[0],
            o[0] + l,
            o[0] + l,
            o[0],
            o[0],
        ],  # x coordinate of points in bottom surface
        [
            o[0],
            o[0] + l,
            o[0] + l,
            o[0],
            o[0],
        ],  # x coordinate of points in upper surface
        [
            o[0],
            o[0] + l,
            o[0] + l,
            o[0],
            o[0],
        ],  # x coordinate of points in outside surface
        [o[0], o[0] + l, o[0] + l, o[0], o[0]],
    ]  # x coordinate of points in inside surface
    y = [
        [
            o[1],
            o[1],
            o[1] + w,
            o[1] + w,
            o[1],
        ],  # y coordinate of points in bottom surface
        [
            o[1],
            o[1],
            o[1] + w,
            o[1] + w,
            o[1],
        ],  # y coordinate of points in upper surface
        [o[1], o[1], o[1], o[1], o[1]],  # y coordinate of points in outside surface
        [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w],
    ]  # y coordinate of points in inside surface
    z = [
        [o[2], o[2], o[2], o[2], o[2]],  # z coordinate of points in bottom surface
        [
            o[2] + h,
            o[2] + h,
            o[2] + h,
            o[2] + h,
            o[2] + h,
        ],  # z coordinate of points in upper surface
        [
            o[2],
            o[2],
            o[2] + h,
            o[2] + h,
            o[2],
        ],  # z coordinate of points in outside surface
        [o[2], o[2], o[2] + h, o[2] + h, o[2]],
    ]  # z coordinate of points in inside surface
    return x, y, z

"""
# az=90,col=10

"""
####################################################
# 3D acoustic scene plotting code                  #
# with directivity pattern for source and receiver #
####################################################
Require position of source and receiver.
Frequency domain filters from interpolated fibo sphere
"""

"""
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

plt.style.use("ggplot")
mic_p_x = room.mic_array.directivity[0].obj_open_sofa_inter.rotated_fibo_points[
    0, :
] / 2 + np.array([[2.31]])
mic_p_y = room.mic_array.directivity[0].obj_open_sofa_inter.rotated_fibo_points[
    1, :
] / 2 + np.array([[1.65]])
mic_p_z = room.mic_array.directivity[0].obj_open_sofa_inter.rotated_fibo_points[
    2, :
] / 4 + np.array([[1.163]])

mic_p_x1 = room.mic_array.directivity[1].obj_open_sofa_inter.rotated_fibo_points[
    0, :
] / 2 + np.array([[3.42]])
mic_p_y1 = room.mic_array.directivity[1].obj_open_sofa_inter.rotated_fibo_points[
    1, :
] / 2 + np.array([[2.48]])
mic_p_z1 = room.mic_array.directivity[1].obj_open_sofa_inter.rotated_fibo_points[
    2, :
] / 4 + np.array([[0.91]])


src_p_x = dir_obj_sr.obj_open_sofa_inter.rotated_fibo_points[0, :] / 2 + np.array(
    [[1.52]]
)
src_p_y = dir_obj_sr.obj_open_sofa_inter.rotated_fibo_points[1, :] / 2 + np.array(
    [[0.883]]
)
src_p_z = dir_obj_sr.obj_open_sofa_inter.rotated_fibo_points[2, :] / 4 + np.array(
    [[1.044]]
)


# src_p=dir_obj_sr.obj_open_sofa_inter.rotated_fibo_points+(np.array([[3.65],[1.004],[1.38]]))

# print(np.sqrt(src_p[0,:]**2+src_p[1,:]**2+src_p[2,:]**2))
# print(src_p.shape)
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")
x = 6
y = 6
z = 2.4
X, Y, Z = cuboid_data([x / 2, y / 2, z / 2], (x, y, z))
ax.plot_surface(
    np.array(X),
    np.array(Y),
    np.array(Z),
    color="deepskyblue",
    rstride=1,
    cstride=1,
    alpha=0.05,
    linewidth=1,
)


# ax.plot3D([0,x,x,0,0,0,x,x,0,0,0,0,x,x,x,x],
#          [0,0,y,y,0,0,0,y,y,y,0,y,y,y,0,0],
#          [0,0,0,0,0,z,z,z,z,0,z,z,0,z,0,z])


# scamap = plt.cm.ScalarMappable(cmap='summer')
# fcolors_ = scamap.to_rgba(np.abs(fft(dir_obj_sr.obj_open_sofa_inter.sh_coeffs_expanded_target_grid,axis=-1)[:,300]),norm=False)
# fcolors_1 = scamap.to_rgba(np.abs(fft(dir_obj_mic.obj_open_sofa_inter.sh_coeffs_expanded_target_grid,axis=-1)[:,30]),norm=False)

ax.scatter(
    src_p_x,
    src_p_y,
    src_p_z,
    c=np.abs(
        fft(dir_obj_sr.obj_open_sofa_inter.sh_coeffs_expanded_target_grid, axis=-1)[
            :, 30
        ]
    ),
    cmap="inferno",
)  # sh_coeffs_expanded_target_grid[:,300],fft(dir_obj_sr.obj_open_sofa_inter.sh_coeffs_expanded_target_grid,axis=-1)
ax.scatter(
    mic_p_x,
    mic_p_y,
    mic_p_z,
    c=np.abs(
        fft(
            room.mic_array.directivity[
                0
            ].obj_open_sofa_inter.sh_coeffs_expanded_target_grid,
            axis=-1,
        )[:, 30]
    ),
    cmap="inferno",
)
# ax.scatter(src_p_x,src_p_y,src_p_z,c=np.abs(dir_obj_sr.obj_open_sofa_inter.freq_angles_fft[:,30]), cmap='inferno')

# ax.scatter(mic_p_x,mic_p_y,mic_p_z,c=np.abs(dir_obj_mic.obj_open_sofa_inter.freq_angles_fft[:,30]), cmap='inferno')
ax.scatter(
    mic_p_x1,
    mic_p_y1,
    mic_p_z1,
    c=np.abs(
        fft(
            room.mic_array.directivity[
                1
            ].obj_open_sofa_inter.sh_coeffs_expanded_target_grid,
            axis=-1,
        )[:, 30]
    ),
    cmap="inferno",
)


ax.text(
    2.31 + 0.5,
    1.65 + 0.5,
    1.163 + 0.5,
    "%s" % ("(Mic az=0,col=0 , pos = [2.31,1.65,1.163])"),
    size=8,
    zorder=1,
    color="k",
)
ax.text(
    1.52 + 0.5,
    0.883 + 0.5,
    1.044 + 0.5,
    "%s" % ("(Src az=45,col=0 , pos = [1.52,0.883,1.044])"),
    size=8,
    zorder=1,
    color="k",
)
ax.set_xlabel("Length")
ax.set_xlim(0, 6)
ax.set_ylabel("Width")
ax.set_ylim(0, 6)
ax.set_zlabel("Height")
ax.set_zlim(0, 2.4)
plt.legend()
plt.show()
# fft(dir_obj_sr.obj_open_sofa_inter.sh_coeffs_expanded_target_grid,axis=-1)
"""

"""
####################################################
# Compare Realism , The code below                 #
# plots dEchorate RIR and pyroom DIRPAT RIR        #
# in time domain and frequency domain              #
# it also plots spectogram, DRR, PSD and  grp del  #
####################################################
Requires dirpat pyroom RIR and path to dEchorate RIR
from one of it's rooms.

"""

"""
plt.style.use('ggplot')

fig = plt.figure(figsize=(12, 20), constrained_layout=True)

spec = fig.add_gridspec(3, 3) #25,35
#rir_3_0=np.load("debug_rir_dir_3.npy")

rir_1_0=rir_1_0

rir_2_0=np.load("room011111_2_14.npy").reshape([240000])

#print("RT60",t60_impulse(rir_1_0,16000,np.array([125 * pow(2,a) for a in range(6)])))


rir_2_0=decimate(rir_2_0,int(round(48000/16000)))[1350:7960]

#Calculate DRR

first_part_d=np.sum(rir_2_0[:230]**2)
first_part_s=np.sum(rir_1_0[:240]**2)

second_part_d=np.sum(rir_2_0[230:]**2)
second_part_s=np.sum(rir_1_0[240:]**2)

print("DRR dEchorate",10*np.log10(first_part_d/second_part_d))
print("DRR Simulated",10*np.log10(first_part_s/second_part_s))














AB_=10 * np.log10(np.abs(fft(rir_2_0)[:(rir_2_0.shape[0] // 2)]) ** 2)
BA_=10 * np.log10(np.abs(fft(rir_1_0)[:(rir_1_0.shape[0] // 2)]) ** 2)
#CA_=10 * np.log10(np.abs(fft(rir_3_0)[:(rir_3_0.shape[0] // 2)]) ** 2)
"""

# Plots the impulse responses in the time domain and also compares them in the frequency domain
"""
ax0 = fig.add_subplot(spec[2, :])
ax0.plot(fftfreq(rir_2_0.shape[0],d=1/16000)[:(rir_2_0.shape[0]//2)],AB_,label="dEchorate Room 011111")
ax0.plot(fftfreq(rir_1_0.shape[0],d=1/16000)[:(rir_1_0.shape[0]//2)],BA_,label="Simulated dEchorate room 011111 with pyroom DIRPAT")
#axs[4].plot(fftfreq(rir_3_0.shape[0],d=1/16000)[:(rir_3_0.shape[0]//2)],CA_,label="Vanilla pyroomacoustics simulated dEchorate Room 011111")

ax1 = fig.add_subplot(spec[1, :])
ax1.plot(np.arange(rir_1_0.shape[0]),rir_1_0,label="Simulated dEchorate room 011111 with pyroom DIRPAT ")
ax1.legend()

ax2 = fig.add_subplot(spec[0, :])
ax2.plot(np.arange(rir_2_0.shape[0]),rir_2_0,label="dEchorate Room 011111",color="steelblue")
ax2.legend()

#axs[2].plot(np.arange(rir_3_0.shape[0]),rir_3_0,label="Vanilla pyroomacoustics simulated dEchorate Room 011111")
"""


"""
rir_1_0=rir_1_0[140:6000]
rir_2_0=rir_2_0[140:6000]
f,t,Sxx=signal.spectrogram(rir_1_0,16000,nperseg=32,noverlap=32//1.3)
f_1,t_1,Sxx_1=signal.spectrogram(rir_2_0,16000,nperseg=32,noverlap=32//1.3)


#Calculate Phase spectrogram
f_a,t_a,Sxx_a=signal.spectrogram(rir_1_0,16000,mode="angle",nperseg=32,noverlap=32//1.3)
f_1_a,t_1_a,Sxx_1_a=signal.spectrogram(rir_2_0,16000,mode="angle",nperseg=32,noverlap=32//1.3)

Sxx_a = Sxx_a*180
Sxx_1_a = Sxx_1_a*180


#Calculate PSD

f_p_1,pxx_1=signal.welch(rir_1_0,16000,nperseg=32,noverlap=32//1.3)
f_p_2, pxx_2 = signal.welch(rir_2_0, 16000, nperseg=32,noverlap=32//1.3)

#Calculate group Delay

Sxx_grp_delay=cal_grp_delay(t_a,f_a,Sxx_a)
Sxx_1_grp_delay = cal_grp_delay(t_1_a, f_1_a, Sxx_1_a)



ax3=fig.add_subplot(spec[0, 0])

mf=ax3.pcolormesh(t,f,10*np.log10(np.abs(Sxx)),shading='gouraud')
ax3.set_xlabel("Time [sec]")
ax3.set_ylabel("Frequency [Hz]")
ax3.set_title("Simulated Spectrogram")
fig.colorbar(mf,use_gridspec=True,ax=ax3)

ax4=fig.add_subplot(spec[1, 0])

g=ax4.pcolormesh(t_1,f_1,10*np.log10(np.abs(Sxx_1)),shading='gouraud')
ax4.set_xlabel("Time [sec]")
ax4.set_ylabel("Frequency [Hz]")
ax4.set_title("dEchorate Spectrogram")
fig.colorbar(g,use_gridspec=True,ax=ax4)


ax5=fig.add_subplot(spec[0, 1])
phase1=ax5.pcolormesh(t_a,f_a,10*np.log10(np.abs(Sxx_a)),shading='gouraud')
ax5.set_xlabel("Time [sec]")
ax5.set_ylabel("Frequency [Hz]")
ax5.set_title("Simulated Phase Spectrogram")
fig.colorbar(phase1,use_gridspec=True,ax=ax5)

ax6=fig.add_subplot(spec[1, 1])
phase2=ax6.pcolormesh(t_1_a,f_1_a,10*np.log10(np.abs(Sxx_1_a)),shading='gouraud')
ax6.set_xlabel("Time [sec]")
ax6.set_ylabel("Frequency [Hz]")
ax6.set_title("dEchorate Phase Spectrogram")
fig.colorbar(phase2,use_gridspec=True,ax=ax6)

ax7=fig.add_subplot(spec[0, 2])
ax8=fig.add_subplot(spec[1, 2])

grp1=ax7.pcolormesh(t_a,f_a,10*np.log10(np.abs(Sxx_grp_delay)),shading='gouraud')
ax7.set_xlabel("Time [sec]")
ax7.set_ylabel("Frequency [Hz]")
ax7.set_title("Simulated grp delay")
fig.colorbar(grp1,use_gridspec=True,ax=ax7)

grp2=ax8.pcolormesh(t_1_a,f_1_a,10*np.log10(np.abs(Sxx_1_grp_delay)),shading='gouraud')
ax8.set_xlabel("Time [sec]")
ax8.set_ylabel("Frequency [Hz]")
ax8.set_title("dEchorate grp delay")
fig.colorbar(grp2,use_gridspec=True,ax=ax8)

ax9=fig.add_subplot(spec[2, :])
ax9.semilogy(f, pxx_1, color="green", label="Simulated")
ax9.semilogy(f, pxx_2, color="blue", label="dEchorate")

ax9.set_ylabel("PSD [V**2/Hz]")
ax9.set_title("PSD")
ax9.set_xlabel("Frequency Hz")
ax9.legend()



plt.legend()
plt.show()


"""


"""
####################################################
# Compare Old pyroom acoustic and dirpat pyroom    #
# plot old RIR and pyroom DIRPAT RIR               #
# in time domain and frequency domain              #
####################################################
Requires path to dirpat pyroom RIR and path to old pyroom generate
RIR.
Simulated in the same frequency , rotation of the source and
receiver should be the same the same goes with the directivity pattern,
acoustic scene should be same . DIrectivity pattern should be imported from SOFA file.

"""


"""

#To plot the newly created sofa files.
rir_1_0=np.load("debug_rir_dir_1.npy")
rir_2_0=np.load("debug_rir_dir_2.npy") #debug_rir_dir_2.npy,room_011111_c_src.npy

#rir_2_0=decimate(rir_2_0, int(round(48000 / 16000)))
pad_zero_rir=np.zeros(rir_1_0.shape[0])
pad_zero_rir[:len(rir_2_0)]=rir_2_0


fig,axs=plt.subplots(2,1,figsize=(25,35))

plt.title("SOFA Files containing frequency independent analytic pattern interpolated on fibo sphere , src rotated az=120 col=31, rec rotated az=46,col=163")


axs[0].plot(np.arange(rir_1_0.shape[0]),pad_zero_rir,c="orange",label="Octave band processing ")
axs[0].legend()
#air_decay = np.exp(-0.5 * air_abs[0] * distance_rir)
#ir_diff[128:N+128]*=air_decay


axs[0].plot(np.arange(rir_1_0.shape[0])-64, rir_1_0, label="DFT domain RIR processing")
axs[0].legend()

#"New Min-phase Method IR With " + txt +" directivity from SOFA files"
AB_=10 * np.log10(np.abs(fft(rir_1_0)[:(rir_1_0.shape[0] // 2)]) ** 2)
BA_=10*np.log10(np.abs(fft(pad_zero_rir)[:(pad_zero_rir.shape[0]//2)])**2)
axs[1].plot(fftfreq(rir_1_0.shape[0],d=1/16000)[:(rir_1_0.shape[0]//2)], AB_,label="DFT domain RIR processing")
axs[1].plot(fftfreq(pad_zero_rir.shape[0],d=1/16000)[:(pad_zero_rir.shape[0]//2)], BA_,label="Octave band processing")
axs[1].legend()

print("RMSE db IN FREQ DOMAIN",np.sqrt(np.mean((AB_-BA_)**2)))
plt.show()


#rir_1_0 = signal.decimate(rir_1_0,int(round(48000/16000)))
#np.save("debug_rir_4.npy",rir_1_0)


"""


####### Fractional Delay Computation Experiment (Interpolation and Look up table) ############
"""

from scipy.fft import fft
fdl=80
lut_gran=20
lut_size = (fdl + 1) * lut_gran + 1
fdl2=(fdl - 1) // 2d
n = np.linspace(-fdl2-1, fdl2 + 1, lut_size)
print(n)
g=[]
k=np.sinc(n)
tau=8
for i in range(81):
    g.append((k[tau] + 0.66 * (k[tau+1] - k[tau])))
    tau+=20
plt.plot(np.arange(81),np.sinc(g))
plt.show()
"""


"""
tau = 0.3  # Fractional delay [samples].
N = 201  # Filter length.
n = np.arange(N)
#print(n)
#print(n-(N-1)/2)
#print(n-(N-1)/2-tau)

# Compute sinc filter.
h = np.sinc(n - (N - 1) / 2 - tau)

# Multiply sinc filter by window
#h *= np.blackman(N)

# Normalize to get unity gain.
#h /= np.sum(h)
#plt.clf()
#plt.plot(np.arange(201),h)
#plt.show()
from scipy.fft import fftfreq,ifft
import math
s=np.zeros(257,dtype=np.complex_)
for a,f in enumerate(fftfreq(512,d=1/16000)):

        l=-2 * 1j * np.pi * f * tau
        s[a]=np.exp(l)

plt.plot(np.arange(257),np.real(s))
plt.show()
"""


"""
impulse_resp=np.zeros(32)
impulse_resp[10]=1
alpha=0.55
si=np.sinc(np.arange(0,32)-0.3)
plt.plot(np.arange(32),np.abs(fft(si)))
plt.show()
wd=np.hanning(32)
ls=[]
plt.clf()
for i in range(32):
        ls.append(alpha*si[i]*wd[i])

plt.plot(np.arange(32),ls)
plt.show()
"""


# dl=np.arange(-40,41,1)
# dl=dl-0.282196044921875
# dl=np.sinc(dl)

# wd=np.hanning(81)
# k=fftconvolve(wd,dl,mode="same")
# plt.plot(np.arange(81),np.abs(fft(wd*dl)))
# plt.show()
