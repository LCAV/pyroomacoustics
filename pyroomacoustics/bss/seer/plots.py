import matplotlib.pyplot as plt
import numpy as np

def plotInitials(ref, room):
    fig = plt.figure()
    fig.set_size_inches(20, 8)
    plt.subplot(2,1,1)
    plt.specgram(ref[0,:,0], NFFT=1024, Fs=room.fs)
    plt.title('Source 0 (clean)')
    plt.subplot(2,1,2)
    plt.specgram(ref[1,:,0], NFFT=1024, Fs=room.fs)
    plt.title('Source 1 (clean)')
    plt.tight_layout(pad=0.5)
    plt.show()

def plotComparaison(ref, y, room):
    fig = plt.figure()
    fig.set_size_inches(20, 8)
    plt.subplot(4,1,1)
    plt.specgram(ref[0,:,0], NFFT=1024, Fs=room.fs)
    plt.title('Source 0 (clean)')
    plt.subplot(4,1,2)
    plt.specgram(ref[1,:,0], NFFT=1024, Fs=room.fs)
    plt.title('Source 1 (clean)')
    plt.subplot(4,1,3)
    plt.specgram(y[0,:,0], NFFT=1024, Fs=room.fs)
    plt.title('Source 0 (clean)')
    plt.subplot(4,1,4)
    plt.specgram(y[1,:,0], NFFT=1024, Fs=room.fs)
    plt.title('Source 1 (clean)')
    plt.tight_layout(pad=0.5)
    plt.show()

def plotVect(Z):
    fig = plt.figure()
    fig.set_size_inches(20, 8)
    plt.subplot(2, 1, 1)
    plt.plot(range(Z.shape[0]),np.real(Z))
    plt.title('Z (real)')
    plt.subplot(2, 1, 2)
    plt.plot(range(Z.shape[0]), np.imag(Z))
    plt.title('Z (imag)')
    plt.show()


