# files
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import matplotlib.patches as patches

# modules
from IQ_map_demap import IQ_LUT
mapping_table, demapping_table = IQ_LUT('16QAM')

# Funcs
def SP(bits, rows, columns):
    # bits - signal to S/P
    # rows - num of subcarriers
    # columns - bits in IQ symbol

    return bits.reshape((rows, columns))

def Mapping(bits):
    return np.array([mapping_table[tuple(b)] for b in bits])

def OFDM_symbol(QAM_payload, K, N, Nfft):
    # QAM_payload - mapped data
    # Nfft - total amount of subcarriers
    # K - amount of data-subcarriers
    # N - total number of OFDM symbols transmitted

    symbols = np.zeros((Nfft,N), dtype=complex) # matrix [Nfft,N]
    data=QAM_payload.reshape((K,N))

    start = Nfft//2-K//2
    symbols[start:start+K,:] = data
    return symbols

def IDFT(OFDM_data,Nfft,N):
    OFDM_data2 = np.fft.ifftshift(OFDM_data,0)
    return np.fft.ifftn(OFDM_data2,[Nfft,N])

def power_set(signal, Sig_pow_dB):
    Sig_pow = 10**(Sig_pow_dB/10)
    power=np.mean(np.abs(signal)**2) # current mean power of the signal
    amp_factor = np.sqrt(Sig_pow/power)
    signal = amp_factor*signal
    return signal

def channel(signal,noise_dB):
    signal_power = np.mean(np.abs(signal)**2)
    sigma2=10**(noise_dB/10)
    SNRdB = 10*np.log10(signal_power/sigma2)
    print('SNR for given noise and signal power: %.4f Signal power in dB: %.4f' % (SNRdB, 10*np.log10(signal_power)))
    
    # Generate noise with given variance
    noise = np.sqrt(sigma2/2) * (np.random.randn(*signal.shape)+1j*np.random.randn(*signal.shape))
    noise_power = 10*np.log10(np.mean(np.abs(noise)**2))
    print('Noise power in dB: %.2f' % (noise_power))

    return noise, noise + signal

def DFT(OFDM_RX):
    return np.fft.fft(OFDM_RX)

def Demapping(QAM):
    # array of possible constellation points
    constellation = np.array([x for x in demapping_table.keys()])
    
    # calculate distance of each RX point to each possible point
    dists = abs(QAM.reshape((-1,1)) - constellation.reshape((1,-1)))
    
    # for each element in QAM, choose the index in constellation 
    # that belongs to the nearest constellation point
    const_index = dists.argmin(axis=1)
    
    # get back the real constellation point
    hardDecision = constellation[const_index]
    
    # transform the constellation point into the bit groups
    return np.vstack([demapping_table[C] for C in hardDecision]), hardDecision

def PS(bits):
    return bits.reshape((-1,))

def main():
    ### Parameters setting
    Fc1 = 15e3 # in kHz n25 band carrier frequency
    Fc2 = 15e3 # in kHz n66 band carrier frequency

    K = 64 # number of used OFDM subcarriers
    df = 2*5/K*1e3 # scs in kHz, 2*Occupied_bandwidth/Subcarrier_num*1e3
    Nfft = int(2048*2)


    B = df*int(Nfft/2) # bandwidth in Hz
    print("Current Bandwidth: in kHz", B)
    print("Current Occupied Bandwidth: in kHz", df*int(K/2))
    Fs = B         # sampling frequency
    N = 100           # number of OFDM symbols transmitted

    ### Power
    Sig_pow_dB = 90 # signal power in dB
    noise_dB = 50  # noise power

    ### Carrier arrangement
    # allCarriers = np.arange(Nfft)  # indices of all subcarriers ([0, 1, ... Nfft-1])
    # dataCarriers = np.arange(K)    # indices of all data-subcarriers ([0, 1, ... K-1])
    mu = 4 # bits per symbol (i.e. 16QAM)
    payloadBits_per_OFDM = K*mu  # number of payload bits per OFDM symbol

    ##### Bits generation
    np.random.seed(6)
    bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, N))

    ##### S/P
    bits_SP = SP(bits, N*K, mu)

    ##### Mapping bits into constellation points
    QAM = Mapping(bits_SP)

    ##### OFDM-symbol creation
    OFDM_data = OFDM_symbol(QAM, K,N, Nfft)

    ##### IDFT operation (freq->time)
    OFDM_time = IDFT(OFDM_data,Nfft,N)

    t = np.arange(0, Nfft/Fs, 1/Fs)  # the time samples
    f = np.arange(-Fs/2, Fs/2, Fs/Nfft)  # the corresponding frequency samples
    # s = s1+s2
    s = OFDM_time
    # plt.figure()
    # plt.plot((10 * np.log10(np.fft.fftshift(np.abs(np.fft.fft(OFDM_time))))))
    # plt.show()
    # t_total = np.arange(len(s))/Fs
    # plt.figure()
    # plt.plot(t_total, s)
    # plt.show()
    s1 = np.multiply(np.transpose(s),np.exp(-1j*2*np.pi*Fc1*t))
    s2 = np.multiply(np.transpose(s),np.exp(1j*2*np.pi*Fc2*t))
    s = np.transpose(s1+s2)

    # Power setting
    s_amp = power_set(s, Sig_pow_dB)
    nperseg = 1024

    s_amp=s_amp*np.sqrt(2*K/Nfft)  # normalization for FFT

    noise, s_rx = channel(s_amp,noise_dB)

    f, Pxx_spec = welch(np.transpose(s_amp), Fs, nperseg=nperseg, nfft=Nfft, return_onesided=False)
    _, noise_spec = welch(np.transpose(noise), Fs, nperseg=nperseg, nfft=Nfft, return_onesided=False)
    _, s_rx_spec = welch(np.transpose(s_rx), Fs, nperseg=nperseg, nfft=Nfft, return_onesided=False)

    Pxx_spec = np.mean(Pxx_spec,0)
    noise_spec = np.mean(noise_spec,0)
    s_rx_spec = np.mean(s_rx_spec,0)

    Pxx_dB=10*np.log10(np.abs(Pxx_spec)*Fs)
    noise_spec_dB= 10*np.log10(np.abs(noise_spec)*Fs)
    s_rx_spec_dB = 10*np.log10(np.abs(s_rx_spec)*Fs)
    
    plt.figure()
    shift = 800e3 # in kHz
    f_MHz = (f+shift) / 1e3
    
    plt.plot(np.fft.fftshift(f_MHz), np.fft.fftshift(Pxx_dB), label='Tx signal')
    plt.plot(np.fft.fftshift(f_MHz), np.fft.fftshift(noise_spec_dB), label='Noise')
    plt.plot(np.fft.fftshift(f_MHz), np.fft.fftshift(s_rx_spec_dB), label='Rx signal', alpha=0.6)
    plt.xlabel('frequency [MHz]')
    plt.ylabel('Signal Power $V^2/Hz$ [dB]')
    # plt.ylim(bottom=0)
    plt.xlim(left=(-2*Fc1+shift)/1e3, right=(2*Fc2+shift)/1e3)

    # Specify the UP-link bandwidth
    bandwidth_up = 70 # in MHz
    Fc_up = (Fc1+shift)/1e3-1.5*bandwidth_up # in MHz
    up_band = patches.Rectangle((Fc_up - bandwidth_up / 2, plt.ylim()[0]),
                               bandwidth_up, plt.ylim()[1] - plt.ylim()[0],
                               linewidth=1, edgecolor='black', facecolor='blue',alpha = 0.5, label='UL band')
    plt.gca().add_patch(up_band)

    # Specify the DL-link bandwidth
    bandwidth_dl = 70 # in MHz
    Fc_dl = (shift)/1e3 # in MHz
    dl_band = patches.Rectangle((Fc_dl - bandwidth_dl / 2, plt.ylim()[0]),
                               bandwidth_dl, plt.ylim()[1] - plt.ylim()[0],
                               linewidth=1, edgecolor='black', facecolor='red',alpha = 0.2, label='DL band')
    plt.gca().add_patch(dl_band)


    #custom_bins = np.arange((-2*Fc1+shift)/1e3, (2*Fc2+shift)/1e3, 5)
    custom_bins = np.arange(np.min(f_MHz), np.max(f_MHz), 5)
    plt.xticks(custom_bins, rotation='vertical')
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

