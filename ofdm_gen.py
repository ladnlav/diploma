# files
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

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
    return np.fft.ifftn(OFDM_data,[Nfft,N])

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
    Fc1 = 100e3 # in Hz n25 band carrier frequency
    Fc2 = 100e3 # in Hz n66 band carrier frequency

    df = 1e3 # scs in Hz
    Nfft = 2048*2
    K = 64 # number of used OFDM subcarriers

    B = df*int(Nfft/2) # bandwidth in Hz
    print("Current Bandwidth: ", B)
    print("Current Occupied Bandwidth: ", df*int(K/2))
    Fs = B         # sampling frequency
    N = 1000           # number of OFDM symbols transmitted

    ### Power
    Sig_pow_dB = 90 # signal power in dB
    noise_dB = 50  # noise power

    ### Carrier arrangement
    # allCarriers = np.arange(Nfft)  # indices of all subcarriers ([0, 1, ... Nfft-1])
    # dataCarriers = np.arange(K)    # indices of all data-subcarriers ([0, 1, ... K-1])
    mu = 4 # bits per symbol (i.e. 16QAM)
    payloadBits_per_OFDM = K*mu  # number of payload bits per OFDM symbol

    ##### Bits generation
    np.random.seed(64)
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
    # s_amp = s
    noise, s_rx = channel(s_amp,noise_dB)
    
    plt.figure()
    scale1 = 800
    scale2 = 1024

    f, Pxx_spec = welch(np.transpose(s_amp), Fs, nperseg=scale1, nfft=scale2, return_onesided=False)
    _, noise_spec = welch(np.transpose(noise), Fs, nperseg=scale1, nfft=scale2, return_onesided=False)
    _, s_rx_spec = welch(np.transpose(s_rx), Fs, nperseg=scale1, nfft=scale2, return_onesided=False)

    Pxx_spec = np.mean(Pxx_spec,0)
    noise_spec = np.mean(noise_spec,0)
    s_rx_spec = np.mean(s_rx_spec,0)

    Pxx_dB=10*np.log10(np.abs(Pxx_spec)*Fs*np.sqrt((1/len(Pxx_spec))))
    noise_spec_dB= 10*np.log10(np.abs(noise_spec)*Fs)
    s_rx_spec_dB = 10*np.log10(np.abs(s_rx_spec)*Fs*np.sqrt((1/len(Pxx_spec))))
    

    plt.plot(np.fft.fftshift(f),Pxx_dB, label='Tx signal')
    plt.plot(np.fft.fftshift(f), noise_spec_dB, label='Noise')
    plt.plot(np.fft.fftshift(f), s_rx_spec_dB, label='Rx signal', alpha=0.6)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('Signal Power [dB]')
    # plt.ylim(bottom=0)
    # plt.xlim(left=-Fc1-40e6, right=Fc1+40e6)
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

