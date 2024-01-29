# files
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.signal import periodogram
import commpy
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

def OFDM_symbol(QAM_payload, allCarriers, K):
    # QAM_payload - mapped data
    # allCarriers - array of data-subcarriers
    # K - total amount of subcarriers

    symbol = np.zeros(K, dtype=complex) # the overall K subcarriers
    symbol[allCarriers] = QAM_payload  # allocate the pilot subcarriers
    return symbol

def IDFT(OFDM_data):
    return np.fft.ifft(OFDM_data)

def addCP(OFDM_time, CP):
    cp = OFDM_time[-CP:]               # take the last CP samples ...
    return np.hstack([cp, OFDM_time])  # ... and add them to the beginning

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
    noise = np.sqrt(sigma2/2) * np.random.randn(*signal.shape)

    noise_power = 10*np.log10(np.mean(noise**2))
    print('Noise power in dB: %.2f' % (noise_power))
    
    # f, noise_spec = welch(noise, 4000000000, nperseg=2*1024, scaling='spectrum', nfft=16*1024)
    # total_power_welch = 10 * np.log10((noise_spec))
    # #print('Welch Noise power in dB: %.4f' % (total_power_welch))
    # plt.plot(f, total_power_welch, label='Noise')
    # plt.show()
    return noise, signal + noise

def removeCP(signal, CP, K):
    return signal[CP:(CP+K)]

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

def matched_filter(Ts, Fs_analog):
    t0 = 3*Ts 

    _, rrc = commpy.filters.rrcosfilter(N=int(2*t0*Fs_analog), alpha=1,Ts=Ts, Fs=Fs_analog)
    t_rrc = np.arange(len(rrc)) / Fs_analog  # the time points that correspond to the filter values
    #plt.plot(t_rrc/Ts, rrc)

    return t_rrc, rrc, t0

def Upconversion(signal, N, ups, rrc, Fc,Fs_analog):

    # Step T1
    x = np.zeros(ups*N, dtype='complex')
    x[::ups] = signal # every ups samples, the value of signal is inserted into the sequence
    # t_x = np.arange(len(x))/Fs_analog

    # Step T2
    u = np.convolve(x, rrc)
    t_u = np.arange(len(u))/Fs_analog

    # Step T3
    i = u.real
    q = u.imag

    # Step T4
    iup = i * np.cos(2*np.pi*t_u*Fc)  
    qup = q * -np.sin(2*np.pi*t_u*Fc)

    # Step T5
    s = iup + qup
    return s

def main():
    ### Parameters setting
    Fc1 = 790e6 # in Hz n25 band carrier frequency
    Fc2 = 820e6 # in Hz n66 band carrier frequency

    B = 5e6 # bandwidth in Hz    
    K = 64 # number of OFDM subcarriers
    Fs_analog = 4*Fc1    # the sampling frequency we use for the discrete simulation of "analog" signals during upconversion
    df = B/K           # scs
    Tu = 1/df      # OFDM symb period
    Ts = Tu/K      # the baseband samples are Ts seconds apart.

    ups = int(Ts*Fs_analog) # number of samples per symbol in the "analog" domain

    CP = K//4  # length of the cyclic prefix: 25% of the block
    N = K+CP           # number of transmitted baseband samples

    ### Power
    Sig_pow_dB = 90 # signal power in dB
    noise_dB = 28+50  # noise power -- 28???

    ### Carrier arrangement
    allCarriers = np.arange(K)  # indices of all subcarriers ([0, 1, ... K-1])
    mu = 4 # bits per symbol (i.e. 16QAM)
    payloadBits_per_OFDM = len(allCarriers)*mu  # number of payload bits per OFDM symbol

    ##### Bits generation
    np.random.seed(64)
    bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))

    ##### S/P
    bits_SP = SP(bits, len(allCarriers), mu)

    ##### Mapping bits into constellation points
    QAM = Mapping(bits_SP)

    ##### OFDM-symbol creation
    OFDM_data = OFDM_symbol(QAM, allCarriers, K)

    ##### IDFT operation (freq->time)
    OFDM_time = IDFT(OFDM_data)

    ##### Adding CP
    OFDM_withCP = addCP(OFDM_time, CP)

    ######## Upconversion
    # Step T0: matched filter design
    t_rrc, rrc, t0 = matched_filter(Ts, Fs_analog)

    # Steps T1-T5:
    s1 = Upconversion(OFDM_withCP, N, ups, rrc, Fc1, Fs_analog)
    s2 = Upconversion(OFDM_withCP, N, ups, rrc, Fc2, Fs_analog)

    s = s1+s2
    # Power setting
    s_amp = power_set(s, Sig_pow_dB)
    noise, s_rx = channel(s_amp,noise_dB)

    plt.figure()
    scale1 = 1024
    scale2 = 2*len(s_amp)

    f, Pxx_spec = welch(s_amp, Fs_analog, nperseg=scale1, scaling='spectrum', nfft=scale2, return_onesided = True)
    _, noise_spec = welch(noise, Fs_analog, nperseg=scale1, scaling='spectrum', nfft=scale2)
    _, s_rx_spec = welch(s_rx, Fs_analog, nperseg=scale1, scaling='spectrum', nfft=scale2)

    Pxx_dB=10*np.log10(Pxx_spec*2.5)
    noise_spec_dB= 10*np.log10(noise_spec)
    s_rx_spec_dB = 10*np.log10(s_rx_spec*2.5)

    plt.plot(f, Pxx_dB, label='Tx signal')
    plt.plot(f, noise_spec_dB, label='Noise')
    plt.plot(f, s_rx_spec_dB, label='Rx signal', alpha=0.6)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('Signal Power [dB]')
    # plt.ylim(bottom=0)
    plt.xlim(left=0, right=Fc2+40e6)
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

