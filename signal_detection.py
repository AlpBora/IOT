import numpy as np
from scipy.signal import kaiserord, lfilter, firwin, freqz, convolve
import os
import matplotlib.pyplot as plt
from scipy.signal import chirp, spectrogram


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w

def create_filter(sample_rate, cutoff_hz, width=1 / 50, ripple_db=60, num_tap=1e4):
    nyq_rate = sample_rate / 2.0
    N, beta = kaiserord(ripple_db, width)
    taps = firwin(N, cutoff_hz / nyq_rate, window='hamming')
    return taps

def get_periodogram_psd_with_len(sig, sig_len,sample_rate, center_freq):
    #from_fft = np.fft.fft(sig, n=sig_len)
    #shifted = np.fft.fftshift(np.abs(from_fft))
    #output_signal = 20 * np.log10(shifted)  # dB
    #sig = sig * np.hamming(len(sig))

    psd = (np.abs(np.fft.fft(sig))/sig_len)**2
    psd_log = 10.0*np.log10(psd)
    psd_shiifted = np.fft.fftshift(psd_log)
    f = np.arange(sample_rate / -2.0, sample_rate / 2.0, sample_rate / sig_len) + center_freq
    return f, psd_shiifted

def fileread(file,dtype):
    samples = np.fromfile(file, dtype= dtype)
    return samples


def convert_iq(samples):
    iq = samples[0::2] + 1j * samples[1::2]
    return iq


def sampling(iq, a, b): #limits of sampling
    sample_iq = iq[a:b]
    return sample_iq


def check_signal(sample_iq, center_freq, sample_rate, cutoff_hz, filter, method, get):
    if filter == 'FIR':
        if method == 'Window':

            taps = create_filter(sample_rate, cutoff_hz= cutoff_hz / 1.8, num_tap=sample_iq.size)
            filtered = convolve(sample_iq, taps, mode='same')


    if get == 'psd filtered':

        return get_periodogram_psd_with_len(filtered, filtered.size, sample_rate, center_freq)
    if get == 'psd':

        return get_periodogram_psd_with_len(sample_iq, sample_iq.size, sample_rate, center_freq)


def show_signal(sample_iq, method):

    if method == 'MA':
        output = moving_average(np.abs(sample_iq), 250)
    return output

def signal_plot(data, get):

    fig, axs = plt.subplots(np.size(get))
    for i in range(np.size(get)):
        if get[i] == 'PSD' or 'PSD Filtered':
            axs[i].plot(data[i][0], data[i][1])
        if get[i] == 'Moving Average' or 'Signal':
            axs[i].plot(data[i])
        axs[i].set_title(get[i], fontsize = 11)
    plt.subplots_adjust(hspace=0.45)
    plt.show()
    """
        for i in range(np.size(get)):
        if get[i] == 'psd':
            axs[i].plot(data[i][0], data[i][1])
            axs[i].set_title('PSD', fontsize=11)
            plt.show()
            print(i)
        if get[i] == 'moving':
            axs[i].plot(data[i])
            axs[i].set_title('Moving Average', fontsize=11)
            plt.show()
            print(i)
        if get[i] == 'signal':
            print(i)
            axs[i].plot(data[i])
            axs[i].set_title('Lora Signal', fontsize=11)
            plt.show()
        if get[i] == 'psd filtered':
            print(i)
            data = data[i]
            axs[i].plot(data[0], data[1])
            axs[i].set_title('PSD Filtered', fontsize=11)
            plt.show()
        plt.subplots_adjust(hspace=0.45)
    """


"""
def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

dat = scipy.io.loadmat('./data.mat')
arr = dat['dn']
snr = signaltonoise(arr)
"""

if __name__ == '__main__':

    directory = r'/home/mp3/Desktop/Lora Dataset/RFFP-dataset/Diff_Days_Outdoor_Setup/Day1/Device1/Record'
    file_path = os.listdir(directory)

    center_freq = 915e6
    sample_rate = 1e6
    cutoff_hz = 500

    for i in range(1):
        signal = fileread(directory + '/' + file_path[i], 'int32')
        signal_iq = convert_iq(signal)
        sample_IQ = sampling(signal_iq, 1000, 100000)  # sınırlar
        moving = show_signal(sample_IQ, method='MA')
        moving_iq = show_signal(signal_iq, method='MA')
        psd_filtered = check_signal(sample_IQ, center_freq, sample_rate, cutoff_hz, filter='FIR', method='Window', get='psd filtered')
        psd = check_signal(sample_IQ, center_freq, sample_rate, cutoff_hz, filter='FIR', method='Window', get='psd')
        # IQ_noise = fileread(directory2 + '\\' + file_path2[0])
        # moving_noise = show_signal(IQ_noise, method='MA')

    plt.plot(np.abs(sample_IQ))
    plt.show()


    #data = [[psd[0], psd[1]], moving, signal, [psd_filtered[0], psd_filtered[1]]]
    #get = ['PSD', 'Moving Average', 'Signal', 'PSD Filtered']
    #signal_plot(data, get)




