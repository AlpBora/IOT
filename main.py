import os

import signal_detection as det
import matplotlib.pyplot as plt
import numpy as np

directory = r'C:\Users\Alp Bora\Desktop\          \IOT\Lora\Records\Bw500Cr45Sf128\Burst'
file_path = os.listdir(directory)

center_freq = 868e6
sample_rate = 5e6
cutoff_hz = 500

for i in range(np.size(file_path)):
    signal = det.fileread(directory + '\\' + file_path[i])
    signal_iq = det.convert_iq(signal)
    sample_IQ = det.sampling(signal_iq, 0, np.size(signal_iq))  #sınırlar
    moving = det.show_signal(sample_IQ, method='MA')
    moving_iq = det.show_signal(signal_iq, method='MA')
    #psd_filtered = det.check_signal(sample_IQ, center_freq, sample_rate, cutoff_hz, filter='FIR', method='Window', get='psd filtered')
    psd = det.check_signal(sample_IQ, center_freq, sample_rate, cutoff_hz, filter='FIR', method='Window', get='psd')
    #IQ_noise = det.fileread(directory2 + '\\' + file_path2[0])
    #moving_noise = det.show_signal(IQ_noise, method='MA')


fig, axs = plt.subplots(3)
axs[0].plot(psd[0], psd[1])
axs[0].set_title('PSD', fontsize= 11)
axs[1].plot(moving)
axs[1].set_title('Moving Average', fontsize= 11)
axs[2].plot(signal)
axs[2].set_title('Lora Signal', fontsize= 11)
plt.subplots_adjust(hspace = 0.45)
#axs[1].plot(psd_filtered[0], psd_filtered[1])
#axs[0].set_title('Moving Average of Abs Sample IQ')
#axs[1].set_title('PSD Filtered')
#axs[1].legend()
plt.show()






