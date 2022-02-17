import matplotlib.pyplot as plt
from numpy import mean
import signal_detection as dt
import numpy as np
import os


def writer(dir, data):
    with open(dir, 'db') as file_handler:
        file_handler.write(data)


def find_threshold(signal):
    #TODO 1.5 * moving avarage
    avg = mean(signal)
    return avg * 1.5


def margin_data(array, i):
    data_start = array[i][0]
    data_end = array[i][1]
    return data_start, data_end


def arrplot(signal):
        plt.plot(signal)
        plt.show()


def cut(iq_moving, threshold):

    array2 = []
    for i in range(np.size(iq_moving)-2):
        if (iq_moving[i] > threshold) and (iq_moving[i+1] > threshold) and (iq_moving[i-1] < threshold):
            for j in range(np.size(iq_moving)-2-i):
                if (iq_moving[j+i] < threshold) and (iq_moving[i + j+1] < threshold) and (iq_moving[j+i-1] > threshold):
                    burst_infos = [i, i+j]
                    array2.append(burst_infos)
                    break

    return array2


def get_moving(iq):
    iq_abs = np.abs(iq[:100000000])
    iq_moving = dt.moving_average(iq_abs, 250)
    return iq_moving


if __name__ == '__main__':
    #input_path = r'C:\Users\Alp Bora\Desktop\IOT\Lora\Records\Bw500Cr45Sf128\Record' #records are located here
    #folder_name = r'C:\\Users\Alp Bora\Desktop\IOT\Lora\Records\Bw500Cr45Sf128\Burst\\' #burst files will be located here
    input_path = r'C:\Users\Alp Bora\Desktop\IOT\Lora\Lora Dataset\RFFP-dataset\Diff_Days_Outdoor_Setup\Day1\Device1\Record'
    folder_name = r'C:\Users\Alp Bora\Desktop\IOT\Lora\Lora Dataset\RFFP-dataset\Diff_Days_Outdoor_Setup\Day1\Device1\Burst'
    file_names = os.listdir(input_path)

    bin_files = [file for file in file_names if 'bin' in file[-3:]]  #list comprehension

    for i in range(np.size(bin_files)):

        input_signal = dt.fileread(input_path + '\\' + bin_files[i], 'int32') #IQIQIQIQ
        iq_signal = dt.convert_iq(input_signal) #Q + Ij
        iq_moving = get_moving(iq_signal)
        threshold = find_threshold(iq_moving)
        print(threshold)
        burst_array = cut(iq_moving, threshold)
        print(burst_array)

        for j in range(len(burst_array)):
            data = margin_data(burst_array, j)
            signal = input_signal[2*data[0] - 10000:2*data[1] + 10000]
            #arrplot(signal)
            burst_no = '{:05d}'.format(j+1)
            record_no = '{:04d}'.format(i+1)
            output_path = folder_name + bin_files[i].replace('.dat', '') + '_burst__' + str(burst_no) + '.dat'
            writer(output_path, signal)

