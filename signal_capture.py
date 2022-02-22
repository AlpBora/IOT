import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import math
import os


def csv_writer(path, data):

    header = ['Burst No', 'Record No', 'Date', 'SNR', 'File Name', 'Path', 'Device ID']
    data.insert(0, header)
    np.savetxt(path, data, delimiter=",", fmt='% s')


def writer(dir, data):
    with open(dir, 'wb') as file_handler:
        file_handler.write(data)


def fileread(file):
    samples = np.fromfile(file, dtype= 'float32')
    return samples


def convert_iq(samples):
    iq = samples[0::2] + 1j * samples[1::2]
    return iq

def signaltonoiseScipy(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

def find_threshold(signal):

    rand =np.random.choice(np.size(signal),10000)
    selected = np.array([])
    for select in rand:
        selected = np.append(selected,signal[select])
    sorted = np.sort(selected.flatten())
    gradient =np.gradient(sorted)
    threshold_index = np.argmax(gradient)
    threshold = sorted[threshold_index]

    return threshold

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w

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
    iq_abs = np.abs(iq[:1000000000])
    iq_moving = moving_average(iq_abs, 250)

    return iq_moving


if __name__ == '__main__':

    input_path = r'/home/mp3/Desktop/Lora Dataset/RFFP-dataset/Device 2/Record/Diff_Days_Outdoor_Setup/Day1'
    spilt_path = input_path.split("/")
    parent_dir = r'/home/mp3/Desktop/Lora Dataset/RFFP-dataset/'
    Device_ID = spilt_path[6]
    folder_name = Device_ID + '/' + 'Burst/' + spilt_path[8] + '/' + spilt_path[9]
    path = os.path.join(parent_dir, folder_name)
    os.makedirs(path)

    record_file_names = os.listdir(input_path)

    bin_files = [file for file in record_file_names if 'dat' in file[-3:]]  #list comprehension
    print(bin_files)

    csv_data = []

    for i in range(np.size(bin_files)):

        input_signal = fileread(input_path + '/' + bin_files[i]) #IQIQIQIQ
        iq_signal = convert_iq(input_signal) #Q + Ij
        iq_moving = get_moving(iq_signal)
        threshold = find_threshold(iq_moving)
        print(threshold)
        abs_iq_signal = np.abs(iq_signal)
        #arrplot(abs_iq_signal)
        burst_array = cut(iq_moving, threshold)
        print(burst_array)
        SNRscipy = 20 * math.log10(abs(signaltonoiseScipy(abs_iq_signal)))
        print("SNR by scipy: {} dB".format(SNRscipy))

        for j in range(len(burst_array)):
            data = margin_data(burst_array, j)
            data_start = data[0]
            data_end = data[1]
            signal = input_signal[2*(data_start-300):2*(data_end+300)]
            SNRscipy = 20 * math.log10(abs(signaltonoiseScipy(np.abs(convert_iq(signal)))))
            #print("SNR by scipy: {} dB".format(SNRscipy))
            #arrplot(np.abs(convert_iq(signal)))
            #arrplot(signal)

            burst_no = '{:05d}'.format(j+1)
            record_no = '{:04d}'.format(i+1)
            time = datetime.now()
            date_time = time.strftime("%m/%d/%Y_%H:%M:%S")
            burst_file_name = bin_files[i].replace('.dat', '') + '_burst__' + str(burst_no) + '.dat'
            csv_file_name = 'csv_file.csv'
            output_path = path + '/' + burst_file_name
            csv_path = path + '/' + csv_file_name
            writer(output_path, signal)
            csv = np.array([burst_no, record_no, date_time, SNRscipy, burst_file_name, output_path, Device_ID])
            csv_data.append(list(csv))

    csv_writer(csv_path, csv_data)
        


