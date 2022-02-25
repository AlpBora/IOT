import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import pandas as pd
import math
import os


def merge_csv(path, csv_name):

    all_filenames = path_finder(path, method = 'for csv')
    # combine all files in the list
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
    # export to csv
    combined_csv.to_csv(os.path.join(path, csv_name), index=False, encoding='utf-8-sig')


def path_finder(path, method):
    if method == 'for burst':
        folders_path = [os.path.join(root, name) for root, dirs, files in os.walk(path) for name in dirs if not name.startswith("Diff")]  # to acces file names : for name in files if name.endswith(".dat")
    if method == 'for csv':
        folders_path = [os.path.join(root, name) for root, dirs, files in os.walk(path) for name in files if name.endswith(".csv")]
    print(folders_path)
    return folders_path


def csv_writer(path, data):

    header = ['Burst No', 'Record No', 'Date', 'SNR', 'Record File Name', 'Burst File Name', 'Record Path', 'Burst Path', 'Dataset', 'Device ID','Day', 'SF & Bandwidth', 'Environment', 'Distance']
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

    rand = np.random.choice(np.size(signal), 10000)
    selected = np.array([])
    for select in rand:
        selected = np.append(selected, signal[select])
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


def main(device_no):

    head_dir = r'/home/mp3/Desktop/Lora Dataset/RFFP-dataset/' + 'Device_' + str(device_no) + '/Record'
    folder_paths = path_finder(head_dir, "for burst")

    for folder_path in folder_paths: #Exp. folder_path=  '/home/mp3/Desktop/Lora Dataset/RFFP-dataset/Device_2/Record/Diff_Days/Day_1'

        if not np.size(os.listdir(folder_path)) == 0:
            input_file_names = os.listdir(folder_path)
            csv_data = []
            split_folder_path = folder_path.split("/")
            Dataset = split_folder_path[5]
            Device_ID = split_folder_path[6]
            Scenario = split_folder_path[8]  # Diff Configuratios, Diff Days, Diff Distances, Diff Locations, Diff Recievers Scenarios
            Variable = split_folder_path[9]  # Configs, Days, Distances, Locations, Recievers

            parent_dir = r'/home/mp3/Desktop/Lora Dataset/RFFP-dataset/'
            folder_name = Device_ID + '/' + 'Burst/' + Scenario + '/' + Variable
            path = os.path.join(parent_dir, folder_name)
            if not os.path.exists(path):
                os.makedirs(path)

            csv_file_name = Scenario + '_' + Variable + '.csv'
            csv_path = os.path.join(path, csv_file_name)

            for counter, input_file_name in enumerate(input_file_names, start=1):

                input_path = os.path.join(folder_path, input_file_name)
                split_path = input_path.split("/")

                Record_file_name = input_file_name
                Record_file_name_split = Record_file_name.split("_") # Exp. IQ_1_Day1_SF7B125_Indoor_5m.dat
                Day = Record_file_name_split[2]
                SF_Bandwidth = Record_file_name_split[3]
                Environment = Record_file_name_split[4]
                Distance = Record_file_name_split[5].split(".")[0]

                input_signal = fileread(input_path) #IQIQIQIQ
                iq_signal = convert_iq(input_signal) #Q + Ij
                iq_moving = get_moving(iq_signal)
                threshold = find_threshold(iq_moving)
                print(threshold)
                abs_iq_signal = np.abs(iq_signal)
                #arrplot(abs_iq_signal)
                burst_array = cut(iq_moving, threshold)
                print(burst_array)

                record_no = '{:04d}'.format(counter)
                time = datetime.now

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
                    burst_file_name = Record_file_name.replace('.dat', '') + '_' + datetime.now().strftime("%y%m%d_%H%M%S") + '_burst__' + str(burst_no) + '.dat'
                    output_path = os.path.join(path, burst_file_name)
                    writer(output_path, signal)
                    csv = np.array([burst_no, record_no, datetime.now().strftime("%d/%m/%Y %H:%M:%S"), SNRscipy, Record_file_name, burst_file_name, input_path, output_path, Dataset, Device_ID, Day, SF_Bandwidth, Environment, Distance])
                    csv_data.append(list(csv))

            csv_writer(csv_path, csv_data)


if __name__ == '__main__':

    for i in range(2):
        Device_No = i + 1
        main(Device_No)
        csv_path = r'/home/mp3/Desktop/Lora Dataset/RFFP-dataset/Device_' + str(Device_No) + '/Burst'
        csv_name = 'Device_' + str(Device_No) + '_Combined.csv'
        merge_csv(csv_path, csv_name)



