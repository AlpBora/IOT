import os
import numpy as np
import pandas as pd


class make_folder:

    def device_folder(parent_folder):
        global parent_folder2, i, j
        device = []
        parent_folder2 = []
        for i in range(25):
            device.append('Device_' + str(i + 1))
        for j in range(25):
            parent_folder2.append(parent_folder + device[j] + '/')
        return parent_folder2

    def day_folder(self):
        global k, path, l
        day_path2 = []
        day_path = 'Record/Diff_Days/'
        day = []
        for k in range(5):
            day.append('Day_' + str(k + 1))
        path = []
        for l in range(5):
            day_path2.append(day_path + day[l])
        return day_path2

    def receiver_folder(self):
        global k, path, l
        receiver_path2 = []
        receiver_path = 'Record/Diff_Receivers/'
        receiver = []
        for k in range(2):
            receiver.append('Recv_' + str(k + 1))
        path = []
        for l in range(2):
            receiver_path2.append(receiver_path + receiver[l])
        return receiver_path2

    def location_folder(self):
        global k, path, l
        location_path2 = []
        location_path = 'Record/Diff_Locations/'
        location = []
        for k in range(3):
            location.append('Location_' + str(k + 1))
        path = []
        for l in range(3):
            location_path2.append(location_path + location[l])
        return location_path2

    def distance_folder(self):
        global k, path, l
        distance_path2 = []
        distance_path = 'Record/Diff_Distances/'
        distance = []
        for k in range(4):
            distance.append(str(5 * (k + 1)) + 'm')
        path = []
        for l in range(4):
            distance_path2.append(distance_path + distance[l])
        return distance_path2

    def config_folder(self):
        global k, path, l
        config_path2 = []
        config_path = 'Record/Diff_Configurations/'
        config = []
        for k in range(4):
            config.append('Config_' + str(k + 1))
        path = []
        for l in range(4):
            config_path2.append(config_path + config[l])
        return config_path2

    def make(parent_folder):
        parent_folder2 = make_folder.device_folder(parent_folder)
        day_path = make_folder.day_folder()
        receiver_path = make_folder.receiver_folder()
        distance_path = make_folder.distance_folder()
        config_path = make_folder.config_folder()
        location_path = make_folder.location_folder()

        for j in range(np.size(parent_folder2)):
            for d in range(np.size(day_path)):
                path.append(os.path.join(parent_folder2[j], day_path[d]))
            for r in range(np.size(receiver_path)):
                path.append(os.path.join(parent_folder2[j], receiver_path[r]))
            for di in range(np.size(distance_path)):
                path.append(os.path.join(parent_folder2[j], distance_path[di]))
            for c in range(np.size(config_path)):
                path.append(os.path.join(parent_folder2[j], config_path[c]))
            for l in range(np.size(location_path)):
                path.append(os.path.join(parent_folder2[j], location_path[l]))

        for i in range(np.size(path)):
            if not os.path.exists(path[i]):
                os.makedirs(path[i])

class rename_files:

    def main(self):

        Source_Path = r'/home/mp3/Desktop/Files'
        Destination = r'/home/mp3/Desktop/Renamed_files'
        files = os.listdir(Source_Path)
        for count, filename in enumerate(files):
            filename_split = filename.split("_")
            file_no = filename_split[1]
            file_no_split = file_no.split(".")
            file_no2 = file_no_split[0]
            dst = "IQ_" + file_no2 + "_" + "Day1_SF7B125_Outdoor_5m.dat"
            print(dst)

            # rename all the files
            os.rename(os.path.join(Source_Path, filename), os.path.join(Destination, dst))
    def rename(self):
        rename_files.main()

class merge_csv:

    def path_finder(path):
        folders_path = [os.path.join(root, name) for root, dirs, files in os.walk(path) for name in files if
                        name.endswith(".csv")]
        print(folders_path)
        return folders_path

    def csv(path, csv_name):
        all_filenames = merge_csv.path_finder(path)
        # combine all files in the list
        combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
        # export to csv
        combined_csv.to_csv(os.path.join(path, csv_name), index=False, encoding='utf-8-sig')

    def merge(path,csv_name):
        #path = '/home/mp3/Desktop/Lora Dataset/RFFP-dataset/Device_2/Burst'
        #csv_name = 'Device_2_Combined.csv'
        merge_csv.csv(path, csv_name)


if __name__ == '__main__':

    merge_csv.merge(r'/home/mp3/Desktop/Lora Dataset/RFFP-dataset/Device_2/Burst', 'Device_2_Combined.csv' )
    make_folder.make(parent_folder = '/home/mp3/Desktop/Lora Dataset/RFFP-dataset/')