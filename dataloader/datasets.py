import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn import preprocessing
import utils


# Class to read the video dataset from the video
class VideoDataset(Dataset):

    def __init__(self, csv_file, video_path, num_frames, categories, transform=None):
        self.df = pd.read_csv(csv_file)
        #self.df = self.df.loc[self.df['Video'] == '040.mp4']

        self.transform = transform
        self.num_frames = num_frames

        self.categories = categories
        if (self.categories == "PIP"):
            label_names = ['Power', 'Precision', 'Intermediate']
            self.label_col_name = "PIP"
        elif (self.categories == "BigCategories"):
            label_names = ['pre-prismatic', 'pow-prismatic', 'circular', 'int-lateral']
            self.label_col_name = "BigCategories"
        else:
            label_names = ['pre-pinch', 'pre-prismatic', 'pre-circular', 'pow-cylindric', 'pow-oblique', 'pow-circular',
                           'int-lateral']
            self.label_col_name = "SmallCategories"

        le = preprocessing.LabelEncoder()
        le.fit(label_names)
        self.label_mapping = dict(zip(label_names, le.transform(label_names)))
        self.video_frames = []
        self.labels = []

        for index, row in self.df.iterrows():
            video_file = video_path + row["Video"]
            start_frame = row["StartFrame"]
            end_frame = row["EndFrame"]
            grasp_type = row[self.label_col_name]

            #frame_ids = utils.sample_frames(self.sampling_mode, start_frame, end_frame, self.num_frames)
            self.video_frames.append({"video_file": video_file, "start_frame": start_frame, "end_frame": end_frame})
            self.labels.append(grasp_type)
            #print("ID: ", row["ID"], " Label: ", grasp_type  , " Start Frame: ", start_frame, " End frame: ", end_frame,  " Video file: ", video_file)


        self.labels = torch.as_tensor(le.transform(self.labels))

    def __len__(self):
        """
        Returns:
            int: number of rows of the csv file (not include the header).
        """
        return len(self.labels)

    def __getitem__(self, index):
        """ get a video and its label """
        video = self.video_frames[index]
        label = self.labels[index]
        if self.transform:
            video = self.transform(video)
        return video, label

    def get_label_mapping(self):
        return sorted(self.label_mapping, key=lambda k: self.label_mapping[k])


class HandDataset(Dataset):

    def __init__(self, csv_file, video_path,num_frames, categories, transform=None):
        self.df = pd.read_csv(csv_file)

        self.transform = transform
        self.num_frames = num_frames

        self.categories = categories
        if (self.categories == "PIP"):
            label_names = ['Power', 'Precision', 'Intermediate']
            self.label_col_name = "PIP"
        elif (self.categories == "BigCategories"):
            label_names = ['pre-prismatic', 'pow-prismatic', 'circular', 'int-lateral']
            self.label_col_name = "BigCategories"
        else:
            label_names = ['pre-pinch', 'pre-prismatic', 'pre-circular', 'pow-cylindric', 'pow-oblique', 'pow-circular',
                           'int-lateral']
            self.label_col_name = "SmallCategories"


        le = preprocessing.LabelEncoder()
        le.fit(label_names)
        self.label_mapping = dict(zip(label_names, le.transform(label_names)))
        self.image_frames = []
        self.labels = []

        for index, row in self.df.iterrows():
            image_file = row["Filename"]
            grasp_type = row[self.label_col_name]

            self.image_frames.append(image_file)
            self.labels.append(grasp_type)

        self.labels = torch.as_tensor(le.transform(self.labels))

    def __len__(self):
        """
        Returns:
            int: number of rows of the csv file (not include the header).
        """
        return len(self.image_frames)

    def __getitem__(self, index):
        """ get a video """
        image = self.image_frames[index]
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        return image,label

    def get_label_mapping(self):
        return sorted(self.label_mapping, key=lambda k: self.label_mapping[k])

#Class to read the video dataset from the pytorch transformed files
class TransformedVideoDataset(Dataset):

    def __init__(self, csv_file, video_path, sampling_mode, num_frames, categories, transform=None):
        self.df = pd.read_csv(csv_file)

        self.transform = transform
        self.num_frames = num_frames


        self.categories = categories
        if (self.categories == "PIP"):
            label_names = ['Power', 'Precision', 'Intermediate']
            self.label_col_name = "PIP"
        elif (self.categories == "BigCategories"):
            label_names = ['pre-pinch', 'pre-prismatic', 'pow-prismatic', 'circular', 'int-lateral']
            self.label_col_name = "BigCategories"
        else:
            label_names = ['pre-pinch', 'pre-prismatic', 'pre-circular', 'pow-cylindric', 'pow-oblique', 'pow-circular',
                           'int-lateral']
            self.label_col_name = "SmallCategories"


        le = preprocessing.LabelEncoder()
        le.fit(label_names)
        self.label_mapping = dict(zip(label_names, le.transform(label_names)))
        self.video_frames = []
        self.labels = []
        for index, row in self.df.iterrows():
            grasp_type = row[self.label_col_name]
            pt_id = row["PT_id"]

            video_file = video_path + str(pt_id) + '.pt'
            #print(video_file)
            self.video_frames.append(video_file)
            self.labels.append(grasp_type)

        self.labels = torch.as_tensor(le.transform(self.labels))


    def __len__(self):
        """
        Returns:
            int: number of rows of the csv file (not include the header).
        """
        return len(self.labels)

    def __getitem__(self, index):
        """ get a video and its label """
        video = self.video_frames[index]
        label = self.labels[index]
        if self.transform:
            video = self.transform(video)
        return video, label

    def get_label_mapping(self):
        return sorted(self.label_mapping, key=lambda k: self.label_mapping[k])


#Class to read the dataset as images
class ImageDataset(Dataset):

    def __init__(self, csv_file, video_path, sampling_mode, num_frames, categories, transform=None):
        self.df = pd.read_csv(csv_file)

        self.transform = transform
        self.num_frames = num_frames
        self.sampling_mode = sampling_mode

        self.categories = categories
        if (self.categories == "PIP"):
            label_names = ['Power', 'Precision', 'Intermediate']
            self.label_col_name = "PIP"
        elif (self.categories == "BigCategories"):
            label_names = ['pre-pinch', 'pre-prismatic', 'pow-prismatic', 'circular', 'int-lateral']
            self.label_col_name = "BigCategories"
        else:
            label_names = ['pre-pinch', 'pre-prismatic', 'pre-circular', 'pow-cylindric', 'pow-oblique', 'pow-circular',
                           'int-lateral']
            self.label_col_name = "SmallCategories"

        le = preprocessing.LabelEncoder()
        le.fit(label_names)
        self.label_mapping = dict(zip(label_names, le.transform(label_names)))
        self.video_frames = []
        self.labels = []

        for index, row in self.df.iterrows():
            video_file = video_path + row["Video"]
            start_frame = row["StartFrame"]
            end_frame = row["EndFrame"]
            grasp_type = row[self.label_col_name]

            frame_ids = utils.sample_frames(self.sampling_mode, start_frame, end_frame, self.num_frames)
            for i in frame_ids:
                self.video_frames.append({"video_file": video_file, "frame_ids": i})
            for i in range(num_frames):
                self.labels.append(grasp_type)

        print(self.video_frames)
        self.labels = torch.as_tensor(le.transform(self.labels)).type(torch.LongTensor)
        print(self.labels)


    def __len__(self):
        """
        Returns:
            int: number of rows of the csv file (not include the header).
        """
        return len(self.labels)

    def __getitem__(self, index):
        """ get a video and its label """
        video = self.video_frames[index]
        label = self.labels[index]
        if self.transform:
            video = self.transform(video)
        return video, label

    def get_label_mapping(self):
        return sorted(self.label_mapping, key=lambda k: self.label_mapping[k])

#Class to read the transformed video dataset as images
class TransformedImageDataset(Dataset):

    def __init__(self, csv_file, video_path, sampling_mode, num_frames, categories, transform=None):
        self.df = pd.read_csv(csv_file)

        self.transform = transform
        self.num_frames = num_frames


        self.categories = categories
        if (self.categories == "PIP"):
            label_names = ['Power', 'Precision', 'Intermediate']
            self.label_col_name = "PIP"
        elif (self.categories == "BigCategories"):
            label_names = ['pre-pinch', 'pre-prismatic', 'pow-prismatic', 'circular', 'int-lateral']
            self.label_col_name = "BigCategories"
        else:
            label_names = ['pre-pinch', 'pre-prismatic', 'pre-circular', 'pow-cylindric', 'pow-oblique', 'pow-circular',
                           'int-lateral']
            self.label_col_name = "SmallCategories"


        le = preprocessing.LabelEncoder()
        le.fit(label_names)
        self.label_mapping = dict(zip(label_names, le.transform(label_names)))
        self.video_frames = []
        self.labels = []
        for index, row in self.df.iterrows():
            grasp_type = row[self.label_col_name]
            pt_id = row["PT_id"]

            video_file = video_path + str(pt_id) + '.pt'

            v = [{"video_file": video_file, "frame_ids": x} for x in range(num_frames)]
            l = [grasp_type]*num_frames
            self.video_frames.extend(v)
            self.labels.extend(l)

        print(self.video_frames)
        self.labels = torch.as_tensor(le.transform(self.labels))


    def __len__(self):
        """
        Returns:
            int: number of rows of the csv file (not include the header).
        """
        return len(self.labels)

    def __getitem__(self, index):
        """ get a video and its label """
        video = self.video_frames[index]
        label = self.labels[index]
        if self.transform:
            video = self.transform(video)
        return video, label

    def get_label_mapping(self):
        return sorted(self.label_mapping, key=lambda k: self.label_mapping[k])


# Class to generate the transformed Dataset.
class GenerateTransformedDataset(Dataset):

    def __init__(self, read_csv, save_csv, video_path, num_frames, num_aug, transform=None):
        self.df = pd.read_csv(read_csv)

        self.transform = transform
        self.num_frames = num_frames
        self.num_aug = num_aug

        new_df = pd.DataFrame(
            columns=["PT_id", "Video", "StartFrame", "EndFrame", "Frames", "PIP", "SmallCategories", "BigCategories"])
        self.video_frames = []
        PT_id = 0
        for index, row in self.df.iterrows():
            video_file = video_path + row["Video"]
            start_frame = row["StartFrame"]
            end_frame = row["EndFrame"]

            small_type = row["SmallCategories"]
            big_type = row["BigCategories"]
            pip_type = row["PIP"]

            if (end_frame - start_frame >= num_frames):

                for i in range(num_aug):
                    #if (i < (num_aug)):
                    #    frame_ids = utils.sample_frames("random", start_frame, end_frame, self.num_frames)
                    #else:
                    frame_ids = utils.sample_frames("uniform", start_frame, end_frame, self.num_frames)
                        # print(i, " " , len(frame_ids), " " ,start_frame, " ", end_frame, " ", frame_ids)

                    self.video_frames.append({"video_file": video_file, "frame_ids": frame_ids})
                    values_to_add = {"PT_id": PT_id, "Video": video_file, "StartFrame": start_frame,
                                     "EndFrame": end_frame, "Frames": frame_ids, "PIP": pip_type,
                                     "SmallCategories": small_type, "BigCategories": big_type}
                    new_df = new_df.append(values_to_add, ignore_index=True)
                    print(values_to_add)
                    PT_id += 1

        new_df.to_csv(save_csv)

    def __len__(self):
        """
        Returns:
            int: number of rows of the csv file (not include the header).
        """
        return len(self.video_frames)

    def __getitem__(self, index):
        """ get a video """
        video = self.video_frames[index]
        if self.transform:
            video = self.transform(video)
        return video
