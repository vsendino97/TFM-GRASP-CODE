import cv2
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import random

def _seconds(value):
    if ("." in value):
        m, s_ms = value.split(':')
        s, ms = s_ms.split('.')
        return float(m) * 60 + float(s) + float(ms)*0.1
    else:
        h, m, s = value.split(':')
        return float(h) * 3600 + float(m) * 60 + float(s)

def _timecode(seconds):
    return '{h:02d}:{m:02d}:{s:02d}:{f:02d}' \
        .format(h=int(seconds / 3600),
                m=int(seconds / 60 % 60),
                s=int(seconds % 60),
                f=round((seconds - int(seconds)) * fps))


def _frames(seconds):
    return seconds * fps

def frames_to_timecode(frames, start=None):
    return _timecode(_seconds(frames) + _seconds(start))

def timecode_to_frames(timecode):
    return _frames(_seconds(timecode))


if __name__ == '__main__':
    random.seed(42)  # Create a fixed state
    df = pd.read_csv("../data/yale_dataset/yale_original_labels.csv")
    columns = ['Grasp', 'Video', 'StartFrame', 'EndFrame', 'PIP', 'Subject', 'Duration', 'BigCategories', 'SmallCategories']

    new_df = pd.DataFrame(columns=columns)

    dirname = os.path.dirname(__file__)
    video_path = os.path.join(dirname, "../video/YaleDataset/")
    print("Dataset path: ", video_path)

    plot = False
    """Select the subset of data for the statistics"""
    print("Statistical properties of subset of data")
    print("-----------------------------------------")
    daux = df
    daux = daux.loc[daux['Grasp'] != 'no grasp']
    daux = daux.loc[daux['blackRatio'] == 0.0]
    daux = daux.loc[daux['Duration'] >= 0.32]
    #daux = daux.loc[daux['Duration'] < 20]

    """Statistical properties of class"""
    print("---Total size: ", len(daux.index))
    print("---Grasp composition \n", daux.groupby('Grasp').size())
    grasp_count = daux.groupby('Grasp').size()
    grasp_pcts = grasp_count.groupby(level=0).apply(lambda x: 100 * x /len(daux.index))
    print("---Grasp Percentage: \n" ,grasp_pcts)
    
    
    """Statistical properties of duration"""
    print("---DURATION:")
    dur = daux['Duration']
    if (plot):
        fig = plt.figure(figsize=(10, 7))
        plt.boxplot(dur)
        plt.show()
    print("Avg: ", dur.mean())
    print("Median: ", dur.median())
    print("1st Quantile: ", dur.quantile(0.25))
    print("3nd Quantile: ", dur.quantile(0.75))
    
    print("Grasp with duration 0s:", len(daux.loc[daux['Duration'] == 0]))
    print("Grasp over the 95 percentile:", len(daux.loc[daux['Duration'] > dur.quantile(0.95)]))


    """Statistical properties of black ratio"""
    print("---BLACK RATIO:")
    bl_ratio = daux['blackRatio']
    if (plot):
        fig = plt.figure(figsize=(10, 7))
        plt.boxplot(bl_ratio)
        plt.show()
    print("Avg: ", bl_ratio.mean())
    print("Median: ", bl_ratio.median())
    print("1st Quantile: ", bl_ratio.quantile(0.25))
    print("3nd Quantile: ", bl_ratio.quantile(0.75))
    
    print ("Grasp with black ratio == 0.0:", len(daux.loc[daux['blackRatio'] == 0]))
    print ("Grasp over the 95 percentile:", len(daux.loc[daux['blackRatio'] > bl_ratio.quantile(0.95)]))

    print("Sample of 5 rows")
    print(df.head())
    grasps = []
    video_files = []
    start_frames = []
    end_frames = []
    pips = []
    durations = []
    timestamp = []
    big_cat = []
    small_cat = []
    subject = []



    small_cat_dict = {
        "thumb-index finger": "pre-pinch",
        "tip pinch": "pre-pinch",

        "thumb-2 finger": "pre-prismatic",
        "thumb-3 finger": "pre-prismatic",
        "thumb-4 finger": "pre-prismatic",
        "writing tripod": "not-used",

        "inferior pincer": "pre-pinch",
        "tripod": "pre-circular",
        "quadpod": "pre-circular",
        "precision disk": "pre-circular",
        "precision sphere": "not-used",

        "large diameter": "pow-cylindric",
        "small diameter": "pow-cylindric",
        "medium wrap": "pow-cylindric",
        "ring": "pow-cylindric",
        "extension type": "not-used",

        "adducted thumb": "pow-oblique",
        "light tool": "pow-oblique",
        "fixed hook": "not-used",
        "palmar": "not-used",
        "index finger extension": "pow-oblique",

        "power disk": "not-used",
        "power sphere": "pow-circular",
        "sphere-3 finger": "pow-circular",
        "sphere-4 finger": "pow-circular",

        "platform": "not-used",
        "no grasp": "not-used",

        "lateral pinch": "int-lateral",
        "lateral tripod": "int-lateral",

        "stick": "not-used",
        "ventral": "not-used",
        "adduction": "pre-pinch",
        "tripod variation": "not-used",
        "parallel extension": "not-used"
    }

    big_cat_dict = {
        "thumb-index finger": "pre-prismatic",
        "tip pinch": "pre-prismatic",

        "thumb-2 finger": "pre-prismatic",
        "thumb-3 finger": "pre-prismatic",
        "thumb-4 finger": "pre-prismatic",
        "writing tripod": "not-used",

        "inferior pincer": "pre-prismatic",
        "tripod": "circular",
        "quadpod": "circular",
        "precision disk": "circular",
        "precision sphere": "not-used",

        "large diameter": "pow-prismatic",
        "small diameter": "pow-prismatic",
        "medium wrap": "pow-prismatic",
        "ring": "pow-prismatic",
        "extension type": "not-used",

        "adducted thumb": "pow-prismatic",
        "light tool": "pow-prismatic",
        "fixed hook": "not-used",
        "palmar": "not-used",
        "index finger extension": "pow-prismatic",

        "power disk": "not-used",
        "power sphere": "circular",
        "sphere-3 finger": "circular",
        "sphere-4 finger": "circular",

        "platform": "not-used",
        "no grasp": "not-used",

        "lateral pinch": "int-lateral",
        "lateral tripod": "int-lateral",

        "stick": "not-used",
        "ventral": "not-used",
        "adduction": "pre-prismatic",
        "tripod variation": "not-used",
        "parallel extension": "not-used"
    }


    for index, row in df.iterrows():
        video_file = str(row["Video"]).zfill(3) + ".mp4"
        start_time = row["TimeStamp"]
        duration = row["Duration"]

        file_path = os.path.join(video_path, video_file)
        #print(file_path)
        
        cap = cv2.VideoCapture(file_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        assert (cap.isOpened())

        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_frame = int(timecode_to_frames(start_time))
        end_frame = int(start_frame + duration * fps)
        cap.release()

        #print("Num frames: ", num_frames, " Start time ", start_time,  " Start frame ", start_frame, " End frame ", end_frame)
        if ((row['blackRatio'] == 0.0) & (row['Grasp'] != 'no grasp') & (row['Duration'] >= 0.32) & (start_frame < num_frames) & (end_frame < num_frames)):
            grasps.append(row['Grasp'])
            pips.append(row['PIP'])
            video_files.append(video_file)
            subject.append(row['Subject'])

            timestamp.append(start_time)
            durations.append(duration)
            start_frames.append(start_frame)
            end_frames.append(end_frame)

            grasp_type = row["Grasp"]
            big_cat.append(big_cat_dict[grasp_type])
            small_cat.append(small_cat_dict[grasp_type])
            """
            print("-----")
            print(video_file, " - ", start_time, " - ", duration, " - ", start_frame, " - ", end_frame, end_frame-start_frame)
            print(grasp_type, " - " , big_cat_dict[grasp_type], " - ", small_cat_dict[grasp_type])
            """
    new_df["Grasp"] = grasps
    new_df["Video"] = video_files
    new_df["StartFrame"] = start_frames
    new_df["EndFrame"] = end_frames
    new_df["PIP"] = pips
    new_df["Duration"] = durations
    new_df["Timestamp"] = timestamp
    new_df["Subject"] = subject
    new_df["BigCategories"] = big_cat
    new_df["SmallCategories"] = small_cat

    new_df.index.names = ['ID']

    print("-----------------------------------------")
    print("---Total size: ", len(new_df.index))
    print("---Big categories composition \n", new_df.groupby('BigCategories').size())
    grasp_count = new_df.groupby('BigCategories').size()
    grasp_pcts = grasp_count.groupby(level=0).apply(lambda x: 100 * x / len(new_df.index))
    print("---Big categories Percentage: \n", grasp_pcts)

    print("---Small categories composition \n", new_df.groupby('SmallCategories').size())
    grasp_count = new_df.groupby('SmallCategories').size()
    grasp_pcts = grasp_count.groupby(level=0).apply(lambda x: 100 * x / len(new_df.index))
    print("---Small categories Percentage: \n", grasp_pcts)

    print("-----------------------------------------")
    new_df = new_df.loc[new_df['SmallCategories'] != 'not-used']
    print("---Final number of grasps: ", len(new_df.index))

    print("---Big categories composition \n", new_df.groupby('BigCategories').size())
    grasp_count = new_df.groupby('BigCategories').size()
    grasp_pcts = grasp_count.groupby(level=0).apply(lambda x: 100 * x / len(new_df.index))
    print("---Big categories Percentage: \n", grasp_pcts)

    print("---Small categories composition \n", new_df.groupby('SmallCategories').size())
    grasp_count = new_df.groupby('SmallCategories').size()
    grasp_pcts = grasp_count.groupby(level=0).apply(lambda x: 100 * x / len(new_df.index))
    print("---Small categories Percentage: \n", grasp_pcts)


    train, test = train_test_split(new_df, test_size=0.16, random_state=42)
    print("-----------------------------------------")
    print("---TRAIN number of grasps: ", len(train.index))
    grasp_count = train.groupby('BigCategories').size()
    grasp_pcts = grasp_count.groupby(level=0).apply(lambda x: 100 * x / len(train.index))
    print("---Big categories Percentage: \n", grasp_pcts)
    grasp_count = train.groupby('SmallCategories').size()
    grasp_pcts = grasp_count.groupby(level=0).apply(lambda x: 100 * x / len(train.index))
    print("---Small categories Percentage: \n", grasp_pcts)

    print("-----------------------------------------")
    print("---TEST number of grasps: ", len(test.index))
    grasp_count = test.groupby('BigCategories').size()
    grasp_pcts = grasp_count.groupby(level=0).apply(lambda x: 100 * x / len(test.index))
    print("---Big categories Percentage: \n", grasp_pcts)
    grasp_count = test.groupby('SmallCategories').size()
    grasp_pcts = grasp_count.groupby(level=0).apply(lambda x: 100 * x / len(test.index))
    print("---Small categories Percentage: \n", grasp_pcts)


    new_df.to_csv("../data/yale_dataset/yale_final_labels.csv")
    train.to_csv("../data/yale_dataset/yale_train.csv")
    test.to_csv("../data/yale_dataset/yale_test.csv")




