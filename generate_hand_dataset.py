import argparse

import cv2
import pandas as pd

from model import handDetector

def main(csv_file, video_path, save_dir, output_path):
    hand_det = handDetector()

    df = pd.read_csv(csv_file)
    #df = df[df["Video"] == "040.mp4"]
    columns = ['Grasp', 'Video', 'Frame', 'PIP', 'BigCategories', 'SmallCategories', 'Filename','Subject','GraspID']
    grasps = []
    video_files = []
    frames = []
    pips = []
    big_cat = []
    small_cat = []
    filenames = []
    subjects = []
    grasp_ids = []

    new_df = pd.DataFrame(columns=columns)

    prev_video = ""
    cap = None
    for index, row in df.iterrows():
        video_file = video_path + str(row["Video"])
        start_frame = row["StartFrame"]
        end_frame = row["EndFrame"]


        if (prev_video != video_file):
            if (cap != None):
                cap.release()
            cap = cv2.VideoCapture(video_file)
            assert (cap.isOpened())
            fps = cap.get(cv2.CAP_PROP_FPS)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            prev_video = video_file

        for i in range(start_frame, end_frame,15):
            # read frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                pr_im = hand_det.predict_hands(frame)
                if (pr_im is not None):
                    filename = output_path+ str(row["Video"]) +"_"+str(i)+".png"
                    cv2.imwrite(filename, pr_im)
                    #cv2.imshow("Image", pr_im)
                    #cv2.waitKey(1)

                    grasps.append(row["Grasp"])
                    video_files.append(row["Video"])
                    frames.append(i)
                    pips.append(row["PIP"])
                    big_cat.append(row["BigCategories"])
                    small_cat.append(row["SmallCategories"])
                    filenames.append(filename)
                    subjects.append(row["Subject"])
                    grasp_ids.append(row["ID"])
        print(video_file)

    cap.release()
    new_df["Grasp"] = grasps
    new_df["Video"] = video_files
    new_df["Frame"] = frames
    new_df["PIP"] = pips
    new_df["BigCategories"] = big_cat
    new_df["SmallCategories"] = small_cat
    new_df["Filename"] = filenames
    new_df["Subject"] = subjects
    new_df["GraspID"] = grasp_ids

    new_df.index.names = ['ID']
    new_df.to_csv(save_dir)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to preprocess the labels from the Yale dataset.')
    parser.add_argument('-c', '--csv_file', default="data/yale_dataset/yale_train.csv", type=str,
                      help='Path to the csv with the original labels')
    parser.add_argument('-v', '--video_path', default="/home/victor.sendinog/data-local/Yale-Grasp-Dataset/", type=str,
                      help='Path to the Yale dataset videos')
    parser.add_argument('-s', '--save_dir', default="data/yale_dataset/yale_hand_labels2.csv", type=str,
                        help='Path to save the csv refering to the extracted hands')
    parser.add_argument('-o', '--output_path', default="/home/victor.sendinog/data-local/hands2/", type=str,
                        help='Path to save the extracted hands')


    args = parser.parse_args()

    csv_file = args.csv_file
    video_path = args.video_path
    save_dir = args.save_dir
    output_path = args.output_path

    main(csv_file, video_path, save_dir, output_path)
