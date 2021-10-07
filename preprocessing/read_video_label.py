import time
import cv2
import pandas as pd
import argparse


def main(csv_file, video_path):

    video_file = video_path.split("/")[-1]

    df = pd.read_csv(csv_file)
    df = df.loc[df['Video'] == video_file]

    font = cv2.FONT_HERSHEY_SIMPLEX
    #bottomLeftCornerOfText = (10, 500)
    bottomLeftCornerOfText = (2, 30)
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2


    vid = cv2.VideoCapture(video_path)

    cv2.namedWindow('Stream',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Stream', (800,600))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    ret,frame=vid.read()
    vw = frame.shape[1]
    vh = frame.shape[0]
    print ("Video size", vw,vh)
    outvideo = cv2.VideoWriter(video_path.replace(".mp4", "-det.mp4"),fourcc,20.0,(vw,vh))

    frames = 0
    actions = 0
    grasp_name = ""
    starttime = time.time()

    while(True):
        print(actions)
        ret, frame = vid.read()

        if (frames == df.iloc[actions]["EndFrame"]):
            actions += 1
            grasp_name = ""

        if (frames == df.iloc[actions]["StartFrame"]):
            start_frame = df.iloc[actions]["StartFrame"]
            end_frame = df.iloc[actions]["EndFrame"]
            grasp_name = df.iloc[actions]["Grasp"]

        if not ret:
            break


        frames += 1
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        cv2.putText(frame, grasp_name,
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)
        cv2.imshow('Stream', frame)
        outvideo.write(frame)
        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            break

    totaltime = time.time() - starttime
    print(frames, "frames", totaltime / frames, "s/frame")
    cv2.destroyAllWindows()
    outvideo.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to read the videos and write the labels at the same time.')
    parser.add_argument('-d', '--dataset', default="yale", type=str,
                      help='Indicate the name of the dataset: yale | nature')
    parser.add_argument('-v', '--video', default="/home/elena/Videos/Yale-Grasp-Dataset/041.mp4", type=str,
                      help='Full path to the video to be shown.')

    args = parser.parse_args()
    csv_file = ""
    if(args.dataset == "yale"):
        csv_file = "../data/yale_dataset/yale_final_labels.csv"
    elif(args.dataset == "nature"):
        csv_file = "../data/nature_dataset/nature_final_labels.csv"
    else:
        print("Invalid dataset.")
        exit()

    video_path = args.video

    print(video_path)
    print(csv_file)
    main(csv_file, video_path)