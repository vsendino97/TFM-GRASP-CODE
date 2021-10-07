import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np


class MidasModel():
    def __init__(self, model_type):
        if (model_type == "DPT_Large" or model_type == "DPT_Hybrid" or  model_type == "MiDaS_small"):
            self.model_type = model_type
            self.midas = torch.hub.load("intel-isl/MiDaS", self.model_type)

            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

            if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
                self.transform = midas_transforms.dpt_transform
            else:
                self.transform = midas_transforms.small_transform

            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            self.midas.to(self.device)
            self.midas.eval()

        else:
            print("Error")

    def predict_batch(self, img):
        input_batch = self.transform(img).to(self.device)

        with torch.no_grad():
            prediction = self.midas(input_batch)


            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            )#.squeeze()


            return prediction.cpu().numpy()


def black_image(im, depth, limit):
    for h in range(depth.shape[0]):
        for w in range(depth.shape[1]):
            if (depth[h][w] < limit):
                im[h,w,:] = 0.0


if __name__ == '__main__':

    model_type = "DPT_Large"
    midas = MidasModel(model_type)

    #filename = "/home/elena/Videos/Yale-Grasp-Dataset/040.mp4"
    #filename = "/home/elena/Videos/Videos-GoPro/GH012471.MP4"
    filename = "/home/elena/Downloads/subject_1_depth_seg_1.mj2"
    #filename2 = "/home/elena/PycharmProjects/GRASP/Gopro/subject_1_gopro_seg_1.mp4"

    cap = cv2.VideoCapture(filename)
    num_frames = 117
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #frames = np.empty([8, 1, height, width])
    assert (cap.isOpened())

    for index in range(1000):
        # read frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, num_frames)
        ret, frame = cap.read()
        if ret:
            orig_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            """
            output = midas.predict_batch(orig_img).squeeze()
            #frames[index,:,:,:] = output
            formatted = (output * 255 / np.max(output)).astype('uint8')

            bl_output = orig_img.copy()
            black_image(bl_output, formatted, 120.0)

            _, axs = plt.subplots(2, 2, figsize=(128, 128))
            axs = axs.flatten()
            for i, ax in zip([orig_img, formatted, bl_output], axs):
                ax.imshow(i)
            """
            plt.imshow(orig_img)
            plt.show()
        num_frames += 1



