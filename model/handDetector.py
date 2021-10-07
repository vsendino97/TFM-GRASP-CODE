# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import cv2
import numpy as np
import torch
from handobj_det.faster_rcnn.resnet import resnet
# from model.nms.nms_wrapper import nms
from handobj_det.roi_layers import nms
from handobj_det.rpn.bbox_transform import bbox_transform_inv
from handobj_det.rpn.bbox_transform import clip_boxes
from handobj_det.utils.blob import im_list_to_blob
# from scipy.misc import imread
from handobj_det.utils.config import cfg
from handobj_det.utils.net_utils import vis_detections_filtered_objects_PIL  # (1) here add a function to viz

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3


class handDetector():
    def __init__(self, thresh_hand=0.5, thresh_obj=0.5):
        self.thresh_hand = thresh_hand
        self.thresh_obj = thresh_obj

        # Load model
        model_dir = os.path.abspath(os.path.dirname(__file__))
        model_dir = os.path.join(model_dir, "../handobj_det/faster_rcnn_1_8_132028.pth")
        print(model_dir)
        if not os.path.exists(model_dir):
            raise Exception('There is no input directory for loading network from ' + model_dir)

        self.pascal_classes = np.asarray(['__background__', 'targetobject', 'hand'])

        self.class_agnostic = False
        self.fasterRCNN = resnet(self.pascal_classes, 101, pretrained=False, class_agnostic=self.class_agnostic)
        self.fasterRCNN.create_architecture()

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        print("load checkpoint %s" % (model_dir))
        if self.device == "cuda":
            checkpoint = torch.load(model_dir)
        else:
            checkpoint = torch.load(model_dir, map_location=(lambda storage, loc: storage))
        self.fasterRCNN.load_state_dict(checkpoint['model'])
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']

        print('load model successfully!')




    def predict_hands(self, im_in):
        # Initilize the tensor holder here.
        im_data = torch.FloatTensor(1)
        im_info = torch.FloatTensor(1)
        num_boxes = torch.LongTensor(1)
        gt_boxes = torch.FloatTensor(1)
        box_info = torch.FloatTensor(1)

        # ship to cuda
        if self.device == "cuda":
            im_data = im_data.cuda()
            im_info = im_info.cuda()
            num_boxes = num_boxes.cuda()
            gt_boxes = gt_boxes.cuda()

        with torch.no_grad():
            if self.device == "cuda":
                cfg.CUDA = True
                self.fasterRCNN.cuda()

            self.fasterRCNN.eval()

            thresh_hand = self.thresh_hand
            thresh_obj = self.thresh_obj
            vis = False


            #print(f'image dir = {"images"}')
            #print(f'save dir = {"images_det"}')

            # bgr
            im = im_in

            blobs, im_scales = self._get_image_blob(im)
            assert len(im_scales) == 1, "Only single-image batch implemented"
            im_blob = blobs
            im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)
            im_data_pt = torch.from_numpy(im_blob)
            im_data_pt = im_data_pt.permute(0, 3, 1, 2)
            im_info_pt = torch.from_numpy(im_info_np)

            with torch.no_grad():
                im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
                im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
                gt_boxes.resize_(1, 1, 5).zero_()
                num_boxes.resize_(1).zero_()
                box_info.resize_(1, 1, 5).zero_()

                # pdb.set_trace()

            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label, loss_list = self.fasterRCNN(im_data, im_info, gt_boxes, num_boxes, box_info)

            scores = cls_prob.data
            boxes = rois.data[:, :, 1:5]

            # extract predicted params
            contact_vector = loss_list[0][0]  # hand contact state info
            offset_vector = loss_list[1][0].detach()  # offset vector (factored into a unit vector and a magnitude)
            lr_vector = loss_list[2][0].detach()  # hand side info (left/right)

            # get hand contact
            _, contact_indices = torch.max(contact_vector, 2)
            contact_indices = contact_indices.squeeze(0).unsqueeze(-1).float()

            # get hand side
            lr = torch.sigmoid(lr_vector) > 0.5
            lr = lr.squeeze(0).float()

            if cfg.TEST.BBOX_REG:
                # Apply bounding-box regression deltas
                box_deltas = bbox_pred.data
                if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                    # Optionally normalize targets by a precomputed mean and stdev
                    if self.class_agnostic:
                        if self.device == "cuda":
                            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                                cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                         + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                        else:
                            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                         + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

                        box_deltas = box_deltas.view(1, -1, 4)
                    else:
                        if self.device == "cuda":
                            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                                cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                         + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                        else:
                            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                         + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                        box_deltas = box_deltas.view(1, -1, 4 * len(self.pascal_classes))

                pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
            else:
                # Simply repeat the boxes, once for each class
                pred_boxes = np.tile(boxes, (1, scores.shape[1]))

            pred_boxes /= im_scales[0]

            scores = scores.squeeze()
            pred_boxes = pred_boxes.squeeze()

            if vis:
                im2show = np.copy(im)
            obj_dets, hand_dets = None, None
            for j in xrange(1, len(self.pascal_classes)):
                # inds = torch.nonzero(scores[:,j] > thresh).view(-1)
                if self.pascal_classes[j] == 'hand':
                    inds = torch.nonzero(scores[:, j] > thresh_hand).view(-1)
                elif self.pascal_classes[j] == 'targetobject':
                    inds = torch.nonzero(scores[:, j] > thresh_obj).view(-1)

                # if there is det
                if inds.numel() > 0:
                    cls_scores = scores[:, j][inds]
                    _, order = torch.sort(cls_scores, 0, True)
                    if self.class_agnostic:
                        cls_boxes = pred_boxes[inds, :]
                    else:
                        cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                    cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1), contact_indices[inds],
                                          offset_vector.squeeze(0)[inds], lr[inds]), 1)
                    cls_dets = cls_dets[order]
                    keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                    cls_dets = cls_dets[keep.view(-1).long()]
                    if self.pascal_classes[j] == 'targetobject':
                        obj_dets = cls_dets.cpu().numpy()
                    if self.pascal_classes[j] == 'hand':
                        hand_dets = cls_dets.cpu().numpy()


            if vis:
                # visualization
                im2show = vis_detections_filtered_objects_PIL(im2show, obj_dets, hand_dets, thresh_hand, thresh_obj)
                im2showRGB = np.array(im2show.convert('RGB'))
                im2showRGB = cv2.cvtColor(im2showRGB, cv2.COLOR_RGB2BGR)
                return im2showRGB

            else:
                if(hand_dets is not None):
                    for hand_idx, i in enumerate(range(np.minimum(10, hand_dets.shape[0]))):
                        bbox = list(int(np.round(x)) for x in hand_dets[i, :4])
                        score = hand_dets[i, 4]
                        lr = hand_dets[i, -1]
                        state = hand_dets[i, 5]


                        if (score > self.thresh_hand and lr==1.0):
                            right_hand = im_in[max(0, bbox[1]-20):min(640, bbox[3]+20), max(0,bbox[0]-20):min(640, bbox[2]+20) ]
                            return right_hand
                else:
                    return None




    def _get_image_blob(self, im):
        """Converts an image into a network input.
        Arguments:
          im (ndarray): a color image in BGR order
        Returns:
          blob (ndarray): a data blob holding an image pyramid
          im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
        """
        im_orig = im.astype(np.float32, copy=True)
        im_orig -= cfg.PIXEL_MEANS

        im_shape = im_orig.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        processed_ims = []
        im_scale_factors = []

        for target_size in cfg.TEST.SCALES:
            im_scale = float(target_size) / float(im_size_min)
            # Prevent the biggest axis from being more than MAX_SIZE
            if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
                im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
            im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                            interpolation=cv2.INTER_LINEAR)
            im_scale_factors.append(im_scale)
            processed_ims.append(im)

        # Create a blob to hold the input images
        blob = im_list_to_blob(processed_ims)

        return blob, np.array(im_scale_factors)



if __name__ == '__main__':
    hand_det = handDetector()

    filename = "/home/elena/Videos/Yale-Grasp-Dataset/040.mp4"

    cap = cv2.VideoCapture(filename)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    assert (cap.isOpened())

    for index in range(600, num_frames, 5):
        # read frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = cap.read()
        if ret:
            pr_im = hand_det.predict_hands(frame)
            print(index)

            cv2.imshow("Image", pr_im)
            cv2.waitKey(1)
