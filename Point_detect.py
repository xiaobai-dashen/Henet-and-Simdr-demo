import cv2
import numpy as np
import torch
from torchvision.transforms import transforms
import torch.nn.functional as F
from lib.config import cfg

from yolov5.YOLOv5 import Yolov5
from lib.utils.transforms import  flip_back_simdr,transform_preds,get_affine_transform
from lib import models
import argparse
import sys
sys.path.insert(0, 'D:\\Study\\SimDR\\yolov5')


class Points():
    def __init__(self,
                 model_name='sa-simdr',
                 resolution=(384,288),
                 opt=None
                ):
        """
        Initializes a new SimpleHRNet object.
        HRNet (and YOLOv3) are initialized on the torch.device("device") and
        its (their) pre-trained weights will be loaded from disk.

        Args:
            c (int): number of channels (when using HRNet model) or resnet size (when using PoseResNet model).
            nof_joints (int): number of joints.
            checkpoint_path (str): path to an official hrnet checkpoint or a checkpoint obtained with `train_coco.py`.
            model_name (str): model name (HRNet or PoseResNet).
                Valid names for HRNet are: `HRNet`, `hrnet`
                Valid names for PoseResNet are: `PoseResNet`, `poseresnet`, `ResNet`, `resnet`
                Default: "HRNet"
            resolution (tuple): hrnet input resolution - format: (height, width).
                Default: (384, 288)
            interpolation (int): opencv interpolation algorithm.
                Default: cv2.INTER_CUBIC
            multiperson (bool): if True, multiperson detection will be enabled.
                This requires the use of a people detector (like YOLOv3).
                Default: True
            return_heatmaps (bool): if True, heatmaps will be returned along with poses by self.predict.
                Default: False
            return_bounding_boxes (bool): if True, bounding boxes will be returned along with poses by self.predict.
                Default: False
            max_batch_size (int): maximum batch size used in hrnet inference.
                Useless without multiperson=True.
                Default: 16
            yolo_model_def (str): path to yolo model definition file.
                Default: "./model/detectors/yolo/config/yolov3.cfg"
            yolo_class_path (str): path to yolo class definition file.
                Default: "./model/detectors/yolo/data/coco.names"
            yolo_weights_path (str): path to yolo pretrained weights file.
                Default: "./model/detectors/yolo/weights/yolov3.weights.cfg"
            device (:class:`torch.device`): the hrnet (and yolo) inference will be run on this device.
                Default: torch.device("cpu")
        """
        self.model_name = model_name
        self.resolution = resolution  # in the form (height, width) as in the original implementation
        self.aspect_ratio = resolution[1]/resolution[0]
        self.yolo_weights_path = opt.detect_weight
        self.flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8],
                           [9, 10], [11, 12], [13, 14], [15, 16]]
        self.device = torch.device(opt.device)
        cfg.defrost()
        if model_name in ('sa-simdr','sasimdr','sa_simdr'):
            if resolution ==(384,288):
                cfg.merge_from_file('./experiments/coco/hrnet/sa_simdr/w48_384x288_adam_lr1e-3_split1_5_sigma4.yaml')
            elif resolution == (256,192):
                cfg.merge_from_file('./experiments/coco/hrnet/sa_simdr/w48_256x192_adam_lr1e-3_split2_sigma4.yaml')
            else:
                raise ValueError('Wrong cfg file')
        elif model_name in ('simdr'):
                if resolution == (384, 288):
                    cfg.merge_from_file('./experiments/coco/hrnet/simdr/nmt_w48_256x192_adam_lr1e-3.yaml')
                elif resolution == (256, 192):
                    cfg.merge_from_file('./experiments/coco/hrnet/simdr/nmt_w48_256x192_adam_lr1e-3.yaml')
                else:
                    raise ValueError('Wrong cfg file')
        elif model_name in ('hrnet','HRnet','Hrnet'):
            if resolution == (384,288):
                cfg.merge_from_file('./experiments/coco/hrnet/heatmap/w48_384x288_adam_lr1e-3.yaml')
            elif resolution == (256,192):
                cfg.merge_from_file('./experiments/coco/hrnet/heatmap/w48_256x192_adam_lr1e-3.yaml')
            else:
                raise ValueError('Wrong cfg file')
        else:
            raise ValueError('Wrong model name.')
        cfg.freeze()
        self.model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
            cfg, is_train=False)

        print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        checkpoint_path = cfg.TEST.MODEL_FILE
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint)

        if 'cuda' in str(self.device):
            print("device: 'cuda' - ", end="")

            if 'cuda' == str(self.device):
                # if device is set to 'cuda', all available GPUs will be used
                print("%d GPU(s) will be used" % torch.cuda.device_count())
                device_ids = None
            else:
                # if device is set to 'cuda:IDS', only that/those device(s) will be used
                print("GPU(s) '%s' will be used" % str(self.device))
                device_ids = [int(x) for x in str(self.device)[5:].split(',')]
        elif 'cpu' == str(self.device):
            print("device: 'cpu'")
        else:
            raise ValueError('Wrong device name.')

        self.model = self.model.to(self.device)
        self.model.eval()
        self.detector = Yolov5(
                               weights=self.yolo_weights_path,
                               opt=opt ,
                               device=self.device)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.resolution[0], self.resolution[1])),  # (height, width)
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / 200, h * 1.0 / 200],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale
    def predict(self, image):
        """
        Predicts the human pose on a single image or a stack of n images.

        Args:
            image (:class:`np.ndarray`):
                the image(s) on which the human pose will be estimated.

                image is expected to be in the opencv format.
                image can be:
                    - a single image with shape=(height, width, BGR color channel)
                    - a stack of n images with shape=(n, height, width, BGR color channel)

        Returns:
            :class:`np.ndarray` or list:
                a numpy array containing human joints for each (detected) person.

                Format:
                    if image is a single image:
                        shape=(# of people, # of joints (nof_joints), 3);  dtype=(np.float32).
                    if image is a stack of n images:
                        list of n np.ndarrays with
                        shape=(# of people, # of joints (nof_joints), 3);  dtype=(np.float32).

                Each joint has 3 values: (y position, x position, joint confidence).

                If self.return_heatmaps, the class returns a list with (heatmaps, human joints)
                If self.return_bounding_boxes, the class returns a list with (bounding boxes, human joints)
                If self.return_heatmaps and self.return_bounding_boxes, the class returns a list with
                    (heatmaps, bounding boxes, human joints)
        """
        if len(image.shape) == 3:
            return self._predict_single(image)
        else:
            raise ValueError('Wrong image format.')

    def sa_simdr_pts(self,img,detection,images,boxes):
        c, s = [], []
        if detection is not None:
            for i, (x1, y1, x2, y2) in enumerate(detection):
                x1 = int(round(x1.item()))
                x2 = int(round(x2.item()))
                y1 = int(round(y1.item()))
                y2 = int(round(y2.item()))
                boxes[i] = [x1, y1, x2, y2]
                w, h = x2 - x1, y2 - y1
                xx1 = np.max((0, x1))
                yy1 = np.max((0, y1))
                xx2 = np.min((img.shape[1] - 1, x1 + np.max((0, w - 1))))
                yy2 = np.min((img.shape[0] - 1, y1 + np.max((0, h - 1))))
                box = [xx1, yy1, xx2 - xx1, yy2 - yy1]
                center, scale = self._box2cs(box)
                c.append(center)
                s.append(scale)

                trans = get_affine_transform(center, scale, 0, np.array(cfg.MODEL.IMAGE_SIZE))
                input = cv2.warpAffine(
                    img,
                    trans,
                    (int(self.resolution[1]), int(self.resolution[0])),
                    flags=cv2.INTER_LINEAR)
                images[i] = self.transform(input)
            if images.shape[0] > 0:
                images = images.to(self.device)
                with torch.no_grad():
                    output_x, output_y = self.model(images)
                    if cfg.TEST.FLIP_TEST:
                        input_flipped = images.flip(3)
                        output_x_flipped_, output_y_flipped_ = self.model(input_flipped)
                        output_x_flipped = flip_back_simdr(output_x_flipped_.cpu().numpy(),
                                                           self.flip_pairs, type='x')
                        output_y_flipped = flip_back_simdr(output_y_flipped_.cpu().numpy(),
                                                           self.flip_pairs, type='y')
                        output_x_flipped = torch.from_numpy(output_x_flipped.copy()).to(self.device)
                        output_y_flipped = torch.from_numpy(output_y_flipped.copy()).to(self.device)

                        # feature is not aligned, shift flipped heatmap for higher accuracy
                        if cfg.TEST.SHIFT_HEATMAP:
                            output_x_flipped[:, :, 0:-1] = \
                                output_x_flipped.clone()[:, :, 1:]
                        output_x = F.softmax((output_x + output_x_flipped) * 0.5, dim=2)
                        output_y = F.softmax((output_y + output_y_flipped) * 0.5, dim=2)
                    else:
                        output_x = F.softmax(output_x, dim=2)
                        output_y = F.softmax(output_y, dim=2)
                    max_val_x, preds_x = output_x.max(2, keepdim=True)
                    max_val_y, preds_y = output_y.max(2, keepdim=True)

                    mask = max_val_x > max_val_y
                    max_val_x[mask] = max_val_y[mask]
                    maxvals = max_val_x.cpu().numpy()

                    output = torch.ones([images.size(0), preds_x.size(1), 2])
                    output[:, :, 0] = torch.squeeze(torch.true_divide(preds_x, cfg.MODEL.SIMDR_SPLIT_RATIO))
                    output[:, :, 1] = torch.squeeze(torch.true_divide(preds_y, cfg.MODEL.SIMDR_SPLIT_RATIO))

                    output = output.cpu().numpy()
                    preds = output.copy()
                    for i in range(output.shape[0]):
                        preds[i] = transform_preds(
                            output[i], c[i], s[i], [cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1]]
                        )
            else:
                preds = np.empty((0, 0, 2), dtype=np.float32)
        return preds
    def simdr_pts(self,img,detection,images,boxes):
        c, s = [], []
        if detection is not None:
            for i, (x1, y1, x2, y2) in enumerate(detection):
                x1 = int(round(x1.item()))
                x2 = int(round(x2.item()))
                y1 = int(round(y1.item()))
                y2 = int(round(y2.item()))
                boxes[i] = [x1, y1, x2, y2]
                w, h = x2 - x1, y2 - y1
                xx1 = np.max((0, x1))
                yy1 = np.max((0, y1))
                xx2 = np.min((img.shape[1] - 1, x1 + np.max((0, w - 1))))
                yy2 = np.min((img.shape[0] - 1, y1 + np.max((0, h - 1))))
                box = [xx1, yy1, xx2 - xx1, yy2 - yy1]
                center, scale = self._box2cs(box)
                c.append(center)
                s.append(scale)

                trans = get_affine_transform(center, scale, 0, np.array(cfg.MODEL.IMAGE_SIZE))
                input = cv2.warpAffine(
                    img,
                    trans,
                    (int(self.resolution[1]), int(self.resolution[0])),
                    flags=cv2.INTER_LINEAR)
                images[i] = self.transform(input)
            if images.shape[0] > 0:
                images = images.to(self.device)
                with torch.no_grad():
                    output_x, output_y = self.model(images)
                    if cfg.TEST.FLIP_TEST:
                        input_flipped = images.flip(3)
                        output_x_flipped_, output_y_flipped_ = self.model(input_flipped)
                        output_x_flipped = flip_back_simdr(output_x_flipped_.cpu().numpy(),
                                                           self.flip_pairs, type='x')
                        output_y_flipped = flip_back_simdr(output_y_flipped_.cpu().numpy(),
                                                           self.flip_pairs, type='y')
                        output_x_flipped = torch.from_numpy(output_x_flipped.copy()).to(self.device)
                        output_y_flipped = torch.from_numpy(output_y_flipped.copy()).to(self.device)

                        # feature is not aligned, shift flipped heatmap for higher accuracy
                        if cfg.TEST.SHIFT_HEATMAP:
                            output_x_flipped[:, :, 0:-1] = \
                                output_x_flipped.clone()[:, :, 1:]
                        output_x = (F.softmax(output_x, dim=2) + F.softmax(output_x_flipped, dim=2)) * 0.5
                        output_y = (F.softmax(output_y, dim=2) + F.softmax(output_y_flipped, dim=2)) * 0.5
                    else:
                        output_x = F.softmax(output_x, dim=2)
                        output_y = F.softmax(output_y, dim=2)
                    max_val_x, preds_x = output_x.max(2, keepdim=True)
                    max_val_y, preds_y = output_y.max(2, keepdim=True)

                    mask = max_val_x > max_val_y
                    max_val_x[mask] = max_val_y[mask]
                    maxvals = max_val_x.cpu().numpy()

                    output = torch.ones([images.size(0), preds_x.size(1), 2])
                    output[:, :, 0] = torch.squeeze(torch.true_divide(preds_x, cfg.MODEL.SIMDR_SPLIT_RATIO))
                    output[:, :, 1] = torch.squeeze(torch.true_divide(preds_y, cfg.MODEL.SIMDR_SPLIT_RATIO))

                    output = output.cpu().numpy()
                    preds = output.copy()
                    for i in range(output.shape[0]):
                        preds[i] = transform_preds(
                            output[i], c[i], s[i], [cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1]]
                        )
            else:
                preds = np.empty((0, 0, 2), dtype=np.float32)
        return preds
    def hrnet_pts(self,img,detection,images,boxes):
        if detection is not None:
            for i, (x1, y1, x2, y2) in enumerate(detection):
                x1 = int(round(x1.item()))
                x2 = int(round(x2.item()))
                y1 = int(round(y1.item()))
                y2 = int(round(y2.item()))

                # Adapt detections to match HRNet input aspect ratio (as suggested by xtyDoge in issue #14)
                correction_factor = self.resolution[0] / self.resolution[1] * (x2 - x1) / (y2 - y1)
                if correction_factor > 1:
                    # increase y side
                    center = y1 + (y2 - y1) // 2
                    length = int(round((y2 - y1) * correction_factor))
                    y1 = max(0, center - length // 2)
                    y2 = min(img.shape[0], center + length // 2)
                elif correction_factor < 1:
                    # increase x side
                    center = x1 + (x2 - x1) // 2
                    length = int(round((x2 - x1) * 1 / correction_factor))
                    x1 = max(0, center - length // 2)
                    x2 = min(img.shape[1], center + length // 2)

                boxes[i] = [x1, y1, x2, y2]
                images[i] = self.transform(img[y1:y2, x1:x2, ::-1])

        if images.shape[0] > 0:
            images = images.to(self.device)

            with torch.no_grad():
                out = self.model(images)

                out = out.detach().cpu().numpy()
                pts = np.empty((out.shape[0], out.shape[1], 2), dtype=np.float32)
                # For each human, for each joint: y, x, confidence
                for i, human in enumerate(out):
                    for j, joint in enumerate(human):
                        pt = np.unravel_index(np.argmax(joint), (self.resolution[0] // 4, self.resolution[1] // 4))
                        # 0: pt_x / (height // 4) * (bb_y2 - bb_y1) + bb_y1
                        # 1: pt_y / (width // 4) * (bb_x2 - bb_x1) + bb_x1
                        # 2: confidences
                        pts[i, j, 0] = pt[1] * 1. / (self.resolution[1] // 4) * (boxes[i][2] - boxes[i][0]) + boxes[i][0]
                        pts[i, j, 1] = pt[0] * 1. / (self.resolution[0] // 4) * (boxes[i][3] - boxes[i][1]) + boxes[i][1]

        else:
            pts = np.empty((0, 0, 2), dtype=np.float32)

        return pts

    def _predict_single(self, image):

        _,detections = self.detector.detect(image)

        nof_people = len(detections) if detections is not None else 0
        boxes = np.empty((nof_people, 4), dtype=np.int32)
        images = torch.empty((nof_people, 3, self.resolution[0], self.resolution[1]))  # (height, width)
        if self.model_name in ('sa-simdr','sasimdr'):
            pts=self.sa_simdr_pts(image,detections,images,boxes)
        elif self.model_name in ('hrnet','HRnet','hrnet'):
            pts = self.hrnet_pts(image, detections, images, boxes)
        elif self.model_name in ('simdr'):
            pts = self.simdr_pts(image, detections, images, boxes)

        return pts