import argparse
import time
import os
import cv2 as cv
import numpy as np
from pathlib import Path
from Point_detect import Points
from lib.utils.visualization import draw_points_and_skeleton,joints_dict

def image_detect(opt):
    skeleton = joints_dict()['coco']['skeleton']
    hrnet_model = Points(model_name='hrnet', opt=opt)

    img0 = cv.imread(opt.source)
    frame = img0.copy()
   #predict
    pred = hrnet_model.predict(img0)
   #vis
    for i, pt in enumerate(pred):
        frame = draw_points_and_skeleton(frame, pt, skeleton)

    #save
    cv.imwrite('test_result.jpg', frame)


def video_detect(opt):

    hrnet_model = Points(model_name='hrnet', opt=opt,resolution=(384,288))
    skeleton = joints_dict()['coco']['skeleton']

    cap = cv.VideoCapture(opt.source)
    if opt.save_video:
        fourcc = cv.VideoWriter_fourcc(*'MJPG')
        out = cv.VideoWriter('data/runs/{}_out.avi'.format(os.path.basename(opt.source).split('.')[0]), fourcc, 24, (int(cap.get(3)), int(cap.get(4))))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        pred = hrnet_model.predict(frame)
        for pt in pred:
            frame = draw_points_and_skeleton(frame,pt,skeleton)
        if opt.show:
            cv.imshow('result', frame)
        if opt.save_video:
            out.write(frame)
        if cv.waitKey(1) == 27:
            break
    out.release()
    cap.release()
    cv.destroyAllWindows()
# video_detect(0)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default=0, help='source')  # file/folder, 0 for webcam
    parser.add_argument('--detect_weight', type=str, default="./yolov5/weights/yolov5x.pt", help='e.g "./yolov5/weights/yolov5x.pt"')
    parser.add_argument('--save_video', action='store_true', default=False,help='save results to *.avi')
    parser.add_argument('--show', action='store_true', default=True, help='save results to *.avi')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all model')
    parser.add_argument('--project', default='data/runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    video_detect(opt)
