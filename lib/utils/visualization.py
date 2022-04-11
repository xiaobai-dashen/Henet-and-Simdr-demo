import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import ffmpeg
import random
import math
import copy
def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img

def joints_dict():
    joints = {
        "coco": {
            "keypoints": {
                0: "nose",
                1: "left_eye",
                2: "right_eye",
                3: "left_ear",
                4: "right_ear",
                5: "left_shoulder",
                6: "right_shoulder",
                7: "left_elbow",
                8: "right_elbow",
                9: "left_wrist",
                10: "right_wrist",
                11: "left_hip",
                12: "right_hip",
                13: "left_knee",
                14: "right_knee",
                15: "left_ankle",
                16: "right_ankle"
            },
            "skeleton": [
                # # [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8],
                # # [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]
                # [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7],
                # [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]
                # [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7],
                # [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],  # [3, 5], [4, 6]
                # [0, 5], [0, 6]
                [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7],
                [6, 8], [7, 9], [8, 10],  [0, 3], [0, 4], [1, 3], [2, 4],  # [3, 5], [4, 6]
                [0, 5], [0, 6]
            ]
        },
        "mpii": {
            "keypoints": {
                0: "right_ankle",
                1: "right_knee",
                2: "right_hip",
                3: "left_hip",
                4: "left_knee",
                5: "left_ankle",
                6: "pelvis",
                7: "thorax",
                8: "upper_neck",
                9: "head top",
                10: "right_wrist",
                11: "right_elbow",
                12: "right_shoulder",
                13: "left_shoulder",
                14: "left_elbow",
                15: "left_wrist"
            },
            "skeleton": [
                # [5, 4], [4, 3], [0, 1], [1, 2], [3, 2], [13, 3], [12, 2], [13, 12], [13, 14],
                # [12, 11], [14, 15], [11, 10], # [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]
                [5, 4], [4, 3], [0, 1], [1, 2], [3, 2], [3, 6], [2, 6], [6, 7], [7, 8], [8, 9],
                [13, 7], [12, 7], [13, 14], [12, 11], [14, 15], [11, 10],
            ]
        },
    }
    return joints


def draw_points(image, points, color_palette='tab20', palette_samples=16, confidence_threshold=0.5,color=None):
    """
    Draws `points` on `image`.

    Args:
        image: image in opencv format
        points: list of points to be drawn.
            Shape: (nof_points, 3)
            Format: each point should contain (y, x, confidence)
        color_palette: name of a matplotlib color palette
            Default: 'tab20'
        palette_samples: number of different colors sampled from the `color_palette`
            Default: 16
        confidence_threshold: only points with a confidence higher than this threshold will be drawn. Range: [0, 1]
            Default: 0.5

    Returns:
        A new image with overlaid points

    """
    try:
        colors = np.round(
            np.array(plt.get_cmap(color_palette).colors) * 255
        ).astype(np.uint8)[:, ::-1].tolist()
    except AttributeError:  # if palette has not pre-defined colors
        colors = np.round(
            np.array(plt.get_cmap(color_palette)(np.linspace(0, 1, palette_samples))) * 255
        ).astype(np.uint8)[:, -2::-1].tolist()

    circle_size = max(1, min(image.shape[:2]) // 80)  # ToDo Shape it taking into account the size of the detection
    # circle_size = max(2, int(np.sqrt(np.max(np.max(points, axis=0) - np.min(points, axis=0)) // 16)))
    for i, pt in enumerate(points):
        if i in [1,2,3,4]:
            continue

        # image = cv2.circle(image, (int(pt[0]), int(pt[1])), circle_size, tuple(colors[i % len(colors)]), -1 , lineType= cv2.LINE_AA)
        image = cv2.circle(image, (int(pt[0]), int(pt[1])), circle_size, tuple(colors[i % len(colors)]), -1 , lineType= cv2.LINE_AA)
        # image = cv2.circle(image, (int(pt[0]), int(pt[1])), circle_size, color, -1 , lineType= cv2.LINE_AA)

    return image


def draw_skeleton(image, points, skeleton, color_palette='Set2', palette_samples=8, person_index=0,
                  confidence_threshold=0.5,sk_color=None):
    """
    Draws a `skeleton` on `image`.

    Args:
        image: image in opencv format
        points: list of points to be drawn.
            Shape: (nof_points, 3)
            Format: each point should contain (y, x, confidence)
        skeleton: list of joints to be drawn
            Shape: (nof_joints, 2)
            Format: each joint should contain (point_a, point_b) where `point_a` and `point_b` are an index in `points`
        color_palette: name of a matplotlib color palette
            Default: 'Set2'
        palette_samples: number of different colors sampled from the `color_palette`
            Default: 8
        person_index: index of the person in `image`
            Default: 0
        confidence_threshold: only points with a confidence higher than this threshold will be drawn. Range: [0, 1]
            Default: 0.5

    Returns:
        A new image with overlaid joints

    """
    try:
        colors = np.round(
            np.array(plt.get_cmap(color_palette).colors) * 255
        ).astype(np.uint8)[:, ::-1].tolist()
    except AttributeError:  # if palette has not pre-defined colors
        colors = np.round(
            np.array(plt.get_cmap(color_palette)(np.linspace(0, 1, palette_samples))) * 255
        ).astype(np.uint8)[:, -2::-1].tolist()
    colors1 = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    canvas = copy.deepcopy(image)
    cur_canvas = canvas.copy()
    for i, joint in enumerate(skeleton):

        if joint == [0,5] or joint == [0,6] :
            pt1, pt2 ,pt3= points[[0,5,6]]

            pt2[[0,1]]= [(pt2[0]+pt3[0])/2,(pt2[1]+pt3[1])/2]
        else:
            pt1, pt2 = points[joint]


        # image = cv2.line(
        #     image, (int(pt1[1]), int(pt1[0])), (int(pt2[1]), int(pt2[0])),
        #     tuple(colors[person_index % len(colors)]), 8,lineType=cv2.LINE_AA
        # )

        length = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(pt1[1] - pt2[1],pt1[0] - pt2[0]))
        polygon = cv2.ellipse2Poly((int(np.mean((pt1[0],pt2[0]))), int(np.mean((pt1[1],pt2[1])))), (int(length / 2), 5), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(cur_canvas, polygon, colors1[i],lineType=cv2.LINE_AA)
        # cv2.fillConvexPoly(cur_canvas, polygon, sk_color,lineType=cv2.LINE_AA)
        canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

    return canvas


def draw_points_and_skeleton(image, points, skeleton, points_color_palette='tab20', points_palette_samples=16,
                             skeleton_color_palette='Set2', skeleton_palette_samples=8, person_index=0,
                             confidence_threshold=0.5,color=None,sk_color=None):
    """
    Draws `points` and `skeleton` on `image`.

    Args:
        image: image in opencv format
        points: list of points to be drawn.
            Shape: (nof_points, 3)
            Format: each point should contain (y, x, confidence)
        skeleton: list of joints to be drawn
            Shape: (nof_joints, 2)
            Format: each joint should contain (point_a, point_b) where `point_a` and `point_b` are an index in `points`
        points_color_palette: name of a matplotlib color palette
            Default: 'tab20'
        points_palette_samples: number of different colors sampled from the `color_palette`
            Default: 16
        skeleton_color_palette: name of a matplotlib color palette
            Default: 'Set2'
        skeleton_palette_samples: number of different colors sampled from the `color_palette`
            Default: 8
        person_index: index of the person in `image`
            Default: 0
        confidence_threshold: only points with a confidence higher than this threshold will be drawn. Range: [0, 1]
            Default: 0.5

    Returns:
        A new image with overlaid joints

    """
    image = draw_skeleton(image, points, skeleton, color_palette=skeleton_color_palette,
                          palette_samples=skeleton_palette_samples, person_index=person_index,
                          confidence_threshold=confidence_threshold,sk_color=sk_color)
    image = draw_points(image, points, color_palette=points_color_palette, palette_samples=points_palette_samples,
                        confidence_threshold=confidence_threshold,color=color)
    return image


def save_images(images, target, joint_target, output, joint_output, joint_visibility, summary_writer=None, step=0,
                prefix=''):
    """
    Creates a grid of images with gt joints and a grid with predicted joints.
    This is a basic function for debugging purposes only.

    If summary_writer is not None, the grid will be written in that SummaryWriter with name "{prefix}_images" and
    "{prefix}_predictions".

    Args:
        images (torch.Tensor): a tensor of images with shape (batch x channels x height x width).
        target (torch.Tensor): a tensor of gt heatmaps with shape (batch x channels x height x width).
        joint_target (torch.Tensor): a tensor of gt joints with shape (batch x joints x 2).
        output (torch.Tensor): a tensor of predicted heatmaps with shape (batch x channels x height x width).
        joint_output (torch.Tensor): a tensor of predicted joints with shape (batch x joints x 2).
        joint_visibility (torch.Tensor): a tensor of joint visibility with shape (batch x joints).
        summary_writer (tb.SummaryWriter): a SummaryWriter where write the grids.
            Default: None
        step (int): summary_writer step.
            Default: 0
        prefix (str): summary_writer name prefix.
            Default: ""

    Returns:
        A pair of images which are built from torchvision.utils.make_grid
    """
    # Input images with gt
    images_ok = images.detach().clone()
    images_ok[:, 0].mul_(0.229).add_(0.485)
    images_ok[:, 1].mul_(0.224).add_(0.456)
    images_ok[:, 2].mul_(0.225).add_(0.406)
    for i in range(images.shape[0]):
        joints = joint_target[i] * 4.
        joints_vis = joint_visibility[i]

        for joint, joint_vis in zip(joints, joints_vis):
            if joint_vis[0]:
                a = int(joint[1].item())
                b = int(joint[0].item())
                # images_ok[i][:, a-1:a+1, b-1:b+1] = torch.tensor([1, 0, 0])
                images_ok[i][0, a - 1:a + 1, b - 1:b + 1] = 1
                images_ok[i][1:, a - 1:a + 1, b - 1:b + 1] = 0
    grid_gt = torchvision.utils.make_grid(images_ok, nrow=int(images_ok.shape[0] ** 0.5), padding=2, normalize=False)
    if summary_writer is not None:
        summary_writer.add_image(prefix + 'images', grid_gt, global_step=step)

    # Input images with prediction
    images_ok = images.detach().clone()
    images_ok[:, 0].mul_(0.229).add_(0.485)
    images_ok[:, 1].mul_(0.224).add_(0.456)
    images_ok[:, 2].mul_(0.225).add_(0.406)
    for i in range(images.shape[0]):
        joints = joint_output[i] * 4.
        joints_vis = joint_visibility[i]

        for joint, joint_vis in zip(joints, joints_vis):
            if joint_vis[0]:
                a = int(joint[1].item())
                b = int(joint[0].item())
                # images_ok[i][:, a-1:a+1, b-1:b+1] = torch.tensor([1, 0, 0])
                images_ok[i][0, a - 1:a + 1, b - 1:b + 1] = 1
                images_ok[i][1:, a - 1:a + 1, b - 1:b + 1] = 0
    grid_pred = torchvision.utils.make_grid(images_ok, nrow=int(images_ok.shape[0] ** 0.5), padding=2, normalize=False)
    if summary_writer is not None:
        summary_writer.add_image(prefix + 'predictions', grid_pred, global_step=step)

    # Heatmaps
    # ToDo
    # for h in range(0,17):
    #     heatmap = torchvision.utils.make_grid(output[h].detach(), nrow=int(np.sqrt(output.shape[0])),
    #                                            padding=2, normalize=True, range=(0, 1))
    #     summary_writer.add_image('train_heatmap_%d' % h, heatmap, global_step=step + epoch*len_dl_train)

    return grid_gt, grid_pred


def check_video_rotation(filename):
    # thanks to
    # https://stackoverflow.com/questions/53097092/frame-from-video-is-upside-down-after-extracting/55747773#55747773

    # this returns meta-data of the video file in form of a dictionary
    meta_dict = ffmpeg.probe(filename)

    # from the dictionary, meta_dict['streams'][0]['tags']['rotate'] is the key
    # we are looking for
    rotation_code = None
    try:
        if int(meta_dict['streams'][0]['tags']['rotate']) == 90:
            rotation_code = cv2.ROTATE_90_CLOCKWISE
        elif int(meta_dict['streams'][0]['tags']['rotate']) == 180:
            rotation_code = cv2.ROTATE_180
        elif int(meta_dict['streams'][0]['tags']['rotate']) == 270:
            rotation_code = cv2.ROTATE_90_COUNTERCLOCKWISE
        else:
            raise ValueError
    except KeyError:
        pass

    return rotation_code
