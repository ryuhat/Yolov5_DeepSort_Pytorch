# Ultralytics YOLO 🚀, GPL-3.0 license

import contextlib
import math
# from pathlib import Path
# from urllib.error import URLError

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

# from ultralytics.yolo.utils import FONT, USER_CONFIG_DIR, threaded

# from .checks import check_font, check_requirements, is_ascii
# from .files import increment_path
# from .ops import clip_coords, scale_image, xywh2xyxy, xyxy2xywh



def draw_arrow(im, pts, vec_len, color=(255, 153, 153), thickness=3, tip_length=0.3, alpha=0.5):
    dx_mean = dx_sum / vec_len
    dy_mean = dy_sum / vec_len
    
    start_point = (int(pts[-1][0]), int(pts[-1][1]))
    end_point = (int(pts[-1][0] + dx_mean * vec_len), int(pts[-1][1] + dy_mean * vec_len))
    
    overlay = im.copy()
    cv2.arrowedLine(overlay, start_point, end_point, color, thickness, tipLength=tip_length, line_type=cv2.LINE_AA)
    im = cv2.addWeighted(overlay, alpha, im, 1 - alpha, 0)
    
    return im

def get_arrow_distance(im, pts, vec_len, color=(255, 153, 153), thickness=3, tip_length=0.3, alpha=0.5):
    dx_mean = dx_sum / vec_len
    dy_mean = dy_sum / vec_len
    # Call draw_arrow() to get the arrow start and end points
    start_point = (int(pts[-1][0]), int(pts[-1][1]))
    end_point = (int(pts[-1][0] + dx_mean * vec_len), int(pts[-1][1] + dy_mean * vec_len))
    
    # Calculate the distance between the start and end points
    distance = ((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2)**0.5
    
    return distance

# from yolov7

# # Plotting utils

# import glob
# import math
# import os
# import random
# from copy import copy
# from pathlib import Path

# import cv2
# import matplotlib
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import torch
# import yaml
# from PIL import Image, ImageDraw, ImageFont
# from scipy.signal import butter, filtfilt

# # from utils.general import xywh2xyxy, xyxy2xywh
# # from utils.metrics import fitness

# # Settings
# matplotlib.rc('font', **{'size': 11})
# matplotlib.use('Agg')  # for writing to files only


# def color_list():
#     # Return first 10 plt colors as (r,g,b) https://stackoverflow.com/questions/51350872/python-from-color-name-to-rgb
#     def hex2rgb(h):
#         return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

#     return [hex2rgb(h) for h in matplotlib.colors.TABLEAU_COLORS.values()]  # or BASE_ (8), CSS4_ (148), XKCD_ (949)


# def hist2d(x, y, n=100):
#     # 2d histogram used in labels.png and evolve.png
#     xedges, yedges = np.linspace(x.min(), x.max(), n), np.linspace(y.min(), y.max(), n)
#     hist, xedges, yedges = np.histogram2d(x, y, (xedges, yedges))
#     xidx = np.clip(np.digitize(x, xedges) - 1, 0, hist.shape[0] - 1)
#     yidx = np.clip(np.digitize(y, yedges) - 1, 0, hist.shape[1] - 1)
#     return np.log(hist[xidx, yidx])


# # def butter_lowpass_filtfilt(data, cutoff=1500, fs=50000, order=5):
# #     # https://stackoverflow.com/questions/28536191/how-to-filter-smooth-with-scipy-numpy
# #     def butter_lowpass(cutoff, fs, order):
# #         nyq = 0.5 * fs
# #         normal_cutoff = cutoff / nyq
# #         return butter(order, normal_cutoff, btype='low', analog=False)

# #     b, a = butter_lowpass(cutoff, fs, order=order)
# #     return filtfilt(b, a, data)  # forward-backward filter

def plot_centroid(V, img, start_point, end_point):
    num = 42
    ratio = 2 * (V-0) / (100 - num) if 2 * (V-0) / (100 - num) < 1 else 1
    b = int(max(0, 255*(1 - ratio)))
    r = int(max(0, 255*(ratio - 1)))
    g = 255 - b - r
    color = (b, g, r)
    tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    # cv2.line(img, start_point, end_point, color, tl)
    # length = end_point-start_point
    # arrow = 
    cv2.arrowedLine(img, end_point, start_point, color, tl)

def rgb(minimum, maximum, value):
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value-minimum) / (maximum - minimum)
    b = int(max(0, 255*(1 - ratio)))
    r = int(max(0, 255*(ratio - 1)))
    g = 255 - b - r
    return b, g, r








def get_intersection_point(center, direction, bbox):
    x, y, w, h = bbox
    cx, cy = center
    dx, dy = direction
    
    # Calculate intersection points with each side of the bbox
    # Top
    if dy < 0:
        ix = cx + dx * (h/2 - cy) / dy
        if ix >= x and ix <= x + w:
            return ix, y
    # Bottom
    elif dy > 0:
        ix = cx + dx * (-h/2 - cy) / dy
        if ix >= x and ix <= x + w:
            return ix, y+h
    # Left
    if dx < 0:
        iy = cy + dy * (w/2 - cx) / dx
        if iy >= y and iy <= y + h:
            return x, iy
    # Right
    elif dx > 0:
        iy = cy + dy * (-w/2 - cx) / dx
        if iy >= y and iy <= y + h:
            return x+w, iy
    
    # No intersection
    return None

def get_intersection_points(center, direction, bbox):
    x, y, w, h = bbox
    cx, cy = center
    dx, dy = direction

    # Calculate intersection points with each side of the bbox
    points = []

    # Top
    if dy < 0:
        ix = cx + dx * (h/2 - cy) / dy
        if ix >= x and ix <= x + w:
            points.append((ix, y))

    # Bottom
    elif dy > 0:
        ix = cx + dx * (-h/2 - cy) / dy
        if ix >= x and ix <= x + w:
            points.append((ix, y+h))

    # Left
    if dx < 0:
        iy = cy + dy * (w/2 - cx) / dx
        if iy >= y and iy <= y + h:
            points.append((x, iy))

    # Right
    elif dx > 0:
        iy = cy + dy * (-w/2 - cx) / dx
        if iy >= y and iy <= y + h:
            points.append((x+w, iy))

    return points


def draw_arrow_with_distance(im, pts, bbox, vec_len=50, color=(255, 0, 0), thickness=2, tip_length=0.2, alpha=0.5):
    dx_mean = dx_sum / vec_len
    dy_mean = dy_sum / vec_len

    center = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
    direction = (dx_mean, dy_mean)
    start_point = (int(pts[-1][0]), int(pts[-1][1]))
    end_point = (int(pts[-1][0] + dx_mean * vec_len), int(pts[-1][1] + dy_mean * vec_len))
    
    overlay = im.copy()
    cv2.arrowedLine(overlay, start_point, end_point, color, thickness, tipLength=tip_length, line_type=cv2.LINE_AA)
    
    intersection = get_intersection_point(center, direction, bbox)
    print("center:", center)
    print("direction:", direction)
    print("start_point:", start_point)
    print("end_point:", end_point)
    print("intersection:", intersection)
    print("bbox:", bbox)
    if intersection:
        # cv2.line(overlay, start_point, intersection, color, thickness, cv2.LINE_AA)
        cv2.circle(overlay, intersection, 5, color, -1)
        distance = round(np.linalg.norm(np.array(start_point) - np.array(intersection)), 2)
        cv2.putText(overlay, f"{distance}", intersection, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness=2)
    
    im = cv2.addWeighted(overlay, alpha, im, 1 - alpha, 0)
    
    return im
def draw_direction(im, bbox, direction, color=(0, 0, 255), thickness=2, alpha=0.5):
    x, y, w, h = bbox
    cx, cy = x + w/2, y + h/2
    dx, dy = direction
    
    length = np.sqrt(dx**2 + dy**2) * w/2
    
    start_point = (int(cx), int(cy))
    end_point = (int(cx + dx * length), int(cy + dy * length))
    
    overlay = im.copy()
    cv2.line(overlay, start_point, end_point, color, thickness, cv2.LINE_AA)
    
    im = cv2.addWeighted(overlay, alpha, im, 1 - alpha, 0)
    
    print(f"Length of line: {length}")
    
    return im

def draw_line_with_intersection(img, start_point, intersection_point, color=(0, 0, 255), thickness=2):
    # Convert the coordinates to integers
    start_point = (int(start_point[0]), int(start_point[1]))
    intersection_point = (int(intersection_point[0]), int(intersection_point[1]))
    
    # Draw the line from the intersection point to the start point
    cv2.line(img, start_point, intersection_point, color, thickness)
    
    return img

# def drawAxis(img, start_pt, vec, colour, length):
#     # アンチエイリアス
#     CV_AA = 16

#     # 終了点
#     end_pt = (int(start_pt[0] + length * vec[0]), int(start_pt[1] + length * vec[1]))

#     # 中心を描画
#     cv2.circle(img, (int(start_pt[0]), int(start_pt[1])), 5, colour, 1)

#     # 軸線を描画
#     cv2.line(img, (int(start_pt[0]), int(start_pt[1])), end_pt, colour, 1, CV_AA);

#     # 先端の矢印を描画
#     angle = math.atan2(vec[1], vec[0])
#     print(angle)

#     qx0 = int(end_pt[0] - 9 * math.cos(angle + math.pi / 4));
#     qy0 = int(end_pt[1] - 9 * math.sin(angle + math.pi / 4));
#     cv2.line(img, end_pt, (qx0, qy0), colour, 1, CV_AA);

#     qx1 = int(end_pt[0] - 9 * math.cos(angle - math.pi / 4));
#     qy1 = int(end_pt[1] - 9 * math.sin(angle - math.pi / 4));
#     cv2.line(img, end_pt, (qx1, qy1), colour, 1, CV_AA);

# def plot_one_box(x, img, color=None, label=None, line_thickness=3):
#     # Plots one bounding box on image img
#     tl = line_thickness
#     # tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
#     color = color or [random.randint(0, 255) for _ in range(3)]
#     c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
#     cx, cy = int((x[0]+x[2])/2), int((x[1]+x[3])/2) 
#     cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
#     # cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
#     # cv2.drawMarker(img, (cx, cy), color, cv2.MARKER_STAR, 20, 1, cv2.LINE_4)
#     if label:
#         tf = max(tl - 1, 1)  # font thickness
#         t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
#         c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
#         cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
#         cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


# def plot_skeleton_kpts(im, kpts, steps, orig_shape=None):
#     #Plot the skeleton and keypointsfor coco datatset
#     palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
#                         [230, 230, 0], [255, 153, 255], [153, 204, 255],
#                         [255, 102, 255], [255, 51, 255], [102, 178, 255],
#                         [51, 153, 255], [255, 153, 153], [255, 102, 102],
#                         [255, 51, 51], [153, 255, 153], [102, 255, 102],
#                         [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
#                         [255, 255, 255]])

#     skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
#                 [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
#                 [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

#     pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
#     pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
#     radius = 5
#     num_kpts = len(kpts) // steps

#     for kid in range(num_kpts):
#         r, g, b = pose_kpt_color[kid]
#         x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
#         if not (x_coord % 640 == 0 or y_coord % 640 == 0):
#             if steps == 3:
#                 conf = kpts[steps * kid + 2]
#                 if conf < 0.5:
#                     continue
#             cv2.circle(im, (int(x_coord), int(y_coord)), radius, (int(r), int(g), int(b)), -1)

#     for sk_id, sk in enumerate(skeleton):
#         r, g, b = pose_limb_color[sk_id]
#         pos1 = (int(kpts[(sk[0]-1)*steps]), int(kpts[(sk[0]-1)*steps+1]))
#         pos2 = (int(kpts[(sk[1]-1)*steps]), int(kpts[(sk[1]-1)*steps+1]))
#         if steps == 3:
#             conf1 = kpts[(sk[0]-1)*steps+2]
#             conf2 = kpts[(sk[1]-1)*steps+2]
#             if conf1<0.5 or conf2<0.5:
#                 continue
#         if pos1[0]%640 == 0 or pos1[1]%640==0 or pos1[0]<0 or pos1[1]<0:
#             continue
#         if pos2[0] % 640 == 0 or pos2[1] % 640 == 0 or pos2[0]<0 or pos2[1]<0:
#             continue
#         cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=2)

# #  save_one_box(bboxes, imc, file=save_dir / 'crops' / txt_file_name / names[c] / f'{id}' / f'{p.stem}.jpg', BGR=True)
# # def save_one_box(xyxy, im, file=Path('im.jpg'), gain=1.02, pad=10, square=False, BGR=False, save=True):
# def save_one_box(xyxy, im, file, gain=1.02, pad=10, square=False, BGR=False, save=True):
#     # Save image crop as {file} with crop size multiple {gain} and {pad} pixels. Save and/or return crop
#     xyxy = torch.tensor(xyxy).view(-1, 4)
#     b = xyxy2xywh(xyxy)  # boxes
#     if square:
#         b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # attempt rectangle to square
#     b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
#     xyxy = xywh2xyxy(b).long()
#     clip_coords(xyxy, im.shape)
#     crop = im[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]), ::(1 if BGR else -1)]
#     src = crop
    
#     gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

#     # ２値化
#     retval, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
#     # retval, bw = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#     # retval, bw = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
#     # X = np.array(contours[i], dtype=np.float).reshape((contours[i].shape[0], contours[i].shape[2]))
#     # X = contours
#     # PCA（１次元）
#     mean, eigenvectors = cv2.PCACompute(bw, mean=np.array([], dtype=np.float), maxComponents=1)

#     # 主成分方向のベクトルを描画
#     pt = (mean[0][0], mean[0][1])
#     vec = (eigenvectors[0][0], eigenvectors[0][1])
#     drawAxis(src, pt, vec, (200, 10, 255), 150)

#     if save:
#         file.parent.mkdir(parents=True, exist_ok=True)  # make directory
#         f = str(increment_path(file).with_suffix('.jpg'))
#         # cv2.imwrite(f, crop)  # save BGR, https://github.com/ultralytics/yolov5/issues/7007 chroma subsampling issue
#         Image.fromarray(bw[..., ::-1]).save(f, quality=95, subsampling=0)  # save RGB
#         # Image.fromarray(crop[..., ::-1]).save(f, quality=95, subsampling=0)  # save RGB
#     return crop

# def increment_path(path, exist_ok=False, sep='', mkdir=False):
#     # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
#     path = Path(path)  # os-agnostic
#     if path.exists() and not exist_ok:
#         path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')

#         # Method 1
#         for n in range(2, 9999):
#             p = f'{path}{sep}{n}{suffix}'  # increment path
#             if not os.path.exists(p):  #
#                 break
#         path = Path(p)

#         # Method 2 (deprecated)
#         # dirs = glob.glob(f"{path}{sep}*")  # similar paths
#         # matches = [re.search(rf"{path.stem}{sep}(\d+)", d) for d in dirs]
#         # i = [int(m.groups()[0]) for m in matches if m]  # indices
#         # n = max(i) + 1 if i else 2  # increment number
#         # path = Path(f"{path}{sep}{n}{suffix}")  # increment path

#     if mkdir:
#         path.mkdir(parents=True, exist_ok=True)  # make directory

#     return path

# def plot_speed(file, path):
#     # path = Path(path)
#     # df = pd.read_csv(path, sep='\s+', names=['time','speed'])
#     # df.plot(x='time', y='speed', figsize=(20, 10))
#     # path = "./runs/track/exp122/tracks/speed.txt"
#     # files = list(Path(path).glob('speed.txt'))
#     # assert len(files), 'No results.txt files found in %s, nothing to plot.' % os.path.abspath(save_dir)
#     file.parent.mkdir(parents=True, exist_ok=True) 
#     df = pd.read_csv(file, sep='\s+', names=['time','speed'])
#     with open(path) as f:
#         lines = f.readlines()
#     # assert os.path.isfile(path)
#     plt.figure()
#     df.plot(x='time', y='speed', figsize=(20, 10))
#     plt.savefig(file / 'speed.jpg')
#     plt.close('all')
#     # if save:
#      # make directory
#     # f = str(increment_path(file).with_suffix('.jpg'))
#     # # cv2.imwrite(f, crop)  # save BGR, https://github.com/ultralytics/yolov5/issues/7007 chroma subsampling issue
#     # Image.fromarray(bw[..., ::-1]).save(f, quality=95, subsampling=0)  # save RGB

#     # plt.savefig(Path(save_dir) / 'LR.png', dpi=200)
#     # df = pd.read_csv("./runs/track/exp125/tracks/speed.txt", sep='\s+', names=['time','speed'])

# def plot_speed_txt(path="./runs/track/exp181/"):  # from utils.plots import *; plot_targets_txt()
#     arr = np.loadtxt(str(Path(path)) + '\\' + 'speed.txt').T
#     # arr = np.loadtxt(Path(path).glob('speed.txt'), dtype=np.float32).T
#     n = arr[0]
#     arr = arr[:,n>8]
#     n, x, y = arr[0], arr[1], arr[2]

#     i = 0
#     dt = 50
#     flag = 0

#     xmin, xmax = t[i]-dt, t[i]+dt
#     ymin, ymax = 0, 20

#     x_list = []
#     # xi = 0
#     for yy in y:
#         # for xx in x:
#         if yy > 10:
#             x_list.append(x)

#     plt.figure(figsize=(20, 10), dpi=200)
#     plt.axhline(y=8, color="gray", linestyle="--")
#     if flag:
#         # t = [675, 1000]
#         plt.ylim(-1, 15)

#     for i in range(len(x)):
#         if y[i] > 8:
#             # print(x[i], n[i])
#             plt.axvline(x=x[i], color="darkorange", linestyle='dashed')

#     for tt in t:
#         if flag:
#             plt.xlim(tt-dt, tt+dt)
#         plt.plot(x,n,'g-', linewidth=1, alpha=0.8, label="number")
#         plt.plot(x,y, 'b-', label="speed [km/h]")
#         plt.legend()
#         # plt.savefig(f"./runs/track/exp{exp}/tracks/" + str(tt) + 'speed.png')
#         plt.savefig(str(Path(path).name) + '.png', dpi=300)
#         # plt.show()

def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))

