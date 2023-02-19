# Plotting utils

import glob
import math
import os
import random
from copy import copy
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import butter, filtfilt

from utils.general import xywh2xyxy, xyxy2xywh
from utils.metrics import fitness

# Settings
matplotlib.rc('font', **{'size': 11})
matplotlib.use('Agg')  # for writing to files only


def color_list():
    # Return first 10 plt colors as (r,g,b) https://stackoverflow.com/questions/51350872/python-from-color-name-to-rgb
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

    return [hex2rgb(h) for h in matplotlib.colors.TABLEAU_COLORS.values()]  # or BASE_ (8), CSS4_ (148), XKCD_ (949)


def hist2d(x, y, n=100):
    # 2d histogram used in labels.png and evolve.png
    xedges, yedges = np.linspace(x.min(), x.max(), n), np.linspace(y.min(), y.max(), n)
    hist, xedges, yedges = np.histogram2d(x, y, (xedges, yedges))
    xidx = np.clip(np.digitize(x, xedges) - 1, 0, hist.shape[0] - 1)
    yidx = np.clip(np.digitize(y, yedges) - 1, 0, hist.shape[1] - 1)
    return np.log(hist[xidx, yidx])


def butter_lowpass_filtfilt(data, cutoff=1500, fs=50000, order=5):
    # https://stackoverflow.com/questions/28536191/how-to-filter-smooth-with-scipy-numpy
    def butter_lowpass(cutoff, fs, order):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        return butter(order, normal_cutoff, btype='low', analog=False)

    b, a = butter_lowpass(cutoff, fs, order=order)
    return filtfilt(b, a, data)  # forward-backward filter

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

def drawAxis(img, start_pt, vec, colour, length):
    # アンチエイリアス
    CV_AA = 16

    # 終了点
    end_pt = (int(start_pt[0] + length * vec[0]), int(start_pt[1] + length * vec[1]))

    # 中心を描画
    cv2.circle(img, (int(start_pt[0]), int(start_pt[1])), 5, colour, 1)

    # 軸線を描画
    cv2.line(img, (int(start_pt[0]), int(start_pt[1])), end_pt, colour, 1, CV_AA);

    # 先端の矢印を描画
    angle = math.atan2(vec[1], vec[0])
    print(angle)

    qx0 = int(end_pt[0] - 9 * math.cos(angle + math.pi / 4));
    qy0 = int(end_pt[1] - 9 * math.sin(angle + math.pi / 4));
    cv2.line(img, end_pt, (qx0, qy0), colour, 1, CV_AA);

    qx1 = int(end_pt[0] - 9 * math.cos(angle - math.pi / 4));
    qy1 = int(end_pt[1] - 9 * math.sin(angle - math.pi / 4));
    cv2.line(img, end_pt, (qx1, qy1), colour, 1, CV_AA);

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness
    # tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cx, cy = int((x[0]+x[2])/2), int((x[1]+x[3])/2) 
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    # cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    # cv2.drawMarker(img, (cx, cy), color, cv2.MARKER_STAR, 20, 1, cv2.LINE_4)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def plot_one_box_PIL(box, img, color=None, label=None, line_thickness=None):
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    line_thickness = line_thickness or max(int(min(img.size) / 200), 2)
    draw.rectangle(box, width=line_thickness, outline=tuple(color))  # plot
    if label:
        fontsize = max(round(max(img.size) / 40), 12)
        font = ImageFont.truetype("Arial.ttf", fontsize)
        txt_width, txt_height = font.getsize(label)
        draw.rectangle([box[0], box[1] - txt_height + 4, box[0] + txt_width, box[1]], fill=tuple(color))
        draw.text((box[0], box[1] - txt_height + 1), label, fill=(255, 255, 255), font=font)
    return np.asarray(img)


def plot_wh_methods():  # from utils.plots import *; plot_wh_methods()
    # Compares the two methods for width-height anchor multiplication
    # https://github.com/ultralytics/yolov3/issues/168
    x = np.arange(-4.0, 4.0, .1)
    ya = np.exp(x)
    yb = torch.sigmoid(torch.from_numpy(x)).numpy() * 2

    fig = plt.figure(figsize=(6, 3), tight_layout=True)
    plt.plot(x, ya, '.-', label='YOLOv3')
    plt.plot(x, yb ** 2, '.-', label='YOLOR ^2')
    plt.plot(x, yb ** 1.6, '.-', label='YOLOR ^1.6')
    plt.xlim(left=-4, right=4)
    plt.ylim(bottom=0, top=6)
    plt.xlabel('input')
    plt.ylabel('output')
    plt.grid()
    plt.legend()
    fig.savefig('comparison.png', dpi=200)


def output_to_target(output):
    # Convert model output to target format [batch_id, class_id, x, y, w, h, conf]
    targets = []
    for i, o in enumerate(output):
        for *box, conf, cls in o.cpu().numpy():
            targets.append([i, cls, *list(*xyxy2xywh(np.array(box)[None])), conf])
    return np.array(targets)


def plot_images(images, targets, paths=None, fname='images.jpg', names=None, max_size=640, max_subplots=16):
    # Plot image grid with labels

    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    # un-normalise
    if np.max(images[0]) <= 1:
        images *= 255

    tl = 3  # line thickness
    tf = max(tl - 1, 1)  # font thickness
    bs, _, h, w = images.shape  # batch size, _, height, width
    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs ** 0.5)  # number of subplots (square)

    # Check if we should resize
    scale_factor = max_size / max(h, w)
    if scale_factor < 1:
        h = math.ceil(scale_factor * h)
        w = math.ceil(scale_factor * w)

    colors = color_list()  # list of colors
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init
    for i, img in enumerate(images):
        if i == max_subplots:  # if last batch has fewer images than we expect
            break

        block_x = int(w * (i // ns))
        block_y = int(h * (i % ns))

        img = img.transpose(1, 2, 0)
        if scale_factor < 1:
            img = cv2.resize(img, (w, h))

        mosaic[block_y:block_y + h, block_x:block_x + w, :] = img
        if len(targets) > 0:
            image_targets = targets[targets[:, 0] == i]
            boxes = xywh2xyxy(image_targets[:, 2:6]).T
            classes = image_targets[:, 1].astype('int')
            labels = image_targets.shape[1] == 6  # labels if no conf column
            conf = None if labels else image_targets[:, 6]  # check for confidence presence (label vs pred)

            if boxes.shape[1]:
                if boxes.max() <= 1.01:  # if normalized with tolerance 0.01
                    boxes[[0, 2]] *= w  # scale to pixels
                    boxes[[1, 3]] *= h
                elif scale_factor < 1:  # absolute coords need scale if image scales
                    boxes *= scale_factor
            boxes[[0, 2]] += block_x
            boxes[[1, 3]] += block_y
            for j, box in enumerate(boxes.T):
                cls = int(classes[j])
                color = colors[cls % len(colors)]
                cls = names[cls] if names else cls
                if labels or conf[j] > 0.25:  # 0.25 conf thresh
                    label = '%s' % cls if labels else '%s %.1f' % (cls, conf[j])
                    plot_one_box(box, mosaic, label=label, color=color, line_thickness=tl)

        # Draw image filename labels
        if paths:
            label = Path(paths[i]).name[:40]  # trim to 40 char
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            cv2.putText(mosaic, label, (block_x + 5, block_y + t_size[1] + 5), 0, tl / 3, [220, 220, 220], thickness=tf,
                        lineType=cv2.LINE_AA)

        # Image border
        cv2.rectangle(mosaic, (block_x, block_y), (block_x + w, block_y + h), (255, 255, 255), thickness=3)

    if fname:
        r = min(1280. / max(h, w) / ns, 1.0)  # ratio to limit image size
        mosaic = cv2.resize(mosaic, (int(ns * w * r), int(ns * h * r)), interpolation=cv2.INTER_AREA)
        # cv2.imwrite(fname, cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))  # cv2 save
        Image.fromarray(mosaic).save(fname)  # PIL save
    return mosaic


def plot_lr_scheduler(optimizer, scheduler, epochs=300, save_dir=''):
    # Plot LR simulating training for full epochs
    optimizer, scheduler = copy(optimizer), copy(scheduler)  # do not modify originals
    y = []
    for _ in range(epochs):
        scheduler.step()
        y.append(optimizer.param_groups[0]['lr'])
    plt.plot(y, '.-', label='LR')
    plt.xlabel('epoch')
    plt.ylabel('LR')
    plt.grid()
    plt.xlim(0, epochs)
    plt.ylim(0)
    plt.savefig(Path(save_dir) / 'LR.png', dpi=200)
    plt.close()


def plot_test_txt():  # from utils.plots import *; plot_test()
    # Plot test.txt histograms
    x = np.loadtxt('test.txt', dtype=np.float32)
    box = xyxy2xywh(x[:, :4])
    cx, cy = box[:, 0], box[:, 1]

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)
    ax.hist2d(cx, cy, bins=600, cmax=10, cmin=0)
    ax.set_aspect('equal')
    plt.savefig('hist2d.png', dpi=300)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
    ax[0].hist(cx, bins=600)
    ax[1].hist(cy, bins=600)
    plt.savefig('hist1d.png', dpi=200)


def plot_targets_txt():  # from utils.plots import *; plot_targets_txt()
    # Plot targets.txt histograms
    x = np.loadtxt('targets.txt', dtype=np.float32).T
    s = ['x targets', 'y targets', 'width targets', 'height targets']
    fig, ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
    ax = ax.ravel()
    for i in range(4):
        ax[i].hist(x[i], bins=100, label='%.3g +/- %.3g' % (x[i].mean(), x[i].std()))
        ax[i].legend()
        ax[i].set_title(s[i])
    plt.savefig('targets.jpg', dpi=200)


def plot_study_txt(path='', x=None):  # from utils.plots import *; plot_study_txt()
    # Plot study.txt generated by test.py
    fig, ax = plt.subplots(2, 4, figsize=(10, 6), tight_layout=True)
    # ax = ax.ravel()

    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 4), tight_layout=True)
    # for f in [Path(path) / f'study_coco_{x}.txt' for x in ['yolor-p6', 'yolor-w6', 'yolor-e6', 'yolor-d6']]:
    for f in sorted(Path(path).glob('study*.txt')):
        y = np.loadtxt(f, dtype=np.float32, usecols=[0, 1, 2, 3, 7, 8, 9], ndmin=2).T
        x = np.arange(y.shape[1]) if x is None else np.array(x)
        s = ['P', 'R', 'mAP@.5', 'mAP@.5:.95', 't_inference (ms/img)', 't_NMS (ms/img)', 't_total (ms/img)']
        # for i in range(7):
        #     ax[i].plot(x, y[i], '.-', linewidth=2, markersize=8)
        #     ax[i].set_title(s[i])

        j = y[3].argmax() + 1
        ax2.plot(y[6, 1:j], y[3, 1:j] * 1E2, '.-', linewidth=2, markersize=8,
                 label=f.stem.replace('study_coco_', '').replace('yolo', 'YOLO'))

    ax2.plot(1E3 / np.array([209, 140, 97, 58, 35, 18]), [34.6, 40.5, 43.0, 47.5, 49.7, 51.5],
             'k.-', linewidth=2, markersize=8, alpha=.25, label='EfficientDet')

    ax2.grid(alpha=0.2)
    ax2.set_yticks(np.arange(20, 60, 5))
    ax2.set_xlim(0, 57)
    ax2.set_ylim(30, 55)
    ax2.set_xlabel('GPU Speed (ms/img)')
    ax2.set_ylabel('COCO AP val')
    ax2.legend(loc='lower right')
    plt.savefig(str(Path(path).name) + '.png', dpi=300)


def plot_labels(labels, names=(), save_dir=Path(''), loggers=None):
    # plot dataset labels
    print('Plotting labels... ')
    c, b = labels[:, 0], labels[:, 1:].transpose()  # classes, boxes
    nc = int(c.max() + 1)  # number of classes
    colors = color_list()
    x = pd.DataFrame(b.transpose(), columns=['x', 'y', 'width', 'height'])

    # seaborn correlogram
    sns.pairplot(x, corner=True, diag_kind='auto', kind='hist', diag_kws=dict(bins=50), plot_kws=dict(pmax=0.9))
    plt.savefig(save_dir / 'labels_correlogram.jpg', dpi=200)
    plt.close()

    # matplotlib labels
    matplotlib.use('svg')  # faster
    ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)[1].ravel()
    ax[0].hist(c, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)
    ax[0].set_ylabel('instances')
    if 0 < len(names) < 30:
        ax[0].set_xticks(range(len(names)))
        ax[0].set_xticklabels(names, rotation=90, fontsize=10)
    else:
        ax[0].set_xlabel('classes')
    sns.histplot(x, x='x', y='y', ax=ax[2], bins=50, pmax=0.9)
    sns.histplot(x, x='width', y='height', ax=ax[3], bins=50, pmax=0.9)

    # rectangles
    labels[:, 1:3] = 0.5  # center
    labels[:, 1:] = xywh2xyxy(labels[:, 1:]) * 2000
    img = Image.fromarray(np.ones((2000, 2000, 3), dtype=np.uint8) * 255)
    for cls, *box in labels[:1000]:
        ImageDraw.Draw(img).rectangle(box, width=1, outline=colors[int(cls) % 10])  # plot
    ax[1].imshow(img)
    ax[1].axis('off')

    for a in [0, 1, 2, 3]:
        for s in ['top', 'right', 'left', 'bottom']:
            ax[a].spines[s].set_visible(False)

    plt.savefig(save_dir / 'labels.jpg', dpi=200)
    matplotlib.use('Agg')
    plt.close()

    # loggers
    for k, v in loggers.items() or {}:
        if k == 'wandb' and v:
            v.log({"Labels": [v.Image(str(x), caption=x.name) for x in save_dir.glob('*labels*.jpg')]}, commit=False)


def plot_evolution(yaml_file='data/hyp.finetune.yaml'):  # from utils.plots import *; plot_evolution()
    # Plot hyperparameter evolution results in evolve.txt
    with open(yaml_file) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)
    x = np.loadtxt('evolve.txt', ndmin=2)
    f = fitness(x)
    # weights = (f - f.min()) ** 2  # for weighted results
    plt.figure(figsize=(10, 12), tight_layout=True)
    matplotlib.rc('font', **{'size': 8})
    for i, (k, v) in enumerate(hyp.items()):
        y = x[:, i + 7]
        # mu = (y * weights).sum() / weights.sum()  # best weighted result
        mu = y[f.argmax()]  # best single result
        plt.subplot(6, 5, i + 1)
        plt.scatter(y, f, c=hist2d(y, f, 20), cmap='viridis', alpha=.8, edgecolors='none')
        plt.plot(mu, f.max(), 'k+', markersize=15)
        plt.title('%s = %.3g' % (k, mu), fontdict={'size': 9})  # limit to 40 characters
        if i % 5 != 0:
            plt.yticks([])
        print('%15s: %.3g' % (k, mu))
    plt.savefig('evolve.png', dpi=200)
    print('\nPlot saved as evolve.png')


def profile_idetection(start=0, stop=0, labels=(), save_dir=''):
    # Plot iDetection '*.txt' per-image logs. from utils.plots import *; profile_idetection()
    ax = plt.subplots(2, 4, figsize=(12, 6), tight_layout=True)[1].ravel()
    s = ['Images', 'Free Storage (GB)', 'RAM Usage (GB)', 'Battery', 'dt_raw (ms)', 'dt_smooth (ms)', 'real-world FPS']
    files = list(Path(save_dir).glob('frames*.txt'))
    for fi, f in enumerate(files):
        try:
            results = np.loadtxt(f, ndmin=2).T[:, 90:-30]  # clip first and last rows
            n = results.shape[1]  # number of rows
            x = np.arange(start, min(stop, n) if stop else n)
            results = results[:, x]
            t = (results[0] - results[0].min())  # set t0=0s
            results[0] = x
            for i, a in enumerate(ax):
                if i < len(results):
                    label = labels[fi] if len(labels) else f.stem.replace('frames_', '')
                    a.plot(t, results[i], marker='.', label=label, linewidth=1, markersize=5)
                    a.set_title(s[i])
                    a.set_xlabel('time (s)')
                    # if fi == len(files) - 1:
                    #     a.set_ylim(bottom=0)
                    for side in ['top', 'right']:
                        a.spines[side].set_visible(False)
                else:
                    a.remove()
        except Exception as e:
            print('Warning: Plotting error for %s; %s' % (f, e))

    ax[1].legend()
    plt.savefig(Path(save_dir) / 'idetection_profile.png', dpi=200)


def plot_results_overlay(start=0, stop=0):  # from utils.plots import *; plot_results_overlay()
    # Plot training 'results*.txt', overlaying train and val losses
    s = ['train', 'train', 'train', 'Precision', 'mAP@0.5', 'val', 'val', 'val', 'Recall', 'mAP@0.5:0.95']  # legends
    t = ['Box', 'Objectness', 'Classification', 'P-R', 'mAP-F1']  # titles
    for f in sorted(glob.glob('results*.txt') + glob.glob('../../Downloads/results*.txt')):
        results = np.loadtxt(f, usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2).T
        n = results.shape[1]  # number of rows
        x = range(start, min(stop, n) if stop else n)
        fig, ax = plt.subplots(1, 5, figsize=(14, 3.5), tight_layout=True)
        ax = ax.ravel()
        for i in range(5):
            for j in [i, i + 5]:
                y = results[j, x]
                ax[i].plot(x, y, marker='.', label=s[j])
                # y_smooth = butter_lowpass_filtfilt(y)
                # ax[i].plot(x, np.gradient(y_smooth), marker='.', label=s[j])

            ax[i].set_title(t[i])
            ax[i].legend()
            ax[i].set_ylabel(f) if i == 0 else None  # add filename
        fig.savefig(f.replace('.txt', '.png'), dpi=200)


def plot_results(start=0, stop=0, bucket='', id=(), labels=(), save_dir=''):
    # Plot training 'results*.txt'. from utils.plots import *; plot_results(save_dir='runs/train/exp')
    fig, ax = plt.subplots(2, 5, figsize=(12, 6), tight_layout=True)
    ax = ax.ravel()
    s = ['Box', 'Objectness', 'Classification', 'Precision', 'Recall',
         'val Box', 'val Objectness', 'val Classification', 'mAP@0.5', 'mAP@0.5:0.95']
    if bucket:
        # files = ['https://storage.googleapis.com/%s/results%g.txt' % (bucket, x) for x in id]
        files = ['results%g.txt' % x for x in id]
        c = ('gsutil cp ' + '%s ' * len(files) + '.') % tuple('gs://%s/results%g.txt' % (bucket, x) for x in id)
        os.system(c)
    else:
        files = list(Path(save_dir).glob('results*.txt'))
    assert len(files), 'No results.txt files found in %s, nothing to plot.' % os.path.abspath(save_dir)
    for fi, f in enumerate(files):
        try:
            results = np.loadtxt(f, usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2).T
            n = results.shape[1]  # number of rows
            x = range(start, min(stop, n) if stop else n)
            for i in range(10):
                y = results[i, x]
                if i in [0, 1, 2, 5, 6, 7]:
                    y[y == 0] = np.nan  # don't show zero loss values
                    # y /= y[0]  # normalize
                label = labels[fi] if len(labels) else f.stem
                ax[i].plot(x, y, marker='.', label=label, linewidth=2, markersize=8)
                ax[i].set_title(s[i])
                # if i in [5, 6, 7]:  # share train and val loss y axes
                #     ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])
        except Exception as e:
            print('Warning: Plotting error for %s; %s' % (f, e))

    ax[1].legend()
    fig.savefig(Path(save_dir) / 'results.png', dpi=200)
    
    
def output_to_keypoint(output):
    # Convert model output to target format [batch_id, class_id, x, y, w, h, conf]
    targets = []
    for i, o in enumerate(output):
        kpts = o[:,6:]
        o = o[:,:6]
        for index, (*box, conf, cls) in enumerate(o.detach().cpu().numpy()):
            targets.append([i, cls, *list(*xyxy2xywh(np.array(box)[None])), conf, *list(kpts.detach().cpu().numpy()[index])])
    return np.array(targets)


def plot_skeleton_kpts(im, kpts, steps, orig_shape=None):
    #Plot the skeleton and keypointsfor coco datatset
    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [255, 255, 255]])

    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
    pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
    radius = 5
    num_kpts = len(kpts) // steps

    for kid in range(num_kpts):
        r, g, b = pose_kpt_color[kid]
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
        if not (x_coord % 640 == 0 or y_coord % 640 == 0):
            if steps == 3:
                conf = kpts[steps * kid + 2]
                if conf < 0.5:
                    continue
            cv2.circle(im, (int(x_coord), int(y_coord)), radius, (int(r), int(g), int(b)), -1)

    for sk_id, sk in enumerate(skeleton):
        r, g, b = pose_limb_color[sk_id]
        pos1 = (int(kpts[(sk[0]-1)*steps]), int(kpts[(sk[0]-1)*steps+1]))
        pos2 = (int(kpts[(sk[1]-1)*steps]), int(kpts[(sk[1]-1)*steps+1]))
        if steps == 3:
            conf1 = kpts[(sk[0]-1)*steps+2]
            conf2 = kpts[(sk[1]-1)*steps+2]
            if conf1<0.5 or conf2<0.5:
                continue
        if pos1[0]%640 == 0 or pos1[1]%640==0 or pos1[0]<0 or pos1[1]<0:
            continue
        if pos2[0] % 640 == 0 or pos2[1] % 640 == 0 or pos2[0]<0 or pos2[1]<0:
            continue
        cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=2)

def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

#  save_one_box(bboxes, imc, file=save_dir / 'crops' / txt_file_name / names[c] / f'{id}' / f'{p.stem}.jpg', BGR=True)
# def save_one_box(xyxy, im, file=Path('im.jpg'), gain=1.02, pad=10, square=False, BGR=False, save=True):
def save_one_box(xyxy, im, file, gain=1.02, pad=10, square=False, BGR=False, save=True):
    # Save image crop as {file} with crop size multiple {gain} and {pad} pixels. Save and/or return crop
    xyxy = torch.tensor(xyxy).view(-1, 4)
    b = xyxy2xywh(xyxy)  # boxes
    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # attempt rectangle to square
    b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
    xyxy = xywh2xyxy(b).long()
    clip_coords(xyxy, im.shape)
    crop = im[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]), ::(1 if BGR else -1)]
    src = crop
    
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # ２値化
    retval, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    # retval, bw = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # retval, bw = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # X = np.array(contours[i], dtype=np.float).reshape((contours[i].shape[0], contours[i].shape[2]))
    # X = contours
    # PCA（１次元）
    mean, eigenvectors = cv2.PCACompute(bw, mean=np.array([], dtype=np.float), maxComponents=1)

    # 主成分方向のベクトルを描画
    pt = (mean[0][0], mean[0][1])
    vec = (eigenvectors[0][0], eigenvectors[0][1])
    drawAxis(src, pt, vec, (200, 10, 255), 150)


    # 輪郭を抽出
    #   contours : [領域][Point No][0][x=0, y=1]
    #   cv2.CHAIN_APPROX_NONE: 中間点も保持する
    #   cv2.CHAIN_APPROX_SIMPLE: 中間点は保持しない
    # contours, hierarchy = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # # img, contours, hierarchy = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # # 各輪郭に対する処理
    # for i in range(0, len(contours)):

    #     # # 輪郭の領域を計算
    #     # area = cv2.contourArea(contours[i])

    #     # # ノイズ（小さすぎる領域）と全体の輪郭（大きすぎる領域）を除外
    #     # if area < 1e2 or 1e5 < area:
    #     #     continue

    #     # # 輪郭を描画する
    #     # cv2.drawContours(src, contours, i, (0, 0, 255), 2, 8, hierarchy, 0)

    #     # # 輪郭データを浮動小数点型の配列に格納
    #     X = np.array(contours[i], dtype=np.float).reshape((contours[i].shape[0], contours[i].shape[2]))
    #     # X = contours
    #     # PCA（１次元）
    #     mean, eigenvectors = cv2.PCACompute(X, mean=np.array([], dtype=np.float), maxComponents=1)

    #     # 主成分方向のベクトルを描画
    #     pt = (mean[0][0], mean[0][1])
    #     vec = (eigenvectors[0][0], eigenvectors[0][1])
    #     drawAxis(src, pt, vec, (255, 255, 0), 150)

    if save:
        file.parent.mkdir(parents=True, exist_ok=True)  # make directory
        f = str(increment_path(file).with_suffix('.jpg'))
        # cv2.imwrite(f, crop)  # save BGR, https://github.com/ultralytics/yolov5/issues/7007 chroma subsampling issue
        Image.fromarray(bw[..., ::-1]).save(f, quality=95, subsampling=0)  # save RGB
        # Image.fromarray(crop[..., ::-1]).save(f, quality=95, subsampling=0)  # save RGB
    return crop

def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')

        # Method 1
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)

        # Method 2 (deprecated)
        # dirs = glob.glob(f"{path}{sep}*")  # similar paths
        # matches = [re.search(rf"{path.stem}{sep}(\d+)", d) for d in dirs]
        # i = [int(m.groups()[0]) for m in matches if m]  # indices
        # n = max(i) + 1 if i else 2  # increment number
        # path = Path(f"{path}{sep}{n}{suffix}")  # increment path

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path

def plot_speed(file, path):
    # path = Path(path)
    # df = pd.read_csv(path, sep='\s+', names=['time','speed'])
    # df.plot(x='time', y='speed', figsize=(20, 10))
    # path = "./runs/track/exp122/tracks/speed.txt"
    # files = list(Path(path).glob('speed.txt'))
    # assert len(files), 'No results.txt files found in %s, nothing to plot.' % os.path.abspath(save_dir)
    file.parent.mkdir(parents=True, exist_ok=True) 
    df = pd.read_csv(file, sep='\s+', names=['time','speed'])
    with open(path) as f:
        lines = f.readlines()
    # assert os.path.isfile(path)
    plt.figure()
    df.plot(x='time', y='speed', figsize=(20, 10))
    plt.savefig(file / 'speed.jpg')
    plt.close('all')
    # if save:
     # make directory
    # f = str(increment_path(file).with_suffix('.jpg'))
    # # cv2.imwrite(f, crop)  # save BGR, https://github.com/ultralytics/yolov5/issues/7007 chroma subsampling issue
    # Image.fromarray(bw[..., ::-1]).save(f, quality=95, subsampling=0)  # save RGB

    # plt.savefig(Path(save_dir) / 'LR.png', dpi=200)
    # df = pd.read_csv("./runs/track/exp125/tracks/speed.txt", sep='\s+', names=['time','speed'])

def plot_speed_txt(path="./runs/track/exp181/"):  # from utils.plots import *; plot_targets_txt()
    # Plot targets.txt histograms
    # x = np.loadtxt('speed.txt', dtype=np.float32).T
    # s = ['x targets', 'y targets', 'width targets', 'height targets']
    # fig, ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
    # ax = ax.ravel()
    # for i in range(4):
    #     ax[i].hist(x[i], bins=100, label='%.3g +/- %.3g' % (x[i].mean(), x[i].std()))
    #     ax[i].legend()
    #     ax[i].set_title(s[i])
    # plt.savefig('targets.jpg', dpi=200)
    # import pandas as pd
    # import numpy as np
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # sns.set()
    # t = [148, 1050]
    # exp = 181
    # path = f"./runs/track/exp{exp}/tracks/speed.txt"
    # arr = np.loadtxt(Path(path).glob('speed.txt'), dtype=np.float32).T
    # # arr = np.loadtxt(path / glob('tracks/speed.txt'))
    # n = arr[0]
    # arr = arr[n>8]
    # n, x, y = arr[0], arr[1], arr[2]
    # exp = 181
    # path = f"./runs/track/exp{exp}/tracks/"
    arr = np.loadtxt(str(Path(path)) + '\\' + 'speed.txt').T
    # arr = np.loadtxt(Path(path).glob('speed.txt'), dtype=np.float32).T
    n = arr[0]
    arr = arr[:,n>8]
    n, x, y = arr[0], arr[1], arr[2]

    i = 0
    dt = 50
    flag = 0

    xmin, xmax = t[i]-dt, t[i]+dt
    ymin, ymax = 0, 20

    x_list = []
    # xi = 0
    for yy in y:
        # for xx in x:
        if yy > 10:
            x_list.append(x)

    plt.figure(figsize=(20, 10), dpi=200)
    plt.axhline(y=8, color="gray", linestyle="--")
    if flag:
        # t = [675, 1000]
        plt.ylim(-1, 15)

    for i in range(len(x)):
        if y[i] > 8:
            # print(x[i], n[i])
            plt.axvline(x=x[i], color="darkorange", linestyle='dashed')

    for tt in t:
        if flag:
            plt.xlim(tt-dt, tt+dt)
        plt.plot(x,n,'g-', linewidth=1, alpha=0.8, label="number")
        plt.plot(x,y, 'b-', label="speed [km/h]")
        plt.legend()
        # plt.savefig(f"./runs/track/exp{exp}/tracks/" + str(tt) + 'speed.png')
        plt.savefig(str(Path(path).name) + '.png', dpi=300)
        # plt.show()

def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))

