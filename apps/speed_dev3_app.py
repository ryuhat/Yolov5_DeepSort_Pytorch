import argparse
import cv2
import os
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import platform
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov8') not in sys.path:
    sys.path.append(str(ROOT / 'yolov8'))  # add yolov5 ROOT to PATH
if str(ROOT / 'trackers' / 'strongsort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strongsort'))  # add strong_sort ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import logging
from yolov8.ultralytics.nn.autobackend import AutoBackend
from yolov8.ultralytics.yolo.data.dataloaders.stream_loaders import LoadImages, LoadStreams
from yolov8.ultralytics.yolo.data.utils import IMG_FORMATS, VID_FORMATS
from yolov8.ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, SETTINGS, callbacks, colorstr, ops
from yolov8.ultralytics.yolo.utils.checks import check_file, check_imgsz, check_imshow, print_args, check_requirements
from yolov8.ultralytics.yolo.utils.files import increment_path
from yolov8.ultralytics.yolo.utils.torch_utils import select_device
from yolov8.ultralytics.yolo.utils.ops import Profile, non_max_suppression, scale_boxes, process_mask, process_mask_native
from yolov8.ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

from trackers.multi_tracker_zoo import create_tracker

from collections import deque
import time
from time import strftime
from time import gmtime
from plotting_speed import plot_centroid, rgb
import yaml
import matplotlib.pyplot as plt
import gradio as gr
import subprocess

@torch.no_grad()
def run(
        source='0',
        yolo_weights=WEIGHTS / 'yolov5m.pt',  # model.pt path(s),
        reid_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        tracking_method='strongsort',
        tracking_config=None,
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=False,  # show results
        save_txt=True,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        save_trajectories=False,  # save trajectories for each track
        save_vid=False,  # save confidences in --save-txt labels
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs' / 'track',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        retina_masks=False,
):

    source = str(source)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    if not isinstance(yolo_weights, list):  # single yolo model
        exp_name = yolo_weights.stem
    elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
        exp_name = Path(yolo_weights[0]).stem
    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    exp_name = name if name else exp_name + "_" + reid_weights.stem
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    is_seg = '-seg' in str(yolo_weights)
    model = AutoBackend(yolo_weights, device=device, dnn=dnn, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_imgsz(imgsz, stride=stride)  # check image size

    # Dataloader
    bs = 1
    if webcam:
        show_vid = check_imshow(warn=True)
        dataset = LoadStreams(
            source,
            imgsz=imgsz,
            stride=stride,
            auto=pt,
            transforms=getattr(model.model, 'transforms', None),
            vid_stride=vid_stride
        )
        bs = len(dataset)
    else:
        dataset = LoadImages(
            source,
            imgsz=imgsz,
            stride=stride,
            auto=pt,
            transforms=getattr(model.model, 'transforms', None),
            vid_stride=vid_stride
        )
    vid_path, vid_writer, txt_path = [None] * bs, [None] * bs, [None] * bs
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup

    # Create as many strong sort instances as there are video sources
    tracker_list = []
    for i in range(bs):
        tracker = create_tracker(tracking_method, tracking_config, reid_weights, device, half)
        tracker_list.append(tracker, )
        if hasattr(tracker_list[i], 'model'):
            if hasattr(tracker_list[i].model, 'warmup'):
                tracker_list[i].model.warmup()
    outputs = [None] * bs

    with open('parameters.yaml', 'r') as file:
        parameters = yaml.safe_load(file)
    # Run tracking
    M = parameters['M']
    pts = [deque(maxlen=M) for _ in range(9999)]
    fn = simple = V_all = num = v_all_th = scale = n = 0

    start = time.perf_counter()
    #model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile(), Profile())
    curr_frames, prev_frames = [None] * bs, [None] * bs
    for frame_idx, batch in enumerate(dataset):
        path, im, im0s, vid_cap, s = batch
        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
        with dt[0]:
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255.0  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            preds = model(im, augment=augment, visualize=visualize)

        # Apply NMS
        with dt[2]:
            if is_seg:
                masks = []
                p = non_max_suppression(preds[0], conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)
                proto = preds[1][-1]
            else:
                p = non_max_suppression(preds, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            
        # Process detections
        for i, det in enumerate(p):  # detections per image
            seen += 1
            if webcam:  # bs >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                p = Path(p)  # to Path
                s += f'{i}: '
                txt_file_name = p.name
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                # video file
                if source.endswith(VID_FORMATS):
                    txt_file_name = p.stem
                    save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                # folder with imgs
                else:
                    txt_file_name = p.parent.name  # get folder name containing current img
                    save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...
            curr_frames[i] = im0

            txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop

            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            
            if hasattr(tracker_list[i], 'tracker') and hasattr(tracker_list[i].tracker, 'camera_update'):
                if prev_frames[i] is not None and curr_frames[i] is not None:  # camera motion compensation
                    tracker_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

            if det is not None and len(det):
                if is_seg:
                    shape = im0.shape
                    # scale bbox first the crop masks
                    if retina_masks:
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], shape).round()  # rescale boxes to im0 size
                        masks.append(process_mask_native(proto[i], det[:, 6:], det[:, :4], im0.shape[:2]))  # HWC
                    else:
                        masks.append(process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True))  # HWC
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], shape).round()  # rescale boxes to im0 size
                else:
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # pass detections to strongsort
                with dt[3]:
                    outputs[i] = tracker_list[i].update(det.cpu(), im0)

                num = 0
                V_sum, V_all, V_a = 0, 0, 0
                angle = 0
                scale = 1
                simple = 0
                ablation = 1
                v_th = 5.2 if M==50 else 8

                v_all_th = 5.2 if scale else 250
                # distance = 0
                
                # draw boxes for visualization
                if len(outputs[i]) > 0:
                    
                    if is_seg:
                        # Mask plotting
                        annotator.masks(
                            masks[i],
                            colors=[colors(x, True) for x in det[:, 5]],
                            im_gpu=torch.as_tensor(im0, dtype=torch.float16).to(device).permute(2, 0, 1).flip(0).contiguous() /
                            255 if retina_masks else im[i]
                        )
                    
                    for j, (output) in enumerate(outputs[i]):
                        
                        bbox = output[0:4]
                        id = output[4]
                        cls = output[5]
                        conf = output[6]
                        # to MOT format
                        bbox_left = output[0]
                        bbox_top = output[1]
                        bbox_w = output[2] - output[0]
                        bbox_h = output[3] - output[1]

                        # L = 1.24
                        # W, H = 1.2, 0.35
                        # k = 12960
                        # ks = 0.000020

                        L = parameters['L']
                        W = parameters['W']
                        H = parameters['H']
                        k = parameters['k']
                        ks = parameters['ks']

                        depth_coef = 1
                        distance_coef = 1
                        x = bbox_left + bbox_w/2
                        y = bbox_top + bbox_h/2
                        l = np.sqrt(bbox_w**2 + bbox_h**2)
                        
                        bbox_right = bbox_left + bbox_w
                        bbox_bottom = bbox_top + bbox_h

                        aspect = 1
                        if scale:
                            sx = W/bbox_w * aspect
                            sy = H/bbox_h * aspect
                            # sz = np.sqrt(sx**2+sy**2)*depth_coef
                            sz = ks / np.sqrt(sx**2+sy**2)*depth_coef
                            # v_th =5.2
                        else:
                            # sx, sy, sz = 0.03, 0.03, 0.03
                            sx, sy, sz = 1, 1, 1
                            # v_th = 1000
                        
                        z = k/l
                        # distance = get_arrow_distance(im0, pts[id], vec_len=10, color=(255, 153, 153), thickness=3, tip_length=0.3, alpha=0.5)
                        # z1 = k/l if distance==0 else k/distance

                        centers = [int(x), int(y), z, frame_idx]
                        pts[id].append(centers)
                        cv2.circle(im0, centers[:2], 1, (0,255,0), 2)
                        # print(pts[id])
                        dR_sum = 0
                        # v_sum = 0 
                        fn_sum = 0
                        V, V_average=0, 0
                        v_xy, v_r, v_s = 0, 0, 0
                        outside = False
                        angle = 0
                        dx_sum = 0
                        dy_sum = 0
                        vec_len = 10

                        dx_mean = dx_sum / vec_len
                        dy_mean = dy_sum / vec_len

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

                        # for jj in range(1, len(pts[id])):
                        #     if jj >= len(pts[id]) - vec_len:
                        #             vec_dx, vec_dy = pts[id][jj][0] - pts[id][jj-1][0], pts[id][jj][1] - pts[id][jj-1][1]
                        #             dx_sum += vec_dx
                        #             dy_sum += vec_dy
                        # distance = get_arrow_distance(im0, pts[id], vec_len=10, color=(255, 153, 153), thickness=3, tip_length=0.3, alpha=0.5)

                        for jj in range(1, len(pts[id])):
                            if pts[id][jj - 1][:2] is None or pts[id][jj][:2] is None:
                                continue
                            ra, rb =  pts[id][jj], pts[id][jj-1] # after and before 

                            
                            # z1 = k/l if distance==0 else k/distance
                            
                            dx, dy, dz = ra[0]-rb[0], ra[1]-rb[1], ra[2]-rb[2]

                            dxy = np.sqrt(dx**2+dy**2)
                            dr = np.sqrt(dx**2+dy**2+dz**2)
                            # dR = dr * ra[3] *18
                            dX, dY, dZ = sx*dx, sy*dy, sz*dz
                            # print(f"dR=({dX}, {dY}, {dZ})")

                            dR = np.sqrt(dX**2 + dY**2 + dZ**2) * distance_coef
                            # dR = np.sqrt((sx*dx)**2 + (sy*dy)**2 + (sz*dz)**2)
                            # print(f"XYZ = ({sx*dx**2}, {sy*dy**2}, {sz*dz**2})")
                            fn = ra[3] - rb[3]
                            if bbox_left < 5 or bbox_right > 1915 or bbox_top < 5 or bbox_bottom > 1075:
                                outside = True
                            else:
                                outside = False

                            V = dR * 30 * 3.6 / fn if scale else dR * 30/ fn
                            v_xy = dxy * 30 * 3.6 / fn
                            v_r = dr * 30 * 3.6 / fn
                            v_s = dR * 30/ fn
                            dR_sum += dR
                            fn_sum += fn

                            V_average = dR_sum * 30 * 3.6 / fn_sum if scale else dR_sum * 30 / fn_sum
                            c_color = rgb(0, 12, V) if M==50 else rgb(0,20,V)

                            thickness = int(np.sqrt(64 / (float(jj + 1))**0.6))
                            cv2.line(im0,(pts[id][jj-1][:2]), (pts[id][jj][:2]),c_color,thickness)

                            if jj >= len(pts[id]) - vec_len:
                                    vec_dx, vec_dy = pts[id][jj][0] - pts[id][jj-1][0], pts[id][jj][1] - pts[id][jj-1][1]
                                    dx_sum += vec_dx
                                    dy_sum += vec_dy

                        im0 = draw_arrow(im0, pts[id], 10, (255, 153, 153), 3, 0.3, 0.5)
                        distance = get_arrow_distance(im0, pts[id], vec_len=10, color=(255, 153, 153), thickness=3, tip_length=0.3, alpha=0.5)
                        
                        num += 1 if not outside else 0

                        V_sum += V_average
                        V_all = V_sum / num if num > 1 else V_sum
                        V_a = V / num if num > 1 else V

                        if save_txt:
                            # Write MOT compliant results to file
                            # with open(txt_path + '.txt', 'a') as f:
                            #     f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                            #                                    bbox_top, bbox_w, bbox_h, -1, -1, -1, i))
                            fx = round(x, 1)
                            fy = round(y, 1)
                            fz = round(z, 1)
                            txt_path2 = str(save_dir / 'tracks' / 'test')
                            txt_path3 = str(save_dir / 'tracks' / 'ablation')
                            txt_path4 = str(save_dir / 'tracks' / 'simple')
                            txt_path5 = str(save_dir / 'tracks' / 'velocity')
                            
                            # fz1 = round(z1, 1)
                            with open(txt_path + '.txt', 'a') as f:
                                    f.write(('%g ' * 8 + '\n') % (frame_idx + 1, id, V_average, V, x,  # MOT format
                                                                y, fz, n))
                            with open(txt_path2 + '.txt', 'a') as f:
                                f.write(('%g ' * 7 + '\n') % (frame_idx + 1, id, x,  # test
                                                                y, fz, V, n))
                            with open(txt_path3 + '.txt', 'a') as f:
                                f.write(('%g ' * 12 + '\n') % (frame_idx + 1, id, V_average, V, x,  # ablation
                                                            y, fz, num, n, v_xy, v_r, v_s))
                            with open(txt_path4 + '.txt', 'a') as f:
                                f.write(('%g ' * 5 + '\n') % (frame_idx + 1, id, x, y, fz))
                            with open(txt_path5 + '.txt', 'a') as f:
                                f.write(('%g ' * 3 + '\n') % (frame_idx + 1, id, V))

                        if save_vid or save_crop or show_vid:  # Add bbox/seg to image
                            c = int(cls)  # integer class
                            id = int(id)  # integer id
                            # label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                            #     (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
                            color = colors(c, True)

                            if scale: 
                              if simple: 
                                  label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                                  (f'{id} {conf:.2f}' if hide_class else f'{id}: {V_average:.1f}km/h ({V:.1f}), {distance:.1f}px'))
                              else:
                                  label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                                  (f'{id} {conf:.2f}' if hide_class else f'{id}: {V_average:.1f}km/h ({V:.1f}), {distance:.1f}px'))

                            else:
                              label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                                  (f'{id} {conf:.2f}' if hide_class else f'{id}: {V_average:.1f}px/s ({V:.1f}) <{angle:.1f}>'))
                     
                            # dx_mean = dx_sum / vec_len
                            # dy_mean = dy_sum / vec_len

                            # start_point = (int(pts[id][-1][0]), int(pts[id][-1][1]))
                            # end_point = (int(pts[id][-1][0] + dx_mean*10), int(pts[id][-1][1] + dy_mean*10))

                            # overlay = im0.copy()
                            # cv2.arrowedLine(im0, start_point, end_point, (255, 153, 153), 3, tipLength=0.3, line_type=cv2.LINE_AA)
                            # alpha = 0.5
                            # im0 = cv2.addWeighted(overlay, alpha, im0, 1 - alpha, 0)

                            if outside:
                                annotator = Annotator(im0, line_width=1, example=str(names))
                                annotator.box_label(bbox, label=None, color=(255, 20, 2))
                                # plot_one_box(bboxes, im0, label=None, color=(255, 20, 2), line_thickness=1)
                            # elif V_average > v_th and simple == 0:
                            elif V_average > 6 and simple == 0:
                                annotator = Annotator(im0, line_width=3, example=str(names))
                                annotator.box_label(bbox, label, color=(2, 2, 255))
                            else:
                                annotator = Annotator(im0, line_width=3, example=str(names))
                                annotator.box_label(bbox, label, color=(2, 200, 2))
                        
                            # annotator.box_label(bbox, label, color=color)
                            
                            if save_trajectories and tracking_method == 'strongsort':
                                q = output[7]
                                tracker_list[i].trajectory(im0, q, color=color)
                            if save_crop:
                                txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                save_one_box(np.array(bbox, dtype=np.int16), imc, file=save_dir / 'crops' / txt_file_name / names[c] / f'{id}' / f'{p.stem}.jpg', BGR=True)
                            
                            # if keypoints:
            else:
                pass
                #tracker_list[i].tracker.pred_n_update_all_tracks()
                
            
            if simple==0 and save_vid:
                if V_all > v_all_th:
                    cv2.putText(im0, 'Speed: {:.1f}km/h'.format(V_all) if scale else 'Speed: {:.1f}px/s'.format(V_all),
                        (430, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 250), thickness=3)    
                else:    
                    cv2.putText(im0, 'Speed: {:.1f}km/h'.format(V_all) if scale else 'Speed: {:.1f}px/s'.format(V_all),
                        (430, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 255, 0), thickness=2)
                cv2.putText(im0, 'num: {:.0f} ({:.0f})'.format(n, num),
                    (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 100), thickness=2)    
                T = (frame_idx + 1) / 30
                T2 = strftime("%M:%S", gmtime(T))
                cv2.putText(im0, f'{T2}',
                    (1820, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 128, 0), thickness=2)

                t_ms = sum([dt.dt for dt in dt if hasattr(dt, 'dt')]) * 1E3
                fps_ms = 1000 / t_ms 
                cv2.putText(im0, 'FPS: {:.0f}'.format(fps_ms),
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), thickness=2)   
            
            # Stream results
            im0 = annotator.result()
            if show_vid:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # 1 millisecond
                    exit()

            # Save results (image with detections)
            if save_vid:
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)

            prev_frames[i] = curr_frames[i]
        
        if V_all > v_all_th:
            # Print total time (preprocessing + inference + NMS + tracking)
            LOGGER.warning(f"{s}{'' if len(det) else '(no detections), '}{sum([dt.dt for dt in dt if hasattr(dt, 'dt')]) * 1E3:.1f}ms, \033[31mspeed:({V_all:.1f})\033[0m, num:({num})")  
        else:
            # Print total time (preprocessing + inference + NMS)
            LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{sum([dt.dt for dt in dt if hasattr(dt, 'dt')]) * 1E3:.1f}ms, speed:({V_all:.1f}), num:({num})")
    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    # t = tuple(x.t / seen * 1E3 if seen > 0 else 0 for x in dt)
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms {tracking_method} update per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        s = f"\n{len(list((save_dir / 'tracks').glob('*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        end = time.perf_counter()
        dt = (end-start)/60
        LOGGER.info(f'({dt:.2f} mins)')
    if update:
        strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning)
        
    #plot
    txt_path5 = str(save_dir / 'tracks' / 'velocity')
    plot_file = txt_path5 + '.txt'
    # frame_idx + 1, id, V
    def plot_velocities(txt_path):
        # Load data from file
        data = np.loadtxt(txt_path)

        # Extract columns of frame_idx, id, and V
        frame_idx = data[:, 0]
        ids = np.unique(data[:, 1]).astype(int)
        V = data[:, -1]

        # Plot frame_idx vs V for each id
        for id in ids:
            mask = (data[:, 1] == id)
            id_frame_idx = frame_idx[mask]
            id_V = V[mask]
            plt.plot(id_frame_idx, id_V, label=f'ID {id}')

        plt.xlabel('frame_idx')
        plt.ylabel('V')
        plt.legend()
        plt.show()

    plot_velocities(plot_file)
    
    def tree(dir_path, padding=''):
        print(padding[:-1] + '+--' + os.path.basename(dir_path) + '/')
        padding += ' '
        files = os.listdir(dir_path)
        for file in files:
            path = os.path.join(dir_path, file)
            if os.path.isdir(path):
                tree(path, padding + '| ')
            else:
                print(padding + '|--' + file)
            
            
    crop_path = str(save_dir / 'crops' / 'tuna')
    # tree(crop_path)
    with open('tree.txt', 'w') as file:
        tree(crop_path, file=file)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=Path, default=WEIGHTS / 'best.pt', help='model.pt path(s)')
    parser.add_argument('--reid-weights', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--tracking-method', type=str, default='bytetrack', help='strongsort, ocsort, bytetrack')
    parser.add_argument('--tracking-config', type=Path, default=None)
    parser.add_argument('--source', type=str, default='./videos/tuna.mp4', help='file/dir/URL/glob, 0 for webcam')  
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt', default=1)
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes', default=1)
    parser.add_argument('--save-trajectories', action='store_true', help='save trajectories for each track')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results', default=1)
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs' / 'track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--retina-masks', action='store_true', help='whether to plot masks in native resolution')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    opt.tracking_config = ROOT / 'trackers' / opt.tracking_method / 'configs' / (opt.tracking_method + '.yaml')
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
    # txt_path5 = str(save_dir / 'tracks' / 'velocity')
    # plot_file = txt_path5 + '.txt'
    # # frame_idx + 1, id, V
    # plot_velocity(plot_file)

