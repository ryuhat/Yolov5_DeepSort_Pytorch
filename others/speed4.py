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
from yolov8.ultralytics.yolo.utils.plotting import plot_one_box, plot_centroid, rgb, drawAxis, plot_speed_txt, angle_between

from collections import deque
# from utils.plots import plot_one_box, plot_centroid, rgb, drawAxis, save_one_box, plot_speed_txt, angle_between

from trackers.multi_tracker_zoo import create_tracker

VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes

M = 90
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
        save_txt=False,  # save results to *.txt
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
        line_thickness=2,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        retina_masks=False,
):

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    if not isinstance(yolo_weights, list):  # single yolo model
        exp_name = yolo_weights.stem
    elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
        exp_name = Path(yolo_weights[0]).stem
        yolo_weights = Path(yolo_weights[0])
    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    exp_name = name if name else exp_name + "_" + strong_sort_weights.stem
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
    save_dir = Path(save_dir)
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    model = attempt_load(Path(yolo_weights), map_location=device)  # load FP32 model
    names, = model.names,
    stride = model.stride.max()  # model stride
    imgsz = check_img_size(imgsz[0], s=stride.cpu().numpy())  # check image size

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride.cpu().numpy())
        nr_sources = 1
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        nr_sources = 1
    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

    # initialize StrongSORT
    cfg = get_config()
    cfg.merge_from_file(opt.config_strongsort)

    # Create as many strong sort instances as there are video sources
    strongsort_list = []
    for i in range(nr_sources):
        strongsort_list.append(
            StrongSORT(
                strong_sort_weights,
                device,
                half,
                max_dist=cfg.STRONGSORT.MAX_DIST,
                max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
                max_age=cfg.STRONGSORT.MAX_AGE,
                n_init=cfg.STRONGSORT.N_INIT,
                nn_budget=cfg.STRONGSORT.NN_BUDGET,
                mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
                ema_alpha=cfg.STRONGSORT.EMA_ALPHA,

            )
        )
        strongsort_list[i].model.warmup()
    outputs = [None] * nr_sources
    prev_outputs = [None] * nr_sources
        
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run tracking
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources
    # x_before, y_before, z_before = 0, 0, 0
    pts = [deque(maxlen=M) for _ in range(9999)]
    fn = 0
    start = time.perf_counter()
    
    for frame_idx, (path, im, im0s, vid_cap) in enumerate(dataset):
        HEIGHT, WIDTH = im0s.shape[0], im0s.shape[1]
        s = ''
        t1 = time_synchronized()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_synchronized()
        dt[0] += t2 - t1
        
        # print(im0s.shape)

        # Inference
        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
        pred = model(im)
        t3 = time_synchronized()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred[0], conf_thres, iou_thres, classes, agnostic_nms)
        dt[2] += time_synchronized() - t3

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if webcam:  # nr_sources >= 1
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
            txt2_path = str(save_dir / 'tracks' / 'speed')  # im.txt
            
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop
            
            fps = vid_cap.get(cv2.CAP_PROP_FPS)
            
            cv2.putText(im0, 'FPS: {:.0f}'.format(fps),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), thickness=2)     

            if cfg.STRONGSORT.ECC:  # camera motion compensation
                strongsort_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to strongsort
                t4 = time_synchronized()
                outputs[i] = strongsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_synchronized()
                dt[3] += t5 - t4
            
                # draw boxes for visualization
                num = 0
                V_sum, V_all, V_a = 0, 0, 0
                angle = 0
                scale = 1
                simple = 0
                ablation = 1
                v_th = 5.2 if M==50 else 8

                v_all_th = 5.2 if scale else 250
                if len(outputs[i]) > 0:
                    for j, (output, conf) in enumerate(zip(outputs[i], confs)):

                        bboxes = output[0:4]
                        bbox_left = output[0]
                        bbox_top = output[1]
                        bbox_w = output[2] - output[0]
                        bbox_h = output[3] - output[1]
                        id = int(output[4])
                        L = 1.24
                        W, H = 1.2, 0.35
                        k = 12960
                        ks = 0.000020

                        depth_coef = 1
                        distance_coef = 1
                        # bias = 0.65
                        
                        # print(f'{id}: {angle}')
                        # new_h = int(w*abs(np.sin(np.radians(angle))) + h*abs(np.cos(np.radians(angle))))
                        # new_w = int(h*abs(np.sin(np.radians(angle))) + w*abs(np.cos(np.radians(angle))))             
                        
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
                            sx, sy, sz = 1, 1, 1
                            # v_th = 1000
                        
                        z = k/l
                        
                        cls = output[5]
                        # center = [int(cx), int(cy)]
                        centers = [int(x), int(y), z, frame_idx]
                        # centers = [int(x), int(y), z, frame_idx, bbox_left, bbox_right, bbox_top, bbox_bottom]
                        # print(z)
                        pts[id].append(centers)
                        cv2.circle(im0, centers[:2], 1, (0,255,0), 2)
                        # print(pts[id])
                        dR_sum = 0
                        # v_sum = 0 
                        fn_sum = 0
                        V, V_average=0, 0
                        v_xy, v_r, v_s = 0, 0, 0
                        flag = False
                        angle = 0
                        # V_max = 10
                        for jj in range(1, len(pts[id])):
                            if pts[id][jj - 1][:2] is None or pts[id][jj][:2] is None:
                                continue
                            # rf = pts[id][jj-5] # -5
                            # rl = pts[id][-1] # last
                            ra, rb =  pts[id][jj], pts[id][jj-1] # after and before 
                            # ra = # after
                            dx, dy, dz = ra[0]-rb[0], ra[1]-rb[1], ra[2]-rb[2]
                            # left_a, left_b = ra[4], rb[4]
                            # right_a, right_b = ra[5], rb[5]
                            # top_a, top_b = ra[6], rb[6]
                            # bottom_a, bottom_b = ra[7], rb[7]
                            
                            angle = angle_between((rb[0], rb[1]), (1, 0)) 
                            # angle = angle_between((1, 0), (rb[0], rb[1])) 
                            # angle = angle_between((ra[0], ra[1]), (rb[0], rb[1])) 
                            # angle = angle_between((ra[0], ra[0]), (rb[0], rb[0])) 
                            # print(f'{id}: {angle}')
                            
                            dxy = np.sqrt(dx**2+dy**2)
                            dr = np.sqrt(dx**2+dy**2+dz**2)
                            # dR = dr * ra[3] *18
                            dX, dY, dZ = sx*dx, sy*dy, sz*dz
                            # print(f"dR=({dX}, {dY}, {dZ})")

                            dR = np.sqrt(dX**2 + dY**2 + dZ**2) * distance_coef
                            # dR = np.sqrt((sx*dx)**2 + (sy*dy)**2 + (sz*dz)**2)
                            # print(f"XYZ = ({sx*dx**2}, {sy*dy**2}, {sz*dz**2})")
                            fn = ra[3] - rb[3]
                            
                            # print(fn)
                            # if left_a == left_b or right_a == right_b or top_a == top_b or bottom_a == bottom_b:
                            # flag = False
                            # @with_goto
                            # if left_a < 5 or right_a > 1915 or top_a < 5 or bottom_a > 1075:
                            if bbox_left < 5 or bbox_right > 1915 or bbox_top < 5 or bbox_bottom > 1075:
                                # print(id, left_a, right_a, top_a, bottom_a)
                                flag = True
                                # num -= 1
                            else:
                                flag = False

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

                        num += 1 if not flag else 0

                        V_sum += V_average
                        V_all = V_sum / num if num > 1 else V_sum
                        V_a = V / num if num > 1 else V
                    
                        if scale: 
                            if simple: 
                                label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                                (f'{id} {conf:.2f}' if hide_class else f'{id}: {V_average:.1f}km/h ({V:.1f})'))
                            else:
                                label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                                (f'{id} {conf:.2f}' if hide_class else f'{id}: {V_average:.1f}km/h ({V:.1f})'))
                            #  <{angle:.1f}>

                        else:
                            label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                                (f'{id} {conf:.2f}' if hide_class else f'{id}: {V_average:.1f}px/s ({V:.1f}) <{angle:.1f}>'))
                                # (f'{id} {conf:.2f}' if hide_class else f'{id} {V:.1f}km/h ({x}, {y}, {z:.1f}) {aspect:.1f} {La:.1f}'))
                            # if pts[id][jj][:2]:
                        # if V_average > 42:
                        if flag:
                            plot_one_box(bboxes, im0, label=None, color=(255, 20, 2), line_thickness=1)
                        elif V_average > v_th and simple == 0:
                            plot_one_box(bboxes, im0, label=label, color=(2, 2, 255), line_thickness=3)
                        else:
                            plot_one_box(bboxes, im0, label=label, color=(2, 200, 2), line_thickness=3)
                        

                        if save_txt and not flag:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            fz = round(z, 1)
                            # Write MOT compliant results to file
                            with open(txt_path + '.txt', 'a') as f:
                                if ablation:
                                    f.write(('%g ' * 12+ '\n') % (frame_idx + 1, id, V_average, V, x,  # MOT format
                                                                y, fz, num, n, v_xy, v_r, v_s))
                                else:
                                    f.write(('%g ' * 8+ '\n') % (frame_idx + 1, id, V_average, V, x,  # MOT format
                                                                y, fz, n))

                                # f.write(('%g ' * 5 + '\n') % (frame_idx + 1, id, x,  # MOT format
                                #                             y, fz))
                                # f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                #                                bbox_top, bbox_w, bbox_h, -1, -1, -1, i))


                        if save_vid or save_crop or show_vid:  # Add bbox to image
                            c = int(cls)  # integer class
                            id = int(id)  # integer id
                            if save_crop:
                                txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                save_one_box(bboxes, imc, file=save_dir / 'crops' / txt_file_name / names[c] / f'{id}' / f'{p.stem}.jpg', BGR=True)
                        # label .end
                # V_average 
                print(f'{s}Done. YOLO:({t3 - t2:.3f}s), StrongSORT:({t5 - t4:.3f}s), speed:({V_all:.1f}), num:({num})')
                # print(f'<<{V_all:.1f}>>')
                if simple==0:
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
                    # cv2.putText(im0, '{:.0f}s'.format(T),
                    #     (1750, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 128, 0), thickness=2)   
                    cv2.putText(im0, f'{T2}',
                        (1820, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 128, 0), thickness=2)
                
                if save_txt:
                    if simple ==0:
                        with open(txt2_path + '.txt', 'a') as f:
                                    f.write(('%g ' * 4 + '\n') % (n, T, V_a, V_all))
                                
            else:
                strongsort_list[i].increment_ages()
                print('No detections')

            # Stream results
            if show_vid:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

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
                        # print("fps={fps}")      
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)

            prev_frames[i] = curr_frames[i]
            prev_outputs[i] = outputs[i]

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms strong sort update per image at shape {(1, 3, imgsz, imgsz)}' % t)
    if save_txt or save_vid:
        s = f"\n{len(list(save_dir.glob('tracks/*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        print(f"Results saved to {colorstr('bold', save_dir)}{s}")
        end = time.perf_counter()
        dt = (end-start)/60
        print(f'<<{dt:.2f} mins>>')
    if update:
        strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning)

# save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
# save_dir = Path(save_dir)
   

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=str, default=WEIGHTS / 'yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--strong-sort-weights', type=str, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--config-strongsort', type=str, default='strong_sort/configs/strong_sort.yaml')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')  
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    # parser.add_argument('--save-3d', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    
    # parser.add_argument('--scale', default=False, action='store_true', help='hide labels')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)