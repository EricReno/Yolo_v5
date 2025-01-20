import os
import cv2
import time
import torch
import onnxruntime
import numpy as np
from config import parse_args
from tracker.vis_tools import plot_tracking
from tracker.byte_tracker import ByteTracker

def nms(bboxes, scores, nms_thresh):
    """"Pure Python NMS."""
    x1 = bboxes[:, 0]  #xmin
    y1 = bboxes[:, 1]  #ymin
    x2 = bboxes[:, 2]  #xmax
    y2 = bboxes[:, 3]  #ymax

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(1e-10, xx2 - xx1)
        h = np.maximum(1e-10, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-14)
        #reserve all the boundingbox whose ovr less than thresh
        inds = np.where(iou <= nms_thresh)[0]
        order = order[inds + 1]

    return keep

def preinfer(image, args):
    image_size = args.image_size
    ratio = [image_size/image.shape[1], image_size/image.shape[0]]

    output = cv2.resize(image, (image_size, image_size))
    output = output.transpose([2, 0, 1]).astype(np.float32)
    output /= 255.
    output = np.expand_dims(output, 0)

    return output, ratio

def postinfer(input, ratio, args):
    bboxes = input[0][:, :4]
    scores = input[0][:, 4:]

    labels = np.argmax(scores, axis=1)
    scores = scores[(np.arange(scores.shape[0]), labels)]
        
    keep = np.where(scores >= args.confidence_threshold)
    bboxes = bboxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    bboxes[..., [0, 2]] /= ratio[0]
    bboxes[..., [1, 3]] /= ratio[1]
    bboxes[..., [0, 2]] = np.clip(bboxes[..., [0, 2]], a_min=0., a_max=(args.image_size/ratio[0]))
    bboxes[..., [1, 3]] = np.clip(bboxes[..., [1, 3]], a_min=0., a_max=(args.image_size/ratio[1]))

    keep = nms(bboxes, scores, args.nms_threshold)

    scores = scores[keep]
    labels = labels[keep]
    bboxes = bboxes[keep]

    return labels, scores, bboxes

def build_tracker(args):
    tracker = ByteTracker(
        track_thresh=args.track_thresh,
        track_buffer=args.track_buffer,
        frame_rate=args.fps,
        match_thresh=args.match_thresh,
        mot20=False
    )
    
    return tracker

def run(args, tracker, detector):
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # for saving
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    save_size = (640, 480)
    cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    save_video_name = os.path.join(cur_time+'.avi')
    vid_writer = cv2.VideoWriter(save_video_name, fourcc, fps, save_size)
    
    # start tracking
    frame_id = 0
    results = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            # ------ Detetion -----
            infer_input, ratio = preinfer(frame, args)
            
            t0 = time.time()
            postinfer_input = detector.run(['output'], {'input': infer_input})
            print("=============== Frame-{} ================".format(frame_id))
            print("detect time: {:.1f} ms".format((time.time() - t0)*1000))
            
            labels, scores, bboxes = postinfer(postinfer_input, ratio, args)
            
            # ------- Track -------
            t2 = time.time()
            if len(bboxes) > 0:
                online_targets = tracker.update(scores, bboxes, labels)
                online_xywhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    xywh = t.xywh
                    tid = t.track_id
                    vertical = xywh[2] / xywh[3] > args.aspect_ratio_thresh
                    if xywh[2] * xywh[3] > args.min_box_area and not vertical:
                        online_xywhs.append(xywh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        results.append(
                            f"{frame_id}, {tid}, {xywh[0]:.2f}, {xywh[1]:.2f}, {xywh[2]:.2f}, {xywh[3]:.2f}, {t.score:.2f}, -1, -1, -1\n"
                        )
                print('tracking time: {:.1f} ms'.format((time.time() - t2)*1000))                    
                
                # plot tracking results
                online_im = plot_tracking(
                    frame, online_xywhs, online_ids, frame_id=frame_id+1, fps= 1. /(time.time() - t0)
                )
            else:
                online_im = frame
            
            # save results
            if args.save:
                frame_resized = cv2.resize(online_im, save_size)
                vid_writer.write(frame_resized)
            # show results
            if args.show:
                cv2.imshow('tracking', online_im)
                ch = cv2.waitKey(0)
                if ch == 27 or ch == ord("q") or ch == ord('Q'):
                    break
        else:
            break
        
        frame_id += 1   
        
    cap.release()
    vid_writer.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    np.random.seed(0)
    args = parse_args()
    
    args.video = '000006.mp4'
    args.onnx = 'deploy/person.onnx'

    providers = [('CUDAExecutionProvider', {'device_id': 0})]
    detector = onnxruntime.InferenceSession(args.onnx, providers=providers)
    
    tracker = build_tracker(args)
    
    run(args, tracker, detector)