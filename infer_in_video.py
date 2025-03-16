import os
import cv2
import numpy as np
import torch
from tracknet import BallTrackerNet
import torch.nn.functional as F
from tqdm import tqdm
from postprocess import postprocess, refine_kps
from homography import get_trans_matrix, refer_kps
import argparse
import json
from utils import json_serialize, replace_file_extension


def read_video(path_video):
    """ Read video file
    :params
        path_video: path to video file
    :return
        frames: list of video frames
        fps: frames per second
    """
    cap = cv2.VideoCapture(path_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    cap.release()
    return frames, fps


def write_video(imgs_new, fps, path_output_video):
    height, width = imgs_new[0].shape[:2]
    out = cv2.VideoWriter(path_output_video, cv2.VideoWriter_fourcc(*'DIVX'),
                          fps, (width, height))
    for num in range(len(imgs_new)):
        frame = imgs_new[num]
        out.write(frame)
    out.release()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='path to model')
    parser.add_argument('--input_path', type=str, help='path to input video')
    parser.add_argument('--output_path', type=str, help='path to output video')
    parser.add_argument('--use_refine_kps', action='store_true', help='whether to use refine kps postprocessing')
    parser.add_argument('--use_homography', action='store_true', help='whether to use homography postprocessing')
    parser.add_argument('--draw_lines', action='store_true', help='whether to draw lines in the video result')
    args = parser.parse_args()

    model = BallTrackerNet(out_channels=15)
    print("Cuda available : ", torch.cuda.is_available())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    OUTPUT_WIDTH = 640
    OUTPUT_HEIGHT = 360

    frames, fps = read_video(args.input_path)
    frames_upd = []

    im = frames[0]
    h, w, c = im.shape
    print('width: ', w)
    print('height: ', h)
    print('channel:', c)

    scaleX = w / OUTPUT_WIDTH
    scaleY = h / OUTPUT_HEIGHT

    print('scaleX:', scaleX)
    print('scaleY:', scaleY)

    framePoints = []
    for image in tqdm(frames):
        img = cv2.resize(image, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
        inp = (img.astype(np.float32) / 255.)
        inp = torch.tensor(np.rollaxis(inp, 2, 0))
        inp = inp.unsqueeze(0)

        out = model(inp.float().to(device))[0]
        pred = F.sigmoid(out).detach().cpu().numpy()

        points = []
        for kps_num in range(14):
            heatmap = (pred[kps_num] * 255).astype(np.uint8)
            x_pred, y_pred = postprocess(heatmap, scaleX, scaleY, low_thresh=170, max_radius=25)
            if args.use_refine_kps and kps_num not in [8, 12, 9] and x_pred and y_pred:
                x_pred, y_pred = refine_kps(image, int(y_pred), int(x_pred), crop_size=40)
            points.append((x_pred, y_pred))

        if args.use_homography:
            matrix_trans = get_trans_matrix(points)
            if matrix_trans is not None:
                points = cv2.perspectiveTransform(refer_kps, matrix_trans)
                points = [np.squeeze(x) for x in points]

        for j in range(len(points)):
            if points[j][0] is not None:
                image = cv2.circle(image, (int(points[j][0]), int(points[j][1])), radius=0, color=(0, 0, 255), thickness=10)

            if args.draw_lines:
                lineColor = (0, 0, 0)
                lineThickness = 2
                image = cv2.line(image, (int(points[0][0]), int(points[0][1])), (int(points[2][0]), int(points[2][1])), thickness=lineThickness, color=lineColor)
                image = cv2.line(image, (int(points[4][0]), int(points[4][1])), (int(points[5][0]), int(points[5][1])), thickness=lineThickness, color=lineColor)
                image = cv2.line(image, (int(points[6][0]), int(points[6][1])), (int(points[7][0]), int(points[7][1])), thickness=lineThickness, color=lineColor)
                image = cv2.line(image, (int(points[1][0]), int(points[1][1])), (int(points[3][0]), int(points[3][1])), thickness=lineThickness, color=lineColor)

                image = cv2.line(image, (int(points[0][0]), int(points[0][1])), (int(points[1][0]), int(points[1][1])), thickness=lineThickness, color=lineColor)
                image = cv2.line(image, (int(points[8][0]), int(points[8][1])), (int(points[9][0]), int(points[9][1])), thickness=lineThickness, color=lineColor)
                image = cv2.line(image, (int(points[10][0]), int(points[10][1])), (int(points[11][0]), int(points[11][1])), thickness=lineThickness, color=lineColor)
                image = cv2.line(image, (int(points[2][0]), int(points[2][1])), (int(points[3][0]), int(points[3][1])), thickness=lineThickness, color=lineColor)

                image = cv2.line(image, (int(points[12][0]), int(points[12][1])), (int(points[13][0]), int(points[13][1])), thickness=lineThickness, color=lineColor)

        framePoints.append(points)

        frames_upd.append(image)

    write_video(frames_upd, fps, args.output_path)
    json_filename = replace_file_extension(args.output_path,".json")

    with open(json_filename,'w', encoding='utf-8') as f:
        json.dump(framePoints,f,ensure_ascii=False, indent=4,cls = json_serialize)

