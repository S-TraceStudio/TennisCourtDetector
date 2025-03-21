import cv2
import numpy as np
import torch
from tracknet import BallTrackerNet
import torch.nn.functional as F
from postprocess import postprocess, refine_kps
from homography import get_trans_matrix, refer_kps
import argparse
from utils import displayDebugImage
from court_line_pixel_detector import CourtLinePixelDetector
from court_line_candidate_detector import CourtLineCandidateDetector
from tennis_court_fitter import TennisCourtFitter

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='path to model')
    parser.add_argument('--input_path', type=str, help='path to input image')
    parser.add_argument('--output_path', type=str, help='path to output image')
    parser.add_argument('--use_refine_kps', action='store_true', help='whether to use refine kps postprocessing')
    parser.add_argument('--use_homography', action='store_true', help='whether to use homography postprocessing')
    parser.add_argument('--debug', action='store_true', help='whether to display images for debugging')
    parser.add_argument('--cv', action='store_true', help='whether to use classical Computer Vision method or ML method')

    args = parser.parse_args()

    if args.cv:
        image = cv2.imread(args.input_path)

        court_line_pixel_detector = CourtLinePixelDetector()
        court_line_candidate_detector = CourtLineCandidateDetector()
        tennis_court_fitter = TennisCourtFitter()

        binary_image = court_line_pixel_detector.run(image,args.debug)
        lines = court_line_candidate_detector.run(binary_image,image)
        model = tennis_court_fitter.run(lines,binary_image,image)


        if args.debug:
            #print("Image")
            #displayDebugImage(image)
            print("Binary image")
            model.draw_model(image,color=(0,255,255))
            displayDebugImage(image)

    else:
        model = BallTrackerNet(out_channels=15)
        print("Cuda available : ", torch.cuda.is_available())
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()

        OUTPUT_WIDTH = 640
        OUTPUT_HEIGHT = 360

        image = cv2.imread(args.input_path)
        img = cv2.resize(image, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
        inp = (img.astype(np.float32) / 255.)
        inp = torch.tensor(np.rollaxis(inp, 2, 0))
        inp = inp.unsqueeze(0)

        h, w, c = image.shape
        print('width: ', w)
        print('height: ', h)
        print('channel:', c)

        scaleX = w / OUTPUT_WIDTH
        scaleY = h / OUTPUT_HEIGHT

        print('scaleX:', scaleX)
        print('scaleY:', scaleY)

        out = model(inp.float().to(device))[0]
        pred = F.sigmoid(out).detach().cpu().numpy()

        points = []
        for kps_num in range(14):
            heatmap = (pred[kps_num]*255).astype(np.uint8)
            x_pred, y_pred = postprocess(heatmap, scaleX, scaleY, low_thresh=170, max_radius=25)
            if args.use_refine_kps and kps_num not in [8, 12, 9] and x_pred and y_pred:
                x_pred, y_pred = refine_kps(image, int(y_pred), int(x_pred), debug=args.debug)
            points.append((x_pred, y_pred))

        if args.use_homography:
            matrix_trans = get_trans_matrix(points)
            if matrix_trans is not None:
                points = cv2.perspectiveTransform(refer_kps, matrix_trans)
                points = [np.squeeze(x) for x in points]

        for j in range(len(points)):
            if points[j][0] is not None:
                image = cv2.circle(image, (int(points[j][0]), int(points[j][1])), radius=0, color=(0, 0, 255), thickness=10)

        cv2.imwrite(args.output_path, image)

        # Display result
        displayDebugImage(image)
