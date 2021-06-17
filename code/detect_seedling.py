from cv2 import cv2
from skimage.util import img_as_ubyte, img_as_float
from pathlib import Path
from tqdm import tqdm

import argparse
import numpy as np
import csv
import os


def normalize(img):
    """Normalize the input image from [0, 1] to [0, 255].
    
    Parameters:
        img: ndarray. img.shape is (m, n, 1)

    Return:
        img_result: ndarray.
            The normalized image.
    """
    img_result = 255 * (img - img.min()) / (img.max() - img.min())
    img_result = img_result.astype(np.uint8)

    return img_result


def complement(img):
    """Calculate complement of the input image.
    
    Parameters:
        img: ndarray. img.shape is (m, n, 1)

    Return:
        img_result: ndarray.
            The complement image.
    """
    img_result = 255 - img

    return img_result

    
def enhance_contrast(img):
    """Enhance contrast of the input image using CLAHE.

    Parameters:
        img: ndarray. img.shape is (m, n, 1)

    Return:
        img_result: ndarray.
            The enhanced image.
    """
    clahe = cv2.createCLAHE(clipLimit = 5)
    img_result = clahe.apply(img)

    return img_result


def calcVEG(img):
    """Generate vi image: VEG.

    Parameters:
        img: ndarray.
            Input image with BGR color space.

    Return:
        VEG: ndarray.
            The vi image.
    """
    B, G, R = cv2.split(img)

    blue = img_as_float(B)
    green = img_as_float(G)
    red = img_as_float(R)

    fraction = green
    denominator = (red**0.667) * (blue**0.333)

    VEG = np.divide(fraction, denominator, out=np.zeros_like(green), where=denominator!=0)
    VEG = normalize(VEG)

    return VEG


def calcCIVE(img):
    """Generate vi image: CIVE.

    Parameters:
        img: ndarray.
            Input image with BGR color space.
            
    Return:
        CIVE: ndarray.
            The vi image.
    """
    B, G, R = cv2.split(img)

    blue = img_as_float(B)
    green = img_as_float(G)
    red = img_as_float(R)

    CIVE = 0.441*red - 0.811*green + 0.385*blue + 18.78745
    CIVE = normalize(CIVE)

    return CIVE

    
def calcExG(img):
    """Generate vi image: ExG.

    Parameters:
        img: ndarray.
            Input image with BGR color space.
            
    Return:
        ExG: ndarray.
            The vi image.
    """        
    B, G, R = cv2.split(img)

    R = img_as_float(R)
    G = img_as_float(G)
    B = img_as_float(B)

    r = np.divide(R, (R+G+B), out=np.zeros_like(R), where=(R+G+B)!=0)
    g = np.divide(G, (R+G+B), out=np.zeros_like(R), where=(R+G+B)!=0)
    b = np.divide(B, (R+G+B), out=np.zeros_like(R), where=(R+G+B)!=0)

    ExG = 2*g - r - b
    ExG = normalize(ExG)

    return ExG
    

def do_preprocess(img, mode='gray'):
    """Generate the pre-processing image by different pre-processing 
    
    Parameters:
        img: ndarray. BGR color space. img.shape is (m, n, 3)
        
        mode: str, default is gray.
            One of the following strings, selecting the mode for pre-processing:
                 - 'gray', 'clahe', 'exg', 'exr', 'mexg', 'tgi', 'com2'

    Return:
        img_result: ndarray.
            The pre-processing image.
    """
    mode = mode.lower()
    
    allowedmode = ['gray', 'clahe', 'exg', 'exr', 'mexg', 'tgi', 'com2']

    if mode not in allowedmode:
        raise ValueError("mode should be 'gray', 'clahe', 'exg', 'exr', 'mexg', 'tgi' or 'com2'")
            
    if mode == 'gray' or mode == 'clahe':
        img_result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    else:
        if mode == 'com2':
            ExG = calcExG(img)
            CIVE = calcCIVE(img)
            VEG = calcVEG(img)
            img_result = 0.36*ExG + 0.47*CIVE + 0.17*VEG
            img_result = normalize(img_result)
        
        elif mode == 'exg':
            img_result = calcExG(img)
        
        else:
            B, G, R = cv2.split(img)

            blue = img_as_float(B)
            green = img_as_float(G)
            red = img_as_float(R)
            
            if mode == 'exr':
                img_result = 1.3*red - green

            elif mode == 'mexg':
                img_result = 1.262*green - 0.884*red - 0.311*blue

            elif mode == 'tgi':
                img_result = green - 0.39*red - 0.61*blue
        
            img_result = normalize(img_result)
        
        if mode == 'exg' or mode == 'mexg' or mode == 'tgi':
            img_result = complement(img_result)
        
    if mode != 'gray':
        img_result = enhance_contrast(img_result)
    
    return img_result


class RiceDetector:
    def __init__(self, minArea=15, maxArea=300):
        parameters = cv2.SimpleBlobDetector_Params()
#         parameters.filterByInertia = False
        parameters.filterByConvexity = False
        parameters.minArea = minArea
        parameters.maxArea = maxArea
        
        self.parameters = parameters
        
    
    def set_thresholdStep(self, test_block, rice_num):
        overflow = 1.2 * rice_num
        thresholdStep = []
        
        params = self.parameters
        
        for i in range(50, 0, -1):
            params.thresholdStep = i
            detector = cv2.SimpleBlobDetector_create(params)
            
            keypoints_blob = detector.detect(test_block)
            kps_blob = len(keypoints_blob)
            
            if kps_blob > overflow:
                break
            
            info = {"thresholdStep": i, "gap": abs(kps_blob - rice_num)}
            thresholdStep.append(info)
        
        # use thresholdStep which gets the closest result with rice_num
        thresholdStep = sorted(thresholdStep, key = lambda item : item["gap"], reverse = False)
        params.thresholdStep = thresholdStep[0]["thresholdStep"]
        self.parameters = params
        
    
    def detect_rice(self, img, mask=None):
        if mask is not None:
            img = cv2.bitwise_and(img, img, mask=mask)
        
        detector = cv2.SimpleBlobDetector_create(self.parameters)
        keypoints = detector.detect(img)
        
        return keypoints
    
    
    def print_params(self):
        parameters = self.parameters
        
        print("thresholdStep =", parameters.thresholdStep)
#         print("minThreshold =", parameters.minThreshold)
#         print("maxThreshold =", parameters.maxThreshold)

#         print("minRepeatability =", parameters.minRepeatability)
#         print("minDistBetweenBlobs =", parameters.minDistBetweenBlobs)

#         print("filterByColor =", parameters.filterByColor)
#         if parameters.filterByColor:
#             print("blobColor =", parameters.blobColor)

#         print("filterByArea =", parameters.filterByArea)
#         if parameters.filterByArea:
#             print("minArea =", parameters.minArea)
#             print("maxArea =", parameters.maxArea)

#         print("filterByCircularity =", parameters.filterByCircularity)
#         if parameters.filterByCircularity:
#             print("minCircularity =", parameters.minCircularity)
#             print("maxCircularity =", parameters.maxCircularity)

#         print("filterByInertia =", parameters.filterByInertia)
#         if parameters.filterByInertia:
#             print("minInertiaRatio =", parameters.minInertiaRatio)
#             print("maxInertiaRatio =", parameters.maxInertiaRatio)

#         print("filterByConvexity =", parameters.filterByConvexity)
#         if parameters.filterByConvexity:
#             print("minConvexity =", parameters.minConvexity)
#             print("maxConvexity =", parameters.maxConvexity)


def _parse_args():
    """Argument parser: parse the folder path and file path of img for setting the parameters of the detector.
    
    Argument:
        --inputpath
        --maskpath
        --outputpath
        --riceImg_outputpath
        --img_for_setting
    """
    parser = argparse.ArgumentParser(description='ArgParser for detect_seedling.py')
    parser.add_argument('--inputpath', default='../input/', type=str, help='folder path to input images')
    parser.add_argument('--maskpath', default='../output/paddy/slic/', type=str, help='folder path to input paddy mask')
    
    parser.add_argument('--outputpath', default='../output/rice/', type=str, help='folder path to output result')
    parser.add_argument('--riceImg_outputpath', default='../output/rice/rice_img/', type=str, help='folder path to save rice_img')
    
    parser.add_argument('--img_for_setting', default='../input_all/DSC07299.JPG', type=str, help='file path of img for setting the parameters of the detector')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    """
    The program detects rice seedling from the input image.

    For each preprocessing image in each input image, the program outputs the detection result as:
        1. an image where the detected seedlings are circled.
        2. a csv file which saves the position of each detected seedling.
    """

    args = _parse_args()
    
    preprocess = ['Gray', 'CLAHE', 'ExG', 'ExR', 'MExG', 'TGI', 'COM2']
    
    inputpath = args.inputpath
    inputfiles = sorted([file for file in os.listdir(inputpath)])
    
    maskpath = args.maskpath
    outputpath = args.outputpath
    
    # build a folder for saving rice img
    riceImg_outputpath = args.riceImg_outputpath
    Path(riceImg_outputpath).mkdir(parents=True, exist_ok=True)
    
    # build folders for saving result by each preprocess
    for name in preprocess:
        directory_path = outputpath + name
        Path(directory_path).mkdir(parents=True, exist_ok=True)
    
    print("set the parameters of the detector for each preprocessing")
    
    detector_list = []
    rice_num = 252

    img = cv2.imread(args.img_for_setting)
    
    for name in preprocess:
        pre_img = do_preprocess(img, mode=name)
        
        print("adjust parameters for the pre-processing:", name)
        detector = RiceDetector()
        test_block = pre_img[2347:2847, 3965:4465].copy()
        detector.set_thresholdStep(test_block, rice_num)
        
        info = {"preprocess": name, "detector": detector}
        detector_list.append(info)
    
#         keypoints_gray = detector.detect_rice(gray, mask=mask_paddy)

    for i in tqdm(range(len(inputfiles))):
        name = inputfiles[i].split(".")[0]
        
        img = cv2.imread(inputpath + inputfiles[i])
        print("read img:", inputfiles[i])
        
        inputmask = name + "_paddymask.jpg"
        paddy_mask = cv2.imread(maskpath + inputmask, cv2.IMREAD_GRAYSCALE)
        print("read mask:", inputmask)
        
        for item in detector_list:
            preprocess_name = item["preprocess"]

            rice_outputpath = outputpath + preprocess_name + "/"

            print("rice seedling detection by", preprocess_name)

            pre_img = do_preprocess(img, mode=preprocess_name)
            img_rice = pre_img.copy()
            
            detector = item["detector"]
#             detector.print_params()
            keypoints_blob = detector.detect_rice(pre_img, mask=paddy_mask)
            
            img_rice = cv2.drawKeypoints(img_rice, keypoints_blob, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imwrite(riceImg_outputpath + name + "_" + preprocess_name + ".jpg", img_rice)
            
            kps_blob = len(keypoints_blob)
            print("kps_" + preprocess_name + "=", kps_blob)
            
            print("write detection result into a csv file")
            seedlings = []
            for keypt in keypoints_blob:
                info = {'keypoints': str(keypt.pt)}
                seedlings.append(info)
            
            csvname = name + "_" + preprocess_name + ".csv"
            with open(rice_outputpath + csvname, 'w', newline='') as csvfile:
                fieldnames = ['keypoints']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for i in seedlings:
                    writer.writerow(i)