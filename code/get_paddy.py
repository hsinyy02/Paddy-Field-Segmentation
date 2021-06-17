import os
from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse

from scipy import ndimage as ndi
from skimage import morphology
from skimage.morphology import disk
from skimage.future import graph
from skimage.segmentation import felzenszwalb, slic, watershed, mark_boundaries
from skimage.util import img_as_ubyte, img_as_float
from skimage.color import label2rgb
from skimage.filters import rank

from pathlib import Path
from tqdm import tqdm


def label2color(label, img, alpha=0.5, boundaries_color=None, boundaries_darken=None):
    """Return an color image where color-coded labels are painted over the image.
    boundaries between labeled regions can be highlighted by setting boundaries_color.
    
    Parameters:
        label: Integer array of labels with the same shape as img.
        
        img: Image used as underlay for labels. If the input is a color image, it’s converted to grayscale before coloring.
        
        alpha: Opacity of colorized labels.
        
        boundaries_color: length-3 sequence, for example, (1, 1, 0).
        Color of boundaries in the output image, which has the same color space with the input image. 
        If None, no boundary is drawn.
        
        boundaries_darken: length-3 sequence, for example, (1, 1, 0).
        Color surrounding boundaries in the output image. Ignored if boundaries_color is None.
    
    Return:
        colorimg: An image where color-coded labels are painted over the input image.
    """    
    
    colorimg = label2rgb(label, image=img, alpha=alpha)
    
    if boundaries_color is not None:
        colorimg = mark_boundaries(colorimg, label, color=boundaries_color, outline_color=boundaries_darken)
    
    colorimg = img_as_ubyte(colorimg)
    
    return colorimg


def watershed_segmentation(img_gray, threshold):
    """Apply watershed segmentation to the input image.
    
    The watershed transformation treats the image as a topographic map, 
    with the pixel value representing its height, and finds the lines that run along the tops of ridges. 
    (From Wikipedia, the free encyclopedia)
    
    This function does the same thing to the example from scikit-image
    https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_marked_watershed.html
    
    Parameters:
        img: ndarray. 
            Input image in grayscale.
        
        threshold: a value for thresholding each pixel to low gradient or not.
    
    Return:
        label_ws: ndarray.
            segmentation result with labels.
    """    

    # denoise img
    denoised = rank.median(img_gray, disk(2))
    
    # find continuous region (low gradient where less than threshold) --> markers
    # disk(5) is used here to get a more smooth image
    markers = rank.gradient(denoised, disk(5)) < threshold
    markers = ndi.label(markers)[0]
    
    # local gradient (disk(2) is used to keep edges thin)
    gradient = rank.gradient(denoised, disk(2))
    
    # process the watershed
    label_ws = watershed(gradient, markers, compactness=0)
    
    return label_ws


class PaddyDetector:
    def __init__(self, min_area=240000, slic_segments=1500, slic_compactness=10, rag_threshold=15, 
                 fz_scale=500, fz_min_size=3000, ws_threshold=30):
        """Parameterized constructor.
        
        Parameters:
            min_area: segment which is smaller than minArea will be ignored while paddy field classification
            
            slic_segments: parameter of n_segments in slic(img, n_segments, compactness).
            
            slic_compactness: parameter of compactness in slic(img, n_segments, compactness).
            
            rag_threshold: threshold value for RAG thresholding.
            
            fz_scale: parameter of scale in felzenszwalb(img, scale, min_size)
            
            fz_min_size: parameter of min_size in felzenszwalb(img, scale, min_size)
            
            ws_threshold: parameter of threshold in watershed_segmentation(img_gray, threshold)
        """
        self.min_area = min_area
        self.slic_segments = slic_segments
        self.slic_compactness = slic_compactness
        self.rag_threshold = rag_threshold
        self.fz_scale = fz_scale
        self.fz_min_size = fz_min_size
        self.ws_threshold = ws_threshold
        

    def build_dataset(self, imgpath, labelpath):
        """Build dataset for paddy classification. data save as dictionary of list. 
        [{"label": label, "hist": histogram}...]
    
        Parameters:
            imgpath: str. Folder path of the images in dataset.
            
            labelpath: str. Folder path of the labeled binary images in dataset.
        """            
        datasetfiles = [file for file in os.listdir(labelpath) if file.endswith('.jpg')]
        datasetfiles.sort()
        
        dataset = []
        # calculate hist of dataset
        for file in datasetfiles:
            label, name = file.split(".")[0].split("_")
            # read dataset img
            img = cv2.imread(imgpath + name + ".JPG")
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # read mask of labeled region
            mask = cv2.imread(labelpath + file, cv2.IMREAD_GRAYSCALE)
            
            # calculate histogram of labeled region
            hist = cv2.calcHist([img_hsv], [0,1], mask, [180,256], [0,180,0,256])
            cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            info = {"label": label, "hist": hist}
            dataset.append(info)
        
        self.dataset = dataset
    

    def is_paddy(self, hist):
        """Check the test region in an image with input histogram is paddy field or not.
    
        Parameters:
            hist: histogram of the test region in an image.
            
        Return:
            A bool value. True means the test region in an image with input histogram is paddy field.
        """    
        # calculate the match score between histogram of current segment 
        # and histogram of every label area in the image of the dataset
        comparehist = []
        for data in self.dataset:
            correl = cv2.compareHist(hist, data["hist"], cv2.HISTCMP_CORREL)
            info = {"label": data["label"], "correl": correl}
            comparehist.append(info)
            
        # sort the match score from the largest to the smallest
        compare = sorted(comparehist, key = lambda item : item["correl"], reverse = True)
        
        # classify current segment as paddy or NOT paddy by 3-NN classification.
        # only paddy_label > 1 will be classified as paddy
        paddy_label = 0
        for i in range(3):
            if compare[i]["correl"] < 0:
                continue
            if compare[i]["label"]=="paddy":
                paddy_label += 1
                    
        if paddy_label > 1:
            return True
        else:
            return False
    
    
    def get_paddy(self, img, mode='slic'):
        """Get a mask of paddy field from the input image with the specified mode.
    
        Parameters:
            img: ndarray.
                Input image with the color space BGR. 
            
            mode: str, optional
                 One of the following strings, selecting the type of method for image segmentation:
                 - 'slic' image segmentation using SLIC algorithm, and then merge the similar segments using RAG thresholding.
                 - 'felzenszwalb' image segmentation using felzenszwalb algorithm
                 - 'watershed' image segmentation using felzenszwalb algorithm
            
        Return:
            mask_paddy: ndarray.
                Return an binary image with white pixels indicate paddy field from the input image.
        """    
        mode = mode.lower()
        
        allowedmode = ['slic', 'felzenszwalb', 'watershed']

        if mode not in allowedmode:
            raise ValueError("mode should be 'slic', 'felzenszwalb', or 'watershed'")
            
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        mask_paddy = np.zeros(img.shape[:2], dtype = "uint8")
        
        if mode == 'slic':
            print("mode slic")
            # apply SLIC algorithm
            label_segment = slic(img, n_segments = self.slic_segments, compactness = self.slic_compactness)
        
            # merge segments which are similar in color by using RAG thresholding
            g = graph.rag_mean_color(img, label_segment)
            label_segment = graph.cut_threshold(label_segment, g, self.rag_threshold)
        
        elif mode == 'felzenszwalb':
            print("mode felzenszwalb")
            label_segment = felzenszwalb(img, scale=self.fz_scale, min_size=self.fz_min_size)
        
        elif mode == 'watershed':
            print("mode watershed")
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            label_segment = watershed_segmentation(img_gray, self.ws_threshold)        
            
        # classify each segment as paddy or not paddy
        label_id = np.unique(label_segment)
        
        for index in label_id:
            # calculate the area of current segment with label==index
            count = np.count_nonzero(label_segment == index)

            # if the area is smaller than 240000 pixels, ignore it
            if count < self.min_area:
                continue
        
            # compute a mask which is the region of current segment
            mask_segment = np.zeros(img.shape[:2], dtype = "uint8")
            mask_segment[label_segment == index] = 255

            # calculate the HS histogram of current segment in HSV color space
            hist_segment = cv2.calcHist([img_hsv], [0,1], mask_segment, [180,256], [0,180,0,256])
            cv2.normalize(hist_segment, hist_segment, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

            # check if current segment is paddy or not.
            paddy_status = self.is_paddy(hist_segment)

            if paddy_status:
                mask_paddy = cv2.bitwise_or(mask_paddy, mask_segment)
            
        return mask_paddy
    
    # morphological operation
    def mask_morphology(self, mask_paddy, mode='slic'):
        """Morphological operation on mask_paddy.
    
        Parameters:
            mask_paddy: ndarray.
                A binary image which is the result image from get_paddy(img, mode). 
            
            mode: str, optional
                 One of the following strings, selecting the case of mask_paddy for morphological operation:
                 - 'slic' & 'felzenszwalb' Apply opening operation the first. Then convex hull operation is applied.
                 - 'watershed' Apply closing operation at first, and then opening, and then convex hull operation.
            
        Return:
            mask_morphology: ndarray.
                Return an binary image after the morphological operation.
        """   
        allowedmode = ['slic', 'felzenszwalb', 'watershed']

        if mode not in allowedmode:
            raise ValueError("mode should be 'slic', 'felzenszwalb', or 'watershed'")
        
        mask_morphology = mask_paddy.copy()
        
        if mode == 'watershed':
            kernel = np.ones((25,25),np.uint8)
            mask_morphology = cv2.morphologyEx(mask_morphology, cv2.MORPH_CLOSE, kernel, iterations=1)
            
        # opening operation
        kernel = np.ones((45,45),np.uint8)
        mask_morphology = cv2.morphologyEx(mask_morphology, cv2.MORPH_OPEN, kernel, iterations=5)
        
        # convex hull
        mask_morphology = morphology.convex_hull_object(mask_morphology)
        mask_morphology = 255*mask_morphology
        mask_morphology = mask_morphology.astype(np.uint8)
        
        return mask_morphology


def _parse_args():
    """Argument parser: parse the folder path.
    
    Argument:
        --inputpath
        --dataset_imgpath
        --dataset_labelpath
        --outputpath
        --slic_outputpath
        --fz_outputpath
        --ws_outputpath
    """
    parser = argparse.ArgumentParser(description='ArgParser for get_paddy.py')
    parser.add_argument('--inputpath', default='../input/', type=str, help='folder path to input images')
    parser.add_argument('--dataset_imgpath', default='../dataset/img/', type=str, help='folder path to input dataset image')
    parser.add_argument('--dataset_labelpath', default='../dataset/label_mask/', type=str, help='folder path to input labeled image of dataset')
    
    parser.add_argument('--outputpath', default='../output/paddy/', type=str, help='folder path to output result')
    parser.add_argument('--slic_outputpath', default='../output/paddy/slic/', type=str, help='folder path to output paddy mask by slic method')
    parser.add_argument('--fz_outputpath', default='../output/paddy/felzenszwalb/', type=str, help='folder path to output paddy mask by felzenszwalb method')
    parser.add_argument('--ws_outputpath', default='../output/paddy/watershed/', type=str, help='folder path to output paddy mask by watershed method')
    
    args = parser.parse_args()
    return args

        
if __name__ == '__main__':
    """
    The program finds the paddy field of the input image with three segmentation methods respectively.
    Outputs will be 3 binary images with pixel value 1 for paddy field and 0 for NOT paddy field.

    For each input image, the program finds the paddy field of the image with a specific method,
    and then outputs a binary mask image with 1 for paddy field, and 0 for not paddy field.
    """

    args = _parse_args()

    inputpath = args.inputpath
    inputfiles = sorted([file for file in os.listdir(inputpath)])
    outputpath = args.outputpath
    
    slic_outputpath = args.slic_outputpath
    fz_outputpath = args.fz_outputpath
    ws_outputpath = args.ws_outputpath
    
    Path(outputpath).mkdir(parents=True, exist_ok=True)
    Path(slic_outputpath).mkdir(parents=True, exist_ok=True)
    Path(fz_outputpath).mkdir(parents=True, exist_ok=True)
    Path(ws_outputpath).mkdir(parents=True, exist_ok=True)
    
    dataset_imgpath = args.dataset_imgpath
    dataset_labelpath = args.dataset_labelpath
   
    pdetector = PaddyDetector()
    
    print("build dataset for paddy field segmentation")
    pdetector.build_dataset(dataset_imgpath, dataset_labelpath)

    for i in tqdm(range(len(inputfiles))):
        name = inputfiles[i].split(".")[0]
        
        print("input img:", inputfiles[i])
        img = cv2.imread(inputpath + inputfiles[i])
        
        print("get paddy by slic")
        start = time.time()
        
        mask_paddy = pdetector.get_paddy(img, mode='slic')
        mask_morphology = pdetector.mask_morphology(mask_paddy, mode='slic')
        
        end = time.time()
        print("running time:", end - start)
  
        cv2.imwrite(slic_outputpath + name + '_paddymask.jpg', mask_morphology)
        
        print("get paddy by felzenszwalb")
        start = time.time()
        
        mask_paddy = pdetector.get_paddy(img, mode='felzenszwalb')
        mask_morphology = pdetector.mask_morphology(mask_paddy, mode='felzenszwalb')
        
        end = time.time()
        print("running time:", end - start)
  
        cv2.imwrite(fz_outputpath + name + '_paddymask.jpg', mask_morphology)
    
        print("get paddy by watershed")
        start = time.time()
        
        mask_paddy = pdetector.get_paddy(img, mode='watershed')
        mask_morphology = pdetector.mask_morphology(mask_paddy, mode='watershed')
        
        end = time.time()
        print("running time:", end - start)
  
        cv2.imwrite(ws_outputpath + name + '_paddymask.jpg', mask_morphology)