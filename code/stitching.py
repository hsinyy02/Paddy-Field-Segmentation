from shapely.geometry import Point
from shapely.geometry import Polygon
# import cv2
from cv2 import cv2
from PIL import Image
from matplotlib import pyplot as plt
from pathlib import Path
from tqdm import tqdm

import argparse
import numpy as np
import math
import ast
import csv
import os
# import time
import heapq
import logging
# logging.basicConfig(level=logging.WARNING)

def pts_distance(pt1, pt2):
    """Calculate distance between pt1 and pt2.

    Parameters:
        pt1: (x, y) tuple. 

        pt2: (x, y) tuple. 
    
    Return:
        dis: float.
            Distance between pt1 and pt2.
    """
    x1, y1 = pt1
    x2, y2 = pt2

    dis = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    
    return dis


def point_homography(point, H):
    """Apply projective transformation (homography) on a point.

    Parameters:
        point: (x, y) tuple. 
            A point for projective transformation.

        H: A homography matrix.
    
    Return:
        homo: (x, y) tuple.
            The point after projective transformation.
    """
    h0, h1, h2 = H[0]
    h3, h4, h5 = H[1]
    h6, h7, h8 = H[2]

    x, y = point
    
    tx = h0*x + h1*y + h2
    ty = h3*x + h4*y + h5
    tz = h6*x + h7*y + h8
    
    homo = (tx/tz, ty/tz)
    
    return homo


def vertices_homography(vertices, H):
    """Apply projective transformation (homography) on a sequence of points.

    Parameters:
        vertices: List of (x, y) tuples. 
            A list for projective transformation.

        H: A homography matrix.
    
    Return:
        vertices_homo: List of (x, y) tuples.
            The list after projective transformation.
    """
    vertices_homo = []

    for vertex in vertices:
        pt_homo = point_homography(vertex, H)
        pt_homo = tuple([math.floor(i) for i in pt_homo])
        vertices_homo.append(pt_homo)
    
    return vertices_homo


def removeOutsider(rice1_H, rice2_H, overlap):
    """Remove rice which is not in the overlap region after homography.

    Parameters:
        rice1_H: List of dictionaries.
            Record the position of rice and that after homography in img1.
            [{"pt": (x1, y1), "homo": (x2, y2)}, ...]

        rice2_H: List of dictionaries.
            Record the position of rice and that after homography in img2.
            [{"pt": (x1, y1), "homo": (x2, y2)}, ...]
        
        overlap: Polygon.
            Overlap region between img1 and img2 after homography.
    """
    # if rice of img1 isn't in the overlap region, remove it from the list.
    for item in reversed(rice1_H):
        homo = Point(item["homo"])

        if not homo.within(overlap):
            rice1_H.remove(item)

    # if rice of img2 isn't in the overlap region, remove it from the list.
    for item in reversed(rice2_H):
        homo = Point(item["homo"])

        if not homo.within(overlap):
            rice2_H.remove(item)


def findRicePair(rice1_H, rice2_H, thresh=5):
    """Find rice pairs whose infinity norm are smaller than the given threshold, 
    and there is only one possible one pair in the infinity norm < thresh.

    Parameters:
        rice1_H: List of dictionaries.
            Record the position of rice and that after homography in img1.
            [{"pt": (x1, y1), "homo": (x2, y2)}, ...]

        rice2_H: List of dictionaries.
            Record the position of rice and that after homography in img2.
            [{"pt": (x1, y1), "homo": (x2, y2)}, ...]
        
        thresh: int or float, optional.
            The given threshold.
    Return:
        rice_pair: List of dictionaries.
            Record the position of rice in img1, the position of rice after homography in img1, 
            and the position of rice in img2.
            [{"rice1": (x1, y1), "rice1_homo": (x2, y2), "rice2": (x3, y3)}, ...]
    """
    rice_pair = []
    R2_H = rice2_H.copy()
    R1_H = rice1_H.copy()
    
    for item in reversed(R1_H):
        pt_original = item["pt"]
        pt_homo = item["homo"]
        x, y = pt_homo
        
        # if there is only 1 rice2 in the block, it will be considered as a rice pair
        pair_item = []
        for check_item in R2_H:
            checkpt = check_item["homo"]
            x1, y1 = checkpt
            
            # if checkpt is in the block, num++
            if (x1 > x-thresh and x1 < x+thresh and
                y1 > y-thresh and y1 < y+thresh):
                pair_item.append(check_item)
        
        if len(pair_item) == 1:
            pairpt = pair_item[0]["pt"]
            info = {"rice1": pt_original, "rice1_homo": pt_homo, "rice2": pairpt}
            rice_pair.append(info)
            R1_H.remove(item)
            R2_H.remove(pair_item[0])
    
    return rice_pair


def findRicePair_heap(rice1_H, rice2_H, thresh=10):
    """Find rice pairs who are the closest to each other, and whose infinity norm are smaller than the given threshold.

    Parameters:
        rice1_H: List of dictionaries.
            Record the position of rice and that after homography in img1.
            [{"pt": (x1, y1), "homo": (x2, y2)}, ...]

        rice2_H: List of dictionaries.
            Record the position of rice and that after homography in img2.
            [{"pt": (x1, y1), "homo": (x2, y2)}, ...]
        
        thresh: int or float, optional.
            The given threshold.
    Return:
        rice_pair: List of dictionaries.
            Record the position of rice in img1, the position of rice after homography in img1, 
            and the position of rice in img2.
            [{"rice1": (x1, y1), "rice1_homo": (x2, y2), "rice2": (x3, y3)}, ...]
    """    
    rice_pair = []
    R1_H = rice1_H.copy()
    R2_H = rice2_H.copy()
    
    for item in reversed(R1_H):
        pair_heap = []
        pt = item["homo"]
        x, y = pt

        # check if any rice2 is in the block
        for check_item in R2_H:
            checkpt = check_item["homo"]
            x1, y1 = checkpt

            # if checkpt is in the block
            if (x1 > x-thresh and x1 < x+thresh and
                y1 > y-thresh and y1 < y+thresh):
                dist = pts_distance(pt, checkpt) 
                heapq.heappush(pair_heap, (dist, checkpt))
            
        # if len(pair_heap)>0 add the "heap" into item, else (if pair_heap is empty) remove the item from rice1_H
        if len(pair_heap):
            item["heap"] = pair_heap
        else:
            R1_H.remove(item)

    for item in reversed(R2_H):
        pair_heap = []
        pt = item["homo"]
        x, y = pt
        # check if any rice1 is in the block
        for check_item in R1_H:
            checkpt = check_item["homo"]
            x1, y1 = checkpt
        
            # if checkpt is in the block
            if (x1 > x-thresh and x1 < x+thresh and
                y1 > y-thresh and y1 < y+thresh):
                dist = pts_distance(pt, checkpt)
                heapq.heappush(pair_heap, (dist, checkpt))
    
        # if len(pair_heap)>0 add the "heap" into item, else (if pair_heap is empty) remove the item from rice2_H
        if len(pair_heap):
            item["heap"] = pair_heap
        else:
            R2_H.remove(item)    
    
    # check if there is a pair
    for item in reversed(R1_H):
        pt_original = item["pt"]
        pt_homo = item["homo"]
        # get homo pt of rice2 which is the cloest to the pt of rice1
        pairpt = item["heap"][0][1]
        # finding item in dicts by value. return None if the item is not found
        pair_item = next((rice for rice in R2_H if rice["homo"] == pairpt), None)
    
        if pair_item is None:
    #         print("the item has been removed")
            continue

        # pop the pair (pt, pairpt) and push it into rice_pair    
        if pair_item["heap"][0][1] == pt_homo:
            info = {"rice1": pt_original, "rice1_homo": pt_homo, "rice2": pair_item["pt"]}
            rice_pair.append(info)
            R1_H.remove(item)
            R2_H.remove(pair_item)
    
    return rice_pair


def find_homography_sift(img1, img2, mask1=None, mask2=None):
    """Find homography matrix with keypoints which were detected by using SIFT.

    Parameters:
        img1: ndarray.
            The source image which can be warpped to the destination image based on the homography.

        img2: ndarray.
            The destination image.

        mask1: ndarray, optional.
            Mask of img1.

        mask2: ndarray, optional.
            Mask of img2.
        
    Return:
        H: ndarray.
            The homography matrix.

        mask: ndarray.
            Indicate the match pair is inlier (1) or outlier (0).
    """    
    sift = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, mask = mask1)
    kp2, des2 = sift.detectAndCompute(img2, mask = mask2)

    bf = cv2.BFMatcher()

    try:
        matches = bf.knnMatch(des1, des2, k=2)
    
    except:
        logging.warning("downsampling")
        
        kp2 = kp2[::2]
        des2 = des2[::2]
    
        matches = bf.knnMatch(des1, des2, k=2)

    good = []
        
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m) 
        
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return H, mask


def find_homography_seedling(img1, img2, rice1, rice2):
    """Find homography matrix with keypoints which were detected by using SIFT, 
    and then refind homography matrix by using rice pairs as matched pairs.

    Parameters:
        img1: ndarray.
            The source image which can be warpped to the destination image based on the homography.

        img2: ndarray.
            The destination image.

        rice1: List of (x, y) tuples.
            Position of seedlings of img1.
        
        rice2: List of (x, y) tuples.
            Position of seedlings of img2.
        
    Return:
        H: ndarray.
            The homography matrix.
    
        mask: ndarray.
            Indicate the match pair is inlier (1) or outlier (0).
    """
    height, width = img1.shape[:2]

    H, masked = find_homography_sift(img1, img2)

    vertices = [(0, 0), (width, 0), (width, height), (0, height)]
    bound = Polygon(vertices)

    # calculate the position of img1's vertices after homography
    vertices_homo = vertices_homography(vertices, H)
    bound_homo = Polygon(vertices_homo)

    # get the overlap region of img1 and img2 after homography
    overlap = bound.intersection(bound_homo)

    # find rice pairs with H
    r1H = []
    r2H = []

    for pt in rice1:
        homopt = point_homography(pt, H)
        info = {"pt": pt, "homo": homopt}
        r1H.append(info)

    for pt in rice2:
        info = {"pt": pt, "homo": pt}
        r2H.append(info)

    # remove rice which is not in the overlap region
    removeOutsider(r1H, r2H, overlap)

    # find rice pairs
    thresh = 5
    ricepairs = findRicePair(r1H, r2H, thresh)

    # find new H by rice pairs
    src_pts = np.float32([pair["rice1"] for pair in ricepairs]).reshape(-1, 1, 2)
    dst_pts = np.float32([pair["rice2"] for pair in ricepairs]).reshape(-1, 1, 2)
        
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return H, mask


def find_homography_iter_seedling(img1, img2, rice1, rice2):
    """First, mask img1 into topleft, topright, bottomleft, bottomright. 
    Next, find homography matrix with sift-based method and find rice pairs respectively. 
    Last, merge rice pairs and then refind homography matrix with rice pairs.

    Parameters:
        img1: ndarray.
            The source image which can be warpped to the destination image based on the homography.

        img2: ndarray.
            The destination image.

        rice1: List of (x, y) tuples.
            Position of seedlings of img1.
        
        rice2: List of (x, y) tuples.
            Position of seedlings of img2.
        
    Return:
        H: ndarray.
            The homography matrix.
    
        mask: ndarray.
            Indicate the match pair is inlier (1) or outlier (0).
    """
    height, width = img1.shape[:2]

    masks = []
    region = [
        {"zone": "lefttop", "corner": [(0, 0), (int(width/2), int(height/2))]},
        {"zone": "righttop", "corner": [(int(width/2), 0), (width, int(height/2))]},
        {"zone": "leftbottom", "corner": [(0, int(height/2)), (int(width/2), height)]},
        {"zone": "rightbottom", "corner": [(int(width/2), int(height/2)), (width, height)]}
    ]

    # compute mask for each region
    for item in region:
        # zone = item["zone"]
        (x1, y1), (x2, y2) = item["corner"]
        mask = np.zeros((height, width), dtype = "uint8")
        mask[y1:y2, x1:x2] = 255
        masks.append(mask)
    
    sift = cv2.xfeatures2d.SIFT_create()

    kp2, des2 = sift.detectAndCompute(img2, mask = None)

    ricepairs = []
    vertices = [(0, 0), (width, 0), (width, height), (0, height)]
    bound = Polygon(vertices)

    for idx, item in enumerate(region):
        zone = item["zone"]
        print("stitching based on " + zone)

        kp1, des1 = sift.detectAndCompute(img1, mask = masks[idx])

        bf = cv2.BFMatcher()

        try:
            matches = bf.knnMatch(des1, des2, k=2)
    
        except:
            logging.warning("downsampling")
        
            kp2 = kp2[::2]
            des2 = des2[::2]
    
            matches = bf.knnMatch(des1, des2, k=2)

        good = []
        
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m) 
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
        # calculate the position of img1's vertices after homography
        vertices_homo = []
        for vx in vertices:
            vx_homo = point_homography(vx, H)
            vertices_homo.append(vx_homo)   
        
        bound_homo = Polygon(vertices_homo)
        
        # if the shape after homography is not simply, jump to next zone
        if not bound_homo.is_simple:
            logging.warning("stitching failed: " + zone)
            continue
            
        # get the overlap region of img1 and img2 after homography
        overlap = bound.intersection(bound_homo)

        # finding ricepair by H_zone
        r1H = []
        r2H = []

        for pt in rice1:
            homopt = point_homography(pt, H)
            info = {"pt": pt, "homo": homopt}
            r1H.append(info)

        for pt in rice2:
            info = {"pt": pt, "homo": pt}
            r2H.append(info)
    
        # remove rice which is not in the overlap region
        removeOutsider(r1H, r2H, overlap)
        
        # find rice pairs
        thresh = 5
        ricepairs_zone = findRicePair(r1H, r2H, thresh)
        ricepairs.append(ricepairs_zone)

    # merge ricepairs and delete duplicated pairs
    ricepairs = [item for ls in ricepairs for item in ls]
    ricepairs = [{k:v for k, v in pair.items() if k == "rice1" or k == "rice2"} for pair in ricepairs]
    # delete the duplicated pairs
    ricepairs = [i for n, i in enumerate(ricepairs) if i not in ricepairs[n + 1:]]
    
    print("# of ricepairs =", len(ricepairs))

    # find new H by rice pairs
    src_pts = np.float32([pair["rice1"] for pair in ricepairs]).reshape(-1, 1, 2)
    dst_pts = np.float32([pair["rice2"] for pair in ricepairs]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)   

    return H, mask


def find_homography(img1, img2, rice1=[], rice2=[], mode="sift"):
    """Find homography matrix with the specified mode.

    Parameters:
        img1: ndarray.
            The source image which can be warpped to the destination image based on the homography.

        img2: ndarray.
            The destination image.

        rice1: List of (x, y) tuples, optional.
            Position of seedlings of img1.
        
        rice2: List of (x, y) tuples, optional.
            Position of seedlings of img2.
        
        mode: str or int, optional.
                One of the following strings or integer, selecting the mode for finding homography:
                 - 'sift' or 0: Find homography using sift-based method.
                 - 'seedling' or 1: Find homography using seedling-based method.
                 - 'iter_seedling' or 2: Find homography using iter_seedling-based method.
        
    Return:
        H: ndarray.
            The homography matrix.
    
        mask: ndarray.
            Indicate the match pair is inlier (1) or outlier (0).
    """    
    if isinstance(mode, str):
        mode = mode.lower()

    allowedmode = ['sift', 'seedling', 'iter_seedling', 0, 1, 2]

    if mode not in allowedmode:
        raise ValueError("mode should be 'sift' or 0, 'seedling' or 1, 'iter_seedling' or 2")

    if mode == "sift" or mode == 0:
        H, masked = find_homography_sift(img1, img2)
    
    elif mode == "seedling" or mode == 1:
        H, masked = find_homography_seedling(img1, img2, rice1, rice2)
    
    else:
        # test if there is enough rice in the overlap region
        # resize img1 and img2
        height, width = img1.shape[:2]
        dsize = (int(width * 0.2), int(height * 0.2))
        
        vertices = [(0, 0), (dsize[0], 0), (dsize[0], dsize[1]), (0, dsize[1])]
        bound = Polygon(vertices)
        
        img1_test = cv2.resize(img1, dsize, interpolation=cv2.INTER_AREA)
        img2_test = cv2.resize(img2, dsize, interpolation=cv2.INTER_AREA)
        H, masked = find_homography_sift(img1_test, img2_test)
        
        # calculate the position of img1_test's vertices after homography
        vertices_homo = []
        for vx in vertices:
            vx_homo = point_homography(vx, H)
            vertices_homo.append(vx_homo)   
        
        bound_homo = Polygon(vertices_homo)
            
        # get the overlap region of img1_test and img2_test after homography
        overlap = bound.intersection(bound_homo)
        
        # calculate # of rice in the overlap region
        r1H = []
        r2H = []

        for pt in rice1:
            pt = tuple([p*0.2 for p in pt])
            homopt = point_homography(pt, H)
            info = {"pt": pt, "homo": homopt}
            r1H.append(info)

        for pt in rice2:
            pt = tuple([p*0.2 for p in pt])
            info = {"pt": pt, "homo": pt}
            r2H.append(info)
    
        # remove rice which is not in the overlap region
        removeOutsider(r1H, r2H, overlap)
   
        print("# of rice1 in overlap =", len(r1H))
        print("# of rice2 in overlap =", len(r2H))
        
        if len(r1H) < 4000 and len(r2H) < 4000:
            logging.warning("not enough # of rice. change mode to seedling")
            H, masked = find_homography_seedling(img1, img2, rice1, rice2)
        else:
            H, masked = find_homography_iter_seedling(img1, img2, rice1, rice2)
    
    return H, masked


def draw_pairs(img_blended, ricepair, circle_radius=5, line_width=3):
    """Draw ricepair on the blended image.
    
    Parameters:
        img_blended: ndarray.
            The blended image.

        ricepair: List of dictionaries.
            [{"rice1": (x1, y1), "rice1_homo": (x2, y2), "rice2": (x3, y3)}, ...]
        
    Return:
        img_pair: ndarray.
            The blended image with ricepairs.
    """
    img_pair = img_blended.copy()
    
    for pair in ricepair:
        pt1 = tuple([math.floor(p) for p in pair["rice1_homo"]])
        pt2 = tuple([math.floor(p) for p in pair["rice2"]])
        
        # rice from img1 will be circle in blue
        cv2.circle(img_pair, pt1, circle_radius, (255, 0, 0), -1)
        # rice from img2 will be circle in red
        cv2.circle(img_pair, pt2, circle_radius, (0, 0, 255), -1)
        # line the ricepair in cyan
        cv2.line(img_pair, pt1, pt2, (255, 255, 0), line_width)
        
    return img_pair


def vertices_translate(vertices, tx, ty):
    vertices_move = []
    for vertex in vertices:
        x, y = vertex
        pt = ((x + tx), (y + ty))
        vertices_move.append(pt)
        
    return vertices_move


def vertices_resize(vertices, ratio):
    vertices_scaling = []
    for vertex in vertices:
        pt_scaling = tuple([math.floor(i*ratio) for i in vertex])
        vertices_scaling.append(pt_scaling)
        
    return vertices_scaling


def get_mosaic_vertices(list_homo, width=6000, height=4000):
    img_vertices = []
    vertices = [(0, 0), (width, 0), (width, height), (0, height)]

    for i in range(len(list_homo) + 1):
        img_vertices.append(vertices)
        
    for i in range(len(list_homo)):
        h11 = list_homo[i]['h11']
        h12 = list_homo[i]['h12']
        h13 = list_homo[i]['h13']
        h21 = list_homo[i]['h21']
        h22 = list_homo[i]['h22']
        h23 = list_homo[i]['h23']
        h31 = list_homo[i]['h31']
        h32 = list_homo[i]['h32']
        h33 = list_homo[i]['h33']
        
        H = [[h11, h12, h13], 
             [h21, h22, h23],
             [h31, h32, h33]]
        
        H_current = np.array(H)
        
        num = i + 1
        for j in range(num):
            img_vertices[j] = vertices_homography(img_vertices[j], H_current)
        
    x_max = [max(v, key = lambda pt:pt[0])[0] for v in img_vertices]
    y_max = [max(v, key = lambda pt:pt[1])[1] for v in img_vertices]
    
    x_min = [min(v, key = lambda pt:pt[0])[0] for v in img_vertices]
    y_min = [min(v, key = lambda pt:pt[1])[1] for v in img_vertices]
    
    print("x_original range:", min(x_min), "~", max(x_max))
    print("y_original range:", min(y_min), "~", max(y_max))
    
    # move the vertices and rescale to smaller one
    tx = 0 - min(x_min)
    ty = 0 - min(y_min)
    
    for i in range(len(img_vertices)):
        poly = img_vertices[i]
        poly = vertices_translate(poly, tx, ty)
        poly = vertices_resize(poly, 0.5)
        img_vertices[i] = poly
    
    x_max = [max(v, key = lambda pt:pt[0])[0] for v in img_vertices]
    y_max = [max(v, key = lambda pt:pt[1])[1] for v in img_vertices]
    
    x_min = [min(v, key = lambda pt:pt[0])[0] for v in img_vertices]
    y_min = [min(v, key = lambda pt:pt[1])[1] for v in img_vertices]

    print("x range:", min(x_min), "~", max(x_max))
    print("y range:", min(y_min), "~", max(y_max))
            
    return img_vertices


def _parse_args():
    """Argument parser: parse the folder path.
    
    Argument:
        --inputpath
        --ricepath
        --outputpath
        --output_blendingpath
        --output_ricepairpath
    """
    parser = argparse.ArgumentParser(description='ArgParser for stitching.py')
    parser.add_argument('--inputpath', default='../input/', type=str, help='folder path to input images')
    parser.add_argument('--ricepath', default='../output/rice/TGI/', type=str, help='folder path to read rice file')
    
    parser.add_argument('--outputpath', default='../output/stitching/', type=str, help='folder path to output result')
    parser.add_argument('--output_blendingpath', default='../output/stitching/blending/', type=str, help='folder path to save blending img')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    """
    The program finds a homography matrix between every two input images.

    For every two input images, a homography matrix will be found, and then output: 
    1. an image of stitching result based on the matrix.

    After all the homography matrices are found, the program will output:
    1. a csv file which saves all the values of homography matrices.
    2. a mosaic image of all the input images.
    """

    args = _parse_args()
    
    inputpath = args.inputpath
    inputfiles = sorted([file for file in os.listdir(inputpath) if file.endswith('.JPG')])
    
    outputpath = args.outputpath
    output_blendingpath = args.output_blendingpath
    
    Path(outputpath).mkdir(parents=True, exist_ok=True)
    Path(output_blendingpath).mkdir(parents=True, exist_ok=True)

    ricepath = args.ricepath
    
    homography_list = []
    
    for i in tqdm(range(len(inputfiles)-1)):
        name_img1 = inputfiles[i].split(".")[0]
        name_img2 = inputfiles[i+1].split(".")[0]
                
        csvname_img1 = name_img1 + "_TGI.csv"
        csvname_img2 = name_img2 + "_TGI.csv"
        
        print("read rice file:", csvname_img1, csvname_img2)

        rice_img1 = []
        rice_img2 = []
        
        # read rice_img1
        with open(ricepath + csvname_img1, newline='') as csvfile:
            csv_reader = csv.DictReader(csvfile)
            csvdata = list(csv_reader)

        for row in csvdata:
            pt = ast.literal_eval(row['keypoints'])
            rice_img1.append(pt)            

        # read rice_img2
        with open(ricepath + csvname_img2, newline='') as csvfile:
            csv_reader = csv.DictReader(csvfile)
            csvdata = list(csv_reader)

        for row in csvdata:
            pt = ast.literal_eval(row['keypoints'])
            rice_img2.append(pt)               
        
        print("read img:", name_img1, name_img2)
        
        img1 = cv2.imread(inputpath + inputfiles[i])
        img2 = cv2.imread(inputpath + inputfiles[i+1])
        
        print("start finding homography")
        
        H, masked3 = find_homography(img1, img2, rice_img1, rice_img2, mode="iter_seedling")
        
        print("H =", H)
        
        h = []
        for row in H:
            h.extend(row)
            
        info = {"name1": name_img1, "name2": name_img2, 
                "h11": h[0], "h12": h[1], "h13": h[2], 
                "h21": h[3], "h22": h[4], "h23": h[5], 
                "h31": h[6], "h32": h[7], "h33": h[8],
                }
        homography_list.append(info)

        print("blending img1 and img2")

        dst = cv2.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))
        result = cv2.addWeighted(dst, 0.5, img2, 0.5, 0)
        cv2.imwrite(output_blendingpath + name_img1 + "_" + name_img2 + ".jpg", result)
        
        rice1_H = []
        rice2_H = []
        
        for pt in rice_img1:
            pt_homo = point_homography(pt, H)
            info = {"pt": pt, "homo": pt_homo}
            rice1_H.append(info)
            
        for pt in rice_img2:
            pt_homo = tuple([p for p in pt])
            info = {"pt": pt, "homo": pt_homo}
            rice2_H.append(info)

        print("finding ricepairs between img1 and img2")
        
        ricepair = findRicePair_heap(rice1_H, rice2_H, thresh=10)
        
        print("# of pairs =", len(ricepair))
        
        img_ricepair = draw_pairs(result, ricepair)
        cv2.imwrite(output_blendingpath + name_img1 + "_" + name_img2 + "_pair.jpg", img_ricepair)
        
    # save homography_list into .csv
    csvname = 'list_homo.csv'
    print("write", csvname)
    with open(outputpath + csvname, 'w', newline='') as csvfile:
        fieldnames = ['name1', 'name2', 'h11', 'h12', 'h13', 'h21', 'h22', 'h23', 'h31', 'h32', 'h33']
        writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
    
        writer.writeheader()
    
        for i in homography_list:
            writer.writerow(i)
    
    # get mosaic_vertices
    print("generate mosaic")
    
    img_vertices = get_mosaic_vertices(homography_list)
    vertices = img_vertices[-1]
    vertices = [tuple([pt*2 for pt in v]) for v in vertices]

    mosaic = np.zeros((7000, 5000, 3), dtype = "uint8")
    mosaic = Image.fromarray(mosaic)

    # get homography by img_vertices, and then generate the mosaic
    for i in range(len(img_vertices)):
        name = inputfiles[i].split(".")[0]
        print("generate mosaic of", name)

        img = cv2.imread(inputpath + inputfiles[i])
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        src = np.float32(vertices)
        dst = np.float32(img_vertices[i])

        M = cv2.getPerspectiveTransform(src, dst)
        
        img_homo = cv2.warpPerspective(img, M, (5000, 7000))
        img_homo = Image.fromarray(img_homo)

        mask = np.zeros(img.shape[:2], dtype = "uint8")
        mask[mask == 0] = 255
        mask = cv2.warpPerspective(mask, M, (5000, 7000))
        mask = Image.fromarray(mask)

        mosaic = Image.composite(img_homo, mosaic, mask)

    # save the mosaic
    mosaic = np.asarray(mosaic)
    cv2.imwrite(output_blendingpath + "mosaic.jpg", mosaic)