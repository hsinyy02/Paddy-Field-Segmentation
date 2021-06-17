from stitching import point_homography, find_homography_sift
from cv2 import cv2
from pathlib import Path
from shapely.geometry import Point
from shapely.geometry import Polygon
from tqdm import tqdm

import argparse
import numpy as np
import math
import ast
import csv
import os
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects


def set_regions():
    """Set the range of each region on the mosaic.
    
    Return:
        regions: List of Polygons.
            12 labeled regions of paddy field.
    """
    regions = []

    list_region = [
        [(2189, 2421), (2216, 2417), (2694, 2861), (2443, 3110), (1995, 2679), (2002, 2617), (2094, 2503)],                             # 1
        [(2694, 2861), (3069, 3225), (2827, 3472), (2443, 3110)],                                                                       # 2
        [(3069, 3225), (3530, 3677), (3568, 3728), (3545, 3773), (3391, 3924), (3356, 3949), (3174, 3806), (2827, 3472)],               # 3
        [(2380, 3131), (2151, 2903), (1994, 2759), (1909, 2705), (1783, 2823), (1692, 2928), (1853, 3092), (2146, 3364)],               # 4
        [(2380, 3131), (2815, 3547), (2582, 3778), (2146, 3364)],                                                                       # 5
        [(2815, 3547), (3061, 3759), (3300, 3978), (3292, 4011), (3154, 4148), (3083, 4191), (2955, 4116), (2750, 3938), (2582, 3778)], # 6
        [(2097, 3465), (1851, 3710), (1381, 3264), (1394, 3212), (1485, 3109), (1577, 3037), (1632, 3052), (1766, 3146)],               # 7
        [(2097, 3465), (1851, 3710), (2231, 4074), (2476, 3829)],                                                                       # 8
        [(2231, 4074), (2476, 3829), (2931, 4265), (2967, 4305), (2919, 4371), (2770, 4496), (2670, 4463), (2559, 4382)],               # 9
        [(1293, 3305), (1384, 3347), (1769, 3709), (1515, 3963), (1086, 3555), (1068, 3515), (1274, 3312)],                             # 10
        [(1769, 3709), (1515, 3963), (1896, 4321), (2159, 4073)],                                                                       # 11
        [(1896, 4321), (2159, 4073), (2354, 4272), (2675, 4567), (2637, 4632), (2471, 4785), (2392, 4770)]                              # 12
    ]

    for region in list_region:
        region = Polygon(region)
        regions.append(region)

    return regions


def plot_regions(mosaic, regions, colors):
    """Plot each region on the mosaic.
    
    Parameters:
        mosaic: ndarray.
            The mosaic image which is in BGR color space.

        regions: List of Polygons.

        colors: List of tuples.
            (B, G, R) color where B, G, R are between 0 to 255.
    """
    fig = plt.figure(figsize = (14, 14))
    plt.imshow(cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))

    for i, region in enumerate(regions):
        idx = i + 1
        x, y = region.centroid.coords[:][0]

        current_color = tuple([value/255 for value in colors[idx]])
        current_color = (current_color[2], current_color[1], current_color[0])

        plt.fill(*region.exterior.xy, color=current_color)
        plt.text(x, y, str(idx), fontsize=16, color='white', path_effects=[
                 path_effects.Stroke(linewidth=1, foreground='black'), path_effects.Normal()])

    plt.axis('off')
    plt.tight_layout()

    return fig


def read_rice(csvpath, csvname):
    rice = []
    with open(csvpath + csvname, newline='') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        csvdata = list(csv_reader)

    for row in csvdata:
        pt = ast.literal_eval(row['keypoints'])
        rice.append(pt)

    return rice


def label_rice(img, mosaic, data_rice, regions, colors):
    """Label the located region of rice seedlings.
    
    Parameters:
        img: ndarray.
            The input image. BGR color space.
            
        mosaic: ndarray.
            The mosaic image which is in BGR color space.
        
        data_rice: List of Dictionaries.
            The position of rice seedlings.
            [{"keypoints": (x, y)}, ...]
            
        regions: List of Polygons.

        colors: List of tuples.
            [(B, G, R), ...] where B, G, R are between 0 to 255.
            
    Return:
        rice_with_label: List of dictionaries.
            The position of rice seedlings and their located region.
            [{"keypoints": (x, y), "label": region_id}, ...]
        
        img_on_mosaic: ndarray.
            Side length of img_on_mosaic is half of that of the mosaic.
        
        mosaic_rice: ndarray.
            Rice seedling on the mosaic. Different region of rice is drawn as different color.
            Side length of the img is half of that of the mosaic.
        
        img_rice: ndarray.
            Rice seedling on the input image. Different region of rice is drawn as different color.
    """
    img_rice = img.copy()
    mosaic_rice = mosaic.copy()

    rice_with_label = []

    H, mask = find_homography_sift(img, mosaic)

    dst = cv2.warpPerspective(img, H, (mosaic.shape[1], mosaic.shape[0]))
    img_on_mosaic = cv2.addWeighted(dst, 0.5, mosaic, 0.5, 0)

    for pt in data_rice:
        homo = point_homography(pt, H)

        label = 0
        checkpt = Point(homo)

        for idx, r in enumerate(regions):
            if checkpt.within(r):
                label = idx + 1
                break
    
        mosaic_rice = cv2.circle(mosaic_rice, (math.floor(homo[0]), math.floor(homo[1])), 3, colors[label], -1)
        img_rice = cv2.circle(img_rice, (math.floor(pt[0]), math.floor(pt[1])), 5, colors[label], 3)

        info = {"keypoints": pt, "label": str(label)}
        rice_with_label.append(info)
    
    resize_dim = (int(img_on_mosaic.shape[1]/2), int(img_on_mosaic.shape[0]/2))
    img_on_mosaic = cv2.resize(img_on_mosaic, resize_dim, interpolation=cv2.INTER_AREA)
    mosaic_rice = cv2.resize(mosaic_rice, resize_dim, interpolation=cv2.INTER_AREA)
    
    return rice_with_label, img_on_mosaic, mosaic_rice, img_rice


def write_rice(csvpath, csvname, data_rice_label):
    with open(csvpath + csvname, 'w', newline='') as csvfile:
        fieldnames = ['keypoints', 'label']

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i in data_rice_label:
            writer.writerow(i)

            
def _parse_args():
    """Argument parser: parse the folder path and file path of the mosaic image.
    
    Argument:
        --inputpath
        --input_ricepath
        --outputpath
        --output_mosaicpath
        --output_labelpath
        --output_labelpath_mosaic
        --output_labelpath_local
        --mosaic_img

    """
    parser = argparse.ArgumentParser(description='ArgParser for label_seedling.py')
    parser.add_argument('--inputpath', default='../input/', type=str, help='folder path to input images')
    parser.add_argument('--input_ricepath', default='../output/rice/TGI/', type=str, help='folder path to read rice file')
    
    parser.add_argument('--outputpath', default='../output/', type=str, help='folder path to output result')
    parser.add_argument('--output_mosaicpath', default='../output/img_on_mosaic/', type=str, help='folder path to save img_on_mosaic')
    parser.add_argument('--output_labelpath', default='../output/rice_label/TGI/', type=str, help='folder path to save labeled rice file')
    parser.add_argument('--output_labelpath_mosaic', default='../output/rice_label/mosaic/', type=str, help='folder path to save mosaic img with labeled rice')
    parser.add_argument('--output_labelpath_local', default='../output/rice_label/local/', type=str, help='folder path to save img with labeled rice')
    
    parser.add_argument('--mosaic_img', default='../mosaic.png', type=str, help='file path of the mosaic image')
    args = parser.parse_args()
    return args
            

if __name__ == '__main__':
    """
    The program finds which zone each rice seedling is located at, and then labels the rice seedlings.

    For each input image, the program outputs the labeling result as:
    1. An image with the input image blending on the mosaic, which can show the location of the input image.
    2. An image with seedlings being labeled in different colors in the mosaic image.
    3. An image with seedlings being labeled in different colors in the input image.
    4. A csv file which saves the position and the zone label of each rice seedling.
    """

    args = _parse_args()
    
    inputpath = args.inputpath
    inputfiles = sorted([file for file in os.listdir(inputpath)])
    
    outputpath = args.outputpath
    
    output_mosaicpath = args.output_mosaicpath
    output_labelpath = args.output_labelpath
    output_labelpath_mosaic = args.output_labelpath_mosaic
    output_labelpath_local = args.output_labelpath_local

    Path(output_mosaicpath).mkdir(parents=True, exist_ok=True)
    Path(output_labelpath).mkdir(parents=True, exist_ok=True)
    Path(output_labelpath_mosaic).mkdir(parents=True, exist_ok=True)
    Path(output_labelpath_local).mkdir(parents=True, exist_ok=True)

    input_ricepath = args.input_ricepath

    colors = [
        (0, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), 
        (155, 255, 0), (155, 0, 255), (155, 0, 0), (0, 155, 255), 
        (0, 255, 0), (255, 0, 255), (0, 0, 155), (255, 0, 0), (0, 155, 0)
    ]

    regions = set_regions()

    mosaic = cv2.imread(args.mosaic_img)

    fig = plot_regions(mosaic, regions, colors)
    plt.savefig(outputpath + "labelregion.jpg")
    
    for i in tqdm(range(len(inputfiles))):
        name = inputfiles[i].split(".")[0]
        rice_csvfile = name + "_TGI.csv"

        print("read img:", inputfiles[i], ", read csv:", rice_csvfile)

        img = cv2.imread(inputpath + inputfiles[i])

        list_rice = read_rice(input_ricepath, rice_csvfile)
        
        # find homography between img and mosaic, and then label rice by using homography
        list_rice_label, img_on_mosaic, mosaic_rice, img_rice = label_rice(img, mosaic, list_rice, regions, colors)
        
        cv2.imwrite(output_mosaicpath + name + "_on_mosaic.jpg", img_on_mosaic)
        cv2.imwrite(output_labelpath_mosaic + name + "_mosaic_rice.jpg", mosaic_rice)
        cv2.imwrite(output_labelpath_local + name + "_rice.jpg", img_rice)

        csvname = rice_csvfile.split(".")[0] + "_label.csv"

        write_rice(output_labelpath, csvname, list_rice_label)    