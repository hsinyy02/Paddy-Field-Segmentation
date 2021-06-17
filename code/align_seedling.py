from stitching import point_homography, findRicePair_heap
from pathlib import Path

# import cv2
from cv2 import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm

import argparse
import numpy as np
import math
import ast
import csv
import os


def show_aligned(aligned_rice_file, img_inputpath, img_outputpath, 
                 aligned_num=4, show_num=10, clip=150, circle_radius=7, color=(255, 0, 0), line_width=3):
    """Show a part of aligned rice in list_aligned_rice
    """
    list_showed = []
    
    with open(aligned_rice_file, newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        csvdata = list(csv_reader)
        
    for row in csvdata:
        if csvdata.index(row) == 0:
            continue
        num = ast.literal_eval(row[1])
        rice = {"id": ast.literal_eval(row[0])}
        if num == aligned_num:
            for i in range(num):
                idx = 2*i + 2
                item = {row[idx]: ast.literal_eval(row[idx+1])}
                rice.update(item)
            
            list_showed.append(rice)
    
    if len(list_showed) == 0:
        print("no rice has aligned_num:", aligned_num)
        
    else:
        for rice in list_showed[:show_num]:
            plt.figure(figsize = (12, 6))
            
            rice_id = rice["id"]
            
            for idx, (key, value) in enumerate(rice.items()):
                if key == "id":
                    continue
                    
                img = cv2.imread(img_inputpath + key + ".JPG")
                pt = tuple([math.floor(p) for p in value])
                
                cv2.circle(img, pt, circle_radius, color, line_width)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                x, y = pt
                xrange = [(x-clip if (x-clip) > 0 else 0), (x+clip if (x+clip) < img.shape[1] else img.shape[1])]
                yrange = [(y-clip if (y-clip) > 0 else 0), (y+clip if (y+clip) < img.shape[0] else img.shape[0])]
                
                cut = img[yrange[0]:yrange[1], xrange[0]:xrange[1]].copy()
                
                plt.subplot(1, aligned_num, idx)
                plt.title(key)
                plt.imshow(cut)
            
            plt.savefig(img_outputpath + "rice_" + str(rice_id) + ".jpg")
            

def write_ricelist(list_rice, outputpath, filename):
    """Write list_rice into a csv file.
    """
    with open(outputpath + filename, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["id", "num", "img", "position"])
        
        for idx, rice_item in enumerate(list_rice):
            row = [idx, len(rice_item)]
            for key, value in rice_item.items():
                row.extend([key, value])
                
            writer.writerow(row)

            
def _parse_args():
    """Argument parser: parse the folder path.
    
    Argument:
        --inputpath
        --ricepath
        --homolist
        --outputpath
    """
    parser = argparse.ArgumentParser(description='ArgParser for align_seedling.py')
    parser.add_argument('--inputpath', default='../input/', type=str, help='folder path to input images')
    parser.add_argument('--ricepath', default='../output/rice_label/TGI/', type=str, help='folder path to read rice file')
    parser.add_argument('--homolist', default='../output/stitching/list_homo.csv', type=str, help='file path to read list of homography')
    
    parser.add_argument('--outputpath', default='../output/alignment/', type=str, help='folder path to output result')
    parser.add_argument('--output_alignedpath', default='../output/alignment/aligned_rice/', type=str, help='folder path to save img of aligned rice')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    """
    The program finds the corresponding rice seedlings between the adjacent images.
    Then each unique seedling will be given an id number.
    For each id number of the seedling, 
    the program will record the id of images which include it, 
    and will also record the position of it in the image.

    All in all, the program will output:
    1. a csv file which saves the id number of the seedling, the id of images which include it, and the position of it in the image.
    2. 10(default) images of seedling which appears in 4 images(default). 
    """

    args = _parse_args()

    inputpath = args.inputpath
    inputfiles = sorted([file for file in os.listdir(inputpath) if file.endswith('.JPG')])
    
    ricepath = args.ricepath
    
    outputpath = args.outputpath
    output_alignedpath = args.output_alignedpath
    
    Path(outputpath).mkdir(parents=True, exist_ok=True)
    Path(output_alignedpath).mkdir(parents=True, exist_ok=True)
    
    list_H = []
    
    csvfile_homolist = args.homolist
    
    with open(csvfile_homolist, newline='') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        csvdata = list(csv_reader)

    for row in csvdata:
        name1 = row['name1']
        name2 = row['name2']
    
        h11 = ast.literal_eval(row['h11'])
        h12 = ast.literal_eval(row['h12'])
        h13 = ast.literal_eval(row['h13'])
        h21 = ast.literal_eval(row['h21'])
        h22 = ast.literal_eval(row['h22'])
        h23 = ast.literal_eval(row['h23'])
        h31 = ast.literal_eval(row['h31'])
        h32 = ast.literal_eval(row['h32'])
        h33 = ast.literal_eval(row['h33'])
        H = [[h11, h12, h13], 
             [h21, h22, h23],
             [h31, h32, h33]]
        H = np.array(H)
    
        info = {"name1": name1, "name2": name2, "H": H}
        list_H.append(info)
    
    list_rice = []
    
    for i in range(len(inputfiles)):
        if i == 0:
            name = inputfiles[i].split(".")[0]
            csvname = name + "_TGI_label.csv"
            print("read:", csvname)
            
            with open(ricepath + csvname, newline='') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                csvdata = list(csv_reader)

            for row in csvdata:
                label = ast.literal_eval(row['label'])
                if label == 0:
                    continue

                pt = ast.literal_eval(row['keypoints'])
                info = {name: pt}
                list_rice.append(info)

            print("# of list_rice =", len(list_rice))            
                    
        elif i == 1:
            name_prev = inputfiles[i-1].split(".")[0]
            name_curr = inputfiles[i].split(".")[0]
            
            rice_prev = []
            rice_curr = []
            
            # read rice_curr
            csvname = name_curr + "_TGI_label.csv"
            print("read:", csvname)
            
            with open(ricepath + csvname, newline='') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                csvdata = list(csv_reader)

            for row in csvdata:
                label = ast.literal_eval(row['label'])
                if label == 0:
                    continue

                pt = ast.literal_eval(row['keypoints'])
                pt_homo = tuple([math.floor(p) for p in pt])
                info = {"pt": pt, "homo": pt_homo}
                rice_curr.append(info)
                
            # get H and transform rice_prev in list_rice into name_curr img plane
            H = list_H[i-1]["H"]
            for rice in list_rice:
                pt = rice[name_prev]
                pt_homo = point_homography(pt, H)
                pt_homo = tuple([math.floor(p) for p in pt_homo])
                info = {"pt": pt, "homo": pt_homo}
                rice_prev.append(info)
                
            print("finding ricepairs between", name_prev, "and", name_curr)
            ricepairs = findRicePair_heap(rice_prev, rice_curr)
            print("# of ricepairs =", len(ricepairs))
            
            for rice in rice_curr:
                pt = rice["pt"]
                # check if rice_curr is linked to rice_prev
                pair = next((item for item in ricepairs if item["rice2"] == pt), None)
                if pair is None:
                    # new rice item to list_rice
                    info = {name_curr: pt}
                    list_rice.append(info)
                else:
                    # link rice_prev to rice_prev in list_rice
                    pt1 = pair["rice1"]
                    item_rice = next(item for item in list_rice if item[name_prev] == pt1)
                    item_rice[name_curr] = pt

            print("# of list_rice =", len(list_rice))            

        else:
            # 對連續三張影像ABC，統一轉換到影像平面B後連結AB、BC，再連結AC
            name1 = inputfiles[i-2].split(".")[0]
            name2 = inputfiles[i-1].split(".")[0]
            name3 = inputfiles[i].split(".")[0]

            H12 = list_H[i-2]["H"]
            H23 = list_H[i-1]["H"]
            H32 = np.linalg.inv(H23)

            rice1 = []
            rice2 = []
            rice3 = []
            
            for item in list_rice:
                # transform rice1 into img plane of name2
                if name1 in item:
                    pt = item[name1]
                    pt_homo = point_homography(pt, H12)
#                     pt_homo = tuple([math.floor(p) for p in pt_homo])
                    info = {"pt": pt, "homo": pt_homo}
                    rice1.append(info)
                    
                if name2 in item:
                    pt = item[name2]
                    pt_homo = pt
#                     pt_homo = tuple([math.floor(p) for p in pt])
                    info = {"pt": pt, "homo": pt_homo}
                    rice2.append(info)
            
            # read rice3
            csvname = name3 + "_TGI_label.csv"
            
            print("read:", csvname)
            
            with open(ricepath + csvname, newline='') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                csvdata = list(csv_reader)
                
            for row in csvdata:
                label = ast.literal_eval(row['label'])
                if label == 0:
                    continue
                    
                pt = ast.literal_eval(row['keypoints'])
                pt_homo = point_homography(pt, H32)
                pt_homo = tuple([math.floor(p) for p in pt_homo])
                info = {"pt": pt, "homo": pt_homo}
                rice3.append(info)
            
            print("finding ricepairs between", name2, "and", name3)
            ricepairs23 = findRicePair_heap(rice2, rice3)
            print("# of ricepairs =", len(ricepairs23))

            print("finding ricepairs between", name1, "and", name3)
            ricepairs13 = findRicePair_heap(rice1, rice3)
            print("# of ricepairs =", len(ricepairs13))
            
            for rice in rice3:
                pt = rice["pt"]
                pair23 = next((item for item in ricepairs23 if item["rice2"] == pt), None)
                pair13 = next((item for item in ricepairs23 if item["rice2"] == pt), None)
                
                # link rice3 to rice2 in list_rice
                if pair23 is not None:
                    pt2 = pair23["rice1"]
                    item_rice = next(item for item in list_rice if name2 in item if item[name2] == pt2)
                    item_rice[name3] = pt
                
                else:
                    # link rice3 to rice1 in list_rice
                    if pair13 is not None:
                        pt1 = pair13["rice1"]
                        item_rice = next(item for item in list_rice if name1 in item if item[name1] == pt1)
                        item_rice[name3] = pt
                    # new rice item to list_rice
                    else:
                        info = {name3: pt}
                        list_rice.append(info)
                
            print("# of list_rice =", len(list_rice))
    
    print("write list_rice into list_rice.csv")
    
    write_ricelist(list_rice, outputpath, "list_rice.csv")
        
    print("generate img of aligned rice from list_rice")
    
    show_aligned(outputpath + "list_rice.csv", inputpath, output_alignedpath)