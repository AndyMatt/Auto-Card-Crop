#!/usr/bin/python3

import sys
import os
import argparse
import cv2
import numpy as np

def PrintError():
    print("""No input files found. 
        
Arguments:
    [file|files]        - Name of file/files to process
    --threshhold_thresh - 0-255, Value for threshold strength.
    --threshhold1_type  - Type of Threshold Method 
        0: THRESH_BINARY        - Preferable for Black Backgrounds
        1: THRESH_BINARY_INV    - Preferable for White Backgrounds
        2: THRESH_TRUNC
        3: THRESH_TOZERO
        4: THRESH_TOZERO_INV
    --blur              - 0-255,  Blur Kernal Size, useful for removing noise
    --pad               - size in px, border around final cropped image
    --outdir            - Specify Output Directory relative to path. '.' is supported.
            
Usage: 
    Rotate.py file1.jpg [file2.jpg ...]
    Rotate.py file1.jpg [file2.jpg ...] --outdir [path]
    Rotate.py -d [InputDirectory]
    Rotate.py file1.jpg --threshhold_thresh 130 --threshhold_type 0""")
    
def RotateImage(img, angle):
	(h,w) = img.shape[:2]
	(cX, cY) = (w // 2, h // 2)

	M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
	
	cos = np.abs(M[0, 0])
	sin = np.abs(M[0, 1])
	width = int((h * sin) + (w * cos))
	height = int((h * cos) + (w * sin))

	#Recalculate Matrix
	M[0, 2] += (width / 2) - cX
	M[1, 2] += (height / 2) - cY

	result = cv2.warpAffine(img, M, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
	return result
	
def GetRectIndecies(x,y,w,h,img):
	(source_height, source_width) = img.shape[:2]
	x0 = max(y-10,0)
	x1 = min(y+h+20,source_height)
	y0 = max(x-10,0)
	y1 = min(x+w+20,source_width)
	
	return x0, x1, y0, y1

def ProcessFile(input):
	##Read file as input
	img = cv2.imread(input)
	if img is None:
		print(f"Skipping {img}: could not read image")
		return None
		
	print(f"Processing {input}")
	
	##Blur source image to remove artifacts
	blurred = cv2.blur(img, (20,20))

	##Desaturate Source to make it easier to find contours
	imgray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
	
	##Change the threshold of the levels to produce simple geometry
	th, threshed = cv2.threshold(imgray, 120, 255, 0)

	##Find Contours in Source
	cnts = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]

	##Create a copy of the threshold result
	canvas  = threshed.copy()

	## sort and choose the largest contour
	cnts = sorted(cnts, key = cv2.contourArea)
	cnt = cnts[-2]

	## approx the contour, so the get the corner points
	arclen = cv2.arcLength(cnt, True)
	approx = cv2.approxPolyDP(cnt, 0.02* arclen, True)
	cv2.drawContours(canvas, [cnt], -1, (255,0,0), 5, cv2.LINE_AA)
	cv2.drawContours(canvas, [approx], -1, (0, 0, 255), 5, cv2.LINE_AA)
	
	##Calculate a bounding box
	contours_op, hierarchy_op = cv2.findContours(threshed, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
	cnts = sorted(cnts, key = cv2.contourArea)
	cnt = cnts[-1]

	##Calculate angle of bounding box for rotation
	_, _, angle = rect = cv2.minAreaRect(cnt)
	if(angle > 45): angle -= 90

	rotated = RotateImage(img, angle)
	
	##Process rotate image
	imgray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)

	##Change the Level Threshold Again
	th, threshed = cv2.threshold(imgray, 150, 255, 0)
	cnts = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
	
	##Find the Contours of the bounding box
	cnts = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	cnts = cnts[0] if len(cnts) == 2 else cnts[1]
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

	# Determine the Bounding box using the contours
	for c in cnts:
		x,y,w,h = cv2.boundingRect(c)
		x0, x1, y0, y1 = GetRectIndecies(x,y,w,h,rotated)
		ROI = rotated[x0:x1, y0:y1]
		break

	#return result
	return ROI
    
# ---------------------------------------------------------------------------
# File System
# ---------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description="Auto-rotate and crop trading card photos (white background).")
    p.add_argument('paths', nargs='*',
                    help='Image file(s), or a directory when -d is used ("." for cwd)')
    p.add_argument('--threshhold_thresh', type=int, default=120,
                    help='Threshold value - card/angle detection (default: 120)')
    p.add_argument('--threshhold1_type', type=int, default=0,
                    help='Threshold type, cv2.threshold type constant (default: 0)')
    p.add_argument('--blur', type=int, default=5, help='Blur kernel size (default: 5)')
    p.add_argument('--pad', type=int, default=10, help='Padding in px around the detected card (default: 10)')
    p.add_argument('--outdir', '--outdir', default=None,
                    help='Output directory (default: overwrite alongside each input as .png)')
    return p
    
def output_path_for(input_path, outdir):
    base = os.path.splitext(os.path.basename(input_path))[0] + '.png'
    directory = outdir if outdir else (os.path.dirname(input_path) or '.')
    return os.path.join(directory, base)
    
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
    
def main():
    args = build_parser().parse_args()
    files = args.paths
    
    if not files:
        PrintError()
        sys.exit(1)
    
    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
        
    try:
        for f in files:
            img = ProcessFile(f)
            if img is None:
                print(f"Failed to detect card bounds in {f}, skipping.")
                continue
				
            out = output_path_for(f, args.outdir)
            cv2.imwrite(out, img)
            print(f"Saved {out}")
    finally:
        print("Complete")
		
if __name__ == '__main__':
    main()