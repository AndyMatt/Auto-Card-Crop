#!/usr/bin/python3

import sys
import os
import argparse
import cv2
import numpy as np

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

def PrintError():
    print("""No input files found. 
        
Arguments:
    [file|files]        - Name of file/files to process
    --dir               - Specify Input directory, '.' is supported.
    --threshhold_thresh - 0-255, Value for threshold strength.
    --threshhold1_type  - Type of Threshold Method 
        0: THRESH_BINARY        - Preferable for Black Backgrounds
        1: THRESH_BINARY_INV    - Preferable for White Backgrounds
        2: THRESH_TRUNC
        3: THRESH_TOZERO
        4: THRESH_TOZERO_INV
    --blur              - 0-255,  Blur Kernal Size, useful for removing noise
    --pad               - Size in px, border around final cropped image
    --outdir            - Specify Output Directory relative to path, '.' is supported.
            
Usage: 
    Rotate.py file1.jpg [file2.jpg ...]
    Rotate.py file1.jpg [file2.jpg ...] --outdir [path]
    Rotate.py --dir [InputDirectory]
    Rotate.py file1.jpg --threshhold_thresh 130 --threshhold_type 0""")
    
# ---------------------------------------------------------------------------
# Calculations
# ---------------------------------------------------------------------------

def transform(pts, img, padding):
	pts = np.array(pts, dtype="float32")

    # Push each corner outward along the direction from center to that corner
	center = pts.mean(axis=0)
	directions = pts - center
	norms = np.linalg.norm(directions, axis=1, keepdims=True)
	unit_dirs = directions / norms
	pts_padded = pts + unit_dirs * padding

	# Axis-aligned bounding box of the padded points
	x_min, y_min = pts_padded.min(axis=0)
	x_max, y_max = pts_padded.max(axis=0)

	# Clamp to image bounds
	h, w = img.shape[:2]
	x_min = int(max(0, np.floor(x_min)))
	y_min = int(max(0, np.floor(y_min)))
	x_max = int(min(w, np.ceil(x_max)))
	y_max = int(min(h, np.ceil(y_max)))

	return img[y_min:y_max, x_min:x_max]
    
# ---------------------------------------------------------------------------
# Image Functions
# ---------------------------------------------------------------------------
def ProcessFile(input, settings):
	##Read file as input
	img = cv2.imread(input)
	if img is None:
		print(f"Skipping {img}: could not read image")
		return None
		
	print(f"Processing {input}")
	
	##Blur source image to remove artifacts
	blurred = cv2.blur(img, (settings["blur_strength"],settings["blur_strength"]))

	##Desaturate Source to make it easier to find contours
	imgray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
	
	##Change the threshold of the levels to produce simple geometry
	th, threshed = cv2.threshold(imgray, settings["threshhold_strength"], 255, settings["threshhold1_type"])

	##Find Contours in Source
	contour_canvas = cv2.cvtColor(threshed, cv2.COLOR_GRAY2BGR)
    
    ##Collect White Pixels to calculate angle and area
	white_pixels = np.argwhere(threshed > 0)
    
    # OpenCV expects points in [x, y] layout, so we flip the column ordering
    # Then reshape to format it correctly for OpenCV geometry functions
	pts = white_pixels[:, ::-1].astype(np.int32)
    
    #Get the outer boundary points (Convex Hull)
	hull = cv2.convexHull(pts)
    
    #Simplify the shape down to its main corners (usually 4 for a sheet)
	epsilon = 0.02 * cv2.arcLength(hull, True)
	approx_corners = cv2.approxPolyDP(hull, epsilon, True)
	
	##Calculate a bounding box
	rect = cv2.minAreaRect(pts)
	box = np.intp(cv2.boxPoints(rect))
	return transform(box,img, settings["border_padding"])
    
# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description="Auto-rotate and crop trading card photos (white background).")
    p.add_argument('paths', nargs='*',
                    help='Image file(s), or a directory when -d is used ("." for cwd)')
    p.add_argument('--dir', '--dir', action='store_true',
                    help='Treat the path argument as a directory and process every image in it')
    p.add_argument('--threshhold_val', type=int, default=120,
                    help='Threshold value - card/angle detection (default: 120)')
    p.add_argument('--threshhold1_type', type=int, default=0,
                    help='Threshold type, cv2.threshold type constant (default: 0)')
    p.add_argument('--blur', type=int, default=10, help='Blur kernel size (default: 5)')
    p.add_argument('--pad', type=int, default=20, help='Padding in px around the detected card (default: 10)')
    p.add_argument('--outdir', '--outdir', default=None,
                    help='Output directory (default: overwrite alongside each input as .png)')
    return p

def get_setting_args(args):
    return {
        "threshhold_strength": getattr(args, "threshhold_val", 120),
        "threshhold1_type": getattr(args, "threshhold1_type", 0),
        "blur_strength": getattr(args, "blur", 10),
        "border_padding": getattr(args, "pad", 20)
    }
      
# ---------------------------------------------------------------------------
# File System
# ---------------------------------------------------------------------------
def collect_files(args):
    if args.dir:
        directory = args.paths[0] if args.paths else '.'
        if not os.path.isdir(directory):
            print(f"Error: '{directory}' is not a directory")
            sys.exit(1)
        return sorted(
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.lower().endswith(IMAGE_EXTS)
        )
    return args.paths
    
def output_path_for(input_path, outdir):
    base = os.path.splitext(os.path.basename(input_path))[0] + '.png'
    directory = outdir if outdir else (os.path.dirname(input_path) or '.')
    return os.path.join(directory, base)
    
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
    
def main():
    args = build_parser().parse_args()
    settings = get_setting_args(args)
    files = collect_files(args)
    
    if not files:
        PrintError()
        sys.exit(1)
    
    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
        
    try:
        for f in files:
            img = ProcessFile(f, settings)
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