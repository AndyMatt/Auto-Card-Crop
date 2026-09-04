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
    --deskew            - Attempt to deskew the detected card/page
            
Usage: 
    Rotate.py file1.jpg [file2.jpg ...]
    Rotate.py file1.jpg [file2.jpg ...] --outdir [path]
    Rotate.py --dir [InputDirectory]
    Rotate.py file1.jpg --threshhold_thresh 130 --threshhold_type 0""")

# ---------------------------------------------------------------------------
# Calculations
# ---------------------------------------------------------------------------

#Orders 4 points in the sequence: top-left, top-right, bottom-right,bottom-left.
def order_points(pts):
  pts = np.array(pts, dtype="float32")

  # Top-left has smallest sum, bottom-right has largest sum
  s = pts.sum(axis=1)
  rect = np.zeros((4, 2), dtype="float32")
  rect[0] = pts[np.argmin(s)]
  rect[2] = pts[np.argmax(s)]

  # Top-right has smallest difference, bottom-left has largest difference
  diff = np.diff(pts, axis=1)
  rect[1] = pts[np.argmin(diff)]
  rect[3] = pts[np.argmax(diff)]

  return rect
  
def perspectiveTransform(pts, img, padding, skew):
    pts_src = order_points(pts)

    # Push each source corner outward along the direction from center to that corner
    center = pts_src.mean(axis=0)
    directions = pts_src - center
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    unit_dirs = directions / norms
    pts_src = (pts_src + unit_dirs * padding)
    
    if not skew:
        # Axis-aligned bounding box of the padded points
        x_min, y_min = pts_src.min(axis=0)
        x_max, y_max = pts_src.max(axis=0)

        # Clamp to image bounds
        h, w = img.shape[:2]
        x_min = int(max(0, np.floor(x_min)))
        y_min = int(max(0, np.floor(y_min)))
        x_max = int(min(w, np.ceil(x_max)))
        y_max = int(min(h, np.ceil(y_max)))

        return img[y_min:y_max, x_min:x_max]

    pts_src = pts_src.astype("float32")
    (tl, tr, br, bl) = pts_src

    # Calculate the true maximum width and height of the (now padded) box
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    # Destination stays a plain flat rectangle — no extra offset needed
    pts_dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32")

    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    return cv2.warpPerspective(img, M, (max_width, max_height))
    
# ---------------------------------------------------------------------------
# Image Functions
# ---------------------------------------------------------------------------
 
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
	if (settings["deskew"] and len(approx_corners) == 4):
		box = approx_corners.reshape(4, 2)
		print("skewing")
	else:
	# Fallback to standard bounding box if shape is too noisy
		rect = cv2.minAreaRect(pts)
		box = np.intp(cv2.boxPoints(rect))

	return perspectiveTransform(box,img, settings["border_padding"],settings["deskew"])
    
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
    p.add_argument('--deskew', action='store_true',
                    help='Attempt to deskewing the detected card/page')
    return p

def get_setting_args(args):
    return {
        "threshhold_strength": getattr(args, "threshhold_val", 120),
        "threshhold1_type": getattr(args, "threshhold1_type", 0),
        "blur_strength": getattr(args, "blur", 10),
        "border_padding": getattr(args, "pad", 20),
        "deskew": args.perspective_skew
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