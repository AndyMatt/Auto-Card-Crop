#!/usr/bin/python3

import sys
import os
import argparse
import cv2
import numpy as np

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

try:
    import tkinter as tk
    from tkinter import ttk
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False

try:
    from PIL import Image, ImageTk, ImageDraw
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    
def PrintError():
    print("""No input files found. 
        
Arguments:
    [file|files]        - Name of file/files to process
    --dir               - Specify Input directory, '.' is supported.
    --threshhold_thresh - 0-255, Value for threshold strength.
    --threshhold1_type  - Type of Threshold Method 
        0: THRESH_BINARY        - Preferable for White Backgrounds
        1: THRESH_BINARY_INV    - Preferable for Black Backgrounds
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
	steps = {'error': None, 'result': None}

	##Read file as input
	img = cv2.imread(input)
	if img is None:
		print(f"Skipping {img}: could not read image")
		return None
		
	steps['source'] = img
    
	##Blur source image to remove artifacts
	blurred = cv2.blur(img, (settings["blur_strength"],settings["blur_strength"]))
	steps['blurred'] = blurred
    
	##Desaturate Source to make it easier to find contours
	imgray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
	
	##Change the threshold of the levels to produce simple geometry
	_, threshed = cv2.threshold(imgray, settings["threshhold_strength"], 255, settings["threshhold1_type"])
	steps['threshhold_preview'] = cv2.cvtColor(threshed, cv2.COLOR_GRAY2BGR)
    
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
	else:
	# Fallback to standard bounding box if shape is too noisy
		rect = cv2.minAreaRect(pts)
		box = np.intp(cv2.boxPoints(rect))
    
    #Draw Previews
	threshold_crop_previw = threshed.copy()
	cv2.drawContours(threshold_crop_previw, [np.int64(box)], 0, (0, 0, 255), 3)
	steps['detected_bounds'] = threshold_crop_previw

	source_crop_preview = img.copy()
	cv2.drawContours(source_crop_preview, [np.int64(box)], 0, (0, 0, 255), 3)
	steps['crop_preview'] = source_crop_preview

	steps['result'] = perspectiveTransform(box,img, settings["border_padding"],settings["deskew"])
	return steps
    
# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description="Auto-rotate and crop trading card photos (white background).")
    p.add_argument('paths', nargs='*',
                    help='Image file(s), or a directory when -d is used ("." for cwd)')
    p.add_argument('--dir', '--dir', action='store_true',
                    help='Treat the path argument as a directory and process every image in it')
    p.add_argument('--ui', action='store_true',
                    help='Open the multi-viewport editor before saving each image')
    p.add_argument('--threshhold_val', type=int, default=120,
                    help='Threshold value - card/angle detection (default: 120)')
    p.add_argument('--threshhold_type', type=int, default=0,
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
        "threshhold1_type": getattr(args, "threshhold_type", 0),
        "blur_strength": getattr(args, "blur", 10),
        "border_padding": getattr(args, "pad", 20),
        "deskew": args.deskew
    }
    
def create_settings_for_ui(threshhold_val=120, threshhold_type=0, blur_size=5, pad=10, deskew=False):
    return {
        "threshhold_strength": threshhold_val,
        "threshhold1_type": threshhold_type,
        "blur_strength": blur_size,
        "border_padding": pad,
        "deskew": deskew
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
# Editor UI
# ---------------------------------------------------------------------------

DISPLAY_STEPS = [
    ('source',              'Original'),
    ('blurred',             'Downsample/Blurred'),
    ('threshhold_preview',  'Threshold'),
    ('detected_bounds',     'Detected Boundary'),
    ('crop_preview',        'Crop Preview'),
    ('result',              'Result'),
]

THRESHOLD_TYPES = [
    (0, "THRESH_BINARY (White Background)"),
    (1, "THRESH_BINARY_INV (Black Backgrounf)"),
    (2, "THRESH_TRUNC"),
    (3, "THRESH_TOZERO"),
    (4, "THRESH_TOZERO_INV"),
]
_THRESH_TYPE_VALUE_TO_LABEL = {value: label for value, label in THRESHOLD_TYPES}
_THRESH_TYPE_LABEL_TO_VALUE = {label: value for value, label in THRESHOLD_TYPES}

THUMB_SIZE = 384
PREVIEW_SIZE = 1024


class CardCropEditor:
    def __init__(self, args):
        self.args = args
        self.last_steps = None
        self.current_img = None
        self.action = None
        self.photo_refs = {}
        self._preview_win = None
        self._preview_img_lbl = None
        self._preview_photo = None
        self._preview_key = None

        self.root = tk.Tk()
        self.root.title("Card Crop Editor")
        self.root.protocol("WM_DELETE_WINDOW", self._on_quit)
        self._wait_var = tk.IntVar(value=0)

        self.threshold_val_var = tk.IntVar(value=args.threshhold_val)
        self.threshold_type_var = tk.IntVar(value=args.threshhold_type)
        self.threshold_type_label_var = tk.StringVar(
            value=_THRESH_TYPE_VALUE_TO_LABEL.get(args.threshhold_type, THRESHOLD_TYPES[0][1]))
        self.blur_var = tk.IntVar(value=args.blur)
        self.pad_var = tk.IntVar(value=args.pad)
        self.deskew = tk.BooleanVar(value=args.deskew)
        self._debounce_id = None

        self._build_controls()
        self._build_viewports()

    # -- layout -------------------------------------------------------
    def _make_slider_group(self, parent, title, var, frm, to):
        """A labeled box containing a value readout and a working slider,
        stacked vertically with pack() so nothing overlaps."""
        box = ttk.LabelFrame(parent, text=title, padding=6)

        val_lbl = ttk.Label(box, text=str(var.get()), width=4)
        val_lbl.pack(side=tk.TOP, anchor='w')

        def on_move(v, var=var, val_lbl=val_lbl):
            var.set(int(float(v)))
            val_lbl.config(text=str(var.get()))

        def on_release(event):
            self._schedule_recompute()

        scale = ttk.Scale(box, from_=frm, to=to, orient=tk.HORIZONTAL,
                           command=on_move, length=150)
        scale.set(var.get())
        scale.bind('<ButtonRelease-1>', on_release)
        scale.bind('<KeyRelease>', on_release)  # arrow-key nudges, not just mouse drag
        scale.pack(side=tk.TOP, fill=tk.X)
        return box

    def _make_threshold_type_group(self, parent):
        """Dropdown for the cv2.threshold `type` argument, showing the enum
        name (and the black/white-background hint) instead of a raw int."""
        box = ttk.LabelFrame(parent, text="Threshold - Type", padding=6)

        combo = ttk.Combobox(box, textvariable=self.threshold_type_label_var,
                              values=[label for _, label in THRESHOLD_TYPES],
                              state='readonly', width=26)
        combo.pack(side=tk.TOP, fill=tk.X)

        def on_select(event):
            self.threshold_type_var.set(
                _THRESH_TYPE_LABEL_TO_VALUE[self.threshold_type_label_var.get()])
            self._schedule_recompute()

        combo.bind('<<ComboboxSelected>>', on_select)
        return box

    def _build_controls(self):
        top = ttk.Frame(self.root, padding=(8,0,0,0))
        top.pack(side=tk.TOP, fill=tk.X)

        self.filename_var = tk.StringVar(value="")
        ttk.Label(top, textvariable=self.filename_var, font=('', 11, 'bold')).pack(
            side=tk.TOP, anchor='w', pady=(0, 6))

        controls_row = ttk.Frame(top)
        controls_row.pack(side=tk.TOP, fill=tk.X)

        self._make_slider_group(controls_row, "Threshold - Value",
                                 self.threshold_val_var, 0, 255).pack(side=tk.LEFT, padx=(0, 8))
        self._make_threshold_type_group(controls_row).pack(side=tk.LEFT, padx=(0, 8))
        self._make_slider_group(controls_row, "Blur kernel",
                                 self.blur_var, 1, 25).pack(side=tk.LEFT, padx=(0, 8))
        self._make_slider_group(controls_row, "Padding",
                                 self.pad_var, 0, 60).pack(side=tk.LEFT, padx=(0, 8))

        deskew_box = ttk.LabelFrame(controls_row, text="Deskew", padding=6)
        ttk.Checkbutton(deskew_box, text="Enabled", variable=self.deskew,
                         command=self._schedule_recompute).pack(side=tk.TOP, anchor='w')
        deskew_box.pack(side=tk.LEFT, padx=(0, 8))

        btns = ttk.Frame(controls_row)
        btns.pack(side=tk.LEFT, padx=(20, 0))
        ttk.Button(btns, text="Reset", command=self._on_reset).pack(side=tk.LEFT, padx=2)
        ttk.Button(btns, text="Skip (n)", command=self._on_skip).pack(side=tk.LEFT, padx=2)
        ttk.Button(btns, text="Save (s)", command=self._on_save).pack(side=tk.LEFT, padx=2)
        ttk.Button(btns, text="Quit (q)", command=self._on_quit).pack(side=tk.LEFT, padx=2)

        self.status_var = tk.StringVar(value="")
        ttk.Label(self.root, textvariable=self.status_var, padding=(8, 0, 8, 6)).pack(
            side=tk.TOP, fill=tk.X)

        self.root.bind('<s>', lambda e: self._on_save())
        self.root.bind('<n>', lambda e: self._on_skip())
        self.root.bind('<q>', lambda e: self._on_quit())

    def _build_viewports(self):
        grid_frame = ttk.Frame(self.root, padding=(8, 2, 8, 8))
        grid_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.panel_labels = {}
        cols = 3
        for i, (key, title) in enumerate(DISPLAY_STEPS):
            r, c = divmod(i, cols)
            cell = ttk.Frame(grid_frame, borderwidth=1, relief='solid', padding=4)
            cell.grid(row=r, column=c, padx=4, pady=4)
            ttk.Label(cell, text=title, font=('', 9, 'bold')).pack(side=tk.TOP)
            img_lbl = ttk.Label(cell, cursor="hand2")
            img_lbl.pack(side=tk.TOP)
            img_lbl.bind('<Button-1>', lambda e, k=key: self._show_preview(k))
            self.panel_labels[key] = img_lbl

    # -- image conversion ----------------------------------------------
    def _to_photo(self, bgr_img, placeholder_text=None, max_dim=THUMB_SIZE):
        if bgr_img is None or bgr_img.size == 0:
            pil = Image.new('RGB', (max_dim, int(max_dim * 0.7)), (50, 50, 50))
            if placeholder_text:
                d = ImageDraw.Draw(pil)
                d.text((10, 10), placeholder_text, fill=(255, 80, 80))
            return ImageTk.PhotoImage(pil)
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil.thumbnail((max_dim, max_dim))
        return ImageTk.PhotoImage(pil)

    # -- click-to-zoom preview -------------------------------------------
    def _ensure_preview_window(self):
        if self._preview_win is not None:
            return
        win = tk.Toplevel(self.root)
        win.title("Preview")
        win.protocol("WM_DELETE_WINDOW", self._hide_preview)
        frame = ttk.Frame(win, padding=6)
        frame.pack(fill=tk.BOTH, expand=True)
        img_lbl = ttk.Label(frame)
        img_lbl.pack(side=tk.TOP)
        ttk.Button(frame, text="Close", command=self._hide_preview).pack(side=tk.TOP, pady=(6, 0))
        win.withdraw()
        self._preview_win = win
        self._preview_img_lbl = img_lbl

    def _refresh_preview_if_open(self):
        if self._preview_win is None or self._preview_key is None:
            return
        if self._preview_win.state() == 'withdrawn':
            return
        self._render_preview(self._preview_key)

    def _render_preview(self, key):
        title = dict(DISPLAY_STEPS).get(key, key)
        arr = self.last_steps.get(key) if self.last_steps else None
        photo = self._to_photo(arr, "N/A" if arr is None else None, max_dim=PREVIEW_SIZE)
        self._preview_photo = photo  # keep reference alive
        self._preview_win.title(f"Preview - {title}")
        self._preview_img_lbl.config(image=photo)

    def _show_preview(self, key):
        self._preview_key = key
        self._ensure_preview_window()
        self._render_preview(key)
        self._preview_win.deiconify()
        self._preview_win.lift()

    def _hide_preview(self):
        if self._preview_win is not None:
            self._preview_win.withdraw()

    # -- recompute / redraw ----------------------------------------------
    def _schedule_recompute(self):
        if self._debounce_id is not None:
            self.root.after_cancel(self._debounce_id)
        self._debounce_id = self.root.after(120, self._recompute)

    def _recompute(self):
        if self.current_img is None:
            return
        ui_settings = create_settings_for_ui(
            self.threshold_val_var.get(),
            self.threshold_type_var.get(),
            self.blur_var.get(),
            self.pad_var.get(),
            self.deskew.get())
        steps = ProcessFile(self.current_img, ui_settings)
        self.last_steps = steps

        for key, _title in DISPLAY_STEPS:
            arr = steps.get(key)
            placeholder = "N/A" if arr is None else None
            photo = self._to_photo(arr, placeholder)
            self.photo_refs[key] = photo  # keep reference alive
            self.panel_labels[key].configure(image=photo)

        self._refresh_preview_if_open()

        if steps.get('error'):
            self.status_var.set(steps['error'])
        else:
            self.status_var.set("Status: OK")

    # -- button handlers ----------------------------------------------
    def _on_reset(self):
        self.threshold_val_var.set(self.args.threshhold_val)
        self.threshold_type_var.set(self.args.threshhold_type)
        self.threshold_type_label_var.set(
            _THRESH_TYPE_VALUE_TO_LABEL.get(self.args.threshhold_type, THRESHOLD_TYPES[0][1]))
        self.blur_var.set(self.args.blur)
        self.pad_var.set(self.args.pad)
        self.deskew.set(self.args.deskew)
        self._recompute()

    def _on_save(self):
        if not self.last_steps or self.last_steps.get('roi') is None:
            self.status_var.set("Cannot save - no crop detected. Adjust sliders first.")
            return
        self.action = 'save'
        self._release()

    def _on_skip(self):
        self.action = 'skip'
        self._release()

    def _on_quit(self):
        self.action = 'quit'
        self._release()

    def _release(self):
        self._wait_var.set(self._wait_var.get() + 1)

    # -- public API ----------------------------------------------------
    def edit(self, path):
        """Show the editor for one image. Blocks until Save/Skip/Quit.
        Returns 'save' | 'skip' | 'quit'. On 'save', self.last_steps['result']
        holds the crop to write out."""                    
        self.current_img = path
        self.action = None
        self.filename_var.set(os.path.basename(path))
        self.status_var.set("")
        self._hide_preview()
        self._recompute()
        self.root.wait_variable(self._wait_var)
        return self.action

    def close(self):
        try:
            self.root.destroy()
        except tk.TclError:
            pass
   
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
        
    if args.ui and not (TKINTER_AVAILABLE and PIL_AVAILABLE):
        missing = []
        if not TKINTER_AVAILABLE:
            missing.append("tkinter (ships with standard Python; on Linux install your distro's python3-tk package)")
        if not PIL_AVAILABLE:
            missing.append("Pillow (pip install pillow)")
        print("--ui requires: " + "; ".join(missing))
        sys.exit(1)
    
    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
 
    editor = CardCropEditor(args) if args.ui else None
    
    try:
        for f in files:
            print(f"Processing {f}")
            if editor:  
                action = editor.edit(f)
                if action == 'quit':
                    print("Quit.")
                    break
                if action == 'skip':
                    print(f"Skipped {f}")
                    continue
                steps = editor.last_steps
            else:
                steps = ProcessFile(f, settings)
                
            roi = steps.get('roi')
            if roi is None:
                print(f"Failed to detect card bounds in {f}, skipping. ({steps.get('error')})")
                continue
				
            out = output_path_for(f, args.outdir)
            cv2.imwrite(out, roi)
            print(f"Saved {out}")
    finally:
        if editor:
            editor.close()
		
if __name__ == '__main__':
    main()