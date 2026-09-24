#!/usr/bin/python3

import sys
import os
import argparse
import cv2
import numpy as np
import ctypes
from ctypes import wintypes

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

try:
    import tkinter as tk
    from tkinter import ttk, filedialog
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

def perspectiveTransform(pts_src, img, skew):
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

def pad_corners(pts, padding):
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
 
    center = rect.mean(axis=0)
    directions = rect - center
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    unit_dirs = directions / norms
    return rect + unit_dirs * padding
    
# ---------------------------------------------------------------------------
# Image Functions
# ---------------------------------------------------------------------------
     
def apply_levels(img, black, white):
    """Photoshop-style black/white point stretch: maps `black` -> 0 and
    `white` -> 255 (clipping anything outside that range), to boost
    contrast on washed-out photos before thresholding. Guards against an
    inverted/zero-width range if the sliders cross."""
    black = min(black, white - 1)
    img_f = img.astype(np.float32)
    stretched = (img_f - black) * (255.0 / (white - black))
    return np.clip(stretched, 0, 255).astype(np.uint8)


def ProcessFile(input, settings):
	steps = {'error': None, 'result': None}

	##Read file as input
	img = cv2.imread(input)
	if img is None:
		print(f"Skipping {img}: could not read image")
		return None
		
	steps['source'] = img

	##Levels stretch (black/white point) to help detection on low-contrast
	##photos - this only feeds the threshold/contour pass below; the final
	##crop still comes from the untouched original `img`
	leveled = apply_levels(img, settings["levels_black"], settings["levels_white"])
	steps['leveled'] = leveled

	##Blur & Desaturate source image to remove artifacts & better to find contours
	blurred = cv2.blur(leveled, (settings["blur_strength"],settings["blur_strength"]))
	imgray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
	
	##Change the threshold of the levels to produce simple geometry
	_, threshed = cv2.threshold(imgray, settings["threshhold_strength"], 255, settings["threshhold1_type"])
	steps['threshhold_preview'] = threshed
    
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
    
	##Apply padding once so the previews and the final crop agree
	padded_box = pad_corners(box, settings["border_padding"])
	padded_box_draw = np.intp(padded_box)

    #Draw Previews
	threshold_crop_previw = cv2.cvtColor(threshed, cv2.COLOR_GRAY2BGR)
	cv2.drawContours(threshold_crop_previw, [padded_box_draw], 0, (0, 0, 255), 3)
	steps['detected_bounds'] = threshold_crop_previw

	source_crop_preview = img.copy()
	cv2.drawContours(source_crop_preview, [padded_box_draw], 0, (0, 0, 255), 3)
	steps['crop_preview'] = source_crop_preview

	steps['result'] = perspectiveTransform(padded_box, img, settings["deskew"])
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
    p.add_argument('--levels_black', type=int, default=0,
                    help='Levels black point, 0-254 (default: 0)')
    p.add_argument('--levels_white', type=int, default=255,
                    help='Levels white point, 1-255 (default: 255)')
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
        "levels_black": getattr(args, "levels_black", 0),
        "levels_white": getattr(args, "levels_white", 255),
        "blur_strength": getattr(args, "blur", 10),
        "border_padding": getattr(args, "pad", 20),
        "deskew": args.deskew
    }
    
def create_settings_for_ui(threshhold_val=120, threshhold_type=0, levels_black=0, levels_white=255, blur_size=5, pad=10, deskew=False):
    return {
        "threshhold_strength": threshhold_val,
        "threshhold1_type": threshhold_type,
        "levels_black": levels_black,
        "levels_white": levels_white,
        "blur_strength": blur_size,
        "border_padding": pad,
        "deskew": deskew
    }
      
# ---------------------------------------------------------------------------
# File System
# ---------------------------------------------------------------------------

def _pick_source_via_dialog():
    if not (TKINTER_AVAILABLE and PIL_AVAILABLE):
        PrintError()
        return None

    picker_root = tk.Tk()
    picker_root.withdraw()  # only the dialogs themselves should be visible

    selected = filedialog.askopenfilenames(
        title="Select card images",
        filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                    ("All files", "*.*")],
        parent=picker_root,
    )
    picker_root.destroy()
    if not selected:
        return None
    args = build_parser().parse_args([])
    args.dir = False
    args.paths = list(selected)

    args.ui = True  # a double-click launch implies the interactive editor
    return args
    
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
# WND PROC Hooks
# ---------------------------------------------------------------------------
class _Win32ResizeHook:
    """Subclasses the toplevel's WndProc to expose true resize-start /
    resize-end signals that Tk has no access to. Windows-only."""

    WM_ENTERSIZEMOVE = 0x0231
    WM_EXITSIZEMOVE = 0x0232
    GWLP_WNDPROC = -4
    
    def __init__(self, tk_root, on_enter=None, on_exit=None):
        self._on_enter = on_enter
        self._on_exit = on_exit

        user32 = ctypes.windll.user32
        tk_root.update()
        self.hwnd = wintypes.HWND(ctypes.windll.user32.GetParent(tk_root.winfo_id()))

        # LRESULT/LONG_PTR are pointer-sized (8 bytes on x64) - getting this
        # wrong is the #1 way this kind of code crashes silently on 64-bit.
        LRESULT = ctypes.c_ssize_t
        self.WNDPROC = ctypes.WINFUNCTYPE(
            LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)

        # *Ptr variants are required for 64-bit safety; 32-bit Python has no
        # SetWindowLongPtrW, so fall back to SetWindowLongW there.
        self._set_wndproc = getattr(user32, "SetWindowLongPtrW", user32.SetWindowLongW)
        self._set_wndproc.restype = ctypes.c_void_p
        self._set_wndproc.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_void_p]

        self._call_wndproc = user32.CallWindowProcW
        self._call_wndproc.restype = LRESULT
        self._call_wndproc.argtypes = [
            ctypes.c_void_p, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]

        # Keep this reference alive for the hook's lifetime - if the ctypes
        # callback object gets garbage collected, Windows will eventually
        # call into freed memory and hard-crash the process.
        self._new_proc = self.WNDPROC(self._wnd_proc)
        self._old_proc = self._set_wndproc(
            self.hwnd, self.GWLP_WNDPROC, ctypes.cast(self._new_proc, ctypes.c_void_p))

    def _wnd_proc(self, hwnd, msg, wparam, lparam):
        #print(f"msg=0x{msg:04X}")
        if msg == 0x0231:
            print("Entered size/move loop")
            self._on_enter()
        elif msg == 0x0232:
            print("Exited size/move loop")
            self._on_exit()
        return self._call_wndproc(self._old_proc, hwnd, msg, wparam, lparam)

    def uninstall(self):
        """Restore the original WndProc. Call this before the Tk window is
        destroyed - otherwise Windows can call into a dead Python callback
        during teardown and crash on exit."""
        if self._old_proc is not None:
            self._set_wndproc(self.hwnd, self.GWLP_WNDPROC, self._old_proc)
            self._old_proc = None

# ---------------------------------------------------------------------------
# Editor UI
# ---------------------------------------------------------------------------

DISPLAY_STEPS = [
    ('source',              'Original'),
    ('leveled',             'Levels'),
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
        self._photo_references = {}
        self._preview_win = None
        self._preview_img_lbl = None
        self._preview_photo = None
        self._preview_key = None
        self._viewportWidth = THUMB_SIZE
        self._viewportHeight = THUMB_SIZE
        self._resize_hook = None
        self._resizePending = False
        self._resizing = False
        self._was_resizing = False

        self.root = tk.Tk()
        self.root.title("Card Crop Editor")
        self.root.protocol("WM_DELETE_WINDOW", self._on_quit)
        self.root.minsize(1280, 800)
        self._wait_var = tk.IntVar(value=0)

        self.threshold_val_var = tk.IntVar(value=args.threshhold_val)
        self.threshold_type_var = tk.IntVar(value=args.threshhold_type)
        self.threshold_type_label_var = tk.StringVar(
            value=_THRESH_TYPE_VALUE_TO_LABEL.get(args.threshhold_type, THRESHOLD_TYPES[0][1]))
        self.levels_black_var = tk.IntVar(value=args.levels_black)
        self.levels_white_var = tk.IntVar(value=args.levels_white)
        self.blur_var = tk.IntVar(value=args.blur)
        self.pad_var = tk.IntVar(value=args.pad)
        self.deskew = tk.BooleanVar(value=args.deskew)
        self._debounce_id = None

        self._build_controls()
        self._build_viewports()

        self.root.update_idletasks()
        
        self._resize_hook = _Win32ResizeHook(self.root, on_enter=self._on_resize_begin, on_exit=self._on_resize_end)        
        self._start_resize_watcher()
        
        self._initialized = True
        
    def _on_resize_begin(self):
        self._resizing = True

    def _on_resize_end(self):
        self._resizing = False
        
    def _start_resize_watcher(self):
        self._was_resizing = False
        self._poll_resize_state()

    def _poll_resize_state(self):
        if self._was_resizing and not self._resizing:
            for key in self.panel_labels:
                self._render_preview(key)
        self._was_resizing = self._resizing
        self.root.after(50, self._poll_resize_state)
    
    def close(self):
        if self._resize_hook is not None:
            self._resize_hook.uninstall()
        try:
            self.root.destroy()
        except tk.TclError:
            pass
                
    def _refresh_all_previews(self):
        if _resizePending:
            _resizePending = False
            for key in self.panel_labels:
                self._render_preview(key)

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
        self._make_slider_group(controls_row, "Levels - Black",
                                 self.levels_black_var, 0, 254).pack(side=tk.LEFT, padx=(0, 8))
        self._make_slider_group(controls_row, "Levels - White",
                                 self.levels_white_var, 1, 255).pack(side=tk.LEFT, padx=(0, 8))
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
            grid_frame.rowconfigure(r, weight=1)
            grid_frame.columnconfigure(c, weight=1)
            cell = ttk.Frame(grid_frame, borderwidth=1, relief='solid', padding=4)
            cell.grid(row=r, column=c, padx=4, pady=4, sticky="nsew")
            cell.rowconfigure(0, weight=0)  # Title row stays compact
            cell.rowconfigure(1, weight=1)  # Debug container takes remaining space
            cell.columnconfigure(0, weight=1)
            title_lbl = ttk.Label(cell, text=title, font=('', 9, 'bold'))
            title_lbl.pack(side=tk.TOP)
            cell.pack_propagate(False)
            debug_panel = tk.Frame(cell, background="magenta", bd=0, highlightthickness=0)
            debug_panel.grid(row=1, column=0, sticky="nsew")
            debug_panel.rowconfigure(0, weight=1)
            debug_panel.columnconfigure(0, weight=1)
            canvas = tk.Canvas(
                debug_panel, 
                width=1, 
                height=1, 
                highlightthickness=0, 
                bd=0, 
                cursor="hand2",
                background="#00ff00" 
            )
            canvas.grid(row=0, column=0, sticky="nsew")
            canvas.bind("<Configure>", lambda event, k=key: self._on_viewport_configure(event, k))
            canvas.bind('<Button-1>', lambda e, k=key: self._show_preview(k))
            #img_container.bind('<Configure>', self._on_label_resize)
            self.panel_labels[key] = canvas

    def _on_viewport_configure(self, event, key):
        """Triggers a clean re-render using a guaranteed, explicitly passed key."""
        # 1. Ignore early structural layout calculations
        if event.width < 10 or event.height < 10:
            return
            
        self._viewportWidth = event.height
        self._viewportHeight = event.height
        self._resizePending = True
        
    def _on_update_preview(self):
        if not hasattr(self, 'last_steps') or not self.last_steps:
            for key in self.panel_labels:
                canvas = self.panel_labels[key]
                canvas.delete("all")
                canvas.create_text(
                    canvas.winfo_width() // 2,
                    canvas.winfo_height() // 2,
                    text="Awaiting Data...",
                    fill="#888888",
                    anchor="center"
                )
            return

        # Re-render every panel now that the drag has settled
        for key in self.panel_labels:
            self._render_preview(key)
            

    def _on_canvas_resize(self, event):
        """Safely resizes the actual image data because canvas layout is locked."""
        canvas = event.widget
        key = getattr(canvas, 'key', None)
        if not key or not self.last_steps or key not in self.last_steps:
            return

        # 1. Get the real bounding box of the canvas
        w, h = event.width, event.height
        if w <= 10 or h <= 10:
            return

        # 2. Determine the maximum dimension to fit inside this viewport (accounting for aspect ratio)
        max_dim = min(w, h)

        # 3. Re-render the image asset at the exact scaled size
        # (This will NOT trigger another resize event because the canvas size is fixed by the grid)
        arr = self.last_steps.get(key)
        photo = self._to_photo(arr, None, max_dim=max_dim)
        
        self._photo_references[key] = photo
        canvas.delete("all")
        canvas.create_image(w // 2, h // 2, image=photo, anchor="center")

    # -- image conversion ----------------------------------------------
    def _to_photo(self, bgr_img, placeholder_text=None, max_dim=THUMB_SIZE, size=None):
        # 1. Handle missing or empty images using basic Pillow layout placeholders
        if bgr_img is None or bgr_img.size == 0:
            tgt_w, tgt_h = size if size else (max_dim, max_dim)
            pil = Image.new('RGB', (tgt_w, tgt_h), (50, 50, 50))
            if placeholder_text:
                d = ImageDraw.Draw(pil)
                d.text((10, 10), placeholder_text, fill=(255, 80, 80))
            return ImageTk.PhotoImage(pil)

        # 2. Extract dimensions directly from the raw NumPy shape tuple
        # OpenCV shape returns (height, width, channels)
        src_h, src_w = bgr_img.shape[:2]

        if size is not None:
            canvas_w, canvas_h = size

            # Calculate fast float scaling vectors
            ratio_w = canvas_w / src_w
            ratio_h = canvas_h / src_h
            scale = min(ratio_w, ratio_h)

            # Round down to valid integer dimensions
            tgt_w = max(1, int(src_w * scale))
            tgt_h = max(1, int(src_h * scale))

            # --- OPTIMIZED OPENCV RESIZE INTERPOLATION ---
            # cv2.INTER_AREA is best for downscaling; cv2.INTER_LINEAR is fastest for upscaling
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            resized_bgr = cv2.resize(bgr_img, (tgt_w, tgt_h), interpolation=interp)
        else:
            # Fallback scaling rule using max_dim constraints via cv2
            scale = max_dim / max(src_w, src_h)
            if scale < 1.0:
                resized_bgr = cv2.resize(bgr_img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            else:
                resized_bgr = bgr_img

        # 3. Final Conversion pass directly to Tkinter layout asset
        rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
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
        arr = self.last_steps.get(key) if self.last_steps else None
        canvas = self.panel_labels.get(key)
        
        if canvas:
            w = canvas.winfo_width()
            h = canvas.winfo_height()
            
            if w <= 10 or h <= 10:
                w, h = 200, 200

            canvas.delete("all")

            # Passes the absolute viewport bounds down to the ratio calculator
            photo = self._to_photo(arr, "N/A" if arr is None else None, size=(w, h))
            
            self._photo_references[key] = photo
            
            canvas.create_image(
                w // 2, 
                h // 2, 
                image=photo, 
                anchor="center",
                tags="preview_img"
            )


    def _show_preview(self, key):
        #self._preview_key = key
        #self._ensure_preview_window()
        self._render_preview(key)
        #self._preview_win.deiconify()
        #self._preview_win.lift()

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
            self.levels_black_var.get(),
            self.levels_white_var.get(),
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
            canvas = self.panel_labels[key]
            if canvas:
                # Clear any previous images drawn on this canvas frame
                canvas.delete("all")

                w = canvas.winfo_width()
                h = canvas.winfo_height()

                if w <= 1 or h <= 1:
                    w, h = 200, 200 
                
                print(w)
                print(h)
                
                # Draw the image positioned exactly at the center of the canvas
                # "anchor='center'" acts like .NET's Center alignment profile
                canvas.create_image(
                    w // 2, 
                    h // 2, 
                    image=photo, 
                    anchor="center",
                    tags="preview_img"
                )
            #self.panel_labels[key].configure(image=photo)

       #self._refresh_preview_if_open()

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
        self.levels_black_var.set(self.args.levels_black)
        self.levels_white_var.set(self.args.levels_white)
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
    if len(sys.argv) == 1:
        args = _pick_source_via_dialog()
        if args is None:
            sys.exit(0)  # user cancelled - quiet exit, not an error
    else:
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