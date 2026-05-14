"""
Mask Alignment Tool
-------------------
Shows the Blender render (left) and the real brain image (right) side by side.
Click corresponding landmark points on each side to compute a homography.
Once happy with the alignment, save the warped masks and run DCT extraction.

Usage:
    python align_masks.py \
        --render  000002.bmp \
        --real    mousebrain2640x640.png \
        --mask    000002.txt \
        --output  000002_aligned.txt

Requirements: Python stdlib + Pillow + numpy (no scipy needed)
"""

import argparse
import os
import sys
import json
import subprocess
import tkinter as tk
from tkinter import messagebox, font as tkfont
import numpy as np
from PIL import Image, ImageDraw, ImageTk


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def compute_homography(src_pts, dst_pts):
    """
    Compute 3x3 homography mapping src_pts → dst_pts via DLT + numpy SVD.
    Requires >= 4 point pairs.
    """
    assert len(src_pts) >= 4 and len(src_pts) == len(dst_pts)
    A = []
    for (x, y), (xp, yp) in zip(src_pts, dst_pts):
        A.append([-x, -y, -1,  0,  0,  0, x*xp, y*xp, xp])
        A.append([ 0,  0,  0, -x, -y, -1, x*yp, y*yp, yp])
    A = np.array(A, dtype=np.float64)
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2]


def apply_homography_pt(H, x, y):
    """Map a single point through homography H."""
    p = H @ np.array([x, y, 1.0])
    return p[0] / p[2], p[1] / p[2]


def warp_polygon(H, pts_norm, W, H_img):
    """Warp normalised polygon coords through H, return pixel coords."""
    out = []
    for nx, ny in pts_norm:
        wx, wy = apply_homography_pt(H, nx * W, ny * H_img)
        out.append((wx, wy))
    return out


# ---------------------------------------------------------------------------
# Mask I/O
# ---------------------------------------------------------------------------

def parse_masks_norm(txt_path):
    """Return list of (class_id, [(nx,ny), ...]) — normalised coords."""
    instances = []
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            cls = int(parts[0])
            coords = list(map(float, parts[1:]))
            pts = [(coords[i], coords[i + 1]) for i in range(0, len(coords) - 1, 2)]
            instances.append((cls, pts))
    return instances


def save_masks_norm(instances, W, H, out_path):
    """Write YOLO-format TXT from (class, pixel_pts) list."""
    with open(out_path, "w") as f:
        for cls, pts_px in instances:
            coords = []
            for x, y in pts_px:
                coords.append(round(np.clip(x / W, 0, 1), 6))
                coords.append(round(np.clip(y / H, 0, 1), 6))
            f.write(f"{cls} " + " ".join(map(str, coords)) + "\n")


# ---------------------------------------------------------------------------
# Overlay renderer
# ---------------------------------------------------------------------------

PALETTE = [
    (255,  80,  80, 110), (80,  255,  80, 110), ( 80, 120, 255, 110),
    (255, 230,  80, 110), (255,  80, 220, 110), ( 80, 230, 230, 110),
]


def render_overlay(base_img: np.ndarray, instances_px, alpha_scale=1.0):
    """Render coloured mask overlay on top of base_img (numpy RGB)."""
    base = Image.fromarray(base_img).convert("RGBA")
    for i, (cls, pts_px) in enumerate(instances_px):
        if len(pts_px) < 3:
            continue
        color = PALETTE[cls % len(PALETTE)]
        color = color[:3] + (int(color[3] * alpha_scale),)
        overlay  = Image.new("RGBA", base.size, (0, 0, 0, 0))
        mask_img = Image.new("L",    base.size, 0)
        ImageDraw.Draw(mask_img).polygon(pts_px, fill=color[3])
        ImageDraw.Draw(overlay).polygon(pts_px, outline=(255, 255, 255, 200))
        colored  = Image.new("RGBA", base.size, color)
        overlay.paste(colored, mask=mask_img)
        base = Image.alpha_composite(base, overlay)
    return np.array(base.convert("RGB"))


def masks_to_pixel(instances_norm, W, H):
    return [(cls, [(nx * W, ny * H) for nx, ny in pts])
            for cls, pts in instances_norm]


# ---------------------------------------------------------------------------
# Main GUI
# ---------------------------------------------------------------------------

class AlignTool:
    POINT_RADIUS = 6
    COLORS = ["#FF3333", "#33FF33", "#3399FF", "#FFD700",
              "#FF33FF", "#33FFFF", "#FF8C00", "#ADFF2F"]

    def __init__(self, render_path, real_path, mask_path, output_path, dct_script):
        self.render_path  = render_path
        self.real_path    = real_path
        self.mask_path    = mask_path
        self.output_path  = output_path
        self.dct_script   = dct_script

        # Load images
        self.render_img = np.array(Image.open(render_path).convert("RGB"))
        self.real_img   = np.array(Image.open(real_path).convert("RGB"))
        self.H, self.W  = self.real_img.shape[:2]

        # Load masks
        self.instances_norm = parse_masks_norm(mask_path)

        # Landmark state
        self.src_pts = []   # points on render (pixels)
        self.dst_pts = []   # points on real   (pixels)
        self.pending_src = None   # point clicked on left, waiting for right
        self.H_matrix = None      # current homography (once >= 4 pairs)

        # Build UI
        self.root = tk.Tk()
        self.root.title("Mask Alignment Tool — click pairs of landmarks")
        self._build_ui()
        self._refresh()
        self.root.mainloop()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        top = tk.Frame(self.root)
        top.pack(fill=tk.BOTH, expand=True)

        # Left canvas — render
        left_frame = tk.Frame(top)
        left_frame.pack(side=tk.LEFT, padx=5, pady=5)
        tk.Label(left_frame, text="Blender render  (click landmarks here first)",
                 font=("Arial", 10, "bold")).pack()
        self.left_canvas = tk.Canvas(left_frame, width=self.W, height=self.H,
                                     cursor="crosshair")
        self.left_canvas.pack()
        self.left_canvas.bind("<Button-1>", self._on_left_click)

        # Right canvas — real brain
        right_frame = tk.Frame(top)
        right_frame.pack(side=tk.LEFT, padx=5, pady=5)
        tk.Label(right_frame, text="Real brain  (then click corresponding point here)",
                 font=("Arial", 10, "bold")).pack()
        self.right_canvas = tk.Canvas(right_frame, width=self.W, height=self.H,
                                      cursor="crosshair")
        self.right_canvas.pack()
        self.right_canvas.bind("<Button-1>", self._on_right_click)

        # Bottom controls
        btm = tk.Frame(self.root)
        btm.pack(fill=tk.X, padx=10, pady=8)

        self.status_var = tk.StringVar(value="Step 1: click a landmark on the LEFT (render), then click its match on the RIGHT (real brain).")
        tk.Label(btm, textvariable=self.status_var, wraplength=700,
                 justify=tk.LEFT, fg="#333333").pack(side=tk.LEFT, expand=True)

        btn_frame = tk.Frame(btm)
        btn_frame.pack(side=tk.RIGHT)

        self.undo_btn = tk.Button(btn_frame, text="Undo last pair",
                                  command=self._undo, state=tk.DISABLED)
        self.undo_btn.pack(side=tk.LEFT, padx=4)

        self.preview_btn = tk.Button(btn_frame, text="Preview alignment",
                                     command=self._preview, state=tk.DISABLED,
                                     bg="#2196F3", fg="white")
        self.preview_btn.pack(side=tk.LEFT, padx=4)

        self.save_btn = tk.Button(btn_frame,
                                  text="Save aligned masks + run DCT",
                                  command=self._save_and_run, state=tk.DISABLED,
                                  bg="#4CAF50", fg="white",
                                  font=("Arial", 10, "bold"))
        self.save_btn.pack(side=tk.LEFT, padx=4)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_left_click(self, event):
        if self.pending_src is not None:
            # Replace pending
            self.pending_src = (event.x, event.y)
        else:
            self.pending_src = (event.x, event.y)
        self._refresh()
        self.status_var.set(
            f"Landmark on render set at ({event.x}, {event.y}).  "
            f"Now click the matching point on the RIGHT image."
        )

    def _on_right_click(self, event):
        if self.pending_src is None:
            self.status_var.set("Click a point on the LEFT first.")
            return
        self.src_pts.append(self.pending_src)
        self.dst_pts.append((event.x, event.y))
        self.pending_src = None

        n = len(self.src_pts)
        self.undo_btn.config(state=tk.NORMAL)

        if n >= 4:
            self._compute_homography()
            self.preview_btn.config(state=tk.NORMAL)
            self.save_btn.config(state=tk.NORMAL)
            self.status_var.set(
                f"{n} pairs — homography computed.  "
                f"Add more points for a better fit, or click Preview / Save."
            )
        else:
            remaining = 4 - n
            self.status_var.set(
                f"{n} pair(s) set.  Need {remaining} more before alignment can be computed."
            )
        self._refresh()

    def _undo(self):
        if self.pending_src is not None:
            self.pending_src = None
        elif self.src_pts:
            self.src_pts.pop()
            self.dst_pts.pop()
        n = len(self.src_pts)
        if n < 4:
            self.H_matrix = None
            self.preview_btn.config(state=tk.DISABLED)
            self.save_btn.config(state=tk.DISABLED)
        else:
            self._compute_homography()
        self.undo_btn.config(state=tk.NORMAL if (self.src_pts or self.pending_src) else tk.DISABLED)
        self.status_var.set(f"Undone. {n} pair(s) remaining.")
        self._refresh()

    # ------------------------------------------------------------------
    # Homography
    # ------------------------------------------------------------------

    def _compute_homography(self):
        try:
            self.H_matrix = compute_homography(self.src_pts, self.dst_pts)
        except Exception as e:
            self.H_matrix = None
            self.status_var.set(f"Homography failed: {e}")

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def _refresh(self):
        self._draw_left()
        self._draw_right()

    def _draw_left(self):
        """Render + mask overlay + src landmark dots."""
        # Mask overlay on render
        inst_px = masks_to_pixel(self.instances_norm, self.W, self.H)
        disp = render_overlay(self.render_img, inst_px, alpha_scale=0.6)
        img = Image.fromarray(disp)
        draw = ImageDraw.Draw(img)

        # Draw confirmed src points
        for i, (x, y) in enumerate(self.src_pts):
            color = self.COLORS[i % len(self.COLORS)]
            r = self.POINT_RADIUS
            draw.ellipse([x - r, y - r, x + r, y + r],
                         fill=color, outline="white")
            draw.text((x + r + 2, y - r), str(i + 1), fill=color)

        # Draw pending
        if self.pending_src:
            x, y = self.pending_src
            r = self.POINT_RADIUS
            draw.ellipse([x - r, y - r, x + r, y + r],
                         fill="#FFFFFF", outline="#AAAAAA")
            draw.text((x + r + 2, y - r), "?", fill="white")

        self._tk_left = ImageTk.PhotoImage(img)
        self.left_canvas.create_image(0, 0, anchor=tk.NW, image=self._tk_left)

    def _draw_right(self):
        """Real brain + (if homography exists) warped mask overlay + dst dots."""
        if self.H_matrix is not None:
            inst_px = self._warped_instances_px()
            disp = render_overlay(self.real_img, inst_px, alpha_scale=0.7)
        else:
            disp = self.real_img.copy()

        img = Image.fromarray(disp)
        draw = ImageDraw.Draw(img)

        for i, (x, y) in enumerate(self.dst_pts):
            color = self.COLORS[i % len(self.COLORS)]
            r = self.POINT_RADIUS
            draw.ellipse([x - r, y - r, x + r, y + r],
                         fill=color, outline="white")
            draw.text((x + r + 2, y - r), str(i + 1), fill=color)

        self._tk_right = ImageTk.PhotoImage(img)
        self.right_canvas.create_image(0, 0, anchor=tk.NW, image=self._tk_right)

    def _preview(self):
        if self.H_matrix is not None:
            self._compute_homography()
            self._draw_right()
            self.status_var.set("Preview updated. Add more landmarks or click Save.")

    # ------------------------------------------------------------------
    # Warped masks
    # ------------------------------------------------------------------

    def _warped_instances_px(self):
        """Apply current homography to all mask polygons, return pixel coords."""
        out = []
        for cls, pts_norm in self.instances_norm:
            pts_src_px = [(nx * self.W, ny * self.H) for nx, ny in pts_norm]
            pts_dst_px = [apply_homography_pt(self.H_matrix, x, y)
                          for x, y in pts_src_px]
            out.append((cls, pts_dst_px))
        return out

    # ------------------------------------------------------------------
    # Save & run DCT
    # ------------------------------------------------------------------

    def _save_and_run(self):
        if self.H_matrix is None:
            messagebox.showerror("Not ready", "Need at least 4 landmark pairs first.")
            return

        # Save aligned TXT
        warped = self._warped_instances_px()
        save_masks_norm(warped, self.W, self.H, self.output_path)

        # Save homography for reference
        h_path = self.output_path.replace(".txt", "_homography.json")
        with open(h_path, "w") as f:
            json.dump({"H": self.H_matrix.tolist(),
                       "src_pts": self.src_pts,
                       "dst_pts": self.dst_pts}, f, indent=2)

        self.status_var.set(f"Aligned masks saved → {os.path.basename(self.output_path)}\nRunning DCT...")
        self.root.update()

        # Run DCT extractor
        out_csv = self.output_path.replace(".txt", "_features.csv")
        result = subprocess.run(
            [sys.executable, self.dct_script,
             "--image",  self.real_path,
             "--mask",   self.output_path,
             "--output", out_csv],
            capture_output=True, text=True,
            cwd=os.path.dirname(os.path.abspath(self.output_path))
        )

        if result.returncode == 0:
            self.status_var.set(
                f"Done!\n"
                f"Aligned masks → {os.path.basename(self.output_path)}\n"
                f"Features CSV  → {os.path.basename(out_csv)}"
            )
            messagebox.showinfo("Complete",
                                f"DCT extraction finished.\n\n"
                                f"CSV saved to:\n{out_csv}")
        else:
            self.status_var.set(f"DCT script error:\n{result.stderr[:300]}")
            messagebox.showerror("DCT Error", result.stderr[:500])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Interactive mask alignment tool")
    parser.add_argument("--render",  required=True, help="Blender render BMP (mask source)")
    parser.add_argument("--real",    required=True, help="Real brain PNG")
    parser.add_argument("--mask",    required=True, help="YOLO .txt mask file")
    parser.add_argument("--output",  required=True, help="Output aligned .txt path")
    parser.add_argument("--dct",     default="dct_feature_extractor.py",
                        help="Path to dct_feature_extractor.py")
    args = parser.parse_args()

    for p in [args.render, args.real, args.mask]:
        if not os.path.exists(p):
            print(f"ERROR: File not found: {p}")
            sys.exit(1)

    AlignTool(
        render_path=args.render,
        real_path=args.real,
        mask_path=args.mask,
        output_path=args.output,
        dct_script=args.dct,
    )


if __name__ == "__main__":
    main()
