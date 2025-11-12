#!/usr/bin/env python
"""
Nathan Herling
Tuesday-November-11-2025
File: mkv_to_gif_converter.py

Convert a fixed segment of a video (.mkv/.mp4/…) to a .gif.
All parameters are hard-coded below.
Supports fractional seconds (e.g., 0.3, 3.7, etc).
"""

import os
from moviepy import VideoFileClip
import inspect

# ===================== CONFIG =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "_data")

INPUT_PATH  = os.path.join(DATA_DIR, "use_for_gif.mp4")
OUTPUT_PATH = os.path.join(BASE_DIR, "laser_cut.gif")

START_TIME  = 0.0     # seconds
END_TIME    = None    # None => use full clip
FPS         = 20      # GIF frames per second
WIDTH       = 600     # width in px (height auto); set to None to keep original
LOOP        = 0       # 0 = infinite (if supported)
# ==================================================


def get_subclip(clip, start_t, end_t):
    """Return a subclip using whichever API this MoviePy version provides."""
    if hasattr(clip, "subclip"):
        return clip.subclip(start_t, end_t)
    if hasattr(clip, "subclipped"):
        return clip.subclipped(start_t, end_t)
    raise AttributeError("Neither 'subclip' nor 'subclipped' found on VideoFileClip.")


def resize_clip(clip, width=None):
    """Resize using resize/resized if width is specified."""
    if width is None:
        return clip

    if hasattr(clip, "resize"):
        return clip.resize(width=width)
    if hasattr(clip, "resized"):
        return clip.resized(width=width)

    raise AttributeError("Neither 'resize' nor 'resized' found on clip object.")


def write_gif_compat(clip, output_path, fps, loop):
    """
    Call write_gif with only the args this MoviePy version supports.
    Handles presence/absence of 'loop' and 'program' cleanly.
    """
    sig = inspect.signature(clip.write_gif)
    kwargs = {}

    if "fps" in sig.parameters:
        kwargs["fps"] = fps
    if "loop" in sig.parameters:
        kwargs["loop"] = loop
    if "program" in sig.parameters:
        kwargs["program"] = "ffmpeg"

    clip.write_gif(output_path, **kwargs)


def mkv_to_gif(input_path, output_path, start_t, end_t, fps, width=None, loop=0):
    """Convert (optionally) a segment of a video file to GIF."""
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"❌ Input file not found:\n{input_path}")

    clip = VideoFileClip(input_path)
    print(f"🎞️  Loaded video: {os.path.basename(input_path)}")
    print(f"   Duration: {clip.duration:.2f} seconds")

    # Normalize start
    start_t = float(start_t)
    if start_t < 0:
        start_t = 0.0

    # Normalize end:
    # - None  -> full duration
    # - number -> clamped to duration
    if end_t is None:
        end_t = clip.duration
    else:
        end_t = float(end_t)
        if end_t > clip.duration:
            end_t = clip.duration

    if end_t <= start_t:
        clip.close()
        raise ValueError(f"End time ({end_t}) must be greater than start time ({start_t}).")

    # Build subclip
    subclip = get_subclip(clip, start_t, end_t)

    # Resize if requested
    subclip = resize_clip(subclip, width=width)

    # Write GIF (version-safe)
    write_gif_compat(subclip, output_path, fps=fps, loop=loop)

    # Cleanup
    clip.close()
    subclip.close()

    print(f"✅ GIF created successfully:\n{os.path.abspath(output_path)}")


if __name__ == "__main__":
    mkv_to_gif(INPUT_PATH, OUTPUT_PATH, START_TIME, END_TIME, FPS, WIDTH, LOOP)
