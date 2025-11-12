#!/usr/bin/env python
"""
Nathan Herling
File: clip_mkv_segment.py

Trim a segment from an MKV file and save it as a new MKV file.
Supports fractional seconds (e.g., 0.3, 3.7).
"""

import os
from moviepy import VideoFileClip
import inspect

# ===================== CONFIG =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "_data")

INPUT_PATH  = os.path.join(DATA_DIR, "use_for_cutting_fig.mkv")
OUTPUT_PATH = os.path.join(BASE_DIR, "use_for_cutting_fig_CLIPPED.mkv")

START_TIME  = 2.9   # seconds
END_TIME    = 16  # seconds
# ==================================================


def get_subclip(clip, start_t, end_t):
    """Return a subclip using whichever API this MoviePy version provides."""
    if hasattr(clip, "subclip"):
        return clip.subclip(start_t, end_t)
    if hasattr(clip, "subclipped"):
        return clip.subclipped(start_t, end_t)
    raise AttributeError("Neither 'subclip' nor 'subclipped' found on VideoFileClip.")


def write_videofile_compat(clip, output_path, **kwargs):
    """
    Call write_videofile() only with arguments supported by the installed MoviePy.
    """
    sig = inspect.signature(clip.write_videofile)
    valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    clip.write_videofile(output_path, **valid_kwargs)


def clip_mkv(input_path, output_path, start_t, end_t):
    """Trim an MKV file to the desired segment."""
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"❌ Input file not found:\n{input_path}")

    clip = VideoFileClip(input_path)
    print(f"🎞️  Loaded video: {os.path.basename(input_path)} ({clip.duration:.2f}s total)")

    if end_t <= start_t:
        clip.close()
        raise ValueError(f"End time ({end_t}) must be greater than start time ({start_t}).")

    start_t = max(0.0, start_t)
    end_t = min(end_t, clip.duration)

    subclip = get_subclip(clip, start_t, end_t)

    print(f"✂️  Clipping from {start_t:.1f}s to {end_t:.1f}s...")

    # Call write_videofile with only supported args
    write_videofile_compat(
        subclip,
        output_path,
        codec="libx264",
        audio_codec="aac",
        fps=clip.fps if hasattr(clip, "fps") else 24,
    )

    clip.close()
    subclip.close()

    print(f"✅ Saved clipped file:\n{os.path.abspath(output_path)}")


if __name__ == "__main__":
    clip_mkv(INPUT_PATH, OUTPUT_PATH, START_TIME, END_TIME)
