"""Frame encoders: H.264 (libx264 on the CPU or NVENC on the GPU) and JPEG. All return bytes. Needs PyAV and simplejpeg."""

import io
from fractions import Fraction

import av
import numpy as np


class H264Encoder:
    """Low-latency H.264 in Annex B (SPS/PPS repeated on every keyframe), no B-frames, one packet per frame."""

    def __init__(self, width, height, backend="libx264", fps=60, quality=26):
        self.ctx = av.CodecContext.create(backend, "w")
        self.ctx.width, self.ctx.height, self.ctx.pix_fmt = width, height, "yuv420p"
        (
            self.ctx.time_base,
            self.ctx.framerate,
            self.ctx.gop_size,
            self.ctx.max_b_frames,
        ) = Fraction(1, fps), fps, 600, 0
        if backend == "libx264":
            self.ctx.options = {
                "preset": "ultrafast",
                "tune": "zerolatency",
                "crf": str(quality),
                "profile": "baseline",
                "threads": "2",
            }
        else:  # h264_nvenc
            self.ctx.options = {
                "preset": "p1",
                "tune": "ull",
                "rc": "constqp",
                "qp": str(quality),
                "zerolatency": "1",
                "profile": "baseline",
                "forced-idr": "1",
                "delay": "0",
                "rc-lookahead": "0",
            }
        self.ctx.open()
        self.index = 0
        self.size = (width, height)

    def encode(self, rgb, keyframe=False):
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        frame.pts = self.index
        self.index += 1
        if keyframe:
            frame.pict_type = av.video.frame.PictureType.I
        packets = self.ctx.encode(frame)
        return b"".join(bytes(p) for p in packets), any(p.is_keyframe for p in packets)


class JpegEncoder:
    def __init__(self, quality=80):
        import simplejpeg

        self.quality = quality
        self.simplejpeg = simplejpeg

    def encode(self, rgb, keyframe=False, quality=None):
        return self.simplejpeg.encode_jpeg(
            np.ascontiguousarray(rgb), quality=quality or self.quality, colorspace="RGB"
        ), True
