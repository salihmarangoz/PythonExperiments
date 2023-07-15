# Backscrub_Alt

I was trying to find an alternative to [floe/backscrub](https://github.com/floe/backscrub) and tried models from [anilsathyan7/Portrait-Segmentation](anilsathyan7/Portrait-Segmentation). Also, `portrait_video.py` writes the output image to stdout in binary which is forwarded to v4l2loopback device using ffmpeg.

## Requirements

- v4l2loopback
- ffmpeg

These are Python requirements:

- PIL
- numba
- cv2
- tensorflow

## Run

```bash
bash run.sh
```

