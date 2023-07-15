# Backscrub_Alt

I was trying to find an alternative to https://github.com/floe/backscrub and tried models from https://github.com/anilsathyan7/Portrait-Segmentation/tree/master/models/portrait_video

Also, portrait_video.py writes the output to stdout as binary which is forwarded to v4l2 device using ffmpeg.

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

