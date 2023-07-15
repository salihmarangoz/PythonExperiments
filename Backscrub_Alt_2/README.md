# Backscrub_Alt_2

I was trying to find an alternative to [floe/backscrub](https://github.com/floe/backscrub). I tried to use optical flow to combine it with the exponential weighted averaging. There are also other experiments with grabcut and bilateral filtering.

This doesn't output to v4l2loopback.

ref: https://github.com/fangfufu/Linux-Fake-Background-Webcam

https://google.github.io/mediapipe/solutions/selfie_segmentation.html

https://google.github.io/mediapipe/getting_started/python.html

https://google.github.io/mediapipe/solutions/pose#static_image_mode

```bash
$ python3 -m venv mp_env && source mp_env/bin/activate
$ pip install mediapipe
$ python app.py
```

