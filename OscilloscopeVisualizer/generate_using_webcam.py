import cv2 as cv
import numpy as np
import soundcard as sc

window_name = "Audio Webcam"
cv.namedWindow(window_name)

vc = cv.VideoCapture(0)
if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

default_speaker = sc.default_speaker()
with default_speaker.player(samplerate=44100) as player:
    while rval:
        img = cv.cvtColor(frame, cv.COLOR_RGBA2GRAY, 0);
        img = cv.resize(img, (0,0), fx=0.33, fy=0.33) 
        img = cv.Canny(image=img, threshold1=50, threshold2=100)
        contours, _ = cv.findContours(img, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        contours = np.vstack(contours).squeeze()
        data = contours.copy()
        data = np.float32(data)
        data[:, 0] = data[:, 0] / img.shape[1] - 0.5
        data[:, 1] = -data[:, 1] / img.shape[0] + 0.5

        print("Playing data with shape:", data.shape)
        player.play(data)

        cv.imshow(window_name, img)
        rval, frame = vc.read()
        key = cv.waitKey(20)
        if key == 27: # exit on ESC
            break

vc.release()
cv.destroyWindow(window_name)