import cv2, sys, time
import numpy as np
import tensorflow as tf
from PIL import Image
from numba import jit
import sys


# Normalize the input image
@jit(nopython=True, nogil=True)
def normalize(img):
    mean = np.array([103.94, 116.78, 123.68], dtype=np.float32)
    val = np.array([0.017, 0.017, 0.017], dtype=np.float32)
    return (img - mean) * val
 
# Alpha blend frame with background
@jit(nopython=True, nogil=True)
def blend(frame, alpha, background):
        #alphargb = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)
        alpha_shape = (frame.shape[0], frame.shape[1], 1)
        return frame * alpha.reshape(alpha_shape) + background * (1-alpha.reshape(alpha_shape))
        #return (frame**(1/2.2) * alpha.reshape(alpha_shape) + background**(1/2.2) * (1-alpha.reshape(alpha_shape)))**2.2

# Initialize tflite-interpreter
interpreter = tf.lite.Interpreter(model_path="portrait_video.tflite", num_threads=1) # Use 'tf.lite' on recent tf versions
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height, width = input_details[0]['shape'][1:3]

# Initialize video capturer
cap = cv2.VideoCapture(0)
#size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) 
size = (640, 480)

background = cv2.imread("background.jpg")
pred_video = None
prev_mask = None


# with Device(device) as out_device:
#     print(dir(BufferType))
#     out_device.set_format(BufferType.VIDEO_OUTPUT, width, height, 'YUYV')
while True:
    # Read the BGR frames 
    ret, frame = cap.read()

    # Resize the image
    image = Image.fromarray(frame)
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    image = np.asarray(image, dtype=np.float32)

    # Normalize the input
    image = normalize(image)

    # Choose prior mask
    if pred_video is None:
        prior = np.zeros((height, width, 1), dtype=np.float32)
    else:
        prior = pred_video

    # Add prior as fourth channel
    image=np.dstack([image,prior])
    prepimg = image[np.newaxis, :, :, :]

    # Invoke interpreter for inference
    interpreter.set_tensor(input_details[0]['index'], prepimg)
    interpreter.invoke()
    outputs = interpreter.get_tensor(output_details[0]['index'])
    outputs = outputs.reshape(height,width,1)

    # Save output to feed subsequent inputs
    
    #outputs = outputs**(1/2)
    #outputs = outputs**(1/1.4)

    # if prev_mask is not None:
    #     prev_mask = outputs*0.8 + prev_mask*0.2
    #     outputs = prev_mask
    # else:
    #     prev_mask = outputs
    #     outputs = prev_mask
    #pred_video = outputs

    #outputs = (outputs - 0.5) * 2 + 0.5
    #outputs = np.clip(outputs, 0.0, 1.0)
    pred_video = outputs


    # Process the output
    outputs = cv2.resize(outputs, size, cv2.INTER_AREA)
    outputs = blend(frame, outputs, background).astype(np.uint8)
    
    # Display the output
    cv2.imshow('Portrait Video',outputs)
    if cv2.waitKey(1)&0xFF == ord('q'):
        break

    #out_device.write(ConvertToYUYV(outputs))
    sys.stdout.buffer.write(outputs.tobytes())

cap.release()
cv2.destroyAllWindows()
