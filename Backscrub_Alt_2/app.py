import cv2
import numpy as np
import mediapipe as mp
mp_selfie_segmentation = mp.solutions.selfie_segmentation

class WebcamSegmentation:
    def __init__(self):
        self.prev_image_u_gray = None
        self.mask = None

    def compute_mask(self, model, image_u):
        rgb_image = cv2.cvtColor(image_u, cv2.COLOR_BGR2RGB)
        rgb_image.flags.writeable = False # To improve performance mark the image as not writeable to pass by reference
        results = model.process(rgb_image)
        return results.segmentation_mask

    def filter_mask(self, image_f, image_u, mask):
        #return self.filter_mask_grabcut_(image_f, image_u, mask)
        #return self.filter_mask_bilateral_(image_f, image_u, mask)
        return self.filter_mask_optical_flow_(image_f, image_u, mask)

    def filter_mask_bilateral_(self, image_f, image_u, mask):
        mask = cv2.ximgproc.jointBilateralFilter(image_f, mask, -1, 40, 10)
        return mask

    def filter_mask_grabcut_(self, image_f, image_u, mask):
        mask = cv2.GaussianBlur(mask,(31,31),0)
        mask_ = np.where(mask>0.95, 2, 3).astype(np.uint8)
        image_ = (image_f*255).astype(np.uint8)

        bgdModel = np.zeros((1,65), np.float64) 
        fgdModel = np.zeros((1,65), np.float64) 
        mask2, bgdModel, fgdModel = cv2.grabCut(image_, mask_, None, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_MASK) 

        mask2 = np.where((mask2==2)|(mask2==0),255, 0).astype('uint8') 
        mask2 = cv2.GaussianBlur(mask2,(11,11),0) / 255.0
        return mask2

    def filter_mask_optical_flow_(self, image_f, image_u, mask):
        if self.prev_image_u_gray is None:
            self.prev_image_u_gray = cv2.cvtColor(image_u, cv2.COLOR_BGR2GRAY)
            self.prev_mask = mask.copy()
            return mask

        def warp_flow(img, flow):
            h, w = flow.shape[:2]
            flow = -flow
            flow[:,:,0] += np.arange(w)
            flow[:,:,1] += np.arange(h)[:,np.newaxis]
            res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
            return res

        def visualize(image, flow):
            hsv = np.zeros_like(image)
            hsv[..., 1] = 255
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang*180/np.pi/2
            hsv[..., 2] = mag*5 # cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            cv2.imshow('frame2', bgr)

        image_u_gray = cv2.cvtColor(image_u, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(self.prev_image_u_gray, image_u_gray, None, 0.5, 3, 40, 3, 5, 1.2, 1)
        visualize(image_u, flow)
        pred_mask = warp_flow(self.prev_mask, flow)
        new_mask = mask * 0.8 + pred_mask * 0.2
        cv2.imshow('mag', new_mask)

        self.prev_image_u_gray = image_u_gray.copy()
        self.prev_mask = new_mask.copy()
        return new_mask

    def add_channels(self, mask, n=3):
        return np.stack((mask,) * n, axis=-1)

    def alpha_composite(self, fg, bg, alpha):
        return fg * alpha + bg * (1-alpha)

    def resize(self, img, scale):
        width = int(img.shape[1] * scale)
        height = int(img.shape[0] * scale)
        dim = (width, height)
        return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    def start_loop(self):
        # For webcam input:
        BG_COLOR = (0, 255, 0)
        cap = cv2.VideoCapture(0)
        with mp_selfie_segmentation.SelfieSegmentation(model_selection=0) as model:
            bg_image = None

            while cap.isOpened():
                success, image_u = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue
                image_f = image_u.astype(np.float32)/255.0

                if bg_image is None:
                    bg_image = np.zeros(image_u.shape, dtype=np.float32)
                    bg_image[:] = BG_COLOR
                    bg_image_f = bg_image/255.0

                # Flip the image horizontally for a later selfie-view display
                #image = cv2.flip(image, 1)

                scale = 0.75
                image_u_mini = self.resize(image_u, scale)
                mask_mini = self.compute_mask(model, image_u_mini)
                mask = self.resize(mask_mini, 1/scale)

                mask_filtered = self.filter_mask(image_f, image_u, mask)
                mask_filtered = self.add_channels(mask_filtered)
                output_image = self.alpha_composite(image_f, bg_image_f, mask_filtered)

                mask = self.add_channels(mask)
                output_image2 = self.alpha_composite(image_f, bg_image_f, mask)
                cv2.imshow('no filtering', output_image2)


                cv2.imshow('MediaPipe Selfie Segmentation', output_image)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
        cap.release()


ws = WebcamSegmentation()
ws.start_loop()