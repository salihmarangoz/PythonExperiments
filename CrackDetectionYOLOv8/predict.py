from ultralytics import YOLO
import cv2
import glob

model = YOLO('runs/segment/train/weights/best.pt')
img_dir = "crack.v2i.yolov8/test/images/*"

cv_window_str = "Press anything to continue, Q to quit"
cv2.namedWindow(cv_window_str, cv2.WINDOW_NORMAL) 

i =0
for name in glob.glob(img_dir):
    print(name)
    img = cv2.imread(name)
    results = model(img)  # list of Results objects
    annotated_frame = results[0].plot()
    cv2.imshow(cv_window_str, annotated_frame)
    #cv2.imwrite("outputs/" + str(i) + ".png", annotated_frame)
    if cv2.waitKey(0) & 0xFF == ord("q"):
        break
    i+=1
    
cv2.destroyAllWindows()