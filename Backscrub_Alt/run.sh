sudo rmmod v4l2loopback
sudo modprobe v4l2loopback devices=2 max_buffers=2 exclusive_caps=1,1 card_label="VirtualCam1","VirtualCam2" video_nr=10,9
python portrait_video.py | ffmpeg -y -f rawvideo -video_size 640x480 -pixel_format bgr24 -i - -f v4l2 -vf format=yuv420p /dev/video10