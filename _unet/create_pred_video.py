import os
import cv2
import numpy as np

model_img_size = (512, 512)
folder_name = 'balra erek jobb szélső ureter'
test_dir = f'../test_videos/{folder_name}'

images_folder = f'{test_dir}/predictions'
images = [img for img in os.listdir(images_folder)]
images.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
video_name = f'{test_dir}/{folder_name}_prediction_video.avi'

video = cv2.VideoWriter(video_name,0,24,(model_img_size[0], model_img_size[1]))

for image in images:
    video.write(cv2.imdecode(np.fromfile(os.path.join(images_folder, image), dtype=np.uint8), cv2.IMREAD_UNCHANGED))

video.release()