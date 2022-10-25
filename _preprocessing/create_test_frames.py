import av
import matplotlib.pyplot as plt
import os

import numpy as np
from skimage import io
import pathlib

video_names = [f for f in os.listdir(f'../test_videos') if os.path.isfile(f'../test_videos/{f}')]

for video_name in video_names:
    container = av.open(f'../test_videos/{video_name}')
    num = 0
    stream = container.streams.video[0]

    get_frame = 1
    i = 0
    for frame in container.decode(stream):
        img = frame.to_image()
        img = np.asarray(img)
        video_dir = video_name.split('.')[0]
        i += 1
        if i % get_frame == 0:#np.round(frame.time,2) % 0.33 == 0:
            try:
                io.imsave(f'../test_videos/{video_dir}/{video_dir}_frame_{num}.png', img[250: 250 + 1024, 250:250 + 1024])
            except:
                p = pathlib.Path(f'../test_videos/{video_dir}')#os.path.mkdir(f'../test_videos/{video_dir}')
                p.mkdir(parents=True, exist_ok=True)
                io.imsave(f'../test_videos/{video_dir}/{video_dir}_frame_{num}.png', img[250: 250 + 1024, 250:250 + 1024])
            num += 1
            print(f'Saving {num}th frame from {video_name}')