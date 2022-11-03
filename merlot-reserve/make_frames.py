import os
import subprocess

def video_to_frames(data_dir):
    vid_names = os.listdir(data_dir)
    print(len(vid_names))
    frame_dir = "/data/raw/frames"
    for vid in vid_names:
        print("MAKING THIS VIDEO INTO MANY PICTURES (FRAMES): ", vid)
        vid_name = os.path.splitext(vid)[0]
        if not os.path.exists(os.path.dirname(vid_name)):
            os.makedirs(os.path.join(frame_dir, vid_name), exist_ok=True)
            vid_path = os.path.join(data_dir, vid)
            output = os.path.join(frame_dir, vid_name, vid_name+"_%03d.jpg")
            subprocess.call('ffmpeg -i {video} -r 3 -q:v 1 {out_name}'.format(video=vid_path, out_name=output), shell=True)

    print("Finished extracting frames.")

video_to_frames("/data/raw/vision/raw")
