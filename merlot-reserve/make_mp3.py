import subprocess
import os

wav_files = os.listdir("/data/raw/acoustic/wav")

for wav in wav_files:
    input_name = os.path.join("/data/raw/acoustic/wav", wav)
    output_name = os.path.join("/data/raw/acoustic_mp3", wav[:-3] + "mp3")
    subprocess.call('ffmpeg -i {video} -ar 22050 -ac 1 {out_name}'.format(video=input_name, out_name=output_name), shell=True)
