import shutil
import os

def move_files():
    path = 'IMU_Data/val'
    destination = "IMU_Data/train"

    files = [file for file in os.listdir(path)]

    for file in files:
        old_path = os.path.join("IMU_Data/val", file)
        new_path = os.path.join(destination, file)
        shutil.move(src=old_path, dst=new_path)

move_files()
