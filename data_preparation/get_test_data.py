import os
from pathlib import Path
import cv2
import numpy as np
import csv
import time
import subprocess
import shutil
import pandas as pd


def read_image(path):
    buf = np.fromfile(path, dtype=np.uint8)
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)

cwd = os.getcwd()
dirct = cwd if os.path.exists(os.path.join(cwd, "annotations")) else os.path.split(cwd)[0]
scene_filter = os.environ.get("SCENE")
sdd = os.path.join(dirct, "stanford_dd_yolo")
img_dir = os.path.join(sdd, "images")
label_dir = os.path.join(sdd, "labels")
_list_dir = [sdd, img_dir, label_dir]

test_csv = os.path.join(sdd, "test.csv")
test_csv_cols = []

for dirs in _list_dir:
    try:
        os.mkdir(dirs)
    except OSError as error:
        print(error)
    
total_labels = 0
start = time.time()
no_of_vid = 0
start = time.time()
#vid_data_dir = ''
for subdir, dirs, files in os.walk(dirct):
    '''print('subdir', subdir)
    print('dirs:', dirs)
    print('files:', files)'''
    
    for file in files:
        if file.endswith(('.mov', '.mp4')):
            #no_of_vid+=1
            f = os.path.join(subdir, file)
            subdir_path = Path(subdir)
            vid_name = subdir_path.name
            parts = subdir_path.parts
            if "video" not in parts:
                continue
            video_idx = parts.index("video")
            if len(parts) <= video_idx + 2:
                continue
            scene_name = parts[video_idx + 1]
            if scene_filter and scene_name != scene_filter:
                continue
            #if os.path.isfile(f):
            print('\n')
            print('current subdirectory: ', subdir)
            #print('count:', c)
            fr_path = os.path.join(subdir, 'frames')
            try:
                os.mkdir(fr_path)
            except OSError as error:
                print(error)
            
            print('Video file: ',f)
            print('Frame dir path: ',fr_path)
            print('\n')
            
            if not os.listdir(fr_path):
                query = "ffmpeg -i " + f + " -qscale:v 2 -crf 18 " + fr_path + "/" + vid_name + "_%d.jpg"
                print(query)
                response = subprocess.Popen(query, shell=True, stdout=subprocess.PIPE).stdout.read()
                s = str(response).encode('utf-8')
            if not os.listdir(fr_path):
                print(f"No frames found for {vid_name}. Skipping.")
                continue
            print(vid_name+" is Done...")
            print('current subdirectory: ', subdir)
            
            label_path = os.path.join(dirct, 'annotations', scene_name, vid_name, "annotations.txt")
            df_labels = pd.read_csv(label_path, names=['id', 'left', 'top', 'right', 'bottom', 'frames','a','b','c','class'], sep=' ')  #we are not concerned with some columns e.g. a,b,c
            df_labels.sort_values(['frames'], axis=0, ascending=True, inplace=True)
            selected = pd.DataFrame(df_labels, columns = ['left', 'top', 'right', 'bottom', 'frames','class'])
            print(selected)
            
            frame_list = []
            frame = 0
            for x, row in selected.iterrows():
                frame = int(row['frames'])
                       
                if frame%89 ==0:
                    
                    fr_name = vid_name+'_'+str(frame+1)+'.jpg'
                    frame_path = os.path.join(fr_path, fr_name)
                    if not os.path.exists(frame_path):
                        continue
                    img = read_image(frame_path)
                    if img is None:
                        continue
                    frame_txt = fr_name.split('.')[0] +'.txt'
                    labels_per_frame = os.path.join(label_dir, frame_txt)
                    
                    rows = selected.loc[selected['frames'] == frame]
                    each_line=[]
                    if frame not in frame_list:
                        frame_list.append(frame)
                    for i, obj in rows.iterrows():
                        class_lbl = 0
                        left  = int(obj['left'])
                        top   = int(obj['top'])
                        right = int(obj['right'])
                        bottom= int(obj['bottom'])
                        _class= obj['class']
                        if _class =='Pedestrian':
                            class_lbl=0
                        elif _class == 'Biker':
                            class_lbl=1
                        elif _class == 'Skater':
                            class_lbl=2
                        elif _class == 'Cart':
                            class_lbl=3
                        elif _class == 'Car':
                            class_lbl=4
                        elif _class == 'Bus':
                            class_lbl=5
                        total_labels+=1
                        x_norm = (left+((right-left)/2))/img.shape[1]
                        y_norm = (top+((bottom-top)/2))/img.shape[0]
                        width_norm = (right-left)/(img.shape[1])
                        height_norm = (bottom-top)/(img.shape[0])
                        _each_line = [class_lbl, x_norm, y_norm, width_norm, height_norm]
                        each_line.append(_each_line)
                
                        with open(labels_per_frame, 'w') as file1:
                            for data in each_line:
                                for _data in data:
                                    file1.writelines("%s " %_data)
                                file1.writelines('\n')
                        file1.close()
            for _frame in frame_list:
                fr_name1 = vid_name+'_'+str(_frame+1)
                shutil.copy(os.path.join(fr_path, fr_name1+'.jpg'), img_dir)
                print(fr_name1)
                test_csv_cols.append({'image': fr_name1+'.jpg',
                                       'label': fr_name1+'.txt'})
            print('Done')
print(f'total {total_labels} annotations')
header = ['image', 'label']
with open(test_csv, 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=header)
    writer.writeheader()
    writer.writerows(test_csv_cols)
time.sleep(1)
end = time.time()
print(f"Time taken: {(end-start)/60} minutes")
