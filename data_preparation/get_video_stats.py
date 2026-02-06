import os
from pathlib import Path
from pathlib import Path
import csv
import time
import subprocess
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dirct =  os.getcwd()                  #r'/home/soumyad/mlproj/sdd
total_labels = 0
total_lbl_per_vid = 0
start = time.time()
new_plots = []
scene = []
#vid_data_dir = ''
for subdir, dirs, files in os.walk(dirct):
    '''print('subdir', subdir)
    print('dirs:', dirs)
    print('files:', files)'''
    no_of_vid = 0
    df_headers = ['Videos', 'Classes', 'Frequency']
    vid_plots = []
    
    for file in files:
        if file.endswith('.txt'):
            f1 = os.path.join(subdir, file)
            ann_path = Path(f1)
            if ann_path.name != "annotations.txt":
                continue

            ann_parts = ann_path.parts
            if "annotations" not in ann_parts:
                continue
            ann_idx = ann_parts.index("annotations")
            if len(ann_parts) <= ann_idx + 2:
                continue

            scene_name = ann_parts[ann_idx + 1]
            video_name = ann_parts[ann_idx + 2]
            if scene_name not in scene:
                scene.append(scene_name)
                new_plots = []
            print('Scenes covered: ', scene)    
            print('Folder for annotattions of {}_{}: '.format(scene_name, video_name), subdir)
            print('Annotations: ', str(ann_path))
            print('\n')
            annotations_dir = ann_path.parents[2]
            vid_data_dir = str(annotations_dir)
            print('vid_data_dir: ', vid_data_dir)
            filenm = os.path.join(vid_data_dir, 'vid_data.csv')
            
            df_labels = pd.read_csv(f1, names=['id', 'left', 'top', 'right', 'bottom', 'frames','a','b','c','classes'], sep=' ')
            selected_labels= pd.DataFrame(df_labels, columns = ['classes'])
            pedestrians = 0
            bikers = 0
            skaters = 0
            cart = 0
            car = 0
            bus = 0
            res = []
            
            for x, row in selected_labels.iterrows():
                classes = row['classes']
                
                if classes =='Pedestrian':
                    pedestrians+=1
                elif classes == 'Biker':
                    bikers+=1
                elif classes == 'Skater':
                    skaters +=1
                elif classes == 'Cart':
                    cart +=1
                elif classes == 'Car':
                    car +=1
                elif classes == 'Bus':
                    bus +=1
                if classes not in res:
                    res.append(classes)
            print('Pedestrians: ', pedestrians)
            print('\n')
            print('Bikers: ', bikers)
            print('\n')
            print('Skaters: ', skaters)
            print('Classes in the video: ', res)
            total_lbl_per_vid = (pedestrians
                            +bikers
                            +skaters
                            +cart
                            +car
                            +bus)
            headers = ['Class', 'Freqs']
            filename = os.path.join(subdir, "freq.csv")
            freqs=[]
            frequency = 0
            
            for _class in res:
                if _class =='Pedestrian':
                    frequency=pedestrians
                elif _class == 'Biker':
                    frequency=bikers
                elif _class == 'Skater':
                    frequency=skaters
                elif _class == 'Cart':
                    frequency=cart
                elif _class == 'Car':
                    frequency=car
                elif _class == 'Bus':
                    frequency=bus
                
                freqs.append({'Class': _class,
                             'Freqs': frequency})
                vid_plots.append({'Videos': video_name,
                                  'Classes': _class,
                                  'Frequency': frequency})
            #new_plots = vid_plots
            new_plots.extend(vid_plots)
            print('vid plot: ', vid_plots)
            print(len(vid_plots)) 
            print('new_plot len', len(new_plots))
            #print('class and frequncy: ', classes, frequency)
            
            with open(filename, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames= headers)
                writer.writeheader()
                writer.writerows(freqs)
            raw_data = pd.read_csv(filename)   
            
            ax = sns.barplot(x='Class', y='Freqs', data=raw_data)
            #ax.set(ylim=(0,total_lbl_per_vid))
            fig = ax.get_figure()
            _fig = os.path.join(subdir, 'data.png')
            fig.savefig(_fig)
            print('Plot saved in: ', _fig)
            print('\n')
            
            vid_data_dir = str(annotations_dir)
            print('vid_data_dir: ', vid_data_dir)
            filenm = os.path.join(vid_data_dir, 'vid_data.csv')
            print('video csv file: ', filenm)
            if new_plots:
                with open(filenm, 'w') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames= df_headers)
                    writer.writeheader()
                    writer.writerows(new_plots)
                raw_data1 = pd.read_csv(filenm)
                if not raw_data1.empty:
                    ax1 = sns.catplot(x='Videos', y='Frequency', hue='Classes', data=raw_data1, kind='bar')
                    ax1.fig.set_figwidth(10)
                    _fig1 = os.path.join(vid_data_dir, 'vid_data.png')
                    ax1.savefig(_fig1)
                    print('Final plot saved in: ', _fig1)
                    print('\n')
        
        total_labels =  total_labels+total_lbl_per_vid

time.sleep(1)
end = time.time()
print('Total annotations: ', total_labels)
print(f"Time taken: {(end-start)/60} minutes")
