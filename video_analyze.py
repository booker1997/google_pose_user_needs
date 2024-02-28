import random
from utils import *
import cv2
import mediapipe as mp
import numpy as np
import sys
from matplotlib import pyplot as plt
from reba_score_class import RebaScoreMIT
import pandas as pd
from reba_video_analyzer import *


test = False
front_view = True
create_csv_from_data = True
show_plots = True
video_file_path = 'scan_video1.avi'

peaks_dataframe,total_dataframe = reba_video_analyzer(video_file_path=video_file_path,
                    test=test,
                    frontview=True,
                    show_plots=show_plots,
                    camera_frames_per_second = 30,
                    create_csv_from_data = create_csv_from_data)


