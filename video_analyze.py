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


test = True
front_view = True
create_csv_from_data = False
show_plots = True

peaks_dataframe = reba_video_analyzer(video_file_path=None,
                    test=test,
                    frontview=True,
                    show_plots=show_plots,
                    camera_frames_per_second = 30,
                    create_csv_from_data = create_csv_from_data)


