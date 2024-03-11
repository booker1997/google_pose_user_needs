from tkinter import *
import tkinter as tk
from tkinter import filedialog
from tkVideoPlayer import TkinterVideo
from datetime import timedelta
from reba_video_analyzer import *


# print(peaks_dataframe)
class VideoWidget():
    def __init__(self,gui_dataframe_output,peaks_dataframe,total_dataframe,video_file_path=None):
        self.root = tk.Tk()
        self.gui_dataframe_output = gui_dataframe_output
        self.peaks_dataframe = peaks_dataframe
        self.total_dataframe=total_dataframe

        # print(self.peaks_dataframe['frame_of_max_peak'])
        # print(self.peaks_dataframe['frame_c_score'])
        self.peak_frames = []
        self.peak_reba_cs = []
        self.total_frames = len(total_dataframe['frame'])
        for i,frame in enumerate(self.gui_dataframe_output['peaks_i']):
            pass
        #     self.play_pause_btn = tk.Button(self.root, text="Play", command=self.play_pause)
        #     self.play_pause_btn.pack()
        # print(self.peak_frames,self.peak_reba_cs)
        self.perc_btn = tk.Button(self.root, text="go to perc", command=self.seek_from_frame)
        self.perc_btn.pack()

        self.root.title("Need Finder Tool Interface")
        self.root.geometry("800x700+290+10")

        

        self.load_btn = tk.Button(self.root, text="Load", command=self.load_video)
        self.load_btn.pack()

        self.vid_player = TkinterVideo(scaled=True, master=self.root)
        self.vid_player.pack(expand=True, fill="both")

        self.play_pause_btn = tk.Button(self.root, text="Play", command=self.play_pause)
        self.play_pause_btn.pack()

        self.skip_plus_5sec = tk.Button(self.root, text="Skip -5 sec", command=lambda: self.skip(-5))
        self.skip_plus_5sec.pack(side="left")

        self.start_time = tk.Label(self.root, text=str(timedelta(seconds=0)))
        self.start_time.pack(side="left")

        self.progress_value = tk.DoubleVar(self.root)

        self.progress_slider = tk.Scale(self.root, variable=self.progress_value, from_=0, to=0, orient="horizontal", command=self.seek)
        # progress_slider.bind("<ButtonRelease-1>", seek)
        self.progress_slider.pack(side="left", fill="x", expand=True)

        self.end_time = tk.Label(self.root, text=str(timedelta(seconds=0)))
        self.end_time.pack(side="left")
        if video_file_path!=None:
            self.video_file_path = video_file_path
            self.load_video()
        else:
            self.video_file_path= None

        self.vid_player.bind("<<Duration>>", self.update_duration)
        self.vid_player.bind("<<SecondChanged>>", self.update_scale)
        self.vid_player.bind("<<Ended>>", self.video_ended )

        self.skip_plus_5sec = tk.Button(self.root, text="Skip +5 sec", command=lambda: self.skip(5))
        self.skip_plus_5sec.pack(side="left")

        

    def update_duration(self,event):
        """ updates the duration after finding the duration """
        duration = self.vid_player.video_info()["duration"]
        self.end_time["text"] = str(timedelta(seconds=duration))
        self.progress_slider["to"] = duration


    def update_scale(self,event):
        """ updates the scale value """
        print(self.vid_player.video_info()['framerate'])
        self.progress_value.set(self.vid_player.current_duration())


    def load_video(self):
        """ loads the video """
        if self.video_file_path!=None:
            file_path = self.video_file_path
        else:
            file_path = filedialog.askopenfilename()

        if file_path:
            self.vid_player.load(file_path)

            self.progress_slider.config(to=0, from_=0)
            self.play_pause_btn["text"] = "Play"
            self.progress_value.set(0)
            print(self.vid_player.video_info()['framerate'])
    def seek_from_frame(self,desired_frame=110):
        # percentage = (desired_frame+1)/self.total_frames
        # rounded_frame = int(self.total_frames*percentage)
        seconds = (desired_frame+1)/29
        # rounded_frame = int(self.total_frames*percentage)
        self.vid_player.seek(seconds)


    def seek(self,value):
        """ used to seek a specific timeframe """
        self.vid_player.seek(int(value))


    def skip(self,value: int):
        """ skip seconds """
        self.vid_player.seek(int(self.progress_slider.get())+value)
        self.progress_value.set(self.progress_slider.get() + value)


    def play_pause(self):
        """ pauses and plays """
        
        if self.vid_player.is_paused():
            self.vid_player.play()
            self.play_pause_btn["text"] = "Pause"

        else:
            self.vid_player.pause()
            self.play_pause_btn["text"] = "Play"
    

    def video_ended(self,event):
        """ handle video ended """
        self.progress_slider.set(self.progress_slider["to"])
        self.play_pause_btn["text"] = "Play"
        self.progress_slider.set(0)
    def run(self):
        self.root.mainloop()
        

if __name__ == "__main__":
    test = False
    front_view = True
    create_csv_from_data = True
    show_plots = False
    video_file_path = 'booker.mp4'
    gui_dataframe_output,peaks_dataframe,total_dataframe = reba_video_analyzer(video_file_path=video_file_path,
                        test=test,
                        frontview=True,
                        show_plots=show_plots,
                        camera_frames_per_second = 30,
                        create_csv_from_data = create_csv_from_data)
    annotated_video_file_path = 'booker_annotated.avi'
    widg = VideoWidget(gui_dataframe_output,peaks_dataframe,total_dataframe,video_file_path)
    widg.run()
    # root.mainloop()
 