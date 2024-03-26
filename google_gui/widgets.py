
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkVideoPlayer import TkinterVideo
from datetime import timedelta
import customtkinter
from reba_video_analyzer import *
import math
import time
import threading
from utils import remake_dicts_from_csv

# print(peaks_dataframe)
class VideoWidget():
    def __init__(self,gui_dataframe_output,peaks_dataframe,reba_data,video_file_path=None):
        # self.root = tk.Tk()
        #style
        self.root = customtkinter.CTk()
        bg_color = "light grey"
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{screen_width}x{screen_height}")
        self.root.config(bg=bg_color)
        self.root.state("zoomed")
        self.root.resizable(False, True)
        self.style = ttk.Style()
        self.style.configure("Custom.TFrame", background=bg_color)
        self.style.configure("Custom.TButton", background=bg_color, font=("Helvetica", 30))
        self.style.configure("Custom.TLabel", background=bg_color, foreground="black", font=("Helvetica", 30))

        # Variables
        self.timer_length = 200
        self.timer_seconds = 0
        self.timer_running = False
        self.display = None

        self.gui_dataframe_output = gui_dataframe_output
        self.peaks_dataframe = peaks_dataframe
        self.reba_data=reba_data
        print(self.gui_dataframe_output)
        # print(self.peaks_dataframe['frame_of_max_peak'])
        # print(self.peaks_dataframe['frame_c_score'])
        self.peak_frames = []
        self.peak_reba_cs = []
        self.total_frames = len(reba_data['frame'])
        event_data = {}
        for i,peak_frame in enumerate(gui_dataframe_output['peaks_i']):
            button_data = {}
            peak_val = gui_dataframe_output['peak_val'][i]
            problem_body_part = self.find_problem_joint(peak_frame)
            button_data['peak_val'] = peak_val
            button_data['peak_frame'] = peak_frame
            button_data['perc_in_vid'] = (peak_frame+1)/self.total_frames
            button_data['bad_joint'] = None
            button_data['text'] = f"Reba score: {peak_val}. {problem_body_part[:-6]} in dangerous position"
            event_data[f'Event {i+1}'] = button_data


        # textOption = {
        #     "Event 1": {'text':"Left Knee in dangerous position",
        #     "Event 2": "Right Ankle in dangerous position",
        #     "Event 3": "Left Elbow in dangerous position"
        # }

        
        # self.configure_rows_columns()

        self.root.title("Need Finder Tool Interface")
        # self.root.geometry("800x700+290+10")
        self.countdown_var = tk.StringVar()
        self.countdown_var_2 = tk.StringVar()
        countdown_var_label_2 = ttk.Label(self.root, textvariable=self.countdown_var_2, style="Custom.TLabel")
        countdown_var_label_2.pack()
        # countdown_var_label_2.grid(row=0, column=0, columnspan=3, pady=10)

        self.start_countdown(self.timer_length) 

        self.load_btn = tk.Button(self.root, text="Load", command=self.load_video)
        self.load_btn.pack()

        self.vid_player = TkinterVideo(scaled=True, master=self.root)
        self.vid_player.pack(expand=True, fill="both")

        self.play_pause_btn = tk.Button(self.root, text="Play", command=self.play_pause)
        self.play_pause_btn.pack()

        self.add_buttons(event_data, self.root)

        self.skip_plus_5sec = tk.Button(self.root, text="Skip -5 sec", command=lambda: self.skip(-5))
        self.skip_plus_5sec.pack(side="left")

        self.start_time = tk.Label(self.root, text=str(timedelta(seconds=0)))
        self.start_time.pack(side="left")

        self.progress_value = tk.DoubleVar(self.root)

        self.progress_slider = tk.Scale(self.root, variable=self.progress_value, from_=0, to=0, orient="horizontal",command=self.seek)
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
        self.vid_player.play()
    def find_problem_joint(self,peak_frame):
        high_score = 0
        high_body_part = None
        for key in self.reba_data:
            if key[-5:] == 'score' and key[0] not in ['a','b','c']:
                print(key,self.reba_data[key][peak_frame],high_score)
                if self.reba_data[key][peak_frame] > high_score:
                    high_score = self.reba_data[key][peak_frame]
                    high_body_part = key

        return high_body_part

    def format_time(self, seconds):
        minutes, seconds = divmod(seconds, 60)
        return f"{minutes:02}:{seconds:02}"
    def text_display_and_seek(self, button_data, button_number, frame):
        if self.display != None:
            self.display.destroy()
        perc = button_data[button_number]['perc_in_vid']
        peak_frame = button_data[button_number]['peak_frame']
        self.seek_from_perc(peak_frame,perc)
        self.display = ttk.Label(frame, text=button_data[button_number]['text'], style="Custom.TLabel")
        self.display.pack()
        # self.display.grid(row=5, column=1)
    def add_buttons(self, data, frame):
        button = tk.IntVar()
        button_counter = 1
        for entry in data:
            button_text = str(f"Event {button_counter}")
            # TO VERIFY INPUT
            # print(f"This is button {button_counter} with text {button_text} which should correspond to {data[button_text]}")
            new_button = tk.Radiobutton(frame, bg="light blue", text=button_text, indicatoron=False, value=button_counter, variable=button, command=(lambda button_label=button_text: self.text_display_and_seek(data, button_label, frame)), font=("Helvetica", 30))
            new_button.pack(side="left")
            # new_button.grid(row=(button_counter + 5), column=0, pady=2)
            button_counter += 1
        return None
    def seek_from_perc(self,frame,perc):
        # percentage = (desired_frame+1)/self.total_frames
        # rounded_frame = int(self.total_frames*percentage)
        seconds = frame/self.frame_rate
        # rounded_frame = int(self.total_frames*percentage)
        val =int(math.floor(seconds))
        print(seconds,val)
        self.vid_player.seek(val)
        self.progress_value.set(val)
        diff = seconds - val
       
        
        # self.play_pause()
    def start_countdown(self, seconds):
        self.timer_seconds = seconds
        self.timer_running = True

        def countdown():
            while self.timer_seconds > 0 and self.timer_running:
                self.countdown_var.set("Time Remaining: " + self.format_time(self.timer_seconds))
                self.countdown_var_2.set("Time Remaining: " + self.format_time(self.timer_seconds))
                time.sleep(1)
                self.timer_seconds -= 1

            if self.timer_running:
                self.root.withdraw()
                

        threading.Thread(target=countdown).start()
    def update_duration(self,event):
        """ updates the duration after finding the duration """
        self.duration = self.vid_player.video_info()["duration"]
        self.end_time["text"] = str(timedelta(seconds=self.duration))
        self.progress_slider["to"] = self.duration


    def update_scale(self,event):
        """ updates the scale value """
        print(self.vid_player.video_info()['framerate'])
        self.frame_rate = int(self.vid_player.video_info()['framerate'])
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
            # self.play_pause_btn["text"] = "Play"
            # self.progress_value.set(0)
            # print(self.vid_player.video_info()['framerate'])
    


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

    new_video = False

    video_file_path = 'booker.mp4'
    if new_video:
        gui_dataframe_output,peaks_dataframe,reba_data = reba_video_analyzer(video_file_path=video_file_path,
                        test=test,
                        frontview=True,
                        show_plots=show_plots,
                        camera_frames_per_second = 30,
                        create_csv_from_data = create_csv_from_data)
    else: 
        gui_dataframe_output = remake_dicts_from_csv('gui_peaks_dataframe.csv')
        peaks_dataframe = remake_dicts_from_csv('peaks_dataframe.csv')
        reba_data = remake_dicts_from_csv('reba_data.csv')

    annotated_video_file_path = 'scan_video1_annotated.avi'
    widg = VideoWidget(gui_dataframe_output,peaks_dataframe,reba_data,video_file_path)
    widg.run()
    # root.mainloop()
 