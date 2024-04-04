
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
    def __init__(self,gui_dataframe_output,peaks_dataframe,reba_data,timer_callback=None,video_file_path=None):
        # self.root = tk.Tk()
        #style
        self.root = customtkinter.CTk()
        self.timer_callback = timer_callback
        self.button_label = 'Problem '
        self.object_data = remake_dicts_from_csv('object_data.csv')
        self.object_labels = ['table top','table leg']
        bg_color = "light grey"
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        font_size = 60
        self.root.geometry(f"{screen_width}x{screen_height}")
        self.root.config(bg=bg_color)
        self.root.state("zoomed")
        self.root.resizable(False, True)
        self.style = ttk.Style()
        self.style.configure("Custom.TFrame", background=bg_color)
        self.style.configure("Custom.TButton", background=bg_color, foreground="black", font=("Helvetica", font_size))
        self.style.configure("Custom.TLabel", background=bg_color, foreground="black", font=("Helvetica", font_size))

        # Variables
        self.timer_length = 200
        self.timer_seconds = 0
        self.timer_running = False
        self.display = None

        self.gui_dataframe_output = gui_dataframe_output
        self.peaks_dataframe = peaks_dataframe
        self.reba_data=reba_data
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
            objects_interacting = []
            if i< len(self.object_data['start_frame']):
                button_data['start_frame'] = self.object_data['start_frame'][i]
                button_data['end_frame'] = self.object_data['end_frame'][i]
                if self.object_data['inter_w_hand'][i] != None:
                    for item in self.object_data['inter_w_hand'][i]:
                        objects_interacting.append(item)
                if self.object_data['inter_w_foot'][i] != None:
                    for item in self.object_data['inter_w_foot'][i]:
                        if item not in objects_interacting:
                            objects_interacting.append(item)
                

            button_data['perc_in_vid'] = (peak_frame+1)/self.total_frames
            button_data['bad_joint'] = None
            button_data['text'] = f"Reba score: {peak_val}. {problem_body_part[:-6]} in dangerous position. \nLook at interactions with the following objects: {objects_interacting} "
            event_data[self.button_label + f'{i+1}'] = button_data


        # textOption = {
        #     "Event 1": {'text':"Left Knee in dangerous position",
        #     "Event 2": "Right Ankle in dangerous position",
        #     "Event 3": "Left Elbow in dangerous position"
        # }


        # self.configure_rows_columns()

        self.root.title("Need Finder Tool Interface")
        # self.root.geometry("800x700+290+10")

        # frame_list = ["timer_frame", "video_frame", "control_frame", "button_frame", "output_frame"]
        # for frame in frame_list:
        #     exec(f"self.{frame} = ttk.Frame(self.root, style='Custom.TFrame')")
        #     exec(f"self.frame.pack(expand=True, fill='both')")
        frame_height = screen_height / 20
        frame_width = screen_width
        self.timer_frame = ttk.Frame(self.root, style="Custom.TFrame", height=frame_height, width=frame_width)
        self.video_frame = ttk.Frame(self.root, style="Custom.TFrame", height=frame_height*15, width=frame_width)
        self.play_frame = ttk.Frame(self.root, style="Custom.TFrame", height=frame_height, width=frame_width)
        self.control_frame = ttk.Frame(self.root, style="Custom.TFrame", height=frame_height, width=frame_width)
        self.button_frame = ttk.Frame(self.root, style="Custom.TFrame", height=frame_height, width=frame_width)
        self.output_frame = ttk.Frame(self.root, style="Custom.TFrame", height=frame_height, width=frame_width)
        self.all_frames = [self.timer_frame,self.video_frame,self.play_frame,self.control_frame,self.button_frame,self.output_frame]
        # Frame borders to help layout
        # self.timer_frame.config(borderwidth=20, relief="ridge")
        # self.video_frame.config(borderwidth=20, relief="ridge")
        # self.play_frame.config(borderwidth=20, relief="ridge")
        # self.control_frame.config(borderwidth=20, relief="ridge")
        # self.button_frame.config(borderwidth=20, relief="ridge")
        # self.output_frame.config(borderwidth=20, relief="ridge")

        self.timer_frame.pack()
        self.video_frame.pack(fill="both", expand=True)
        self.play_frame.pack()
        self.control_frame.pack()
        self.button_frame.pack()
        self.output_frame.pack()

        self.countdown_var = tk.StringVar()
        # self.countdown_var_2 = tk.StringVar()
        countdown_var_label = ttk.Label(self.timer_frame, textvariable=self.countdown_var, style="Custom.TLabel")
        countdown_var_label.pack()
        # countdown_var_label_2.grid(row=0, column=0, columnspan=3, pady=10)

        # self.start_countdown(self.timer_length,self.hide_all_screens)

        # self.load_btn = tk.Button(self.root, text="Load", command=self.load_video)
        # self.load_btn.pack()

        self.vid_player = TkinterVideo(scaled=True, master=self.video_frame)
        self.vid_player.pack(fill="both", expand=True)

        self.play_pause_btn = tk.Button(self.play_frame, text="Play", command=self.play_pause, font=("Helvetica", 50))
        self.play_pause_btn.pack()

        # self.skip_plus_5sec = tk.Button(self.control_frame, text="Skip -5 sec", command=lambda: self.skip(-5), font=("Helvetica", 30))
        # self.skip_plus_5sec.pack(side="left", expand=True, fill="both")

        self.start_time = tk.Label(self.control_frame, text=str(timedelta(seconds=0)), font=("Helvetica", 30))
        self.start_time.pack(side="left", expand=True, fill="both")

        self.progress_value = tk.DoubleVar(self.root)

        self.progress_slider = tk.Scale(self.control_frame, variable=self.progress_value, from_=0, to=0, orient="horizontal",command=self.seek, font=("Helvetica", 50), width=50, length=screen_width)
        # progress_slider.bind("<ButtonRelease-1>", seek)
        self.progress_slider.pack(side="left", fill="x", expand=True)

        self.end_time = tk.Label(self.control_frame, text=str(timedelta(seconds=0)), font=("Helvetica", 30))
        self.end_time.pack(side="left", expand=True, fill="both")
        if video_file_path!=None:
            self.video_file_path = video_file_path
            self.load_video()
        else:
            self.video_file_path= None

        self.vid_player.bind("<<Duration>>", self.update_duration)
        self.vid_player.bind("<<SecondChanged>>", self.update_scale)
        self.vid_player.bind("<<Ended>>", self.video_ended )

        # self.skip_plus_5sec = tk.Button(self.control_frame, text="Skip +5 sec", command=lambda: self.skip(5), font=("Helvetica", 30))
        # self.skip_plus_5sec.pack(side="left", expand=True, fill="both")

        self.add_buttons(event_data, self.button_frame)


    def find_problem_joint(self,peak_frame):
        high_score = 0
        high_body_part = None
        for key in self.reba_data:
            if key[-5:] == 'score' and key[0] not in ['a','b','c']:
                # print(key,self.reba_data[key][peak_frame],high_score)
                if self.reba_data[key][peak_frame] > high_score:
                    high_score = self.reba_data[key][peak_frame]
                    high_body_part = key

        return high_body_part

    def format_time(self, seconds):
        minutes, seconds = divmod(seconds, 60)
        return f"{minutes:02}:{seconds:02}"

    def text_display_and_seek(self, button_data, button_number):
        if self.display != None:
            self.display.destroy()
        perc = button_data[button_number]['perc_in_vid']
        peak_frame = button_data[button_number]['peak_frame']
        start_frame = button_data[button_number]['start_frame']
        self.seek_from_perc(start_frame)
        self.display = ttk.Label(self.output_frame, text=button_data[button_number]['text'], style="Custom.TLabel",font=("Helvetica", 30))
        self.display.pack()
        # self.display.grid(row=5, column=1)

    def add_buttons(self, data, frame):
        button = tk.IntVar()
        button_counter = 1
        # print(data)
        for entry in data:
            button_text = str(self.button_label + f"{button_counter}")
            # TO VERIFY INPUT
            # print(f"This is button {button_counter} with text {button_text} which should correspond to {data[button_text]}")
            new_button = tk.Radiobutton(frame, bg="red", text=button_text, indicatoron=False, value=button_counter, variable=button, command=(lambda button_label=button_text: self.text_display_and_seek(data, button_label)), font=("Helvetica", 30))
            new_button.pack(side="left", anchor="center")
            # new_button.grid(row=(button_counter + 5), column=0, pady=2)
            button_counter += 1
        return None

    def seek_from_perc(self,frame):
        # percentage = (desired_frame+1)/self.total_frames
        # rounded_frame = int(self.total_frames*percentage)
        seconds = frame/self.frame_rate
        # rounded_frame = int(self.total_frames*percentage)
        val =int(math.floor(seconds))
        self.vid_player.seek(val)
        self.progress_value.set(val)
        diff = seconds - val


        # self.play_pause()

    def start_countdown(self, seconds, callback):
        self.timer_seconds = seconds
        self.timer_running = True

        def countdown():
            if self.timer_seconds > 0 and self.timer_running:
                self.root.after(1000, countdown)  # Schedule the next countdown in 1 second
                self.timer_seconds -= 1
            elif self.timer_running:
                callback()

        def update_gui():
            if self.timer_running:
                self.countdown_var.set("Time Remaining: " + self.format_time(self.timer_seconds))
                self.root.after(1000, update_gui)  # Schedule the next GUI update in 1 second

        threading.Thread(target=countdown).start()  # Start the countdown in a separate thread
        update_gui()

    def update_duration(self,event):
        """ updates the duration after finding the duration """
        self.duration = self.vid_player.video_info()["duration"]
        self.end_time["text"] = str(timedelta(seconds=self.duration))
        self.progress_slider["to"] = self.duration


    def update_scale(self,event):
        """ updates the scale value """
        # print(self.vid_player.video_info()['framerate'])
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

            # self.progress_slider.config(to=0, from_=0)



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
    def hide_all_screens(self):
        for frame in self.all_frames:
            frame.destroy()
        self.timer_callback()
    def run(self):
        # Start the countdown in a separate thread
        threading.Thread(target=self.start_countdown, args=(self.timer_length, self.hide_all_screens)).start()
        # Run the Tkinter main loop in the main thread
        self.root.mainloop()
        # self.root.mainloop()


if __name__ == "__main__":
    test = False
    front_view = True
    create_csv_from_data = True
    show_plots = False

    new_video = False

    # video_file_path = 'booker.mp4'
    video_file_path = 'scan_video1_with_masks.avi'
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
    
    widg = VideoWidget(gui_dataframe_output,peaks_dataframe,reba_data,video_file_path=video_file_path)
    widg.run()
    # root.mainloop()
