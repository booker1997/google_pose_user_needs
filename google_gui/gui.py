import tkinter as tk
from tkinter import ttk
# from tkinter import *
import customtkinter
import time
import threading
from widgets import *

# customtkinter.set_appearance_mode("dark")

class ExperimentGUI:

    def __init__(self, root):

        # test = False
        # front_view = True
        # create_csv_from_data = True
        # show_plots = False
        # gui_dataframe_output,peaks_dataframe,total_dataframe = reba_video_analyzer(video_file_path=video_file_path,
        #                     test=test,
        #                     frontview=True,
        #                     show_plots=show_plots,
        #                     camera_frames_per_second = 30,
        #                     create_csv_from_data = create_csv_from_data)
        # self.video_widg = VideoWidget(gui_dataframe_output,peaks_dataframe,total_dataframe,video_file_path)

        self.root = root
        self.root.title("Experiment GUI")


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

        # Screens
        self.instruction()
        self.part1_instruction()
        # self.part1()
        # self.part2()
        # self.end_screen()

        # Show the initial screen
        self.show_start_screen()

    def instruction(self):
        self.start_frame = ttk.Frame(self.root, padding="20", style="Custom.TFrame")

        label = ttk.Label(self.start_frame, text="Welcome to the Experiment! \n\n\nIn this study, you will be given a video of a person performing an activity." +
                          "The video will be about 5-minutes long. \n\nYour task is to observe this video and identify the needs of the person in the video." +
                          " Identify as many needs as possible and try to go \nbeyond the obvious needs. \n\nYou will have total 30 minutes for the task. After" +
                          " the first 15 minutes, you will get a 2-minute break, followed by another 15 minutes. \n\n\nIf you have any questions regarding the study," +
                          " you may ask the researcher now. If you are ready to begin the task, please click Start.", style="Custom.TLabel")
        label.grid(row=0, column=0, pady=10, sticky="nesw")

        start_button = customtkinter.CTkButton(self.start_frame, text="Start", command=self.show_screen_1, corner_radius=0)
        start_button.grid(row=1, column=0, pady=10)

    def part1_instruction(self):
        self.screen_1_frame = ttk.Frame(self.root, padding="20", style="Custom.TFrame")
        label = ttk.Label(self.screen_1_frame, text="For the next 15 minutes, you will be asked to observe a given video and identify as many needs of the person in the video as possible."+
                          " We encourage you \nto try to go beyond the obvious needs. \n\n15 minutes will begin when you click Next.", style="Custom.TLabel")
        label.grid(row=0, column=0, pady=10)

        next_button = customtkinter.CTkButton(self.screen_1_frame, text="Next", command=self.root.destroy, corner_radius=0) #command=self.show_screen_2
        next_button.grid(row=1, column=0, pady=10)

    def part1(self):
        self.screen_2_frame = ttk.Frame(self.root, padding="20", style="Custom.TFrame")

        self.countdown_var = tk.StringVar()
        countdown_var_label = ttk.Label(self.screen_2_frame, textvariable=self.countdown_var, style="Custom.TLabel")
        countdown_var_label.grid(row=0, column=0, columnspan=3, pady=10)

        self.start_countdown(self.timer_length, self.show_screen_3)  # 15 minutes countdown

        # BOOKER, DISPLAY VIDEO HERE!!!!!!!!!!!!!!!!!!!!
        gui_dataframe_output = remake_dicts_from_csv('gui_peaks_dataframe.csv')
        peaks_dataframe = remake_dicts_from_csv('peaks_dataframe.csv')
        reba_data = remake_dicts_from_csv('reba_data.csv')
        video_file_path = 'scan_video1_with_masks.avi'
        widg = VideoWidget(self.root,gui_dataframe_output,peaks_dataframe,reba_data,timer_callback=self.show_screen_3,video_file_path=video_file_path)
        widg.run()
        textOption = {
            "Event 1": "Left Knee in dangerous position",
            "Event 2": "Right Ankle in dangerous position",
            "Event 3": "Left Elbow in dangerous position"
        }

        self.add_button(textOption, self.screen_2_frame)
        self.configure_rows_columns()


    def part2(self):
        self.screen_3_frame = ttk.Frame(self.root, padding="20", style="Custom.TFrame")

        self.countdown_var_2 = tk.StringVar()
        countdown_var_label_2 = ttk.Label(self.screen_3_frame, textvariable=self.countdown_var_2, style="Custom.TLabel")
        countdown_var_label_2.grid(row=0, column=0, columnspan=3, pady=10)

        self.start_countdown(2*self.timer_length, self.show_end_screen)  # 15 minutes countdown

        # BOOKER, DISPLAY VIDEO HERE!!!!!!!!!!!!!!!!!!!!
        
        # video = ttk.Label(self.screen_3_frame, text="VIDEO GOES HERE", font=("Helvetica", 120), background="red", borderwidth=5, relief="raised", padding="0.4i",)
        # video.grid(row=1, column=0, columnspan=2, rowspan=4, pady=10)

        textOption = {
            "Event 1": "Left Knee in dangerous position",
            "Event 2": "Right Ankle in dangerous position",
            "Event 3": "Left Elbow in dangerous position"
        }

        self.add_button(textOption, self.screen_3_frame)
        self.configure_rows_columns()

    def end_screen(self):
        self.end_frame = ttk.Frame(self.root, padding="20", style="Custom.TFrame")
        label = ttk.Label(self.end_frame, text="End of Experiment", style="Custom.TLabel")
        label.grid(row=0, column=0, pady=10)

        # Video can be added here

        restart_button = ttk.Button(self.end_frame, text="Restart", command=threading.Thread(target=self.show_start_screen).start, style="Custom.TButton")
        restart_button.grid(row=1, column=0, pady=10)

    def start_countdown(self, seconds, callback):
        self.timer_seconds = seconds
        self.timer_running = True

        def countdown():
            while self.timer_seconds > 0 and self.timer_running:
                self.countdown_var.set("Time Remaining: " + self.format_time(self.timer_seconds))
                self.countdown_var_2.set("Time Remaining: " + self.format_time(self.timer_seconds))
                time.sleep(1)
                self.timer_seconds -= 1

            if self.timer_running:
                callback()

        threading.Thread(target=countdown).start()

    def format_time(self, seconds):
        minutes, seconds = divmod(seconds, 60)
        return f"{minutes:02}:{seconds:02}"

    def show_start_screen(self):
        self.hide_all_screens()
        self.start_frame.grid()

    def show_screen_1(self):
        self.hide_all_screens()
        self.screen_1_frame.grid()

    def show_screen_2(self):
        self.hide_all_screens()
        self.screen_2_frame.grid()

    def show_screen_3(self):
        self.hide_all_screens()
        self.screen_3_frame.grid()

    def show_end_screen(self):
        self.hide_all_screens()
        self.end_frame.grid()

    def hide_all_screens(self):
        self.start_frame.grid_forget()
        self.screen_1_frame.grid_forget()
        # self.screen_2_frame.grid_forget()
        # self.screen_3_frame.grid_forget()
        # self.end_frame.grid_forget()

    def text_display(self, button_data, button_number, frame):
        if self.display != None:
            self.display.destroy()
        self.display = ttk.Label(frame, text=button_data[button_number], style="Custom.TLabel")
        self.display.grid(row=5, column=1)

    def add_button(self, data, frame):
        button = tk.IntVar()
        button_counter = 1
        for entry in data:
            button_text = str(f"Event {button_counter}")
            # TO VERIFY INPUT
            # print(f"This is button {button_counter} with text {button_text} which should correspond to {data[button_text]}")
            new_button = tk.Radiobutton(frame, bg="light blue", text=button_text, indicatoron=False, value=button_counter, variable=button, command=(lambda button_label=button_text: self.text_display(data, button_label, frame)), font=("Helvetica", 30))
            new_button.grid(row=(button_counter + 5), column=0, pady=2)
            button_counter += 1
        return None

    def count_rows_columns(self):
        max_row = 0
        max_column = 0

        # Iterate over all widgets in the grid
        for widget in self.root.grid_slaves():
            row = int(widget.grid_info()['row'])
            column = int(widget.grid_info()['column'])
            max_row = max(max_row, row)
            max_column = max(max_column, column)

        return max_row + 1, max_column + 1

    def configure_rows_columns(self):
        rows, columns = self.count_rows_columns()
        for i in range(rows):
            self.root.grid_rowconfigure(i, weight=1)
        for j in range(columns):
            self.root.grid_columnconfigure(j, weight=1)



if __name__ == "__main__":
    root = customtkinter.CTk()

    #prep everything
    app = ExperimentGUI(root)
    
    video_file_path = 'videos/scan_video1_with_masks.avi'
    gui_dataframe_output = remake_dicts_from_csv('data/'+video_file_path[7:-4]+'_gui_peaks_dataframe.csv')
    peaks_dataframe = remake_dicts_from_csv('data/'+video_file_path[7:-4]+'_peaks_dataframe.csv')
    reba_data = remake_dicts_from_csv('data/'+video_file_path[7:-4]+'_reba_data.csv')
    object_data = remake_dicts_from_csv('data/'+video_file_path[7:-4]+'_object_data.csv')
    entry_list=[]

    # run the windows back to back
    root.mainloop()
    video_window = VideoWidget(video_file_path,gui_dataframe_output,peaks_dataframe,reba_data,object_data=object_data)
    video_window.run()
    
