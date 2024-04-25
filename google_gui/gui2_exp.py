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
        self.root.title("User Needs Experiment")

        # bg_color = "white"
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{screen_width}x{screen_height}")
        self.root.config()
        self.root.state("zoomed")
        self.root.resizable(False, True)
        self.style = ttk.Style()
        # self.style.configure("Custom.TFrame", background=bg_color)
        # self.style.configure("Custom.TButton", background=bg_color, font=("Helvetica", 20))
        # self.style.configure("Custom.TTitle", background=bg_color, font=("Helvetica", 30), anchor=tk.CENTER)
        # self.style.configure("Custom.TLabel", background=bg_color, font=("Helvetica", 20))

        # # Variables
        # self.timer_length = 60
        # self.timer_seconds = 0
        # self.timer_running = False
        # self.display = None

        # Screens
        self.instruction()
        # self.part1_instruction()
        # self.part1()
        # self.part2()
        # self.end_screen()

        # Show the initial screen
        self.show_start_screen()


    def instruction(self):
        self.start_frame = ttk.Frame(self.root, padding="50", width=1400)

        # title = ttk.Label(self.start_frame, text="Welcome to the Experiment!", font=("Arial", 25), justify=tk.CENTER)
        # title.grid(row=0, column=0, pady=10)

        label = ttk.Label(self.start_frame, text="You will now be given the video analysis results from an AI-based observer tool."+
                          " The results are identified problem points, and when you click on the problem points, you will receive further details about them."+
                          "\n\nFor the next 15 minutes, please use these results and your own observations to add more user needs to your list. Identify as many needs"+ 
                          " as possible and try to go beyond the obvious needs.\n", font=("Arial", 25), wraplength=1400)
        label.grid(row=0, column=0, pady=10)

        start_button = customtkinter.CTkButton(self.start_frame, text="Start", command=self.root.destroy, corner_radius=0, font=("Arial", 25))
        start_button.grid(row=1, column=0, pady=5)

    # def part1_instruction(self):
    #     self.screen_1_frame = ttk.Frame(self.root, padding="50", width=1400)
    #     label = ttk.Label(self.screen_1_frame, text="For the next 15 minutes, you will be asked to observe a given video and identify as many needs of the person in the video as possible."+
    #                       " We encourage you to try to go beyond the obvious needs. \n\n15 minutes will begin when you click Next.", font=("Arial", 20), wraplength=1400)
    #     label.grid(row=0, column=0, pady=10)

    #     # next_button = customtkinter.CTkButton(self.screen_1_frame, text="Next", command=self.show_screen_2, corner_radius=0)

    #     next_button = customtkinter.CTkButton(self.screen_1_frame, text="Next", command=self.root.destroy, corner_radius=0, font=("Arial", 20)) #command=self.show_screen_2
    #     next_button.grid(row=1, column=0, pady=10)

    # def part1(self):
    #     self.screen_2_frame = ttk.Frame(self.root, padding="50", width=1400)

    #     self.countdown_var = tk.StringVar()
    #     # print(self.countdown_var)
    #     countdown_var_label = ttk.Label(self.screen_2_frame, textvariable=self.countdown_var, font=("Arial", 18), wraplength=1400)
    #     countdown_var_label.grid(row=0, column=0, columnspan=3, pady=10)

    #     self.start_countdown(self.timer_length, self.show_screen_3)  # 15 minutes countdown

    #     # BOOKER, DISPLAY VIDEO HERE!!!!!!!!!!!!!!!!!!!!
    #     gui_dataframe_output = remake_dicts_from_csv('gui_peaks_dataframe.csv')
    #     peaks_dataframe = remake_dicts_from_csv('peaks_dataframe.csv')
    #     reba_data = remake_dicts_from_csv('reba_data.csv')
    #     video_file_path = 'scan_video1_with_masks.avi'
    #     widg = VideoWidget(self.root,gui_dataframe_output,peaks_dataframe,reba_data,timer_callback=self.show_screen_3,video_file_path=video_file_path)
    #     widg.run()
    #     textOption = {
    #         "Event 1": "Left Knee in dangerous position",
    #         "Event 2": "Right Ankle in dangerous position",
    #         "Event 3": "Left Elbow in dangerous position"
    #     }

    #     self.add_button(textOption, self.screen_2_frame)
    #     self.configure_rows_columns()


    # def part2(self):
    #     self.screen_3_frame = ttk.Frame(self.root, padding="50", width=1400)

    #     self.countdown_var_2 = tk.StringVar()
    #     countdown_var_label_2 = ttk.Label(self.screen_3_frame, textvariable=self.countdown_var_2, font=("Arial", 18), wraplength=1400)
    #     countdown_var_label_2.grid(row=0, column=0, columnspan=3, pady=10)

    #     self.start_countdown(2*self.timer_length, self.show_end_screen)  # 15 minutes countdown

    #     # BOOKER, DISPLAY VIDEO HERE!!!!!!!!!!!!!!!!!!!!
        
    #     # video = ttk.Label(self.screen_3_frame, text="VIDEO GOES HERE", font=("Helvetica", 120), background="red", borderwidth=5, relief="raised", padding="0.4i",)
    #     # video.grid(row=1, column=0, columnspan=2, rowspan=4, pady=10)

    #     textOption = {
    #         "Event 1": "Left Knee in dangerous position",
    #         "Event 2": "Right Ankle in dangerous position",
    #         "Event 3": "Left Elbow in dangerous position"
    #     }

    #     self.add_button(textOption, self.screen_3_frame)
    #     self.configure_rows_columns()

    # def end_screen(self):
    #     self.end_frame = ttk.Frame(self.root, padding="50", width=1400)
    #     label = ttk.Label(self.end_frame, text="End of Experiment", font=("Arial", 18), wraplength=1400)
    #     label.grid(row=0, column=0, pady=10)

    #     # Video can be added here

    #     restart_button = ttk.Button(self.end_frame, text="Restart", command=threading.Thread(target=self.show_start_screen).start, font=("Arial", 18))
    #     restart_button.grid(row=1, column=0, pady=10)

    # def start_countdown(self, seconds, callback):
    #     self.timer_seconds = seconds
    #     self.timer_running = True

    #     def countdown():
    #         while self.timer_seconds > 0 and self.timer_running:
    #             self.countdown_var.set("Time Remaining: " + self.format_time(self.timer_seconds))
    #             self.countdown_var_2.set("Time Remaining: " + self.format_time(self.timer_seconds))
    #             time.sleep(1)
    #             self.timer_seconds -= 1

    #         if self.timer_running:
    #             callback()

    #     threading.Thread(target=countdown).start()

    # def format_time(self, seconds):
    #     minutes, seconds = divmod(seconds, 60)
    #     return f"{minutes:02}:{seconds:02}"

    def show_start_screen(self):
        self.hide_all_screens()
        self.start_frame.grid()

    # def show_screen_1(self):
    #     self.hide_all_screens()
    #     self.screen_1_frame.grid()

    # def show_screen_2(self):
    #     self.hide_all_screens()
    #     self.screen_2_frame.grid()

    # def show_screen_3(self):
    #     self.hide_all_screens()
    #     self.screen_3_frame.grid()

    # def show_end_screen(self):
    #     self.hide_all_screens()
    #     self.end_frame.grid()

    def hide_all_screens(self):
        self.start_frame.grid_forget()
        # self.screen_1_frame.grid_forget()
        # self.screen_2_frame.grid_forget()
        # self.screen_3_frame.grid_forget()
        # self.end_frame.grid_forget()

    def text_display(self, button_data, button_number, frame):
        if self.display != None:
            self.display.destroy()
        self.display = ttk.Label(frame, text=button_data[button_number], font=("Arial", 18))
        self.display.grid(row=5, column=1)

    def add_button(self, data, frame):
        button = tk.IntVar()
        button_counter = 1
        for entry in data:
            button_text = str(f"Event {button_counter}")
            # TO VERIFY INPUT
            # print(f"This is button {button_counter} with text {button_text} which should correspond to {data[button_text]}")
            new_button = tk.Radiobutton(frame, bg="light blue", text=button_text, indicatoron=False, value=button_counter, variable=button, command=(lambda button_label=button_text: self.text_display(data, button_label, frame)), font=("Helvetica", 20))
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

    # Prep everything
    app = ExperimentGUI(root)
    
    #####VIDEO SHOULD BE ANNOTATED
    # video_file_path = 'videos/scan_video1_with_masks_annotated.avi'
    # video_file_path = 'videos/opening_door_annotated.avi'
    video_file_path = 'videos/setting_up_desk_annotated.mov'

    gui_dataframe_output = remake_dicts_from_csv('data/'+video_file_path[7:-14]+'_gui_peaks_dataframe.csv')
    peaks_dataframe = remake_dicts_from_csv('data/'+video_file_path[7:-14]+'_peaks_dataframe.csv')
    reba_data = remake_dicts_from_csv('data/'+video_file_path[7:-14]+'_reba_data.csv')
    object_data = remake_dicts_from_csv('data/'+video_file_path[7:-14]+'_object_data.csv')
    count=2
    condition=2
    entry_list=[]

    # Run the windows back to back
    root.mainloop()

    video_window = VideoWidget(video_file_path, gui_dataframe_output, peaks_dataframe, reba_data, count, condition, object_data=object_data)
    video_window.run()

    # root = customtkinter.CTk()
    # root.mainloop()