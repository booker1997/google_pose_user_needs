import tkinter as tk
from tkinter import ttk
# from tkinter import *
import customtkinter
import time
import threading

# customtkinter.set_appearance_mode("dark")

class ExperimentGUI:

    def __init__(self, root):
        self.root = root
        self.root.title("Experiment GUI")

        # screen_width = self.root.winfo_screenwidth()
        # screen_height = self.root.winfo_screenheight()
        # window_width = int(screen_width * 0.8)
        # window_height = int(screen_height * 0.8)
        # self.root.geometry(f"{window_width}x{window_height}")
        # self.root.geometry("1500x1000")

        # Variables
        self.timer_seconds = 0
        self.timer_running = False

        # Screens
        self.instruction()
        self.part1_instruction()
        self.part1()
        self.create_screen_3()
        self.create_end_screen()

        # Show the initial screen
        self.show_start_screen()

    def instruction(self):
        self.start_frame = ttk.Frame(self.root, padding="20")
        # self.start_frame.grid(row=0, column=0, sticky="nsew")

        label = ttk.Label(self.start_frame, text="Welcome to the Experiment! \n\n\nIn this study, you will be given a video of a person performing an activity." +
                          "The video will be about 5-minutes long. \n\nYour task is to observe this video and identify the needs of the person in the video." + 
                          " Identify as many needs as possible and try to go \nbeyond the obvious needs. \n\nYou will have total 30 minutes for the task. After" +
                          " the first 15 minutes, you will get a 2-minute break, followed by another 15 minutes. \n\n\nIf you have any questions regarding the study," +
                          " you may ask the researcher now. If you are ready to begin the task, please click Start.", font=("Helvetica", 12))
        label.grid(row=0, column=0, pady=10)

        start_button = customtkinter.CTkButton(self.start_frame, text="Start", command=self.show_screen_1)
        start_button.grid(row=1, column=0, pady=10)

    def part1_instruction(self):
        self.screen_1_frame = ttk.Frame(self.root, padding="20")
        label = ttk.Label(self.screen_1_frame, text="For the next 15 minutes, you will be asked to observe a given video and identify as many needs of the person in the video as possible."+
                          " We encourage you \nto try to go beyond the obvious needs. \n\n15 minutes will begin when you click Next.",font=("Helvetica", 12))
        label.grid(row=0, column=0, pady=10)

        next_button = customtkinter.CTkButton(self.screen_1_frame, text="Next", command=self.show_screen_2)
        next_button.grid(row=1, column=0, pady=10)

    def part1(self):
        self.screen_2_frame = ttk.Frame(self.root, padding="20")

        countdown_label = ttk.Label(self.screen_2_frame, text="Time remaining: ")
        countdown_label.grid(row=0, column=0, pady=10)

        self.countdown_var = tk.StringVar()
        countdown_var_label = ttk.Label(self.screen_2_frame, textvariable=self.countdown_var)
        countdown_var_label.grid(row=1, column=0, pady=10)

        self.start_countdown(900, self.show_screen_3)  # 15 minutes countdown

        label = ttk.Label(self.screen_2_frame, text="You now have 15 minutes to observe the video below and identify as many needs of the person in the video as possible."+
                          " We encourage you to try to go beyond the obvious needs. \n\n\nDISPLAY VIDEO HERE.", font=("Helvetica", 12))
        label.grid(row=2, column=0, pady=10)

        # BOOKER, DISPLAY VIDEO HERE!!!!!!!!!!!!!!!!!!!! It will go to row=3, column=0
        
        add_button = customtkinter.CTkButton(self.screen_2_frame, text="Add Entry", command=self.add_entry)
        add_button.grid(row=3, column=1, pady=10)

        input_label = tk.Label(self.screen_2_frame, text="Need #1:")
        input_label.grid(row=4, column=1, pady=10)
        user_input = ttk.Entry(self.screen_2_frame, width=50, font=("Arial", 12), justify='right')
        user_input.grid(row=4, column=2, pady=10)

    def create_screen_3(self):
        self.screen_3_frame = ttk.Frame(self.root, padding="20")
        label = ttk.Label(self.screen_3_frame, text="Screen 3")
        label.grid(row=0, column=0, pady=10)

        countdown_label = ttk.Label(self.screen_3_frame, text="Time remaining: ")
        countdown_label.grid(row=1, column=0, pady=10)

        self.countdown_var_2 = tk.StringVar()
        countdown_var_label_2 = ttk.Label(self.screen_3_frame, textvariable=self.countdown_var_2)
        countdown_var_label_2.grid(row=2, column=0, pady=10)

        self.start_countdown(120, self.show_end_screen)  # 2 minutes countdown

    def create_end_screen(self):
        self.end_frame = ttk.Frame(self.root, padding="20")
        label = ttk.Label(self.end_frame, text="End of Experiment")
        label.grid(row=0, column=0, pady=10)

        # Video can be added here

        # restart_button = ttk.Button(self.end_frame, text="Restart", command=self.restart_experiment)
        # restart_button.grid(row=1, column=0, pady=10)

    def start_countdown(self, seconds, callback):
        self.timer_seconds = seconds
        self.timer_running = True

        def countdown():
            while self.timer_seconds > 0 and self.timer_running:
                self.countdown_var.set(self.format_time(self.timer_seconds))
                self.countdown_var_2.set(self.format_time(self.timer_seconds))
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
        self.screen_2_frame.grid_forget()
        self.screen_3_frame.grid_forget()
        self.end_frame.grid_forget()
    
    def add_entry(self):
        row = len(entry_list)+5
        input_label = tk.Label(self.screen_2_frame, text="Need #"+str(row-3)+":")
        input_label.grid(row=row, column=1, pady=10)
        new_entry = tk.Entry(self.screen_2_frame, width=50, font=("Arial", 12), justify='right')
        new_entry.grid(row=row, column=2, pady=10)
        entry_list.append(new_entry)
    

if __name__ == "__main__":
    root = customtkinter.CTk()
    app = ExperimentGUI(root)
    entry_list=[]
    root.mainloop()