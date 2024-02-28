from tkinter import *
from tkVideoPlayer import TkinterVideo

root = Tk()

videoplayer = TkinterVideo(master=root, scaled=True)
videoplayer.load(r"scan_video1.avi")
videoplayer.pack(expand=True, fill="both")

videoplayer.play() # play the video

root.mainloop()
# def callback():
#    Label(win, text="Hello World!", font=('Century 20 bold')).pack(pady=4)
# #Create a Label and a Button widget
# btn=Button(win, text="Press Enter", command= callback)
# btn.pack(ipadx=10)
# win.bind('<Return>',lambda event:callback())
# win.mainloop()