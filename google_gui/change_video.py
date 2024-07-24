import cv2

# Open the video file
cap = cv2.VideoCapture('vacuum_new.MOV')

# Get the frame rate of the input video
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the output video and its FPS
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps_new = 30
out = cv2.VideoWriter('vacuum_new_30.mp4', fourcc, fps_new, (1920, 1080))

# Capture the video frame by frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame
    frame = cv2.resize(frame, (1920, 1080))

    # Write the frame to the output video
    out.write(frame)

# Close the video files
cap.release()
out.release()