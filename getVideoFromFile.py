
import cv2

def play_video(video_path):
    """Play a video file with pause, seek, and info display."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video info: {width}x{height}, {fps:.2f} FPS, {total_frames} frames, {total_frames/fps:.2f} seconds")
    print("Controls: [q] quit, [space] pause, [f] forward 10s, [b] back 10s")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read frame.")
            break

        # Show current time
        current_pos = cap.get(cv2.CAP_PROP_POS_MSEC)
        minutes = int(current_pos // 60000)
        seconds = int((current_pos % 60000) // 1000)
        cv2.putText(frame, f'Time: {minutes:02d}:{seconds:02d}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Video Player', frame)
        key = cv2.waitKey(25) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):  # Pause
            print("Paused. Press any key to continue.")
            cv2.waitKey(0)
        elif key == ord('f'):  # Forward 10s
            current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame + int(fps * 10))
        elif key == ord('b'):  # Back 10s
            current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current_frame - int(fps * 10)))

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Change the path as needed
    play_video('assets/videos/video1.mp4')