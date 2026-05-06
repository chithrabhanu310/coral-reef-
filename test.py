import cv2
import numpy as np

video_path = r"C:/Users/MRIDULA/Downloads/17 (1).MP4"

cap = cv2.VideoCapture(video_path)

bleach_percentages = []
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    frame = cv2.resize(frame, (640, 480))
    h, w, _ = frame.shape
    roi = frame[int(h*0.3):int(h*0.9), int(w*0.1):int(w*0.9)]  # Adjust if needed

    #  Convert to HSV , HSV ranges , Healthy coral (colored)
   
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_healthy = np.array([5, 80, 80])
    upper_healthy = np.array([25, 255, 255])

    # Bleached coral 
    lower_bleach = np.array([0, 20, 180])
    upper_bleach = np.array([180, 80, 255])

    # Create color masks
    mask_healthy = cv2.inRange(hsv, lower_healthy, upper_healthy)
    mask_bleach = cv2.inRange(hsv, lower_bleach, upper_bleach)

    # Texture filter using edges 
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)

    # Keep bleached regions that also have texture (edges)
    mask_bleach = cv2.bitwise_and(mask_bleach, edges)

    # noice removal 
    kernel = np.ones((5, 5), np.uint8)
    mask_bleach = cv2.morphologyEx(mask_bleach, cv2.MORPH_OPEN, kernel)
    mask_healthy = cv2.morphologyEx(mask_healthy, cv2.MORPH_OPEN, kernel)

    #  Count pixels 
    bleached_pixels = cv2.countNonZero(mask_bleach)
    healthy_pixels = cv2.countNonZero(mask_healthy)
    total_coral_pixels = bleached_pixels + healthy_pixels

    if total_coral_pixels > 0:
        bleaching_percentage = (bleached_pixels / total_coral_pixels) * 100
    else:
        bleaching_percentage = 0

    bleach_percentages.append(bleaching_percentage)

    #   results 
    display_frame = roi.copy()
    cv2.putText(display_frame, f"Bleaching: {bleaching_percentage:.2f}%", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.imshow("Coral Region", display_frame)
    cv2.imshow("Bleached Mask", mask_bleach)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Print average bleaching percentage
avg_bleaching = np.mean(bleach_percentages)
print(f"\nAverage Coral Bleaching Percentage: {avg_bleaching:.2f}%")