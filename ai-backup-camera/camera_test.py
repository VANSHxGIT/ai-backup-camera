import cv2
# Open default webcam
cap = cv2.VideoCapture(0)
# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()
while True:
    # Read frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break
    # Show frame
    cv2.imshow("AI Backup Camera", frame)
    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Cleanup
cap.release()
cv2.destroyAllWindows()