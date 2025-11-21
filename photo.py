import cv2

# opening camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # reading the video from camera
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Display the resulting frame
    cv2.imshow('Camera Feed', frame)
    key = cv2.waitKey(1) 

    if key == 27:
        break

    # taking photo when pressing space
    if key == 32:
        name = input("write name and press enter to save photo: ")
        filename = f"contacts_data/{name}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Photo saved as {filename}")

cap.release()
cv2.destroyAllWindows()

