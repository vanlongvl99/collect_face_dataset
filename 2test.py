import cv2

cap = cv2.VideoCapture(2)

while True:
    ret, frame =  cap.read()
    # print(frame)

    # frame_resize = cv2.flip(frame, 0)
    cv2.imshow('frame_resize', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()