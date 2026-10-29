import cv2

cap = cv2.VideoCapture("data/sample.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)

frame_idx = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break

    cv2.putText(frame, f"frame {frame_idx}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    if frame_idx == 400:
        cv2.rectangle(frame, (485, 415), (605, 615), (0, 255, 0), 2)

    cv2.imshow("video", frame)
    delay = 0 if frame_idx == 400 else int(1000 / fps)
    if cv2.waitKey(delay) == ord("q"):
        break

    frame_idx += 1