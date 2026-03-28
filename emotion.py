
import cv2
from deepface import DeepFace
import threading

current_emotion = "Detecting..."
current_color   = (200, 200, 200)
analyzing       = False

# Emotion → display color (BGR)
EMOTION_COLORS = {
    "happy":    (0, 220, 100),    # green
    "sad":      (200, 80, 50),    # blue-ish
    "angry":    (0, 0, 220),      # red
    "surprise": (0, 200, 255),    # yellow
    "fear":     (130, 0, 180),    # purple
    "disgust":  (0, 140, 80),     # dark green
    "neutral":  (180, 180, 180),  # gray
}

# Friendly display names
EMOTION_LABELS = {
    "happy":    "😊 Happy",
    "sad":      "😢 Sad",
    "angry":    "😠 Angry",
    "surprise": "😲 Surprised",
    "fear":     "😨 Fearful",
    "disgust":  "🤢 Disgusted",
    "neutral":  "😐 Neutral",
}


def analyze_frame(frame):
    """Run DeepFace emotion analysis in a background thread."""
    global current_emotion, current_color, analyzing
    try:
        result = DeepFace.analyze(
            frame,
            actions=["emotion"],
            enforce_detection=False,   # don't crash if no face found
            silent=True,
        )
        emotion = result[0]["dominant_emotion"].lower()
        current_emotion = EMOTION_LABELS.get(emotion, emotion.capitalize())
        current_color   = EMOTION_COLORS.get(emotion, (255, 255, 255))
    except Exception:
        current_emotion = "No face found"
        current_color   = (100, 100, 100)
    finally:
        analyzing = False


def main():
    global analyzing

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    print("Emotion Detector started.")
    print("First analysis may take a few seconds to load the model.")
    print("Press 'Q' to quit.\n")

    frame_count = 0
    ANALYZE_EVERY = 15    # run DeepFace every N frames (keeps it smooth)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        # Run analysis in background thread every N frames
        frame_count += 1
        if frame_count % ANALYZE_EVERY == 0 and not analyzing:
            analyzing = True
            thread = threading.Thread(
                target=analyze_frame,
                args=(frame.copy(),),
                daemon=True,
            )
            thread.start()

        # ── Dark overlay bar at bottom ───────────────────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 100), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # ── Emotion text ─────────────────────────────────────────────────────
        text      = current_emotion
        font      = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.6
        thickness  = 3

        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = (w - text_w) // 2
        text_y = h - 30

        # Shadow
        cv2.putText(frame, text, (text_x + 2, text_y + 2),
                    font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        # Colored text
        cv2.putText(frame, text, (text_x, text_y),
                    font, font_scale, current_color, thickness, cv2.LINE_AA)

        # ── Top status bar ───────────────────────────────────────────────────
        cv2.rectangle(frame, (0, 0), (w, 38), (30, 30, 30), -1)
        cv2.putText(frame, "Face Emotion Detector", (10, 26),
                    font, 0.7, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(frame, "Press Q to quit", (w - 160, 26),
                    font, 0.55, (140, 140, 140), 1, cv2.LINE_AA)

        cv2.imshow("Emotion Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Emotion Detector stopped.")


if __name__ == "__main__":
    main()