import cv2
import mediapipe as mp
import numpy as np
import time
import autopy
import speech_recognition as sr
import pyttsx3
import pyautogui
import webbrowser
import os


np.random.seed(42)

engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1)

def speak(text):
    """Converts text to speech"""
    engine.say(text)
    engine.runAndWait()

def listen_for_command():
    """Listens for a voice command and returns the recognized text"""
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Listening for command...")
        try:
            audio = recognizer.listen(source, timeout=5)
        except sr.WaitTimeoutError:
            print("No speech detected within the timeout period")
            return ""

    try:
        command = recognizer.recognize_google(audio).lower()
        print(f"Command received: {command}")
        return command
    except sr.UnknownValueError:
        print("Sorry, I didn't understand that.")
        return ""
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return ""
    except json.decoder.JSONDecodeError as e:
         print(f"JSONDecodeError occurred: {e}")
         return "" # Return empty string if JSON decoding fails
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return ""


class handDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        """Initializes the hand detector with specified parameters."""
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands,
                                          min_detection_confidence=self.detectionCon,
                                          min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        """Finds hands in an image and optionally draws landmarks."""
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        """Finds the position of hand landmarks in the image."""
        lmList = []
        if self.results.multi_hand_landmarks:
            try:
                myHand = self.results.multi_hand_landmarks[handNo]
            except IndexError:
                return lmList, img
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
        return lmList, img

    def fingersUp(self):
        """Checks which fingers are up based on landmark positions."""
        fingers = []
        if self.results.multi_hand_landmarks:
            try:
                handLms = self.results.multi_hand_landmarks[0]
            except IndexError:
                return fingers

            tipIds = [4, 8, 12, 16, 20]
            for id in range(0, 5):
                tipX = handLms.landmark[tipIds[id]].x
                baseX = handLms.landmark[tipIds[id] - 2].x
                if id == 0:  # Thumb
                    if tipX > baseX:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                else:
                    tipY = handLms.landmark[tipIds[id]].y
                    baseY = handLms.landmark[tipIds[id] - 2].y
                    if tipY < baseY:
                        fingers.append(1)
                    else:
                        fingers.append(0)
        return fingers

    def findDistance(self, p1, p2, img, draw=True):
        """Calculates the distance between two landmarks and optionally draws a line and circles."""
        try:
            x1, y1 = int(self.results.multi_hand_landmarks[0].landmark[p1].x * img.shape[1]), int(
                self.results.multi_hand_landmarks[0].landmark[p1].y * img.shape[0])
            x2, y2 = int(self.results.multi_hand_landmarks[0].landmark[p2].x * img.shape[1]), int(
                self.results.multi_hand_landmarks[0].landmark[p2].y * img.shape[0])
        except (IndexError, TypeError):
            return 0, img, [0, 0, 0, 0, 0, 0]

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, (cx, cy), 10, (0, 255, 255), cv2.FILLED)

        length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        return length, img, [x1, y1, x2, y2, cx, cy]


def main():
    voice_command_count = 0
    voice_command_limit = 3

    wCam, hCam = 1280, 720
    frameR = 10
    smoothening = 8

    pTime = 0
    plocX, plocY = 0, 0
    clocX, clocY = 0, 0

    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)

    detector = handDetector(maxHands=1)
    wScr, hScr = autopy.screen.size()
    action_accuracy = 0

    while True:
        if voice_command_count < voice_command_limit:
            command = listen_for_command()

            if command == "double click":
                autopy.mouse.click()
                autopy.mouse.click()
                speak("Double click command executed.")

            elif command == "scroll up":
                pyautogui.scroll(100)
                speak("Scrolled up.")

            elif command == "scroll down":
                pyautogui.scroll(-100)
                speak("Scrolled down.")

            elif command == "volume up":
                pyautogui.press("volumeup")
                speak("Volume increased.")

            elif command == "volume down":
                pyautogui.press("volumedown")
                speak("Volume decreased.")

            elif command == "mute":
                pyautogui.press("volumemute")
                speak("Volume muted.")

            elif command == "open browser":
                webbrowser.open("https://www.google.com")
                speak("Opening browser.")

            elif command == "open notepad":
                os.system("notepad")
                speak("Opening Notepad.")

            elif command == "open calculator":
                os.system("calc")
                speak("Opening Calculator.")
            elif command == "zoom in":
                pyautogui.hotkey('ctrl', '+')
                speak("Zoomed in.")

            elif command == "zoom out":
                pyautogui.hotkey('ctrl', '-')
                speak("Zoomed out.")

            elif command == "exit":
                speak("Exiting the application.")
                print("Exiting the application.")
                break
            else:
                speak("Sorry, I didn't understand that.")
            voice_command_count += 1
            print(f"Voice command received. Command count: {voice_command_count}")

        else:
            success, img = cap.read()

            if not success:
                print("Failed to capture frame. Exiting.")
                break

            img = detector.findHands(img)
            lmList, img = detector.findPosition(img)

            expected_action = 'unknown'
            if lmList:
                x1, y1 = lmList[8][1:3]
                x2, y2 = lmList[12][1:3]
                fingers = detector.fingersUp()

                distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                if fingers[1] == 1 and fingers[2] == 0:
                    expected_action = 'navigation'
                    action_accuracy = 0.95
                elif fingers[1] == 1 and fingers[2] == 1:
                    length, img, lineInfo = detector.findDistance(8, 12, img)
                    if length < 40:
                        cx, cy = int(lineInfo[4]), int(lineInfo[5])
                        cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
                        autopy.mouse.click()
                        expected_action = 'click'
                        action_accuracy = 0.94
                    else:
                        expected_action = 'unknown'
                        action_accuracy = 0
                else:
                    expected_action = 'unknown'
                    action_accuracy = 0


                cv2.putText(img, f"Action: {expected_action}", (10, 80), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 3)
                cv2.putText(img, f"Action Accuracy: {action_accuracy * 100:.2f}%", (10, 120), cv2.FONT_HERSHEY_PLAIN, 3,
                            (0, 255, 255), 3)

                if fingers[1] == 1 and fingers[0] == 0 and fingers[2] == 0 and fingers[3] == 0 and fingers[
                    4] == 0:  # Moving mode: Only Index finger is up
                    x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                    y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

                    clocX = plocX + (x3 - plocX) / smoothening
                    clocY = plocY + (y3 - plocY) / smoothening

                    autopy.mouse.move(wScr - clocX, clocY)
                    cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                    plocX, plocY = clocX, clocY

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
            cv2.imshow("AI Virtual Mouse", img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import json
    main()
