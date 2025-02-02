import cv2
import mediapipe as mp
import numpy as np
import time
import pygame

pygame.mixer.init()

piano_progressions_original = {i: pygame.mixer.Sound(f"wavs/progression_{i}.wav") for i in range(1, 21)}
current_piano_chord_index = None
current_piano_channel = None
current_speed_factor = 1.0

def change_playback_speed(sound, speed):
    arr = pygame.sndarray.array(sound)
    n_samples, n_channels = arr.shape
    new_length = int(n_samples / speed)
    new_arr = np.zeros((new_length, n_channels), dtype=arr.dtype)
    for ch in range(n_channels):
        new_arr[:, ch] = np.interp(
            np.linspace(0, n_samples, new_length, endpoint=False),
            np.arange(n_samples),
            arr[:, ch]
        ).astype(arr.dtype)
    return pygame.sndarray.make_sound(new_arr)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)

MENU_HOME = "home"
MENU_KITS = "kits"
MENU_LAYERS = "layers"
MENU_EXPORT = "export"
current_menu = MENU_HOME
current_instrument = None
global_bpm = 120
ignore_volume_adjustment = False
dragging_kit = None
dragging_pos = None
initial_pinch_position = None
last_back_time = 0
BACK_COOLDOWN = 0.5
toggle_pressed = False

menu_buttons = [
    {"name": "Kits", "pos": [250, 100]},
    {"name": "Layers", "pos": [600, 100]},
    {"name": "Export", "pos": [950, 100]},
]

back_button = {"name": "Back", "pos": [1150, 50], "radius": 30, "font_scale": 0.5}

kits = [
    {"name": "Drums", "pos": [300, 300], "volume": 50, "chord_index": 0, "chords": ["Pattern1", "Pattern2", "Pattern3"], "active": False},
    {"name": "Piano", "pos": [600, 450], "volume": 50, "chord_index": 0, "chords": [str(i) for i in range(1, 21)], "active": False},
    {"name": "Guitar", "pos": [900, 300], "volume": 50, "chord_index": 0, "chords": ["Em", "G", "D", "C"], "active": False},
]

instrument_prev_x = None
swipe_threshold = 40

def detect_pinch(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    distance = np.linalg.norm(
        np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_tip.x, index_tip.y])
    )
    return distance < 0.05

def last_three_fingers_closed(hand_landmarks):
    middle_closed = hand_landmarks.landmark[12].y > hand_landmarks.landmark[10].y
    ring_closed = hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y
    pinky_closed = hand_landmarks.landmark[20].y > hand_landmarks.landmark[18].y
    return middle_closed and ring_closed and pinky_closed

def find_nearest_button(x, y, buttons):
    for button in buttons:
        dist = np.sqrt((x - button["pos"][0])**2 + (y - button["pos"][1])**2)
        if dist < 100:
            return button
    return None

def find_nearest_kit(x, y):
    for kit in kits:
        dist = np.sqrt((x - kit["pos"][0])**2 + (y - kit["pos"][1])**2)
        if dist < 100:
            return kit
    return None

def draw_buttons(frame, buttons):
    for button in buttons:
        radius = button.get("radius", 50)
        cv2.circle(frame, tuple(button["pos"]), radius, (255, 255, 255), -1, cv2.LINE_AA)
        draw_centered_text(frame, button["name"], button["pos"][0], button["pos"][1], font_scale=button.get("font_scale", 0.7))

def draw_centered_text(frame, text, x, y, font_scale=0.7, thickness=2, color=(0,0,0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = x - text_size[0] // 2
    text_y = y + text_size[1] // 2
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)

def draw_menu(frame):
    if current_menu == MENU_HOME:
        draw_buttons(frame, menu_buttons)
        draw_centered_text(frame, "Home Menu", 640, 50, font_scale=1)
    
    elif current_menu == MENU_KITS:
        for kit in kits:
            cv2.circle(frame, tuple(kit["pos"]), 50, (255, 255, 255), -1, cv2.LINE_AA)
            draw_centered_text(frame, kit["name"], kit["pos"][0], kit["pos"][1])
        draw_buttons(frame, [back_button])
        draw_centered_text(frame, "Select a Kit", 640, 50, font_scale=1)

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        draw_menu(frame)

        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                hand_label = results.multi_handedness[idx].classification[0].label.lower()
                x = int(hand_landmarks.landmark[8].x * 1280)
                y = int(hand_landmarks.landmark[8].y * 720)

                if hand_label == "right":
                    if current_menu == MENU_HOME and detect_pinch(hand_landmarks):
                        selected_button = find_nearest_button(x, y, menu_buttons)
                        if selected_button:
                            name = selected_button["name"].lower()
                            if name == "kits":
                                current_menu = MENU_KITS

                    elif current_menu == MENU_KITS and detect_pinch(hand_landmarks):
                        kit = find_nearest_kit(x, y)
                        if kit:
                            current_instrument = kit
                            current_menu = kit["name"].lower()

                elif hand_label == "left" and current_menu in ["drums", "piano", "guitar"]:
                    if detect_pinch(hand_landmarks):
                        global_bpm = int(((720 - y) / 720) * 200) + 60

        if current_menu in ["drums", "piano", "guitar"] and current_instrument is not None:
            if current_instrument["name"].lower() == "piano" and current_instrument.get("active", False):
                speed_factor = global_bpm / 120.0
                chord_num = int(current_instrument["chords"][current_instrument["chord_index"]])
                if (current_piano_chord_index != current_instrument["chord_index"]) or (abs(current_speed_factor - speed_factor) > 0.01):
                    if current_piano_channel is not None:
                        current_piano_channel.stop()
                    new_sound = change_playback_speed(piano_progressions_original[chord_num], speed_factor)
                    current_piano_channel = new_sound.play(loops=-1)
                    current_speed_factor = speed_factor
                    current_piano_chord_index = current_instrument["chord_index"]

        cv2.imshow("Hand Gesture Music UI", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
