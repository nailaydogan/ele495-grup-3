# pip install mediapipe (it will install openCV and numPy together)

import cv2
import math
import time
import mediapipe as mp
import subprocess
import json

mp_hands = mp.solutions.hands

# .-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------.
# | Palm Status |    Handedness    | Angle Requirement |                Finger Requirement                 |     State Requirement      |      Outcome     |        State Change        |
# |-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
# |  Palm Far   |        -         |         -         |                      All Open                     |           CLOSED           |      Open TV     |            OPEN            |
# |-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
# |  Palm Near  |        -         |         -         |                     All Closed                    |            OPEN            |     Close TV     |           CLOSED           |
# |  Palm Far   |  Only Left Hand  |      Angle 0      |             Thumb Open, Others Closed             |            OPEN            |     Volume Up    |             -              |
# |  Palm Near  |  Only Left Hand  |      Angle 0      |             Thumb Open, Others Closed             |            OPEN            |    Volume Down   |             -              |
# |  Palm Far   | Only Right Hand  |     Angle 180     |             Thumb Closed, Others Open             |            OPEN            | Previous Channel |             -              |
# |  Palm Far   |  Only Left Hand  |      Angle 0      |             Thumb Closed, Others Open             |            OPEN            |   Next Channel   |             -              |
# |  Palm Near  |        -         |      Angle 90     |                      All Open                     |            OPEN            |    Toggle Mute   |             -              |
# |  Palm Near  |  Only Left Hand  |      Angle 90     |       Thumb-Index Finger Open, Others Closed      |            OPEN            |   Begin Channel  |   ENTERING_CHANNEL_FIRST   |
# |-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
# |  Palm Near  |  Only Left Hand  |         -         |                     All Closed                    |      ENTERING_CHANNEL      |   Channel -> 0   |            OPEN            |
# |  Palm Near  |  Only Left Hand  |         -         |             Thumb Open, Others Closed             |      ENTERING_CHANNEL      |   Channel -> 1   |            OPEN            |
# |  Palm Near  |  Only Left Hand  |         -         |     Thumb to Index Finger Open, Others Closed     |      ENTERING_CHANNEL      |   Channel -> 2   |            OPEN            |
# |  Palm Near  |  Only Left Hand  |         -         |     Thumb to Middle Finger Open, Others Closed    |      ENTERING_CHANNEL      |   Channel -> 3   |            OPEN            |
# |  Palm Near  |  Only Left Hand  |         -         |      Thumb to Ring Finger Open, Others Closed     |      ENTERING_CHANNEL      |   Channel -> 4   |            OPEN            |
# |  Palm Near  |  Only Left Hand  |         -         |                      All Open                     |      ENTERING_CHANNEL      |   Channel -> 5   |            OPEN            |
# |  Palm Near  |  Only Left Hand  |         -         |             Pinky Open, Others Closed             |      ENTERING_CHANNEL      |   Channel -> 6   |            OPEN            |
# |  Palm Near  |  Only Left Hand  |         -         |      Pinky to Ring Finger Open, Others Closed     |      ENTERING_CHANNEL      |   Channel -> 7   |            OPEN            |
# |  Palm Near  |  Only Left Hand  |         -         |     Pinky to Middle Finger Open, Others Closed    |      ENTERING_CHANNEL      |   Channel -> 8   |            OPEN            |
# |  Palm Near  |  Only Left Hand  |         -         |     Pinky to Index Finger Open, Others Closed     |      ENTERING_CHANNEL      |   Channel -> 9   |            OPEN            |
# |  Palm Far   |  Only Right Hand |         -         |                      All Open                     |      ENTERING_CHANNEL      |   Stop Editing   |            OPEN            |
# '-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

class RemoteController():
    # States
    CLOSED = 0
    OPEN = 1
    ENTERING_CHANNEL = 2

    # Possible readings
    CLOSE_TV = 10
    OPEN_TV = 11
    VOLUME_UP = 12
    VOLUME_DOWN = 13
    PREV_CHANNEL = 14
    NEXT_CHANNEL = 15
    TOGGLE_MUTE = 16
    GO_CHANNEL = 17
    STOP_EDITTING = 18

    # Numbers
    NUMBER_ZERO = 20
    NUMBER_ONE = 21
    NUMBER_TWO = 22
    NUMBER_THREE = 23
    NUMBER_FOUR = 24
    NUMBER_FIVE = 25
    NUMBER_SIX = 26
    NUMBER_SEVEN = 27
    NUMBER_EIGHT = 28
    NUMBER_NINE = 29

    # ir-ctl related options
    EXE_PATH = "/usr/bin/ir-ctl"
    PROTOCOL = "nec"
    DEVICE = "/dev/lirc0"

    def __init__(self, enable_debug=False, acceptance_count=50, rejection_count=10, sleep_between_decimals=0.75):
        self.hand_interpreter = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=1)
        self.capture = cv2.VideoCapture(0)

        self.enable_debug = enable_debug
        self.acceptance_count = acceptance_count
        self.rejection_count = rejection_count
        self.sleep_between_decimals = sleep_between_decimals

        self.last_read_input = -1
        self.last_read_input_count = 0
        self.last_read_threshold_count = 0

        self.state = RemoteController.CLOSED
        self.is_muted = False
        self.channel_number = []

        fp = open('key_codes.json', 'r')
        self.key_codes = json.load(fp)
        fp.close()

    # It basically returns wrist coordinates and MCP (bottom of any finger) averaged (between middle and ring fingers) coordinates
    def important_coord_calculation(self):
        wrist_coords = (self.landmarks[mp_hands.HandLandmark.WRIST].x * self.width, self.landmarks[mp_hands.HandLandmark.WRIST].y * self.height)
        mcp_average_coords = ((self.landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x + self.landmarks[mp_hands.HandLandmark.RING_FINGER_MCP].x) * self.width / 2.0, (self.landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y + self.landmarks[mp_hands.HandLandmark.RING_FINGER_MCP].y) * self.height / 2.0)

        return (wrist_coords, mcp_average_coords)

    # First value in radians, second in degrees
    def angle_calculation(self, wrist_coords, mcp_average_coords):
        angle_rad = math.atan2(wrist_coords[1] - mcp_average_coords[1], mcp_average_coords[0] - wrist_coords[0])
        angle_deg = (angle_rad / (2 * math.pi) * 360.0 + 360.0) % 360.0

        if self.enable_debug:
            cv2.putText(self.image, "Angle: %.2f" % angle_deg, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return (angle_rad, angle_deg)

    # Order: thumb, index finger, middle finger, ring finger, pinky, circle center (x, y), radius
    def finger_inside_palm(self, wrist_coords, mcp_average_coords):
        center = ((wrist_coords[0] + mcp_average_coords[0]) / 2.0, (wrist_coords[1] + mcp_average_coords[1]) / 2.0)
        radius = math.sqrt((wrist_coords[0] - mcp_average_coords[0]) ** 2 + (wrist_coords[1] - mcp_average_coords[1]) ** 2)
        squared_radius = radius ** 2

        thumb_tip = (self.landmarks[mp_hands.HandLandmark.THUMB_TIP].x * self.width, self.landmarks[mp_hands.HandLandmark.THUMB_TIP].y * self.height)
        index_finger_tip = (self.landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * self.width, self.landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * self.height)
        middle_finger_tip = (self.landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * self.width, self.landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * self.height)
        ring_finger_tip = (self.landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].x * self.width, self.landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y * self.height)
        pinky_tip = (self.landmarks[mp_hands.HandLandmark.PINKY_TIP].x * self.width, self.landmarks[mp_hands.HandLandmark.PINKY_TIP].y * self.height)

        is_thumb_inside = ((thumb_tip[0] - center[0]) ** 2 + (thumb_tip[1] - center[1]) ** 2) < squared_radius
        is_index_finger_inside = ((index_finger_tip[0] - center[0]) ** 2 + (index_finger_tip[1] - center[1]) ** 2) < squared_radius
        is_middle_finger_inside = ((middle_finger_tip[0] - center[0]) ** 2 + (middle_finger_tip[1] - center[1]) ** 2) < squared_radius
        is_ring_finger_inside = ((ring_finger_tip[0] - center[0]) ** 2 + (ring_finger_tip[1] - center[1]) ** 2) < squared_radius
        is_pinky_inside = ((pinky_tip[0] - center[0]) ** 2 + (pinky_tip[1] - center[1]) ** 2) < squared_radius

        if self.enable_debug:
            cv2.putText(self.image, "Values: {}, {}, {}, {}, {}".format(is_thumb_inside, is_index_finger_inside, is_middle_finger_inside, is_ring_finger_inside, is_pinky_inside), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            self.image = cv2.circle(self.image, (int(center[0]), int(center[1])), int(radius), (0, 255, 0), 2)

        return (is_thumb_inside, is_index_finger_inside, is_middle_finger_inside, is_ring_finger_inside, is_pinky_inside, center, radius)

    def hand_orientation(self):
        avg_coord = ((self.landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x + self.landmarks[mp_hands.HandLandmark.RING_FINGER_MCP].x) / 2.0, 1.0 - (self.landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y + self.landmarks[mp_hands.HandLandmark.RING_FINGER_MCP].y) / 2.0, (self.landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].z + self.landmarks[mp_hands.HandLandmark.RING_FINGER_MCP].z) / 2.0)
        wrist_coord = (self.landmarks[mp_hands.HandLandmark.WRIST].x, 1.0 - self.landmarks[mp_hands.HandLandmark.WRIST].y, self.landmarks[mp_hands.HandLandmark.WRIST].z)
        thumb_cmc_coord = (self.landmarks[mp_hands.HandLandmark.THUMB_CMC].x, 1.0 - self.landmarks[mp_hands.HandLandmark.THUMB_CMC].y, self.landmarks[mp_hands.HandLandmark.THUMB_CMC].z)

        vector_1 = (avg_coord[0] - wrist_coord[0], avg_coord[1] - wrist_coord[1], avg_coord[2] - wrist_coord[2])
        vector_2 = (thumb_cmc_coord[0] - wrist_coord[0], thumb_cmc_coord[1] - wrist_coord[1], thumb_cmc_coord[2] - wrist_coord[2])

        orientation = (vector_1[1] * vector_2[2] - vector_1[2] * vector_2[1], vector_1[2] * vector_2[0] - vector_1[0] * vector_2[2], vector_1[0] * vector_2[1] - vector_1[1] * vector_2[0])

        is_palm_near = False if (orientation[2] > 0 and self.hand_label == "Left") or (orientation[2] < 0 and self.hand_label == "Right") else True

        if self.enable_debug:
            cv2.putText(self.image, "Orientation: {}".format("Palm Near" if is_palm_near else "Palm Far"), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return (orientation, is_palm_near)

    def state_process(self, conditions_inputs_and_outcomes):
        has_match = False

        for element in conditions_inputs_and_outcomes:
            if element[0]:
                self.last_read_threshold_count = 0

                if self.last_read_input != element[1]:
                    self.last_read_input = element[1]
                    self.last_read_input_count = 0
                else:
                    self.last_read_input_count += 1

                if self.last_read_input_count >= self.acceptance_count:
                    self.state = element[2]

                    self.last_read_input = -1
                    self.last_read_input_count = 0
                    self.last_read_threshold_count = 0

                    if len(element) == 4:
                        element[3]()
                    elif len(element) == 5:
                        element[3](element[4])

                has_match = True

                break

        if not has_match and self.last_read_input != -1:
            self.last_read_threshold_count += 1

            if self.last_read_threshold_count >= self.rejection_count:
                self.last_read_input = -1
                self.last_read_input_count = 0
                self.last_read_threshold_count = 0

        if self.enable_debug:
            cv2.putText(self.image, "State: {}".format(self.state), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(self.image, "Input: {}".format(self.last_read_input), (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(self.image, "Input Count: {}".format(self.last_read_input_count), (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(self.image, "Threshold Count: {}".format(self.last_read_threshold_count), (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    def reset(self):
        self.last_read_input = -1
        self.last_read_input_count = 0
        self.last_read_threshold_count = 0

        self.state = RemoteController.CLOSED
        self.is_muted = False
        self.channel_number = []

    def ir_command_helper(self, hex_code):
        command = [self.EXE_PATH, "-S", "{}:{}".format(self.PROTOCOL, hex_code), "-d", self.DEVICE]
        result = subprocess.call(command)

        return result

    def string_channel_number(self):
        return "".join(map(lambda x: str(x), self.channel_number)).lstrip("0")

    def open_tv_function(self):
        self.ir_command_helper(self.key_codes["KEY_POWER"])
        print('TV opened')

    def close_tv_function(self):
        self.ir_command_helper(self.key_codes["KEY_POWER"])
        print('TV closed')

    def volume_up_function(self):
        self.ir_command_helper(self.key_codes["KEY_VOLUME_UP"])
        print('Volume increased')

    def volume_down_function(self):
        self.ir_command_helper(self.key_codes["KEY_VOLUME_DOWN"])
        print('Volume decreased')

    def prev_channel_function(self):
        self.ir_command_helper(self.key_codes["KEY_PREV_CHANNEL"])
        print('Go previous channel')

    def next_channel_function(self):
        self.ir_command_helper(self.key_codes["KEY_NEXT_CHANNEL"])
        print('Go next channel')

    def toggle_mute(self):
        self.ir_command_helper(self.key_codes["KEY_MUTE"])
        self.is_muted = not self.is_muted

        if self.is_muted:
            print('Now muted')
        else:
            print('Mute deactivated')

    def go_channel(self):
        print('Started channel selection process')

    def add_decimal(self, decimal):
        self.channel_number.append(decimal)

        print("Current number: {}".format(self.string_channel_number()))

    def stop_editting(self):
        new_list = []
        is_elapsed = False

        for decimal in self.channel_number:
            if decimal != 0:
                is_elapsed = True

            if is_elapsed:
                new_list.append(decimal)

        if len(new_list) == 0:
            print("No channel number is entered!")
        else:
            for decimal in new_list:
                if decimal == 0:
                    self.ir_command_helper(self.key_codes["KEY_ZERO"])
                elif decimal == 1:
                    self.ir_command_helper(self.key_codes["KEY_ONE"])
                elif decimal == 2:
                    self.ir_command_helper(self.key_codes["KEY_TWO"])
                elif decimal == 3:
                    self.ir_command_helper(self.key_codes["KEY_THREE"])
                elif decimal == 4:
                    self.ir_command_helper(self.key_codes["KEY_FOUR"])
                elif decimal == 5:
                    self.ir_command_helper(self.key_codes["KEY_FIVE"])
                elif decimal == 6:
                    self.ir_command_helper(self.key_codes["KEY_SIX"])
                elif decimal == 7:
                    self.ir_command_helper(self.key_codes["KEY_SEVEN"])
                elif decimal == 8:
                    self.ir_command_helper(self.key_codes["KEY_EIGHT"])
                elif decimal == 9:
                    self.ir_command_helper(self.key_codes["KEY_NINE"])

                time.sleep(self.sleep_between_decimals)

            print("Go to channel number {}".format(self.string_channel_number()))

            self.channel_number = []

    def main_execution(self):
        while self.capture.isOpened():
            success, image = self.capture.read()

            start_time = time.time()

            if not success:
                print("Ignoring empty camera frame.")
                continue

            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

            self.height, self.width = image.shape[:2]

            image.flags.writeable = False
            results = self.hand_interpreter.process(image)

            image.flags.writeable = True
            self.image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                hand_landmark = results.multi_hand_landmarks[0]
                self.landmarks = hand_landmark.landmark

                handedness = results.multi_handedness[0]
                handedness_classification = handedness.classification[0]
                self.hand_label = handedness_classification.label

                (wrist_coords, mcp_average_coords) = self.important_coord_calculation()

                (angle_rad, angle_deg) = self.angle_calculation(wrist_coords, mcp_average_coords)

                (is_thumb_inside, is_index_finger_inside, is_middle_finger_inside, is_ring_finger_inside, is_pinky_inside, center, radius) = self.finger_inside_palm(wrist_coords, mcp_average_coords)

                (orientation, is_palm_near) = self.hand_orientation()

                if self.state == RemoteController.CLOSED:
                    open_tv_payload = (
                        not is_palm_near and not is_thumb_inside and not is_index_finger_inside and not is_middle_finger_inside and not is_ring_finger_inside and not is_pinky_inside,
                        RemoteController.OPEN_TV,
                        RemoteController.OPEN,
                        self.open_tv_function,
                    )

                    self.state_process((open_tv_payload, ))
                elif self.state == RemoteController.OPEN:
                    close_tv_payload = (
                        is_palm_near and is_thumb_inside and is_index_finger_inside and is_middle_finger_inside and is_ring_finger_inside and is_pinky_inside,
                        RemoteController.CLOSE_TV,
                        RemoteController.CLOSED,
                        self.close_tv_function,
                    )
                    volume_up_payload = (
                        self.hand_label == "Left" and (angle_deg < 30.0 or angle_deg > 330.0) and not is_palm_near and not is_thumb_inside and is_index_finger_inside and is_middle_finger_inside and is_ring_finger_inside and is_pinky_inside,
                        RemoteController.VOLUME_UP,
                        RemoteController.OPEN,
                        self.volume_up_function,
                    )
                    volume_down_payload = (
                        self.hand_label == "Left" and (angle_deg < 30.0 or angle_deg > 330.0) and is_palm_near and not is_thumb_inside and is_index_finger_inside and is_middle_finger_inside and is_ring_finger_inside and is_pinky_inside,
                        RemoteController.VOLUME_DOWN,
                        RemoteController.OPEN,
                        self.volume_down_function,
                    )
                    prev_channel_payload = (
                        self.hand_label == "Right" and (angle_deg < 210.0 or angle_deg > 150.0) and not is_palm_near and is_thumb_inside and not is_index_finger_inside and not is_middle_finger_inside and not is_ring_finger_inside and not is_pinky_inside,
                        RemoteController.PREV_CHANNEL,
                        RemoteController.OPEN,
                        self.prev_channel_function,
                    )
                    next_channel_payload = (
                        self.hand_label == "Left" and (angle_deg < 30.0 or angle_deg > 330.0) and not is_palm_near and is_thumb_inside and not is_index_finger_inside and not is_middle_finger_inside and not is_ring_finger_inside and not is_pinky_inside,
                        RemoteController.NEXT_CHANNEL,
                        RemoteController.OPEN,
                        self.next_channel_function,
                    )
                    toggle_mute_payload = (
                        (angle_deg < 120.0 or angle_deg > 60.0) and is_palm_near and not is_thumb_inside and not is_index_finger_inside and not is_middle_finger_inside and not is_ring_finger_inside and not is_pinky_inside,
                        RemoteController.TOGGLE_MUTE,
                        RemoteController.OPEN,
                        self.toggle_mute,
                    )
                    begin_channel_payload = (
                        self.hand_label == "Left" and (angle_deg < 120.0 or angle_deg > 60.0) and is_palm_near and is_thumb_inside and is_index_finger_inside and not is_middle_finger_inside and not is_ring_finger_inside and not is_pinky_inside,
                        RemoteController.GO_CHANNEL,
                        RemoteController.ENTERING_CHANNEL,
                        self.go_channel,
                    )

                    self.state_process((close_tv_payload, volume_up_payload, volume_down_payload, prev_channel_payload, next_channel_payload, toggle_mute_payload, begin_channel_payload))
                elif self.state == RemoteController.ENTERING_CHANNEL:
                    number_zero_payload = (
                        self.hand_label == "Left" and is_palm_near and is_thumb_inside and is_index_finger_inside and is_middle_finger_inside and is_ring_finger_inside and is_pinky_inside,
                        RemoteController.NUMBER_ZERO,
                        RemoteController.ENTERING_CHANNEL,
                        self.add_decimal,
                        0,
                    )
                    number_one_payload = (
                        self.hand_label == "Left" and is_palm_near and not is_thumb_inside and is_index_finger_inside and is_middle_finger_inside and is_ring_finger_inside and is_pinky_inside,
                        RemoteController.NUMBER_ONE,
                        RemoteController.ENTERING_CHANNEL,
                        self.add_decimal,
                        1,
                    )
                    number_two_payload = (
                        self.hand_label == "Left" and is_palm_near and not is_thumb_inside and not is_index_finger_inside and is_middle_finger_inside and is_ring_finger_inside and is_pinky_inside,
                        RemoteController.NUMBER_TWO,
                        RemoteController.ENTERING_CHANNEL,
                        self.add_decimal,
                        2,
                    )
                    number_three_payload = (
                        self.hand_label == "Left" and is_palm_near and not is_thumb_inside and not is_index_finger_inside and not is_middle_finger_inside and is_ring_finger_inside and is_pinky_inside,
                        RemoteController.NUMBER_THREE,
                        RemoteController.ENTERING_CHANNEL,
                        self.add_decimal,
                        3,
                    )
                    number_four_payload = (
                        self.hand_label == "Left" and is_palm_near and not is_thumb_inside and not is_index_finger_inside and not is_middle_finger_inside and not is_ring_finger_inside and is_pinky_inside,
                        RemoteController.NUMBER_FOUR,
                        RemoteController.ENTERING_CHANNEL,
                        self.add_decimal,
                        4,
                    )
                    number_five_payload = (
                        self.hand_label == "Left" and is_palm_near and not is_thumb_inside and not is_index_finger_inside and not is_middle_finger_inside and not is_ring_finger_inside and not is_pinky_inside,
                        RemoteController.NUMBER_FIVE,
                        RemoteController.ENTERING_CHANNEL,
                        self.add_decimal,
                        5,
                    )
                    number_six_payload = (
                        self.hand_label == "Left" and is_palm_near and is_thumb_inside and is_index_finger_inside and is_middle_finger_inside and is_ring_finger_inside and not is_pinky_inside,
                        RemoteController.NUMBER_SIX,
                        RemoteController.ENTERING_CHANNEL,
                        self.add_decimal,
                        6,
                    )
                    number_seven_payload = (
                        self.hand_label == "Left" and is_palm_near and is_thumb_inside and is_index_finger_inside and is_middle_finger_inside and not is_ring_finger_inside and not is_pinky_inside,
                        RemoteController.NUMBER_SEVEN,
                        RemoteController.ENTERING_CHANNEL,
                        self.add_decimal,
                        7,
                    )
                    number_eight_payload = (
                        self.hand_label == "Left" and is_palm_near and is_thumb_inside and is_index_finger_inside and not is_middle_finger_inside and not is_ring_finger_inside and not is_pinky_inside,
                        RemoteController.NUMBER_EIGHT,
                        RemoteController.ENTERING_CHANNEL,
                        self.add_decimal,
                        8,
                    )
                    number_nine_payload = (
                        self.hand_label == "Left" and is_palm_near and is_thumb_inside and not is_index_finger_inside and not is_middle_finger_inside and not is_ring_finger_inside and not is_pinky_inside,
                        RemoteController.NUMBER_NINE,
                        RemoteController.ENTERING_CHANNEL,
                        self.add_decimal,
                        9,
                    )
                    stop_editting = (
                        self.hand_label == "Right" and not is_palm_near and not is_thumb_inside and not is_index_finger_inside and not is_middle_finger_inside and not is_ring_finger_inside and not is_pinky_inside,
                        RemoteController.STOP_EDITTING,
                        RemoteController.OPEN,
                        self.stop_editting,
                    )

                    self.state_process((number_zero_payload, number_one_payload, number_two_payload, number_three_payload, number_four_payload, number_five_payload, number_six_payload, number_seven_payload, number_eight_payload, number_nine_payload, stop_editting))

            stop_time = time.time()
            fps = 1.0 / (stop_time - start_time)

            if self.enable_debug:
                cv2.putText(self.image, "FPS: %.2f" % fps, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow('Remote Control', self.image)

            key = cv2.waitKey(5) & 0xFF

            if key == 27:
                break
            elif key == 114 and self.enable_debug:
                self.reset()

        self.hand_interpreter.close()
        self.capture.release()

if __name__ == "__main__":
    controller = RemoteController(enable_debug=True, acceptance_count=25, rejection_count=10, sleep_between_decimals=0.75)
    controller.main_execution()