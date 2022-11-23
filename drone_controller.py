from djitellopy import Tello
from utils import GestureFilter


class DroneController:
    """
    class for controlling tello drone
    """

    def __init__(self, tello: Tello):
        self.tello = tello
        # connect tello drone
        self.tello.connect()
        self._is_langding = False
        # intialize the speed of the drone
        self.forward_backward_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0

    def command_control(self, filter: GestureFilter):
        """
        control the drone based on the given gesture after filtering
        """
        command = filter.get_gesture()
        print("command", command)

        # if the drone is landing, do not send any other commands
        # if the drone is not landing, send the command
        if not self._is_langding:

            if command == "FORWARD":
                self.forward_backward_velocity = 10
            elif command == "BACK":
                self.forward_backward_velocity = -10

            elif command == "LEFT":
                self.left_right_velocity = -30
            elif command == "RIGHT":
                self.left_right_velocity = 30

            elif command == "UP":
                self.up_down_velocity = 10
            elif command == "DOWN":
                self.up_down_velocity = -10

            elif command == "LAND":
                self._is_langding = True
                self.forward_backward_velocity = (
                    self.left_right_velocity
                ) = self.up_down_velocity = self.yaw_velocity = 0
                self.tello.land()

            elif command == "STOP":
                self.forward_backward_velocity = (
                    self.left_right_velocity
                ) = self.up_down_velocity = self.yaw_velocity = 0

            # send the command to the drone
            self.tello.send_rc_control(
                self.left_right_velocity,
                self.forward_backward_velocity,
                self.up_down_velocity,
                self.yaw_velocity,
            )
