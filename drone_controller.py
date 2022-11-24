from djitellopy import Tello
from utils import GestureFilter


class DroneController:
    """
    class for controlling tello drone
    """

    def __init__(self, tello: Tello, velocity: int = 50):
        self.tello = tello
        # self.tello.TAKEOFF_TIMEOUT = 3
        # connect tello drone
        self.tello.connect()
        self._is_landing = False
        self._is_flying = True
        # intialize the speed of the drone
        self.velocity = velocity
        self.forward_backward_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0

    def command_control(self, filter: GestureFilter):
        """
        control the drone based on the given gesture after filtering
        """
        command = filter.get_gesture()
        # print("command", command)

        # if the drone is landing, do not send any other commands
        # if the drone is not landing, send the command
        if not self._is_landing:

            if command == "LAND":
                self._is_landing = True
                self.yaw_velocity = 0
                self.up_down_velocity = 0
                self.left_right_velocity = 0
                self.forward_backward_velocity = 0
                self.tello.land()
                self._is_landing = False

            elif command == "STOP":
                self.yaw_velocity = 0
                self.up_down_velocity = 0
                self.left_right_velocity = 0
                self.forward_backward_velocity = 0

            elif command == "FORWARD":
                self.forward_backward_velocity = self.velocity
                self.yaw_velocity = 0
                self.up_down_velocity = 0
                self.left_right_velocity = 0

            elif command == "BACK":
                self.forward_backward_velocity = -self.velocity
                self.yaw_velocity = 0
                self.up_down_velocity = 0
                self.left_right_velocity = 0

            elif command == "UP":
                self.up_down_velocity = self.velocity
                self.yaw_velocity = 0
                self.left_right_velocity = 0
                self.forward_backward_velocity = 0

            elif command == "LEFT":
                self.left_right_velocity = self.velocity
                self.yaw_velocity = 0
                self.up_down_velocity = 0
                self.forward_backward_velocity = 0

            elif command == "RIGHT":
                self.left_right_velocity = -self.velocity
                self.yaw_velocity = 0
                self.up_down_velocity = 0
                self.forward_backward_velocity = 0

            elif command == "TAKEOFF":
                if not self.tello.is_flying:
                    self.tello.takeoff()

            # send the command to the drone
            self.tello.send_rc_control(
                self.left_right_velocity,
                self.forward_backward_velocity,
                self.up_down_velocity,
                self.yaw_velocity,
            )
