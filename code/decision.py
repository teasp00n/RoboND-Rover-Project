import numpy as np


def _get_steer_angle(Rover):
    """
    Inspects the rover state and determines the best way to turn.
    """
    return np.clip(np.mean(Rover.nav_angles * 180 / np.pi), -15, 15)


def _are_we_stuck(Rover):
    """
    Determines if we are stuck or not by looking at the previous positions we
    have recorded and checks the delta between the largest and smallest is over
    some arbitrary threshold.
    """
    recent_positions_np = np.array(Rover.recent_positions)
    delta_x = recent_positions_np[:, 0].ptp()
    delta_y = recent_positions_np[:, 1].ptp()
    delta = np.hypot(delta_x, delta_y)
    Rover.post_stuck_leway += 1
    return delta < 0.1 and Rover.throttle > 0 and Rover.post_stuck_leway >= Rover.max_post_stuck_leway


# This is where you can build a decision tree for determining throttle, brake
# and steer commands based on the output of the perception_step() function
def decision_step(Rover):
    # Implement conditionals to decide what to do given perception data
    # Here you're all set up with some basic functionality but you'll need to
    # improve on this decision tree to do a good job of navigating
    # autonomously!

    # Example:
    # Check if we have vision data to make decisions with
    if Rover.nav_angles is not None:

        # Check for Rover.mode status
        if Rover.mode == 'forward':
            # Steady as she goes!
            if Rover.see_sample:
                Rover.max_vel = 1
            else:
                Rover.max_vel = 2

            # print(np.array(Rover.recent_positions[1]).ptp())
            # Check the extent of navigable terrain
            # Check if we see a rock, if we do we don't care if we are running
            # out of space. we must have that rock!
            if (Rover.see_sample and not Rover.near_sample) or len(Rover.nav_angles) >= Rover.stop_forward:
                # If mode is forward, navigable terrain looks good
                # and velocity is below max, then throttle
                if Rover.vel < Rover.max_vel:
                    # Set throttle value to throttle setting
                    Rover.throttle = Rover.throttle_set
                else:  # Else coast
                    Rover.throttle = 0
                Rover.brake = 0
                # Set steering to average angle clipped to the range +/- 15
                Rover.steer = _get_steer_angle(Rover)
            # If there's a lack of navigable terrain pixels then go to 'stop' mode
            # Or if we are near the rock, we have to stop to pick it up
            elif Rover.near_sample or len(Rover.nav_angles) < Rover.stop_forward:
                # Set mode to "stop" and hit the brakes!
                Rover.throttle = 0
                # Set brake to stored brake value
                Rover.brake = Rover.brake_set
                Rover.steer = 0
                Rover.mode = 'stop'

            if _are_we_stuck(Rover):
                Rover.mode = 'stuck'

        # If we're already in "stop" mode then make different decisions
        elif Rover.mode == 'stop':
            # If we're in stop mode but still moving keep braking
            if Rover.vel > 0.2:
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0
            # If we're not moving (vel < 0.2) then do something else
            elif Rover.vel <= 0.2:
                # Now we're stopped and we have vision data to see if there's a path forward
                if not Rover.see_sample and len(Rover.nav_angles) < Rover.go_forward:
                    Rover.throttle = 0
                    # Release the brake to allow turning
                    Rover.brake = 0
                    # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
                    Rover.steer = -15
                # If we're stopped but see sufficient navigable terrain in front then go!
                if not Rover.see_sample and len(Rover.nav_angles) >= Rover.go_forward:
                    # Set throttle back to stored value
                    Rover.throttle = Rover.throttle_set
                    # Release the brake
                    Rover.brake = 0
                    # Set steer to mean angle
                    Rover.steer = np.clip(
                        np.mean(Rover.nav_angles * 180 / np.pi), -15, 15)
                    Rover.mode = 'forward'

        elif Rover.mode == 'stuck':
            if Rover.spin_ticks <= Rover.max_spin_ticks:
                Rover.throttle = 0
                # Release the brake to allow turning
                Rover.brake = 0
                # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
                Rover.steer = -15
                Rover.spin_ticks += 1
            else:
                Rover.spin_ticks = 0
                Rover.post_stuck_leway = 0
                Rover.mode = 'forward'


    # Just to make the rover do something
    # even if no modifications have been made to the code
    else:
        Rover.throttle = Rover.throttle_set
        Rover.steer = 0
        Rover.brake = 0

    # If in a state where want to pickup a rock send pickup command
    if Rover.near_sample and Rover.vel == 0 and not Rover.picking_up:
        Rover.send_pickup = True

    return Rover
