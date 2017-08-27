import numpy as np
import cv2


MAX_ROLL = 0.5
MAX_PITCH = 0.5

OBSTACLE_CHANNEL = 0
ROCK_CHANNEL = 1
NAVIGABLE_CHANNEL = 2

SCALE = 10
BOTTOM_OFFSET = 6


# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def _color_thresh(img, rgb_thresh=(160, 160, 160)):
    """
    Returns an array of 1s and 0s where a 1 indicates the source pixel was
    above the given threshold and 0 indicates it was not. The returned array
    will be of the same shape as the source array, but is a single channel.
    """
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:, :, 0])

    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:, :, 0] > rgb_thresh[0]) \
        & (img[:, :, 1] > rgb_thresh[1]) \
        & (img[:, :, 2] > rgb_thresh[2])

    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select


# TODO: investigate other methods for this
# TODO: come up with our own selector based on images
def _find_rocks(img, selector=(110, 110, 50)):
    """
    Returns an array of 1s and 0s where a 1 indicates a pixel matched the given
    selector, and a 0 indicates it has not.
    This is a variation of _color_thresh specifically for finding rocks (the
    selector is a little different).
    """
    gold_pix = ((img[:, :, 0] > selector[0])
                & (img[:, :, 1] > selector[1])
                & (img[:, :, 2] < selector[2]))

    rock_map = np.zeros_like(img[:, :, 0])
    rock_map[gold_pix] = 1
    return rock_map


# Define a function to convert from image coords to rover coords
def _rover_coords(binary_img):
    """
    Returns coordinates with the rover camera as the origin (0, 0)
    """
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at
    # the center bottom of the image.
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1] / 2).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def _to_polar_coords(x_pixel, y_pixel):
    """
    Converts cartesian coordinates to polar coordinates
    """
    # Convert (x_pixel, y_pixel) to (distance, angle)
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles


# Define a function to map rover space pixels to world space
def _rotate_pix(xpix, ypix, yaw):
    """
    Applies a rotation matrix to the pixels, xpix and ypix, given the angle as
    yaw.
    """
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result
    return xpix_rotated, ypix_rotated


def _translate_pix(x, y, x_offset, y_offset, scale):
    """
    Applies a simple translation matrix to x and y given the offsets for each
    and the scale.
    """
    # Apply a scaling and a translation
    xpix_translated = (x / scale) + x_offset
    ypix_translated = (y / scale) + y_offset
    # Return the result
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def _pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    """
    Converts from rover coordinate space to world coordinate space by:
    1. Apply a rotation matrix on the coordinates to correct for the rovers
    current yaw.
    2. Apply a translation matrix on the resultant coordinates to adjust for
    the rovers current position in the world.
    3. Clip the resulting pixels to ensure they fall within the bounds of our
    world coordinate space.
    """
    # Apply rotation
    xpix_rot, ypix_rot = _rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = _translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world


def _field_of_view(img, src, dst):
    """
    Provides a mask to index our array such that we don't map values out of
    view of the cameras field of view
    """
    M = cv2.getPerspectiveTransform(src, dst)
    fov = cv2.warpPerspective(np.ones_like(img[:, :, 0]), M, (img.shape[1],
                              img.shape[0]))
    return fov


# Define a function to perform a perspective transform
def _perspect_transform(img, src, dst):
    """
    Applies a perspective transform of the image given the source and
    destination pixel positions.
    """
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1],
                                 img.shape[0]))  # keep same size as input image
    return warped


def _should_map(roll, pitch):
    """
    Checks to see if we should map the pixels we see given our current
    telemetry.
    """
    return _from_origin(roll) < MAX_ROLL and _from_origin(pitch) < MAX_PITCH


def _from_origin(degrees):
    """
    Returns degrees from origin. EG 359 -> 1, 5 -> 5
    """
    if degrees < 180:
        return degrees
    else:
        return 360 - degrees


# Apply the above functions in succession and update the Rover state
# accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # TODO:
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
    # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
    #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
    #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image

    # 5) Convert map image pixel values to rover-centric coords
    # 6) Convert rover-centric pixel values to world coordinates
    # 7) Update Rover worldmap (to be displayed on right side of screen)
    # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
    #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
    #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1

    # 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
    # Rover.nav_dists = rover_centric_pixel_distances
    # Rover.nav_angles = rover_centric_angles

    source = np.float32([[14, 140], [301, 140], [200, 96], [118, 96]])

    destination = np.float32([[Rover.img.shape[1] / 2 - (SCALE / 2), Rover.img.shape[0] - BOTTOM_OFFSET],
                              [Rover.img.shape[1] / 2 + (SCALE / 2), Rover.img.shape[0] - BOTTOM_OFFSET],
                              [Rover.img.shape[1] / 2 + (SCALE / 2), Rover.img.shape[0] - 2 * (SCALE / 2) - BOTTOM_OFFSET],
                              [Rover.img.shape[1] / 2 - (SCALE / 2), Rover.img.shape[0] - 2 * (SCALE / 2) - BOTTOM_OFFSET],
                              ])

    fov = _field_of_view(Rover.img, source, destination)
    warped = _perspect_transform(Rover.img, source, destination)

    navigable_threshold = (160, 160, 160)
    navigable_terrain = _color_thresh(warped, navigable_threshold)
    Rover.vision_image[:, :, NAVIGABLE_CHANNEL] = navigable_terrain * 255

    obstacles = np.absolute(np.float32(navigable_terrain) - 1) * fov
    Rover.vision_image[:, :, OBSTACLE_CHANNEL] = obstacles * 255

    worldmap_size = Rover.worldmap.shape[0]

    rover_x, rover_y = _rover_coords(navigable_terrain)
    rover_x_world, rover_y_world = _pix_to_world(rover_x, rover_y,
                                                 Rover.pos[0], Rover.pos[1], Rover.yaw, worldmap_size, SCALE)

    obstacles_x, obstacles_y = _rover_coords(obstacles)
    obstacles_x_world, obstacles_y_world = _pix_to_world(obstacles_x, obstacles_y,
                                                         Rover.pos[0], Rover.pos[1], Rover.yaw, worldmap_size, SCALE)

    # we are only mapping in one plane. we need to make sure we don't map anything when the robot is all topsy-turvy
    if _should_map(Rover.roll, Rover.pitch):
        # we are sure about navigable terrain
        Rover.worldmap[rover_y_world, rover_x_world, NAVIGABLE_CHANNEL] += 10  # we are sure about navigable terrain
        Rover.worldmap[rover_y_world, rover_x_world, OBSTACLE_CHANNEL] -= 10  # we are sure about navigable terrain

        # slowly build up confidence that a pixel is in fact an obstacle. If it
        # isn't then we will set it to 0 when we see it as navigable anyway
        Rover.worldmap[obstacles_y_world, obstacles_x_world, OBSTACLE_CHANNEL] += 1

    rock_map = _find_rocks(warped)
    if rock_map.any():
        rock_x, rock_y = _rover_coords(rock_map)
        rock_x_world, rock_y_world = _pix_to_world(rock_x, rock_y, Rover.pos[0], Rover.pos[1], Rover.yaw, worldmap_size, SCALE)

        rock_dist, rock_angles = _to_polar_coords(rock_x, rock_y)
        rock_anchor_index = np.argmin(rock_dist)
        rock_anchor_x = rock_x_world[rock_anchor_index]
        rock_anchor_y = rock_y_world[rock_anchor_index]

        # Add the rock to our vision
        Rover.vision_image[:, :, ROCK_CHANNEL] = rock_map * 255

        if _should_map(Rover.roll, Rover.pitch):
            Rover.worldmap[rock_anchor_x, rock_anchor_y, ROCK_CHANNEL] = 255

        # if we found a rock lets go towards it
        Rover.see_sample = 1
        Rover.nav_dists = rock_dist
        Rover.nav_angles = rock_angles

    else:
        Rover.vision_image[:, :, ROCK_CHANNEL] = rock_map * 0
        # no rock so we abide by our general navigation principles
        Rover.see_sample = 0
        dist, angles = _to_polar_coords(rover_x, rover_y)
        Rover.nav_dists = dist
        Rover.nav_angles = angles

    return Rover
