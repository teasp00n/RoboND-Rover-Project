import numpy as np
import cv2


# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160)):
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
def find_rocks(img, selector=(110, 110, 50)):
    gold_pix = ((img[:, :, 0] > selector[0])
               & (img[:, :, 1] > selector[1])
               & (img[:, :, 2] < selector[2]))

    color_select = np.zeros_like(img[:, :, 0])
    color_select[gold_pix] = 1
    return color_select


# Define a function to convert from image coords to rover coords
def object_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at
    # the center bottom of the image.
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1] / 2).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle)
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles


# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))

    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result
    return xpix_rotated, ypix_rotated


def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale):
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world


def field_of_view(img, src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    fov = cv2.warpPerspective(np.ones_like(img[:, :, 0]), M, (img.shape[1],
                              img.shape[0]))
    return fov


# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1],
                                 img.shape[0]))  # keep same size as input image
    return warped


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

    scale = 10
    bottom_offset = 6

    source = np.float32([[14, 140], [301, 140], [200, 96], [118, 96]])

    destination = np.float32([[Rover.img.shape[1] / 2 - (scale / 2), Rover.img.shape[0] - bottom_offset],
                              [Rover.img.shape[1] / 2 + (scale / 2), Rover.img.shape[0] - bottom_offset],
                              [Rover.img.shape[1] / 2 + (scale / 2), Rover.img.shape[0] - 2 * (scale / 2) - bottom_offset],
                              [Rover.img.shape[1] / 2 - (scale / 2), Rover.img.shape[0] - 2 * (scale / 2) - bottom_offset],
                              ])

    fov = field_of_view(Rover.img, source, destination)
    warped = perspect_transform(Rover.img, source, destination)

    navigable_threshold = (160, 160, 160)
    navigable_terrain = color_thresh(warped, navigable_threshold)
    Rover.vision_image[:, :, 2] = navigable_terrain * 255

    obstacles = np.absolute(np.float32(navigable_terrain) - 1) * fov
    Rover.vision_image[:, :, 0] = obstacles * 255

    worldmap_size = Rover.worldmap.shape[0]

    rover_x, rover_y = object_coords(navigable_terrain)
    rover_x_world, rover_y_world = pix_to_world(rover_x, rover_y,
                                                Rover.pos[0], Rover.pos[1], Rover.yaw, worldmap_size, scale)

    obstacles_x, obstacles_y = object_coords(obstacles)
    obstacles_x_world, obstacles_y_world = pix_to_world(obstacles_x, obstacles_y,
                                                        Rover.pos[0], Rover.pos[1], Rover.yaw, worldmap_size, scale)

    rock_map = find_rocks(warped)
    if rock_map.any():
        rock_x, rock_y = object_coords(rock_map)
        rock_x_world, rock_y_world = pix_to_world(rock_x, rock_y,
                                                  Rover.pos[0], Rover.pos[1], Rover.yaw, worldmap_size, scale)

        rock_dist, rock_angles = to_polar_coords(rock_x, rock_y)
        rock_anchor_index = np.argmin(rock_dist)
        rock_anchor_x = rock_x_world[rock_anchor_index]
        rock_anchor_y = rock_y_world[rock_anchor_index]

        Rover.worldmap[rock_anchor_x, rock_anchor_y, 1] += 1
        Rover.vision_image[:, :, 1] = rock_map * 255
        Rover.samples_located += 1

        # if we found a rock lets go towards it
        Rover.see_sample = 1
        Rover.nav_dists = rock_dist
        Rover.nav_angles = rock_angles

    else:
        Rover.vision_image[:, :, 1] = 0
        # no rock so we abide by our general navigation principles
        Rover.see_sample = 0
        dist, angles = to_polar_coords(rover_x, rover_y)
        Rover.nav_dists = dist
        Rover.nav_angles = angles

    # we are only mapping in one plane. we need to make sure we don't map anything when the robot is all topsy-turvy
    if Rover.roll < 3 and Rover.pitch < 3:
        Rover.worldmap[rover_y_world, rover_x_world, 2] += 1
        Rover.worldmap[obstacles_y_world, obstacles_x_world, 0] += 1

    return Rover
