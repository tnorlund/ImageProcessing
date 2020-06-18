import cv2
import json
import os
import numpy as np
from matplotlib import pyplot as plt

def read_images(config_file, image_dir):
    with open(config_file) as f:
        config = json.load(f)
    for image in config.keys():
        config[image]["IMAGE"] = cv2.imread(os.path.join(image_dir, image))
        config[image]["MATCHED-CORNERS"] = {}
        for face in config[image]["FACES"].keys():
            config[image]["MATCHED-CORNERS"][face] = {}
            known_corners = {
                corner_name:config[image]["CORNERS"][corner_name] 
                for corner_name in config[image]["CORNERS"] 
                if face in corner_name
            }
            for square_coord in config[image]["FACES"][face]:
                distances = np.array([
                    np.sqrt(
                        np.power((corner_coord[0] - square_coord[0]), 2) \
                        + np.power((corner_coord[1] - square_coord[1]), 2)
                    )
                    for corner_coord in known_corners.values()
                ])
                closest_corner = list(known_corners.keys())[
                    np.where(distances == distances.min())[0][0]
                ]
                config[image]["MATCHED-CORNERS"][face][closest_corner + "-FACE"] = square_coord
                del known_corners[closest_corner]
    return(config)

def match_points(image_1, image_2, config):
    matched_points = {}
    matched_points[image_1] = []
    matched_points[image_2] = []
    for corner in set(config[image_1]["CORNERS"]) & set(config[image_2]["CORNERS"]):
        matched_points[image_1].append(
            config[image_1]["CORNERS"][corner]
        )
        matched_points[image_2].append(
            config[image_2]["CORNERS"][corner]
        )
    for face in set(config[image_1]["MATCHED-CORNERS"]) & set(config[image_2]["MATCHED-CORNERS"]):
        for face_value in set(config[image_1]["MATCHED-CORNERS"][face]) & set(config[image_2]["MATCHED-CORNERS"][face]):
            matched_points[image_1].append(
                config[image_1]["MATCHED-CORNERS"][face][face_value]
            )
            matched_points[image_2].append(
                config[image_2]["MATCHED-CORNERS"][face][face_value]
            )
    return(matched_points)

def show_images(image_1, image_2, config, selected_feature=None):
    fig, axes = plt.subplots(
        1, 2,
        figsize = (
            config[image_1]["IMAGE"].shape[1]/750 * 2, 
            config[list(set(config))[0]]["IMAGE"].shape[0]/750
        )
    )
    for image_index in range(2):
        if image_index == 0:
            image_name = image_1
        else:
            image_name = image_2
        axes[image_index].imshow(config[image_name]["IMAGE"][...,::-1])
        for face in config[image_name]["FACES"].keys():
            for point in config[image_name]["FACES"][face]:
                axes[image_index].scatter(point[0], point[1], color=face)
        for corner_color_key in config[image_name]["CORNERS"].keys():
            corner_colors = corner_color_key.split("-")
            for color_index in range(len(corner_colors)):
                axes[image_index].scatter(
                    config[image_name]["CORNERS"][corner_color_key][0],
                    config[image_name]["CORNERS"][corner_color_key][1],
                    color=corner_colors[color_index],
                    s=5 * (color_index + 1),
                    zorder=3 - color_index
                )
    if selected_feature != None:
        matched_points = match_points(image_1, image_2, config)
        axes[0].scatter(
            matched_points[image_1][selected_feature][0],
            matched_points[image_1][selected_feature][1],
            marker="X",
            color="Black",
            s=20,
            zorder=2
        )
        axes[0].scatter(
            matched_points[image_1][selected_feature][0],
            matched_points[image_1][selected_feature][1],
            marker="o",
            color="White",
            s=20,
            zorder=1
        )
        axes[1].scatter(
            matched_points[image_2][selected_feature][0],
            matched_points[image_2][selected_feature][1],
            marker="X",
            color="Black",
            s=20,
            zorder=2
        )
        axes[1].scatter(
            matched_points[image_2][selected_feature][0],
            matched_points[image_2][selected_feature][1],
            marker="o",
            color="White",
            s=20,
            zorder=1
        )
    plt.show()