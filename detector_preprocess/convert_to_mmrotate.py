"""Convert files to mmrotate format.

Input is a directory containing pairs of PNG and JSON files:
- {label}.png
- {label}.json

The JSON should be a list of points like this:
    [
        [
            [stern_x, stern_y],
            [bow_x, bow_y],
            [port_x, port_y],
            [starboard_x, starboard_y],
        ],
        [
            ...
        ],
        ...
    ]

Output is a directory containing pairs of PNG and TXT files:
- images/{label}.png
- labelTxt/{label}.txt

The TXT should be a list of oriented bounding boxes like this:
    x1 y1 x2 y2 x3 y3 x4 y4 category difficult

Difficult should always be 0, category should always be "vessel".
The points are clockwise order starting from the stern/port corner of the vessel.
"""

import argparse
import json
import math
import os
import shutil

crop_size = 1024
padding = 2

def get_center(points):
    x = 0
    y = 0
    for point in points:
        x += point[0]
        y += point[1]
    return (x/len(points), y/len(points))

def point_subtract(point1, point2):
    return (point1[0]-point2[0], point1[1]-point2[1])

def point_add(point1, point2):
    return (point1[0]+point2[0], point1[1]+point2[1])

def point_scale(point, scale):
    return (point[0]*scale, point[1]*scale)

def point_norm(point):
    return math.sqrt(point[0]**2 + point[1]**2)

def in_range(point):
    return point[0] >= 0 and point[1] >= 0 and point[0] < crop_size and point[1] < crop_size

def convert_json(src_fname, dst_fname):
    with open(src_fname, 'r') as f:
        labels = json.load(f)

    with open(dst_fname, 'w') as f:
        for stern, bow, port, starboard in labels:
            center = get_center([stern, bow])
            # Get half width/length vectors to convert stern/bow/port/starboard centers into vessel rotated rectangle.
            dx_vector = point_scale(point_subtract(starboard, port), 0.5)
            dy_vector = point_scale(point_subtract(bow, stern), 0.5)
            # Add padding.
            dx_vector = point_add(dx_vector, point_scale(dx_vector, padding/point_norm(dx_vector)))
            dy_vector = point_add(dy_vector, point_scale(dy_vector, padding/point_norm(dy_vector)))
            # Get rectangle corners.
            corner1 = point_add(center, point_add(dx_vector, dy_vector))
            corner2 = point_add(center, point_add(dx_vector, point_scale(dy_vector, -1)))
            corner3 = point_add(center, point_add(point_scale(dx_vector, -1), point_scale(dy_vector, -1)))
            corner4 = point_add(center, point_add(point_scale(dx_vector, -1), dy_vector))
            corners = [corner1, corner2, corner3, corner4]
            if all([not in_range(corner) for corner in corners]):
                continue
            f.write('{} {} {} {} {} {} {} {} vessel 0\n'.format(
                corner1[0], corner1[1],
                corner2[0], corner2[1],
                corner3[0], corner3[1],
                corner4[0], corner4[1],
            ))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", help="Source folder")
    parser.add_argument("--dst", help="Destination folder")
    args = parser.parse_args()

    src_dir = args.src
    dst_img_dir = os.path.join(args.dst, 'images')
    dst_npy_dir = os.path.join(args.dst, 'imagesNpy')
    dst_label_dir = os.path.join(args.dst, 'labelTxt')

    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_npy_dir, exist_ok=True)
    os.makedirs(dst_label_dir, exist_ok=True)

    for fname in os.listdir(src_dir):
        prefix = fname.split('.')[0]

        if fname.endswith('.json'):
            convert_json(os.path.join(src_dir, fname), os.path.join(dst_label_dir, prefix+'.txt'))

        elif fname.endswith('.png'):
            shutil.copyfile(os.path.join(src_dir, fname), os.path.join(dst_img_dir, prefix+'.png'))

        elif fname.endswith('.npy'):
            shutil.copyfile(os.path.join(src_dir, fname), os.path.join(dst_npy_dir, prefix+'.npy'))
