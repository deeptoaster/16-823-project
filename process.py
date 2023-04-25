from argparse import ArgumentParser, Namespace
import cv2
import numpy as np
from pathlib import PurePath


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        "--image",
        help="path to lighting data as a single multi-channel image or multiple intensity images",
        nargs="+",
        required=True,
        type=PurePath,
    )
    parser.add_argument(
        "-l",
        "--light",
        help="lighting intensity and direction in the form l_x,l_y,l_z",
        nargs="*",
        type=parse_floats,
    )
    parser.add_argument(
        "-m", "--mask", help="path to object mask", required=True, type=PurePath
    )
    parser.add_argument(
        "-r",
        "--mirror",
        help="coefficients representing mirror plane a x + b x + c x + d = 0 in the form a,b,c,d",
        type=parse_floats,
    )
    return parser.parse_args()


def parse_floats(floats: str) -> list[float]:
    return [float(float_str) for float_str in floats.split(",")]


arguments = parse_arguments()
mask = cv2.imread(str(arguments.mask), flags=cv2.IMREAD_GRAYSCALE)
images = np.dstack(
    [
        cv2.imread(
            str(image_path),
            flags=cv2.IMREAD_GRAYSCALE
            if len(arguments.image) >= 3
            else cv2.IMREAD_COLOR,
        )
        for image_path in arguments.image
    ]
)
