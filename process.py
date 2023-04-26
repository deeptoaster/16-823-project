from argparse import Action, ArgumentParser, ArgumentError, Namespace
import cv2
from matplotlib import pyplot as plt
import numpy as np
from numpy.typing import NDArray
from pathlib import PurePath
from typing import Callable, Optional


def make_parse_floats(action: Action) -> Callable[[str], list[float]]:
    def parse_floats(floats: str) -> list[float]:
        floats_split = floats.split(",")
        if len(floats_split) != 3:
            raise ArgumentError(action, "expected triple of floats")
        return [float(float_str) for float_str in floats_split]

    return parse_floats


def parse_ranges(ranges: str) -> list[int]:
    splits = (range_str.split("-") for range_str in ranges.split(","))
    return sum([list(range(int(split[0]), int(split[-1]) + 1)) for split in splits], [])


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(required=True)
    default_parser = subparsers.add_parser(
        "default", help="load images and lighting by command line"
    )
    default_parser.add_argument(
        "image",
        help="path to lighting data as a single multi-channel image or multiple intensity images",
        nargs="+",
        type=PurePath,
    )
    light_argument = default_parser.add_argument(
        "-l",
        "--light",
        action="append",
        help="lighting intensity and direction in the form l_x,l_y,l_z",
        required=True,
    )
    light_argument.type = make_parse_floats(light_argument)
    default_parser.add_argument(
        "-m", "--mask", help="path to object mask", type=PurePath
    )
    mirror_argument = default_parser.add_argument(
        "-r",
        "--mirror",
        help="coefficients representing mirror plane a x + b x + c x + d = 0 in the form a,b,c,d",
    )
    mirror_argument.type = make_parse_floats(mirror_argument)
    diligent_parser = subparsers.add_parser(
        "diligent", help="load images and lighting from a DiLiGenT directory"
    )
    diligent_parser.add_argument(
        "directory", help="path to DiLiGent directory", type=PurePath
    )
    diligent_parser.add_argument(
        "-i",
        "--indices",
        help="indices or ranges of indices of images to use",
        type=parse_ranges,
    )
    return parser.parse_args()


def load_images(
    image_paths: list[PurePath], mask_path: Optional[PurePath]
) -> NDArray[np.single]:
    mask = (
        cv2.imread(str(mask_path), flags=cv2.IMREAD_GRAYSCALE)
        if mask_path is not None
        else np.ones((1, 1))
    )
    images = [
        cv2.imread(str(image_path), flags=cv2.IMREAD_COLOR)
        for image_path in image_paths
    ]
    return (
        np.dstack(
            [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]
            if len(image_paths) >= 3
            else images
        )
        * (mask != 0)[:, :, np.newaxis]
    )


arguments = parse_arguments()
if "directory" in arguments:
    with open(arguments.directory / "filenames.txt") as image_names:
        images = load_images(
            [
                arguments.directory / image_name.rstrip()
                for image_index, image_name in enumerate(image_names)
                if arguments.indices is None or image_index in arguments.indices
            ],
            arguments.directory / "mask.png",
        )
    lights = (
        np.loadtxt(arguments.directory / "light_directions.txt")[
            arguments.indices or slice(None)
        ]
        * cv2.cvtColor(
            np.loadtxt(arguments.directory / "light_intensities.txt", dtype=np.single)[
                np.newaxis, arguments.indices or slice(None)
            ],
            cv2.COLOR_RGB2GRAY,
        ).T
    )
else:
    if len(arguments.image) != len(arguments.light):
        raise ArgumentError(None, "number of lights must match number of images")
    images = load_images(arguments.image, arguments.mask)
    lights = np.array(arguments.light)
lights_pinv = np.linalg.inv(lights.T @ lights) @ lights.T
normals_with_albedos = np.tensordot(images, lights_pinv, ([2], [1]))
albedos = np.linalg.norm(normals_with_albedos, axis=2)
normals = normals_with_albedos / albedos[:, :, np.newaxis]
if "directory" in arguments:
    figure, (axis_gt, axis_computed) = plt.subplots(1, 2)
    figure.set(figwidth=11)
    axis_gt.axis(False)
    axis_gt.imshow(
        np.loadtxt(arguments.directory / "normal.txt").reshape(normals.shape)
    )
else:
    figure, axis_computed = plt.subplots(1, 1)
    figure.set(figwidth=6)
axis_computed.axis(False)
axis_computed.imshow(normals)
plt.show()
