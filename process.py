from argparse import Action, ArgumentParser, ArgumentError, Namespace
import cv2
import itertools
from matplotlib import pyplot as plt
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from typing import Callable, Iterable, Optional


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


def add_mirror_argument(parser: ArgumentParser) -> None:
    mirror_argument = parser.add_argument(
        "-r", "--mirror", help="unit normal of mirror in scene in the form r_x,r_y,r_z"
    )
    mirror_argument.type = make_parse_floats(mirror_argument)


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
        type=Path,
    )
    light_argument = default_parser.add_argument(
        "-l",
        "--light",
        action="append",
        help="lighting intensity and direction in the form l_x,l_y,l_z",
        required=True,
    )
    light_argument.type = make_parse_floats(light_argument)
    default_parser.add_argument("-m", "--mask", help="path to object mask", type=Path)
    add_mirror_argument(default_parser)
    diligent_parser = subparsers.add_parser(
        "diligent", help="load images and lighting from a DiLiGenT directory"
    )
    diligent_parser.add_argument(
        "directory", help="path to DiLiGent directory", type=Path
    )
    diligent_parser.add_argument(
        "-i",
        "--indices",
        help="indices or ranges of indices of images to use",
        type=parse_ranges,
    )
    add_mirror_argument(diligent_parser)
    return parser.parse_args()


def load_images(
    image_paths: list[Path], mask_path: Optional[Path]
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


def calculate_normals(
    images: NDArray[np.single], lights: NDArray[np.single]
) -> tuple[NDArray[np.single], NDArray[np.single], NDArray[np.single]]:
    pinv = np.linalg.pinv(np.array(lights))
    normals_and_albedos = np.tensordot(images, pinv, ([2], [1]))
    albedos = np.linalg.norm(normals_and_albedos, axis=2)
    return (
        np.nan_to_num(normals_and_albedos / albedos[:, :, np.newaxis]),
        albedos,
        np.sum(
            np.square(
                images
                - np.moveaxis(
                    np.tensordot(lights, normals_and_albedos, ([1], [2])), 0, 2
                )
            ),
            2,
        ),
    )


def select_normals_and_albedos(
    images: NDArray[np.single],
    candidate_lights: Iterable[
        tuple[NDArray[np.single], NDArray[np.single], NDArray[np.single]]
    ],
) -> tuple[
    NDArray[np.single],
    NDArray[np.single],
    list[NDArray[np.single]],
    list[NDArray[np.single]],
]:
    candidate_normals = []
    candidate_albedos = []
    candidate_errors = []
    for light_combination in candidate_lights:
        normals, albedos, errors = calculate_normals(
            images, np.array(light_combination)
        )
        candidate_normals.append(normals)
        candidate_albedos.append(albedos)
        candidate_errors.append(errors)
    candidate_normals.pop()
    candidate_albedos.pop()
    candidate_errors.pop()
    indices = np.argmin(candidate_errors, axis=0)
    return (
        np.take_along_axis(
            np.array(candidate_normals),
            indices[np.newaxis, :, :, np.newaxis],
            0,
        )[0],
        np.take_along_axis(np.array(candidate_albedos), indices[np.newaxis], 0)[0],
        candidate_normals,
        candidate_albedos,
    )


def prepare_normals(normals: NDArray[np.single]) -> NDArray[np.single]:
    return (normals / 2 + 0.5) * np.logical_and.reduce(normals != 0, axis=2)[
        :, :, np.newaxis
    ]


arguments = parse_arguments()
if "directory" in arguments:
    with open(arguments.directory / "filenames.txt") as image_names:
        mask_path = arguments.directory / "mask.png"
        images = load_images(
            [
                arguments.directory / image_name.rstrip()
                for image_index, image_name in enumerate(image_names)
                if arguments.indices is None or image_index in arguments.indices
            ],
            mask_path if mask_path.exists() else None,
        )
    index_slice = arguments.indices if arguments.indices is not None else slice(None)
    direct_lights = np.loadtxt(arguments.directory / "light_directions.txt")[
        index_slice
    ]
    light_intensities_path = arguments.directory / "light_intensities.txt"
    if light_intensities_path.exists():
        direct_lights *= cv2.cvtColor(
            np.loadtxt(light_intensities_path, dtype=np.single)[
                np.newaxis, index_slice
            ],
            cv2.COLOR_RGB2GRAY,
        ).T
else:
    if len(arguments.image) != len(arguments.light):
        raise ArgumentError(None, "number of lights must match number of images")
    images = load_images(arguments.image, arguments.mask)
    direct_lights = np.array(arguments.light)
if arguments.mirror is not None:
    mirrored_lights = direct_lights - 2 * np.outer(
        direct_lights @ arguments.mirror, arguments.mirror
    )
    direct_and_mirrored_lights = direct_lights + mirrored_lights
    (
        normals,
        _albedos,
        candidate_normals,
        _candidate_albedos,
    ) = select_normals_and_albedos(
        images,
        itertools.product(
            *zip(direct_lights, mirrored_lights, direct_and_mirrored_lights)
        ),
    )
    figure, axes = plt.subplots(5, 6, layout="tight")
    figure.set(figheight=11, figwidth=16)
    for axis in axes.flat:
        axis.axis(False)
    for axis, candidate_normal in zip(axes.flat, candidate_normals):
        axis.imshow(prepare_normals(candidate_normal))
else:
    normals, _albedos, _errors = calculate_normals(images, direct_lights)
if "directory" in arguments and (arguments.directory / "normal.txt").exists():
    figure, (axis_gt, axis_computed) = plt.subplots(1, 2)
    figure.set(figwidth=11)
    axis_gt.axis(False)
    axis_gt.imshow(
        prepare_normals(
            np.loadtxt(arguments.directory / "normal.txt").reshape(normals.shape)
        )
    )
else:
    figure, axis_computed = plt.subplots(1, 1)
    figure.set(figwidth=6)
axis_computed.axis(False)
axis_computed.imshow(prepare_normals(normals))
plt.show()
