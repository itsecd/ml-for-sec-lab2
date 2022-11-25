"""Реализация НЗБ-встраивания."""
from __future__ import annotations

import itertools
import random
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import List, Tuple


def get_binary_plane(image: np.ndarray, binary_num: int) -> np.ndarray:
    """Выдача битовой плоскости определенного порядка"""
    num = 2 ** (binary_num - 1)
    return np.bitwise_and(image, num)


def set_binary_value(pixel: int, value: int, binary_num: int, image_bits: int = 8) -> int:
    """Заменяет значение в бите."""
    max_bits = 2 ** image_bits - 1
    copy_number = pixel & (max_bits ^ (1 << (binary_num - 1)))
    new_pixel_value = copy_number + (value << (binary_num - 1))
    return new_pixel_value


def white_noise(q: float, n1: int, n2: int) -> np.ndarray:
    """Генерация белого шума."""
    nb = int(n1 * n2 * q)
    return np.random.randint(0, 2, size=(nb, ), dtype=np.uint8)


def get_random_image_coordinates(image: np.ndarray, size: int) -> List[Tuple[int, int]]:
    """Генерация случайных координат изображений."""
    n_rows, n_cols = image.shape
    pixels_combinations = list(itertools.product(range(n_rows), range(n_cols)))
    return random.sample(pixels_combinations, size)


def lsb_replacement(image: np.ndarray, q: float, binary_num: int = 1) -> np.ndarray:
    """НЗБ-встраивание по псевдослучайным координатам."""
    noise_image: np.ndarray = image.copy()
    rows, cols = noise_image.shape
    noise_vector = white_noise(q, rows, cols)
    noise_vector_len = len(noise_vector)
    random_coordinates = get_random_image_coordinates(noise_image, noise_vector_len)
    for index, pixel_coordinate in enumerate(random_coordinates):
        i, j = pixel_coordinate
        noise_image[i][j] = set_binary_value(noise_image[i][j], noise_vector[index], binary_num)
    return noise_image
