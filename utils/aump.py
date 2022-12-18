"""Реализация алгоритма AUMP."""
from __future__ import annotations

import numpy as np
import numpy.linalg as lin
from math import sqrt

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Tuple


def aump(input_image: np.ndarray, m: int, d: int) -> float:
    """
    AUMP НЗБ детектор.

    :param input_image - входное изображение.
    :param m - размер окна в пикселях.
    :param d = q - 1 - степень полинома для прогонки.

    :return beta - статистика обнаружения (beta = \hat{\Lambda}^{\star{X}}).
    """
    image = input_image.astype(np.float64)
    image_pred, _, weights = predict_aump(image, m, d)  # полиномиальное предсказание, веса
    residual = image - image_pred  # вычет
    X_bar = (image + 1) - 2 * np.mod(image, 2)  # переворот
    beta = np.sum(np.multiply(np.multiply(weights, (image - X_bar)), residual))  # статистика обнаружения
    return beta


def predict_aump(
    image_input: np.ndarray,
    m: int,
    d: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Предсказатель пикселей путем подгонки локального полинома степени d = q - 1.
    m должно разделить количество пикселей в строке.

    :param image_input - изображение для предсказания.
    :param m - размер окна в пикселях.
    :param d = q - 1 - степень полинома для прогонки.

    :return image_pred - предсказанное изображение.
    :return sigma - локальные различия.
    :return weights - веса.
    """
    input_image_rows, input_image_cols = image_input.shape

    sig_th = 1
    q = d + 1
    k_n = (input_image_rows * input_image_cols) // m  # количество блоков в m пикселях
    y = np.zeros((m, k_n))  # y хранит все блоки пикселей в колонках
    S = np.zeros_like(image_input)  # межпиксельное различие
    image_pred = np.zeros_like(image_input)  # предсказанное изображение
    flatten_image = image_input.flatten("F")

    H = np.zeros((m, q))
    x1: np.ndarray = np.arange(1, m + 1) / m

    for i in range(0, q):
        H[:, i] = np.power(x1.T, i)

    for i in range(1, m + 1):  # формируем блоки пикселей
        # aux = image_input[:, i::m]
        if i != m:
            aux = flatten_image[(i - 1) * k_n:i * k_n]
        else:
            aux = flatten_image[(i - 1) * k_n:]
        y[i-1] = aux

    p = lin.lstsq(H, y)[0]
    y_pred = H @ p

    for i in range(0, m):
        image_pred[:, i::m] = np.reshape(y_pred[i, :], image_input[:, i::m].shape)

    sig2 = np.sum(np.power(y - y_pred, 2), axis=0) / (m - q)
    sig2 = np.maximum((sig_th ** 2) * np.ones_like(sig2.shape), sig2)

    Sy = np.ones((m, 1)) * sig2

    for i in range(0, m):
        S[:, i::m] = np.reshape(Sy[i, :], image_input[:, i::m].shape)

    s_n2 = k_n / np.sum(1/sig2)

    S[S == 0] = 1 # защита от деления на ноль

    w = sqrt(s_n2 / (k_n * (m-q))) / S

    return image_pred, S, w


def predict_aump_old(
    image_input: np.ndarray,
    m: int,
    d: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Предсказатель пикселей путем подгонки локального полинома степени d = q - 1.
    m должно разделить количество пикселей в строке.

    :param image_input - изображение для предсказания.
    :param m - размер окна в пикселях.
    :param d = q - 1 - степень полинома для прогонки.

    :return image_pred - предсказанное изображение.
    :return sigma - локальные различия.
    :return weights - веса.
    """
    input_image_rows, input_image_cols = image_input.shape

    signum_threshold = 1  # порог численной сходимости
    q = d + 1  # количество параметров на каждый блок
    k_n = (input_image_rows * input_image_cols) // m  # количество блоков в m пикселях
    y = np.zeros((m, k_n))  # y хранит все блоки пикселей в колонках
    sigma = np.zeros_like(image_input)  # межпиксельное различие
    image_pred = np.zeros_like(image_input)  # предсказанное изображение
    flatten_image = image_input.flatten("F")  # колонно-ориентированное сплющивание

    vandermonde_matrix = np.zeros((m, q))  # матрица Вандермонда
    x_1: np.ndarray = np.arange(1, m + 1) / m
    for i in range(1, q + 1):
        vandermonde_matrix[:, i - 1] = np.power(x_1.T, i - 1)

    for i in range(1, m + 1):  # формируем блоки пикселей
        if i != m:
            aux = flatten_image[(i - 1) * k_n:i * k_n]
        else:
            aux = flatten_image[(i - 1) * k_n:]
        y[i-1] = aux

    polynomial = np.zeros_like(y).astype(np.float64)
    operations = k_n // q
    # поблочное деление матрицы блоков пикселей на матрицу Вандермонда
    for num_operation in range(1, operations + 1):
        y_part = y[:, (num_operation - 1) * q:num_operation * q]
        aux = y_part / vandermonde_matrix
        polynomial[:, (num_operation - 1) * q:num_operation * q] = aux

    polynomial = np.nan_to_num(polynomial)
    y_pred = np.zeros_like(y).astype(np.float64)
    # поблочное умножение матрицы полиномов на матрицу Вандермонда
    for num_operation in range(1, operations + 1):
        polynomial_part = polynomial[:, (num_operation - 1) * q:num_operation * q]
        aux = vandermonde_matrix * polynomial_part
        y_pred[:, (num_operation - 1) * q:num_operation * q] = aux

    flatten_y_pred = np.nan_to_num(y_pred).flatten()  # построчно-ориентированное сплющивание
    for i in range(1, input_image_cols + 1):
        if i != input_image_cols:
            aux = flatten_y_pred[(i - 1) * input_image_rows:i * input_image_rows]
        else:
            aux = flatten_y_pred[(i - 1) * input_image_rows:]
        image_pred[:, i-1] = aux

    sigma_2 = np.sum((y - y_pred) ** 2, axis=0) / (m - q)  # различие в k-м блоке
    le01 = signum_threshold ** 2 * np.ones(sigma_2.shape)
    sigma_2 = le01 if np.all(le01 >= sigma_2) else sigma_2

    sigma_y = np.ones((m, 1)) * sigma_2  # различие во всех пикселях
    flatten_sigma_y = sigma_y.flatten()
    for i in range(1, input_image_cols + 1): # преобразуем к размеру входного изображения
        if i != input_image_cols:
            aux = flatten_sigma_y[(i - 1) * input_image_rows:i * input_image_rows]
        else:
            aux = flatten_sigma_y[(i - 1) * input_image_rows:]
        sigma[:, i-1] = aux

    sigma_n2 = k_n / np.sum(1 / sigma_2)
    weights = np.sqrt(sigma_n2 / (k_n * (m - q))) / sigma  # веса

    return image_pred, sigma, weights


if __name__ == "__main__":
    import cv2

    IMAGE_PATH = "../images/Image00001.tif"
    image_sample = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

    print(aump(image_sample, 128, 7))
