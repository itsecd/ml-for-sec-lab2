# from multiprocessing import current_process
from multiprocessing.pool import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

DATASETS_DIR = 'DATASETS'
GT_IMAGES_DIR = 'BOWS2'
PROCESSES_COUNT = 10


def get_pseudorandom_coord_list(image_shape, seed=None):
    x_shape, y_shape = image_shape
    coordinate_pairs = np.stack(np.meshgrid(np.arange(x_shape), np.arange(y_shape)), axis=-1).reshape(-1, 2)
    if seed:
        np.random.seed(seed)
    return np.random.permutation(coordinate_pairs)


def get_binary_plane(image, plane_number):
    # plane number is indexing from 1

    return np.bitwise_and(image, 1 << (plane_number - 1)).astype(np.uint8)


def generate_payload(N1, N2, q):
    """(White noise)"""
    Nb = int(N1 * N2 * q)
    return np.random.randint(0, 1 + 1, size=(Nb,), dtype=np.uint8)


def lsb_replacement(image_, bit_num, position_pairs, payload):
    image = image_.copy()
    for i, payload_bit in enumerate(payload):
        x, y = position_pairs[i]

        # example: let bit_num = 3
        # np.binary_repr(1 << (bit_num - 1))         = '00000100'
        # np.binary_repr(255)                        = '11111111'
        # np.binary_repr(255 ^ (1 << (bit_num - 1))) = '11111011'
        image[x, y] &= (255 ^ (1 << (bit_num - 1)))  # copy all except bit_num'th bit, bit_num'th bit equals 0
        image[x, y] += payload_bit << (bit_num - 1)  # adding '00000*00' where * is payload_bit
    return image


def get_hist_t(hist_e) -> np.ndarray:
    assert len(hist_e) % 2 == 0
    return np.array([(hist_e[2 * (i // 2)] + hist_e[2 * (i // 2) + 1]) / 2 for i in range(len(hist_e))])


def get_chi_components(hist_e, hist_t) -> np.ndarray:
    diffs_chi = (hist_e - hist_t) ** 2
    non_zero_mask = hist_t > 0
    diffs_chi[non_zero_mask] /= hist_t[non_zero_mask]
    return diffs_chi


def get_features(image):
    hist_e = np.histogram(image, bins=256, range=(0, 256))[0]
    hist_t = get_hist_t(hist_e)

    # i-th and (i+1)-th features are equal because they hist_e[i] and hist_e[i+1] are equally spaced
    # from hist_t[i] = hist_t[i+1]
    # so every other one can be eliminated
    return get_chi_components(hist_e, hist_t)[::2]


def process_one_file(params):
    # print(params)

    filename, need_noise, q, coord_seed = params

    # print(f'{current_process()} is processing file {filename.stem}')

    im = Image.open(filename)
    image = np.array(im)

    if need_noise:
        image = lsb_replacement(image,
                                1,
                                get_pseudorandom_coord_list(image.shape, seed=coord_seed),
                                generate_payload(*image.shape, q))

    features = get_features(image)

    return list(features) + [1 if need_noise else 0]


def prepare_dataset(all_files_list: list, q, no_noise_size=0.5, coord_seed=None) -> pd.DataFrame:
    # dataset = pd.DataFrame(columns=(list(map(str, range(256)))+['noised']))

    # need_noise = k > len(all_files_list) * no_noise_size

    need_noise_arr = np.zeros(shape=(len(all_files_list),), dtype=bool)
    need_noise_arr[int(len(all_files_list) * no_noise_size):] = True

    rows = []

    with Pool(processes=PROCESSES_COUNT) as p:
        rows = p.map(process_one_file, list(zip(all_files_list,
                                                list(need_noise_arr),
                                                [q] * len(all_files_list),
                                                [coord_seed] * len(all_files_list))))

    dataset = pd.DataFrame(columns=(list(map(str, range(len(rows[0]) - 1))) + ['noised']), data=rows)

    # k = 0
    # for filename in tqdm(all_files_list):
    #     k += 1
    #     dataset.loc[k - 1] = process_one_file((filename, need_noise))

    dataset = dataset.astype({'noised': int})
    return dataset


if __name__ == "__main__":

    all_files = list(Path(GT_IMAGES_DIR).glob('*.tif'))

    Path(DATASETS_DIR).mkdir(parents=True, exist_ok=True)

    for q_percent in tqdm(range(20, 100 + 1 , 10)):
        q = q_percent / 100
        df = prepare_dataset(all_files, q=q)
        df.to_csv(f'{DATASETS_DIR}/q={q_percent}%.csv', index=False)
