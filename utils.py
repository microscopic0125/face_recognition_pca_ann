import os, cv2, numpy as np # pyright: ignore[reportMissingImports]

IMG_SIZE = (64, 64)
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.pgm', '.tif', '.tiff'}


def _is_image_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS


def load_orl_dataset(base_path='data'):
    X = []
    y = []
    label_map = {}

    if not os.path.isdir(base_path):
        raise ValueError(f'Dataset path does not exist: {base_path}')

    for root, _, files in os.walk(base_path):
        if not files:
            continue

        label_name = os.path.basename(root)
        if label_name == os.path.basename(base_path):
            continue

        for f in sorted(files):
            if not _is_image_file(f):
                continue
            fp = os.path.join(root, f)
            img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, IMG_SIZE)
            if label_name not in label_map:
                label_map[label_name] = len(label_map)
            X.append(img.flatten() / 255.0)
            y.append(label_map[label_name])

    if len(X) == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int32)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)