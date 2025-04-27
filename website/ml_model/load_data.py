import glob
import os
import shutil

import kagglehub
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

matplotlib.use('Agg')


class LoadDataset:

    def __init__(self, class_names):
        self.class_names = class_names
        self.data = None
        self.labels = None
        self.wavelength = None

    def load_mat_data(self, path):
        mat_files = glob.glob(os.path.join(path, "*.mat"))

        self.data = None
        self.labels = None

        for mat_file in mat_files:
            mat = sio.loadmat(mat_file)
            for key, value in mat.items():
                if isinstance(value, np.ndarray):
                    if value.ndim == 3 and self.data is None:
                        self.data = value
                    elif value.ndim == 2 and np.issubdtype(value.dtype, np.integer) and self.labels is None:
                        self.labels = value

        if self.data is None or self.labels is None:
            raise ValueError("Could not automatically find data and label arrays.")

    def load_numpy_data(self, path):
        npy_data = glob.glob(os.path.join(path, '*.npy'))

        self.data = None
        self.labels = None

        for file in npy_data:
            arr = np.load(file)
            if arr.ndim == 3 and self.data is None:
                self.data = arr
            elif arr.ndim == 2 and np.issubdtype(arr.dtype, np.integer) and self.labels is None:
                self.labels = arr

        if self.data is None or self.labels is None:
            raise ValueError('No data loaded')

    def load_npz_data(self, path):
        npz_data = np.load(path)

        self.data = None
        self.labels = None

        for key in npz_data:
            arr = npz_data[key]
            if arr.ndim == 3 and self.data is None:
                self.data = arr
            elif arr.ndim == 2 and np.issubdtype(arr.dtype, np.integer) and self.labels is None:
                self.labels = arr

        if self.data is None or self.labels is None:
            raise ValueError('No data loaded')

    @staticmethod
    def unarchive_data(archive_format, path):
        filename = os.path.join('Ground_Detection_App', 'website', 'static', 'uploads', path)
        extract_dir = os.path.join('Ground_Detection_App', 'website', 'static', 'uploads')

        shutil.unpack_archive(filename, extract_dir, archive_format)

    def load_data(self, path):

        if "static\\uploads\\" not in path:
            path = eval(path)

        if os.path.isdir(path):
            if any(f.endswith('.mat') for f in os.listdir(path)):
                self.load_mat_data(path)
            elif any(f.endswith('.npy') for f in os.listdir(path)):
                self.load_numpy_data(path)
            else:
                raise ValueError("Unsupported files in folder.")
        elif path.endswith('.npz'):
            self.load_npz_data(path)
        else:
            raise ValueError("Unsupported file format.")

        self.wavelength = np.linspace(430, 860, self.data.shape[2])

        return self.wavelength, self.labels, self.data
