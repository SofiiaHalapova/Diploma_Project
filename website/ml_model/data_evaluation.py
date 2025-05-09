import os
import random
import string

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from ..ml_model.load_data import LoadDataset

matplotlib.use('Agg')


class DataEvaluation:

    def __init__(self, class_names):
        self.class_names = class_names
        self.data = None  # hyperspectral image cube
        self.labels = None  # the ground truth classifications
        self.wavelength = None  # the wavelength values for each spectral band

    def load_data(self, path):
        ld = LoadDataset(self.class_names)
        self.wavelength, self.labels, self.data = ld.load_data(path)

    @staticmethod
    def save_plot_to_file(plot_type, save_dir):

        random_string = "".join(random.choice(string.ascii_letters + string.digits) for _ in range(5))

        full_img_path = os.path.join(save_dir, f'{plot_type}_{random_string}.png')

        plt.savefig(full_img_path)
        plt.close()

        return f'imgs/{plot_type}_{random_string}.png'

    def create_false_color(self, bands=None):

        if bands is None:
            bands = [30, 20, 10]
        false_color = np.dstack([self.data[:, :, b] for b in bands])
        return (false_color - false_color.min()) / (false_color.max() - false_color.min())

    def plot_class_distribution(self, save_path,):

        if self.labels is None:
            raise ValueError("Labels not defined")

        unique, counts = np.unique(self.labels, return_counts=True)

        plt.figure(figsize=(15, 5))

        bars = plt.bar(
            [self.class_names.get(i, f'Невідомий матеріал ({i})') for i in unique],
            counts
        )

        plt.xticks(rotation=45, ha='right')
        plt.title('Розподіл класів у зображенні')
        plt.xlabel('Клас')
        plt.ylabel('Кількість пікселів')

        for bar in bars:
            plt.text(bar.get_x() + bar.get_width() / 2.,
                     bar.get_height(),
                     f'{int(bar.get_height())}',
                     ha='center', va='bottom'
                     )

        plt.tight_layout()

        static_img_path = os.path.dirname(save_path) if save_path else None

        return self.save_plot_to_file('class_distribution', save_dir=static_img_path)

    # Візуалізація спектральних характеристик
    def plot_sample_spectra(self, save_path, selected_classes=None):

        if selected_classes is None:
            selected_classes = [2, 5, 7]
        plt.figure(figsize=(15, 5))

        for class_id in selected_classes:
            # маска, тобто обрані елементи для класифікації
            mask = self.labels == class_id
            if np.any(mask):
                # обчислення середнього спектру для всіх пікселів, що задовольняють умови, визначені маскою
                mean_spectrum = np.mean(self.data[mask], axis=0)
                plt.plot(self.wavelength, mean_spectrum, label=self.class_names[class_id])

        plt.title('Середні спектри обраних класів')
        plt.xlabel('Довжина хвилі спектру (nm)')
        plt.ylabel('Відбивна здатність матеріалу')
        plt.legend()
        plt.grid(True)

        static_img_path = os.path.dirname(save_path) if save_path else None

        return self.save_plot_to_file('sample_spectra', save_dir=static_img_path)

    def plot_comprehensive_review(self, save_path):
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))

        false_color = self.create_false_color()
        axes[0, 0].imshow(false_color)
        axes[0, 0].set_title('Кольорове зображення')

        # Ground truth
        im = axes[0, 1].imshow(self.labels)
        axes[0, 1].set_title('Фактичні дані грунту')
        plt.colorbar(im, ax=axes[0, 1])

        # Single band visualization
        mid_band = self.data.shape[2] // 2
        im = axes[1, 0].imshow(self.data[:, :, mid_band])
        axes[1, 0].set_title(f'Окремий спектральний канал ({int(self.wavelength[mid_band])}nm)')
        plt.colorbar(im, ax=axes[1, 0])

        spectral_variance = np.std(self.data, axis=2)
        im = axes[1, 1].imshow(spectral_variance)
        axes[1, 1].set_title('Спектральна дисперсія')
        plt.colorbar(im, ax=axes[1, 1])

        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        static_img_path = os.path.dirname(save_path) if save_path else None

        return self.save_plot_to_file('comprehensive_review', save_dir=static_img_path)
