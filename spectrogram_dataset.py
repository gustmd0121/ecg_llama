import numpy as np
import os
import torch
import wfdb
import librosa
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

class ECGProcessor:
    def __init__(self, image_processor):
        self.image_processor = image_processor

    def generate_spectrogram(self, ecg_signal):
        if ecg_signal is None or np.isnan(ecg_signal).any():
            return self.create_placeholder_image()

        spectrograms = []
        for channel in ecg_signal:
            channel = channel.numpy()  # Convert to numpy array
            S = librosa.stft(channel, n_fft=2048, hop_length=512)
            S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
            spectrograms.append(S_db)

        S_db_concatenated = np.concatenate(spectrograms, axis=-1)
        min_val = S_db_concatenated.min()
        max_val = S_db_concatenated.max()
        if max_val != min_val:
            S_db_concatenated = (S_db_concatenated - min_val) / (max_val - min_val)
        else:
            S_db_concatenated = np.zeros_like(S_db_concatenated)

        cm = plt.get_cmap('viridis')
        spectrogram_colormap = cm(S_db_concatenated)
        spectrogram_image = (spectrogram_colormap[..., :3] * 255).astype(np.uint8)

        buf = BytesIO()
        Image.fromarray(spectrogram_image).save(buf, format='PNG')
        buf.seek(0)
        spectrogram_image = Image.open(buf).convert('RGB')
        return spectrogram_image

    def create_placeholder_image(self):
        placeholder_image = np.zeros((256, 256, 3), dtype=np.uint8)
        return Image.fromarray(placeholder_image)

    def process_ecg_file(self, ecg_file_base, output_dir):
        try:
            ecg, _ = wfdb.rdsamp(ecg_file_base)
            ecg = torch.from_numpy(ecg.T)
        except Exception as e:
            ecg = None

        spectrogram_image = self.generate_spectrogram(ecg)

        relative_path = os.path.relpath(ecg_file_base, input_dir)
        output_path = os.path.join(output_dir, relative_path + '.png')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        spectrogram_image.save(output_path)

def main():
    global input_dir
    input_dir = '/nfs_data_storage/mimic-iv-ecg/files'
    output_dir = '/nfs_edlab/hschung/ecg_plots_spectrograms/files/'
    os.makedirs(output_dir, exist_ok=True)

    image_processor = None  # Initialize your image processor here
    ecg_processor = ECGProcessor(image_processor)

    # Process all ECG files in the input directory
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.dat'):
                ecg_file_base = os.path.join(root, file[:-4])
                ecg_processor.process_ecg_file(ecg_file_base, output_dir)

if __name__ == "__main__":
    main()