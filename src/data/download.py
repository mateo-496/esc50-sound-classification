import requests
import zipfile
import io
import os
import shutil
from pathlib import Path
from tqdm import tqdm
import sys

from src.config.config import DownloadConfig

config = DownloadConfig()

class ESC50Downloader:
    def __init__(self, repo_url: str = config.repo_url, repo_dst_dir: str = config.repo_dst_dir) -> None:
        self.repo_url = repo_url
        self.repo_dst_dir = Path(repo_dst_dir)
        self.audio_dst_dir = config.audio_dst_dir
        self.paths_to_delete = config.paths_to_delete
        self.extracted_dir = config.extracted_dir 
        self.audio_src_dir = config.audio_src_dir 

    def download_and_extract(self) -> None:
        os.makedirs(self.repo_dst_dir, exist_ok=True)
        print(f"Downloading from {self.repo_url}")

        response = requests.get(self.repo_url, stream=True)
        response.raise_for_status()

        total = int(response.headers.get("content-length", 0))
        buffer = io.BytesIO()

        with tqdm(total=total, unit="B", unit_scale=True, desc="Downloading") as bar:
            for chunk in response.iter_content(chunk_size=8192):
                buffer.write(chunk)
                bar.update(len(chunk))

        buffer.seek(0)
        with zipfile.ZipFile(buffer) as z:
            print(f"Extracting to {self.repo_dst_dir}")
            z.extractall(self.repo_dst_dir)
        print("Done extracting.")

    def clean_files(self) -> None:
        for f in self.paths_to_delete:
            path = os.path.join(self.extracted_dir, f)
            if os.path.isfile(path):
                os.remove(path)
                print(f"Deleted file: {path}")
            elif os.path.isdir(path):
                shutil.rmtree(path)
                print(f"Deleted directory: {path}")

    def move_audio_files(self) -> None:
        os.makedirs(self.audio_dst_dir, exist_ok=True)
        print(f"Moving audio files from {self.audio_src_dir} to {self.audio_dst_dir}")
        
        for filename in os.listdir(self.audio_src_dir):
            src_file = os.path.join(self.audio_src_dir, filename)
            dst_file = os.path.join(self.audio_dst_dir, filename)
            if os.path.isfile(src_file):
                shutil.move(src_file, dst_file)
        print(f"Moved all audio files to {self.audio_dst_dir}")

    def download_clean(self) -> None:
        self.download_and_extract()
        self.clean_files()
        self.move_audio_files()
