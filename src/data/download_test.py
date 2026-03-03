import requests 
from requests.models import PreparedRequest
import time
import random
import json
import os
from tqdm import tqdm
from typing import Dict, Union, List
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from src.config.config import ScrappingTestConfig

class Downloader:
    def __init__(self, config: ScrappingTestConfig = ScrappingTestConfig()) -> None:
        self.config = config

    def _build_url_request(self, category: str, durations: List[str]) -> str:
        params = self.config.params.copy()
        req = PreparedRequest()
        params = self.config.params.copy()
        params["q"] = category
        params["durations"] = ",".join(durations)
        req.prepare_url(self.config.base_url, params=params)
        return req.url

    def _fetch_mp3_urls(self, url: str) -> List[str]:
        options = webdriver.ChromeOptions()
        options.set_capability("goog:loggingPrefs", {"performance": "ALL"})

        self.driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=options
        )
        self.driver.delete_all_cookies()
        self.driver.get_log("performance")
        self.driver.get(url)
        time.sleep(5)
        logs = self.driver.get_log("performance")
        mp3_urls = []

        for entry in logs:
            message = json.loads(entry["message"])["message"]
            if message["method"] == "Network.responseReceived":
                url = message["params"]["response"]["url"]
                if ".mp3" in url:
                    mp3_urls.append(url)

        mp3_urls = list(set(mp3_urls))

        self.driver.quit()

        return mp3_urls

    def _download_mp3(self, mp3_url: str, save_dir: str) -> bool:
        filename = mp3_url.split("/")[-1]
        filepath = os.path.join(save_dir, filename)

        try:
            with requests.get(mp3_url, headers=self.config.headers, stream=True) as r:
                r.raise_for_status()
                with open(filepath, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            return True

        except Exception as e:
            print(f"Failed to download {mp3_url}: {e}")
            return False

    def download(self) -> None:
        os.makedirs("data/test", exist_ok=True)

        for i, category in enumerate(tqdm(self.config.labels), start=1):
            print(f"Category {i}/{len(self.config.labels)}: {category}...")
            save_dir = f"data/test/{category}"
            os.makedirs(save_dir, exist_ok=True)
                       
            downloaded = 0
            retries = 0
            durations_to_try = self.config.duration_ranges.copy()
            while downloaded < self.config.min_files_per_category and durations_to_try:
                time.sleep(random.uniform(1, 4))

                current_durations = [durations_to_try.pop(0)]
                print(current_durations)
                url = self._build_url_request(category, current_durations)
            
                while retries < self.config.max_retries:
                    try:
                        mp3_urls = self._fetch_mp3_urls(url)
                        break
                    except RuntimeError:
                        retries += 1
                        print(f"Rate limited. Cooling down {self.config.cooldown_seconds}s...")
                        time.sleep(self.config.cooldown_seconds)
                
                # mp3_urls = self._fetch_mp3_urls(url)
                for mp3_url in mp3_urls:
                    if self._download_mp3(mp3_url, save_dir):
                        downloaded+=1
                    if downloaded > self.config.min_files_per_category:
                        break
                print(f"{category}: {downloaded} files downloaded so far")
            print(f"Finished {category}: {downloaded} files total\n")
            sleep = random.uniform(5, 15)
            print(f"Sleeping for {sleep}")
            time.sleep(sleep)
downloader = Downloader()

downloader.download()