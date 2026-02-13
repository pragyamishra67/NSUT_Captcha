import random
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from captcha_predictor.config.settings import (
    DEFAULT_REQUEST_ACCEPT,
    IMS_BASE_URL,
    IMS_LEGACY_LOGIN_URL,
    IMS_LOGIN_URL,
)


class CaptchaFetcher:
    def __init__(self):
        self.session = requests.Session()
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
        ]
        self._configure_session()

    def _configure_session(self):
        headers = {
            "Accept": DEFAULT_REQUEST_ACCEPT,
            "Referer": IMS_BASE_URL,
            "User-Agent": random.choice(self.user_agents),
        }
        self.session.headers.update(headers)

    def fetch_single_image(self):
        try:
            self.session.get(IMS_BASE_URL)
            self.session.get(IMS_LEGACY_LOGIN_URL)
            login_url = IMS_LOGIN_URL
            response = self.session.get(login_url)
            soup = BeautifulSoup(response.text, "html.parser")
            captcha_img = soup.find("img", {"id": "captchaimg"})
            if not captcha_img:
                return None, "Captcha element not found."
            captcha_url = urljoin(login_url, captcha_img["src"])
            img_response = self.session.get(
                captcha_url, headers={"Referer": login_url}
            )
            return img_response.content, None
        except Exception as e:
            return None, str(e)

