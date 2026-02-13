from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CaptchaFetchResult:
    image_bytes: Optional[bytes]
    error: Optional[str]

