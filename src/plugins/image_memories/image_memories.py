from dataclasses import dataclass
from datetime import datetime
import logging
import os
import random
import time

import requests
from requests import Response
from PIL import Image, ImageColor, ImageOps, ImageDraw, ImageFont
from io import BytesIO

from plugins.base_plugin.base_plugin import BasePlugin

from utils.image_utils import pad_image_blur, resize_image


logger = logging.getLogger(__name__)

def add_text_overlay(image: Image.Image, year: int, location: str | None = None, font_size_percent: float = 0.04) -> Image.Image:
    """
    Add text overlay with date (and optionally location) at the bottom of the image.
    Uses a semi-transparent background for better readability.

    Args:
        image: The image to add text to
        year: The year to display
        location: Optional location name to display
        font_size_percent: Font size as a percentage of image height (default 0.04 = 4%)
    """
    # Create a copy to avoid modifying the original
    img = image.copy()

    # Calculate font size relative to image height
    font_size = int(img.height * font_size_percent)

    # Prepare the overlay with alpha channel
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Build text string
    if location:
        text = f"{year} · {location}"
    else:
        text = str(year)

    # Load font
    try:
        # Use Jost-SemiBold from static fonts
        font_path = os.path.join(os.path.dirname(__file__), '..', '..', 'static', 'fonts', 'Jost-SemiBold.ttf')
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        logger.warning(f"Could not load custom font: {e}. Using default font.")
        font = ImageFont.load_default()

    # Get text bounding box
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Position at bottom right corner with padding (also relative to image size)
    padding = int(text_height * 0.5)  # 50% of text height as padding

    # Account for bbox offset (top-left of bbox might not be at 0,0)
    bbox_x_offset = bbox[0]
    bbox_y_offset = bbox[1]

    # Calculate position so the visual text box is padding away from edges
    x = img.width - text_width - padding - bbox_x_offset
    y = img.height - text_height - padding - bbox_y_offset

    # Draw semi-transparent background rectangle
    # Use the actual bbox dimensions for the background
    background_box = [
        x + bbox_x_offset - padding,
        y + bbox_y_offset - padding,
        x + bbox_x_offset + text_width + padding,
        y + bbox_y_offset + text_height + padding
    ]
    draw.rectangle(background_box, fill=(0, 0, 0, 180))  # Black with ~70% opacity

    # Draw white text
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))

    # Composite the overlay onto the original image
    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    img = Image.alpha_composite(img, overlay)

    # Convert back to RGB if needed
    if img.mode == 'RGBA':
        rgb_img = Image.new('RGB', img.size, (255, 255, 255))
        rgb_img.paste(img, mask=img.split()[3])  # Use alpha channel as mask
        return rgb_img

    return img

@dataclass
class Memory:
    id: str
    year: int
    width: int
    height: int
    country: str | None = None
    city: str | None = None
    people: list[str] | None = None


class ImmichProvider:
    def __init__(self, base_url: str, key: str, orientation: str):
        self.base_url = base_url
        self.key = key
        self.orientation = orientation
        self.headers = {"x-api-key": self.key}

    def send_request(self, endpoint: str, params: dict | None = None) -> Response:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                r = requests.get(f"{self.base_url}{endpoint}", headers=self.headers, params=params, timeout=30)
                r.raise_for_status()
                return r
            except TimeoutError:
                logger.warning(f"Request to {endpoint} timed out (attempt {attempt + 1}/{max_retries}). Retrying...")
            # sleep before retrying (exponential backoff)
            time.sleep(2 ** attempt)
        raise RuntimeError(f"Failed to get a successful response from {endpoint} after {max_retries} attempts.")

    def get_todays_memories(self) -> list[dict]:
        params = {
            "for": datetime.now().isoformat(),
            "order": "desc"
        }
        r = self.send_request("/api/memories", params=params)
        return r.json()

    def get_asset_data(self) -> list[Memory]:
        memories = self.get_todays_memories()
        return [Memory(
            id=asset["id"],
            year=memory["data"]["year"],
            width=asset["width"],
            height=asset["height"],
        ) for memory in memories for asset in memory.get("assets", []) if asset["type"] == "IMAGE"]

    def get_image(self, prev_index: int, randomize: bool, require_face: bool = False) -> tuple[Image.Image | None, int, Memory | None]:
        try:
            logger.info("Getting asset IDs for todays memories")
            asset_data = self.get_asset_data()
        except Exception as e:
            logger.error(f"Error grabbing image from {self.base_url}: {e}")
            return None, -1, None

        if not asset_data:
            logger.error("No memories found for today.")
            return None, -1, None

        filtered_assets = [a for a in asset_data if
                           (self.orientation == "horizontal" and a.width >= a.height) or
                            (self.orientation == "vertical" and a.height >= a.width)]

        if not filtered_assets:
            logger.error("No suitable images found for the specified orientation.")
            return None, -1, None

        # Try to find an image with a face if required
        start_index = (prev_index + 1) % len(filtered_assets)

        indices_to_check = [(start_index + i) % len(filtered_assets) for i in range(len(filtered_assets))]
        if randomize:
            indices_to_check = indices_to_check[:-1] # exclude prev_index
            random.shuffle(indices_to_check)
            # make sure to check the previous index last
            indices_to_check.append(prev_index)

        # find the next image with a face, if required
        for current_index in indices_to_check:
            asset = filtered_assets[current_index]

            # get additional metadata for text overlay
            r = self.send_request(f"/api/assets/{asset.id}").json()
            asset.country = r.get("exifInfo", {}).get("country", None)
            asset.city = r.get("exifInfo", {}).get("city", None)
            asset.people = [person.get("name") for person in r.get("people", [])]
            logger.info(f"Downloading image {asset.id}")
            r = self.send_request(f"/api/assets/{asset.id}/original")
            img = Image.open(BytesIO(r.content))
            img = ImageOps.exif_transpose(img)

            if require_face and not asset.people:
                continue

            return img, current_index, asset

        # If no image with face was found after checking all, return None
        logger.warning("No images with faces found in memories")
        return None, -1, None

class ImageMemories(BasePlugin):
    def generate_settings_template(self):
        template_params = super().generate_settings_template()
        template_params['api_key'] = { # type: ignore
            "required": True,
            "service": "Immich",
            "expected_key": "IMMICH_KEY"
        }
        return template_params

    def generate_image(self, settings, device_config):
        orientation = device_config.get_config("orientation")
        img = None
        memory = None
        prev_index = settings.get('_lastMemoryIndex', -1)
        randomize = settings.get('randomize') == 'true'

        match settings.get("memoriesProvider"):
            case "Immich":
                key = device_config.load_env_key("IMMICH_KEY")
                if not key:
                    raise RuntimeError("Immich API Key not configured.")

                url = settings.get('url')
                if not url:
                    raise RuntimeError("URL is required.")

                provider = ImmichProvider(url, key, orientation)
                require_face = settings.get('requireFace') == 'true'
                img, idx, memory = provider.get_image(prev_index, randomize, require_face)
                if not img:
                    raise RuntimeError("Failed to load image, please check logs.")
                settings['_lastMemoryIndex'] = idx

        if img is None:
            raise RuntimeError("Failed to load image, please check logs.")

        if settings.get('padImage') == "true":
            dimensions = device_config.get_resolution()

            if orientation == "vertical":
                dimensions = dimensions[::-1]

            if settings.get('backgroundOption') == "blur":
                img = pad_image_blur(img, dimensions) # type: ignore
            else:
                background_color = ImageColor.getcolor(settings.get('backgroundColor') or (255, 255, 255), "RGB")
                img = ImageOps.pad(img, dimensions, color=background_color, method=Image.Resampling.LANCZOS)
        else:
            # need to resize here so text overlay scales correctly
            img = resize_image(
                img,
                device_config.get_resolution(),
                image_settings=[]
            )

        # Apply text overlay if enabled
        if settings.get('showDate') == 'true' and memory:
            match settings.get('showLocation'):
                case "country":
                    location = memory.country
                case "city":
                    location = memory.city
                case _:
                    location = None

            # Font size setting is a percentage (1-10), convert to decimal (0.01-0.10)
            font_size_percent = float(settings.get('fontSize', 4)) / 100.0
            img = add_text_overlay(
                img,
                memory.year,
                location,
                font_size_percent
            )

        return img
