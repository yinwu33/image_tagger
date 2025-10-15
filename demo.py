from __future__ import annotations

from pathlib import Path
from dotenv import load_dotenv

from image_tagger import create_tagger


if __name__ == "__main__":
    load_dotenv()
    image_path = "example.jpg"

    tagger = create_tagger(name="openai")
    tags = tagger.tag(image_path)

    print("Generated tags:")
    for tag in tags:
        print(f"- {tag}")
