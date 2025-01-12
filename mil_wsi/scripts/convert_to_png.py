import os
import argparse
import openslide
from PIL import Image
import matplotlib.pyplot as plt


def convert_to_png(source: str, save_dir: str, custom_image: str, file_extension: str):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Directory created: {save_dir}")
    else:
        print(f"Directory already exists: {save_dir}")

    files = [f for f in os.listdir(source) if os.path.isfile(os.path.join(source, f))]
    files = [x for x in files if file_extension.lower() in x.lower()]

    # Standardize custom_image input if provided
    if len(custom_image) != 0:
        if not custom_image.endswith(f".{file_extension}"):
            custom_image += f".{file_extension}"
        files = [os.path.join(source, custom_image)]
    else:
        files = [os.path.join(source, f) for f in files]

    # Check file extension and process accordingly
    if file_extension.lower() in ['svs', 'ndpi', 'tiff']:
        print(f"Conversion from {file_extension.upper()} to PNG")

        for file in files:
            print(f"Processing file: {file}")
            filename = os.path.basename(file).split(f".{file_extension}")[0]
            
            try:
                slide = openslide.OpenSlide(file)
                thumbnail = slide.get_thumbnail((1024, 1024))  # Resize thumbnail to 1024x1024
                thumbnail.save(f"{save_dir}/{filename}.png", format="PNG")
                print(f"Saved: {save_dir}/{filename}.png")
            except Exception as e:
                print(f"Error processing file {file}: {e}")
    else:
        raise Exception(f"Currently we do not support the {file_extension} conversion to PNG")


parser = argparse.ArgumentParser(description="Convert WSI files to PNG")
parser.add_argument("--source", type=str, required=True, help="Path to folder containing raw WSI image files")
parser.add_argument("--save_dir", type=str, required=True, help="Directory to save processed data")
parser.add_argument(
    "--custom_image",
    type=str,
    default="",
    help="Name of custom image that wants to be converted to PNG. If omitted, all files with the extension will be converted",
)
parser.add_argument(
    "--file_extension", type=str, default="svs", help="Extension of the files that will be converted to PNG"
)

if __name__ == "__main__":
    args = parser.parse_args()

    convert_to_png(
        source=args.source,
        save_dir=args.save_dir,
        custom_image=args.custom_image,
        file_extension=args.file_extension,
    )

    

