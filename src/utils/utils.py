from pathlib import Path
import shutil


def delete_directory_contents(directory: Path, recursive: bool = False):
    if not directory.is_dir():
        raise ValueError(f"{directory} is not a valid directory.")

    for item in directory.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            if recursive:
                shutil.rmtree(item)
            else:
                print(f"Skipping subdirectory: {item}")

    if recursive:
        print(f"All contents in {directory} have been deleted.")
    else:
        print(
            f"All files in {directory} have been deleted. Subdirectories were skipped."
        )
