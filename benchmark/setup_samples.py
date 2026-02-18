import os

# Script to help setup benchmark CCTV samples.
# Note: SharePoint links require manual download via browser.

SAMPLES_DIR = "benchmark/cctv_samples"
SAMPLES = {
    "C1HighRes - Human.mp4": "https://drive.google.com/file/d/1rUlnJr5g4Tj6WsLfKStRAJq2_P96b6Uj/view?usp=drive_link",
    "C1LowRes - Human.mp4": "https://drive.google.com/file/d/1UjZv3yxt-28pmyIy2W9N4TAfLYpRZqqQ/view?usp=drive_link",
    "C2HighRes - Car.mp4": "https://drive.google.com/file/d/1EGluAF3Y6q_H4Kg7ZXsOBxAG6YJoio_z/view?usp=drive_link",
    "C2LowRes - Car.mp4": "https://drive.google.com/file/d/1IKayZA9K4UqCOqkSqvjJW8TONXUY0Qig/view?usp=drive_link",
    "C3HighRes - Car.mp4": "https://drive.google.com/file/d/1XPW5fHKhgmTo2H_ScXC1RYGqePXw7QhT/view?usp=drive_link",
    "C3LowRes - Car.mp4": "https://drive.google.com/file/d/1AmEw-l0qa6HBisYxm6YfN5D-a-JsvAfm/view?usp=drive_link",
    "C4HighRes - Human.mp4": "https://drive.google.com/file/d/1jtIhRHc8no55JAmf2wltdE65lArCHB3t/view?usp=drive_link",
    "C4LowRes - Human.mp4": "https://drive.google.com/file/d/1Z01n9zQ7BMXG_FOG-wQmG4QTXctkS9vD/view?usp=drive_link",
    "C5HighResPTZ - Car.mp4": "https://drive.google.com/file/d/1ev5oXKDYHac0FiUDpICQcAWGXUhXPWpg/view?usp=drive_link",
    "C5LowResPTZ - Car.mp4": "https://drive.google.com/file/d/11pgoDhvAXxmXlA2qOkPfjPJPk7bwmvX0/view?usp=drive_link",
    "C6HighResPTZ - Human.mp4": "https://drive.google.com/file/d/1hCTfCLiUpW2sN48vBWb6QmmtlAJ94t_I/view?usp=drive_link",
    "C6LowResPTZ - Human.mp4": "https://drive.google.com/file/d/11kZOHq86kuju9qr9h18vvJBWGPMcNeOu/view?usp=drive_link",
}


def download_from_gdrive(url, dest_path):
    """Downloads a file from Google Drive using its sharing URL."""
    try:
        import requests  # type: ignore
    except ImportError:
        print("Error: 'requests' library not found. Installing...")
        import subprocess

        subprocess.check_call(["pip", "install", "requests"])
        import requests  # type: ignore

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                return value
        return None

    def save_response_content(response, dest_path):
        CHUNK_SIZE = 32768
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    # Extract ID from the view link
    # https://drive.google.com/file/d/ID/view?usp=drive_link
    if "/file/d/" in url:
        file_id = url.split("/file/d/")[1].split("/")[0]
    else:
        print(f"Skipping: Could not parse GDrive ID from {url}")
        return False

    download_url = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(download_url, params={"id": file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": file_id, "confirm": token}
        response = session.get(download_url, params=params, stream=True)

    print(f"Downloading to {dest_path}...")
    save_response_content(response, dest_path)
    return True


def setup():
    if not os.path.exists(SAMPLES_DIR):
        os.makedirs(SAMPLES_DIR)
        print(f"Created directory: {SAMPLES_DIR}")

    print("\n--- CCTV Sample Setup ---")
    print("Checking for samples in:", os.path.abspath(SAMPLES_DIR))

    for filename, url in SAMPLES.items():
        filepath = os.path.join(SAMPLES_DIR, filename)
        if not os.path.exists(filepath):
            print(f"[MISSING] {filename}")
            success = download_from_gdrive(url, filepath)
            if success:
                print(f"[DONE]    Downloaded {filename}")
            else:
                print(f"[FAILED]  Could not download {filename}")
        else:
            print(f"[FOUND]   {filename}")

    print("\nCheck complete. You are ready to run the benchmark!")


if __name__ == "__main__":
    setup()
