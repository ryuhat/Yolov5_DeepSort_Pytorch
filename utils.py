from pathlib import Path

def get_latest_exp_number(track_folder):
    # track_folder = "../runs/track"
    track_path = Path(track_folder)
    exp_folders = [f for f in track_path.iterdir() if f.is_dir() and f.name.startswith("exp") and f.name[3:].isdigit()]
    if not exp_folders:
        return 0
    latest_exp_folder = max(exp_folders, key=lambda f: int(f.name[3:]))
    latest_exp_number = int(latest_exp_folder.name[3:])
    return latest_exp_number
