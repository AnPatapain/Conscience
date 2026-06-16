import fiftyone as fo
import fiftyone.zoo as foz
from pathlib import Path
import csv

ROOT = Path.cwd()
RAW_DATA_DIR = ROOT / "data" / "raw" / "fiftyone"
OUT_CSV = ROOT / "data" / "annotations" / "sessions.csv"

fo.config.dataset_zoo_dir = RAW_DATA_DIR
dataset = foz.load_zoo_dataset(
    'activitynet-200',
    splits = ['train'],
    classes = ['Smoking a cigarette'],
    max_samples = 5,
)
dataset.compute_metadata()

rows = []

def main():
    for i, sample in enumerate(dataset, start=1):
        print(sample)
        rows.append({
            'session_id': f'S{i:03d}',
            'video_path': sample.filepath,
            'duration_sec': sample.metadata.duration,
            'viewpoint': 'n/a',
            'scenario': 'smoking_prep',
            'notes': 'n/a',
        })

    # Write to data/annotations/sessions.csv
    with open(OUT_CSV, 'w', encoding='utf-8', newline='') as csvfile:
        fieldnames = ['session_id', 'video_path', 'duration_sec', 'viewpoint', 'scenario', 'notes']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

if __name__ == '__main__':
    main()