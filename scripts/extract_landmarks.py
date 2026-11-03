import argparse
from isl_data_collection.extraction import extract_dataset

parser = argparse.ArgumentParser(description="Extract MediaPipe hand landmarks from labelled ISL clips.")
parser.add_argument("--force", action="store_true", help="Re-extract clips even when features already exist.")
parser.add_argument("--frame-stride", type=int, default=2, help="Process every Nth source frame.")
args = parser.parse_args()
print(extract_dataset(force=args.force, frame_stride=max(1, args.frame_stride)))
