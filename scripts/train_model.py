import argparse
from isl_data_collection.training import train_category

parser = argparse.ArgumentParser(description="Train one signer-independent ISL category model.")
parser.add_argument("--category", choices=["word", "alphabet", "number"], required=True)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch-size", type=int, default=8)
args = parser.parse_args()
print(train_category(args.category, epochs=args.epochs, batch_size=args.batch_size))
