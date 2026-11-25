#!/usr/bin/env bash
# One command: convert Yelp -> project format, then tune + evaluate.
# Usage:  bash experiments/yelp/run_yelp.sh ~/Downloads/yelp
set -euo pipefail
YELP_DIR="${1:?Give the folder that contains yelp_academic_dataset_business.json}"
python experiments/yelp/prepare_yelp.py --yelp-dir "$YELP_DIR" --out data_yelp
python experiments/evaluate_hybrid.py --data-dir data_yelp \
  --md experiments/yelp/RESULTS.md --title "Yelp results (real data)"
echo
echo "Done. Results table: experiments/yelp/RESULTS.md"
