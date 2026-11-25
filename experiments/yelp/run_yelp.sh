#!/usr/bin/env bash
# One command: convert Yelp -> project format, evaluate the original hybrid (v1),
# then the improved hybrid (v2). Usage:  bash experiments/yelp/run_yelp.sh <yelp folder>
set -euo pipefail
YELP_DIR="${1:?Give the folder that contains yelp_academic_dataset_business.json}"
python experiments/yelp/prepare_yelp.py --yelp-dir "$YELP_DIR" --out data_yelp
python experiments/evaluate_hybrid.py --data-dir data_yelp \
  --md experiments/yelp/RESULTS.md --title "Yelp results (real data)"
python experiments/evaluate_hybrid_v2.py --data-dir data_yelp \
  --md experiments/yelp/RESULTS_v2.md --title "Yelp results: improved hybrid (v2)"
echo
echo "Done. Results: experiments/yelp/RESULTS.md and experiments/yelp/RESULTS_v2.md"
