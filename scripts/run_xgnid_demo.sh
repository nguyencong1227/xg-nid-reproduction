#!/usr/bin/env bash
# End-to-end XG-NID demo on the two CIC-IoT2023 captures in data/cic_pcap.
#
# The quotas are scaled down from the paper's 20,000 train / 4,000 test per
# class (Table 4) because only two captures are available here, and Benign is
# harvested from the non-attacker flows inside those captures rather than from
# the dedicated benign captures.  For a full run, drop --train-per-class,
# --test-cap and --benign-from-background.
set -euo pipefail
cd "$(dirname "$0")/.."

PCAPS=${PCAPS:-'data/cic_pcap/*.pcap'}
WORK=${WORK:-work/xgnid}

python -u -m xgnid.cli extract   --pcaps "$PCAPS"        --out "$WORK/01_flows"
python -u -m xgnid.cli features  --in  "$WORK/01_flows"  --out "$WORK/02_features"
python -u -m xgnid.cli dataset   --in  "$WORK/02_features" --out "$WORK/03_dataset" \
                                 --train-per-class 4000 --test-cap 800 \
                                 --benign-from-background
python -u -m xgnid.cli graphs    --dataset "$WORK/03_dataset" --out "$WORK/04_graphs"
python -u -m xgnid.cli train     --graphs  "$WORK/04_graphs"  --out "$WORK/05_run" \
                                 --epochs 25 --batch-size 128
python -u -m xgnid.cli baselines --dataset "$WORK/03_dataset" --out "$WORK/06_baselines" \
                                 --max-train 6000
python -u -m xgnid.cli report    --run "$WORK/05_run" --baselines "$WORK/06_baselines"
python -u -m xgnid.cli explain   --run "$WORK/05_run" --graphs "$WORK/04_graphs" --num 6
python -u -m xgnid.cli infer     --pcap data/cic_pcap/XSS.pcap --run "$WORK/05_run" \
                                 --out "$WORK/07_infer/XSS_predictions.csv"

echo
echo "Results:      $WORK/05_run/results.json"
echo "Comparison:   $WORK/05_run/REPORT.md"
echo "Explanations: $WORK/05_run/explanations_test/explanations.txt"
echo "Predictions:  $WORK/07_infer/XSS_predictions.csv"
