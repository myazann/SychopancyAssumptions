#!/usr/bin/env bash

set -euo pipefail

# =============================================================================
# Directory config
# =============================================================================

ASSUMPTIONS_DIR="${ASSUMPTIONS_DIR:-/assumption_data}"

BASE_OUT_DIR="${BASE_OUT_DIR:-/output/assumption_probes}"

declare -a MODEL_CONFIGS=(
  "meta-llama/Llama-3.3-70B-Instruct|llama70b|20,30,40,50,60,70|8|llama70b"
)

# =============================================================================
# Data config
# =============================================================================

# Filenames (relative to ASSUMPTIONS_DIR)
TRAIN_FILENAME="train.csv"
TEST_FILENAME="test.csv"

# Name of the column in your CSV that holds the assumption score labels
SCORE_COL="assumption_score" # e.g., validation_seeking_score, emotional_support_seeking_score, etc.

# =============================================================================
# Probe config
# =============================================================================

POOLING="mean"      
QUANTILE="0.3"     
BATCH_SIZE="8"       
TRAIN_ALPHA="10.0"  
TRAIN_SEED="42"      

# =============================================================================
# Generation config
# =============================================================================

ALPHAS="-4,-2,-1,-0.5,0,0.5,1,2,4"

MAX_NEW_TOKENS="400"   
SEED="0"              

for config in "${MODEL_CONFIGS[@]}"; do

  IFS='|' read -r model model_short layers gen_batch_size model_tag <<< "$config"

  echo ""
  echo "============================================================"
  echo "Model : $model"
  echo "Tag   : $model_tag"
  echo "Layers: $layers"
  echo "============================================================"

  # Full paths to train and test data
  train_file="$ASSUMPTIONS_DIR/$TRAIN_FILENAME"
  test_file="$ASSUMPTIONS_DIR/$TEST_FILENAME"

  # Directory where trained probe weights and metadata will be saved
  probe_dir="$BASE_OUT_DIR/probes/$model_tag"

  # Generation output paths
  gen_dir="$BASE_OUT_DIR/generations/$model_tag"
  out_jsonl="$gen_dir/steered.jsonl"
  out_wide="$gen_dir/steered_wide.csv"

  # ELEPHANT scoring output prefix (scorer appends its own suffixes)
  elephant_out_prefix="$BASE_OUT_DIR/scores/$model_tag/elephant"

  mkdir -p "$probe_dir"
  mkdir -p "$gen_dir"
  mkdir -p "$BASE_OUT_DIR/scores/$model_tag"

  if [[ ! -f "$train_file" ]]; then
    echo "ERROR: Training file not found: $train_file" >&2
    exit 1
  fi

  if [[ ! -f "$test_file" ]]; then
    echo "ERROR: Test file not found: $test_file" >&2
    exit 1
  fi

  # ===========================================================================
  # Step 1 — Train assumption probe
  # ===========================================================================
  echo ""
  echo "--- Step 1: Training probe ---"

  python3 train_assumption_probe.py \
    --data         "$train_file"  \
    --model        "$model"       \
    --out_dir      "$probe_dir"   \
    --pooling      "$POOLING"     \
    --layers       "$layers"      \
    --batch_size   "$BATCH_SIZE"  \
    --alpha        "$TRAIN_ALPHA" \
    --seed         "$TRAIN_SEED"  \
    --score        "$SCORE_COL"   \
    --load_in_4bit

  echo "Probe saved to: $probe_dir"

  # ===========================================================================
  # Step 2 — Evaluate AUC by source on test set
  # ===========================================================================
  echo ""
  echo "--- Step 2: Evaluating AUC by source ---"

  python3 evaluate_auc_by_source.py \
    --data       "$test_file"   \
    --probe_dir  "$probe_dir"   \
    --model      "$model"       \
    --quantile   "$QUANTILE"    \
    --score      "$SCORE_COL"   \
    --batch_size "$BATCH_SIZE"  \
    --load_in_4bit

  # ===========================================================================
  # Step 3 — Generate steered responses (skip if output already exists)
  # ===========================================================================
  echo ""
  echo "--- Step 3: Generating steered responses ---"

  if [[ -f "$out_wide" ]]; then
    echo "Skipping generation — output already exists: $out_wide"
  else
    python3 sample_and_generate_steered.py \
      --data           "$test_file"      \
      --prompt_col     user_text         \
      --n              "$N_GEN_INTERNAL" \
      --seed           "$SEED"           \
      --model          "$model"          \
      --batch_size     "$gen_batch_size" \
      --probe_dir      "$probe_dir"      \
      --alphas         "$ALPHAS"         \
      --do_sample                        \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --out_jsonl      "$out_jsonl"      \
      --out_csv_wide   "$out_wide"       \
      --use_4bit

    echo "Generations saved to: $out_wide"
  fi

  # ===========================================================================
  # Step 4 — Score steered responses with ELEPHANT
  # ===========================================================================
  echo ""
  echo "--- Step 4: ELEPHANT scoring ---"

  if [[ ! -f "$out_wide" ]]; then
    echo "WARNING: Generation output not found, skipping ELEPHANT scoring: $out_wide" >&2
  else
    python3 ../elephant_scorer_5pointscale.py \
      "$out_wide"            \
      "$elephant_out_prefix"

    echo "ELEPHANT scores saved to: ${elephant_out_prefix}*"
  fi

  echo ""
  echo "Finished model: $model_tag"

done

echo ""
echo "============================================================"
echo "All models complete."
echo "============================================================"