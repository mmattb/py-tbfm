#!/bin/bash
# Cross-animal generalization: train on one monkey, TTA on the other.
#
# Fold G2J: train on all 22 MonkeyG sessions → TTA on MonkeyJ sessions
# Fold J2G: train on all 18 MonkeyJ sessions → TTA on MonkeyG sessions
#
# Uses the same hyperparams as the main random_folds experiments.
# gather_session_ids() in multisession.py will set held_out = all - held_in,
# so no code change needed to split by animal.
#
# Usage: bash scripts/train_cross_animal.sh [output_dir]

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="${1:-cross_animal_${TIMESTAMP}}"
mkdir -p "${OUTPUT_BASE}"

echo "Cross-animal generalization training started at ${TIMESTAMP}"
echo "Output directory: ${OUTPUT_BASE}"

# -----------------------------------------------------------------------
# All MonkeyG sessions (22 total)
# -----------------------------------------------------------------------
MONKEY_G_SESSIONS="MonkeyG_20150914_Session1_S1,MonkeyG_20150914_Session3_S1,MonkeyG_20150915_Session2_S1,MonkeyG_20150915_Session3_S1,MonkeyG_20150915_Session4_S1,MonkeyG_20150915_Session5_S1,MonkeyG_20150916_Session4_S1,MonkeyG_20150917_Session1_M1,MonkeyG_20150917_Session1_S1,MonkeyG_20150917_Session2_M1,MonkeyG_20150917_Session2_S1,MonkeyG_20150917_Session3_M1,MonkeyG_20150917_Session3_S1,MonkeyG_20150918_Session1_M1,MonkeyG_20150918_Session1_S1,MonkeyG_20150921_Session3_S1,MonkeyG_20150921_Session5_S1,MonkeyG_20150922_Session1_S1,MonkeyG_20150922_Session2_S1,MonkeyG_20150922_Session3_S1,MonkeyG_20150925_Session1_S1,MonkeyG_20150925_Session2_S1"

# -----------------------------------------------------------------------
# All MonkeyJ sessions (18 total)
# -----------------------------------------------------------------------
MONKEY_J_SESSIONS="MonkeyJ_20160426_Session1_S1,MonkeyJ_20160426_Session2_S1,MonkeyJ_20160426_Session3_S1,MonkeyJ_20160428_Session2_S1,MonkeyJ_20160428_Session3_S1,MonkeyJ_20160429_Session1_S1,MonkeyJ_20160429_Session3_S1,MonkeyJ_20160502_Session1_S1,MonkeyJ_20160624_Session3_S1,MonkeyJ_20160624_Session4_S1,MonkeyJ_20160625_Session4_S1,MonkeyJ_20160625_Session5_S1,MonkeyJ_20160627_Session1_S1,MonkeyJ_20160627_Session2_S1,MonkeyJ_20160630_Session1_S1,MonkeyJ_20160630_Session3_S1,MonkeyJ_20160702_Session2_S1,MonkeyJ_20160702_Session4_S1"

# -----------------------------------------------------------------------
# Hyperparams matching the main random_folds experiments
# -----------------------------------------------------------------------
NUM_BASES=100
LATENT_DIM=96
BASIS_RESIDUAL_RANK=16
TRAIN_SIZE=5000
BATCH_SIZE=500

# G→J: train on 22 MonkeyG, evaluate TTA on MonkeyJ
echo "Starting G→J fold on GPU 0..."
python tma_standalone.py \
    ${NUM_BASES} 22 0 false ${BASIS_RESIDUAL_RANK} ${TRAIN_SIZE} true \
    --latent-dim ${LATENT_DIM} \
    --batch-size-per-session ${BATCH_SIZE} \
    --out-dir "${OUTPUT_BASE}/fold_G2J" \
    --held-in-sessions "${MONKEY_G_SESSIONS}" &
PID_G2J=$!
echo "G→J: started (PID ${PID_G2J})"

# J→G: train on 18 MonkeyJ, evaluate TTA on MonkeyG
echo "Starting J→G fold on GPU 1..."
python tma_standalone.py \
    ${NUM_BASES} 18 1 false ${BASIS_RESIDUAL_RANK} ${TRAIN_SIZE} true \
    --latent-dim ${LATENT_DIM} \
    --batch-size-per-session ${BATCH_SIZE} \
    --out-dir "${OUTPUT_BASE}/fold_J2G" \
    --held-in-sessions "${MONKEY_J_SESSIONS}" &
PID_J2G=$!
echo "J→G: started (PID ${PID_J2G})"

wait ${PID_G2J}
EC=$?
if [ $EC -ne 0 ]; then echo "ERROR: G→J fold failed (exit ${EC})"; exit 1; fi
echo "G→J fold complete."

wait ${PID_J2G}
EC=$?
if [ $EC -ne 0 ]; then echo "ERROR: J→G fold failed (exit ${EC})"; exit 1; fi
echo "J→G fold complete."

echo ""
echo "Both cross-animal training runs complete. Results in: ${OUTPUT_BASE}"
echo ""
echo "Next: run TTA with tta_random_folds.sh or manually:"
echo "  python tta_testing.py --model-paths fold_G2J:${OUTPUT_BASE}/fold_G2J \\"
echo "    --unfreeze-bases --progressive-unfreezing-threshold 0 \\"
echo "    --max-adapt-sessions 20 --tta-epochs 7001 \\"
echo "    --output-dir ${OUTPUT_BASE}/tta_G2J"
echo ""
echo "  python tta_testing.py --model-paths fold_J2G:${OUTPUT_BASE}/fold_J2G \\"
echo "    --unfreeze-bases --progressive-unfreezing-threshold 0 \\"
echo "    --max-adapt-sessions 20 --tta-epochs 7001 \\"
echo "    --output-dir ${OUTPUT_BASE}/tta_J2G"
