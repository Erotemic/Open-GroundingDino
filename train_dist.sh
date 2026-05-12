#!/bin/bash
GPU_NUM=$1
CFG=$2
DATASETS=$3
OUTPUT_DIR=$4
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
PRETRAIN_MODEL_PATH=${PRETRAIN_MODEL_PATH:-"/path/to/groundingdino_swint_ogc.pth"}
TEXT_ENCODER_TYPE=${TEXT_ENCODER_TYPE:-"/path/to/bert-base-uncased"}
PYTHON_BIN=${PYTHON_BIN:-python}
AMP_FLAG=()
if [ "${USE_AMP:-0}" = "1" ] || [ "${USE_AMP:-}" = "true" ]; then
        AMP_FLAG+=(--amp)
fi
echo "
GPU_NUM = $GPU_NUM
CFG = $CFG
DATASETS = $DATASETS
OUTPUT_DIR = $OUTPUT_DIR
NNODES = $NNODES
NODE_RANK = $NODE_RANK
PORT = $PORT
MASTER_ADDR = $MASTER_ADDR
PRETRAIN_MODEL_PATH = $PRETRAIN_MODEL_PATH
TEXT_ENCODER_TYPE = $TEXT_ENCODER_TYPE
USE_AMP = ${USE_AMP:-0}
"

# Change ``pretrain_model_path`` to use a different pretrain.
# (e.g. GroundingDINO pretrain, DINO pretrain, Swin Transformer pretrain.)
# If you don't want to use any pretrained model, just ignore this parameter.

if [ "${GPU_NUM}" = "1" ]; then
        "${PYTHON_BIN}" main.py \
                --output_dir "${OUTPUT_DIR}" \
                -c "${CFG}" \
                --datasets "${DATASETS}"  \
                --pretrain_model_path "${PRETRAIN_MODEL_PATH}" \
                "${AMP_FLAG[@]}" \
                --options text_encoder_type="$TEXT_ENCODER_TYPE"
else
        "${PYTHON_BIN}" -m torch.distributed.run \
                --nproc_per_node="${GPU_NUM}" \
                --nnodes="${NNODES}" \
                --node_rank="${NODE_RANK}" \
                --master_addr="${MASTER_ADDR}" \
                --master_port="${PORT}" \
                main.py \
                --output_dir "${OUTPUT_DIR}" \
                -c "${CFG}" \
                --datasets "${DATASETS}"  \
                --pretrain_model_path "${PRETRAIN_MODEL_PATH}" \
                "${AMP_FLAG[@]}" \
                --options text_encoder_type="$TEXT_ENCODER_TYPE"
fi
