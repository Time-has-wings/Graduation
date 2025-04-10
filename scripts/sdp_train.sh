export cudart_path="/usr/local/cuda-12.1/targets/x86_64-linux/lib/libcudart.so"
# set various optimazation flags
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# set the launcher and trainer
LAUNCHER="python3 -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7"
# LAUNCHER="python3 -m paddle.distributed.launch --gpus 0,1,2,3"

TRAINER="./parallelism/pure_shard.py"

# set all configurations
PARSER_ARGS="
    --vocab_size 32000 \
    --seq_length 1024 \
    --global_batch_size 8 \
    --seed 1234 \
    --ngpu 8 \
    --num_epochs 1 \
    --iter_upper_limit 21 \
    --hidden_size 4096 \
    --num_hidden_layers 4 \
    --rms_norm_eps 1e-6 \
    --num_attention_heads 32 \
    --dp_degree 1 \
    --mp_degree 1 \
    --pp_degree 1 \
    --sdp_degree 8 \
    --mixed_precision bf16 \
    --micro_batch_size 1 \
"

# run the training
${LAUNCHER} ${TRAINER} ${PARSER_ARGS}