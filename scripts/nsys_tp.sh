nsys profile \
    -w true \
    -t cuda,nvtx,osrt,cudnn,cublas \
    -s cpu \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --cudabacktrace=true \
    -x true \
    --force-overwrite true \
    -o ./nsys/paddle-$(date +"%Y%m%d_%H%M%S") \
    bash ./scripts/tp_train.sh