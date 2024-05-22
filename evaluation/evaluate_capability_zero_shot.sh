DEVICE=6
export HF_HOME=... # Path to your huggingface cache directory
RESULTS=... # Path to save results

MODEL=... # Path to edited model
model_args=pretrained=${MODEL}

mkdir -p ${RESULTS}
echo ""
echo "${MODEL}"

CUDA_VISIBLE_DEVICES=${DEVICE} lm_eval --model hf \
    --model_args ${model_args}\
    --batch_size auto \
    --tasks rte,boolq,hellaswag,winogrande,openbookqa,arc_easy,arc_challenge \
    --num_fewshot 0 \
    --seed 20 \
    --output_path ${RESULTS}/commonsense_reasoning_${MODEL}.json
