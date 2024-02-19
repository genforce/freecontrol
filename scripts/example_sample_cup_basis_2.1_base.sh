PROMPT="A photo of a cat, with a simple background, best quality, extremely detailed"
NEGATIVE_PROMPT="fewer digits, cropped, worst quality, low quality"
SD_VERSION="1.5"
MODEL_NAME="naive"
HEIGHT=512
WIDTH=512
SEED=2024
NUM_STEPS=200
OUTPUT_CLASS="cu"

python sample_semantic_bases.py --prompt "${PROMPT}" \
--negative_prompt "${NEGATIVE_PROMPT}" \
--sd_version ${SD_VERSION} \
--model_name ${MODEL_NAME} \
--height ${HEIGHT} \
--width ${WIDTH} \
--seed ${SEED} \
--num_steps ${NUM_STEPS} \
--output_class ${OUTPUT_CLASS} \
--num_images 20 \
--num_batch 1 \
--log \
