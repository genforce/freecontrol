PROMPT="A photo of a car, on the road, best quality, extremely detailed"
NEGATIVE_PROMPT="fewer digits, cropped, worst quality, low quality"
SD_VERSION="XL_1.0"
MODEL_NAME="naive"
HEIGHT=1024
WIDTH=1024
SEED=28988
NUM_STEPS=200
OUTPUT_CLASS="car"

python sample_semantic_bases.py \
--config_path "config/sdxl_base.yaml" \
--prompt "${PROMPT}" \
--negative_prompt "${NEGATIVE_PROMPT}" \
--sd_version ${SD_VERSION} \
--model_name ${MODEL_NAME} \
--height ${HEIGHT} \
--width ${WIDTH} \
--seed ${SEED} \
--num_steps ${NUM_STEPS} \
--output_class ${OUTPUT_CLASS} \
--num_images 5 \
--num_batch 4 \
--log \