#!/bin/bash

# CSV_PATH = <path to direct outcome dataset (CSV)>
# DATASET = <name of dataset>
# TEXT = <name of text column>
# OUTCOME = <name of outcome column>
# NUM_LABELS = <2 for binary outcome, 1 for continuous outcome>
# RESULTS_DIR = <path to results directory>
# PRETRAINED_MODEL_PATH = <path to HuggingFace pre-trained model to optimize>
# MODEL_NAME = <name of pre-trained model>
# TOKENIZER_PATH = <path to tokenizer for pre-trained model>
# FT_MODEL_PATH = <path to pre-trained model that has been fine-tuned on texts from the direct outcome dataset>
# OUTCOME_MODEL_PATH = <path to outcome model after it has been trained>
# OPTIMIZED_MODEL_PATHS = <list of paths to CPO, OO-RLHF, and DR-CPO models after they have been optimized>
# EPOCHS = <number of epochs to train>
# BATCH_SIZE = <batch size>
# PROMPT_CSV = <path to CSV containing prompts to use to generate texts in optimized models>
# PROMPT_COL = <name of column containing prompts>
# NUM_PROMPTS = <number of prompts to be used for generation. If this number is greater than the size of PROMPT_CSV, prompts will be reused at random.>

SHARED_MODELS=(bias_corrected_rlhf bias_corrected_rlhf_ablationrlhfpaired)
OUTPUT_NAMES=(cpo oo_rlhf)

# Training outcome model
python train_outcome_model.py \
    --csv-path $CSV_PATH \
    --dataset $DATASET \
    --text $TEXT \
    --outcome $OUTCOME \
    --num-labels $NUM_LABELS \
    --results-dir $RESULTS_DIR \
    --model-path $PRETRAINED_MODEL_PATH \
    --model-name $MODEL_NAME \
    --tokenizer-path $TOKENIZER_PATH \
    --epochs $EPOCHS

# Precomputing P^{f_0}
python get_pretrained_logits.py \
    --csv-path $CSV_PATH \
    --dataset $DATASET \
    --text $TEXT \
    --model-path $FT_MODEL_PATH \
    --model-name $MODEL_NAME \
    --tokenizer-path $TOKENIZER_PATH \
    --batch-size $BATCH_SIZE \
    --results-dir $RESULTS_DIR \
    --paired \
    --outcome-model-path $OUTCOME_MODEL_PATH \
    --num-labels $NUM_LABELS
printf 'Finished computing Hatespeech confounded\n'

# Training CPO
python train.py \
    --csv-path $CSV_PATH \
    --dataset $DATASET \
    --text $TEXT \
    --outcome $OUTCOME \
    --opt-type bias_corrected_rlhf \
    --model-path $FT_MODEL_PATH \
    --outcome-model-path $OUTCOME_MODEL_PATH \
    --model-name $MODEL_NAME \
    --tokenizer-path $TOKENIZER_PATH \
    --num-labels $NUM_LABELS \
    --batch-size $BATCH_SIZE \
    --results-dir $RESULTS_DIR \
    --epochs $EPOCHS
    
# Training OO-RLHF
python train.py \
    --csv-path $CSV_PATH \
    --dataset $DATASET \
    --text $TEXT \
    --outcome $OUTCOME \
    --opt-type bias_corrected_rlhf \
    --model-path $FT_MODEL_PATH \
    --outcome-model-path $OUTCOME_MODEL_PATH \
    --model-name $MODEL_NAME \
    --tokenizer-path $TOKENIZER_PATH \
    --num-labels $NUM_LABELS \
    --batch-size $BATCH_SIZE \
    --results-dir $RESULTS_DIR \
    --c-ipw 0.0 --c-rlhf 1.0 --paired \
    --epochs $EPOCHS

# Training DR-CPO
python train.py \
    --csv-path $CSV_PATH \
    --dataset $DATASET \
    --text $TEXT \
    --outcome $OUTCOME \
    --opt-type bias_corrected_rlhf \
    --model-path $FT_MODEL_PATH \
    --outcome-model-path $OUTCOME_MODEL_PATH \
    --model-name $MODEL_NAME \
    --tokenizer-path $TOKENIZER_PATH \
    --num-labels $NUM_LABELS \
    --batch-size $BATCH_SIZE \
    --results-dir ${RESULTS_DIR}_paired \
    --paired \
    --epochs $EPOCHS

# Generate texts from CPO and OO-RLHF models
for i in "${!SHARED_MODELS[@]}"; do
    python generate_from_csv.py --generate-outcomes \
        --tokenizer-path $TOKENIZER_PATH \
        --model-path ${OPTIMIZED_MODEL_PATHS[$i]} \
        --model-name $MODEL_NAME \
        --outcome-model-path $OUTCOME_MODEL_PATH \
        --prompt-csv $PROMPT_CSV \
        --prompt-col $PROMPT_COL \
        --results-dir $RESULTS_DIR \
        --output-name ${OUTPUT_NAMES[$i]} \
        --dataset $DATASET \
        --num-labels $NUM_LABELS \
        --batch-size $BATCH_SIZE \
        --output-dir $OUTPUT_DIR \
        --generate-text \
        --num-prompts $NUM_PROMPTS
done

# Generate texts from DR-CPO model
python generate_from_csv.py --generate-outcomes \
    --tokenizer-path $TOKENIZER_PATH \
    --model-path $DRCPO_MODEL_PATH \
    --model-name $MODEL_NAME \
    --outcome-model-path $OUTCOME_MODEL_PATH \
    --prompt-csv $PROMPT_CSV \
    --prompt-col $PROMPT_COL \
    --results-dir $RESULTS_DIR \
    --output-name dr_cpo \
    --dataset $DATASET \
    --num-labels $NUM_LABELS \
    --batch-size $BATCH_SIZE \
    --output-dir $OUTPUT_DIR \
    --generate-text \
    --num-prompts $NUM_PROMPTS