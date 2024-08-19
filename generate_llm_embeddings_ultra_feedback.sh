
# Set model_type to be 'gpt2' or 'llama' for model_type
# Set other_subsets to be 'ultra_feedback', 'pos_neg', 'set', or 'single'
model_type=$1
other_subsets=$2

# Generate LLM embeddings for UltraFeedback dataset
if [ "${other_subsets}" = "ultra_feedback" ]; then
    subsets="helpfulness honesty instruction_following truthfulness"
    postfix=""
elif [ "${other_subsets}" = "pos_neg" ]; then
    subsets="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"
    postfix="_pos_neg"
elif [ "${other_subsets}" = "set" ]; then
    subsets="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"
    postfix="_subset"
elif [ "${other_subsets}" = "single" ]; then
    subsets="8 4 2 1"
    postfix="_single"
elif [ "${other_subsets}" = "84" ]; then
    subsets="8 4"
    postfix="_84"
else
    echo "Invalid!"
fi

echo "${subsets}"
#
#for subset in ${subsets}
#do
#    python -m hidden_context.data_utils.data_processing --output_dir "data/UltraFeedback${postfix}_in_context_fixed_finegrained_filtered_last_token_8/" \
#    --data_path "data/UltraFeedback${postfix}_finegrained_filtered" --data_subset ${subset} --data_split test --model_type ${model_type} \
#    --other_subsets ${other_subsets} --add_controversial True
#
#    python -m hidden_context.data_utils.data_processing --output_dir "data/UltraFeedback${postfix}_in_context_fixed_finegrained_filtered_last_token_8/" \
#    --data_path "data/UltraFeedback${postfix}_finegrained_filtered" --data_subset ${subset} --data_split train --model_type ${model_type} \
#    --other_subsets ${other_subsets} --add_controversial True
#done
#

#for subset in ${subsets}
#do
#    python -m hidden_context.data_utils.data_processing --output_dir "data/UltraFeedback${postfix}_in_context_finegrained_filtered_last_token_8/" \
#    --data_path "data/UltraFeedback${postfix}_finegrained_filtered" --data_subset ${subset} --data_split train --model_type ${model_type} \
#    --other_subsets ${other_subsets} --add_controversial True --with_embeddings False
#
#    python -m hidden_context.data_utils.data_processing --output_dir "data/UltraFeedback${postfix}_in_context_finegrained_filtered_last_token_8/" \
#    --data_path "data/UltraFeedback${postfix}_finegrained_filtered" --data_subset ${subset} --data_split test --model_type ${model_type} \
#    --other_subsets ${other_subsets} --add_controversial True --with_embeddings False
#done

#survey_size=100
#for subset in ${subsets}
#do
#    python -m hidden_context.data_utils.add_survey_contexts --output_dir "data/large_survey_${survey_size}/" \
#    --data_path "data/UltraFeedback${postfix}_finegrained_filtered" --data_subset ${subset} --data_split train --model_type ${model_type} \
#    --other_subsets ${other_subsets} --add_controversial True --with_embeddings True --survey_size $survey_size
#
#    python -m hidden_context.data_utils.add_survey_contexts --output_dir "data/large_survey_${survey_size}/" \
#    --data_path "data/UltraFeedback${postfix}_finegrained_filtered" --data_subset ${subset} --data_split test --model_type ${model_type} \
#    --other_subsets ${other_subsets} --add_controversial True --with_embeddings True --survey_size $survey_size
#done

# Final version for four users
#survey_size=100
#for subset in ${subsets}
#do
#    python -m hidden_context.data_utils.add_survey_contexts --output_dir "data/P_4_survey_${survey_size}_large/" \
#    --data_path "data/UltraFeedback_single_P_4" --data_subset ${subset} --data_split train --model_type ${model_type} \
#    --other_subsets ${other_subsets} --add_controversial True --with_embeddings True --survey_size $survey_size --num_duplicates 8
#
#    python -m hidden_context.data_utils.add_survey_contexts --output_dir "data/P_4_survey_${survey_size}_large/" \
#    --data_path "data/UltraFeedback_single_P_4" --data_subset ${subset} --data_split test --model_type ${model_type} \
#    --other_subsets ${other_subsets} --add_controversial True --with_embeddings True --survey_size $survey_size --num_duplicates 8
#done

# Variable length version for two users
#survey_size=100
#for subset in ${subsets}
#do
#    python -m hidden_context.data_utils.add_survey_contexts --output_dir "data/P_survey_${survey_size}_variable/" \
#    --data_path "data/UltraFeedback_84_P" --data_subset ${subset} --data_split train --model_type ${model_type} \
#    --other_subsets ${other_subsets} --add_controversial True --with_embeddings True --survey_size $survey_size --num_duplicates 8
#
#    python -m hidden_context.data_utils.add_survey_contexts --output_dir "data/P_survey_${survey_size}_variable/" \
#    --data_path "data/UltraFeedback_84_P" --data_subset ${subset} --data_split test --model_type ${model_type} \
#    --other_subsets ${other_subsets} --add_controversial True --with_embeddings True --survey_size $survey_size --num_duplicates 8
#done

survey_size=100
for subset in ${subsets}
do
    python -m hidden_context.data_utils.add_survey_contexts --output_dir "data/P_survey_${survey_size}_variable_random_contexts/" \
    --data_path "data/UltraFeedback_84_P_random_contexts" --data_subset ${subset} --data_split train --model_type ${model_type} \
    --other_subsets ${other_subsets} --add_controversial True --with_embeddings True --survey_size $survey_size --num_duplicates 8 \
    --random_contexts True

    python -m hidden_context.data_utils.add_survey_contexts --output_dir "data/P_survey_${survey_size}_variable_random_contexts/" \
    --data_path "data/UltraFeedback_84_P_random_contexts" --data_subset ${subset} --data_split test --model_type ${model_type} \
    --other_subsets ${other_subsets} --add_controversial True --with_embeddings True --survey_size $survey_size --num_duplicates 8 \
    --random_contexts True
done




# Final version for two users
#survey_size=16
#for subset in ${subsets}
#do
#    python -m hidden_context.data_utils.add_survey_contexts --output_dir "data/P_survey_${survey_size}_final/" \
#    --data_path "data/UltraFeedback_${other_subsets}_P" --data_subset ${subset} --data_split train --model_type ${model_type} \
#    --other_subsets ${other_subsets} --add_controversial True --with_embeddings True --survey_size $survey_size --num_duplicates 8 --fixed_context_length True
#
#    python -m hidden_context.data_utils.add_survey_contexts --output_dir "data/P_survey_${survey_size}_final/" \
#    --data_path "data/UltraFeedback_${other_subsets}_P" --data_subset ${subset} --data_split test --model_type ${model_type} \
#    --other_subsets ${other_subsets} --add_controversial True --with_embeddings True --survey_size $survey_size --num_duplicates 8 --fixed_context_length True
#done
