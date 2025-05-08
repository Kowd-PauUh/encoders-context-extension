#!/bin/sh

# the benchmarking results will be written under results/ directory by MTEB 

models="intfloat/e5-large-v2 FacebookAI/roberta-large"
interpolations="linear cubic"
ctx_space="512 1024 1536 2048 2560 3072"

for model in $models; do
    for interpolation in $interpolations; do
        for ctx in $ctx_space; do
            echo ============================================
            echo Evaluating $model with $interpolation interpolation on ctx$ctx

            # infer offset from the model
            if [ "$model" = "intfloat/e5-large-v2" ]; then
                offset=0
            fi
            if [ "$model" = "FacebookAI/roberta-large" ]; then
                offset=2
            fi

            model_dir=$model-ctx$ctx-$interpolation

            # create model with interpolated embeddings
            python3 context_extension/interpolate_embeddings.py \
                --model_name_or_path=$model \
                --max_seq_length=$ctx \
                --embeddings_attr_name="embeddings.position_embeddings" \
                --offset=$offset \
                --interpolation_type=$interpolation \
                --output_dir=$model_dir
            
            # benchmark model
            python3 context_extension/benchmark_model.py --model_name_or_path=$model_dir

            # remove model
            rm -r $model_dir
        done
    done
done
