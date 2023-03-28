GPU=1
DATA_PATH=$pwd/${DOMAIN}_data
FAIRSEQ_PATH=$pwd/fairseq

for DOMAIN in wmt ; do
    # If domain is wmt
    if [ "$DOMAIN" = "wmt" ]; then
        LANGUAGES="cs de et fi fr gu ha is ja kk km lt lv ps ru ta tr uk zh"
    #iterate the following code for each model and language pair
    for LANGUAGE in $LANGUAGES ; do
        for DIRECTION in to ; do
            DATA_LANG=$DATA_PATH/test

            # if DIRECTION is "from", then set "lang_pair" to "en-$LANGUAGE"
            if [ "$DIRECTION" = "from" ]; then
                lang_pair="en-$LANGUAGE"
            # if DIRECTION is "to", then set "lang_pair" to "$LANGUAGE-en"
            elif [ "$DIRECTION" = "to" ]; then
                lang_pair="$LANGUAGE-en"
            fi

            echo "Preparing ${lang_pair}"
            # Get source language
            SRC_LANGUAGE=${lang_pair%-*}
            # Get target language
            TGT_LANGUAGE=${lang_pair#*-}

            DATA_BIN=$FAIRSEQ_PATH/data-bin/m2m_natural_hallucinations/$DOMAIN/$lang_pair
            overwrite=true

            SPM_MODEL=$FAIRSEQ_PATH/spm.128k.model
            DICT=$FAIRSEQ_PATH/data_dict.128k.txt

            #only run the following code if DATA_BIN does not exist or overwrite is true
            if [ ! -d "$DATA_BIN" ] || [ "$overwrite" = true ]; then
                mkdir -p $DATA_BIN
                # if $DATA_PATH/spm does not exist, then create it
                if [ ! -d "$DATA_PATH/spm" ]; then
                    mkdir -p $DATA_PATH/spm
                fi
                echo "Tokenizing..."
                python $FAIRSEQ_PATH/scripts/spm_encode.py \
                    --model $SPM_MODEL \
                    --output_format=piece \
                    --inputs=$DATA_PATH/$lang_pair.$SRC_LANGUAGE \
                    --outputs=$DATA_PATH/spm/test.spm.$SRC_LANGUAGE 

                python $FAIRSEQ_PATH/scripts/spm_encode.py \
                    --model $SPM_MODEL \
                    --output_format=piece \
                    --inputs=$DATA_PATH/$lang_pair.$TGT_LANGUAGE \
                    --outputs=$DATA_PATH/spm/test.spm.$TGT_LANGUAGE

                echo "binarizing..."
                fairseq-preprocess \
                --source-lang $SRC_LANGUAGE --target-lang $TGT_LANGUAGE \
                --testpref $DATA_PATH/spm/test.spm \
                --thresholdsrc 0 --thresholdtgt 0 \
                --destdir $DATA_BIN \
                --srcdict $DICT --tgtdict $DICT
            fi


            for model in small100 418M 1.2B 12B; do
                echo "Generating $model"
                GENERATE_PATH=$DATA_BIN/generation/${model}/lang_pair
                if [ "$model" = "418M" ] || [ "$model" = "1.2B" ]; then
                    CUDA_VISIBLE_DEVICES=${GPU} fairseq-generate $DATA_BIN --beam 4 --sacrebleu --scoring sacrebleu --path $FAIRSEQ_PATH/checkpoints/M2M/${model}_last_checkpoint.pt --fixed-dictionary $FAIRSEQ_PATH/model_dict.128k.txt -s $SRC_LANGUAGE -t $TGT_LANGUAGE --remove-bpe 'sentencepiece' --task translation_multi_simple_epoch --lang-pairs $FAIRSEQ_PATH/language_pairs_small_models.txt --decoder-langtok --encoder-langtok src --gen-subset test --results-path $GENERATE_PATH
                elif [ "$model" = "small100" ]; then
                    CUDA_VISIBLE_DEVICES=${GPU} fairseq-generate $DATA_BIN --beam 4 --sacrebleu --scoring sacrebleu --path $FAIRSEQ_PATH/checkpoints/M2M/${model}_last_checkpoint.pt --fixed-dictionary $FAIRSEQ_PATH/model_dict.128k.txt -s $SRC_LANGUAGE -t $TGT_LANGUAGE --remove-bpe 'sentencepiece' --task translation_multi_simple_epoch --lang-pairs $FAIRSEQ_PATH/language_pairs_small_models.txt --encoder-langtok tgt --gen-subset test --results-path $GENERATE_PATH
                elif [ "$model" = "12B" ]; then
                    CUDA_VISIBLE_DEVICES=${GPUS} fairseq-generate $DATA_BIN --batch-size 1 --beam 4 --path $FAIRSEQ_PATH/checkpoints/M2M/${model}_last_checkpoint.pt --fixed-dictionary $FAIRSEQ_PATH/model_dict.128k.txt -s $SRC_LANGUAGE -t $TGT_LANGUAGE --remove-bpe 'sentencepiece' --task translation_multi_simple_epoch --lang-pairs $FAIRSEQ_PATH/language_pairs.txt --decoder-langtok --encoder-langtok src --gen-subset test --fp16 --dataset-impl mmap --distributed-world-size 1 --distributed-no-spawn --pipeline-model-parallel --pipeline-chunks 1 --pipeline-encoder-balance '[26]' --pipeline-encoder-devices '[0]' --pipeline-decoder-balance '[3,22,1]' --pipeline-decoder-devices '[0,1,0]' --results-path $GENERATE_PATH
                fi 
                cat ${GENERATE_PATH}/generate-test.txt  | grep -P '^H-'  | cut -c 3- | sort -n -k 1 | awk -F "\t" '{print $NF}' > ${GENERATE_PATH}/sys.txt
                sacrebleu $DATA_PATH/$lang_pair.$TGT_LANGUAGE < ${GENERATE_PATH}/sys.txt --tokenize spm --metrics bleu chrf --chrf-word-order 2 -f text > ${GENERATE_PATH}/metrics.txt
                # Print the first line of the metrics file that contains the spBLEU score
                head -n 1 ${GENERATE_PATH}/metrics.txt
            done
        done
    done
done