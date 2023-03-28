GPU=0
PROJECT_DIR=/home/nunomg/llm-hallucination
FAIRSEQ_PATH=${PROJECT_DIR}/fairseq
STOPES_PATH=${PROJECT_DIR}/stopes

for DOMAIN in chat health news tico iwslt ; do
    # Define a list of languages to iterate over in a for loop; the list should be such that we can code for LANGUAGE in $LANGUAGES and LANGUAGE will be the language code
    # If domain is wmt
    if [ "$DOMAIN" = "chat" ]; then
        LANG_PAIRS="en-ru en-wo"
    elif [ "$DOMAIN" = "health" ]; then
        LANG_PAIRS="en-ru en-wo"
    elif [ "$DOMAIN" = "news" ]; then
        LANG_PAIRS="en-ru en-wo"
    fi

    DATA_PATH=/home/nunomg/llm-hallucination/nllb_md_data/$DOMAIN

    if [ "$DOMAIN" = "tico" ]; then
        LANGUAGES="am ar bn zh fr ha hi id km mr ne ps pt ru es so sw tl ta zu"
        # LANG_PAIRS are combinations of the languages in LANGUAGES with English
        LANG_PAIRS=""
        for lang in $LANGUAGES; do
            LANG_PAIRS="$LANG_PAIRS $lang-en"
        done
        # Add the reverse direction
        for lang in $LANGUAGES; do
            LANG_PAIRS="$LANG_PAIRS en-$lang"
        done
        DATA_PATH=/home/nunomg/llm-hallucination/${DOMAIN}_data
    fi

    if [ "$DOMAIN" = "iwslt" ]; then
        LANGUAGES="de it nl ro"
        LANG_PAIRS=""
        for lang in $LANGUAGES; do
            LANG_PAIRS="$LANG_PAIRS $lang-en"
        done
        # Add the reverse direction
        for lang in $LANGUAGES; do
            LANG_PAIRS="$LANG_PAIRS en-$lang"
        done
        DATA_PATH=/home/nunomg/llm-hallucination/${DOMAIN}_data
    fi

    echo $LANG_PAIRS

    #iterate the following code for each model and language pair
    for lang_pair in $LANG_PAIRS ; do
        for MODEL in small100 418M 1.2B 12B; do
            echo "Preparing ${lang_pair}"

            # Get source language
            SRC_LANGUAGE=${lang_pair%-*}
            # Get target language
            TGT_LANGUAGE=${lang_pair#*-}

            # Generate DATA_BIN for alti scores
            DATA_BIN=$FAIRSEQ_PATH/alti-bin/m2m_natural_hallucinations/specialized_domains/$lang_pair/${MODEL}

            # Obtain the path to the model translations
            GEN_DATA_BIN=$FAIRSEQ_PATH/data-bin/m2m_specialized_domains/$DOMAIN/$lang_pair
            GENERATE_PATH=$GEN_DATA_BIN/generation/${MODEL}/lang_pair

            SPM_MODEL=$FAIRSEQ_PATH/spm.128k.model
            DICT=$FAIRSEQ_PATH/data_dict.128k.txt
            MODEL_DICT=$FAIRSEQ_PATH/model_dict.128k.txt
            LANG_PAIR_PATH=$FAIRSEQ_PATH/language_pairs.txt

            #only run the following code if DATA_BIN does not exist or overwrite is true
            overwrite=false
            if [ ! -d "$DATA_BIN" ] || [ "$overwrite" = true ]; then
                mkdir -p $DATA_BIN
                DATA_PATH=${DATA_BIN}/translations
                # if $DATA_PATH/spm does not exist, then create it
                if [ ! -d "$DATA_PATH/spm" ]; then
                    mkdir -p $DATA_PATH/spm
                fi
                
                cat ${GENERATE_PATH}/generate-test.txt  | grep -P '^S-'  | cut -c 3- | sort -n -k 1 | awk -F "\t" '{print $NF}' > ${DATA_PATH}/$lang_pair.$SRC_LANGUAGE
                cat ${GENERATE_PATH}/generate-test.txt  | grep -P '^H-'  | cut -c 3- | sort -n -k 1 | awk -F "\t" '{print $NF}' > ${DATA_PATH}/$lang_pair.$TGT_LANGUAGE

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


            # The script takes an argparse with 8 arguments: data_path, df_path, fairseq_path, source_lang, target_lang, ckpt_path, spm_model, data_dict
            # Run the script on GPU

            # if model is not 12B, run the code below:
            if [ "$MODEL" != "12B" ]; then
                # confirm whether hydra-core version is 1.0.7
                if [ "$(pip3 freeze | grep hydra-core | cut -d '=' -f 3)" != "1.0.7" ]; then
                    echo "Installing hydra-core==1.0.7..."
                    pip3 install hydra-core==1.0.7
                fi
                script=${STOPES_PATH}/stopes/eval/alti/run_alti.py
                echo "Detecting with ALTI..."
                CUDA_VISIBLE_DEVICES=${GPU} python3 ${STOPES_PATH}/stopes/eval/alti/run_alti.py \
                    --data_path $DATA_BIN \
                    --df_path $GEN_DATA_BIN/dataframes/df_gen_stats_${SRC_LANGUAGE}${TGT_LANGUAGE}_${MODEL}.pkl \
                    --source_lang $SRC_LANGUAGE \
                    --target_lang $TGT_LANGUAGE \
                    --ckpt_path $FAIRSEQ_PATH/checkpoints/M2M/${MODEL}_last_checkpoint.pt \
                    --model_dict $MODEL_DICT
            # if model is 12B, run the code below:
            elif [ "$MODEL" == "12B" ]; then
                if [ "$(pip3 freeze | grep hydra-core | cut -d '=' -f 3)" != "1.0.0" ]; then
                    echo "Installing hydra-core==1.0.0..."
                    pip3 install hydra-core==1.0.0
                fi
                echo "Detecting with ALTI..."
                CUDA_VISIBLE_DEVICES=${GPUS} python3 $PROJECT_DIR/hallucinations/score_with_alti_12B.py \
                    --data_path $DATA_BIN \
                    --df_path $GEN_DATA_BIN/dataframes/df_gen_stats_${SRC_LANGUAGE}${TGT_LANGUAGE}_${MODEL}.pkl \
                    --src_lang $SRC_LANGUAGE \
                    --tgt_lang $TGT_LANGUAGE \
                    --ckpt_path $FAIRSEQ_PATH/checkpoints/M2M/${MODEL}_last_checkpoint.pt \
                    --model_dict $MODEL_DICT \
                    --lang_pair_path $LANG_PAIR_PATH
            fi
        done
    done
done