GPU=0
DOMAIN=flores
PROJECT_DIR=$pwd
FAIRSEQ_PATH=${PROJECT_DIR}/fairseq
STOPES_PATH=${PROJECT_DIR}/stopes

# de nl sv fr pt es cs pl ru fi hu lt el tr ja ko vi zh bn hi ta id ar he fa sw ast cy az oc hr ps
# sv fr pt es cs pl ru fi hu lt el tr ja ko vi zh bn hi ta id ar he fa sw ast cy az oc hr ps
# de nl sv fr pt es cs pl ru fi hu lt el tr ja ko vi zh bn hi ta id ar he fa sw ast cy az oc hr ps
for DOMAIN in flores ; do
    # Flores languages
    if [ "$DOMAIN" = "flores" ]; then
        LANGUAGES="de nl sv fr pt es cs pl ru fi hu lt el tr ja ko vi zh bn hi ta id ar he fa sw ast cy az oc hr ps"
    # WMT languages
    elif [ "$DOMAIN" = "wmt" ]; then
        LANGUAGES="cs de et fi fr gu ha is ja kk km lt lv ps ru ta tr uk zh"
    fi
    #iterate the following code for each model and language pair
    for LANGUAGE in $LANGUAGES ; do
        for DIRECTION in to from ; do
            for MODEL in small100 418M 1.2B 12B; do
                # if DIRECTION is "from", then set "lang_pair" to "en-$LANGUAGE"
                if [ "$DIRECTION" = "from" ]; then
                    lang_pair="en-$LANGUAGE"
                # if DIRECTION is "to", then set "lang_pair" to "$LANGUAGE-en"
                elif [ "$DIRECTION" = "to" ]; then
                    lang_pair="$LANGUAGE-en"
                fi

                echo "Preparing ${lang_pair} for ${MODEL} on ${DOMAIN}..."

                # Get source language
                SRC_LANGUAGE=${lang_pair%-*}
                # Get target language
                TGT_LANGUAGE=${lang_pair#*-}

                # Generate DATA_BIN for alti scores
                DATA_BIN=$FAIRSEQ_PATH/alti-bin/m2m_natural_hallucinations/$DOMAIN/$lang_pair/${MODEL}

                # Obtain the path to the model translations
                GEN_DATA_BIN=$FAIRSEQ_PATH/data-bin/m2m_natural_hallucinations/$DOMAIN/$lang_pair
                GENERATE_PATH=$GEN_DATA_BIN/generation/${MODEL}/lang_pair

                SPM_MODEL=$FAIRSEQ_PATH/spm.128k.model
                DICT=$FAIRSEQ_PATH/data_dict.128k.txt
                MODEL_DICT=$FAIRSEQ_PATH/model_dict.128k.txt

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
                elif [ "$MODEL" = "12B" ]; then
                    # confirm whether hydra-core version is 1.0.7
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
done