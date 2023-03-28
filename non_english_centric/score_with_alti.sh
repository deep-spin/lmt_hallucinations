GPU=0
PROJECT_DIR=/home/nunomg/llm-hallucination
FAIRSEQ_PATH=${PROJECT_DIR}/fairseq
STOPES_PATH=${PROJECT_DIR}/stopes

# Mapping from language code to flores language code
declare -A lang_map
lang_map=( ["nl"]="nld" ["en"]="eng" ["de"]="deu" ["fr"]="fra" ["sv"]="swe" ["pt"]="por" ["es"]="spa"\
            ["cs"]="ces" ["pl"]="pol" ["ru"]="rus" ["fi"]="fin" ["hu"]="hun" ["lt"]="lit" ["el"]="ell" ["tr"]="tur" ["ja"]="jpn"\
            ["vi"]="vie" ["bn"]="ben" ["hi"]="hin" ["id"]="ind" ["mk"]="mkd" ["ta"]="tam" ["sw"]="swh" ["mr"]="mar"\
            ["ko"]="kor" ["zh"]="zho_simpl" ["tl"]="tgl" ["sr"]="srp" ["km"]="khm" ["ar"]="ara" ["he"]="heb" ["fa"]="fas"\
            ["af"]="afr" ["xh"]="xho" ["zu"]="zul" ["kk"]="kaz" ["hr"]="hrv" ["be"]="bel" ["sr"]="srp" ["ro"]="ron"\
            ["uk"]="ukr" ["hy"]="hye" ["sk"]="slk" ["it"]="ita")
LANGUAGES="hi-bn hi-mr hi-ta af-xh af-zu xh-zu ar-fr fr-sw kk-ru zh-ta de-hr de-hu nl-fr nl-de be-ru hr-sr hr-hu hr-cs hr-sk el-tr cs-sk fi-sw it-fr it-de ro-ru ro-uk hy-hr hy-sr ro-de ro-hu ro-tr ro-hy uk-ru"
#LANGUAGES="hi-bn hi-mr hi-ta af-xh af-zu xh-zu ar-fr fr-sw kk-ru zh-ta de-hr de-hu nl-fr nl-de be-ru hr-sr hr-hu hr-cs hr-sk el-tr cs-sk fi-sw it-fr it-de ro-ru ro-uk hy-hr hy-sr ro-de ro-hu ro-tr ro-hy uk-ru"


#iterate the following code for each model and language pair
for lang_pair in $LANGUAGES ; do
    for MODEL in small100 418M 1.2B 12B; do
        echo "Preparing ${lang_pair}"

        # Get source language
        SRC_LANGUAGE=${lang_pair%-*}
        # Get target language
        TGT_LANGUAGE=${lang_pair#*-}

        # Generate DATA_BIN for alti scores
        DATA_BIN=$FAIRSEQ_PATH/alti-bin/m2m_natural_hallucinations/non_english_centric/$lang_pair/${MODEL}

        # Obtain the path to the model translations
        GEN_DATA_BIN=$FAIRSEQ_PATH/data-bin/m2m_non_english_centric/$lang_pair
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