GPU=1
DOMAIN=flores
DATA_PATH=$pwd/flores_data/flores101_dataset
FAIRSEQ_PATH=$pwd/fairseq

# FLORES SETUP
# Bridge languages
# en nl de sv fr pt es cs pl ru fi hu lt el tr ja ko vi zh bn hi ta id ar he fa sw
# Additional low-resource languages
# ast cy az oc hr ps

declare -A lang_map
lang_map=( ["nl"]="nld" ["en"]="eng" ["de"]="deu" ["fr"]="fra" ["sv"]="swe" ["pt"]="por" ["es"]="spa"\
            ["cs"]="ces" ["pl"]="pol" ["ru"]="rus" ["fi"]="fin" ["hu"]="hun" ["lt"]="lit" ["el"]="ell" ["tr"]="tur" ["ja"]="jpn"\
            ["vi"]="vie" ["bn"]="ben" ["hi"]="hin" ["id"]="ind" ["mk"]="mkd" ["ta"]="tam" ["sw"]="swh" ["mr"]="mar"\
            ["ko"]="kor" ["zh"]="zho_simpl" ["tl"]="tgl" ["sr"]="srp" ["km"]="khm" ["ar"]="ara" ["he"]="heb" ["fa"]="fas"\
            ["ast"]="ast" ["cy"]="cym" ["be"]="bel" ["oc"]="oci" ["hr"]="hrv" ["ps"]="pus" ["az"]="azj" )

#iterate the following code for each model and language pair
for LANGUAGE in de nl sv fr pt es cs pl ru fi hu lt el tr ja ko vi zh bn hi ta id ar he fa sw ast cy az oc hr ps ; do
    for DIRECTION in from to ; do
        echo "Preparing ${LANGUAGE}"
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

        # Get source language code for flores
        SRC_LANG_CODE=${lang_map[$SRC_LANGUAGE]}
        # Get target language code for flores
        TGT_LANG_CODE=${lang_map[$TGT_LANGUAGE]}

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
                --inputs=$DATA_PATH/test/$SRC_LANG_CODE.test \
                --outputs=$DATA_PATH/spm/test.spm.$SRC_LANGUAGE 

            python $FAIRSEQ_PATH/scripts/spm_encode.py \
                --model $SPM_MODEL \
                --output_format=piece \
                --inputs=$DATA_PATH/test/$TGT_LANG_CODE.test \
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
            sacrebleu $DATA_PATH/test/$TGT_LANG_CODE.test < ${GENERATE_PATH}/sys.txt --tokenize spm --metrics bleu chrf --chrf-word-order 2 -f text > ${GENERATE_PATH}/metrics.txt
            # Print the first line of the metrics file with contains the spBLEU score
            head -n 1 ${GENERATE_PATH}/metrics.txt
        done
    done
done