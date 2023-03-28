# Download all data
# FLORES
wget https://web.tecnico.ulisboa.pt/~ist178550/flores_test.zip
unzip flores_test.zip
rm -r flores_test.zip

# WMT   
# Set main path
main_path=${pwd}

# Check if sacrebleu is installed; if not, download it via pip install sacrebleu
if ! command -v sacrebleu &> /dev/null
then
    pip install sacrebleu
fi

# Path to save the mafand_mt files
save_path=$main_path/wmt_data

# Create the save folder path if it doesn't exist
if [ ! -d "$save_path" ]; then
    mkdir -p $save_path
fi

# Create bash mapping between languages and WMT test set year, e.g. {"en-ps": 20, "en-gu": 19, ...}
declare -A wmt_test_set_year
wmt_test_set_year["en-de"]=19
wmt_test_set_year["en-ru"]=19
wmt_test_set_year["en-zh"]=19
wmt_test_set_year["en-lt"]=19
wmt_test_set_year["en-fr"]=14
wmt_test_set_year["en-lv"]=17
wmt_test_set_year["en-tr"]=17
wmt_test_set_year["en-et"]=18
wmt_test_set_year["en-fi"]=17
wmt_test_set_year["en-cs"]=18
wmt_test_set_year["en-kk"]=19
wmt_test_set_year["en-gu"]=19
wmt_test_set_year["en-ps"]=20
wmt_test_set_year["en-km"]=20
wmt_test_set_year["en-ta"]=20
wmt_test_set_year["en-is"]=21
wmt_test_set_year["en-ha"]=21
wmt_test_set_year["en-uk"]=22
wmt_test_set_year["en-ja"]=22

# For each language pair, run the following code: sacrebleu -t wmt{year} -l {lang_pair} --echo src > wmt20.{lang_pair}.{src_lang}; and sacrebleu -t wmt{year} -l {lang_pair} --echo ref > wmt20.{lang_pair}.{trg_lang}
for lang_pair in "${!wmt_test_set_year[@]}"; do
    year=${wmt_test_set_year[$lang_pair]}
    src_lang=${lang_pair%-*}
    trg_lang=${lang_pair#*-}
    sacrebleu -t wmt$year -l $lang_pair --echo src > $save_path/wmt$year.$lang_pair.$src_lang
    sacrebleu -t wmt$year -l $lang_pair --echo ref > $save_path/wmt$year.$lang_pair.$trg_lang
done

# Do the same but for the reverse direction of the language pairs
for lang_pair in "${!wmt_test_set_year[@]}"; do
    year=${wmt_test_set_year[$lang_pair]}
    rev_lang_pair=${lang_pair#*-}-${lang_pair%-*}
    src_lang=${rev_lang_pair%-*}
    trg_lang=${rev_lang_pair#*-}
    sacrebleu -t wmt$year -l $rev_lang_pair --echo src > $save_path/wmt$year.$rev_lang_pair.$src_lang
    sacrebleu -t wmt$year -l $rev_lang_pair --echo ref > $save_path/wmt$year.$rev_lang_pair.$trg_lang
done

# All files start with wmt{year}.{lang_pair}.{src_lang} or wmt{year}.{lang_pair}.{trg_lang}; 
# remove the wmt{year} prefix from the file name
for file in $save_path/wmt*; do
    # rename the file by removing the wmt{year} prefix from the file name
    mv $file $(echo $file | sed 's/wmt[0-9]*\.//')
done

# TICO
# Set main path
main_path=${pwd}

# Path to save the WAT files
save_path=$main_path/tico_data
# Create the save folder path if it doesn't exist
if [ ! -d "$save_path" ]; then
    mkdir -p $save_path
fi

cd $save_path

# Download the data
wget https://web.tecnico.ulisboa.pt/~ist178550/tico_data.zip
# Extract the data
unzip tico_data.zip
rm tico_data.zip

cd $save_path/concatenated_test


# # Each file in the folder is of the sort "test.{lang1}" and "test.{lang2}"
# # Copy the files to the save_path directory with the format "{lang1}-{lang2}.{lang1}" and "{lang1}-{lang2}.{lang2}"
languages="am ar bn es fr ha hi id km mr ne ps pt ru so sw ta tl zh zu"
# Create files by pairing each language with english
for lang1 in $languages; do
    cp test.$lang1 $save_path/$lang1-en.$lang1
    cp test.en $save_path/$lang1-en.en

    # Add the reverse direction
    cp test.$lang1 $save_path/en-$lang1.$lang1
    cp test.en $save_path/en-$lang1.en
    
done


# Remove the test files
rm test.*

cd .. 
rm -r concatenated_test