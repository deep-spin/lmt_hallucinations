import argparse
import pandas as pd
import torch
import os

from tqdm import tqdm
from fairseq.alti import AltiMultilingualHubInterface

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--df_path', type=str)
    parser.add_argument("--src_lang", type=str)
    parser.add_argument("--tgt_lang", type=str)
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--model_dict", type=str)
    parser.add_argument("--lang_pair_path", type=str)
    return parser.parse_args()

def contribution_source(total_rollout, src_tensor):
    # Get total source contribution
    src_total_alti = total_rollout[-1][:,:src_tensor.size(0)].sum(dim=-1)
    # We delete the first prediction (only source and bos contribution)
    src_total_alti = src_total_alti[1:]
    return src_total_alti

def main():
    args = parse_args()

    model_dict = args.model_dict
    ckpt_path = args.ckpt_path
    source_lang = args.src_lang
    target_lang = args.tgt_lang

    # checkpoint_dir is the folder where the checkpoint is located
    # checkpoint_file is the name of the checkpoint file
    checkpoint_dir = os.path.dirname(ckpt_path)
    checkpoint_file = os.path.basename(ckpt_path)

    # the checkpoint file finishes with "model_last_checkpoint.pt"; obtain the name of the model
    model_name = checkpoint_file.split("_")[0]

    lang_pairs = args.lang_pair_path
    with open(lang_pairs, "r") as f:
        lang_pairs = f.read()[:-1]

    hub = AltiMultilingualHubInterface.from_pretrained(
        checkpoint_dir=checkpoint_dir,
        checkpoint_file=checkpoint_file,
        data_name_or_path=args.data_path,
        source_lang=source_lang,
        target_lang=target_lang,
        lang_pairs=lang_pairs,
        fixed_dictionary=model_dict,
        pipeline=True,
        distributed_world_size=1,
        dataset_impl="mmap",
        fp16=True,
        distributed_no_spawn=True,
        pipeline_model_parallel=True,
        pipeline_chunks=1,
        pipeline_encoder_balance='[26]',
        pipeline_encoder_devices='[0]',
        pipeline_decoder_balance='[3,22,1]',
        pipeline_decoder_devices='[0,1,0]',
    )
    
    dataset_to_use = pd.read_pickle(args.df_path)
    # Compute alti for source in dataset_to_use
    alti_scores = []
    for idx in tqdm(dataset_to_use.index):
        src_tensor = torch.tensor(dataset_to_use.loc[idx, 'src_ids'])
        tgt_tensor = torch.tensor(dataset_to_use.loc[idx, 'mt_ids'][:-1])

        # Add id 2 to the beginning of the target sentence
        tgt_tensor = torch.cat((torch.tensor([2]), tgt_tensor))

        src_sent = dataset_to_use.loc[idx, 'src']
        tgt_sent = dataset_to_use.loc[idx, 'mt']
        
        with torch.no_grad():
            total_alti = hub.get_contribution_rollout(src_tensor, tgt_tensor, 'l1', norm_mode='min_sum')['total']
        alti_score = contribution_source(total_alti, src_tensor)
        alti_score = torch.mean(alti_score).item()
        alti_scores.append({"alti": alti_score, "src": src_sent, "tgt": tgt_sent})

    dataset_to_use['alti'] = [alti_scores[i]['alti'] for i in range(len(alti_scores))]
    
    print(f"Saving alti scores for model {model_name} to pickle file.")
    alti_path = os.path.join(args.data_path, "alti", model_name)
    if not os.path.exists(alti_path):
        os.makedirs(alti_path)
    dataset_to_use.to_pickle(os.path.join(alti_path, "df_w_alti.pkl"))


    print(dataset_to_use['alti'].describe(percentiles=[.25, .5, .75]))
     # Save that info to a text file
    with open(os.path.join(args.data_path, "alti", model_name, "alti_scores.txt"), "w") as f:
        f.write(dataset_to_use['alti'].describe(percentiles=[.25, .5, .75]).to_string())



if __name__ == "__main__":
    main()