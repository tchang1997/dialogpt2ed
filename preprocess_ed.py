from argparse import ArgumentParser
import csv
import json
import os
import random

import pandas as pd
from tqdm import tqdm

import logging
logger = logging.getLogger(__file__)

def load_ed(filename):
    with open(filename, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONE)
        row_num = 0
        lines = []
        for row in csv_reader:
            if row_num == 0:
                col_names = row
            else:
                lines.append({name:val for name,val in zip(col_names, row)})
            row_num += 1
    print('Loaded {} lines from {}'.format(len(lines), filename))
    return lines


def merge_ed_rows(ed_rows):
    dialogues = []
    curr_dialogue = []
    for idx, row in enumerate(ed_rows):
        conv_id = row['conv_id']
        if curr_dialogue == []:
            curr_dialogue.append(row)
        else:
            if conv_id == curr_dialogue[0]['conv_id']:
                curr_dialogue.append(row)
            else: # new dialogue
                dialogues.append(curr_dialogue)
                curr_dialogue = []
                curr_dialogue.append(row)
    dialogues.append(curr_dialogue)
    return dialogues


def process_utterance(utterance):
    utterance = utterance.lower()
    utterance = utterance.replace('_comma_', ',')
    return utterance


def make_dialogue(dialogue_in, all_utterances, num_candidates=20):
    turns = []

    # make turns
    for i in range(1, len(dialogue_in), 2):

        # get candidates
        gold_utterance = dialogue_in[i]['utterance']
        while True:
            candidates = random.sample(all_utterances, num_candidates-1)
            if gold_utterance not in candidates:
                break
        candidates.append(gold_utterance)  # length num_candidates
        candidates = [process_utterance(c) for c in candidates]

        # get history
        history = [row['utterance'] for row in dialogue_in[:i]]
        history = [process_utterance(h) for h in history]

        # append
        turns.append({'candidates': candidates, 'history': history})

    dialogue_out = {
        'emotion': dialogue_in[0]['context'],
        'situation': process_utterance(dialogue_in[0]['prompt']),
        'turns': turns,
       }

    return dialogue_out


def make_ed_data(ed_rows):

    # merge the rows into dialogues
    ed_merged_rows = merge_ed_rows(ed_rows)

    # get set of all utterances
    all_utterances = []
    for dialogue in ed_merged_rows:
        for row in dialogue:
            all_utterances.append(row['utterance'])
    all_utterances = set(all_utterances)

    # convert to new format
    ed_data = []
    print('Converting {} dialogues to new format...'.format(len(ed_merged_rows)))
    for idx, dialogue in enumerate(tqdm(ed_merged_rows)): # TODO: multiprocessing.Pool
        ed_data.append(make_dialogue(dialogue, all_utterances))
    return ed_data


def save_ed_data(data_dict, ed_path_out):
    print('Saving to {}...'.format(ed_path_out))
    with open(ed_path_out, 'w') as f:
        json.dump(data_dict, f)
    print('Done!')


def parser():
    psr = ArgumentParser()
    psr.add_argument("--data-dir", type=str, required=True, help="Input directory for ED data.")
    psr.add_argument("--out-file", type=str, required=True, help="Output file for merged, preprocessed data.")
    psr.add_argument("--split", type=str, default=['train', 'valid', 'test'], nargs='+', choices=['train', 'valid', 'test'], help="Splits to preprocess.")
    psr.add_argument("--force-preprocess", action='store_true', help="Force preprocess the dataset if out_file exists.")
    return psr


if __name__ == '__main__':
    args = parser().parse_args()
    data_dict = dict()
    if os.path.isfile(args.out_file) and not args.force_preprocess:
        logger.warning("Specified output file already exists; previewing only. Use the `--force-preprocess` command line option if to clobber the output file.")
    else:
        if 'train' in args.split:
            ed_rows_train = load_ed(os.path.join(args.data_dir, "train.csv"))
            data_dict['train'] = make_ed_data(ed_rows_train)
        if 'valid' in args.split:
            ed_rows_valid = load_ed(os.path.join(args.data_dir, "valid.csv"))
            data_dict['valid'] = make_ed_data(ed_rows_valid)
        if 'test' in args.split:
            ed_rows_test = load_ed(os.path.join(args.data_dir, "test.csv"))
            data_dict['test'] = make_ed_data(ed_rows_test)
        save_ed_data(data_dict, args.out_file)

    print("Preview:")
    with open(args.out_file, 'r') as f:
        test_json = json.load(f)
    demo_key = next(iter(test_json.keys()))
    print(pd.DataFrame(test_json[demo_key]).head())
