from __future__ import annotations

import os
import pathlib
import csv
#import gradio as gr
import numpy as np
import torch
import torchaudio
from fairseq2.assets import InProcAssetMetadataProvider, asset_store
from huggingface_hub import snapshot_download
from seamless_communication.inference import Translator
import argparse
import glob
from lang_list import (
    ASR_TARGET_LANGUAGE_NAMES,
    LANGUAGE_NAME_TO_CODE,
    S2ST_TARGET_LANGUAGE_NAMES,
    S2TT_TARGET_LANGUAGE_NAMES,
    T2ST_TARGET_LANGUAGE_NAMES,
    T2TT_TARGET_LANGUAGE_NAMES,
    TEXT_SOURCE_LANGUAGE_NAMES,
)


AUDIO_SAMPLE_RATE = 16000.0
MAX_INPUT_AUDIO_LENGTH = 60  # in seconds
DEFAULT_TARGET_LANGUAGE = "Spanish"




parser = argparse.ArgumentParser(description='Manual Evaluation code with Speech-to-Text-Translation and ASR')

parser.add_argument('--input_speech_dir', type=str, 
					help='Path of directory containing wav speech_files', required=True)

parser.add_argument('--input_speech_file', type=str, help='.wav file', required=False)

parser.add_argument('--source_language', type=str, 
					help='source language of the wav files', required=True)
parser.add_argument('--target_language', type=str, 
					help='target language for the wav text transcription', required=True)

parser.add_argument('--s2tt',action='store_true', help='Speech to Text Conversion')
parser.add_argument('--asr',action='store_true', help='ASR')
args = parser.parse_args()












def preprocess_audio(input_audio: str) -> None:
	    arr, org_sr = torchaudio.load(input_audio)
	    new_arr = torchaudio.functional.resample(arr, orig_freq=org_sr, new_freq=AUDIO_SAMPLE_RATE)
	    max_length = int(MAX_INPUT_AUDIO_LENGTH * AUDIO_SAMPLE_RATE)
	    if new_arr.shape[1] > max_length:
	        new_arr = new_arr[:, :max_length]
	        gr.Warning(f"Input audio is too long. Only the first {MAX_INPUT_AUDIO_LENGTH} seconds is used.")
	    # name = input_audio.split('/')[-1]
	    # if not os.path.isdir('preprocessed_audio_samples'):
	    # 	os.mkdir('preprocessed_audio_samples')
	    torchaudio.save(input_audio, new_arr, sample_rate=int(AUDIO_SAMPLE_RATE))


def main():
	if not os.path.isdir(args.input_speech_dir):
		raise ValueError('--input_speech_dir argument must be a valid path to the input wav files directory')
	if not args.input_speech_dir:
		if not os.path.isfile(args.input_speech_file):
			raise ValueError('--input_speech_fil argument must be a valid path to the input wav file path if the input speech directory is not given')
	
	
	if torch.cuda.is_available():
	    device = torch.device("cuda:0")
	    dtype = torch.float16
	else:
	    device = torch.device("cpu")
	    dtype = torch.float32

	translator = Translator(
	model_name_or_card="seamlessM4T_v2_large",
	vocoder_name_or_card="vocoder_v2",
	device=device,
	dtype=dtype,
	apply_mintox=True,
	)
	if args.s2tt:
		if args.input_speech_dir:
			dir_name = args.input_speech_dir.split('/')[-1]
			speech_files = glob.glob(f'{dir_name}/*.wav')
			text = []
			for i in speech_files:
				preprocess_audio(i)
				name = i.split('/')[-1]
				source_language_code = LANGUAGE_NAME_TO_CODE[args.source_language]
				target_language_code = LANGUAGE_NAME_TO_CODE[args.target_language]
				out_texts, _ = translator.predict(input=i,
		        task_str="S2TT",
		        src_lang=source_language_code,
		        tgt_lang=target_language_code)
				#print(str(out_texts[0]))
				text.append((name.strip('.wav'),str(out_texts[0])))
			with open(f'{dir_name}_s2tt.txt', 'a', newline='') as f:
				writer = csv.writer(f,delimiter = '\t')
				writer.writerows(text)

		else:
			name = args.input_speech_file.split('/')[-1]
			text = []
			preprocess_audio(args.input_speech_file)
			source_language_code = LANGUAGE_NAME_TO_CODE[args.source_language]
			target_language_code = LANGUAGE_NAME_TO_CODE[args.target_language]
			out_texts, _ = translator.predict(input=i,
	        task_str="S2TT",
	        src_lang=source_language_code,
	        tgt_lang=target_language_code)
			#print(str(out_texts[0]))
			text.append((name.strip('.wav'),str(out_texts[0])))

			with open(f'{name}_s2tt.txt', 'a', newline='') as f:
			    writer = csv.writer(f,delimiter = '\t')
			    writer.writerows(text)

	if args.asr:
		if args.input_speech_dir:
			dir_name = args.input_speech_dir.split('/')[-1]
			speech_files = glob.glob(f'{dir_name}/*.wav')
			text = []
			for i in speech_files:
				preprocess_audio(i)
				name = i.split('/')[-1]
				target_language_code = LANGUAGE_NAME_TO_CODE[args.target_language]
				out_texts, _ = translator.predict(
		        input=i,
		        task_str="ASR",
		        src_lang=target_language_code,
		        tgt_lang=target_language_code,)
				#print(str(out_texts[0]))
				text.append((name.strip('.wav'),str(out_texts[0])))
			with open(f'{dir_name}_asr.txt', 'a', newline='') as f:
				writer = csv.writer(f,delimiter = '\t')
				writer.writerows(text)

		else:
			name = args.input_speech_file.split('/')[-1]
			text = []
			preprocess_audio(args.input_speech_file)
			#source_language_code = LANGUAGE_NAME_TO_CODE[args.source_language]
			target_language_code = LANGUAGE_NAME_TO_CODE[args.target_language]
			out_texts, _ = translator.predict(input=i,
	        task_str="ASR",
	        src_lang=target_language_code,
	        tgt_lang=target_language_code)
			#print(str(out_texts[0]))
			text.append((name.strip('.wav'),str(out_texts[0])))

			with open(f'{name}_asr.txt', 'a', newline='') as f:
			    writer = csv.writer(f,delimiter = '\t')
			    writer.writerows(text)




if __name__=="__main__":
    main()
