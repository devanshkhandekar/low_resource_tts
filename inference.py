import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from pathlib import Path

import soundfile as sf
import torch
from espnet2.text.token_id_converter import TokenIDConverter
from parallel_wavegan.utils import load_model as load_vocoder
from espnet2.bin.tts_inference import Text2Speech
import sox
import glob
import json
import argparse
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import soundfile as sf
import torch
from packaging.version import parse as V
from typeguard import check_argument_types

from espnet2.fileio.npy_scp import NpyScpWriter
from espnet2.gan_tts.vits import VITS
from espnet2.tasks.tts import TTSTask
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.tts.fastspeech import FastSpeech
from espnet2.tts.fastspeech2 import FastSpeech2
from espnet2.tts.tacotron2 import Tacotron2
from espnet2.tts.transformer import Transformer
from espnet2.tts.utils import DurationCalculator
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool, str2triple_str, str_or_none
from espnet.utils.cli_utils import get_commandline_args



import librosa

logging.getLogger('numba').setLevel(logging.WARNING)
from scipy.io.wavfile import write
from transformers import WavLMModel

from FreeVC import utils
from FreeVC.models import SynthesizerTrn
from FreeVC.mel_processing import mel_spectrogram_torch
from FreeVC.speaker_encoder.voice_encoder import SpeakerEncoder



from nsnet2_denoiser import NSnet2Enhancer
enhancer = NSnet2Enhancer(fs=16000)
import soundfile as sf

language_code = {'unknown': 0,'czech' : 1 ,'german': 2,'greek' : 3, 'english-UK' : 4,'english-US' : 5,'spanish' : 6, 'estonian' : 7, 'finnish' : 8,
 'french' : 9,'croatian' : 10,'hungarian' : 11, 'italian'  : 12,'lithuanian' : 13,'dutch' : 14,'polish' : 15,'romanian' : 16,'russian'  : 17,
'slovak' : 18,'slovenian' : 19, 'ukrainian' : 20 } 

from nsnet2_denoiser import NSnet2Enhancer
import soundfile as sf



enhancer = NSnet2Enhancer(fs=16000)

parser = argparse.ArgumentParser(description='Inference code for TTS with Voice Conversion')

parser.add_argument('--model_path', type=str, 
					help='Name of saved checkpoint to load weights from', required=True)

parser.add_argument('--input_file', type=str, help='input file for text2speech', required=True)

parser.add_argument('--training_config_file', type=str, 
					help='config.yaml for training', required=True)
parser.add_argument('--inference_config_file', type=str, 
					help='config.yaml for inference', required=True)
parser.add_argument('--output_dir', type=str, help='output directory to save the wav files', 
								default=None)

parser.add_argument('--source_language',help=['unknown','czech','german','greek','english-UK','english-US','spanish','estonian','finnish','french','croatian','hungarian','italian','lithuanian','dutch','polish','romanian','russian','slovak','slovenian','ukrainian'],type=str, required=True)


parser.add_argument('--vocoder_file', type=str, help='hifigan vocoder checkpoint file', default='./vocoder/hifigan16k_libritts_css10_vctk/checkpoint-2000000steps.pkl')
parser.add_argument('--vocoder_config', type=str, help='hifigan vocoder config file', default='./vocoder/hifigan16k_libritts_css10_vctk/config.yml')

parser.add_argument('--voice_conversion', type=bool, help='Voice Conversion', default=False)

parser.add_argument('--vc_target_speaker', type=str, help='[p226_002.wav , p225_001.wav]', default= 'p226_002.wav')
parser.add_argument('--concatenate', type=bool, help='concatenate the generated wav files', default=False)



args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 0 


language_list = ['unknown','czech','german','greek','english-UK','english-US','spanish','estonian','finnish','french','croatian','hungarian','italian','lithuanian','dutch','polish','romanian','russian','slovak','slovenian','ukrainian']

def main():
	if not os.path.isfile(args.model_path):
		raise ValueError('--model_path argument must be a valid path to the saved checkpoint file')

	if not os.path.isfile(args.input_file):
		raise ValueError('--input_file argument must be a valid path to the input tts document')

	if not os.path.isfile(args.training_config_file):
		raise ValueError('--training_config_file argument must be a valid path to the training config file')

	if not os.path.isfile(args.inference_config_file):
		raise ValueError('--inference_config_file argument must be a valid path to the inference config file')

	if not os.path.isfile(args.vocoder_config):
		raise ValueError('--vocoder_config argument must be a valid path to the vocoder config file')

	if not os.path.isfile(args.vocoder_file):
		raise ValueError('--vocoder_file argument must be a valid path to the vocoder_file')

	if args.source_language not in language_list:
		raise ValueError(f'--source_language argument must be a valid language from {language_list}')

	if args.voice_conversion:
		if not os.path.isfile(args.vc_target_speaker):
			raise ValueError(f'--vc_target_speaker argument must be either of p226_002.wav,p225_001.wav file path')

	num_workers = 1
	batch_size = 1
	lid = torch.tensor(language_code[args.source_language]).to(device).long()
	text2speech_kwargs = dict(
        train_config=args.training_config_file,
        model_file=args.model_path,
        threshold=0.5,
        maxlenratio=10.0,
        minlenratio=0.0,
        use_teacher_forcing=False,
        use_att_constraint=True,
        backward_window=1,
        forward_window=3,
        speed_control_alpha=1.0,
        noise_scale=0.667,
        noise_scale_dur=0.8,
        vocoder_config=args.vocoder_config,
        vocoder_file=args.vocoder_file,
        dtype='float32',
        device='cuda',
        seed=seed,
        always_fix_seed=False
    )
	print('Loading TTS model')
	text2speech = Text2Speech.from_pretrained(**text2speech_kwargs)
	if args.voice_conversion:
		print("Loading FreeVC Voice Conversion Model")
		hps = utils.get_hparams_from_file("FreeVC/configs/freevc.json")
		freevc = SynthesizerTrn(
		    hps.data.filter_length // 2 + 1,
		    hps.train.segment_size // hps.data.hop_length,
		    **hps.model).to(device)
		_ = freevc.eval()
		_ = utils.load_checkpoint("FreeVC/checkpoints/freevc.pth", freevc, None)
		smodel = SpeakerEncoder('FreeVC/speaker_encoder/ckpt/pretrained_bak_5805000.pt')
		cmodel = WavLMModel.from_pretrained("microsoft/wavlm-large").to(device)


	text2speech.tts.use_gst = False
	text2speech.tts.spk_embed_dim = None
	text2speech.tts.use_encoder_w_lid = True

	data_path_and_name_and_type = [(args.input_file,'text','text')]
	key_file = args.input_file

	loader = TTSTask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype='float32',
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=TTSTask.build_preprocess_fn(text2speech.train_args, False),
        collate_fn=TTSTask.build_collate_fn(text2speech.train_args, False),
        allow_variable_data_keys=True,
        inference=True)
	if args.output_dir is None:
		output_dir = f"{args.input_file.split('/')[-1]}_inference_result"
	Path(f"{output_dir}").mkdir(parents=True, exist_ok=True)
	Path(f"{output_dir}/norm").mkdir(parents=True, exist_ok=True)
	Path(f"{output_dir}/denorm").mkdir(parents=True, exist_ok=True)
	Path(f"{output_dir}/denoised").mkdir(parents=True, exist_ok=True)
	with NpyScpWriter(output_dir+"/norm",output_dir+"/norm/feats.scp",) as norm_writer,NpyScpWriter(
		output_dir+"/denorm", output_dir+"/denorm/feats.scp") as denorm_writer:
		for idx, (keys, batch) in enumerate(loader, 1):
			key = keys[0]
			_bs = len(next(iter(batch.values())))
			batch = {k: v[0] for k, v in batch.items() if not k.endswith("_lengths")}
			batch.update(lids=lid)
			output_dict = text2speech(**batch)
			feat_gen = output_dict["feat_gen"]
			norm_writer[key] = output_dict["feat_gen"].cpu().numpy()
			denorm_writer[key] = output_dict["feat_gen_denorm"].cpu().numpy()
			wav = output_dict["wav"]
			denoised_wav = enhancer(wav.cpu().numpy(), 16000)
			if args.voice_conversion:
				spk = Path(args.vc_target_speaker).name.split('_')[0]
				Path(f"{output_dir}/voice_conversion_{spk}").mkdir(parents=True, exist_ok=True)
				with torch.no_grad():
					wav_tgt, _ = librosa.load(args.vc_target_speaker, sr=hps.data.sampling_rate)
					wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)
					g_tgt = smodel.embed_utterance(wav_tgt)
					g_tgt = torch.from_numpy(g_tgt).unsqueeze(0).to(device)
					#wav_src, _ = librosa.load(src, sr=hps.data.sampling_rate)
					wav_src = torch.from_numpy(denoised_wav).unsqueeze(0).to(device,dtype=torch.float)
					
					c = cmodel(wav_src).last_hidden_state.transpose(1, 2).to(device)
					audio = freevc.infer(c, g=g_tgt)
					audio = audio[0][0].data.cpu().float().numpy()
					sf.write(f"{output_dir}/voice_conversion_{spk}/{key}_vc_{spk}.wav",audio,hps.data.sampling_rate,"PCM_16")
		            #write(out, hps.data.sampling_rate, audio)
			else:
				sf.write(f"./{output_dir}/denoised/{key}_denoised.wav",denoised_wav,text2speech.fs,"PCM_16")

	if args.concatenate :
		if args.voice_conversion:
			wav_files = glob.glob(f"{output_dir}/voice_conversion_{spk}/*.wav")
			concat_name = 'vc_' + args.vc_target_speaker
		else:
			wav_files = glob.glob(f"./{output_dir}/denoised/*.wav")
			concat_name = 'denoised'
		cbn = sox.Combiner()
		cbn.convert(samplerate=16000, n_channels=2)
		#cbn.pitch(3.0)
		cbn.build(wav_files, f'{output_dir}_concatenate_{concat_name}_output.wav', 'concatenate')
					
				

if __name__=="__main__":
    main()
