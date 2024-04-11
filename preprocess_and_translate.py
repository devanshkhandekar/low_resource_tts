import docx2txt 
import pandas as pd
import os
from sentsplit.segment import SentSplit
from pathlib import Path
import argparse
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
model = M2M100ForConditionalGeneration.from_pretrained("seyoungsong/flores101_mm100_175M")
tokenizer: M2M100Tokenizer = M2M100Tokenizer.from_pretrained("seyoungsong/flores101_mm100_175M")

tokenizer.lang_token_to_id = {t: i for t, i in zip(tokenizer.all_special_tokens, tokenizer.all_special_ids) }#if i > 5}
tokenizer.lang_code_to_token = {s.strip("_"): s for s in tokenizer.lang_token_to_id}
tokenizer.lang_code_to_id = {s.strip("_"): i for s, i in tokenizer.lang_token_to_id.items()}
tokenizer.id_to_lang_token = {i: s for s, i in tokenizer.lang_token_to_id.items()}


parser = argparse.ArgumentParser(description='Preprocessing and Translation script')
parser.add_argument('--input_file', type=str, help='input docx file for text2speech', required=True)
parser.add_argument('--source_language', type=str, 
					help='source language of the Text Document', default = 'English')
parser.add_argument('--target_language', type=str, 
					help='target language for Translation of the Text Document', required=True)
language_dict = {'Afrikaans':'af', 'Amharic' :'am', 'Arabic' : 'ar', 'Asturian' : 'ast', 'Azerbaijani' : 'az', 'Bashkir' : 'ba', 'Belarusian' :'be', 'Bulgarian' : 'bg', 'Bengali' : 'bn', 'Breton' :'br', 'Bosnian' :'bs', 'Catalan' :'ca', 'Cebuano' :'ceb', 'Czech' :'cs', 'Welsh': 'cy', 'Danish': 'da', 'German' :'de', 'Greeek' :'el', 'English' :'en', 'Spanish' :'es', 'Estonian' :'et', 'Persian' :'fa', 'Fulah' :'ff', 'Finnish' :'fi', 'French' :'fr', 'Western Frisian' :'fy', 'Irish' :'ga', 'Gaelic' : 'gd', 'Galician' :'gl', 'Gujarati' :'gu', 'Hausa': 'ha', 'Hebrew' :'he', 'Hindi' :'hi', 'Croatian' :'hr', 'Haitian': 'ht', 'Hungarian' :'hu', 'Armenian': 'hy', 'Indonesian' :'id', 'Igbo' :'ig', 'Iloko': 'ilo', 'Icelandic' :'is', 'Italian' :'it', 'Japanese' :'ja', 'Javanese' :'jv', 'Georgian' :'ka', 'Kazakh' :'kk', 'Central Khmer' :'km', 'Kannada' :'kn', 'Korean' : 'ko', 'Luxembourgish':'lb', 'Ganda' :'lg', 'Lingala' :'ln', 'Lao': 'lo', 'Lithuanian' :'lt', 'Latvian': 'lv', 'Malagasy':'mg', 'Macedonian' :'mk', 'Malayalam' :'ml', 'Mongolian' :'mn', 'Marathi' :'mr', 'Malay' :'ms', 'Burmese' :'my', 'Nepali' :'ne', 'Dutch':'nl', 'Norwegian' :'no', 'Northern Sotho' :'ns', 'Occitan' :'oc', 'Oriya' :'or', 'Punjabi' :'pa', 'Polish' :'pl', 'Pashto' :'ps', 'Portuguese' :'pt', 'Romanian':'ro', 'Russian' :'ru', 'Sindhi' :'sd','Sinhalese' :'si', 'Slovak' :'sk', 'Slovenian' :'sl', 'Somali' :'so', 'Albanian' :'sq', 'Serbian' :'sr', 'Swati' :'ss', 'Sundanese' :'su', 'Swedish' :'sv', 'Swahili' :'sw', 'Tamil' :'ta', 'Thai' :'th', 'Tagalog' :'tl', 'Tswana' :'tn', 'Turkish' :'tr', 'Ukrainian' :'uk', 'Urdu' :'ur', 'Uzbek' :'uz', 'Vietnamese' :'vi', 'Wolof':'wo', 'Xhosa' :'xh', 'Yiddish' :'yi', 'Yoruba' :'yo', 'Chinese' :'zh', 'Zulu' :'zu'}
args = parser.parse_args()
def main():
	if not os.path.isfile(args.input_file):
		raise ValueError('--input_file argument must be a valid path to the .docx file')

	if args.target_language not in language_dict:
		raise ValueError("--target_language must be one of 'Afrikaans', 'Amharic', 'Arabic', 'Asturian', 'Azerbaijani', 'Bashkir', 'Belarusian', 'Bulgarian', 'Bengali', 'Breton', 'Bosnian', 'Catalan', 'Cebuano', 'Czech', 'Welsh', 'Danish', 'German', 'Greeek', 'English', 'Spanish', 'Estonian', 'Persian', 'Fulah', 'Finnish', 'French', 'Western Frisian', 'Irish', 'Gaelic', 'Galician', 'Gujarati', 'Hausa', 'Hebrew', 'Hindi', 'Croatian', 'Haitian', 'Hungarian', 'Armenian', 'Indonesian', 'Igbo', 'Iloko', 'Icelandic', 'Italian', 'Japanese', 'Javanese', 'Georgian', 'Kazakh', 'Central Khmer', 'Kannada', 'Korean', 'Luxembourgish', 'Ganda', 'Lingala', 'Lao', 'Lithuanian', 'Latvian', 'Malagasy', 'Macedonian', 'Malayalam', 'Mongolian', 'Marathi', 'Malay', 'Burmese', 'Nepali', 'Dutch', 'Norwegian', 'Northern Sotho', 'Occitan', 'Oriya', 'Punjabi', 'Polish', 'Pashto', 'Portuguese', 'Romanian', 'Russian', 'Sindhi', 'Sinhalese', 'Slovak', 'Slovenian', 'Somali', 'Albanian', 'Serbian', 'Swati', 'Sundanese', 'Swedish', 'Swahili', 'Tamil', 'Thai', 'Tagalog', 'Tswana', 'Turkish', 'Ukrainian', 'Urdu', 'Uzbek', 'Vietnamese', 'Wolof', 'Xhosa', 'Yiddish', 'Yoruba', 'Chinese', 'Zulu' ")
	
	text_read = docx2txt.process(args.input_file)
	sent_splitter = SentSplit('en')


	# segment a single line
	sentences = sent_splitter.segment(text_read)
	sentences = [i for i in sentences if i != '\n']
	Path("./Preprocessed_Documents").mkdir(parents=True, exist_ok=True)
	with open(f'./Preprocessed_Documents/{Path(args.input_file).stem}_doc', 'w+') as f:
	     
	    # write elements of list
	    for items in sentences:
	        f.write('%s' %items)
	     
	    print("File written successfully")

	f.close()
	translation_language = args.target_language

	translation_code = language_dict
	name = Path(f'./Preprocessed_Documents/{Path(args.input_file).stem}_doc').stem
	eng_text = open(f'./Preprocessed_Documents/{Path(args.input_file).stem}_doc','r')
	data = eng_text.read()
	data_into_list = data.split("\n")
	eng_text.close()
	translated_doc = []
	Path("Translated_Documents").mkdir(parents=True, exist_ok=True)
	for i in data_into_list:
	    tokenizer.src_lang = translation_code[args.source_language]
	    encoded_hi = tokenizer(i, return_tensors="pt")
	    generated_tokens = model.generate(**encoded_hi, forced_bos_token_id=tokenizer.get_lang_id(translation_code[translation_language]),max_new_tokens=4000)
	    translated_doc.append(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0])
	    translated_index = [f'{translation_language}_{name}_{i+1}' for i in range(len(translated_doc))]
	    translated_df = pd.DataFrame(list(zip(translated_index,translated_doc)),columns=['Name','Text'])
	    translated_df.to_csv(f'./Translated_Documents/{translation_language}_{name}',sep='\t',index=False,header=False)
	 
 












if __name__=="__main__":
    main()


