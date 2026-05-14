import sys
import csv
import torch
import pickle
import pandas as pd
from camel_tools.disambig.bert import BERTUnfactoredDisambiguator
from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer
from camel_tools.utils.dediac import dediac_ar
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.utils.charsets import AR_LETTERS_CHARSET
from camel_tools.utils.charsets import AR_DIAC_CHARSET
from camel_tools.dialectid import DialectIdentifier

def stringContainsArabic(word):
    for c in word:
        if c in AR_LETTERS_CHARSET:
            return True
    return False

def get_readability_dict(readability_level_file):
  csvreader_readability_level = csv.DictReader(readability_level_file)
  dict_readability={}
  for row in csvreader_readability_level:
    dict_readability[int(row['snippet_id'])-1]=float(row['readability_level'])
  return dict_readability

def get_dict_aldi(aldi_level_file):
  csvreader_aldi_level = csv.DictReader(aldi_level_file)
  dict_aldi={}
  for row in csvreader_aldi_level:
    #if row['ALDi']=='undefined':
    #  dict_aldi[int(row['snippet_id'])-1]=0
    #else:
    dict_aldi[int(row['snippet_id'])-1]=float(row['ALDi'])
  return dict_aldi

def get_samer_readability_dict(df):
  dict_samer={}
  samer_lemma_pos_verb_list=df['lemma#pos']
  samer_readability_list=df['readability (rounded average)']
  for samer_index in range(len(samer_lemma_pos_verb_list)):
    samer_lemma_pos_verb=samer_lemma_pos_verb_list[samer_index]
    samer_readability=samer_readability_list[samer_index]
    lemma=dediac_ar(samer_lemma_pos_verb.split('#')[0])
    pos=samer_lemma_pos_verb.split('#')[1]
    samer_lemma_pos_verb_id=lemma+"#"+pos
    dict_samer[(samer_lemma_pos_verb_id)]=samer_readability
  return dict_samer

def update_samer_readability(lemmas_all,pos_all,samer_word_level_scores,dict_samer):
  for lemma_index in range(len(lemmas_all)):
    lemma=lemmas_all[lemma_index]
    if stringContainsArabic(lemma):
      if lemma!='-no lex-':
        pos=pos_all[lemma_index]
        lemma_pos=dediac_ar(lemma)+"#"+pos
        if lemma_pos in dict_samer:
          word_readability_level=dict_samer[lemma_pos]
          samer_word_level_scores.append(word_readability_level)
          
def update_morph_dicts(disambig,snippet_morph_richness,morph_features_dict_mapping):
  for d in disambig:
    d_analysis= d.analyses[0].analysis
    word=d.word
    for morph_feature in morph_features_dict_mapping:
      morph_feature_dict=morph_features_dict_mapping[morph_feature]
      if (morph_feature in d_analysis) and (d_analysis[morph_feature] not in ['na','u','no']):
        morph_feature_value=d_analysis[morph_feature]
        if morph_feature_value in morph_feature_dict:
          morph_feature_dict[morph_feature_value]=morph_feature_dict[morph_feature_value]+1
    if 'bw' in d_analysis and stringContainsArabic(word):
      bw=d_analysis['bw']
      snippet_morph_richness.append(len(bw.split('+')))

def get_surface_features(pos_tags,sentence_tokenized):
  num_tokens=len(sentence_tokenized)
  punc_count=0
  word_count=0
  digit_count=0
  index=0
  for pos_tag in pos_tags:
    if pos_tag=='punc':
      punc_count+=1
    elif pos_tag=='digit':
      digit_count+=1
    else:
      word_count+=1
    index+=1
  return num_tokens,word_count,punc_count,digit_count

def update_token_type_ratio_dict(disambig,word_set,lemma_set):
  for d in disambig:
    d_analysis= d.analyses[0].analysis
    word=d.word
    pos=d_analysis.get('pos',"-no pos-")
    lemma=d_analysis.get('lex',"-no lex-")
    if pos not in ['punc','digit']:
      word_set.add(word)
      lemma_set.add(lemma)

def initialize_morph_dicts():
  pos_dict={"noun":0,"noun_prop":0,"noun_num":0,"noun_quant":0,"adj":0,"adj_comp":0,"adj_num":0,
    "adv":0,"adv_interrog":0,"adv_rel":0,"pron":0,"pron_dem":0,"pron_exclam":0,"pron_interrog":0,
    "pron_rel":0,"verb":0,"verb_pseudo":0,"verb_nom":0,"part":0,"part_dem":0,"part_det":0,"part_focus":0,
    "part_fut":0,"part_interrog":0,"part_neg":0,"part_restrict":0,"part_verb":0,"part_voc":0,"prep":0,
    "abbrev":0,"conj":0,"conj_sub":0,"interj":0,"punc":0,"digit":0,"latin":0,"foreign":0}
  prc0_dict={'0':0,'Aa_prondem':0,'Al_det':0,'AlmA_neg':0,'lA_neg':0,'mA_neg':0,'ma_neg':0,'mA_part':0,'mA_rel':0}
  prc1_dict={'0':0,'<i$_interrog':0,'bi_part':0,'bi_prep':0,'bi_prog':0,'Ea_prep':0,'EalaY_prep':0,'fiy_prep':0,'hA_dem':0,'Ha_fut':0,'ka_prep':0,
  'la_emph':0,'la_prep':0,'la_rc':0,'libi_prep':0,'laHa_emphfut':0,'laHa_rcfut':0,'li_jus':0,'li_sub':0,'li_prep':0,
  'min_prep':0,'sa_fut':0,'ta_prep':0,'wa_part':0,'wa_prep':0,'wA_voc':0,'yA_voc':0}
  prc2_dict={'0':0,'fa_conj':0,'fa_conn':0,'fa_rc':0,'fa_sub':0,'wa_conj':0,'wa_part':0,'wa_sub':0}
  prc3_dict={'0':0,'>a_ques':0}
  per_dict={'1':0,'2':0,'3':0}
  asp_dict={'c':0,'i':0,'p':0}
  vox_dict={'a':0,'p':0}
  mod_dict={'i':0,'j':0,'s':0}
  form_gen_dict={'f':0,'m':0}
  gen_dict={'f':0,'m':0}
  form_num_dict={'s':0,'d':0,'p':0}
  num_dict={'s':0,'d':0,'p':0}
  stt_dict={'c':0,'d':0,'i':0}
  cas_dict={'n':0,'a':0,'g':0}
  enc0_dict={'0':0,'1s_dobj':0,'1s_poss':0,'1s_pron':0,'1p_dobj':0,'1p_poss':0,'1p_pron':0,'2d_dobj':0,'2d_poss':0,'2d_pron':0,
  '2p_dobj':0,'2p_poss':0,'2p_pron':0,'2fs_dobj':0,'2fs_poss':0,'2fs_pron':0,'2fp_dobj':0,'2fp_poss':0,'2fp_pron':0,'2ms_dobj':0,
  '2ms_poss':0,'2ms_pron':0,'2mp_dobj':0,'2mp_poss':0,'2mp_pron':0,'3d_dobj':0,'3d_poss':0,'3d_pron':0,'3p_dobj':0,'3p_poss':0,
  '3p_pron':0,'3fs_dobj':0,'3fs_poss':0,'3fs_pron':0,'3fp_dobj':0,'3fp_poss':0,'3fp_pron':0,'3ms_dobj':0,'3ms_poss':0,'3ms_pron':0,
  '3mp_dobj':0,'3mp_poss':0,'3mp_pron':0,'Ah_voc':0,'lA_neg':0,'ma_interrog':0,'mA_interrog':0,'man_interrog':0,'ma_rel':0,'mA_rel':0,
  'man_rel':0,'ma_sub':0,'mA_sub':0}
  morph_features_dict_mapping={'asp':asp_dict,'cas':cas_dict,'form_gen':form_gen_dict,'form_num':form_num_dict,'gen':gen_dict,'mod':mod_dict,'num':num_dict,
    'per':per_dict,'stt':stt_dict,'vox':vox_dict,'pos':pos_dict,'prc0':prc0_dict,'prc1':prc1_dict,'prc2':prc2_dict,'prc3':prc3_dict,'enc0':enc0_dict}
  return pos_dict,morph_features_dict_mapping

def update_surface_features_dict(snippet_surface_features_dict,tokens_per_line,words_per_line,punc_per_line,digits_per_line):
  snippet_surface_features_dict['tokens_per_line'].append(tokens_per_line)
  if tokens_per_line==0:
    snippet_surface_features_dict['words_per_line'].append(0)
    snippet_surface_features_dict['punc_per_line'].append(0)
    snippet_surface_features_dict['digits_per_line'].append(0)
  else:
    snippet_surface_features_dict['words_per_line'].append(words_per_line/tokens_per_line)
    snippet_surface_features_dict['punc_per_line'].append(punc_per_line/tokens_per_line)
    snippet_surface_features_dict['digits_per_line'].append(digits_per_line/tokens_per_line)

def update_tensors(snippet_index,dict_features,all_snippets_tensor_lists,all_snippets_tensor_lists_langFeatures,
                   all_snippets_tensor_lists_lexicalFeatures,all_snippets_tensor_lists_syntacticFeatures,all_snippets_tensor_lists_surfaceFeatures):
  dict_features_sorted=dict(sorted(dict_features.items()))
  snippet_tensor_values=[]
  snippet_tensor_values_langFeatures=[]
  snippet_tensor_values_lexicalFeatures=[]
  snippet_tensor_values_syntacticFeatures=[]
  snippet_tensor_values_surfaceFeatures=[]
  for k,v in dict_features_sorted.items():
    snippet_tensor_values.append(v)
    if k.startswith("lang_"): snippet_tensor_values_langFeatures.append(v)
    if k.startswith("lexical_"): snippet_tensor_values_lexicalFeatures.append(v)
    if k.startswith("morph_"): snippet_tensor_values_syntacticFeatures.append(v)
    if k.startswith("surface_"): snippet_tensor_values_surfaceFeatures.append(v)
  snippet_tensor = torch.Tensor(snippet_tensor_values)
  snippet_tensor_lang = torch.Tensor(snippet_tensor_values_langFeatures)
  snippet_tensor_lexical = torch.Tensor(snippet_tensor_values_lexicalFeatures)
  snippet_tensor_syntactic = torch.Tensor(snippet_tensor_values_syntacticFeatures)
  snippet_tensor_surface = torch.Tensor(snippet_tensor_values_surfaceFeatures)
  all_snippets_tensor_lists.append(snippet_tensor)
  all_snippets_tensor_lists_langFeatures.append(snippet_tensor_lang)
  all_snippets_tensor_lists_lexicalFeatures.append(snippet_tensor_lexical)
  all_snippets_tensor_lists_syntacticFeatures.append(snippet_tensor_syntactic)
  all_snippets_tensor_lists_surfaceFeatures.append(snippet_tensor_surface)
  if snippet_index==0:
    print("langFeatures: ", snippet_tensor_lang.size())
    print("lexicalFeatures: ",snippet_tensor_lexical.size())
    print("syntacticFeatures: ",snippet_tensor_syntactic.size())
    print("surfaceFeatures: ",snippet_tensor_surface.size())

def get_diac_count(sentence_tokenized):
  diac_count_sentence=[]
  total_number_ar_words=0
  for token in sentence_tokenized:
    if stringContainsArabic(token):
      total_number_ar_words+=1
      count_diac=0
      for c in token:
        if c in AR_DIAC_CHARSET:
          count_diac+=1
      diac_count_sentence.append(count_diac)
  return diac_count_sentence,total_number_ar_words

def main():
  all_snippets_tensor_lists=[]
  all_snippets_tensor_lists_langFeatures=[]
  all_snippets_tensor_lists_lexicalFeatures=[]
  all_snippets_tensor_lists_syntacticFeatures=[]
  all_snippets_tensor_lists_surfaceFeatures=[]
  input_corpus_csv_file=open(sys.argv[1], newline='')
  csvreader = csv.DictReader(input_corpus_csv_file)
  content_case=sys.argv[5] #content_styled/content_example

  unfactored_glf = BERTUnfactoredDisambiguator.pretrained(model_name='glf')
  unfactored_msa = BERTUnfactoredDisambiguator.pretrained(model_name='msa')
  db_dir="DB_DIR"
  db = MorphologyDB(db_dir+"/MSA/calima-msa-s31_0.4.2.utf8.db", 'a')
  analyzer = Analyzer(db, 'ADD_PROP', cache_size=100000)
  unfactored_msa._analyzer = analyzer
  with open(db_dir+'/MSA/disambig_ranking_cache/calima-msa-s31/default_cache.pickle', 'rb') as f:
      unfactored_msa._ranking_cache = pickle.load(f)

  did = DialectIdentifier.pretrained()
  df = pd.read_csv(sys.argv[2], sep='\t')
  dict_samer=get_samer_readability_dict(df)
  snippet_index=0
  count=0
  for row in csvreader:
    if content_case!="content" or (content_case=="content" and 'test' in row['set']): #reading all entries from TST output or only test snippets from A3D dataset
      content=row[content_case]
      corpus=row['corpus']
      unfactored_msa_gulf=unfactored_msa
      if corpus=='Gumar':
          unfactored_msa_gulf=unfactored_glf

      snippet_samer_word_level_scores=[]
      snippet_morph_richness=[]
      pos_extra_info={"punc":0,"digit":0,"latin":0,"foreign":0}
      snippet_surface_features_dict={'tokens_per_line':[],'words_per_line':[],'punc_per_line':[],'digits_per_line':[]}

      pos_dict,morph_features_dict_mapping=initialize_morph_dicts()

      word_diac_count=[]
      word_set=set()
      lemma_set=set()
      total_number_words,total_number_ar_words=0,0
      for content_sentence in content.split("\n"):
        if len(content_sentence.strip())>0:
          
          #tokenize and disambiguate
          sentence_tokenized=simple_word_tokenize(content_sentence)
          disambig_GULF_content_example = unfactored_msa_gulf.disambiguate(sentence_tokenized)
          diac_count_sentence,total_number_ar_words_sentence=get_diac_count(sentence_tokenized)
          total_number_ar_words+=total_number_ar_words_sentence
          word_diac_count.extend(diac_count_sentence)

          #update morphological features for words in this line
          update_morph_dicts(disambig_GULF_content_example,snippet_morph_richness,morph_features_dict_mapping)

          #update samer readability level
          lemmas_all = [d.analyses[0].analysis.get('lex',"-no lex-") for d in disambig_GULF_content_example]
          pos_tags = [d.analyses[0].analysis.get('pos',"-no pos-") for d in disambig_GULF_content_example]
          update_samer_readability(lemmas_all,pos_tags,snippet_samer_word_level_scores,dict_samer)

          #supdate urface feaures
          tokens_per_line,words_per_line,punc_per_line,digits_per_line=get_surface_features(pos_tags,sentence_tokenized)
          update_surface_features_dict(snippet_surface_features_dict,tokens_per_line,words_per_line,punc_per_line,digits_per_line)
          total_number_words+=words_per_line

          #update token type ratio
          update_token_type_ratio_dict(disambig_GULF_content_example,word_set,lemma_set)

      #========> langugage choice:
      dict_features={}
      prediction_scores = did.predict([content])[0].scores
      for prediction_scores_dialect in prediction_scores:
        dict_features["lang_did_"+prediction_scores_dialect]=prediction_scores[prediction_scores_dialect]
      if total_number_words==0:
        dict_features["lang_foreign_dist"]=0
      else:
        dict_features["lang_foreign_dist"]=(pos_dict['latin']+pos_dict['foreign'])/total_number_words
      aldi_level_file=open(sys.argv[4], newline='')
      dict_aldi=get_dict_aldi(aldi_level_file)
      dict_features["lang_aldi"]=dict_aldi[snippet_index]
      
      #========> syntactic features
      for pos_extra_info_key in pos_extra_info:
        pos_extra_info[pos_extra_info_key]=pos_dict[pos_extra_info_key]
        pos_dict.pop(pos_extra_info_key)
      for morph_feature in morph_features_dict_mapping:
        morph_feature_dict=morph_features_dict_mapping[morph_feature]
        for sub_feature in morph_feature_dict:
          if total_number_ar_words==0:
            dict_features["morph"+"_"+morph_feature+"_"+sub_feature]=0
          else:
            dict_features["morph"+"_"+morph_feature+"_"+sub_feature]=morph_feature_dict[sub_feature]/total_number_ar_words
      if len(snippet_morph_richness)==0:
        dict_features["morph_morphRichness"]=0
      else:
        dict_features["morph_morphRichness"]=sum(snippet_morph_richness)/len(snippet_morph_richness)

      #========> surface features
      for surface_feature in snippet_surface_features_dict:
        dict_features["surface_"+surface_feature]=sum(snippet_surface_features_dict[surface_feature])/len(snippet_surface_features_dict[surface_feature])
      if len(word_diac_count)==0:
        dict_features["surface_avg_diac_count"]=0
      else:  
        dict_features["surface_avg_diac_count"]=sum(word_diac_count)/len(word_diac_count)

      #========> lexical features
      readability_level_file=open(sys.argv[3], newline='')
      dict_readability=get_readability_dict(readability_level_file)
      dict_features["lexical_readability_bareq"]=dict_readability[snippet_index]
      if len(snippet_samer_word_level_scores)==0:
        dict_features["lexical_readability_samer"]=0
      else:
        dict_features["lexical_readability_samer"]=sum(snippet_samer_word_level_scores)/len(snippet_samer_word_level_scores)
      if total_number_words==0:
        dict_features["lexical_token_type_ratio"]=0
      else:
        dict_features["lexical_token_type_ratio"]=len(word_set)/total_number_words

      #=========> create features tensor
      update_tensors(snippet_index,dict_features,all_snippets_tensor_lists,all_snippets_tensor_lists_langFeatures,
                     all_snippets_tensor_lists_lexicalFeatures, all_snippets_tensor_lists_syntacticFeatures,all_snippets_tensor_lists_surfaceFeatures)
      print("line index: ",snippet_index)
      snippet_index+=1
    count+=1
  print("count",count)

  #save all snippet-level pos distributions
  torch.save(all_snippets_tensor_lists, sys.argv[6])
  torch.save(all_snippets_tensor_lists_langFeatures, sys.argv[6]+"_langFeatures")
  torch.save(all_snippets_tensor_lists_lexicalFeatures, sys.argv[6]+"_lexicalFeatures")
  torch.save(all_snippets_tensor_lists_syntacticFeatures, sys.argv[6]+"_syntacticFeatures")
  torch.save(all_snippets_tensor_lists_surfaceFeatures, sys.argv[6]+"_surfaceFeatures")

if __name__=="__main__":
  main()