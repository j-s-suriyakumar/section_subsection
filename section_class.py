import json
import re
import pandas as pd
from urllib.parse import unquote_plus
from io import StringIO
from rapidfuzz import fuzz, process
from dateutil.parser import parse
import logging
from tqdm import tqdm
import numpy as np 
import time
from sklearn.cluster import KMeans
from date_time_extractor import DatetimeExtractor
import date_finder as dtf
import time_finder as tf
from constant import date_tags, suppress_date_tag_Section_end, suppress_date_tag_section
from constant import adm_dis_tag
from post_process import Postprocess
from IPython.display import HTML
import os
from pyhtml2pdf import converter
import warnings
warnings.filterwarnings("ignore")

class Section:
    def __init__(self, textract_file, constant_path, File_Name) -> None:
        self.textract_file = textract_file
        self.constant_path = constant_path
        self.File_Name = File_Name
        self.SECTION_END_PATTERNS = ['electronically witnessed', 'electronically signed', 'addendum by', 
                                    'addendum', 'signed by', 'added by', 'edited by','authenticated by',
                                    'consults by', 'reviewed by', 'transcribv by', 'transcribe by','transcribed by', 'review by', 
                                    'sign by', 'verify by', 'signed on','authorized by', 'testing performed by','performed by','entered by']
        self.SECTION_END_PATTERNS.sort(key=len, reverse=True)
        self.REPLACE_STRINGS = 'continued'

        # Load stop words
        stop_words = pd.read_csv(rf"{self.constant_path}\stop_words_v2.csv")
        self.stop_words_list = stop_words['Stop_words'].tolist()
        logging.info('Stop words loaded')

    def toc_checker(self, df):

        # load the sections
        section_data = pd.read_excel(rf"{self.constant_path}\old_section_subsection_list.xlsx",sheet_name='Section')
        logging.info('Pre defined Section Names loaded')

        # Load sub section data
        sub_section_data = pd.read_excel(rf"{self.constant_path}\old_section_subsection_list.xlsx",sheet_name='SubSection')
        logging.info('Pre defined Sub Section Names loaded')

        # Load Date formats
        date_variations = pd.read_csv(rf"{self.constant_path}\date_str_variations 1.csv")
        date_formats = date_variations['Date_string_types'].tolist()
        logging.info('Date formats loaded')

        corpus =' '.join(df['Text'].apply(lambda x : str(x)).to_list())
        table_of_content_match = re.findall(r"(?:Table of Contents?)", corpus)
        if len(table_of_content_match) > 0:
            print('is toc present')
            # read both df replace is_include col toc-->yes
            condition_idx_ls = section_data[ section_data['is_include'] =='toc'].index.to_list()
            for idx in condition_idx_ls:
                section_data.at[idx, 'is_include'] = 'Yes'

            condition_idx_ls_ = sub_section_data[ sub_section_data['is_include'] =='toc'].index.to_list()
            for idx in condition_idx_ls_:
                sub_section_data.at[idx, 'is_include'] = 'Yes'
            
            # And remove Labs: as a section
            mask_for_complete_df = section_data['section_name'].str.contains('Labs:', case=False)
            mask_for_Final_df = sub_section_data['section_name'].str.contains('Labs:', case=False)

            section_data = section_data[~mask_for_complete_df]
            sub_section_data = sub_section_data[~mask_for_Final_df]

        else:
            self.sub_section_data = sub_section_data[sub_section_data['is_include']=='Yes']
            self.section_data = section_data[section_data['is_include']=='Yes']

        self.rel_sec = self.section_data[self.section_data['is_relevant']== 'Yes']['section_name'].to_list()
        self.irrel_sec = self.section_data[self.section_data['is_relevant']== 'No']['section_name'].to_list()
        self.irrel_sec_lower = list(set([ele.lower() for ele in self.irrel_sec]))
        self.total_sections = list(set(self.rel_sec + self.irrel_sec))
        self.sec_list = sorted(self.total_sections, key=len, reverse=True)

    def assign_index(df):
        """ 
        args:
            DataFrame: df
        return:
            if Count word Ids matched with the sum(count of Ids in Line)
                DataFrame
            else 
                None
        """
        df_word = len(df[df['BlockType']=='WORD'])
        
        df = df[df['BlockType']=='LINE']
        df = df.reset_index()
        df.drop(columns={'index'},inplace=True)
        idx_len = 0
        for i,j in df.iterrows():
            for k in eval(j['Relationships']):
                idx_len += len(k['Ids'])
                df.at[i,'word_index'] = str([idx_len-len(k['Ids']),idx_len])
        
        if df_word == idx_len:
            logging.info('Index count match')
            return df
        else:
            logging.error('Index count did not match')
            return None

    def text_cleanup(self, text):
        '''
            # t1 - Remove non-alphanumeric characters (except whitespace)
            # t2 - Convert text to lowercase, strip whitespace, and split into words
            # t3 - Remove empty strings from the list of words
            # stopword_count -  Count the number of stopwords in the text
            # non_stopword_count -  Remove stopwords from the list of words
            # non_stopword_text - Join the list of words back into a string with single spaces
            # extra_whitespaces_removed - Replace multiple consecutive spaces with a single space
        args:
            text: string (raw text)
            stop: list of stopwords (from stop_words.csv)
        return:
            raw_str_match: string (preprocessed text)
            len_words_raw_text: int (length of the preprocessed text)
            stopword_count: int (number of stopwords in the preprocessed text)
        '''
        t1 = re.sub(r'([^A-Za-z0-9\s]+?)', '', str(text))
        t2 = t1.lower().strip().split()
        t3 = list(filter(None, t2))
        
        stopword_count = len([i for i in t3 if i in self.stop_words_list])
        non_stopword_count = [i for i in t3 if i not in self.stop_words_list]
        non_stopword_count_length = len([i for i in t3 if i not in self.stop_words_list])
        non_stopword_text = ' '.join(non_stopword_count)
        non_extra_whitespaces_text = non_stopword_text.replace(' +', ' ')
        
        return non_extra_whitespaces_text, non_stopword_count_length, stopword_count
    
    def update_cleaned_dataframe(self, i,txt, sec_subsec_list, sec_name, average_score,section_entity, df):
            
        '''
        # Updates the dataframe with the section name, average score, section entity and relevant section
        
        args:
            i: int (index)
            sec_name: string (section name)
            average_score: int (average score)
            section_entity: string (section entity)
        return:
            None
        '''
        if section_entity in ('SECTION','SECTION END'):
            section_name = sec_name
            df.loc[i, 'entity'] = section_name
            df.loc[i, 'score'] = average_score
            relevancy = 'No' if sec_name.lower() in self.irrel_sec_lower else 'Yes'
            df.loc[i, 'section_entity'] = section_entity
            df.loc[i, 'is_relevant'] = relevancy

        elif section_entity in ('SUB SECTION'):
            section_name = sec_name
            df.loc[i, 'sub_section_entity'] = section_name
            df.loc[i, 'sub_section_entity_score'] = average_score
            relevancy = 'No' if sec_name.lower() in self.irrel_sub_sec else 'Yes'
            df.loc[i, 'section_entity'] = section_entity
            df.loc[i, 'is_sub_section_relevant'] = relevancy
    
    def get_lookup_table_df(self, df):
        df['text_len'] = df['Text'].str.len()
        df['Start'] = df['text_len'].shift(fill_value=0).cumsum() + df.apply(lambda row: row.name, axis=1)
        df['End'] = df['Start']+df['text_len']
        df.drop('text_len',  axis=1, inplace=True)
        return df
    
    def get_corpus(self, d1):
        corpus = ' '.join(d1['Text'].apply(lambda x: str(x)))
        return corpus
    
    def supress_date(self, date_tags_list,suppress_date_tag):
        date_tags_true = []
        suppress_date_ls = list(map(lambda x: x.lower(),suppress_date_tag))
        for i in date_tags_list:
            if str(i[2]).lower() not in suppress_date_ls:
                date_tags_true.append(i)
        if len(date_tags_true)==0:
            return date_tags_true
        else:
            return date_tags_true
        
    def sub_sec_detection(self,i, non_extra_whitespaces_text, all_sub_sections_list_sorted, cleaned_subsec, irrel_sub_sec, non_stopword_count_length,df): 
        
        matcher = process.extractOne(non_extra_whitespaces_text, cleaned_subsec, scorer=fuzz.token_sort_ratio)
        
        if matcher[1] >= 75:
            ratio_partial = round(fuzz.partial_ratio(str(non_extra_whitespaces_text),matcher[0]), 2)
            average_score = round((matcher[1] + ratio_partial)/2)
            original_sub_sec = all_sub_sections_list_sorted[matcher[2]]
            if len(original_sub_sec.split()) ==1 and average_score > 95 and (non_stopword_count_length >= len(original_sub_sec.split()) or average_score >= 95):
                self.update_cleaned_dataframe(i, non_extra_whitespaces_text, self.sec_list, cleaned_subsec[matcher[2]], average_score,'SUB SECTION',df)

            elif len(original_sub_sec.split()) > 1 and average_score > 87.5 and (non_stopword_count_length >= len(original_sub_sec.split()) or average_score >= 87.5):
                self.update_cleaned_dataframe(i, non_extra_whitespaces_text, self.sec_list, cleaned_subsec[matcher[2]], average_score,'SUB SECTION',df)

    def get_result(self, df,index,range1,range2,suppress_date_tags):
        corpus = self.get_corpus(df[index+range1:index+range2])
        Section_dates = self.supress_date(DatetimeExtractor.get_date_time_from_corpus_v2(corpus.lower(),date_tags),suppress_date_tags)
        return Section_dates
        
    def update_cleaned_dataframe_v2(self, i,txt, sec_subsec_list, sec_name, average_score,section_entity,irr_i, df):
        
        '''
        # Updates the dataframe with the section name, average score, section entity and relevant section
        
        args:
            i: int (index)
            sec_name: string (section name)
            average_score: int (average score)
            section_entity: string (section entity)
        return:
            None
        '''
        if section_entity in ('SUB SECTION'):
            df.loc[irr_i, 'entity'] = ''
            df.loc[i, 'is_relevant'] = str('nan')
            df.loc[i, 'score'] = np.NaN

            section_name = sec_name
            df.loc[i, 'sub_section_entity'] = section_name
            df.loc[i, 'sub_section_entity_score'] = average_score
            relevancy = 'No' if sec_name.lower() in self.irrel_sub_sec else 'Yes'
            df.loc[i, 'section_entity'] = section_entity
            df.loc[i, 'is_sub_section_relevant'] = relevancy

    def sub_sec_detection_v2(self, i, non_extra_whitespaces_text, all_sub_sections_list_sorted, cleaned_subsec, irrel_sub_sec, non_stopword_count_length,irr_indexes, df): 
        
        matcher = process.extractOne(non_extra_whitespaces_text, cleaned_subsec, scorer=fuzz.token_sort_ratio)
        
        if matcher[1] >= 75:
            ratio_partial = round(fuzz.partial_ratio(str(non_extra_whitespaces_text),matcher[0]), 2)
            average_score = round((matcher[1] + ratio_partial)/2)
            original_sub_sec = all_sub_sections_list_sorted[matcher[2]]
            if len(original_sub_sec.split()) ==1 and average_score > 95 and (non_stopword_count_length >= len(original_sub_sec.split()) or average_score >= 95):
                self.update_cleaned_dataframe_v2(i, non_extra_whitespaces_text, self.sec_list, cleaned_subsec[matcher[2]], average_score,'SUB SECTION',irr_indexes,df)
            
            elif len(original_sub_sec.split()) > 1 and average_score > 87.5 and (non_stopword_count_length >= len(original_sub_sec.split()) or average_score >= 87.5):
                self.update_cleaned_dataframe_v2(i, non_extra_whitespaces_text, self.sec_list, cleaned_subsec[matcher[2]], average_score,'SUB SECTION',irr_indexes,df)
            
            else:
                df.loc[irr_indexes, 'entity'] = ''
                df.loc[i, 'is_relevant'] =str('nan')
                df.loc[i, 'score'] = np.NaN
                df.loc[i,'section_entity'] = str('nan')
        else:
            df.loc[irr_indexes, 'entity'] = ''
            df.loc[i, 'is_relevant'] = str('nan')
            df.loc[i, 'score'] = np.NaN
            df.loc[i,'section_entity'] = str('nan')

    def section_logic(self, df):
        self.section_start_time = time.time()
        logging.info('Detecting SECTIONs and SECTION END')
        b = []
        for i, txt in tqdm(df.iterrows()):
            if len(str(txt['Text'])) == 0:
                continue
            non_extra_whitespaces_text,non_stopword_count_length,stopword_count = self.text_cleanup(str(txt['Text'])) 
            non_extra_whitespaces_text = str(non_extra_whitespaces_text)

            clean_txt_length = len(str(non_extra_whitespaces_text))
            matcher = process.extractOne(non_extra_whitespaces_text.replace("continued","").strip(), self.cleaned_sections, scorer=fuzz.token_sort_ratio)
            
            if matcher[1] >= 75:
                partial_ratio_score = fuzz.partial_ratio(non_extra_whitespaces_text.replace("continued","").strip(), matcher[0], processor=lambda x: x.lower().strip())
                average_score = (matcher[1] + partial_ratio_score)/2
                split_master_sec_by_words = matcher[0].split(' ')
                is_num = re.sub(r'([^0-9]+?)', '', str(matcher[0]))
                is_raw_text_greater = len(non_extra_whitespaces_text.split(' ')) >= len(
                    split_master_sec_by_words)
                
                if len(split_master_sec_by_words) == 1 and average_score > 95 and clean_txt_length >= 3 and is_raw_text_greater and stopword_count < 3:
                    b.append((i,non_extra_whitespaces_text, self.sec_list[matcher[2]], average_score,'SECTION',is_raw_text_greater,split_master_sec_by_words,non_extra_whitespaces_text.split(' ')))
                    self.update_cleaned_dataframe(i, non_extra_whitespaces_text, self.sec_list, self.sec_list[matcher[2]], average_score,'SECTION',df)
                
                elif len(split_master_sec_by_words) == 1 and average_score > 87.5 and is_num == '' and clean_txt_length >= 3 and is_raw_text_greater and self.REPLACE_STRINGS in non_extra_whitespaces_text and stopword_count < 3:
                    self.update_cleaned_dataframe(i,non_extra_whitespaces_text, self.sec_list, self.sec_list[matcher[2]], average_score,'SECTION',df)
                    b.append((i, non_extra_whitespaces_text, self.sec_list[matcher[2]], average_score,'SECTION',is_raw_text_greater,split_master_sec_by_words,non_extra_whitespaces_text.split(' ')))
                
                elif len(split_master_sec_by_words) > 1 and average_score > 87.5 and is_num == '' and clean_txt_length >= 3 and is_raw_text_greater and stopword_count < 3:
                    self.update_cleaned_dataframe(i, non_extra_whitespaces_text, self.sec_list, self.sec_list[matcher[2]], average_score,'SECTION',df)
                    b.append((i, non_extra_whitespaces_text, self.sec_list[matcher[2]], average_score,'SECTION',is_raw_text_greater,split_master_sec_by_words,non_extra_whitespaces_text.split(' ')))
            
            elif len(non_extra_whitespaces_text.strip().lower()) >=7:
                for sec_end_pattern in self.SECTION_END_PATTERNS:
                    if sec_end_pattern in txt['Text'].lower():
                        b.append((i, non_extra_whitespaces_text, sec_end_pattern, 100,'SECTION END','','',''))
                        self.update_cleaned_dataframe(i, non_extra_whitespaces_text, self.sec_list, '', 100,'SECTION END',df)
                        break
        print("The length of b is : ", len(b))
        logging.info('Detected SECTIONs and SECTION END')
        self.section_end_time = time.time()

    def suppress_section(self,df):
        page_ls = set(df[df['section_entity'] == "SECTION"]['Page'].values)

        suppress_section_idx = []
        for page in tqdm(page_ls):
            idx_ls = df[(df['Page'] == page) & (df['section_entity'] == "SECTION")].index
            th_top = np.average(df[df['Page'] == page]['Geometry.BoundingBox.Height'])
            if len(idx_ls)>0:
                for idx in idx_ls:
                    # Check for the height of the current txt, w.r.t previous and next text, if the height is less than average consider it as in one line 
                    if (abs(df.iloc[idx]['Geometry.BoundingBox.Top'] - df.iloc[idx-1]['Geometry.BoundingBox.Top']) < th_top or \
                        abs(df.iloc[idx]['Geometry.BoundingBox.Top'] - df.iloc[idx+1]['Geometry.BoundingBox.Top']) < th_top) and \
                        not(df.iloc[idx]['Text'].isupper()):
                        suppress_section_idx.append(idx)

        updated_ss_idx = []
        for i in suppress_section_idx:
            temp = df[df['Page'] == df.iloc[i]['Page']].copy()
            temp['Y'] = 0.0
            k_range = range(1,3)
            k_means_var = [KMeans(n_clusters = k).fit(temp[['Geometry.BoundingBox.Left','Y']]) for k in k_range]
            centroids = [X.cluster_centers_ for X in k_means_var]
            if df.iloc[i]['Geometry.BoundingBox.Left'] >= min(0.48,max(centroids[1][0][0],centroids[1][1][0])):
                updated_ss_idx.append(i)

        df.loc[updated_ss_idx,'section_entity'] = str('nan')
        df.loc[updated_ss_idx,'entity'] = str('nan')
        df.loc[updated_ss_idx,'score'] = np.nan

        df['entity'] = df['entity'].mask(df['entity']=='nan', None).ffill()
        df['entity'] = df['entity'].mask(df['entity']=='n', None).ffill()
        df['entity'] = df['entity'].mask(df['entity']=='', None).ffill() 

    def get_main_section_dict(self, df):
        df1 = df[['Page','Text','section_entity','word_index','entity','is_relevant']].drop_duplicates() 
        df1 = df1[df1['is_relevant']=='Yes']
        detected_sections = list(set(df1['entity'].dropna()))

        self.main_sec_dict = {}
        for i,j in self.sub_section_data.iterrows():
            for k in detected_sections:
                if j['section_name'] == k:
                    self.main_sec_dict[j['section_name']] = j['main_section']

        df['main_section'] = df['entity'].replace(self.main_sec_dict)

    def sub_section_logic(self,df):
        self.sub_section_start_time = time.time()
        logging.info('Detecting SUB SECTIONs')
        for k, v in list(set(self.main_sec_dict.items())):
            i = self.main_sec_dict.get(k)
            # Step 1: Find the index of the mapped main section
            # Step 2: Get subsections related to the mapped main section
            # Step 3: Get indexes of rows in relevant_sections_df where 'section' matches 'i'
            # Step 4: Extract irrelevant subsections
            # Step 5: Extract all subsections and sort them by length in reverse order
            # Step 6: Return the extracted information
            mapped_main_section = list(set(self.sub_section_data[self.sub_section_data['section_name'] == k]['main_section']))
            logging.info(f'Detecting SUB SECTIONs for {k} --> {mapped_main_section}')
            if len(mapped_main_section) > 0:
                mapped_main_section = mapped_main_section[0]

                sec_idxs = df[df['entity'] == k].index.tolist() 
                
                all_subsec_df = self.sub_section_data[self.sub_section_data['main_section'] == mapped_main_section] 
                        
                rel_sub_sec_list = all_subsec_df[all_subsec_df['is_relevant'] == 'Yes']['sub_section_name'].to_list() 
                rel_sub_sec_list = [ele.lower().strip() for ele in rel_sub_sec_list] 
                
                self.irrel_sub_sec = all_subsec_df[all_subsec_df['is_relevant'] == 'No']['sub_section_name'].to_list() 
                self.irrel_sub_sec = [ele.lower() for ele in self.irrel_sub_sec]
            
                all_sub_sections = all_subsec_df['sub_section_name'].to_list()
                all_sub_sections_list_sorted = sorted(all_sub_sections, key = len, reverse = True)
            
                if len(mapped_main_section) == 0:
                    logging.error('No mapped main section found for the section {}'.format(i))
            
                cleaned_subsec = [self.text_cleanup(ele)[0] for ele in all_sub_sections_list_sorted] 
                for idx in sec_idxs:
                    raw_cleaned_txt = str(df['Text'][idx])
                    
                    if ':' in raw_cleaned_txt and df['section_entity'][idx] == 'nan':
                        for txt_split in raw_cleaned_txt.split(":"):
                        # txt_split = raw_cleaned_txt.split(":")[0]
                            txt_colon_split = str(txt_split).strip()
                            non_extra_whitespaces_text, non_stopword_count_length, stopword_count = self.text_cleanup(txt_colon_split)
                            self.sub_sec_detection(idx, non_extra_whitespaces_text, all_sub_sections_list_sorted, cleaned_subsec, self.irrel_sub_sec, non_stopword_count_length,df)
                    
                    elif df['section_entity'][idx] == 'nan':
                        non_extra_whitespaces_text, non_stopword_count_length, stopword_count = self.text_cleanup(raw_cleaned_txt)
                        matcher = process.extractOne(non_extra_whitespaces_text, cleaned_subsec, scorer=fuzz.token_sort_ratio)
                        ratio_partial = round(fuzz.partial_ratio(str(non_extra_whitespaces_text),matcher[0]), 2)
                        average_score = round((matcher[1] + ratio_partial)/2)

                        if stopword_count <= 2 or average_score>98:
                            self.sub_sec_detection(idx, non_extra_whitespaces_text, all_sub_sections_list_sorted, cleaned_subsec, self.irrel_sub_sec, non_stopword_count_length,df)
                time.sleep(5)

        logging.info('Detected SUB SECTIONs')
        self.sub_section_end_time = time.time()

    def update_Main_Section(self,df):
        #------------------------------- Section -> Main Section Update-------------------------------
        logging.info('Updating the Main Section')
        df_updated = df[(df['section_entity'] == "SECTION") | (df['section_entity'] == "SECTION END") | (df['section_entity'] == "SUB SECTION")].copy()
        df_updated_idxs = df_updated.index.to_list()

        prev_main_section = None
        need_to_change_df = []
        for idx in range(len(df_updated)):
            if df_updated.iloc[idx]['main_section'] != prev_main_section and prev_main_section is not None:
                if df_updated.iloc[idx-1]['section_entity'] != "SECTION END":
                    need_to_change_df.append(df_updated.iloc[idx])
                    continue
                if df_updated.iloc[idx]['section_entity'] == "SECTION END":
                    need_to_change_df.append(df_updated.iloc[idx])
                    continue
                prev_main_section = df_updated.iloc[idx]['main_section']
            else:
                prev_main_section = df_updated.iloc[idx]['main_section']

        need_to_change_df_idx = pd.DataFrame(need_to_change_df).index.to_list()

        idx_ss_ls = list(set(need_to_change_df_idx).symmetric_difference(set(df_updated.index.to_list())))
        for idx in idx_ss_ls:
            df.at[idx,'Updated_main_section'] = df.iloc[idx]['main_section']
        
        df['Updated_main_section'] = df['Updated_main_section'].mask(df['Updated_main_section']=='nan', None).ffill()
        # --------------------------------------------------------------------------------------------
        # ---------------Update the Sections to Sub Sections under each main section.-----------------
        main_sec_ls = df['Updated_main_section'].value_counts().index
        for i in main_sec_ls:
            sec_ls = self.sub_section_data[self.sub_section_data['main_section'] == i]['section_name'].to_list()
            for j in list(df[df['Updated_main_section'] == i]['entity'].value_counts().index):
                indexes = df[(df['Updated_main_section'] == i) & (df['entity'] == j) & (df['section_entity'] == 'SECTION')].index
                irr_indexes = df[(df['Updated_main_section'] == i) & (df['entity'] == j)].index
                if j not in sec_ls:
                    raw_cleaned_txt = str(j)

                    all_subsec_df = self.sub_section_data[self.sub_section_data['main_section'] == i]

                    rel_sub_sec_list = all_subsec_df[all_subsec_df['is_relevant'] == 'Yes']['sub_section_name'].to_list() 
                    rel_sub_sec_list = [ele.lower().strip() for ele in rel_sub_sec_list]

                    self.irrel_sub_sec = all_subsec_df[all_subsec_df['is_relevant'] == 'No']['sub_section_name'].to_list() 
                    self.irrel_sub_sec = [ele.lower() for ele in self.irrel_sub_sec]

                    all_sub_sections = all_subsec_df['sub_section_name'].to_list()
                    all_sub_sections_list_sorted = sorted(all_sub_sections, key = len, reverse = True)

                    cleaned_subsec = [self.text_cleanup(ele)[0] for ele in all_sub_sections_list_sorted] 
                    if ':' in raw_cleaned_txt:
                        txt_colon_split = str(raw_cleaned_txt).strip()
                        non_extra_whitespaces_text, non_stopword_count_length, stopword_count = self.text_cleanup(txt_colon_split)
                        self.sub_sec_detection_v2(indexes, non_extra_whitespaces_text, all_sub_sections_list_sorted, cleaned_subsec, self.irrel_sub_sec, non_stopword_count_length,irr_indexes,df)

                    else:
                        non_extra_whitespaces_text, non_stopword_count_length, stopword_count = self.text_cleanup(raw_cleaned_txt)
                        matcher = process.extractOne(non_extra_whitespaces_text, cleaned_subsec, scorer=fuzz.token_sort_ratio)
                        ratio_partial = round(fuzz.partial_ratio(str(non_extra_whitespaces_text),matcher[0]), 2)
                        average_score = round((matcher[1] + ratio_partial)/2)

                        if stopword_count <= 2 or average_score>98:
                            self.sub_sec_detection_v2(indexes, non_extra_whitespaces_text, all_sub_sections_list_sorted, cleaned_subsec, self.irrel_sub_sec, non_stopword_count_length,irr_indexes,df)

        df['entity'] = df['entity'].mask(df['entity']=='', None).ffill()
        # --------------------------------------------------------------------------------------------

    def sectionDates(self, df):
        lookup_section_start = df[df['section_entity'] == "SECTION"].index
        lookup_section_end = df[df['section_entity'] == "SECTION END"].index

        crp = self.get_corpus(df)
        p_obj = Postprocess(adm_dis_tag)
        self.yr_ = p_obj.get_adm_discharge_date(crp)

        Section_dates_ls = []
        result = []

        for idx in lookup_section_start:
            Section_date_ = self.get_result(df,idx,0,10,suppress_date_tag_section)
            if len(Section_date_) > 0:
                if eval(str(Section_date_))[0][1]:
                    Section_date = p_obj.date_parser(eval(str(Section_date_))[0][0] + " " + eval(str(Section_date_))[0][1], (self.yr_, 1, 1))
                    Section_dates_ls.append([df.iloc[idx]['Page'], df.iloc[idx]['score'],df.iloc[idx]['entity'],df.iloc[idx]['section_entity'],df.iloc[idx]['main_section'],df.iloc[idx]['Updated_main_section'],Section_date]) 
                    df.at[idx,'Date/Time'] = str(Section_date_)
                    df.at[idx,'Post Process Date/Time'] = str(Section_date)
                else:
                    Section_date = p_obj.date_parser(eval(str(Section_date_))[0][0], (self.yr_, 1, 1))
                    Section_dates_ls.append([df.iloc[idx]['Page'], df.iloc[idx]['score'],df.iloc[idx]['entity'],df.iloc[idx]['section_entity'],df.iloc[idx]['main_section'],df.iloc[idx]['Updated_main_section'],Section_date]) 
                    df.at[idx,'Date/Time'] = str(Section_date_)
                    df.at[idx,'Post Process Date/Time'] = str(Section_date)
            else:
                Section_dates_ls.append([df.iloc[idx]['Page'], df.iloc[idx]['score'],df.iloc[idx]['entity'],df.iloc[idx]['section_entity'],df.iloc[idx]['main_section'],df.iloc[idx]['Updated_main_section']])

        try:
            self.Sections_DF = pd.DataFrame(Section_dates_ls)
            self.Sections_DF.rename(columns={0:'Page Number',1:'score',2:'entity',3:'section_entity',4:'main_section',5:'Updated_main_section',6:'Dates'},inplace=True)
        except:
            logging.error('Sections not detected')
            pass

    def update_sectionEnd_date(self, Section_end_date_,Section_end_date, df, df_end, i, j):
        df_end.at[j,'Date/Time'] = str(Section_end_date_)
        df.at[i,'Date/Time'] = str(Section_end_date_)
        df_end.at[j,'Post Process Date/Time'] = str(Section_end_date)
        df.at[i,'Post Process Date/Time'] = str(Section_end_date)

    def sectionEndinfo(self,df):
        #----------------------------- Section End -> Physician Name, Date ---------------------------
        logging.info('Extracting Physician Name, Date')
        START_PATTERN = [' [[Bb]y]?\s*:?\s*', ' EDT', ' EST', ' [[Ss]igned]?\s*:\s*',' [[Ss]igned]?\s*,\s*','SIGNED',' [[Ee]ditor]?\s*:\s*',
                        '[[Aa]ddendum]?\s*:\s*', 'AM', 'PM', 'PST','PDT']

        CREDS = ['M\s*?\.?\s*?D', 'DO', 'DPM', 'PA', 'PA-C', 'CRNA', 'RN',
                'D\s*?\.\s*?O\.', 'PHD', 'MBBS', 'NP', 'APRN', 'DNP-APRN', 
                ]

        END_PATTERN = ['\bPT\b','\(','\d', ' on', ' at','\[','\*','DATE:']

        SP = "|".join(START_PATTERN)
        CR = "|".join(CREDS)
        EP = "|".join(END_PATTERN)
        p_obj = Postprocess(adm_dis_tag)
        j=0
        df_end = pd.DataFrame()
        pattern1 = rf"(?:{SP})[^A-Za-z0-9]*(.*?)(?:{EP})|(?:{SP})[^A-Za-z0-9]*(.*)(?:{EP})?"
        for i in df[df['section_entity'] == "SECTION END"].index.to_list():
            text = " ".join(df.iloc[i:i+1]['Text'])
            df_end.at[j,'Page'] = int(df.iloc[i]['Page'])
            df_end.at[j,'Index'] = int(i)
            res = ""
            try:
                non_extra_whitespaces_text,_, _ = self.text_cleanup("".join(re.findall(pattern1,text)[0]).strip())
                if len(non_extra_whitespaces_text) > 0:
                    df_end.at[j,'Text'] = text
                    df_end.at[j,'Physician Name'] = "".join(re.findall(pattern1,text)[0]).strip()
                else:
                    text = " ".join(df.iloc[i:i+2]['Text'])
                    if len("".join(re.findall(pattern1,text)[0]).strip()) > 0:
                        df_end.at[j,'Text'] = text
                        df_end.at[j,'Physician Name'] = "".join(re.findall(pattern1,text)[0]).strip()
            except:
                text = " ".join(df.iloc[i:i+2]['Text'])
                try:
                    if len("".join(re.findall(pattern1,text)[0]).strip()) > 0:
                        df_end.at[j,'Text'] = text
                        df_end.at[j,'Physician Name'] = "".join(re.findall(pattern1,text)[0]).strip()
                except:
                    print(i)
            
            # Date Logic
            Section_end_date_ = self.get_result(df,i,0,4,suppress_date_tag_Section_end)
            if len(Section_end_date_) != 0:
                if eval(str(Section_end_date_))[0][1]:
                    Section_end_date = p_obj.date_parser(eval(str(Section_end_date_))[0][0] + " " + eval(str(Section_end_date_))[0][1], (self.yr_, 1, 1))
                    self.update_sectionEnd_date(Section_end_date_,Section_end_date, df, df_end, i, j)
                else:
                    Section_end_date = p_obj.date_parser(eval(str(Section_end_date_))[0][0], (self.yr_, 1, 1))
                    self.update_sectionEnd_date(Section_end_date_,Section_end_date, df, df_end, i, j)
            else:
                Section_end_date_ = self.get_result(df,i,-2,0,suppress_date_tag_Section_end)
                if len(Section_end_date_) != 0:
                    if eval(str(Section_end_date_))[0][1]:
                        Section_end_date = p_obj.date_parser(eval(str(Section_end_date_))[0][0] + " " + eval(str(Section_end_date_))[0][1], (self.yr_, 1, 1))
                        self.update_sectionEnd_date(Section_end_date_,Section_end_date, df, df_end, i, j)
                        
                    else:
                        Section_end_date = p_obj.date_parser(eval(str(Section_end_date_))[0][0], (self.yr_, 1, 1))
                        self.update_sectionEnd_date(Section_end_date_,Section_end_date, df, df_end, i, j)
            j+=1

        pattern_CREDS = rf"\b({CR})\b"
        df_end = df_end.reset_index()
        df_end.drop(columns={'index'},inplace=True)
        for i in tqdm(range(len(df_end))):
            try:
                if ",".join(re.findall(pattern_CREDS,df_end.iloc[i]['Physician Name'])).strip():
                    df_end.at[i,'CREDS_phy'] = ",".join(re.findall(pattern_CREDS,df_end.iloc[i]['Physician Name'])).strip()
                else:
                    df_end.at[i,'CREDS_phy'] = ",".join(re.findall(pattern_CREDS,df_end.iloc[i]['Text'])).strip()
            except:
                pass
        
        self.df_end = df_end

    def section_subsection_algorithm(self):
        # Configure the logging system
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s')
        logging.info('FUNCTION START: rev_section_subsection.py')
        df = pd.read_csv(rf"{self.textract_file}", na_filter=False)
        df = Section.assign_index(df)
        self.toc_checker(df)

        logging.info('FUNCTION START: str_preprocessing')
        self.cleaned_sections = [self.text_cleanup(sec)[0] for sec in self.sec_list]
        logging.info('FUNCTION END: str_preprocessing')

        self.section_logic(df)
        self.suppress_section(df)
        self.get_main_section_dict(df)
        self.sub_section_logic(df)
        self.update_Main_Section(df)
        self.get_lookup_table_df(df)
        self.sectionDates(df)
        self.sectionEndinfo(df)
        return df

    def ChartOrderLogic(self, df):
        #----------------------------------------- Chart Order----------------------------------------
        logging.info('Chart Order')
        df_CO = df[(df['section_entity'] == "SECTION") | (df['section_entity'] == "SUB SECTION") | (df['section_entity'] == "SECTION END")]
        req_ls = ['Page','entity','score','section_entity','sub_section_entity','Updated_main_section','Post Process Date/Time']
        df_CO = df_CO[[i for i in req_ls if i in df.columns]]
        df_CO.rename(columns={"Updated_main_section":"Main_Section"},inplace=True)
        df_CO = df_CO.reset_index()

        for i in range(len(df_CO)):
            try:
                df_CO.at[i,'Date'] = df_CO.iloc[i]['Post Process Date/Time'].split(" ")[0]
            except:
                df_CO.at[i,'Date'] = df_CO.iloc[i]['Post Process Date/Time']

        count = 1
        flag = 0
        for i in range(len(df_CO)):
            if (df_CO.iloc[i]['section_entity'] == "SECTION" or  df_CO.iloc[i]['section_entity'] == "SUB SECTION") and flag == 0:
                flag = 0
                continue
            if df_CO.iloc[i]['section_entity'] == "SECTION END":
                flag = 1
                df_CO.at[i,'Group'] = f'Group{count}'
            if df_CO.iloc[i]['section_entity'] == "SECTION" or df_CO.iloc[i]['section_entity'] == "SUB SECTION" and flag == 1:
                flag = 0
                count += 1
        # If there is no Section End present at last, we need to consider it as one group
        if df_CO.iloc[len(df_CO)-1]['Group'] == None or df_CO.iloc[len(df_CO)-1]['Group'] == 'nan':
            df_CO.at[len(df_CO)-1,'Group'] = f'Group{count}'

        df_CO['Group'] = df_CO['Group'].mask(df_CO['Group']=='nan', None).bfill()

        for i in list(set(df_CO['Group'].to_list())):
            index = df_CO[df_CO['Group'] == i].index[-1]
            df_CO.at[index,'Range'] = str((df_CO[df_CO['Group'] == i].iloc[0]['Page'],df_CO[df_CO['Group'] == i].iloc[-1]['Page']))
            try:
                df_CO.at[index,'Group Date'] = max([i for i in df_CO[(df_CO['Group'] == i) & (df_CO['section_entity'] == "SECTION END")]['Post Process Date/Time'].to_list() if i != 'nan'])
            except:
                pass

        # If we're fiding a SUB SECTION after SECTION END, we're suppressing them !
        keys = df_CO[df_CO['Range'] != 'nan'].index.to_list()
        values = df_CO[df_CO['Range'] != 'nan']['Range'].to_list()
        pairs = zip(keys, values)
        d = dict(pairs)

        for k,v in d.items():
            try:
                if df_CO.iloc[k+1]['section_entity'] == "SUB SECTION":
                    grp_value = df_CO.iloc[k+1]['Group']
                    main_section_name_index = df_CO[df_CO['Group'] == grp_value].index
                    df_CO.loc[main_section_name_index, 'Main_Section'] = df_CO.iloc[k]['Main_Section']
                    df_CO['Group'] = df_CO['Group'].replace(grp_value,'nan')
                    df_CO.at[k,'Range'] = 'nan'
            except:
                pass

        df_CO['Group'] = df_CO['Group'].mask(df_CO['Group']=='nan', None).ffill()
        df_CO['Range'] = 'nan'
        grp_set = set(df_CO['Group'].to_list())
        for i in grp_set:
            index = df_CO[df_CO['Group'] == i].index[-1]
            df_CO.at[index,'Range'] = str((df_CO[df_CO['Group'] == i].iloc[0]['Page'],df_CO[df_CO['Group'] == i].iloc[-1]['Page']))
            try:
                df_CO.at[index,'Group Date'] = max([i for i in df_CO[(df_CO['Group'] == i) & (df_CO['section_entity'] == "SECTION END")]['Post Process Date/Time'].to_list() if i != 'nan'])
            except:
                pass

        # If we're fiding a SECTION and SECTION END in a same page, we're suppressing them !
        keys = df_CO[df_CO['Range'] != 'nan'].index.to_list()
        values = df_CO[df_CO['Range'] != 'nan']['Range'].to_list()
        pairs = zip(keys, values)
        d = dict(pairs) 
        prev_key = None
        for k,v in d.items():
            try:
                if abs(eval(v)[0] - eval(v)[1]) == 0 and prev_key != None and df_CO.iloc[prev_key]['Main_Section'] == df_CO.iloc[k]['Main_Section']:
                    # print(k,v)
                    df_CO['Group'] = df_CO['Group'].replace(df_CO.iloc[k]['Group'],'nan')
            except:
                pass
            prev_key = k

        df_CO['Group'] = df_CO['Group'].mask(df_CO['Group']=='nan', None).ffill()
        df_CO['Range'] = 'nan'
        grp_set = set(df_CO['Group'].to_list())
        df_CO['Group Date'] = ''
        for i in grp_set:
            index = df_CO[df_CO['Group'] == i].index[-1]
            df_CO.at[index,'Range'] = str((df_CO[df_CO['Group'] == i].iloc[0]['Page'],df_CO[df_CO['Group'] == i].iloc[-1]['Page']))
            try:
                df_CO.at[index,'Group Date'] = max([i for i in df_CO[(df_CO['Group'] == i) & (df_CO['section_entity'] == "SECTION END")]['Post Process Date/Time'].to_list() if i != 'nan'])
            except:
                pass
            try:
                df_CO.at[index,'_Score_'] = df_CO[(df_CO['Group'] == i) & (df_CO['section_entity'] == 'SECTION')].iloc[0]['score']
            except:
                pass

        df_CO['Group Date'] = pd.to_datetime(df_CO['Group Date'])
        df_CO.drop(columns={'Date'},inplace=True)
        df_CO_date_order = df_CO[df_CO['Range'] != 'nan'].sort_values(by=['Group Date'])

        self.df_CO = df_CO
        self.df_CO_date_order = df_CO_date_order

    def ChartOrder(self,df):
        self.ChartOrderLogic(df)

    def IndexPage(self, index_path, df):
        logging.info('Adding an index page with hyperlinks associated with it')
        File_name = self.File_Name
        df_CO_Index = pd.DataFrame({"Chart Order": ['Demographics (DEMO) [Face sheet, Coding Worksheet, Financial]',
                                    'Emergency Department (ED)',
                                    'History and Physical (H&P)',
                                    'Progress Notes including Consults (Physician/QHP)',
                                    'Operative/Procedure Note',
                                    'Discharge Summary (DS)',
                                    'Therapy Notes [PT/OT/ST etc.]',
                                    'Dietary/Nutritional Notes',
                                    'Nursing Documentation',
                                    'LABS',
                                    'Imaging',
                                    'Orders',
                                    'Miscellaneous [Consent]'],
                                })
        Chart_Order_df = pd.read_excel(rf"{self.constant_path}\Chart Order.xlsx",sheet_name='Sheet2')
        Chart_Order_df['Chart order'] = Chart_Order_df['Chart order'].mask(Chart_Order_df['Chart order']==np.NaN, None).ffill()

        self.df_CO_date_order = self.df_CO_date_order.reset_index()
        for i in range(len(self.df_CO_date_order)):
            entity = self.df_CO_date_order.iloc[i]['Main_Section']
            try:
                C_O = Chart_Order_df[Chart_Order_df['Main Section'] == entity]['Chart order'].values[0]
                self.df_CO_date_order.at[i, 'Chart Order'] = C_O
            except:
                pass

        df_order_entity = self.df_CO_date_order[self.df_CO_date_order['Range'] != 'nan'].sort_values(by=['Chart Order','Group Date'])

        def make_clickable(page, name, File_name):
            if page != '':
                # return '<a href="C:\\Users\\Rahuly\\Desktop\\RevMax Ai\\Medical Records\\{}.pdf#page={}" rel="noopener noreferrer" target="_blank">{}</a>'.format(File_name, str(page.split("-")[0]), name)
                return '<a href="{}.pdf#page={}" rel="noopener noreferrer" target="_blank">{}</a>'.format(File_name, eval(page)[0], name)
            else:
                return '{}'.format(name)

        df_order_entity['entity'] = df_order_entity.apply(lambda x: make_clickable(x['Range'], x['entity'], File_name), axis=1)
        # recheck
        for i in df_CO_Index['Chart Order']:
            if len(df_order_entity[df_order_entity['Chart Order'] == i]):
                with open(rf"{index_path}\{File_name}_index.html", "a") as f:
                    f.write("<center><strong>" + i + "</strong> </center><br>\n")
                    html = df_order_entity[df_order_entity['Chart Order'] == i][['entity','Group Date','Range','_Score_']].reset_index().drop(columns={'index'}).rename(columns={'entity':'Section Name','Range':'Page Range'}).to_html(render_links=True, escape=False)
                    f.write("<center>" + html + "</center><br>\n")
                    f.write("<br><br>\n")
                print(i)
        # recheck-----
        logging.info('Converting DataFrame to html')
        html = df_order_entity.to_html(render_links=True, escape=False)

        logging.info('Converting html to pdf')
        path = os.path.abspath(rf"{index_path}\{self.File_Name}_index.html")
        converter.convert(f'file://{path}', rf'{index_path}\{self.File_Name}_index.pdf')

    def save_result(self, path_to_save_result, df):
        logging.info('Saving the file')
        # -------------------------------------------------------------------------------------------- 
        from openpyxl import load_workbook
        import pandas as pd
        save_file_start_time = time.time()
        df_raw = pd.read_csv(rf"{self.textract_file}\{self.File_Name}.csv",na_filter=False)
        with pd.ExcelWriter(rf'{path_to_save_result}\{self.File_Name}_section_subsection.xlsx') as writer:
            df_raw.to_excel(writer, sheet_name="RAW")
            df.to_excel(writer,sheet_name="Processed")
            # df.drop(columns={'main_section'}).rename(columns={'Updated_main_section':'main_section'}).to_excel(writer,sheet_name="Processed")
            self.Sections_DF.to_excel(writer,sheet_name="SECTION")
            self.df_end.to_excel(writer,sheet_name="SECTION END")
            self.df_CO.to_excel(writer, sheet_name = "CHART ORDER")
            self.df_CO_date_order.to_excel(writer, sheet_name='Date Order',index=False)
            self.df_CO_date_order[self.df_CO_date_order['Range'] != 'nan'].sort_values(by=['Chart Order','Group Date']).to_excel(writer, sheet_name='Chart View',index=False)
        logging.info('Saved')
        save_file_end_time = time.time()

        logs_start_time = time.time()
        with open(rf"C:\Users\Rahuly\Desktop\RevMax Ai\Logs_v2\{self.File_Name}.txt", 'w') as f:
            f.write(f"SECTION Page List : {list(set(df[df['section_entity'] == 'SECTION']['Page'].to_list()))}")
            for i in list(set(df[df['section_entity'] == 'SECTION']['entity'].values)):
                f.write(f"\n{i} : {sorted(list(set(df[(df['section_entity'] == 'SECTION') & (df['entity'] == i)]['Page'].to_list())))}")

            if "sub_section_entity" in df.columns:
                f.write(f"\n\nSUB SECTION Page List : {list(set(df[df['section_entity'] == 'SUB SECTION']['Page'].to_list()))}")
                for i in list(set(df[df['section_entity'] == 'SUB SECTION']['sub_section_entity'].values)):
                    f.write(f"\n{i} : {sorted(list(set(df[(df['section_entity'] == 'SUB SECTION') & (df['sub_section_entity'] == i)]['Page'].to_list())))}")

            f.write(f"\n\nSECTION END Page List : {list(set(df[df['section_entity'] == 'SECTION END']['Page'].to_list()))}")

        logs_end_time = time.time()