import json
import re
import pandas as pd
from urllib.parse import unquote_plus
from io import StringIO
from rapidfuzz import fuzz, process
from dateutil.parser import parse
import logging
from constants import REPLACE_STRINGS, SECTION_END_PATTERNS

# Configure the logging system
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s')
logging.info('FUNCTION START: rev_section_subsection.py')

# Load stop words
stop_words = pd.read_csv(r'revdata\stop_words_v2.csv')
stop = stop_words['stop_words'].tolist()
logging.info('Stop words loaded')

# load the sections
section_data = pd.read_csv(r'revdata\final_sections.csv')
logging.info('Pre defined Section Names loaded')

rel_sec = section_data[section_data['is_relevant']
                       == 'Yes']['section_name'].to_list()
irrel_sec = section_data[section_data['is_relevant']
                         == 'No']['section_name'].to_list()

irrel_sec_lower = list(set([ele.lower() for ele in irrel_sec]))
# combine all the sections
total_sections = list(set(rel_sec + irrel_sec))
sec_list = sorted(total_sections, key=len, reverse=True)

# Load sub section data
sub_section_data = pd.read_csv(r'revdata\final_subsections.csv')
logging.info('Pre defined Sub Section Names loaded')

# Load Date formats
date_variations = pd.read_csv(r'revdata\date_str_variations.csv')
date_formats = date_variations['date_formats'].tolist()
logging.info('Date formats loaded')

# Load test data
df = pd.read_csv(r'revdata\test_sec_doc.csv')
df = df[df['BlockType'] == 'LINE']
# text preprocessing function


def text_cleanup(text, stop):
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

    stopword_count = len([i for i in t3 if i in stop])
    non_stopword_count = [i for i in t3 if i not in stop]
    non_stopword_count_length = len([i for i in t3 if i not in stop])
    non_stopword_text = ' '.join(non_stopword_count)
    non_extra_whitespaces_text = non_stopword_text.replace(' +', ' ')

    return non_extra_whitespaces_text, non_stopword_count_length, stopword_count

def update_cleaned_dataframe(i,sec_subsec_list, sec_name, average_score,section_entity):
        
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

        section_name = sec_name
        cleaned_df.loc[i, section_entity] = section_name
        cleaned_df.loc[i, 'SCORE'] = average_score
        relevancy = 'No' if sec_subsec_list[section_name].lower(
        ) in irrel_sec_lower  else 'Yes'
        cleaned_df.loc[i, 'SECTION_ENTITY'] = section_entity
        cleaned_df.loc[i, 'IS_RELEVANT'] = relevancy
        logging.info('UPDATED DATA INFO: {} ----- {} ----- {} ----- {} ----- {}'.format(i, txt, section_name, average_score, section_entity))

# Preprocess the text
logging.info('FUNCTION START: str_preprocessing')
cleaned_sections = [text_cleanup(sec, stop) for sec in sec_list]
logging.info('FUNCTION END: str_preprocessing')
clean_text_tuple = df['Text'].apply(text_cleanup, args=(stop,))

cleaned_df = pd.DataFrame(list(clean_text_tuple), columns=[
                          'non_extra_whitespaces_text', 'non_stopword_count_length', 'stopword_count'])
logging.info('Cleaned text dataframe created')

clean_text_list = cleaned_df['non_extra_whitespaces_text'].tolist()
stop_word_count_list = cleaned_df['stopword_count'].tolist()


for i, txt in cleaned_df.iterrows():

    SCORE = ''
    SECTION = ''
    RELEVANT_SECTION = ''
    SECTION_ENTITY = ''
    clean_txt_length = len(str(txt['non_extra_whitespaces_text']))

    # Section Matching
    matcher = process.extractOne(
        txt['non_extra_whitespaces_text'], cleaned_sections, scorer=fuzz.token_set_ratio)
    logger.info(
        'SECTION MATCHER: {} ----- {}'.format(txt['non_extra_whitespaces_text'], matcher))
    
    if matcher[1] >= 75:
        partial_ratio_score = fuzz.partial_ratio(
            txt['non_extra_whitespaces_text'], matcher[0], processor=lambda x: x.lower().strip())
        average_score = (matcher[1] + partial_ratio_score)/2
        logger.info(
            'SECTION MATCHER: {} ----- {}'.format(txt['non_extra_whitespaces_text'], average_score))
        split_master_sec_by_words = matcher[0].split()
        is_num = re.sub(r'([^0-9]+?)', '', str(matcher[0]))
        is_raw_text_greater = len(txt['non_extra_whitespaces_text']) >= len(
            split_master_sec_by_words)

    if len(split_master_sec_by_words) == 1 and average_score > 95 and clean_txt_length >= 3 and is_raw_text_greater and txt['stopword_count'] < 3:
        update_cleaned_dataframe(i, sec_list, sec_list[matcher[2]], average_score,'SECTION')

    elif len(split_master_sec_by_words) == 1 and average_score > 87.5 and is_num == '' and clean_txt_length >= 3 and is_raw_text_greater and REPLACE_STRINGS in str(txt['non_extra_whitespaces_text']) and txt['stopword_count'] < 3:
        update_cleaned_dataframe(i, sec_list, sec_list[matcher[2]], average_score,'SECTION')

    elif len(split_master_sec_by_words) > 1 and average_score > 87.5 and is_num == '' and clean_txt_length >= 3 and is_raw_text_greater and txt['stopword_count'] < 3:
        update_cleaned_dataframe(i, sec_list, sec_list[matcher[2]], average_score,'SECTION')

    elif SECTION_ENTITY == '':
        for sec_end_pattern in SECTION_END_PATTERNS:
            if sec_end_pattern in str(txt['non_extra_whitespaces_text']).lower():
                update_cleaned_dataframe(i, sec_list, sec_end_pattern, 100,'SECTION END')
                break

cleaned_df['SECTION'] = cleaned_df['SECTION'].mask(cleaned_df['SECTION'] =='', None).ffill() #ffill of section column
relevant_sections_df = cleaned_df[cleaned_df['relevant_sec'] == 'Yes']
relevant_section_list = list(set(relevant_sections_df['section']))
relevant_section_list = list(filter(None, relevant_section_list))
filtered_relevant_section_list = list(set(relevant_section_list).difference(irrel_sec))


for i in filtered_relevant_section_list:
    # Step 1: Find the index of the mapped main section
    # Step 2: Get subsections related to the mapped main section
    # Step 3: Get indexes of rows in relevant_sections_df where 'section' matches 'i'
    # Step 4: Extract irrelevant subsections
    # Step 5: Extract all subsections and sort them by length in reverse order
    # Step 6: Return the extracted information
    mapped_main_section = list(set(sub_section_data[sub_section_data['section_name'] == i]['main_section']))[0]
    logging.info('MAPPED MAIN SECTION: {}'.format(mapped_main_section))

    all_subsec_df = sub_section_data[sub_section_data['main_section'] == mapped_main_section]
    logging.info('List of subsections related to main section: {}'.format(all_subsec_df['sub_section'].tolist()))
    
    sec_idxs = relevant_sections_df[relevant_sections_df['section'] == i].index.tolist() 
    logging.info('Indexes of rows in relevant_sections_df where section matches i: {} - {}'.format(max(sec_idxs),min(sec_idxs)))

    rel_sub_sec_list = all_subsec_df[all_subsec_df['is_relevant'] == 'Yes']['sub_section_name'].to_list()
    rel_sub_sec_list = [ele.lower() for ele in rel_sub_sec_list]
    logging.info('Irrelevant subsections for the section {}: {}'.format(i, rel_sub_sec_list))

    all_sub_sections = all_subsec_df['sub_section_name'].to_list()
    all_sub_sections_list_sorted = sorted(all_sub_sections, key = len, reverse = True)
    logging.info('All subsections for the section {}: {}'.format(i, all_sub_sections_list_sorted))

    if len(mapped_main_section) == 0:
        logging.error('No mapped main section found for the section {}'.format(i))

def sub_sec_detection():
    # sub_section identification function    
    ratio_sort = process.extractOne(non_extra_whitespaces_text, cleaned_subsec, scorer=fuzz.token_sort_ratio)
    if ratio_sort[1] >= 75:
        ratio_partial = round(fuzz.partial_ratio(str(non_extra_whitespaces_text),ratio_sort[0]), 2)
        average_ratio = round((ratio_sort[1] + ratio_partial)/2)
        original_sub_sec = all_sub_sections_list_sorted[ratio_sort[2]]
        
        if len(original_sub_sec.split()) ==1 and average_ratio > 95 and non_stopword_count_length >= len(original_sub_sec.split()):
            update_cleaned_dataframe(i, rel_sub_sec_list, all_sub_sections_list_sorted[ratio_sort[2]], average_score,'SUB SECTION')
        elif len(original_sub_sec.split()) > 1 and average_ratio > 87.5 and non_stopword_count_length >= len(original_sub_sec.split()):
            update_cleaned_dataframe(i, rel_sub_sec_list, all_sub_sections_list_sorted[ratio_sort[2]], average_score,'SUB SECTION')
            pass


for i in filtered_relevant_section_list:

    cleaned_subsec = [text_cleanup(ele, stop)[0] for ele in all_sub_sections_list_sorted]
    for idx in sec_idxs:
        raw_cleaned_txt = str(relevant_sections_df['non_extra_whitespaces_text'][idx])
        if ':' in raw_cleaned_txt and relevant_sections_df['SECTION_ENTITY'][idx] == '':
            txt_colon_split = str(raw_cleaned_txt.split(':')[0]).strip()
            non_extra_whitespaces_text, non_stopword_count_length, stopword_count = text_cleanup(txt_colon_split, stop)
            # replace below line after cleaning up the below sub_section identification function
            relevant_sections_df = sub_sec_detection(relevant_sections_df, non_extra_whitespaces_text, all_sub_sections_list_sorted, cleaned_subsec, irrel_sub_sec, non_stopword_count_length, 87.5)
        elif relevant_sections_df['SECTION_ENTITY'][idx] == '':
            non_extra_whitespaces_text, non_stopword_count_length, stopword_count = text_cleanup(raw_cleaned_txt, stop)
            if stopword_count <= 2:
            # replace below line after cleaning up the below sub_section identification function
                relevant_sections_df = sub_sec_detection(relevant_sections_df, non_extra_whitespaces_text, all_sub_sections_list_sorted, cleaned_subsec, irrel_sub_sec, non_stopword_count_length, 87.5)


def date_extractor():
    pass

def physician_creds_extractor():
    pass

def physican_extractor():
    pass
