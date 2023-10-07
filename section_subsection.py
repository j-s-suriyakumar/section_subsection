import json
import re
import boto3
import pandas as pd
from urllib.parse import unquote_plus
from io import StringIO
from rapidfuzz import fuzz, process
from dateutil.parser import parse

# s3_c = boto3.client('s3')

## Similarity method and string processing functions

# def lambda_handler(event, context):
# def lambda_handler():
    
    # file_obj = event["Records"][0]
    # bucket_name = str(file_obj["s3"]["bucket"]["name"])
    # file_name = unquote_plus(str(file_obj["s3"]["object"]["key"]))
    file_path = r"C:\Users\chirag\Downloads\229954040.309398918.H68414674.csv"
    file_name = r'229954040.309398918.H68414674.csv'
    # print(file_name, bucket_name,1)
    print(file_name)
    
    ## Read data to run on
    # csv_obj = s3_c.get_object(Bucket=bucket_name, Key=file_name)
    # print(file_name, bucket_name, 2)
    
    # body = csv_obj['Body']
    # csv_string = body.read().decode('utf-8')
    # df1 = pd.read_csv(StringIO(csv_string))
    df1 = pd.read_csv(file_path)

    # print("input file read:", file_name)
    
    ## Read stop_words list
    # csv_obj = s3_c.get_object(Bucket='human-review-stage', Key='stop_words_v2.csv')
    # body = csv_obj['Body']
    # csv_string = body.read().decode('utf-8')
    # stop_words = pd.read_csv(StringIO(csv_string))
    stop_words = pd.read_csv('./constant/stop_words_v2.csv')
    stop = stop_words['Stop_words'].to_list()
    
    print("stop_words read")
    
    ## Creation of required lists
    # csv_obj = s3_c.get_object(Bucket='human-review-stage', Key='Final_Sections.csv')
    # body = csv_obj['Body']
    # csv_string = body.read().decode('utf-8')
    # completed_data = pd.read_csv(StringIO(csv_string))
    completed_data = pd.read_csv('./constant/Final_Sections.csv')
    
    completed_data_rel = completed_data[completed_data['Is Relevant'] == 'Yes']
    completed_data_irrel = completed_data[completed_data['Is Relevant'] == 'No']
    rel_sections = completed_data_rel['Section Name'].to_list()
    irrel_sections = completed_data_irrel['Section Name'].to_list()
    irrelvant_sections_lower = list(set([ele.lower() for ele in irrel_sections]))
    total_sections = list(set(rel_sections + irrel_sections))
    sec_list1 = sorted(total_sections, key = len, reverse = True)
    
    # csv_obj = s3_c.get_object(Bucket='human-review-stage', Key='Final_Subsections.csv')
    # body = csv_obj['Body']
    # csv_string = body.read().decode('utf-8')
    # Final_data = pd.read_csv(StringIO(csv_string))
    Final_data =  pd.read_csv('./constant/Final_Subsections.csv')
    
    print("sections and subsections masterlist read")
    
    ## date_strings list
    # csv_obj = s3_c.get_object(Bucket='human-review-stage', Key='date_str_variations.csv')
    # body = csv_obj['Body']
    # csv_string = body.read().decode('utf-8')
    # date_str = pd.read_csv(StringIO(csv_string))
    date_str = pd.read_csv('./constant/date_str_variations.csv')
    date_strings = date_str['Date string types'].to_list()
    date_strings.sort(key = len, reverse = True)
    
    Main_date_string = [ele.replace('date','').strip() for ele in date_strings if ele.endswith('date')][:-1]
    Main_date_string.sort(key = len, reverse = True)
        
    creds = [' MD ',' DO ',' DPM ',' PA ',' PA-C ',' CRNA ',' RN ',' D.O. ',' PHD ',' MBBS ', ' NP ', ' APRN ', ' DNP-APRN ']
    creds.sort(key = len, reverse = True)
    my_list = ['electronically witnessed' ,'electronically signed', 'addendum by', 'addendum', 'signed by', 'added by', 'edited by','authenticated by','consults by','signed on']
    my_list.sort(key = len, reverse = True)
    my_list1 = ['electronically witnessed' ,'electronically signed', 'addendum by', 'addendum', 'signed by', 'added by', 'edited by','authenticated by','consults by', 'reviewed by','transcribv by','transcribe by','review by','sign by','verify by','perform','signed on']
    my_list1.sort(key = len, reverse = True)
    
    print("Read the required dataframes.")
    
    ######################################################################################################################################################
    
    def str_preprocessing(text, stop):
        string = re.sub(r'([^A-Za-z0-9\s]+?)', '', text)
        string1 = string.lower().strip().split()
        string1 = list(filter(None, string1))#[ele for ele in string1 if ele != '']
        No_of_stopwords = len([i for i in string1 if i in stop])
        len_words_raw_text = [i for i in string1 if i not in stop]
        raw_str_match = ' '.join(len_words_raw_text)
        raw_str_match = raw_str_match.replace(' +', ' ')
        return raw_str_match, len(len_words_raw_text), No_of_stopwords
    
    #section identification function
    def sec_marking(df1, m,score1,irrelvant_sections_lower,original_sec, sec,section_notation,column,mapped_list_str):
        df1['SECTION_ENTITY'][m] = section_notation
        df1['Score'][m] = score1
        df1[mapped_list_str][m] = original_sec
        if sec.lower() in irrelvant_sections_lower:
            df1[column][m] = 'No'
        else:
            df1[column][m] = 'Yes'
        
        return df1
    
    #subsection identification function
    def sub_sec_detection(df1, text, sub_sec_list1,sub_sec_list2, irrel_sub_sections, length, score):
        sub_section_notation = 'SUB_SECTION'
        column = 'relevant_sub_sec'
        mapped_list_str = 'sub_section'
        
        ratio_sort = process.extractOne(text, sub_sec_list2, scorer=fuzz.token_sort_ratio)
        if ratio_sort[1] >= 75:
            ratio_partial = round(fuzz.partial_ratio(str(text),ratio_sort[0]), 2)
            average_ratio = round((ratio_sort[1] + ratio_partial)/2)
            original_sub_sec = sub_sec_list1[ratio_sort[2]]
            
            if len(original_sub_sec.split()) ==1 and average_ratio > 95 and length >= len(original_sub_sec.split()):
                df1 = sec_marking(df1, 
                                  index,average_ratio, 
                                  irrel_sub_sections, 
                                  original_sub_sec,
                                  sub_sec_list1[ratio_sort[2]],
                                  sub_section_notation,
                                  column,
                                  mapped_list_str)
    
            elif len(original_sub_sec.split()) > 1 and average_ratio > 87.5 and length >= len(original_sub_sec.split()):
                df1 = sec_marking(df1, index,average_ratio, irrel_sub_sections, original_sub_sec,sub_sec_list1[ratio_sort[2]],sub_section_notation, column, mapped_list_str)
            else:
                None
        return df1
    
    def sub_sec_list_suppress(i, Final_data):
        common_section = Final_data[Final_data['Section Name'] == i].index.to_list()[0]
        indexes = df1[df1['section'] == i].index.tolist() #taking only indexws of the section
        ###### getting subsections list for the respected section
        split_df = Final_data[Final_data['Main_section'] == Final_data['Main_section'][common_section]]
        
        irrel_sub_sections = split_df[split_df['is Relevant'] == 'No']['Sub Section Name'].to_list()
        irrel_sub_sections = [ele.lower() for ele in irrel_sub_sections]#list(map(lambda x: x.lower(), irrel_sub_sections))#
        Sub_section_list = split_df['Sub Section Name'].to_list()
        sub_sec_list1 = sorted(Sub_section_list, key = len, reverse = True)
        
        return indexes, irrel_sub_sections, sub_sec_list1
    
    def map_section_list(df1):
        df1['section'] = df1['section'].mask(df1['section']=='', None).ffill() #ffill of section column
        dff = df1[df1['relevant_sec'] == 'Yes']
        Map_section_list = list(set(dff['section']))
        #removing unwanted strings from the list of mapped sections
        Map_section_list1 = list(filter(None, Map_section_list))
        # Map_section_list1=[x for x in Map_section_list if x != None and str(x) != ''] 
        Map_section_list1 = list(set(Map_section_list1).difference(irrel_sections))
        return df1, Map_section_list1

     ##Date regex
    
    def date_regex(text):
        pattern = r"((?:(?:\d{1,2}[/ -])?(?:\d{1,2}[/ -])?\d{2,4})|(?:(?:January|Jan|February|Feb|March|Mar|April|Apr|May|June|July|August|Aug|September|Sep|October|Oct|November|Nov|December|Dec)\s\d{1,2}(st|nd|rd|th)?)(?:,?\s\d{2,4})?|(?:(?:Monday|Mon|Tuesday|Tue|Wednesday|Wed|Thursday|Thur|Friday|Fri|Saturday|Sat|Sunday|Sun),?\s)?(?:(?:January|Jan|February|Feb|March|Mar|April|Apr|May|June|July|August|Aug|September|Sep|October|Oct|November|Nov|December|Dec)\s\d{1,2}(st|nd|rd|th)?,?\s\d{2,4})|(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s\d{2})|(?:(?:\d{1,2}|0[1-9]|1[0-2])(?:[-/])(?:\d{1,2}|0[1-9]|[1-2][0-9]|3[0-1])(?:[-/])\d{2,4}))"
        reg_date = re.findall(pattern, text)
        date_elements = [ele for ele in [item for t in reg_date for item in t] if (ele != '') if len(ele) > 6]
        date_elements_2 = [ele for ele in [item for t in reg_date for item in t] if (ele != '') if len(ele) == 6]
        date_elements2 = [ele for ele in date_elements_2 if any(x for x in ele.split("-" or "/" or " ") if len(x) > 2) is False]
        date_elements = date_elements + date_elements2
        return date_elements
    
        
    def tag_check(str_above_date):
        if len(str_above_date) > 20:
            check = [ele for ele in date_strings if ele in str(str_above_date).lower()]
            if len(check) != 0:
                str_above_date = check[0]
        else:
            str_above_date = str_above_date
        return str_above_date
                                
    def string_split(str_before_date):
        if ';' in str_before_date:
            str_before_date = str_before_date.split(';')[-1]
        else:
            str_before_date = str_before_date
        return str_before_date

    def date_tag(string, n, data):
        tag_str = ["dt\tm:","dt\tm","date", "dt ","dt:", "dos ", "dos:", "adm:","d/c:","d/s:","dob","printed","generated", "date of birth"]
        date_tags = []
        if dates_present_list[n] != []:
            dates_list = dates_present_list[n]
            l = len(dates_list)
            c = 0
            
            while c < l:
                if c == 0:
                    if count-1 > 1:
                        str_before_date = str(list1[n].split(dates_list[c])[0])
                        str_before_date = string_split(str_before_date)
                        str_above_date = str(list1[n-1])
                        str_above_date = string_split(str_above_date)
    
                        ## Case 1
                        if str_before_date != "" and [x for x in tag_str if x in str_before_date.lower()] != []:
                            str_before_date1 = [ele for ele in [x for x in tag_str if x in str_before_date.lower()] if ele in ['dob','dos','adm:','d/c:','d/s:']]
                            if len(str_before_date1) >0:
                                date_tags.append(str_before_date1[0])
                            else:
                                date_tags.append(str_before_date)
                        ## Case 2
                        elif str_above_date != "" and [x for x in tag_str if x in str_above_date.lower()] != []:
                            str_above_date = tag_check(str_above_date)
                            date_tags.append(str_above_date)
                        else:
                            None
                    
                    else:
                        str_before_date = str(list1[n].split(dates_list[c])[0])
                        str_before_date = string_split(str_before_date)
    
                        ## Case 1
                        if str_before_date != "" and [x for x in tag_str if x in str_before_date.lower()] != []:
                             str_before_date1 = [ele for ele in [x for x in tag_str if x in str_before_date.lower()] if ele in ['dob','dos','adm:','d/c:','d/s:']]
                             if len(str_before_date1) >0:
                                 date_tags.append(str_before_date1[0])
                             else:
                                 date_tags.append(str_before_date)
                        else:
                            None
    
                else:
                    str_before_date = str((list1[n].split(dates_list[c])[0]).split(dates_list[c-1])[-1])
                    str_before_date = string_split(str_before_date)
    
                    if str_before_date != "" and [x for x in tag_str if x in str_before_date.lower()] != []:
                        str_before_date1 = [ele for ele in [x for x in tag_str if x in str_before_date.lower()] if ele in ['dob','dos','adm:','d/c:','d/s:']]
                        if len(str_before_date1) >0:
                            str_before_date = tag_check(str_before_date1[0])
                            date_tags.append(str_before_date)
                        else:
                            str_before_date = tag_check(str_before_date)
                            date_tags.append(str_before_date)
                    
                    elif str_before_date.strip() in ["-", "to"]:
                        if date_tags == []:
                            None
                        else:
                            date_tags.append(date_tags[-1])
    
                    else:
                        None
                c+= 1
        return date_tags
    
    print("defined all required functions.")
    
        ######################################################################################################################################################
    
    ## Run
    df1[['Page_LINK','SECTION_ENTITY', 'Score', 'section', 'relevant_sec','sub_section','relevant_sub_sec','dates_present','date_tags','concat_datetag','provider_cred','provider_name']] = ''
    # df1["date_tags"] = [list() for x in range(len(df1.index))]
    # df1.rename(columns = {'Text':'No PHI text'}, inplace = True)
    max_num_page = max(df1['Page'].to_list())
    
    # for index, pg in df1['Page'].items():
    #     df1['Page_LINK'][index] = 'https://human-review-stage.s3.us-east-2.amazonaws.com/revmaxai-digitized-output/' + str(file_name.replace('flattened-csv/','').replace('.csv','')) +'/'+ str(file_name.replace('flattened-csv/','').replace('.csv','')) + '_page_'  + str(pg) + '_of_' + str(max_num_page) + '.pdf'
    
    sec_list2 = [str_preprocessing(ele, stop)[0] for ele in sec_list1]
    
    list1 = df1['Text'].to_list()
    list_pre = [str_preprocessing(str(ele).replace('(continued)',''), stop)[0] for ele in list1]
    list_noofstop = [str_preprocessing(str(ele).replace('(continued)',''), stop)[2] for ele in list1]
    
    score_list = []
    section_list = []
    SECTION_ENTITY_list = []
    relevant_sec_list = []
    dates_present_list = []
    date_tag_list = []
    string_list = []
    count = 0
    string = ''
    for x in list_pre:
        
        count +=1
        matcher = process.extractOne(x, sec_list2, scorer=fuzz.token_sort_ratio)
        Score = ''
        section = ''
        relevant_sec = ''
        SECTION_ENTITY = ''
        if matcher[1] >= 75:
            partial_rat = fuzz.partial_ratio(x, matcher[0].lower().strip())
            average_score = (matcher[1] + partial_rat)/2
            split_text = matcher[0].split() #sec words split
            number_check = re.sub(r'([^0-9]+?)', '', str(matcher[0]))
            No_of_stop_raw = list_noofstop[count-1]
            word_check = len(x.split()) >= len(matcher[0].split())
            len_of_string = len(x)
                
            if len(split_text) == 1 and average_score > 95 and len_of_string >= 3 and word_check == True and No_of_stop_raw <= 2:
                Score = average_score
                section = sec_list1[matcher[2]]
                if sec_list1[matcher[2]].lower() in irrelvant_sections_lower:
                    relevant_sec = 'No'
                else:
                    relevant_sec = 'Yes'
                SECTION_ENTITY = 'SECTION'
                string = relevant_sec
            elif len(split_text) == 1 and average_score > 87.5 and number_check == '' and len_of_string >= 3 and word_check == True and '(continued)' in list1[count-1] and No_of_stop_raw <= 2:
                Score = average_score
                section = sec_list1[matcher[2]]
                if sec_list1[matcher[2]].lower() in irrelvant_sections_lower:
                    relevant_sec = 'No'
                else:
                    relevant_sec = 'Yes'
                SECTION_ENTITY = 'SECTION'
                string = relevant_sec
    
            elif len(split_text) > 1 and average_score > 87.5 and number_check == '' and len_of_string >= 3 and word_check == True and No_of_stop_raw <= 2:
                Score = average_score
                section = sec_list1[matcher[2]]
                if sec_list1[matcher[2]].lower() in irrelvant_sections_lower:
                    relevant_sec = 'No'
                else:
                    relevant_sec = 'Yes'
                SECTION_ENTITY = 'SECTION'
                string = relevant_sec
    
            else:
                # string = ''
                None
        x2 = list1[count-1]
        if any(substring in str(x2).lower() for substring in my_list): #check if need to update with my_list1
            SECTION_ENTITY = 'SECTION_END'
        else:
            None
        
        string_list.append(string)
        score_list.append(Score)
        section_list.append(section)
        SECTION_ENTITY_list.append(SECTION_ENTITY)
        relevant_sec_list.append(relevant_sec)
    
        if string == 'Yes':
            # print(True)
            x1 = list1[count-1]
            n = count-1
            for substr in my_list: #[electronically signed..]
                if substr in str(x1).lower():
                    prv_name_str1 = str(x1)[str(x1).lower().find(substr) + len(substr):]
                    
                    if len(prv_name_str1.strip()) == 0:
                        num = 10
                        for i in range(1,num+1):
                            prv_name_str1 = list1[n+i]
                        
                            # cred = [c for c in creds if c in prv_name_str1]
                            cred = [c for c in creds if c.strip() in prv_name_str1]
                            if cred != []:
                                df1["provider_cred"][n+i] = str(cred[0])
                                prv_name = prv_name_str1.split(cred[0])[0].replace(",","").strip()
                                df1["provider_name"][n+i] = prv_name
    
                    else:
                        # cred = [c for c in creds if c in prv_name_str1]
                        cred = [c for c in creds if c.strip() in prv_name_str1]
                        if cred != []:
                            df1["provider_cred"][n] = str(cred[0])
                            prv_name = prv_name_str1.split(cred[0])[0].replace(",","").strip()
                            df1["provider_name"][n] = prv_name
            
            if any(substring in str(x1).lower() for substring in my_list1):
                dates_present_list.append(date_regex(str(x1)))
                date_tag_list.append(['signed by date'])
            elif any(substring in str(x1).lower() for substring in ['printed', 'generated on','ted on']):
                dates_present_list.append(date_regex(str(x1)))
                date_tag_list.append(['printed date'])
            else:
                dates_present_list.append(date_regex(str(x1)))
                date_tag_list.append(date_tag(x1, count-1, dates_present_list))
    
        else:
            date_tag_list.append([])
            dates_present_list.append([])
                
    df1['date_tags'] = date_tag_list
    df1['dates_present'] = dates_present_list
    df1['Score'] = score_list
    df1['section'] = section_list
    df1['SECTION_ENTITY'] = SECTION_ENTITY_list
    df1['relevant_sec'] = string_list
    
    df1, Map_section_list1 = map_section_list(df1)
    
    for i in Map_section_list1:
        # try:
        indexes, irrel_sub_sections, sub_sec_list1 = sub_sec_list_suppress(i, Final_data)
        sub_sec_list2 = [str_preprocessing(ele, stop)[0] for ele in sub_sec_list1]
    
        #################sub-section mapping ##################
        for index in indexes:
            j = str(df1['Text'][index])
            if ':' in j and df1['SECTION_ENTITY'][index] == '':
                string = j.split(':')[0]
                raw_str_match, length, No_of_stop_raw_sub_sec = str_preprocessing(string, stop)
    
                df1 = sub_sec_detection(df1, raw_str_match, sub_sec_list1, sub_sec_list2, irrel_sub_sections, length, 87.5)
            elif df1['SECTION_ENTITY'][index] == '':
                string = j
                raw_str_match, length, No_of_stop_raw_sub_sec = str_preprocessing(string, stop)
    
                if No_of_stop_raw_sub_sec <= 2:
                    df1 = sub_sec_detection(df1, raw_str_match, sub_sec_list1, sub_sec_list2, irrel_sub_sections, length, 87.5)
                else:
                    None
    
    
    admit_date = ''
    discharge_date = ''

    for index, j in df1['Page'].items():
        if j <= 20 and df1['relevant_sec'][index] != 'Yes':
            if any(substring in str(df1['Text'][index]).lower() for substring in my_list1):
                df1['dates_present'][index] = date_regex(str(df1['Text'][index]))
                if df1['dates_present'][index] != []:
                    df1.at[index, 'date_tags'] = ['signed by date']
            elif any(substring in str(df1['Text'][index]).lower() for substring in ['printed', 'generated on','ted on']):
                df1['dates_present'][index] = date_regex(str(df1['Text'][index]))
                if df1['dates_present'][index] != []:
                    df1.at[index, 'date_tags'] = ['printed date']
            else:
                df1['dates_present'][index] = date_regex(str(df1['Text'][index]))
                if df1['dates_present'][index] != []:
                    df1.at[index, 'date_tags'] = date_tag(str(df1['Text'][index]), index, df1['dates_present'])  
        
    
    adm_date_list = []
    dis_date_list = []
    for m,n in df1['date_tags'].items():
        cnt = 0
        if df1['date_tags'][m] != []:
            for ele in n:
                cnt += 1
                if 'admission date' in ele.lower() or 'admit date' in ele.lower() or 'adm date' in ele.lower() or 'admit dt' in ele.lower() or 'registration date'  in ele.lower() or 'admit' in ele.lower() or 'adm:' in ele.lower():
                    date_list = df1['dates_present'][m]
                    admit_date = date_list[cnt-1]
                    adm_date_list.append(admit_date)
                elif 'discharge date' in ele.lower()  or 'discharge' in ele.lower() or 'dis date' in ele.lower() or 'dischg date' in ele.lower() or 'd/c:' in ele.lower() or 'd/c' in ele.lower():
                    date_list1 = df1['dates_present'][m]
                    discharge_date = date_list1[cnt-1]
                    dis_date_list.append(df1['dates_present'][m][cnt-1])
                else:
                    None
    if len(adm_date_list) >0:
        admit_date = min(adm_date_list)
    if len(dis_date_list) >0:
        discharge_date = max(dis_date_list)
    
    
    def date_checking(df1, min_date, max_date):
        for m,n in df1['dates_present'].items():
            ser_dates = []
            cnt = 0
            if n != []:
                for ele in n:
                    cnt +=1
                    try:
                        date_parse = parse(ele)
                        if min_date != '' and max_date != '':
                            if parse(min_date) <= date_parse <= parse(max_date):
                                ser_dates.append(n[cnt-1])
                            else:
                                if len(df1['date_tags'][m]) > 0 and len(df1['date_tags'][m]) > cnt:
                                    df1['date_tags'][m].pop(cnt-1)
                        else:
                            ser_dates.append(n[cnt-1])
                    except:
                        if len(df1['date_tags'][m]) > 0 and len(df1['date_tags'][m]) > cnt:
                            df1['date_tags'][m].pop(cnt-1)
                            
                        
            df1.at[m, 'dates_present'] = ser_dates
            if ser_dates == []:
                df1.at[m, 'date_tags'] = []
            if ser_dates != [] and df1['date_tags'][m] == []:
                df1.at[m, 'date_tags'] = ['service_date']
        return df1
    

    df1 = date_checking(df1, admit_date, discharge_date)
    
    for index,j in df1.iterrows():
        if len(j['date_tags']) == 1:
            date_string = [ele for ele in date_strings if ele in j['date_tags'][0].lower()]
            if len(date_string) >= 1 :
                df1.at[index, "concat_datetag"] = [date_string[0] + ':' + '-'.join(df1['dates_present'][index])]
            else:
                df1.at[index, "concat_datetag"] = [df1['date_tags'][index][0] + ':' + '-'.join(df1['dates_present'][index])]
        elif len(j['date_tags']) >1:
            concat_dt = []
            c = 0
            for ele1 in j['date_tags']:
                date_string = [ele for ele in date_strings if ele in ele1.lower()]
                if len(date_string) >0 and len(j['dates_present']) > c:
                    concat_date = date_string[0] + ':' + j['dates_present'][c]
                else:
                    if len(j['dates_present']) > c:
                        concat_date = ele1 + ':' + j['dates_present'][c]
                concat_dt.append(concat_date)
                c +=1
            df1.at[index, "concat_datetag"] = concat_dt
            
    print("Ran all identification, saving file")

    # csv_buf = StringIO()
    # df1.to_csv(csv_buf, header=True, index=False)
    # csv_buf.seek(0)
    # s3_c.put_object(Bucket=bucket_name, Body=csv_buf.getvalue(), Key=file_name.replace("flattened-csv/","section_subsection_identification/"))
        
    # print("saved the output file", file_name.replace("flattened-csv/","section_subsection_identification/"))
    df1.to_csv('./batch6/229954040.309398918.H68414674_sec_subSec.csv', header=True, index=False)
    return 'file saved!'

lambda_handler()