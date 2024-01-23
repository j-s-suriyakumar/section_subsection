import date_finder as dtf
import time_finder as tf
from rapidfuzz import fuzz, process
import json
import re


class DatetimeExtractor:
    def __init__(self, corpus):
        self.corpus = corpus

    @staticmethod
    def suppress_datetime(datetime_ls):
        """ 
        args:
            list: datetime_ls
        return:
            list: suppressed list
        """
        suppressed_list = []
        for datetime_tuple in datetime_ls:
            _, is_valid, _, _ = datetime_tuple
            if is_valid:
                suppressed_list.append(datetime_tuple)
        return suppressed_list

    @staticmethod
    def check_valid_range_datetime(date_obj, label, range_dict={'day': (1, 31), 'month': (1, 12), 'year': (1900, 2026), 'year_': (1, 99)}):
        """
        args:   int: date_obj
                label: str {'hours','minutes','seconds'}   
        return: bool
        """
        if date_obj != None:
            if label == 'hours':
                return 0 <= date_obj <= 23
            elif label == 'minutes':
                return 0 <= date_obj <= 59
            elif label == 'seconds':
                return 0 <= date_obj <= 59
            elif label == 'day':
                return range_dict.get('day')[0] <= date_obj <= range_dict.get('day')[1]
            elif label == 'month':
                return range_dict.get('month')[0] <= date_obj <= range_dict.get('month')[1]
            elif label == 'year':
                return range_dict.get('year')[0] <= date_obj <= range_dict.get('year')[1]
            elif label == 'year_':
                return range_dict.get('year_')[0] <= date_obj <= range_dict.get('year_')[1]
            else:
                return False
        else:
            return False

    @staticmethod
    def validate_time(time_str, is_json_obj=False):
        """
        args: (str :'adm: 20-jun-2023 23:00:54') or (tf object)
            is_json_obj: bool   
        return: list : [(text,bool)]
        """
        if is_json_obj:
            time_obj = time_str
        else:
            time_obj = json.loads(tf.run(time_str))
        # print(time_obj)
        return_ls = []
        if len(time_obj) > 0:
            for time_dict in time_obj:
                valid_datetime = False
                text, hr, min, sec, start, end = time_dict.get('text'), time_dict.get('hours'), time_dict.get(
                    'minutes'), time_dict.get('seconds'), time_dict.get('start'), time_dict.get('end')

                # condition_1:min & hr must be present
                if (hr != None and min != None):
                    # condition_2: (0=<hours<24| 0<=minutes<=59 | 0<seconds<=59|00:00:00-23:59:59)
                    valid_datetime = all([DatetimeExtractor.check_valid_range_datetime(hr, 'hours'),
                                          DatetimeExtractor.check_valid_range_datetime(min, 'minutes'),])
                elif (hr != None and sec != None):
                    # condition_2: (0=<hours<24| 0<=minutes<=59 | 0<seconds<=59|00:00:00-23:59:59)
                    valid_datetime = all([DatetimeExtractor.check_valid_range_datetime(hr, 'hours'),
                                          DatetimeExtractor.check_valid_range_datetime(sec, 'seconds')])
                else:
                    valid_datetime = False

                # check if detected time is part of date or any other text (e.g 11/02/2034 here 2034 is not time)
                start_, end_ = start - 1, end + 1
                start_end_time_txt = time_str[0 if start_ < 0 else start_:end_]
                other_text_count = len(re.findall(
                    r"[^0-9:\s\(\)ampmistedt-]", start_end_time_txt, re.I))
                if other_text_count > 0 and valid_datetime == True:
                    valid_datetime = False
                return_ls.append((text, valid_datetime, start, end))
            return return_ls
        else:
            return []

    @staticmethod
    def validate_date(date_str, is_json_obj=False, range_dict={'day': (1, 31), 'month': (1, 12), 'year': (1900, 2026), 'year_': (1, 99)}):
        """
        args:   date_str: str
                is_json_obj : bool (default: False)
                custome_range_dict: dict {'day':(),'month':(),'year':()}
        return: list: [(text,bool)]
        """
        if is_json_obj:
            date_obj = date_str
        else:
            date_obj = json.loads(dtf.run(date_str))

        return_ls = []
        if len(date_obj) > 0:
            for date_dict in date_obj:
                valid_date = False
                text, day, month, year, start, end = date_dict.get('text'), date_dict.get('day'), date_dict.get(
                    'month'), date_dict.get('year'), date_dict.get('start'), date_dict.get('end')
                # day/month/year must be present
                if (day != None and month != None and year != None):
                    valid_date = all([DatetimeExtractor.check_valid_range_datetime(day, 'day', range_dict),
                                      DatetimeExtractor.check_valid_range_datetime(
                        month, 'month', range_dict),
                        DatetimeExtractor.check_valid_range_datetime(year, 'year_', range_dict) if year < 100 else DatetimeExtractor.check_valid_range_datetime(year, 'year', range_dict)])
                # month/day must be present
                elif (day != None and month != None):
                    valid_date = all([DatetimeExtractor.check_valid_range_datetime(day, 'day', range_dict),
                                      DatetimeExtractor.check_valid_range_datetime(
                        month, 'month', range_dict),
                    ])
                elif year != None:
                    # condition_2: (0=<hours<24| 0<=minutes<=59 | 0<seconds<=59|00:00:00-23:59:59)
                    valid_date = all([DatetimeExtractor.check_valid_range_datetime(year, 'year', range_dict),
                                      ])
                else:
                    valid_date = False

                # check the length of 'Text', it should be greater than 3
                if len(text) <= 3 and valid_date == True:
                    valid_date = False
                return_ls.append((text, valid_date, start, end))
        else:
            return_ls = []
        return return_ls

    @staticmethod
    def get_date_tag(datetime_str, date_tags, date_start, lookbehind_idx=18, ):
        """ 
        args:   str: datetime_string
                list: date_tag_list
                int: start index of detected date
                int: lookbehind index (how much index you want to lookbehind from begining of date)
        return: str: associated tag string  
        """
        datetime_str_original = datetime_str
        processed_date_tags = [str(tag).lower().strip() for tag in date_tags]
        if date_start < lookbehind_idx:
            datetime_str = datetime_str[0:date_start]
        else:
            datetime_str = datetime_str[date_start-lookbehind_idx:date_start]

        datetime_str = " ".join(
            re.sub(r"[^a-zA-Z]", " ", datetime_str).strip().split()).lower()

        matcher = process.extractOne(
            datetime_str, processed_date_tags, scorer=fuzz.token_sort_ratio)
        detected_tags = [i for i in processed_date_tags if i in datetime_str_original]
        if detected_tags:
            return detected_tags[0]
        
        if matcher[1] >= 50:
            partial_ratio = fuzz.partial_ratio(
                datetime_str, matcher[0])
            avg_ratio = (matcher[1]+partial_ratio)/2
            if avg_ratio >= 62:
                return matcher[0].lower()

    @staticmethod
    def get_date_time_from_corpus(corpus, date_tags) -> list:
        extracted_date = DatetimeExtractor.validate_date(corpus)
        extracted_time = DatetimeExtractor.validate_time(corpus)

        # supress datetime (as it's detected False Positive as well so keep only True date/time)
        suppressed_date = DatetimeExtractor.suppress_datetime(extracted_date)
        suppressed_time = DatetimeExtractor.suppress_datetime(extracted_time)

        reversed_date_ls = reversed(suppressed_date)
        time_ls = suppressed_time.copy()
        # here we get final output
        final_date_list = []
        for detected_date in reversed_date_ls:
            date, _, date_start, date_end = detected_date
            tag = DatetimeExtractor.get_date_tag(corpus, date_tags, date_start)
            if len(time_ls) > 0:
                for detected_time in reversed(time_ls):
                    time, _, time_start, time_end = detected_time
                    if time_start > date_end:
                        final_date_list.append((date, time, tag))
                        time_ls.remove(detected_time)
            else:
                final_date_list.append((date, None, tag))
        return final_date_list

    @staticmethod
    def get_date_time_from_corpus_v2(corpus, date_tags):
        extracted_date = DatetimeExtractor.validate_date(corpus)

        # supress datetime (as it's detected False Positive as well so keep only True date/time)
        suppressed_date = DatetimeExtractor.suppress_datetime(extracted_date)
        final_date_list = []
        for detected_date in suppressed_date:
            date, _, date_start, date_end = detected_date
            tag = DatetimeExtractor.get_date_tag(corpus, date_tags, date_start)
            extracted_time = DatetimeExtractor.validate_time(
                corpus[date_end:date_end+15])
            suppressed_time = DatetimeExtractor.suppress_datetime(
                extracted_time)
            if len(suppressed_time) > 0:
                time, _, time_start, time_end = suppressed_time[-1]
                if time:
                    final_date_list.append((date, time, tag))
            else:
                final_date_list.append((date, None, tag))
        return final_date_list