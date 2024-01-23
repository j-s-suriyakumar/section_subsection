from datetime import datetime
from dateutil.parser import parse
from date_time_extractor import DatetimeExtractor
import pandas as pd
import random

class Postprocess:
    def __init__(self, adm_tag_list):
        self.adm_tag_list = adm_tag_list

    def date_parser(self, date_str, default_date=(2020, 1, 1)):
        yy, mm, dd = default_date
        default_date_ = datetime(yy, mm, dd)

        date_str_obj = DatetimeExtractor.get_date_time_from_corpus_v2(date_str,['None'])
        if date_str_obj and date_str_obj[0][1] and len(date_str_obj[0][1])<=4 and ':' not in date_str_obj[0][1]:
            # add extra 00 at the end of time
            date_str = f"{date_str_obj[0][0]} {date_str_obj[0][1]+'00'}"
        # parse the date
        try:
            parsed_date = parse(date_str, default=default_date_)
            day = parsed_date.day
            month = parsed_date.month
            year = parsed_date.year

            hour = parsed_date.hour
            minute = parsed_date.minute
            seconds = parsed_date.second
            return f"{month}-{day}-{year} {hour}:{minute}:{seconds}"
        except Exception as e:
            pass

    def get_adm_discharge_date(self, corpus):
        # if corpus is highly dense, in order to reduce the time to bring year of adm/dsch date
        # use the 25 % of corpus
        threshold = 100000
        if len(corpus) >= threshold:
            trimmed_corpus = corpus[:threshold]
        else:
            trimmed_corpus = corpus

        detected_date_tag = DatetimeExtractor.get_date_time_from_corpus_v2(
            trimmed_corpus, self.adm_tag_list)
        
        filter_date_list = [i for i in detected_date_tag if i[-1] and len(i[0]) > 4]
        random_date = [parse(i[0]).year for i in random.choices(
            filter_date_list, k=5)]
        adm_year = max(set(random_date), key=random_date.count)
        # for dt in detected_date_tag:
        #     if dt[2] and len(dt[0]) > 4:
        #         adm_year = parse(dt[0]).year
        #         break
        return adm_year
