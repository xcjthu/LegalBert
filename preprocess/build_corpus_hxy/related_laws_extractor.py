#!/usr/bin/python
# encoding:utf-8
"""
Extract all possible laws from the doc. This is to say, we need a further selection
according to the formatted laws in plm_data_process.py.
"""

import json
import sys
import traceback

sys.path.append('/data/private/hxy/PLM')

from cn2an import an2cn
from cn2an import cn2an
from build_corpus.const import *
from build_corpus.config import *


def _get_date(WB):
    """
    Extract date from WB. Due to the misprint in the doc, it can't be extracted
    totally right, but the accuracy is enough.

    :param WB:
    :return:
    """

    res = re.search(p_date, WB)
    if res is not None:
        try:
            year = ""
            for ch in res.group("year"):
                year += str(cn2num_dict[ch])
            year = int(year)
            if year < TOO_EARLY_CASE_YEAR:
                year = DATE_IGNORE_MARK

        except:
            print(traceback.print_exc())
            year = DATE_IGNORE_MARK

        try:
            if res.group("month") == '元':
                month = 1
            else:
                month = cn2an(res.group("month"), mode='normal')
        except:
            print(traceback.print_exc())
            month = DATE_IGNORE_MARK

        try:
            day = cn2an(res.group("day"), mode='normal')
        except:
            print(traceback.print_exc())
            day = DATE_IGNORE_MARK
        date = {"year": year, "month": month, "day": day}

        return date
    else:
        print("empty WB:", WB)
        return {"year": 0, "month": 0, "day": 0}


def _convert_an_to_cn(ch):
    if ch.isdigit():  # 是数字的话
        ch = an2cn(ch, 'low')
    return ch


def _remove_repeated_terms(related_terms_list):
    # print("************")
    # print(related_terms_list)

    tmp_terms = []
    for related_terms in related_terms_list:

        for t1 in related_terms:
            flag = False
            for t2 in tmp_terms:
                if t1['term'] == t2['term'] and t1['kuan'] == t2['kuan'] and t1['xiang'] == t2['xiang']:
                    flag = True
                    break
            if not flag:
                tmp_terms.append(t1)

    # print("777777777777777")
    # print(tmp_terms)

    return tmp_terms


def _combine_related_laws(related_laws_list):
    # print(related_laws_list)

    tmp_related_laws = {}

    for related_laws in related_laws_list:
        for law_name in related_laws.keys():
            if law_name not in tmp_related_laws.keys():
                tmp_related_laws[law_name] = _remove_repeated_terms([related_laws[law_name]])
            else:
                tmp_related_laws[law_name] = _remove_repeated_terms(
                    [related_laws[law_name], tmp_related_laws[law_name]])
    # print("=========================")
    # print(tmp_related_laws)

    return tmp_related_laws


def _convert_doc_bookmark(doc):
    """
    Standardize the bookmarks in all the part of doc through the method _convert_bookmark.

    :param doc:
    :return:
    """


    if 'SS' in doc.keys():
        doc['SS'] = _convert_bookmark(doc['SS'])

    if 'LY' in doc.keys():
        doc['LY'] = _convert_bookmark(doc['LY'])

    if 'JG' in doc.keys():
        doc['JG'] = _convert_bookmark(doc['JG'])

    if 'FZ' in doc.keys():
        doc['FZ'] = _convert_bookmark(doc['FZ'])


def _convert_bookmark(lawname):
    """
    The chinese bookmarks have lots of different types which will
    affect the recognition of law names, so we standardize them here.

    :param lawname:
    :return:
    """

    lawname = lawname.replace("＜", "《")
    lawname = lawname.replace("＞", "》")
    lawname = lawname.replace('﹤', '《')
    lawname = lawname.replace('﹥', '》')
    lawname = lawname.replace('〈', "《")
    lawname = lawname.replace('〉', '》')

    lawname = lawname.replace("（", "(")
    lawname = lawname.replace("）", ")")

    return lawname


def _extract_from_para(para):
    """
    Extract all possible related terms, according to bookmarks and keywords like "条","款".

    :param para:
    :return:
    """



    if para is None or len(para) == 0:  # no LY
        return {}

    related_laws = {}

    para = para.split("《")

    for para_part in para:
        it = re.finditer(p_lawname_with_right_bookmark, para_part)
        for i_lawname in it:
            ln = i_lawname.group("lawname")

            ln = _convert_bookmark(ln)

            term_nums = i_lawname.group("term_nums")[:60]  # only get 60 character

            # related term for one law
            related_term = []
            it_termnum1 = re.finditer(p_termnum1, term_nums)

            # 如果termnum1匹配不到 换termnum2
            for i_termnum1 in it_termnum1:
                one_term = {"term": TERM_IGNORE_MARK, "kuan": TERM_IGNORE_MARK, "xiang": TERM_IGNORE_MARK}

                one_term["term"] = _convert_an_to_cn(
                    i_termnum1.group('term_num1') if i_termnum1.group('term_num1') is not None else i_termnum1.group(
                        "term_num2"))

                if i_termnum1.group("kuan") is not None:
                    one_term["kuan"] = i_termnum1.group("kuan")
                if i_termnum1.group("xiang") is not None:
                    if one_term['kuan'] == TERM_IGNORE_MARK:
                        one_term["kuan"] = '一'
                    one_term["xiang"] = i_termnum1.group("xiang")

                related_term.append(one_term)

            # else:
            #     it_termnum2 = re.finditer(p_termnum2, term_nums)
            #
            #     for i_termnum2 in it_termnum2:
            #         one_term = {"term": IGNORE_MARK, "kuan": IGNORE_MARK, "xiang": IGNORE_MARK}
            #
            #         one_term["term"] = convert_an_to_cn(i_termnum2.group('term_num'))
            #
            #         if i_termnum2.group("kuan") is not None:
            #             one_term["kuan"] = i_termnum2.group("kuan")
            #             if i_termnum2.group("xiang") is not None:
            #                 one_term["xiang"] = i_termnum2.group("xiang")
            #
            #         related_term.append(one_term)

            if len(related_term) == 0:
                continue

            if ln in related_laws.keys():
                related_laws[ln].extend(related_term)
            else:
                related_laws[ln] = related_term

    return related_laws


def _clean_SS(SS: str, all_law_list):
    """

    :param SS:
    :param all_law_list:
    :return:
    """

    # FIXME(huxueyu@buaa.edu.cn): This method generally has the same purpose of removing law info in the doc
    #  to prevent information leakage, with the method who has the same name _clean_SS in ac_data_process.py,
    #  but has a different implementation. Actually, there is a conflict here and would better be fixed.


    SS = _convert_bookmark(SS)

    for name in all_law_list:
        SS = SS.replace("《" + name + "》", "《》")

    return SS


def extract_related_laws(start=0, end=TOTAL_MS_NUM):
    """
    Extract the whole doc datasets and the output is like XXX.pb2_plm.json.

    :param start:
    :param end:
    :return:
    """


    all_shorten_formatted_laws_name = json.load(open(all_shorten_formatted_laws_name_path))
    all_formatted_laws_name = json.load(open(all_formatted_laws_name_path))

    all_ir_list = all_shorten_formatted_laws_name + all_formatted_laws_name

    for fp_num in range(start, end):
        try:
            with open(ms_raw_dir + '/' + str(fp_num) + ".pb2.json", "r", encoding='utf-8') as f:
                docs = json.load(f)
            print("example docs: ", fp_num)

        except Exception as e:
            print(e)
            print("error:" + str(fp_num) + ".pb2.json")
            continue

        for case_num, doc in enumerate(docs):

            LY_related_laws = _extract_from_para(doc.get("LY"))
            JG_related_laws = _extract_from_para(doc.get("JG"))
            FZ_related_laws = _extract_from_para(doc.get("FZ"))

            # remove repeated term
            related_laws = _combine_related_laws([LY_related_laws, JG_related_laws, FZ_related_laws])

            doc.update({"related_laws": related_laws})

            # get date from WB
            doc.update({"date": _get_date(doc.get("WB"))})

            # clean SS
            if 'SS' in doc.keys():
                doc['SS'] = _clean_SS(doc['SS'], all_ir_list)
            # print(doc['LY'])
            # print(LY_related_laws)
        # if fp_num > 10:
        #     exit(0)
        with open(ms_plm_dir + '/' + str(fp_num) + ".pb2_plm.json", "w", encoding='utf-8') as f:
            json.dump(docs, f, ensure_ascii=False)





if __name__ == '__main__':
    extract_related_laws()
