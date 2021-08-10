# encoding:utf-8
"""
Match the related laws extracted by related_laws_extractor.py with the formatted laws.
"""


import sys

sys.path.append('/data/private/hxy/PLM')

import json
import random
from random import shuffle
import traceback

from transformers import BertTokenizer

from build_corpus.const import *
from build_corpus.config import *
from build_corpus.helper import read_formatted_laws


def _compare_date(date1, date2):
    year1, month1, day1 = date1['year'], date1['month'], date1['day']
    year2, month2, day2 = date2['year'], date2['month'], date2['day']

    if year1 > year2:
        return 1
    elif year1 < year2:
        return -1
    elif year1 == year2:
        if month1 > month2:
            return 1
        elif month1 < month2:
            return -1
        elif month1 == month2:
            if day1 > day2:
                return 1
            elif day1 < day2:
                return -1
            elif day1 == day2:
                return 0


def _max_date_law(laws):
    """
    Return the law which was lastly released from laws.

    :param laws:
    :return:
    """


    max_date = {'year': -100, 'month': -100, 'day': -100}

    max_date_law = None

    for law in laws:
        if _compare_date(law['date'], max_date) > 0:
            max_date = law['date']
            max_date_law = law
    return max_date_law


def _is_valid_doc(doc):
    """
    Judge if the doc has a valid date, because some dates of case are not extracted rightly.

    :param doc:
    :return:
    """


    if 'SS' not in doc.keys() or len(doc['SS']) == 0:
        return False

    if 'date' not in doc.keys() or (
            doc['date']['year'] == DATE_IGNORE_MARK and doc['date']['month'] == DATE_IGNORE_MARK and doc['date'][
        'day'] == DATE_IGNORE_MARK):
        return False

    if 'related_laws' not in doc.keys():
        return False

    return True


def _is_legal_term_kuan_xiang(content, term_kuan_xiang):
    """
    Judge if the given term_kuan_xiang is legal, since some of the extracted term_kuan_xiang
    can't be found.

    :param content:
    :param term_kuan_xiang:
    :return:
    """



    pos_text = None

    term = term_kuan_xiang['term']
    kuan = term_kuan_xiang['kuan']
    xiang = term_kuan_xiang['xiang']

    for chapter in content:
        if term in chapter.keys():
            if type(chapter[term]) == str:
                pos_text = chapter[term]

            else:
                if kuan == TERM_IGNORE_MARK:
                    pos_text = "".join(chapter[term].values())
                    break
                else:

                    if kuan in chapter[term].keys():
                        pos_text = chapter[term][kuan]
                        break

    if pos_text is not None:
        return True
    else:
        return False


def _transform_law_name(law_name):
    try:
        if law_name.startswith("中华人民共和") and law_name[6] != '国':
            law_name = "中国人民共和国" + law_name[6:]
        if law_name.startswith("中国人民共和国"):
            law_name = '中华人民共和国' + law_name[7:]

        if law_name in ir_dict.keys():
            law_name = ir_dict[law_name]
    except Exception as e:
        traceback.print_exc()

    return law_name


def _transform_sfjs_name(sfjs_name):
    try:
        sfjs_name = sfjs_name.replace("（", "(")
        sfjs_name = sfjs_name.replace("）", ")")

        if sfjs_name in ir_dict.keys():
            sfjs_name = ir_dict[sfjs_name]
    except Exception as e:
        traceback.print_exc()

    return sfjs_name


def get_pos_neg(content, term_kuan_xiang):
    """
    Return the text of term according to the term_kuan_xiang, also return
    the negative samples, which are the terms in the same chapter of the positive one.

    :param content:
    :param term_kuan_xiang:
    :return:
    """


    pos_text = None
    neg_text_list = None

    term = term_kuan_xiang['term']
    kuan = term_kuan_xiang['kuan']
    xiang = term_kuan_xiang['xiang']

    for chapter in content:

        if term in chapter.keys():
            if type(chapter[term]) == str:
                pos_text = chapter[term]
                neg_text_list = chapter.copy()
                del neg_text_list[term]
                neg_text_list = list(neg_text_list.values())
            else:
                if kuan == TERM_IGNORE_MARK:
                    pos_text = "".join(chapter[term].values())
                    other_terms = chapter.copy()
                    del other_terms[term]

                    for k in other_terms.keys():
                        t = other_terms[k]
                        if neg_text_list is None:
                            neg_text_list = ["".join(t.values())]
                        else:
                            neg_text_list.append("".join(t.values()))

                    break
                else:

                    if kuan in chapter[term].keys():
                        pos_text = chapter[term][kuan]
                        other_terms = chapter.copy()
                        del other_terms[term]

                        for k in other_terms.keys():
                            t = other_terms[k]
                            if neg_text_list is None:
                                neg_text_list = ["".join(t.values())]
                            else:
                                neg_text_list.append("".join(t.values()))

                        break

    return pos_text, neg_text_list


def search_one_law_with_date(law_name, case_date, formatted_laws):
    """
    Search one law according to its name and the date of case, which means it will
    return the latest law but not later then the date of case .

    :param law_name:
    :param case_date:
    :param formatted_laws:
    :return:
    """


    possible_laws = []

    law_name = _transform_law_name(law_name)

    for f in formatted_laws:
        if law_name == f['title'] or (len(law_name) >= 10 and law_name in f['title'] and '人民代表大会' not in f['title']):
            if _compare_date(case_date, f['date']) >= 0:
                possible_laws.append(f)

    if len(possible_laws) == 0:
        return None
    elif len(possible_laws) == 1:
        return possible_laws[0]
    else:
        return _max_date_law(possible_laws)


def search_all_laws(law_name, all_formatted_laws, case_date=FAKE_DATE):
    """
    Search one law according to its name and the date of case, in the order of
    fl, sfjs, xzfg and dfxfg.

    :param law_name:
    :param all_formatted_laws:
    :param case_date:
    :return:
    """

    formatted_fl = all_formatted_laws.get('formatted_fl')
    formatted_sfjs = all_formatted_laws.get('formatted_sfjs')
    formatted_xzfg = all_formatted_laws.get('xzfg')
    formatted_dfxfg = all_formatted_laws.get('dfxfg')

    target_law = None
    if target_law is None and formatted_fl is not None:
        target_law = search_one_law_with_date(law_name, case_date, formatted_fl)
    if target_law is None and formatted_sfjs is not None:
        target_law = search_one_law_with_date(law_name, case_date, formatted_sfjs)

    if target_law is None and formatted_xzfg is not None:
        for formatted_law in formatted_xzfg:
            if _transform_sfjs_name(law_name) == formatted_law['title'] or (
                    len(law_name) >= 10 and law_name in formatted_law['title']):
                target_law = formatted_law
                break

    if target_law is None and formatted_dfxfg is not None:
        for formatted_law in formatted_dfxfg:
            if _transform_sfjs_name(law_name) == formatted_law['title'] or (
                    len(law_name) >= 10 and law_name in formatted_law['title']):
                target_law = formatted_law
                break

    return target_law


def get_term_ids_from_one_doc(doc, all_formatted_laws):
    """
    Search all the related laws of one doc through the method search_all_laws.

    :param doc:
    :param all_formatted_laws:
    :return:
    """


    related_term_ids = []

    case_date = doc['date']
    related_laws = doc['related_laws']

    for rl_key in related_laws.keys():

        target_law = search_all_laws(rl_key, all_formatted_laws, case_date)
        if target_law is None:
            # print("there is no info: ", rl_key)
            continue

        law_id = target_law['id_']
        content = target_law['content']

        for term_kuan_xiang in related_laws[rl_key]:
            if _is_legal_term_kuan_xiang(content, term_kuan_xiang):
                related_term_ids.append((law_id, term_kuan_xiang['term'], term_kuan_xiang['kuan']))

    return related_term_ids


def update_all_term_ids(start=0, end=TOTAL_MS_NUM):

    # FIXME(huxueyu@buaa.edu.cn): Use this method to update the result generated
    #  by related_laws_extractor.py. But actually these two steps can be merged to
    #  improve efficiency.



    all_formatted_laws = read_formatted_laws(return_dict=True)

    for fp_num in range(start, end):
        print("ms docs: ", fp_num)
        with open(ms_plm_dir + '/' + str(fp_num) + ".pb2_plm.json", "r", encoding='utf-8') as f:
            docs = json.load(f)

        doc_with_term_ids = []
        for doc in docs:
            if not _is_valid_doc(doc):
                continue

            term_ids = get_term_ids_from_one_doc(doc, all_formatted_laws)

            if len(term_ids) > 0:
                doc.update({'term_ids': term_ids})
                doc_with_term_ids.append(doc)

        with open(ms_plm_ids_dir + '/' + str(fp_num) + ".pb2_plm_ids.json", 'w', encoding='utf-8') as f:
            json.dump(doc_with_term_ids, f, ensure_ascii=False)


def create_examples(start=0, end=TOTAL_XS_NUM):
    example_id = 0
    pos_num = 0
    neg_num = 0

    with open(formatted_fl_path, 'r', encoding='utf-8') as f:
        formatted_fl = json.load(f)

    with open('/data/private/hxy/data/all_national_laws/formatted_laws/formatted_sfjs_date.json', 'r',
              encoding='utf-8') as f:
        formatted_sfjs = json.load(f)

    with open(formatted_dfxfg_path, 'r', encoding='utf-8') as f:
        formatted_dfxfg = json.load(f)

    for fp_num in range(start, end):
        print("ms docs: ", fp_num)
        with open(ms_plm_dir + '/' + str(fp_num) + ".pb2_plm.json", "r", encoding='utf-8') as f:
            docs = json.load(f)

        examples = []

        for doc in docs:
            if 'SS' not in doc.keys() or len(doc['SS']) == 0:
                continue

            SS = doc['SS']
            date = doc['date']

            date['year'] = 10000 if date['year'] == 0 else date['year']
            date['month'] = 10000 if date['month'] == 0 else date['month']
            date['day'] = 10000 if date['day'] == 0 else date['day']

            if 'related_laws' in doc.keys():
                related_laws = doc['related_laws']
            else:
                continue

            for rl_key in related_laws.keys():
                target_law = search_one_law_with_date(rl_key, date, formatted_fl)
                if target_law is None:
                    target_law = search_one_law_with_date(rl_key, date, formatted_sfjs)

                if target_law is None:
                    for formatted_law in formatted_dfxfg:
                        if _transform_sfjs_name(rl_key) == formatted_law['title'] or (
                                len(rl_key) >= 10 and rl_key in formatted_law['title']):
                            target_law = formatted_law
                            break

                if target_law is None:
                    # print("there is no info: ", rl_key)
                    continue

                # positive 0 negative 1

                content = target_law['content']
                for term_kuan_xiang in related_laws[rl_key]:
                    pos_text, neg_text_list = get_pos_neg(content, term_kuan_xiang)
                    if pos_text is None or neg_text_list is None:

                        if pos_text is None:
                            print("no such term", rl_key, term_kuan_xiang)

                        continue

                    # print(rl_key, term_kuan_xiang)
                    # print(pos_text)

                    example = {'id_': example_id, 'SS': SS, 'law': pos_text, 'label': POSITIVE_LABEL}
                    example_id += 1
                    pos_num += 1
                    examples.append(example)

                    if len(neg_text_list) != 0:
                        example = {'id_': example_id, 'SS': SS, 'law': random.choice(neg_text_list),
                                   'label': NEGATIVE_LABEL}
                        example_id += 1
                        neg_num += 1
                        examples.append(example)

        with open(plm_examples_dir + '/' + str(fp_num) + ".pb2_plm_examples.json", 'w', encoding='utf-8') as f:
            json.dump(examples, f, ensure_ascii=False)

    print(pos_num)
    print(neg_num)
    print("total: ", example_id)


def _get_both_sep_index(input_ids, tokenizer):
    sep_index_count = 0
    sep_index_one = -1
    sep_index_two = -1

    for i in range(len(input_ids)):
        if input_ids[i] == tokenizer.sep_token_id:
            if sep_index_one == -1:
                sep_index_one = i
                sep_index_count += 1
            else:
                sep_index_two = i
                sep_index_count += 1

    assert sep_index_count == 2
    return sep_index_one, sep_index_two


def convert_example_to_feature(model_path, start, end):
    tokenizer = BertTokenizer.from_pretrained(model_path)

    for fp_num in range(start, end):

        features = []

        try:

            with open(plm_examples_dir + '/' + str(fp_num) + ".pb2_plm_examples.json", "r", encoding='utf-8') as f:
                examples = json.load(f)

            print("example docs: ", fp_num)

        except Exception as e:
            print(e)
            print("error:" + str(fp_num) + ".pb2_plm_examples.json")
            continue

        shuffle(examples)

        # total = len(examples)
        # too_long_num = 0

        for step, e in enumerate(examples):

            id_ = e['id_']
            SS = e['SS']
            law = e['law']
            next_label = e['label']

            if len(SS) + len(law) > MAX_LENGTH - 3:
                law = law[:MAX_LAW_LENGTH]

            if len(SS) + len(law) > MAX_LENGTH - 3:
                SS = SS[:MAX_LENGTH - 3 - len(law)]
                assert len(SS) + len(law) == (MAX_LENGTH - 3)

            encoding_dict = tokenizer.encode_plus(text=law,
                                                  text_pair=SS,
                                                  add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]'
                                                  truncation=True,
                                                  max_length=MAX_LENGTH,
                                                  padding='max_length',
                                                  return_attention_mask=True,  # 返回 attn. masks.
                                                  )

            input_ids = encoding_dict["input_ids"]
            token_type_ids = encoding_dict['token_type_ids']
            attention_mask = encoding_dict['attention_mask']

            cls_index = 0
            sep_index_one, sep_index_two = _get_both_sep_index(input_ids, tokenizer)

            assert len(input_ids) == MAX_LENGTH
            assert len(token_type_ids) == MAX_LENGTH
            assert len(attention_mask) == MAX_LENGTH
            assert input_ids[cls_index] == tokenizer.cls_token_id
            assert input_ids[sep_index_one] == tokenizer.sep_token_id
            assert input_ids[sep_index_two] == tokenizer.sep_token_id

            mask_index = random.sample(
                list(range(cls_index + 1, sep_index_one)) + list(range(sep_index_one + 1, sep_index_two)),
                int((sep_index_two - sep_index_one - 2) * 0.15))

            mlm_label = [MLM_IGNORE_ID] * MAX_LENGTH
            for i in mask_index:
                mlm_label[i] = input_ids[i]
                if random.random() < 0.8:
                    input_ids[i] = tokenizer.mask_token_id
                elif random.random() < 0.5:
                    rand_index = random.randint(0, tokenizer.vocab_size - 1)
                    while rand_index in tokenizer.all_special_ids:
                        rand_index = random.randint(0, tokenizer.vocab_size - 1)
                    input_ids[i] = rand_index

                else:
                    pass

            feature = {'id_': id_, 'input_ids': input_ids, 'token_type_ids': token_type_ids,
                       'attention_mask': attention_mask, 'mlm_label': mlm_label, 'next_label': next_label}

            features.append(feature)

        with open(plm_features_dir + '/' + str(fp_num) + ".pb2_plm_features.json", "w", encoding='utf-8') as f:
            json.dump(features, f, ensure_ascii=False)


if __name__ == '__main__':
    illegal_term_num = 0
    total_term_num = 0
    update_all_term_ids(0, TOTAL_MS_NUM)
    # print(illegal_term_num)
    # print(total_term_num)
    # print(illegal_term_num / total_term_num)
