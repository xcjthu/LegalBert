# encoding:utf-8
"""
Build dataset from xs docs for the crime prediction task.
"""
import sys

sys.path.append('/mnt/datadisk0/hxy/PLM')

import os
import random

from build_corpus.plm_data_process import *
from build_corpus.config import *
from build_corpus.const import *
from build_corpus.helper import read_formatted_laws
from build_corpus.retrieval import find_term_by_index


def _is_right_doc(doc):
    TITLE = doc.get("TITLE")
    date = doc.get("date")
    ajay_id = doc.get("AJAY")
    related_laws = doc.get("related_laws")
    JG = doc.get("JG")

    # 只选一审刑事判决书
    if '一审刑事判决书' not in TITLE or 'SS' not in doc.keys() or len(doc['SS']) < 10:
        return False

    if date['year'] == DATE_IGNORE_MARK or date['year'] < 1990:
        return False

    if len(ajay_id) < 1:
        return False

    if 'related_laws' not in doc.keys() or len(related_laws) == 0:
        return False

    if 'JG' not in doc.keys() or len(JG) == 0:
        return False

    return True


def _clean_SS(SS, all_law_set):
    SS = SS.replace(" ", "").replace("\t", "")

    new_SS = []

    split_SS = SS.split("。")

    for i in split_SS:
        res1 = re.search(p_empty_bookmark, i)
        res2 = re.search(p_sfjs_bookmark, i)
        if res1 is not None or res2 is not None:
            pass

        else:
            res3 = re.search(p_bookmark, i)

            if res3 is not None and res3.group('lawname') in all_law_set:
                pass
            else:
                new_SS.append(i)
    new_SS = '。'.join(new_SS)

    return new_SS


def get_number_from_string(s):
    for x in s:
        if not (x in num_list):
            print(s)
            raise ValueError

    value = 0
    try:
        value = int(s)
    except ValueError:
        nowbase = 1
        addnew = True
        for a in range(len(s) - 1, -1, -1):
            if s[a] == u'十':
                if nowbase >= 10000:
                    nowbase = 100000
                else:
                    nowbase = 10
                addnew = False
            elif s[a] == u'百':
                if nowbase >= 10000:
                    nowbase = 1000000
                else:
                    nowbase = 100
                addnew = False
            elif s[a] == u'千':
                if nowbase >= 10000:
                    nowbase = 10000000
                else:
                    nowbase = 1000
                addnew = False
            elif s[a] == u'万':
                nowbase = 10000
                addnew = False
            else:
                value = value + nowbase * num_list[s[a]]
                nowbase = nowbase * 10
                addnew = True

        if not (addnew):
            value += nowbase

    return value


def parse_money(doc):
    if "JG" not in doc.keys():
        return []

    JG = doc['JG']

    result_list = []

    rex = re.compile(u"人民币([" + num_str + "]*)元")
    result = rex.finditer(JG)

    for x in result:
        datax = get_number_from_string(x.group(1))
        result_list.append(datax)
        # print(x.group(1), datax)

    return result_list


def parse_date_with_year_and_month_begin_from(s, begin, delta):
    # erf = open("error.log", "a")
    pos = begin + delta
    num1 = 0
    while s[pos] in num_list:
        if s[pos] == u"十":
            if num1 == 0:
                num1 = 1
            num1 *= 10
        elif s[pos] == u"百" or s[pos] == u"千" or s[pos] == u"万":
            # print("0 " + s[begin - 10:pos + 20], file=erf)
            return None
        else:
            num1 = num1 + num_list[s[pos]]

        pos += 1

    num = 0
    if s[pos] == u"年":
        num2 = 0
        pos += 1
        if s[pos] == u"又":
            pos += 1
        while s[pos] in num_list:
            if s[pos] == u"十":
                if num2 == 0:
                    num2 = 1
                num2 *= 10
            elif s[pos] == u"百" or s[pos] == u"千" or s[pos] == u"万":
                # print("1 " + s[begin - 10:pos + 20], file=erf)
                return None
            else:
                num2 = num2 + num_list[s[pos]]

            pos += 1
        if s[pos] == u"个":
            pos += 1
        if num2 != 0 and s[pos] != u"月":
            # print("2 " + s[begin - 10:pos + 20], file=erf)
            return None
        num = num1 * 12 + num2
    else:
        if s[pos] == u"个":
            pos += 1
        if s[pos] != u"月":
            # print("3 " + s[begin - 10:pos + 20], file=erf)
            return None
        else:
            num = num1

    pos += 1
    # print(num,s[x.start():pos])
    return num


def parse_term_of_imprisonment(doc):
    result = {}

    JG = doc['JG'].replace("b", '')

    # 有期徒刑
    youqi_arr = []
    pattern = re.compile(u"有期徒刑")
    for x in pattern.finditer(JG):
        pos = x.start()
        num = parse_date_with_year_and_month_begin_from(JG, pos, len(u"有期徒刑"))
        if num is not None:
            youqi_arr.append(num)

    # 拘役
    juyi_arr = []
    pattern = re.compile(u"拘役")
    for x in pattern.finditer(JG):
        pos = x.start()
        num = parse_date_with_year_and_month_begin_from(JG, pos, len(u"拘役"))
        if num is not None:
            juyi_arr.append(num)

    # 管制
    guanzhi_arr = []
    pattern = re.compile(u"管制")
    for x in pattern.finditer(JG):
        pos = x.start()
        num = parse_date_with_year_and_month_begin_from(JG, pos, len(u"管制"))
        if num is not None:
            guanzhi_arr.append(num)

    # 无期
    forever = False
    if JG.count("无期徒刑") != 0:
        forever = True

    # 死刑
    dead = False
    if JG.count("死刑") != 0 and JG.count("缓期"):
        dead = True

    result["youqi"] = youqi_arr
    result["juyi"] = juyi_arr
    result["guanzhi"] = guanzhi_arr
    result["wuqi"] = forever
    result["sixing"] = dead

    # print(doc)

    return result


def _is_right_parse(parse_result):
    if len(parse_result) == 0 or (
            len(parse_result['juyi']) + len(parse_result['guanzhi']) + len(parse_result['youqi']) +
            parse_result['wuqi'] + parse_result['sixing']) != 1:
        return False

    if (len(parse_result['juyi']) > 0 and parse_result['juyi'][0] > 12) or (
            len(parse_result['guanzhi']) > 0 and parse_result['guanzhi'][0] > 36) or (
            len(parse_result['youqi']) > 0 and parse_result['youqi'][0] > 240):
        return False
    return True


def _format_parse(parse_result):
    if len(parse_result['youqi']) == 1:
        term_of_imprisonment = {
            'guanzhi': 0,
            'juyi': 0,
            'imprisonment': parse_result['youqi'][0],
            'death_penalty': parse_result['sixing'],
            'life_imprisonment': parse_result['wuqi']}

    elif len(parse_result['guanzhi']) == 1:
        term_of_imprisonment = {
            'guanzhi': parse_result['guanzhi'][0],
            'juyi': 0,
            'imprisonment': 0,
            'death_penalty': parse_result['sixing'],
            'life_imprisonment': parse_result['wuqi']}

    elif len(parse_result['juyi']) == 1:
        term_of_imprisonment = {
            'guanzhi': 0,
            'juyi': parse_result['juyi'][0],
            'imprisonment': 0,
            'death_penalty': parse_result['sixing'],
            'life_imprisonment': parse_result['wuqi']}

    else:
        term_of_imprisonment = {
            'guanzhi': 0,
            'juyi': 0,
            'imprisonment': 0,
            'death_penalty': parse_result['sixing'],
            'life_imprisonment': parse_result['wuqi']}

    return term_of_imprisonment


def _is_same_law(law1, law2):
    law1 = law1.strip()
    law2 = law2.strip()
    if law1 == law2:
        return True

    if len(law1) > 30 and law1[10:25] in law2:
        return True

    return False


def _convert_kuan_to_term(input):
    if type(input) == list:
        output = []
        for i in input:
            tmp = [i[0], i[1], TERM_IGNORE_MARK]
            if tmp not in output:
                output.append(tmp)
    else:
        raise NotImplementedError

    return output


def extract_JP(start=0, end=SMALL_JP_FILE_NUM):
    # read
    id2ay = json.load(open(mapping_path))['id2ay']
    all_shorten_formatted_laws_name = json.load(open(all_shorten_formatted_laws_name_path))
    all_formatted_laws_name = json.load(open(all_formatted_laws_name_path))
    all_lawname_list = all_shorten_formatted_laws_name + all_formatted_laws_name
    all_lawname_set = set(all_lawname_list)

    formatted_fl, formatted_sfjs, formatted_xzfg, formatted_dfxfg = read_formatted_laws()
    all_formatted_laws = formatted_fl + formatted_sfjs + formatted_xzfg + formatted_dfxfg

    # combine same laws
    already_exist_laws = set({})
    already_exist_laws_content = {}
    lookup_dict = {}

    # statistics
    total_lengths = []

    for fp_num in range(start, end):
        print("xs docs: ", fp_num)
        with open(xs_plm_ids_dir + '/' + str(fp_num) + ".pb2_plm_ids.json", encoding='utf-8') as f:
            docs = json.load(f)
        docs_with_jp = []
        for doc in docs:
            if not _is_right_doc(doc):
                continue

            # get AYAY
            ajay_id = doc.get("AJAY")
            doc.update({"crime": ajay_id})
            for a in ajay_id:
                doc['SS'] = doc['SS'].replace(id2ay[a], "")
            doc['SS'] = _clean_SS(doc['SS'], all_lawname_set)

            # get related laws
            related_laws = _convert_kuan_to_term(doc.get("term_ids"))
            del doc['term_ids']

            # get term of imprisonment
            try:
                parse_result = parse_term_of_imprisonment(doc)
            except:
                traceback.print_exc()
                continue
            if not _is_right_parse(parse_result):
                continue
            term_of_imprisonment = _format_parse(parse_result)
            doc['term_of_imprisonment'] = term_of_imprisonment

            # clean same law
            new_related_laws = []
            for new_term in related_laws:
                if tuple(new_term) in lookup_dict.keys():
                    new_related_laws.append(lookup_dict[tuple(new_term)])
                elif tuple(new_term) in already_exist_laws:
                    new_related_laws.append(new_term)
                else:
                    new_term_content = find_term_by_index(tuple(new_term), all_formatted_laws)[0]
                    for already_term in already_exist_laws:

                        if _is_same_law(already_exist_laws_content[already_term], new_term_content):
                            lookup_dict[tuple(new_term)] = list(already_term)
                            new_related_laws.append(list(already_term))
                            break
                    else:
                        already_exist_laws.add(tuple(new_term))
                        already_exist_laws_content[tuple(new_term)] = \
                            find_term_by_index(tuple(new_term), all_formatted_laws)[0]
                        new_related_laws.append(new_term)

            doc['related_laws'] = new_related_laws

            docs_with_jp.append(doc)

            # statistics
            total_lengths.append(len(doc['SS']))

        with open(xs_jp_dir + '/' + str(fp_num) + ".pb2_jp.json", "w", encoding='utf-8') as f:
            json.dump(docs_with_jp, f, ensure_ascii=False)

    print(sum(total_lengths) / len(total_lengths))


def combine_same_term():
    formatted_fl, formatted_sfjs, formatted_xzfg, formatted_dfxfg = read_formatted_laws()
    all_formatted_laws = formatted_fl + formatted_sfjs + formatted_xzfg + formatted_dfxfg

    already_exist_laws = set({})
    lookup_dict = {}

    for fp_num in range(SMALL_JP_FILE_NUM):
        print("xs_jp docs: ", fp_num)
        with open(xs_small_jp_dir + '/' + str(fp_num) + ".pb2_jp.json", encoding='utf-8') as f:
            docs = json.load(f)

        for doc in docs:
            related_laws = doc['related_laws']
            new_related_laws = []
            for new_term in related_laws:
                if tuple(new_term) in lookup_dict.keys():
                    new_related_laws.append(lookup_dict[tuple(new_term)])
                elif tuple(new_term) in already_exist_laws:
                    new_related_laws.append(new_term)
                else:
                    new_term_content = find_term_by_index(tuple(new_term), all_formatted_laws)[0]
                    for already_term in already_exist_laws:
                        if find_term_by_index(already_term, all_formatted_laws)[0] == new_term_content:
                            # print('new', tuple(new_term))
                            # print('exists', already_term)
                            lookup_dict[tuple(new_term)] = list(already_term)
                            new_related_laws.append(list(already_term))
                            break
                    else:
                        already_exist_laws.add(tuple(new_term))
                        new_related_laws.append(new_term)

            new_related_laws = _convert_kuan_to_term(new_related_laws)
            doc['related_laws'] = new_related_laws

        with open(xs_jp_comb_dir + '/' + str(fp_num) + ".pb2_jp_comb.json", "w", encoding='utf-8') as f:
            json.dump(docs, f, ensure_ascii=False)
    print(len(already_exist_laws))
    print(len(lookup_dict))


def create_large_corpus():
    os.makedirs(xs_large_jp_dir, exist_ok=True)

    with open(original_jp_info_path, encoding='utf-8') as f:
        info = json.load(f)

    crime_ids = {k[0] for k in info['crime_id_dis']}
    laws = {tuple(k[0]) for k in info['term_dis']}

    # crime_ids = {crime_id: 0 for crime_id in crime_ids.keys()}
    # term_dis = {tuple(law[0]): 0 for law in term_dis}

    for fp_num in range(0, LARGE_JP_FILE_NUM):
        print("xs docs: ", fp_num)
        with open(xs_jp_dir + '/' + str(fp_num) + ".pb2_jp.json", "r", encoding='utf-8') as f:
            docs = json.load(f)

        doc_large = []
        for doc in docs:

            # maybe multilabel
            crime_id = doc['crime']
            flag = False
            for c in crime_id:
                if c not in crime_ids:
                    flag = True
                    break
            if flag is True:
                continue

            related_laws = doc['related_laws']
            related_laws = [id_ for id_ in related_laws if tuple(id_) in laws]

            if len(related_laws) == 0:
                continue

            doc['related_laws'] = related_laws
            doc_large.append(doc)

        with open(xs_large_jp_dir + '/' + str(fp_num) + ".pb2_large_jp.json", "w", encoding='utf-8') as f:
            json.dump(doc_large, f, ensure_ascii=False)


def create_small_corpus():
    os.makedirs(xs_small_jp_dir, exist_ok=True)

    for fp_num in range(0, LARGE_JP_FILE_NUM):
        print("xs docs: ", fp_num)
        with open(xs_large_jp_dir + '/' + str(fp_num) + ".pb2_large_jp.json", "r", encoding='utf-8') as f:
            docs = json.load(f)

        doc_small = []
        for doc in docs:
            if random.random() < 0.1:
                doc_small.append(doc)

        with open(xs_small_jp_dir + '/' + str(fp_num) + ".pb2_small_jp.json", "w", encoding='utf-8') as f:
            json.dump(doc_small, f, ensure_ascii=False)


def add_small_corpus():
    with open(large_jp_info_path, encoding='utf-8') as f:
        large_info = json.load(f)

    with open(small_jp_info_path, encoding='utf-8') as f:
        small_info = json.load(f)

    large_crime_ids = {k[0] for k in large_info['crime_id_dis']}
    large_laws = {tuple(k[0]) for k in large_info['term_dis']}

    small_crime_ids = {k[0] for k in small_info['crime_id_dis']}
    small_laws = {tuple(k[0]) for k in small_info['term_dis']}

    need_crime_ids = large_crime_ids - small_crime_ids
    need_laws = large_laws - small_laws
    need_crime_ids_dis = {k[0]: max(k[1] * 0.10, 12) for k in large_info['crime_id_dis']}
    need_laws_dis = {tuple(k[0]): max(k[1] * 0.10, 12) for k in large_info['term_dis'] if tuple(k[0]) in need_laws}

    cur_crime_ids_dis = {k: 0 for k in large_crime_ids}
    cur_laws_dis = {k: 0 for k in need_laws}

    doc_small = []
    for fp_num in range(0, LARGE_JP_FILE_NUM):
        print("xs docs: ", fp_num)
        with open(xs_large_jp_dir + '/' + str(fp_num) + ".pb2_large_jp.json", "r", encoding='utf-8') as f:
            docs = json.load(f)

        for doc in docs:

            flag = False
            for crime_id in doc['AJAY']:
                if crime_id in large_crime_ids and cur_crime_ids_dis[crime_id] < 15:
                    flag = True
                    break
            if flag is True:
                for crime_id in doc['AJAY']:
                    cur_crime_ids_dis[crime_id] += 1
                doc_small.append(doc)
                continue

            if doc['AJAY'][0] in need_crime_ids and cur_crime_ids_dis[doc['AJAY'][0]] < need_crime_ids_dis[
                doc['AJAY'][0]]:
                cur_crime_ids_dis[doc['AJAY'][0]] += 1
                doc_small.append(doc)
                continue

            if len(set({tuple(i) for i in doc['related_laws']}).intersection(need_laws)) == 0:
                continue

            flag = False
            for i in doc['related_laws']:
                if tuple(i) not in large_laws:
                    flag = True
                    break
            if flag is True:
                continue
            flag = False
            for i in doc['related_laws']:
                if tuple(i) in need_laws and cur_laws_dis[tuple(i)] < 15:
                    break
            else:
                for i in doc['related_laws']:
                    if tuple(i) in need_laws and cur_laws_dis[tuple(i)] > need_laws_dis[tuple(i)]:
                        flag = True
                    break

            if flag is True:
                continue

            for i in doc['related_laws']:
                if tuple(i) in need_laws:
                    cur_laws_dis[tuple(i)] += 1
            doc_small.append(doc)
            print(len(doc_small))

    with open(xs_small_jp_dir + '/' "0.pb2_added_small_jp.json", "w", encoding='utf-8') as f:
        json.dump(doc_small, f, ensure_ascii=False)


def statistics(mode, start=0, end=SMALL_JP_FILE_NUM):
    if mode == 'original':
        dir_path = xs_jp_dir
        suffix = ".pb2_jp.json"
        info_output_path = original_jp_info_path

    elif mode == 'combine':
        dir_path = xs_jp_comb_dir
        suffix = ".pb2_jp_comb.json"
        info_output_path = comb_jp_info_path

    elif mode == 'sample':
        dir_path = xs_sample_jp_dir
        suffix = ".pb2_sample_jp.json"
        info_output_path = sample_jp_info_path

    elif mode == 'large':
        dir_path = xs_large_jp_dir
        suffix = ".pb2_large_jp.json"
        info_output_path = large_jp_info_path

    elif mode == 'small':
        dir_path = xs_small_jp_dir
        suffix = ".pb2_small_jp.json"
        info_output_path = small_jp_info_path

    elif mode == 'added_small':
        dir_path = xs_small_jp_dir
        suffix = ".pb2_small_jp.json"
        info_output_path = added_small_jp_info_path
    else:
        raise NotImplementedError

    id2ay = json.load(open(mapping_path))['id2ay']

    total_num = 0
    SS_lengths = []
    crime_dis = {}
    crime_id_dis = {}
    term_dis = {}
    death_penalty_num = 0
    life_imprisonment_num = 0
    imprison_dis = {}

    for fp_num in range(start, end):
        print("xs docs: ", fp_num)
        with open(dir_path + '/' + str(fp_num) + suffix, "r", encoding='utf-8') as f:
            docs = json.load(f)

        for doc in docs:
            SS_lengths.append(len(doc.get("SS")))
            total_num += 1

            if len(doc.get("crime")) > 0:
                for crime_id in doc.get("crime"):
                    crime = id2ay[crime_id]

                    if crime_id not in crime_id_dis.keys():
                        crime_id_dis[crime_id] = 1
                    else:
                        crime_id_dis[crime_id] += 1

                    if crime not in crime_dis.keys():
                        crime_dis[crime] = 1
                    else:
                        crime_dis[crime] += 1

            term_ids = doc.get("related_laws")
            for l in term_ids:
                l = tuple(l)
                if l not in term_dis.keys():
                    term_dis[l] = 1
                else:
                    term_dis[l] += 1

            # imprisonment

            term_of_imprisonment = doc['term_of_imprisonment']
            imprisonment = term_of_imprisonment['imprisonment']
            death_penalty = term_of_imprisonment['death_penalty']
            life_imprisonment = term_of_imprisonment['life_imprisonment']

            if death_penalty is True:
                death_penalty_num += 1
            elif life_imprisonment is True:
                life_imprisonment_num += 1
            else:
                if imprisonment not in imprison_dis.keys():
                    imprison_dis[imprisonment] = 1
                else:
                    imprison_dis[imprisonment] += 1

    if mode == 'added_small':
        with open(xs_small_jp_dir + '/' "0.pb2_added_small_jp.json", encoding='utf-8') as f:
            docs = json.load(f)

        for doc in docs:
            SS_lengths.append(len(doc.get("SS")))
            total_num += 1
            print(total_num)

            if len(doc.get("crime")) > 0:

                crime_id = doc.get("crime")[0]
                crime = id2ay[crime_id]

                if crime_id not in crime_id_dis.keys():

                    crime_id_dis[crime_id] = 1
                else:
                    crime_id_dis[crime_id] += 1

                if crime not in crime_dis.keys():
                    crime_dis[crime] = 1
                else:
                    crime_dis[crime] += 1

            term_ids = doc.get("related_laws")
            for l in term_ids:
                l = tuple(l)
                if l not in term_dis.keys():
                    term_dis[l] = 1
                else:
                    term_dis[l] += 1

            # imprisonment

            term_of_imprisonment = doc['term_of_imprisonment']
            imprisonment = term_of_imprisonment['imprisonment']
            death_penalty = term_of_imprisonment['death_penalty']
            life_imprisonment = term_of_imprisonment['life_imprisonment']

            if death_penalty is True:
                death_penalty_num += 1
            elif life_imprisonment is True:
                life_imprisonment_num += 1
            else:
                if imprisonment not in imprison_dis.keys():
                    imprison_dis[imprisonment] = 1
                else:
                    imprison_dis[imprisonment] += 1

    avg_length = sum(SS_lengths) / len(SS_lengths)

    if mode == 'added_small' or mode == 'small' or mode == 'large':
        crime_id_dis = [(k, i) for k, i in crime_id_dis.items()]
    else:
        crime_id_dis = [(k, i) for k, i in crime_id_dis.items() if i >= CRIME_NUM_THRESHOLD]

    crime_id_dis.sort(key=lambda x: x[1], reverse=True)

    if mode == 'added_small' or mode == 'small' or mode == 'large':
        crime_dis = [(k, i) for k, i in crime_dis.items()]
    else:
        crime_dis = [(k, i) for k, i in crime_dis.items() if i >= CRIME_NUM_THRESHOLD]
    crime_dis.sort(key=lambda x: x[1], reverse=True)

    if mode == 'added_small' or mode == 'small' or mode == 'large':
        term_dis = [(k, i) for k, i in term_dis.items()]
    else:
        term_dis = [(k, i) for k, i in term_dis.items() if i >= XS_LAW_THRESHOLD]
    term_dis.sort(key=lambda x: x[1], reverse=True)

    imprison_dis = [(k, i) for k, i in imprison_dis.items()]
    imprison_dis.sort(key=lambda x: x[0])
    max_imprison_time = imprison_dis[0][0]
    min_imprison_time = imprison_dis[-1][0]

    imprison_dis.append(['death_penalty', death_penalty_num])
    imprison_dis.append(['life_imprisonment', life_imprisonment_num])

    xs_jp_info = {'total_num': total_num, 'avg_length': avg_length, 'crime_num': len(crime_dis),
                  'term_num': len(term_dis), 'max_imprison_time': max_imprison_time,
                  'min_imprison_time': min_imprison_time, 'crime_id_dis': crime_id_dis, 'crime_dis': crime_dis,
                  'term_dis': term_dis,
                  'imprison_dis': imprison_dis}

    with open(info_output_path, 'w', encoding='utf-8') as f:
        json.dump(xs_jp_info, f, ensure_ascii=False)


def check_kill_law():
    for fp_num in range(1000, 1100):
        print("xs docs: ", fp_num)
        with open(xs_jp_dir + '/' + str(fp_num) + ".pb2_jp.json", encoding='utf-8') as f:
            docs = json.load(f)

        for doc in docs:
            SS = doc['SS']
            SS = SS.split("。")
            for i in SS:
                if '《' in i:
                    print(i)


def imprisonment_statistics():
    juyi_num = 0
    guanzhi_num = 0
    huanxing_num = 0
    total = 0

    for fp_num in range(100):
        print("xs docs: ", fp_num)
        with open(xs_jp_dir + '/' + str(fp_num) + ".pb2_jp.json", encoding='utf-8') as f:
            docs = json.load(f)
        for doc in docs:
            if not _is_right_doc(doc):
                continue
            total += 1

            if '管制' in doc['JG']:
                guanzhi_num += 1
                continue

            if '拘役' in doc['JG']:
                juyi_num += 1
                continue

            if '缓刑' in doc['JG']:
                huanxing_num += 1
                continue

    print(juyi_num / total)
    print(guanzhi_num / total)
    print(huanxing_num / total)


def tmp():
    count = 0
    total = 0
    for fp_num in range(0, 100):
        print("xs docs: ", fp_num)
        with open(xs_plm_ids_dir + '/' + str(fp_num) + ".pb2_plm_ids.json", encoding='utf-8') as f:
            docs = json.load(f)
        docs_with_jp = []
        for doc in docs:

            total += 1
            if len(doc['AJAY']) > 1:
                count += 1

    print(count / total)


if __name__ == '__main__':
    # extract_JP(0, LARGE_JP_FILE_NUM)
    # statistics('original', 0, LARGE_JP_FILE_NUM)

    # create_large_corpus()
    # statistics('large', 0, LARGE_JP_FILE_NUM)
    # create_small_corpus()
    # statistics('small', 0, LARGE_JP_FILE_NUM)
    add_small_corpus()
    statistics('added_small', 0, LARGE_JP_FILE_NUM)
    # create_small_corpus()
    # check_kill_law()
