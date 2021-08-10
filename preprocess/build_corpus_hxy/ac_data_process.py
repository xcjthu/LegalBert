# encoding:utf-8
"""
Extract aciton causes from ms docs for the action cause prediction task.
"""
import sys
import os

sys.path.append('/mnt/datadisk0/hxy/PLM')

from build_corpus.retrieval import find_term_by_index
from build_corpus.plm_data_process import *
from build_corpus.helper import read_formatted_laws


def ac_format():
    """
    This method has been abandoned.
    Format all action causes from actioncause.txt, but now we can action causes
    from mapping_v2.json which is associated with the documents dataset.

    :return:
    """

    all_ac = []

    with open("../data/actioncause/actioncause.txt", "r") as f:
        line = f.readline()
        while line:

            res = re.search(p_actioncause, line)

            if res is None:
                line = f.readline()
                continue

            ac = res.group("actioncause").strip()
            all_ac.append(ac)

            line = f.readline()

    return all_ac


def _is_right_doc(doc):
    """
    Judge if the doc is used to construct the dataset.

    :param doc:
    :return:
    """


    TITLE = doc.get("TITLE")
    date = doc.get("date")
    ajay_id = doc.get("AJAY")
    related_laws = doc.get("related_laws")

    # 只选一审民事判决书
    if '一审民事判决书' not in TITLE or 'SS' not in doc.keys() or len(doc['SS']) < 50:
        return False

    if date['year'] == DATE_IGNORE_MARK or date['year'] < 1990:
        return False

    if len(ajay_id) < 1:
        return False

    if 'related_laws' not in doc.keys() or len(related_laws) == 0:
        return False

    return True


def _clean_SS(SS, all_law_set):
    """
    Remove the sentences in the SS which contains the law in the all_law_set.

    :param SS:
    :param all_law_set:
    :return:
    """


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


def _convert_kuan_to_term(input):
    """
    Sometimes we only need the related laws accurate to the level of tiao(条) instead of kuan(款),
    so we replace kuan with TERM_IGNORE_MARK.

    :param input:
    :return:
    """


    if type(input) == list:
        output = []
        for i in input:
            tmp = [i[0], i[1], TERM_IGNORE_MARK]
            if tmp not in output:
                output.append(tmp)
    else:
        raise NotImplementedError

    return output


def _is_same_law(law1, law2):
    """
    Judge if two terms from different versions of one law are the same. Obviously,
    there are a large number of terms unchanged after the revision of laws.

    :param law1:
    :param law2:
    :return:
    """


    law1 = law1.strip()
    law2 = law2.strip()
    if law1 == law2:
        return True

    if len(law1) > 30 and law1[10:25] in law2:
        return True

    return False


def parse_one_doc(doc, id2ay, all_formatted_laws, already_exist_laws, already_exist_laws_content, lookup_dict,
                  all_law_set):

    """
    Parse all needed information from one doc.There are three stages.
    First, get the id of actioncause from AJAY.
    Second, clean the SS through the method _clean_SS.
    Third, get the related laws. Given that there are terms from different versions of law factually same, use
    _is_same_law to merge.

    :param doc:
    :param id2ay:
    :param all_formatted_laws:
    :param already_exist_laws:
    :param already_exist_laws_content:
    :param lookup_dict:
    :param all_law_set:
    :return:
    """


    ajay_id = doc.get("AJAY")[0]
    doc.update({"actioncause": ajay_id})

    doc['SS'] = _clean_SS(doc['SS'].replace(id2ay[ajay_id], ""), all_law_set)

    related_laws = _convert_kuan_to_term(doc.get("term_ids"))
    del doc['term_ids']
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
                    # print('new', tuple(new_term))
                    # print('exists', already_term)
                    lookup_dict[tuple(new_term)] = list(already_term)
                    new_related_laws.append(list(already_term))
                    break
            else:
                already_exist_laws.add(tuple(new_term))
                already_exist_laws_content[tuple(new_term)] = find_term_by_index(tuple(new_term), all_formatted_laws)[0]
                new_related_laws.append(new_term)

    doc['related_laws'] = new_related_laws

    return doc


def extract_ac_and_related_laws(start=0, end=SMALL_AC_FILE_NUM):
    """
    Extract actioncauses and related_laws from XXX.pb2_plm_ids.json.

    :param start: the start index of docs
    :param end: the end index of docs
    :return:
    """



    os.makedirs(ms_ac_dir, exist_ok=True)

    # get the mapping
    id2ay = json.load(open(mapping_path))['id2ay']

    # get all formatted laws
    all_shorten_formatted_laws_name = json.load(open(all_shorten_formatted_laws_name_path))
    all_formatted_laws_name = json.load(open(all_formatted_laws_name_path))
    all_lawname_list = all_shorten_formatted_laws_name + all_formatted_laws_name
    all_lawname_set = set(all_lawname_list)

    formatted_fl, formatted_sfjs, formatted_xzfg, formatted_dfxfg = read_formatted_laws()
    all_formatted_laws = formatted_fl + formatted_sfjs + formatted_xzfg + formatted_dfxfg

    # prepare for merging the same terms from different versions of law
    already_exist_laws = set({})
    already_exist_laws_content = {}
    lookup_dict = {}

    for fp_num in range(start, end):
        print("ms docs: ", fp_num)
        with open(ms_plm_ids_dir + '/' + str(fp_num) + ".pb2_plm_ids.json", "r", encoding='utf-8') as f:
            docs = json.load(f)

        docs_with_ac = []
        for doc_num, doc in enumerate(docs):

            if not _is_right_doc(doc):
                continue

            new_doc = parse_one_doc(doc, id2ay, all_formatted_laws, already_exist_laws, already_exist_laws_content,
                                    lookup_dict, all_lawname_set)

            docs_with_ac.append(new_doc)

        with open(ms_ac_dir + '/' + str(fp_num) + ".pb2_ac.json", "w", encoding='utf-8') as f:
            json.dump(docs_with_ac, f, ensure_ascii=False)





# def create_large_corpus():
#     os.makedirs(ms_large_ac_dir, exist_ok=True)
#
#     with open(original_ac_info_path, encoding='utf-8') as f:
#         info = json.load(f)
#
#     ac_ids = {k[0] for k in info['ac_id_dis']}
#     laws = {tuple(k[0]) for k in info['term_dis']}
#
#     # ac_ids = {ac_id: 0 for ac_id in ac_ids.keys()}
#     # term_dis = {tuple(law[0]): 0 for law in term_dis}
#
#     for fp_num in range(0, LARGE_AC_FILE_NUM):
#         print("ms docs: ", fp_num)
#         with open(ms_ac_dir + '/' + str(fp_num) + ".pb2_ac.json", "r", encoding='utf-8') as f:
#             docs = json.load(f)
#
#         doc_large = []
#         for doc in docs:
#
#             # maybe multilabel
#             ac_id = doc['actioncause']
#             flag = False
#
#             if ac_id not in ac_ids:
#                 flag = True
#
#             if flag is True:
#                 continue
#
#             related_laws = doc['related_laws']
#             related_laws = [id_ for id_ in related_laws if tuple(id_) in laws]
#
#             if len(related_laws) == 0:
#                 continue
#
#             doc['related_laws'] = related_laws
#             doc_large.append(doc)
#
#         with open(ms_large_ac_dir + '/' + str(fp_num) + ".pb2_large_ac.json", "w", encoding='utf-8') as f:
#             json.dump(doc_large, f, ensure_ascii=False)
#
#
# def create_small_corpus():
#     os.makedirs(ms_small_ac_dir, exist_ok=True)
#
#     for fp_num in range(0, LARGE_AC_FILE_NUM):
#         print("ms docs: ", fp_num)
#         with open(ms_large_ac_dir + '/' + str(fp_num) + ".pb2_large_ac.json", "r", encoding='utf-8') as f:
#             docs = json.load(f)
#
#         doc_small = []
#         for doc in docs:
#             if random.random() < 0.1:
#                 doc_small.append(doc)
#
#         with open(ms_small_ac_dir + '/' + str(fp_num) + ".pb2_small_ac.json", "w", encoding='utf-8') as f:
#             json.dump(doc_small, f, ensure_ascii=False)
#
#
# def add_small_corpus():
#     with open(large_ac_info_path, encoding='utf-8') as f:
#         large_info = json.load(f)
#
#     with open(small_ac_info_path, encoding='utf-8') as f:
#         small_info = json.load(f)
#
#     large_ac_ids = {k[0] for k in large_info['ac_id_dis']}
#     large_laws = {tuple(k[0]) for k in large_info['term_dis']}
#
#     small_ac_ids = {k[0] for k in small_info['ac_id_dis']}
#     small_laws = {tuple(k[0]) for k in small_info['term_dis']}
#
#     need_ac_ids = large_ac_ids - small_ac_ids
#     need_laws = large_laws - small_laws
#     need_ac_ids_dis = {k[0]: max(k[1] * 0.10, 12) for k in large_info['ac_id_dis']}
#     need_laws_dis = {tuple(k[0]): max(k[1] * 0.10, 12) for k in large_info['term_dis'] if tuple(k[0]) in need_laws}
#
#     cur_ac_ids_dis = {k: 0 for k in large_ac_ids}
#     cur_laws_dis = {k: 0 for k in need_laws}
#
#     doc_small = []
#     for fp_num in range(0, LARGE_AC_FILE_NUM):
#         print("xs docs: ", fp_num)
#         with open(ms_large_ac_dir + '/' + str(fp_num) + ".pb2_large_ac.json", "r", encoding='utf-8') as f:
#             docs = json.load(f)
#
#         for doc in docs:
#
#             flag = False
#             ac_id = doc['AJAY'][0]
#             if ac_id in large_ac_ids and cur_ac_ids_dis[ac_id] < 15:
#                 flag = True
#
#
#             if flag is True:
#
#                 cur_ac_ids_dis[ac_id] += 1
#                 doc_small.append(doc)
#                 continue
#
#             if doc['AJAY'][0] in need_ac_ids and cur_ac_ids_dis[doc['AJAY'][0]] < need_ac_ids_dis[
#                 doc['AJAY'][0]]:
#                 cur_ac_ids_dis[doc['AJAY'][0]] += 1
#                 doc_small.append(doc)
#                 continue
#
#             if len(set({tuple(i) for i in doc['related_laws']}).intersection(need_laws)) == 0:
#                 continue
#
#             flag = False
#             for i in doc['related_laws']:
#                 if tuple(i) not in large_laws:
#                     flag = True
#                     break
#             if flag is True:
#                 continue
#             flag = False
#             for i in doc['related_laws']:
#                 if tuple(i) in need_laws and cur_laws_dis[tuple(i)] < 15:
#                     break
#             else:
#                 for i in doc['related_laws']:
#                     if tuple(i) in need_laws and cur_laws_dis[tuple(i)] > need_laws_dis[tuple(i)]:
#                         flag = True
#                     break
#
#             if flag is True:
#                 continue
#
#             for i in doc['related_laws']:
#                 if tuple(i) in need_laws:
#                     cur_laws_dis[tuple(i)] += 1
#             doc_small.append(doc)
#             print(len(doc_small))
#
#     with open(ms_small_ac_dir + '/' "0.pb2_added_small_ac.json", "w", encoding='utf-8') as f:
#         json.dump(doc_small, f, ensure_ascii=False)
#
#
# def statistics(mode, start=0, end=LARGE_AC_FILE_NUM):
#     if mode == 'original':
#         dir_path = ms_ac_dir
#         suffix = ".pb2_ac.json"
#         info_output_path = original_ac_info_path
#
#     elif mode == 'large':
#         dir_path = ms_large_ac_dir
#         suffix = ".pb2_large_ac.json"
#         info_output_path = large_ac_info_path
#
#     elif mode == 'small':
#         dir_path = ms_small_ac_dir
#         suffix = ".pb2_small_ac.json"
#         info_output_path = small_ac_info_path
#
#     elif mode == 'added_small':
#         dir_path = ms_small_ac_dir
#         suffix = ".pb2_small_ac.json"
#         info_output_path = added_small_ac_info_path
#
#     else:
#         raise NotImplementedError
#
#     id2ay = json.load(open(mapping_path))['id2ay']
#
#     SS_lengths = []
#     ac_dis = {}
#     ac_id_dis = {}
#     term_dis = {}
#     total_num = 0
#
#     for fp_num in range(start, end):
#         print("ms docs: ", fp_num)
#         with open(dir_path + '/' + str(fp_num) + suffix, "r", encoding='utf-8') as f:
#             docs = json.load(f)
#
#         for doc in docs:
#             SS_lengths.append(len(doc.get("SS")))
#             total_num += 1
#             ac_id = doc.get("actioncause")
#
#             ac = id2ay[ac_id]
#
#             if ac_id not in ac_id_dis.keys():
#                 ac_id_dis[ac_id] = 1
#             else:
#                 ac_id_dis[ac_id] += 1
#
#             if ac not in ac_dis.keys():
#                 ac_dis[ac] = 1
#             else:
#                 ac_dis[ac] += 1
#
#             related_laws = doc.get("related_laws")
#             for l in related_laws:
#                 l = tuple(l)
#                 if l not in term_dis.keys():
#                     term_dis[l] = 1
#                 else:
#                     term_dis[l] += 1
#
#     if mode == 'added_small':
#         with open(ms_small_ac_dir + '/' "0.pb2_added_small_ac.json", encoding='utf-8') as f:
#             docs = json.load(f)
#
#         for doc in docs:
#             SS_lengths.append(len(doc.get("SS")))
#             total_num += 1
#             ac_id = doc.get("actioncause")
#
#             ac = id2ay[ac_id]
#
#             if ac_id not in ac_id_dis.keys():
#                 ac_id_dis[ac_id] = 1
#             else:
#                 ac_id_dis[ac_id] += 1
#
#             if ac not in ac_dis.keys():
#                 ac_dis[ac] = 1
#             else:
#                 ac_dis[ac] += 1
#
#             related_laws = doc.get("related_laws")
#             for l in related_laws:
#                 l = tuple(l)
#                 if l not in term_dis.keys():
#                     term_dis[l] = 1
#                 else:
#                     term_dis[l] += 1
#
#     if mode == 'original':
#         ac_id_dis = [(k, i) for k, i in ac_id_dis.items() if i >= MIN_AC_THRESHOLD]
#         ac_id_dis.sort(key=lambda x: x[1], reverse=True)
#     else:
#         ac_id_dis = [(k, i) for k, i in ac_id_dis.items()]
#         ac_id_dis.sort(key=lambda x: x[1], reverse=True)
#
#     if mode == 'original':
#         ac_dis = [(k, i) for k, i in ac_dis.items() if i >= MIN_AC_THRESHOLD]
#         ac_dis.sort(key=lambda x: x[1], reverse=True)
#     else:
#         ac_dis = [(k, i) for k, i in ac_dis.items()]
#         ac_dis.sort(key=lambda x: x[1], reverse=True)
#
#     if mode == 'original':
#         term_dis = [(k, i) for k, i in term_dis.items() if i >= MS_LAW_THRESHOLD]
#         term_dis.sort(key=lambda x: x[1], reverse=True)
#     else:
#         term_dis = [(k, i) for k, i in term_dis.items()]
#         term_dis.sort(key=lambda x: x[1], reverse=True)
#
#     avg_length = sum(SS_lengths) / len(SS_lengths)
#
#     print(ac_dis)
#     print(term_dis)
#
#     print(avg_length)
#     print(len(term_dis))
#     print(total_num)
#
#     ms_ac_info = {'total_num': total_num, 'avg_length': avg_length, 'ac_num': len(ac_dis), 'ac_dis': ac_dis,
#                   'ac_id_dis': ac_id_dis,
#                   'term_num': len(term_dis), 'term_dis': term_dis}
#
#     with open(info_output_path, 'w', encoding='utf-8') as f:
#         json.dump(ms_ac_info, f, ensure_ascii=False)


if __name__ == '__main__':
    # extract_ac_and_related_laws(0, LARGE_AC_FILE_NUM)
