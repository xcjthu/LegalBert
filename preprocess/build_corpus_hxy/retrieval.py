# encoding:utf-8
import json

from build_corpus.config import *
from build_corpus.const import *
from build_corpus.helper import read_formatted_laws


def _find_law_by_id(law_id: int, all_formatted_laws: list):
    for law in all_formatted_laws:
        if law['id_'] == law_id:
            return law

    raise ValueError


def _get_content_and_chapter(content: dict, term: str, kuan: str):
    target_term = None
    other_terms = None  # may be empty

    for chapter in content:
        if term in chapter.keys():
            if kuan == TERM_IGNORE_MARK:

                target_term = "".join(chapter[term].values())
                other_terms = chapter.copy()
                del other_terms[term]
                break
            else:
                if kuan in chapter[term].keys():
                    target_term = chapter[term][kuan]
                    other_terms = chapter.copy()
                    del other_terms[term]
                    break

    return target_term, other_terms


def get_id_to_law_name():
    formatted_fl, formatted_sfjs, formatted_xzfg, formatted_dfxfg = read_formatted_laws()

    all_formatted_laws = formatted_fl + formatted_sfjs

    for i in all_formatted_laws:
        del i['head']
        del i['content']
        del i['file_path']
    with open(id_to_law_name_dict,'w',encoding='utf-8') as f:
        json.dump(all_formatted_laws,f,ensure_ascii=False)


def find_term_by_index(index: tuple, formatted_laws: list):
    """
    :param index: (law_id, term, kuan)
    :param formatted_laws: please use read_formatted_laws() to get
    :return: other_terms may be empty if the chapter only has one term
    """

    target_law = _find_law_by_id(index[0], formatted_laws)
    target_term, other_terms = _get_content_and_chapter(target_law['content'], index[1], index[2])

    return target_term, other_terms, target_law




# if __name__ == '__main__':
#     formatted_fl, formatted_sfjs, formatted_xzfg, formatted_dfxfg = read_formatted_laws()
#     all_formatted_laws = formatted_fl + formatted_sfjs + formatted_xzfg + formatted_dfxfg
#
#     # for fp_num in range(0, TOTAL_MS_NUM):
#     #     print("ms docs: ", fp_num)
#     #     with open(ms_plm_ids_dir + '/' + str(fp_num) + ".pb2_plm_ids.json", "r", encoding='utf-8') as f:
#     #         docs = json.load(f)
#     #
#     #     for doc in docs:
#     #         term_ids = doc['term_ids']
#     #
#     #         for index in term_ids:
#     #             target_term, other_terms, target_law = find_term_by_index(index, all_formatted_laws)
#     #             print(target_term)
#     #             # print(other_terms)
#     #             # print(target_law)
#     #         break
#     #     break
#
#     print(find_term_by_index((506, '二百五十三', -100),all_formatted_laws))

if __name__ == '__main__':
    # with open(large_ac_info_path) as f:
    #     info = json.load(f)['term_dis']
    #
    # ids = [i[0][0] for i in info]
    # print(ids)
    # print(max(ids))
    get_id_to_law_name()

