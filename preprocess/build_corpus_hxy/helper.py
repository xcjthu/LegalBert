# encoding:utf-8
import json

from build_corpus.config import *


def read_formatted_laws(return_dict=False):
    """
    Read all formatted laws.

    :param return_dict:
    :return:
    """

    with open(formatted_fl_path, 'r', encoding='utf-8') as f:
        formatted_fl = json.load(f)

    with open(formatted_sfjs_path, 'r', encoding='utf-8') as f:
        formatted_sfjs = json.load(f)

    with open(formatted_xzfg_path, 'r', encoding='utf-8') as f:
        formatted_xzfg = json.load(f)

    with open(formatted_dfxfg_path, 'r', encoding='utf-8') as f:
        formatted_dfxfg = json.load(f)

    if not return_dict:
        return formatted_fl, formatted_sfjs, formatted_xzfg, formatted_dfxfg
    else:
        return {'formatted_fl': formatted_fl, 'formatted_sfjs': formatted_sfjs, 'formatted_xzfg': formatted_xzfg,
                'formatted_dfxfg': formatted_dfxfg}
