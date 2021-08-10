"""
Two parts have been included in this script, the file paths and some configs
for constructing the dataset.
"""


# ac_extractor
ac_examples_output_path = '/mnt/datadisk0/hxy/PLM/data/ac_prediction/examples/'

original_ac_info_path = '/mnt/datadisk0/hxy/data/ms/original_ac_info.json'
sample_ac_info_path = '/mnt/datadisk0/hxy/data/ms/sample_ac_info.json'
large_ac_info_path = '/mnt/datadisk0/hxy/data/ms/large_ac_info.json'
small_ac_info_path = '/mnt/datadisk0/hxy/data/ms/small_ac_info.json'
added_small_ac_info_path = '/mnt/datadisk0/hxy/data/ms/added_small_ac_info.json'

# ljp extractor
original_jp_info_path = '/mnt/datadisk0/hxy/data/xs/original_jp_info.json'
comb_jp_info_path = '/mnt/datadisk0/hxy/data/xs/comb_jp_info.json'
sample_jp_info_path = '/mnt/datadisk0/hxy/data/xs/sample_jp_info.json'
large_jp_info_path = '/mnt/datadisk0/hxy/data/xs/large_jp_info.json'
small_jp_info_path = '/mnt/datadisk0/hxy/data/xs/small_jp_info.json'
added_small_jp_info_path = '/mnt/datadisk0/hxy/data/xs/added_small_jp_info.json'

civil_law_statistics_path = '../result/civil_law_statistics.json'
civil_law_dist_path = '../result/civil_law_dist.json'

# pku and national laws
pku_fl_dir_path = "/mnt/datadisk0/hxy/data/all_pku_laws/pku_fl"
pku_sfjs_dir_path = '/mnt/datadisk0/hxy/data/all_pku_laws/pku_sfjs'
national_fl_dir_path = "/mnt/datadisk0/hxy/data/all_national_laws/national_fl"
national_xzfg_dir_path = "/mnt/datadisk0/hxy/data/all_national_laws/national_xzfg"
national_dfxfg_dir_path = '/mnt/datadisk0/hxy/data/all_national_laws/national_dfxfg_1_1539_docx'

# ms and xs
ms_raw_dir = "/mnt/datadisk0/hxy/data/ms/ms_raw"
ms_ac_dir = "/mnt/datadisk0/hxy/data/ms/ms_ac"
ms_large_ac_dir = "/mnt/datadisk0/hxy/data/ms/ms_large_ac"
ms_small_ac_dir = "/mnt/datadisk0/hxy/data/ms/ms_small_ac"
ms_sample_ac_dir = "/mnt/datadisk0/hxy/data/ms/ms_sample_ac"
ms_plm_dir = "/mnt/datadisk0/hxy/data/ms/ms_plm"
ms_plm_ids_dir = "/mnt/datadisk0/xcj/plm_data/data/ms_plm"

xs_raw_dir = "/mnt/datadisk0/hxy/data/xs/xs_raw"
xs_plm_dir = "/mnt/datadisk0/hxy/data/xs/xs_plm"
xs_jp_dir = "/mnt/datadisk0/hxy/data/xs/xs_jp"
xs_small_jp_dir = "/mnt/datadisk0/hxy/data/xs/xs_small_jp"
xs_large_jp_dir = "/mnt/datadisk0/hxy/data/xs/xs_large_jp"
xs_jp_comb_dir = "/mnt/datadisk0/hxy/data/xs/xs_jp_comb"
xs_sample_jp_dir = "/mnt/datadisk0/hxy/data/xs/xs_sample_jp"
xs_plm_ids_dir = "/mnt/datadisk0/xcj/plm_data/data/xs_plm"

plm_examples_dir = "/mnt/datadisk0/hxy/data/plm_data/examples/v2"
plm_features_dir = "/mnt/datadisk0/hxy/data/plm_data/features/v1_base"

# formatted laws
formatted_fl_path = '/mnt/datadisk0/hxy/data/formatted_laws/formatted_fl.json'
formatted_sfjs_path = '/mnt/datadisk0/hxy/data/formatted_laws/formatted_sfjs.json'
formatted_dfxfg_path = '/mnt/datadisk0/hxy/data/formatted_laws/formatted_dfxfg.json'
formatted_xzfg_path = '/mnt/datadisk0/hxy/data/formatted_laws/formatted_xzfg.json'
all_formatted_laws_name_path = '/mnt/datadisk0/hxy/data/formatted_laws/all_formatted_laws_name.json'
all_shorten_formatted_laws_name_path = '/mnt/datadisk0/hxy/data/formatted_laws/all_shorten_formatted_laws_name.json'
id_to_law_name_dict = '/mnt/datadisk0/hxy/data/formatted_laws/id2name.json'

# name date pair
fl_name_date_path = '/mnt/datadisk0/hxy/data/all_pku_laws/fl_name_date.json'
sfjs_name_date_path = "/mnt/datadisk0/hxy/data/all_pku_laws/sfjs_name_date.json"

# anay mapping
mapping_path = '/mnt/datadisk0/hxy/data/mapping_v2.json'

MIN_AC_THRESHOLD = 50
MS_LAW_THRESHOLD = 1000
CRIME_NUM_THRESHOLD = 12
XS_LAW_THRESHOLD = 400

TOTAL_MS_NUM = 23273
TOTAL_XS_NUM = 6077
SMALL_AC_FILE_NUM = 200
LARGE_AC_FILE_NUM = 2100
SMALL_JP_FILE_NUM = 300
LARGE_JP_FILE_NUM = 2400
