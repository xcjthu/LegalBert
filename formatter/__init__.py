import logging

from .Basic import BasicFormatter
from .VanillaLFMFormatter import VanillaLFMFormatter
from .DocLawFormatter import DocLawFormatter
from .NSPDocLawFormatter import NSPDocLawFormatter
from .ContrastiveLawFormatter import ContrastiveLawFormatter
from .UnContrastiveFormatter import UnContrastiveFormatter

logger = logging.getLogger(__name__)

formatter_list = {
    "Basic": BasicFormatter,
    'VanillaLFM': VanillaLFMFormatter,
    "DocLaw": DocLawFormatter,
    "NSPLaw": NSPDocLawFormatter,
    "ContrastiveLaw": ContrastiveLawFormatter,
    "UnContrastive": UnContrastiveFormatter
}


def init_formatter(config, mode, *args, **params):
    temp_mode = mode
    if mode != "train":
        try:
            config.get("data", "%s_formatter_type" % temp_mode)
        except Exception as e:
            logger.warning(
                "[reader] %s_formatter_type has not been defined in config file, use [dataset] train_formatter_type instead." % temp_mode)
            temp_mode = "train"
    which = config.get("data", "%s_formatter_type" % temp_mode)

    if which in formatter_list:
        formatter = formatter_list[which](config, mode, *args, **params)

        return formatter
    else:
        logger.error("There is no formatter called %s, check your config." % which)
        raise NotImplementedError
