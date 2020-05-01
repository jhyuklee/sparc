import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def check_diff(model_a, model_b):
    a_set = set([a for a in model_a.keys()])
    b_set = set([b for b in model_b.keys()])
    if a_set != b_set:
        logger.info('load with different params =>')
    if len(a_set - b_set) > 0:
        logger.info('Loaded weight does not have ' + str(a_set - b_set))
    if len(b_set - a_set) > 0:
        logger.info('Model code does not have: ' + str(b_set - a_set))
