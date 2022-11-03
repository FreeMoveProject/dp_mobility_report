from typing import List
from dp_mobility_report import constants as const

# only select analyses that are present in both reports
def combine_analysis_exclusion(analysis_exclusion_proposal: List[str], analysis_exclusion_benchmark: List[str]) -> List[str]:
    analysis_exclusion = list(set(analysis_exclusion_proposal + analysis_exclusion_benchmark))
    analysis_exclusion.sort(key=lambda i: const.ELEMENTS.index(i))
    return analysis_exclusion

# make sure there is a valid measure selected for each analysis
def validate_measure_selection(measure_selection):
    pass