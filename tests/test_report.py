import geopandas as gpd
import pandas as pd
import pytest

from dp_mobility_report import MobilityReport
from dp_mobility_report import constants as const
from dp_mobility_report.report import report


@pytest.fixture
def test_mreport():
    """Create a test report."""
    test_data = pd.read_csv("tests/test_files/test_data.csv")
    test_tessellation = gpd.read_file("tests/test_files/test_tessellation.geojson")
    return MobilityReport(test_data, test_tessellation, privacy_budget=None)


def test_report_elements(test_mreport):
    test_mreport.analysis_selection = [const.ALL]
    elements = report.report_elements(test_mreport)
    assert (
        list(elements.keys())
        == const.OVERVIEW_ELEMENTS
        + const.PLACE_ELEMENTS
        + const.OD_ELEMENTS
        + const.USER_ELEMENTS
    )

    test_mreport.analysis_selection = [const.OVERVIEW]
    elements = report.report_elements(test_mreport)
    assert list(elements.keys()) == const.OVERVIEW_ELEMENTS

    test_mreport.analysis_selection = [const.PLACE_ANALYSIS]
    elements = report.report_elements(test_mreport)
    assert list(elements.keys()) == const.PLACE_ELEMENTS

    test_mreport.analysis_selection = [const.OD_ANALYSIS]
    elements = report.report_elements(test_mreport)
    assert list(elements.keys()) == const.OD_ELEMENTS

    test_mreport.analysis_selection = [const.USER_ANALYSIS]
    elements = report.report_elements(test_mreport)
    assert list(elements.keys()) == const.USER_ELEMENTS

    test_mreport.analysis_selection = [const.OVERVIEW, const.USER_ANALYSIS]
    elements = report.report_elements(test_mreport)
    assert list(elements.keys()) == const.OVERVIEW_ELEMENTS + const.USER_ELEMENTS
