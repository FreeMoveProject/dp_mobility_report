import pytest

import pandas as pd
import geopandas as gpd
from dp_mobility_report.md_report import MobilityDataReport
from dp_mobility_report.report import report
from dp_mobility_report import constants as const



@pytest.fixture
def test_mdreport():
    """Create a test report."""
    test_data = pd.read_csv("tests/test_files/test_data.csv")
    test_tessellation = gpd.read_file("tests/test_files/test_tessellation.geojson")
    return MobilityDataReport(test_data, test_tessellation, privacy_budget=None)


def test_report_elements(test_mdreport):
    test_mdreport.analysis_selection = [const.ALL]
    elements = report.report_elements(test_mdreport)
    assert list(elements.keys()) == const.OVERVIEW_ELEMENTS + const.PLACE_ELEMENTS + const.OD_ELEMENTS + const.USER_ELEMENTS

    test_mdreport.analysis_selection = [const.OVERVIEW]
    elements = report.report_elements(test_mdreport)
    assert list(elements.keys()) == const.OVERVIEW_ELEMENTS

    test_mdreport.analysis_selection = [const.PLACE_ANALYSIS]
    elements = report.report_elements(test_mdreport)
    assert list(elements.keys()) == const.PLACE_ELEMENTS

    test_mdreport.analysis_selection = [const.OD_ANALYSIS]
    elements = report.report_elements(test_mdreport)
    assert list(elements.keys()) == const.OD_ELEMENTS

    test_mdreport.analysis_selection = [const.USER_ANALYSIS]
    elements = report.report_elements(test_mdreport)
    assert list(elements.keys()) == const.USER_ELEMENTS

    test_mdreport.analysis_selection = [const.OVERVIEW, const.USER_ANALYSIS]
    elements = report.report_elements(test_mdreport)
    assert list(elements.keys()) == const.OVERVIEW_ELEMENTS + const.USER_ELEMENTS