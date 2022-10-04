import geopandas as gpd
import pandas as pd
import pytest

from dp_mobility_report import constants as const
from dp_mobility_report.md_report import MobilityDataReport
from dp_mobility_report.report import report


@pytest.fixture
def test_mdreport():
    """Create a test report."""
    test_data = pd.read_csv("tests/test_files/test_data.csv")
    test_tessellation = gpd.read_file("tests/test_files/test_tessellation.geojson")
    return MobilityDataReport(test_data, test_tessellation, privacy_budget=None)


def test_report_elements(test_mdreport):

    elements = report.report_elements(test_mdreport)
    assert (
        list(elements.keys())
        == const.OVERVIEW_ELEMENTS
        + const.PLACE_ELEMENTS
        + const.OD_ELEMENTS
        + const.USER_ELEMENTS
    )

    test_mdreport.analysis_exclusion = (
        const.PLACE_ELEMENTS + const.OD_ELEMENTS + const.USER_ELEMENTS
    )
    elements = report.report_elements(test_mdreport)
    assert list(elements.keys()) == const.OVERVIEW_ELEMENTS

    test_mdreport.analysis_exclusion = (
        const.OVERVIEW_ELEMENTS + const.OD_ELEMENTS + const.USER_ELEMENTS
    )
    elements = report.report_elements(test_mdreport)
    assert list(elements.keys()) == const.PLACE_ELEMENTS

    test_mdreport.analysis_exclusion = (
        const.OVERVIEW_ELEMENTS + const.PLACE_ELEMENTS + const.USER_ELEMENTS
    )
    elements = report.report_elements(test_mdreport)
    assert list(elements.keys()) == const.OD_ELEMENTS

    test_mdreport.analysis_exclusion = (
        const.OVERVIEW_ELEMENTS + const.PLACE_ELEMENTS + const.OD_ELEMENTS
    )
    elements = report.report_elements(test_mdreport)
    assert list(elements.keys()) == const.USER_ELEMENTS

    test_mdreport.analysis_exclusion = const.PLACE_ELEMENTS + const.OD_ELEMENTS
    elements = report.report_elements(test_mdreport)
    assert list(elements.keys()) == const.OVERVIEW_ELEMENTS + const.USER_ELEMENTS

    test_mdreport.analysis_exclusion = (
        [const.VISITS_PER_TILE_TIMEWINDOW] + const.OD_ELEMENTS
    )
    elements = report.report_elements(test_mdreport)
    assert (
        list(elements.keys())
        == const.OVERVIEW_ELEMENTS + [const.VISITS_PER_TILE] + const.USER_ELEMENTS
    )
