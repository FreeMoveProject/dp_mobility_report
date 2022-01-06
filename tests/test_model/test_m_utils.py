#!/usr/bin/env python

import pytest

import numpy as np
from dp_mobility_report import constants as const
from dp_mobility_report.model import m_utils

def test_haversine_distxsdf():
    const.UID
    haversine_dist = m_utils.haversine_dist([52.5217, 13.4135, 52.5162, 13.3777])
    assert round(haversine_dist) == 2498

def test_cut_outliers():
    data = np.array([1,4,5,6,10])
    cut_data, n_outliers = m_utils.cut_outliers(data, min_value=2, max_value=9)
    assert cut_data.tolist() == [4,5,6]
    assert n_outliers == 2
    cut_data, n_outliers = m_utils.cut_outliers(data, min_value=1, max_value=10)
    assert cut_data.tolist() == data.tolist()
    assert n_outliers == 0
    cut_data, n_outliers = m_utils.cut_outliers(data, min_value=5, max_value=5)
    assert cut_data.tolist() == [5]
    cut_data, n_outliers = m_utils.cut_outliers(data, min_value=None, max_value=None)
    assert cut_data.tolist() == data.tolist()
    assert n_outliers == 0
    cut_data, n_outliers = m_utils.cut_outliers(data, min_value=None, max_value=9)
    assert cut_data.tolist() == [1,4,5,6]
    assert n_outliers == 1
    cut_data, n_outliers = m_utils.cut_outliers(data, min_value=2, max_value=None)
    assert cut_data.tolist() == [4,5,6,10]
    assert n_outliers == 1
    with pytest.raises(ValueError):
        m_utils.cut_outliers(data, min_value=4, max_value=2)
    

def test_hist_section():
    series = np.array([1,1,3,3,5,5,7,7])

    hist_sec = m_utils.hist_section(series, None, 1)
    assert hist_sec.data[0].tolist() == [2, 0, 2, 0, 2, 0, 2]
    assert hist_sec.data[1].round(2).tolist() == [1.0, 1.86, 2.71, 3.57, 4.43, 5.29, 6.14, 7.0]
    assert hist_sec.privacy_budget == None
    assert hist_sec.n_outliers == None
    assert hist_sec.quartiles.tolist() == [1,2.5,4,5.5,7]