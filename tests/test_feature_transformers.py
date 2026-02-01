from datetime import date

import numpy as np
import pandas as pd

from nowcast_data.features import MissingnessFilter, QuarterlyFeatureBuilder, compute_gdp_qoq_saar


def test_quarterly_feature_builder_no_leakage_ordering() -> None:
    df = pd.DataFrame(
        {
            "asof_date": [date(2020, 7, 15), date(2020, 7, 15), date(2020, 8, 15), date(2020, 8, 15)],
            "ref_quarter_end": [
                pd.Timestamp("2020-06-30"),
                pd.Timestamp("2020-03-31"),
                pd.Timestamp("2020-06-30"),
                pd.Timestamp("2020-03-31"),
            ],
            "unrate": [6.0, 5.0, 9.0, 7.0],
        }
    )

    builder = QuarterlyFeatureBuilder(predictor_keys=["unrate"])
    out = builder.transform(df)

    expected = pd.Series([1.0, np.nan, 2.0, np.nan], index=df.index, name="unrate__qoq")
    pd.testing.assert_series_equal(out["unrate__qoq"], expected)
    pd.testing.assert_series_equal(out["unrate__level"], df["unrate"], check_names=False)
    expected_yoy = pd.Series([np.nan, np.nan, np.nan, np.nan], index=df.index, name="unrate__yoy")
    pd.testing.assert_series_equal(out["unrate__yoy"], expected_yoy)
    pd.testing.assert_series_equal(
        out["unrate__isna"], pd.Series([0, 0, 0, 0], index=df.index), check_names=False
    )


def test_compute_gdp_qoq_saar() -> None:
    series = pd.Series([100.0, 101.0, 102.0])
    out = compute_gdp_qoq_saar(series)
    annualization = 400.0
    expected = pd.Series(
        [
            np.nan,
            np.log(101.0 / 100.0) * annualization,
            np.log(102.0 / 101.0) * annualization,
        ]
    )
    np.testing.assert_allclose(out.values, expected.values, rtol=1e-8, atol=1e-12, equal_nan=True)


def test_missingness_filter_drops_columns() -> None:
    df = pd.DataFrame(
        {
            "a": [np.nan, np.nan, 1.0, np.nan],
            "b": [1.0, np.nan, 2.0, 3.0],
            "c": [1.0, 2.0, 3.0, 4.0],
        }
    )
    filt = MissingnessFilter(max_nan_frac=0.5)
    filt.fit(df)
    out = filt.transform(df)
    assert list(out.columns) == ["b", "c"]


def test_quarterly_feature_builder_short_history_outputs_nan_changes() -> None:
    df = pd.DataFrame(
        {
            "asof_date": [date(2020, 7, 15)],
            "ref_quarter_end": [pd.Timestamp("2020-06-30")],
            "unrate": [6.0],
        }
    )

    builder = QuarterlyFeatureBuilder(predictor_keys=["unrate"])
    out = builder.transform(df)

    assert set(out.columns) == {
        "unrate__level",
        "unrate__qoq",
        "unrate__yoy",
        "unrate__isna",
    }
    assert out["unrate__level"].iloc[0] == 6.0
    assert np.isnan(out["unrate__qoq"].iloc[0])
    assert np.isnan(out["unrate__yoy"].iloc[0])
    assert out["unrate__isna"].iloc[0] == 0


def test_quarterly_feature_builder_passthrough_non_predictor_cols() -> None:
    df = pd.DataFrame(
        {
            "asof_date": [date(2020, 7, 15), date(2020, 7, 15)],
            "ref_quarter_end": [pd.Timestamp("2020-03-31"), pd.Timestamp("2020-06-30")],
            "unrate": [5.0, 6.0],
            "y_asof_latest_growth": [1.23, 4.56],
        }
    )

    out = QuarterlyFeatureBuilder(predictor_keys=["unrate"]).transform(df)

    assert "y_asof_latest_growth" in out.columns
    pd.testing.assert_series_equal(
        out["y_asof_latest_growth"], df["y_asof_latest_growth"], check_names=False
    )
    assert {"unrate__level", "unrate__qoq", "unrate__yoy", "unrate__isna"}.issubset(out.columns)
    pd.testing.assert_series_equal(out["unrate__level"], df["unrate"], check_names=False)
