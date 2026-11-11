from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from sim_utils import collapse_abo
from sim_utils import haversine_NM


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_collapse_abo(polars_frame_function):

    df = polars_frame_function(
        data={
            'DON_ABO': [
                'A  ',
                'A1 ',
                'A2 ',
                'AB ',
                'A1B',
                'A2B',
                'B  ',
                'O  ',
            ],
        },
        schema={'DON_ABO': pl.String},
    )
    expected = polars_frame_function(
        data={
            'DON_ABO': [
                'A',
                'A',
                'A',
                'AB',
                'AB',
                'AB',
                'B',
                'O',
            ],
        },
        schema={'DON_ABO': pl.Enum(('A', 'B', 'AB', 'O'))},
    )
    df = df.with_columns(
        collapse_abo(pl.col('DON_ABO'))
        .alias('DON_ABO'),
    )

    assert_frame_equal(df, expected)


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_haversine_NM(polars_frame_function):

    df = polars_frame_function(
        data={
            'can_lat': [42],
            'can_lon': [-92],
            'don_lat': [36],
            'don_lon': [-100],
        },
    )

    expected = polars_frame_function(
        data={
            'can_lat': [42],
            'can_lon': [-92],
            'don_lat': [36],
            'don_lon': [-100],
            'distance_NM': [518.31],
        },
    )

    df = (
        df
        .with_columns(
            haversine_NM(
                pl.col('can_lat'),
                pl.col('can_lon'),
                pl.col('don_lat'),
                pl.col('don_lon'),
            ).alias('distance_NM'),
        )
    )

    assert_frame_equal(df, expected, rel_tol=0.01)
