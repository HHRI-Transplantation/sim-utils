from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from sim_utils import abo_compatibility
from sim_utils import age_at_listing
from sim_utils import collapse_abo
from sim_utils import haversine_NM


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_abo_compatibility(polars_frame_function):

    data = (
        # don_abo, can_abo, abo_compatibility
        ('A  ', 'A  ', 'Identical'),
        ('A  ', 'A1 ', 'Identical'),
        ('A  ', 'A2 ', 'Identical'),
        ('A  ', 'AB ', 'Compatible'),
        ('A  ', 'A1B', 'Compatible'),
        ('A  ', 'A2B', 'Compatible'),
        ('A  ', 'O  ', 'Incompatible'),
        ('A  ', 'B  ', 'Incompatible'),
        ('A1 ', 'A  ', 'Identical'),
        ('A1 ', 'A1 ', 'Identical'),
        ('A1 ', 'A2 ', 'Identical'),
        ('A1 ', 'AB ', 'Compatible'),
        ('A1 ', 'A1B', 'Compatible'),
        ('A1 ', 'A2B', 'Compatible'),
        ('A1 ', 'O  ', 'Incompatible'),
        ('A1 ', 'B  ', 'Incompatible'),
        ('A2 ', 'A  ', 'Identical'),
        ('A2 ', 'A1 ', 'Identical'),
        ('A2 ', 'A2 ', 'Identical'),
        ('A2 ', 'AB ', 'Compatible'),
        ('A2 ', 'A1B', 'Compatible'),
        ('A2 ', 'A2B', 'Compatible'),
        ('A2 ', 'O  ', 'Incompatible'),
        ('A2 ', 'B  ', 'Incompatible'),
        ('AB ', 'A  ', 'Incompatible'),
        ('AB ', 'A1 ', 'Incompatible'),
        ('AB ', 'A2 ', 'Incompatible'),
        ('AB ', 'AB ', 'Identical'),
        ('AB ', 'A1B', 'Identical'),
        ('AB ', 'A2B', 'Identical'),
        ('AB ', 'O  ', 'Incompatible'),
        ('AB ', 'B  ', 'Incompatible'),
        ('A1B', 'A  ', 'Incompatible'),
        ('A1B', 'A1 ', 'Incompatible'),
        ('A1B', 'A2 ', 'Incompatible'),
        ('A1B', 'AB ', 'Identical'),
        ('A1B', 'A1B', 'Identical'),
        ('A1B', 'A2B', 'Identical'),
        ('A1B', 'O  ', 'Incompatible'),
        ('A1B', 'B  ', 'Incompatible'),
        ('A2B', 'A  ', 'Incompatible'),
        ('A2B', 'A1 ', 'Incompatible'),
        ('A2B', 'A2 ', 'Incompatible'),
        ('A2B', 'AB ', 'Identical'),
        ('A2B', 'A1B', 'Identical'),
        ('A2B', 'A2B', 'Identical'),
        ('A2B', 'O  ', 'Incompatible'),
        ('A2B', 'B  ', 'Incompatible'),
        ('O  ', 'A  ', 'Compatible'),
        ('O  ', 'A1 ', 'Compatible'),
        ('O  ', 'A2 ', 'Compatible'),
        ('O  ', 'AB ', 'Compatible'),
        ('O  ', 'A1B', 'Compatible'),
        ('O  ', 'A2B', 'Compatible'),
        ('O  ', 'O  ', 'Identical'),
        ('O  ', 'B  ', 'Compatible'),
        ('B  ', 'A  ', 'Incompatible'),
        ('B  ', 'A1 ', 'Incompatible'),
        ('B  ', 'A2 ', 'Incompatible'),
        ('B  ', 'AB ', 'Compatible'),
        ('B  ', 'A1B', 'Compatible'),
        ('B  ', 'A2B', 'Compatible'),
        ('B  ', 'O  ', 'Incompatible'),
        ('B  ', 'B  ', 'Identical'),
    )

    expected = polars_frame_function(
        data=data,
        orient='row',
        schema={
            'don_abo': pl.String,
            'can_abo': pl.String,
            'abo_compatibility': pl.Enum((
                'Identical',
                'Compatible',
                'Incompatible',
            )),
        },
    )

    frame = expected.drop('abo_compatibility')
    frame = (
        frame
        .with_columns(
            abo_compatibility(
                pl.col('can_abo'),
                pl.col('don_abo'),
            )
            .alias('abo_compatibility'),
        )
    )
    assert_frame_equal(frame, expected)


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_collapse_abo(polars_frame_function):

    frame = polars_frame_function(
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
    frame = frame.with_columns(
        collapse_abo(pl.col('DON_ABO'))
        .alias('DON_ABO'),
    )

    assert_frame_equal(frame, expected)


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_haversine_NM(polars_frame_function):

    frame = polars_frame_function(
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

    frame = (
        frame
        .with_columns(
            haversine_NM(
                pl.col('can_lat'),
                pl.col('can_lon'),
                pl.col('don_lat'),
                pl.col('don_lon'),
            ).alias('distance_NM'),
        )
    )

    assert_frame_equal(frame, expected, rel_tol=0.01)


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_age_at_listing(polars_frame_function):

    frame = polars_frame_function(
        data={
            'CAN_AGE_IN_MONTHS_AT_LISTING': [
                12*0, 12*18 - 1, 12*18, 12*40, 12*72, None,
            ],
        },
        schema={
            'CAN_AGE_IN_MONTHS_AT_LISTING': pl.Int64,
        },
    )
    expected = polars_frame_function(
        data={
            'CAN_AGE_IN_MONTHS_AT_LISTING': [
                12*0,
                12*18 - 1,
                12*18,
                12*40,
                12*72,
                None,
            ],
            'Age at Listing': [
                'Pediatric',
                'Pediatric',
                'Adult',
                'Adult',
                'Adult',
                'Adult',
            ],
        },
        schema={
            'CAN_AGE_IN_MONTHS_AT_LISTING': pl.Int64,
            'Age at Listing': pl.Enum(('Pediatric', 'Adult')),
        },
    )

    frame = frame.with_columns(age_at_listing().alias('Age at Listing'))
    assert_frame_equal(frame, expected)
