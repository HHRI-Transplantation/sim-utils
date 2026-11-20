from __future__ import annotations

from typing import TypeVar

import polars as pl

PolarsFrame = TypeVar('PolarsFrame', pl.DataFrame, pl.LazyFrame)


def abo_compatibility(
    can_abo: pl.Expr,
    don_abo: pl.Expr,
) -> pl.Expr:

    can_abo_clean = collapse_abo(can_abo)
    don_abo_clean = collapse_abo(don_abo)

    return (
        pl.when(don_abo_clean == can_abo_clean)
        .then(pl.lit('Identical'))
        .when(don_abo_clean == pl.lit('O'))
        .then(pl.lit('Compatible'))
        .when(can_abo_clean == pl.lit('AB'))
        .then(pl.lit('Compatible'))
        .otherwise(pl.lit('Incompatible'))
        .cast(pl.Enum(('Identical', 'Compatible', 'Incompatible')))
    )


def age_at_listing() -> pl.Expr:

    return (
        pl.when(pl.col('CAN_AGE_IN_MONTHS_AT_LISTING') < pl.lit(18*12))
        .then(pl.lit('Pediatric'))
        .otherwise(pl.lit('Adult'))
        .cast(pl.Enum(('Pediatric', 'Adult')))
    )


def collapse_abo(abo: pl.Expr) -> pl.Expr:

    return (
        pl.when(abo.str.strip_chars() == pl.lit('O'))
        .then(pl.lit('O'))
        .when(abo.str.strip_chars().is_in(['A', 'A1', 'A2']))
        .then(pl.lit('A'))
        .when(abo.str.strip_chars() == pl.lit('B'))
        .then(pl.lit('B'))
        .when(abo.str.strip_chars().is_in(['AB', 'A1B', 'A2B']))
        .then(pl.lit('AB'))
        .otherwise(None)
        .cast(pl.Enum(('A', 'B', 'AB', 'O')))
    )


def _degrees_to_radians(degrees: pl.Expr) -> pl.Expr:

    return degrees*(3.14159/180)


def haversine_NM(
    can_lat: pl.Expr,
    can_lon: pl.Expr,
    don_lat: pl.Expr,
    don_lon: pl.Expr,
) -> pl.Expr:
    '''

        Calculates the haversine distance in nautical miles
        between the candidate and donor coordinates. Coordinates
        are assumed to be in degrees.

    '''

    can_lat_r = _degrees_to_radians(can_lat)
    can_lon_r = _degrees_to_radians(can_lon)
    don_lat_r = _degrees_to_radians(don_lat)
    don_lon_r = _degrees_to_radians(don_lon)

    radius_NM = 3443.92

    return (
        2*radius_NM*(
            (don_lat_r - can_lat_r).truediv(2).sin().pow(2) +
            (can_lat_r.cos() * don_lat_r.cos()) *
            (don_lon_r - can_lon_r).truediv(2).sin().pow(2)
        ).sqrt().arcsin()
    )
