"""The convention of the maps of the European grid, one for all studies.

The maps plot longitude against latitude, with one unit of latitude drawn EUROPE_STRETCH times
as long as one unit of longitude. True shapes at 52 degrees north would need 1/cos(52 deg) =
1.62; the maps are flattened to 1.25 so that the continent fits the width of a text column, and
every map uses the same value, so that the continent has one shape in all figures. A script
takes it with

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))    # the studies folder
    from shared.maps import EUROPE_STRETCH
"""

EUROPE_STRETCH = 1.25
