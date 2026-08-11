# Data provenance

The numerical tensors were derived from Sentinel-1 radar and Sentinel-2
Level-2A optical observations. Forest strata were assigned from a 20 m Taiwan
forest-type product. Retrospective event labels were rasterized from public
event-inventory polygons after temporal and spatial auditing.

The release excludes source imagery, source rasters, and source polygons. It
contains only model-ready numerical features, split identifiers, predictions,
quality masks, rasterized analytical labels, and reduced cartographic context.
WGS84 coordinates in `study_windows.csv` identify analysis-window centres.

Natural Earth coastlines are public-domain cartographic context. Sentinel data
remain subject to the Copernicus data terms. Users who reconstruct the upstream
pipeline should obtain source products directly from their providers.

