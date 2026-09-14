# Measured reflectance spectra

35 material samples from the [USGS Spectral Library Version 7](https://www.usgs.gov/data/usgs-spectral-library-version-7-data), by Kokaly, Clark, Swayze, Livo, Hoefen, Pearson, Wise, Benzel, Lowers, Driscoll, and Klein (2017). The data release is marked [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/).

Use **Spectrum > Load SPD** to open a file. For a diffuse/base material, select the material and **Scattering (base / diffuse)**, then click **Assign**. Subsequent edits update the assigned source. Use **Save SPD** outside the built-in spectrum directory to save an edited version.

## Data preparation

- Source: [native measured ASCII spectra, splib07a](https://www.sciencebase.gov/catalog/item/586e8c88e4b0f5ce109fccae), archive `ASCIIdata_splib07a.zip`.
- Each SPD contains 441 original ASD channels spanning nominally 390-830 nm. Channels outside this range are omitted.
- Wavelengths are converted from micrometers to nanometers using decimal arithmetic; the source's numerical wavelength precision is retained.
- Reflectance numbers are copied verbatim. No resampling, smoothing, normalization, clipping, or simplification is applied. All imported values are finite and within 0-1, with no deleted/bad channels in this interval.
- The renderer interpolates these samples onto its wavelength grid. The original measurement resolution is distinct from the channel spacing.
- These are measured sample reflectances, not complete angular BSDF measurements. Source filenames and record identifiers are embedded in every SPD.
- Wavelength file: `ASCIIdata_splib07a/splib07a_Wavelengths_ASD_0.35-2.5_microns_2151_ch.txt`.
- Downloaded: 2026-09-14.
- Archive SHA-256: `d232645740869a82aafcad5839448c50b1dc72965ce042d1374f29b7a798a91c`.

## Samples

| File | Measured sample | Original archive entry |
| --- | --- | --- |
| [asphalt_old_road.spd](asphalt_old_road.spd) | Old black road asphalt | `ASCIIdata_splib07a/ChapterA_ArtificialMaterials/splib07a_Asphalt_GDS376_Blck_Road_old_ASDFRa_AREF.txt` |
| [brick_red_paving.spd](brick_red_paving.spd) | Red paving brick | `ASCIIdata_splib07a/ChapterA_ArtificialMaterials/splib07a_Brick_GDS349_Paving_Red_ASDFRa_AREF.txt` |
| [brick_tan_paving.spd](brick_tan_paving.spd) | Tan paving brick | `ASCIIdata_splib07a/ChapterA_ArtificialMaterials/splib07a_Brick_GDS347_Paving_Tan_ASDFRa_AREF.txt` |
| [concrete_light_gray.spd](concrete_light_gray.spd) | Light gray road concrete | `ASCIIdata_splib07a/ChapterA_ArtificialMaterials/splib07a_Concrete_GDS375_Lt_Gry_Road_ASDFRa_AREF.txt` |
| [pine_wood.spd](pine_wood.spd) | New pine wood beam | `ASCIIdata_splib07a/ChapterA_ArtificialMaterials/splib07a_Wood_Beam_GDS363_Nw_Pine_2X4_ASDFRa_AREF.txt` |
| [cotton_white.spd](cotton_white.spd) | White cotton fabric | `ASCIIdata_splib07a/ChapterA_ArtificialMaterials/splib07a_Cotton_Fabric_GDS437_White_ASDFRa_AREF.txt` |
| [nylon_red.spd](nylon_red.spd) | Red nylon fabric | `ASCIIdata_splib07a/ChapterA_ArtificialMaterials/splib07a_Nylon_Fabric_GDS431_Red_RpSt_ASDFRa_AREF.txt` |
| [nylon_green.spd](nylon_green.spd) | Green nylon fabric | `ASCIIdata_splib07a/ChapterA_ArtificialMaterials/splib07a_Nylon_Fabric_GDS432_Grn_RpSt_ASDFRa_AREF.txt` |
| [nylon_blue.spd](nylon_blue.spd) | Blue nylon fabric | `ASCIIdata_splib07a/ChapterA_ArtificialMaterials/splib07a_Nylon_Fabric_GDS433_Blu_RpSt_ASDFRa_AREF.txt` |
| [painted_aluminum_gray.spd](painted_aluminum_gray.spd) | Light gray painted aluminum | `ASCIIdata_splib07a/ChapterA_ArtificialMaterials/splib07a_Painted_Aluminum_GDS333_LgGr_ASDFRa_AREF.txt` |
| [oak_leaf_fresh.spd](oak_leaf_fresh.spd) | Fresh oak leaf | `ASCIIdata_splib07a/ChapterV_Vegetation/splib07a_Oak_Oak-Leaf-1_fresh_ASDFRa_AREF.txt` |
| [oak_leaf_dried.spd](oak_leaf_dried.spd) | Dried oak leaf | `ASCIIdata_splib07a/ChapterV_Vegetation/splib07a_Oak_Oak-Leaf-2_dried_ASDFRa_AREF.txt` |
| [grass_dry_golden.spd](grass_dry_golden.spd) | Golden dry grass | `ASCIIdata_splib07a/ChapterV_Vegetation/splib07a_Grass_Golden_Dry_GDS480_ASDFRa_AREF.txt` |
| [sagebrush_leaves_dry.spd](sagebrush_leaves_dry.spd) | Dry sagebrush leaves | `ASCIIdata_splib07a/ChapterV_Vegetation/splib07a_Sagebrush_Sage-Leaves-1_dry_ASDFRa_AREF.txt` |
| [hematite.spd](hematite.spd) | Hematite mineral | `ASCIIdata_splib07a/ChapterM_Minerals/splib07a_Hematite_HS45.3_ASDFRb_AREF.txt` |
| [goethite.spd](goethite.spd) | Goethite mineral | `ASCIIdata_splib07a/ChapterM_Minerals/splib07a_Goethite_GDS134_ASDFRb_AREF.txt` |
| [malachite.spd](malachite.spd) | Malachite mineral | `ASCIIdata_splib07a/ChapterM_Minerals/splib07a_Malachite_HS254.3B_ASDFRb_AREF.txt` |
| [gypsum_selenite.spd](gypsum_selenite.spd) | Gypsum selenite | `ASCIIdata_splib07a/ChapterM_Minerals/splib07a_Gypsum_HS333.3B_(Selenite)_ASDFRa_AREF.txt` |
| [flower_geranium_red_orange.spd](flower_geranium_red_orange.spd) | Red-orange geranium flower | `ASCIIdata_splib07a/ChapterV_Vegetation/splib07a_Flower_Geranium-1_Red-Orange_ASDFRa_AREF.txt` |
| [flower_pansy_yellow.spd](flower_pansy_yellow.spd) | Yellow pansy flower | `ASCIIdata_splib07a/ChapterV_Vegetation/splib07a_Flower_Pansy-1_Yellow_ASDFRa_AREF.txt` |
| [flower_petunia_pink.spd](flower_petunia_pink.spd) | Pink petunia flower | `ASCIIdata_splib07a/ChapterV_Vegetation/splib07a_Flower_Petunia-1_Pink_ASDFRa_AREF.txt` |
| [flower_petunia_purple.spd](flower_petunia_purple.spd) | Purple petunia flower | `ASCIIdata_splib07a/ChapterV_Vegetation/splib07a_Flower_Petunia-2_Purple_ASDFRa_AREF.txt` |
| [flower_petunia_white.spd](flower_petunia_white.spd) | White petunia flower | `ASCIIdata_splib07a/ChapterV_Vegetation/splib07a_Flower_Petunia-3_White_ASDFRa_AREF.txt` |
| [flower_platycodon_purple.spd](flower_platycodon_purple.spd) | Purple platycodon flower | `ASCIIdata_splib07a/ChapterV_Vegetation/splib07a_Flower_Platycodon-1_Purple_ASDFRa_AREF.txt` |
| [willow_leaves_dry.spd](willow_leaves_dry.spd) | Dry willow leaves | `ASCIIdata_splib07a/ChapterV_Vegetation/splib07a_Willow_Willow-Leaves-1_dry_ASDFRa_AREF.txt` |
| [spruce_needles.spd](spruce_needles.spd) | Engelmann spruce needles | `ASCIIdata_splib07a/ChapterV_Vegetation/splib07a_Engelmann-Spruce_ES-Needls-1_ASDFRa_AREF.txt` |
| [lodgepole_pine_needles.spd](lodgepole_pine_needles.spd) | Lodgepole pine needles | `ASCIIdata_splib07a/ChapterV_Vegetation/splib07a_Lodgepole-Pine_LP-Needles-1_ASDFRa_AREF.txt` |
| [lichen_acarospora.spd](lichen_acarospora.spd) | Acarospora lichen | `ASCIIdata_splib07a/ChapterV_Vegetation/splib07a_Lichen_Acarospora-1_ASDFRb_AREF.txt` |
| [lichen_xanthoparmelia.spd](lichen_xanthoparmelia.spd) | Xanthoparmelia lichen | `ASCIIdata_splib07a/ChapterV_Vegetation/splib07a_Lichen_Xanthoparmelia-1_ASDFRb_AREF.txt` |
| [cedar_fresh.spd](cedar_fresh.spd) | Fresh cedar shake | `ASCIIdata_splib07a/ChapterA_ArtificialMaterials/splib07a_Cedar_Shake_GDS357_Fresh_ASDFRa_AREF.txt` |
| [cedar_weathered.spd](cedar_weathered.spd) | Highly weathered cedar shake | `ASCIIdata_splib07a/ChapterA_ArtificialMaterials/splib07a_Cedar_Shake_GDS361_HiWeather_ASDFRa_AREF.txt` |
| [cedar_weathered_moss.spd](cedar_weathered_moss.spd) | Highly weathered cedar shake with moss | `ASCIIdata_splib07a/ChapterA_ArtificialMaterials/splib07a_Cedar_Shake_GDS360_H_Weamoss_ASDFRa_AREF.txt` |
| [plywood_pine.spd](plywood_pine.spd) | Fresh pine plywood | `ASCIIdata_splib07a/ChapterA_ArtificialMaterials/splib07a_Plywood_GDS365_Fresh_Pine_ASDFRa_AREF.txt` |
| [burlap_brown.spd](burlap_brown.spd) | Brown burlap fabric | `ASCIIdata_splib07a/ChapterA_ArtificialMaterials/splib07a_Burlap_Fabric_GDS430_Brown_ASDFRa_AREF.txt` |
| [cardboard_brown.spd](cardboard_brown.spd) | Brown corrugated cardboard | `ASCIIdata_splib07a/ChapterA_ArtificialMaterials/splib07a_Cardboard_GDS371_Brn_Corgted_ASDFRa_AREF.txt` |
