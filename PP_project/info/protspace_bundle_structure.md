# ProtSpace Bundle Structure Explanation

This bundle contains multiple logical tables and a settings object. It
does **not** contain raw embeddings or models---only the
already-computed coordinates and metadata needed for visualization.

## 1. selected_annotations

**Rows:** 1428\
**Columns:** 15

This table contains metadata for each protein / point.

Example columns: - protein_id -- unique identifier used across tables -
major_group -- high-level classification - group -- sub-classification
used for coloring in the UI - sub_group -- optional finer grouping -
membran_prediction -- prediction about membrane association - seq_start
-- sequence start information - number_cysteines -- number of cysteine
residues - data_souNumber of bundle parts: 4

============================================================
selected_annotations
rows: 1428
cols: 15
schema:
protein_id: string
major_group: string
group: string
sub_group: string
membran_prediction: string
seq_start: string
number_cysteines: string
data_source: string
species: string
protein_name: string
gene_name: string
ec: string
keyword: string
protein_families: string
uniprot_kb_id: string
-- schema metadata --
pandas: '{"index_columns": [{"kind": "range", "name": null, "start": 0, "' + 2072

preview:
   protein_id  major_group        group sub_group membran_prediction  \
0  1217Boiga_  Plesiotypic  Plesiotypic      <NA>           Secreted   
1  1221Boiga_  Plesiotypic  Plesiotypic      <NA>           Secreted   
2  1223Boiga_  Plesiotypic  Plesiotypic      <NA>           Secreted   
3  1225Boiga_  Plesiotypic  Plesiotypic      <NA>           Secreted   
4  1227Boiga_  Plesiotypic  Plesiotypic      <NA>           Secreted   

  seq_start number_cysteines data_source                        species  \
0      <NA>               10        <NA>  Boiga_dendrophila_dendrophila   
1      <NA>               10        <NA>  Boiga_dendrophila_dendrophila   
2      <NA>               10        <NA>  Boiga_dendrophila_gemmicincta   
3      <NA>               10        <NA>  Boiga_dendrophila_gemmicincta   
4      <NA>               10        <NA>  Boiga_dendrophila_gemmicincta   

  protein_name gene_name ec keyword protein_families uniprot_kb_id  
0                                                                   
1                                                                   
2                                                                   
3                                                                   
4                                                                   

============================================================
projections_metadata
rows: 2
cols: 3
schema:
projection_name: string
dimensions: int64
info_json: string
-- schema metadata --
pandas: '{"index_columns": [{"kind": "range", "name": null, "start": 0, "' + 624

preview:
  projection_name  dimensions  \
0          UMAP_2           2   
1           PCA_2           2   

                                           info_json  
0  {"n_components": 2, "n_neighbors": 50, "min_di...  
1  {"n_components": 2, "svd_solver": "arpack", "e...  

============================================================
projections_data
rows: 2856
cols: 5
schema:
projection_name: string
identifier: string
x: float
y: float
z: null
-- schema metadata --
pandas: '{"index_columns": [{"kind": "range", "name": null, "start": 0, "' + 818

preview:
  projection_name  identifier         x         y     z
0          UMAP_2  1217Boiga_  3.258224  9.650118  None
1          UMAP_2  1221Boiga_  2.979928  9.721151  None
2          UMAP_2  1223Boiga_  3.388507  9.573641  None
3          UMAP_2  1225Boiga_  3.385113  9.440028  None
4          UMAP_2  1227Boiga_  3.127588  9.303226  None

============================================================
settings
top-level keys: ['group', 'major_group']

{
  "group": {
    "includeShapes": false,
    "shapeSize": 30,
    "sortMode": "size-desc",
    "hiddenValues": [],
    "enableDuplicateStackUI": false,
    "selectedPaletteId": "kellys",
    "maxVisibleValues": 6,
    "categories": {
      "Short-chain": {
        "zOrder": 0,
        "color": "#63CBE5",
        "shape": "circle"
      },
      "Plesiotypic": {
        "zOrder": 1,
        "color": "#67BD45",
        "shape": "circle"
      },
      "Non-standard": {
        "zOrder": 2,
        "color": "#788E42",
        "shape": "circle"
      },
      "Long-chain": {
        "zOrder": 3,
        "color": "#24638F",
        "shape": "circle"
      },
      "LYPD2": {
        "zOrder": 4,
        "color": "#3D5BA9",
        "shape": "circle"
      },
      "PSCA": {
        "zOrder": 5,
        "color": "#968AC2",
        "shape": "circle"
      },
      "Reptilian Ly6 group 1": {
        "zOrder": 6,
        "color": "#BD522A",
        "shape": "circle"
      },
      "SLURP1": {
        "zOrder": 7,
        "color": "#488BCA",
        "shape": "circle"
      },
      "Ly6E": {
        "zOrder": 8,
        "color": "#ED2024",
        "shape": "circle"
      },
      "Ly6K": {
        "zOrder": 9,
        "color": "#ECB9B9",
        "shape": "circle"
      },
      "Reptilian Ly6 group 4": {
        "zOrder": 10,
        "color": "#F47745",
        "shape": "circle"
      },
      "frog Ly6": {
        "zOrder": 11,
        "color": "#00A79D",
        "shape": "circle"
      },
      "Reptilian Ly6 group 3": {
        "zOrder": 12,
        "color": "#F7921E",
        "shape": "circle"
      },
      "Ly6H": {
        "zOrder": 13,
        "color": "#9A4C9D",
        "shape": "circle"
      },
      "Reptilian Ly6 group 2": {
        "zOrder": 14,
        "color": "#F8AC4F",
        "shape": "circle"
      },
      "Reptilian Ly6 group 1-2": {
        "zOrder": 15,
        "color": "#BD522A",
        "shape": "circle"
      },
      "GPIHBP1": {
        "zOrder":rce -- dataset source - species -- organism name -
protein_name -- protein description - gene_name -- gene name - ec --
enzyme commission number - keyword -- annotation keywords -
protein_families -- family classification - uniprot_kb_id -- UniProt
identifier

This table is essentially **metadata used for filtering, labeling, and
coloring points in the visualization.**

The key linking field is:

protein_id

------------------------------------------------------------------------

## 2. projections_metadata

**Rows:** 2

Defines which dimensionality reduction projections exist in the bundle.

Columns: - projection_name -- name of projection (e.g., UMAP_2, PCA_2) -
dimensions -- number of dimensions - info_json -- parameters used to
generate the projection

Example:

UMAP_2 → 2D UMAP projection\
PCA_2 → 2D PCA projection

The JSON field stores parameters such as: - n_neighbors - min_dist -
svd_solver - explained variance

------------------------------------------------------------------------

## 3. projections_data

**Rows:** 2856

This is the **actual embedding coordinates used for plotting.**

Columns: - projection_name -- which projection this row belongs to -
identifier -- matches protein_id - x -- x coordinate - y -- y
coordinate - z -- optional third dimension (None for 2D)

Example:

UMAP_2 \| 1217Boiga\_ \| 3.25 \| 9.65

Because there are 1428 proteins and **two projections**, the table
contains:

1428 × 2 = 2856 rows

So yes:

**The UMAP coordinates are already stored here.**\
No UMAP computation happens in the viewer.

The viewer simply: 1. Reads this table 2. Filters rows for
projection_name == "UMAP_2" 3. Plots (x, y)

------------------------------------------------------------------------

## 4. settings

A JSON configuration used by the UI.

Example top-level keys: - group - major_group

Each key defines how a metadata column should be visualized.

Example configuration:

-   color palette
-   shape settings
-   category colors
-   z-order for rendering
-   hidden categories

Example:

"Plesiotypic": { "color": "#67BD45", "shape": "circle" }

This controls how the category appears in the scatter plot.

------------------------------------------------------------------------

# Overall Data Model

Protein metadata → selected_annotations

Projection definitions → projections_metadata

Coordinates for visualization → projections_data

UI display settings → settings

------------------------------------------------------------------------

# Key Insight

The bundle already contains **fully computed UMAP/PCA coordinates**.

The ProtSpace viewer is only a **visualization layer**, not a
computation engine.
