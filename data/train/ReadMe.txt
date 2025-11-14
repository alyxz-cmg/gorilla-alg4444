------------------------------------------
GENERAL INFORMATION
------------------------------------------
Challenge website: https://monkey.grand-challenge.org/

Tutorials: https://github.com/computationalpathologygroup/monkey-challenge

------------------------------------------
DATA INFORMATION
------------------------------------------
- annotations: 
Dot annotations of the monocytes and lymphocytes (collectively inflammatory cells) in the grand-challenge json format as well as in xml format (which can be loaded in ASAP).

- images/tissue-masks: 
Tissue masks for the region (polygon) of interests as a binary tif file.

- images/pas-cpg:
Scanned tif of the PAS WSI using the CPG scanning profile at Radboud. The annotations are made on these files, and all other tif images are registered to them.

- images/pas-diagnostic:
Scanned tif of the PAS WSI using the Diagnostic scanning profile at Radboud.

- images/pas-original:
Tif of the PAS WSI scanned at the center of origin.

- images/ihc: 
Will be uploaded asap. Contains the IHC double stains for the monocytes (red) and lymphocytes (brown).

- context-information.xlsx: 
Overview of the data quality, naming, diagnosis and other information (see sheet "explanations").


------------------------------------------
FOLDER STRUCTURE
------------------------------------------
├── ReadMe.txt
├── annotations/
│   ├── json/
│   │   ├── A_P000001_inflammatory-cells.json
│   │   ├── A_P000001_monocytes.json
│   │   ├── A_P000001_lymphocytes.json
│   │   ├── A_P000002_inflammatory-cells.json
│   │   └── (...).json
│   └── xml/
│       ├── A_P000001.xml
│       ├── A_P000002.xml
│       └── (...).xml
├── images/
│   ├── tissue-masks/
│   │   ├── A_P000001_tissue-mask.tif
│   │   ├── A_P000002_tissue-mask.tif
│   │   └── (...).tif
│   ├── pas-cpg/
│   │   ├── A_P000001_PAS_CPG.tif
│   │   ├── A_P000001_PAS_CPG.tif
│   │   └── (...).tif
│   ├── pas-diagnostic/
│   │   ├── A_P000001_PAS_Diagnostic.tif
│   │   ├── A_P000001_PAS_Diagnostic.tif
│   │   └── (...).tif
│   ├── pas-original/
│   │   ├── A_P000001_PAS_Original.tif
│   │   ├── A_P000001_PAS_Original.tif
│   │   └── (...).tif
│   └── ihc/
│       ├── A_P000001_IHC_CPG.tif
│       ├── A_P000001_IHC_CPG.tif
│       └── (...).tif
└── metadata/
    └── context-information.xlsx