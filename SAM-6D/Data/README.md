
## Data Structure
Our data structure in `Data` folder is constructed as follows: 
```
Data
├── MegaPose-Training-Data
    ├── MegaPose-GSO
        ├──google_scanned_objects
        ├──templates
        └──train_pbr_web
    ├── MegaPose-ShapeNetCore
        ├──shapenetcorev2
        ├──templates
        └──train_pbr_web
├── BOP   # https://bop.felk.cvut.cz/datasets/
    ├──tudl
    ├──lmo
    ├──ycbv
    ├──icbin
    ├──hb
    ├──itodd
    └──tless
└── BOP-Templates
    ├──tudl
    ├──lmo
    ├──ycbv
    ├──icbin
    ├──hb
    ├──itodd
    └──tless
```


## Data Download

### Training Datasets
For training the Pose Estimation Model, you may download the rendered images of [c](https://github.com/thodan/bop_toolkit/blob/master/docs/bop_challenge_2023_training_datasets.md) provided by BOP official in the respective `MegaPose-Training-Data/MegaPose-GSO/train_pbr_web` and `MegaPose-Training-Data/MegaPose-ShapeNetCore/train_pbr_web` folders. 

We use the [pre-processed object models](https://www.paris.inria.fr/archive_ylabbeprojectsdata/megapose/tars/) of the two datasets, provided by [MegePose](https://github.com/megapose6d/megapose6d), and download them in the `MegaPose-Training-Data/MegaPose-GSO/google_scanned_objects` and `MegaPose-Training-Data/MegaPose-ShapeNetCore/shapenetcorev2` folders, respectively.


### BOP Datasets
To evaluate our SAM-6D on BOP datasets, you may download the test data and the object CAD models of the seven core datasets from [BOP official](https://bop.felk.cvut.cz/datasets/). For each dataset, the structure could be constructed as follows:

```
BOP
├── lmo
    ├──models           # object CAD models 
    ├──test             # bop19 test set
    ├──(train_pbr)      # maybe used in instance segmentation
    ...
...
```

You may also download the `train_pbr` data of the datasets for template selection in the Instance Segmentation Model following [CNOS](https://github.com/nv-nguyen/cnos?tab=readme-ov-file).



## Template Rendering

### Requirements

* blenderproc
* trimesh
* numpy 
* cv2


### Template Rendering of Training Objects

We generate two-view templates for each training object via [Blenderproc](https://github.com/DLR-RM/BlenderProc). You may run the following commands to render the templates for `MegaPose-GSO` dataset:

```
cd ../Render/
blenderproc run render_gso_templates.py
```
and the commands for `shapenetcorev2` dataset:

```
cd ../Render/
blenderproc run render_shapenet_templates.py
```

### Template Rendering of Test Objects
We generate 42 templates for each test object following the [CNOS](https://github.com/nv-nguyen/cnos?tab=readme-ov-file) via [Blenderproc](https://github.com/DLR-RM/BlenderProc). 


#### Custom Objects

For a custom object, you can run the following commands to render the templates:
```
cd ../Render/
blenderproc run render_custom_templates.py --cad_path $CAD_PATH --output_dir $OUTPUT_DIR
```
The string "CAD_PATH" is the path to your CAD and the string "OUTPUT_DIR" is the path to save templates.


#### BOP Datasets
You may run the following commands to render the tempates for the objects in BOP datasets:
```
cd ../Render/
blenderproc run render_bop_templates.py --dataset_name $DATASET
```
The string "DATASET" could be set as `lmo`, `icbin`, `itodd`, `hb`, `tless`, `tudl` or `ycbv`. We also offer downloadable rendered templates [[link](https://drive.google.com/drive/folders/1fXt5Z6YDPZTJICZcywBUhu5rWnPvYAPI?usp=sharing)].


## Acknowledgements
- [MegaPose](https://github.com/megapose6d/megapose6d)
- [CNOS](https://github.com/nv-nguyen/cnos?tab=readme-ov-file)
