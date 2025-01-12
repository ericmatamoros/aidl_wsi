# mil_wsi

This repository contains the required functions & classes for the project: mil_wsi

## Set up virtual environment & install dependencies

Download the project file.

Next, use the environment configuration file to create a conda environment:
```
conda env create -f env.yml
```

Activate the environment:
```
conda activate wsi
```

Deactivate the environment:
```
conda deactivate wsi
```

# Data

Images to be processed should be placed in the directoryF: aidl_wsi/data
Our pipeline currently supports common standard WSI formats including: SVS, NDPI, TIFF.

# Results

Both intermediate & final outputs will be stored in the results/ folder. Please note that some intermediate steps will create the output in specific folders
created inside the results/ one.

# Execute the pipeline

There are some templates already build as VSCode pipelines which can be found in the .vscode/ folder. To execute a given task:

Command + Shift + P -> Run Task -> (Choose step to execute)

The Pipeline task will sequentially run all the steps.

## Configuration Files

Configuration parameters can be found inside the folder mil_wsi/config/, with the config.yaml filename.

## Contributors

- [Eric Matamoros](ericmatamoros1999@gmail.com)
- [Núria Blasco](nuriablasco35@gmail.com)
- [Marcel Maragall](marcelmaragall@gmail.com)
- [Mireia Torres](miretorres.macia@gmail.com)