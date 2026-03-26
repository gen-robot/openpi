# Rosbag Converter
This is the default Rosbag converter from ZBL robot, which is encrypted for commercial use.
We need a separate python environment to run this command as this does not fits the version of the venv in openpi repository.
## Python Environment Setup
```
cd examples/x2robot/rosbag_converter
conda create -n rosbag-converter python=3.10
conda activate rosbag-converter
pip install -r requirements.txt
```
## Usage
```
# this script is not run at the root folder of the repo
cd examples/x2robot/rosbag_converter
python3 batch_process.py source_datasets target_datasets

# optional: you can remove the record folder at the converted dataset
cd target_datasets
rm -r record
```
