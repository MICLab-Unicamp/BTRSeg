#!/bin/bash
git clone https://github.com/MICLab-Unicamp/BTRSeg
cp -r BTRSeg/* .
rm -rfd BTRSeg
mv data.zip data
python3 -m pip install dlpt_installer/DLPT*.whl
jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --no-browser