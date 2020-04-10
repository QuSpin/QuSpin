#!/bin/bash


cd sphinx
make html
cd ../
cp -r sphinx/_build/html/* docs/


