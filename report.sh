#!/bin/bash
#
# generate an HTML report of the last training session
#

jupyter nbconvert --execute ../code/report.ipynb --no-input --to html --output-dir ./
