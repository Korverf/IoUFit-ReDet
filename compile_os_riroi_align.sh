#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

echo "Building os riroi align op..."
cd mmdet/ops/os_riroi_align
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace