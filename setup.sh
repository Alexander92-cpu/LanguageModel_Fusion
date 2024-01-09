#!/usr/bin/env bash

if [ ! -d kenlm ]; then
    git clone https://github.com/kpu/kenlm.git
fi

if [ ! -d kenlm/build ]; then
    mkdir kenlm/build
    cd kenlm/build
    cmake ..
    make -j 4
fi