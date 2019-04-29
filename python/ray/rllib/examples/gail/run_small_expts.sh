#!/bin/bash
mkdir -p small-expts/
for batch_size in 128 512; do
    for hiddens in 128,128 400,300; do
        export CUDA_VISIBLE_DEVICES=""
        for type in single dist; do
        for cores in 1 2 4 8 16 32; do
            python dist_vs_single.py --expt ${type} --cores ${cores} --csv-out small-expts/all.csv \
                | tee small-expts/run-${type}-cores${cores}-net${hiddens}-batch${batch_size}.txt \
                || break
        done
        done
        export CUDA_VISIBLE_DEVICES="0"
        python dist_vs_single.py --expt single --cores 4 --csv-out small-expts/all.csv \
            | tee small-expts/run-single-gpu-net${hiddens}-batch${batch_size}.txt \
            || break
    done
done
