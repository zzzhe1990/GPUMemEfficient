#!/bin/bash

METRICS="gld_efficiency,gst_efficiency,shared_efficiency,tex_cache_hit_rate,l2_tex_read_hit_rate"
SPATH='/home/yuanzhe/Desktop/GPUMemEfficient/SmithWaterman/'
cd $SPATH
nvprof --metrics $METRICS --log-file share_nvprof_1080ti_128_32.log ./GPU_Hyperlane_Share 15 15 32 128 1 
nvprof --metrics $METRICS --log-file share_nvprof_1080ti_256_32.log ./GPU_Hyperlane_Share 15 15 32 256 1
SPATH='/home/yuanzhe/Desktop/GPUMemEfficient/EditDistance/'
cd $SPATH
nvprof --metrics $METRICS --log-file share_nvprof_1080ti_128_32.log ./GPU_Hyperlane_Share 15 15 32 128 1
nvprof --metrics $METRICS --log-file share_nvprof_1080ti_256_32.log ./GPU_Hyperlane_Share 15 15 32 256 1
SPATH='/home/yuanzhe/Desktop/GPUMemEfficient/2D-SOR/nested_loop/'
cd $SPATH
nvprof --metrics $METRICS --log-file share_nvprof_1080ti_128_32.log ./GPU_Hyperlane_Share 15 15 32 128 1
nvprof --metrics $METRICS --log-file share_nvprof_1080ti_256_32.log ./GPU_Hyperlane_Share 15 15 32 256 1
SPATH='/home/yuanzhe/Desktop/GPUMemEfficient/SAT/'
cd $SPATH
nvprof --metrics $METRICS --log-file share_nvprof_1080ti_128_32.log ./GPU_Hyperlane_Share 15 15 32 128 1
nvprof --metrics $METRICS --log-file share_nvprof_1080ti_256_32.log ./GPU_Hyperlane_Share 15 15 32 256 1
