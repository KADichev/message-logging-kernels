#!/bin/bash

make
mv jacobi jacobi.std
mv lulesh lulesh.std
mv cg cg.std

sed -i "s/\/\/#define SCALE_FREQ/#define SCALE_FREQ/g" mammut_config.h
make
mv jacobi jacobi.fs
mv lulesh lulesh.fs
mv cg cg.fs
sed -i "s/#define SCALE_FREQ/\/\/#define SCALE_FREQ/g" mammut_config.h


sed -i "s/\/\/#define SCALE_MOD/#define SCALE_MOD/g" mammut_config.h
make
mv jacobi jacobi.ms
mv lulesh lulesh.ms
mv cg cg.ms
sed -i "s/#define SCALE_MOD/\/\/#define SCALE_MOD/g" mammut_config.h



sed -i "s/#define LOG_BFR_DEPTH 75/#define LOG_BFR_DEPTH 0/g" mammut_config.h
make
mv jacobi jacobi.global
mv lulesh lulesh.global
mv cg cg.global
sed -i "s/#define LOG_BFR_DEPTH 0/#define LOG_BFR_DEPTH 75/g" mammut_config.h
