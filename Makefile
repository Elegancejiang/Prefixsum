all:
	nvcc -std=c++11 -gencode arch=compute_86,code=sm_86 -O3  prefixsum.cu -o  prefixsum  --expt-relaxed-constexpr -w