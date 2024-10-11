#include <stdio.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
// #include <cub/cub.cuh>
#include <sys/time.h>


struct timeval prefixsum_begin;
struct timeval prefixsum_end;
double prefixsum1_time = 0;
double prefixsum2_time = 0;
double prefixsum3_time = 0;
double prefixsum4_time = 0;
double prefixsum5_time = 0;
double prefixsum6_time = 0;
double prefixsum7_time = 0;
double prefixsum8_time_128 = 0;
double prefixsum8_time_256 = 0;
double prefixsum8_time_512 = 0;
double prefixsum8_time_1024 = 0;
double prefixsum_trust_time = 0;

// GPU memory
size_t max_memory_GPU = 0;
size_t now_memory_GPU = 0;
size_t freeMem, totalMem, usableMem;

void GPU_Memory()
{
	// 获取设备属性
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);

	// 检测当前程序执行前显存已使用的大小
	cudaMemGetInfo(&freeMem, &totalMem);
	usableMem = totalMem - freeMem;

	printf("Total GPU Memory:                   %zu B %zu KB %zu MB %zu GB\n", totalMem, totalMem / 1024,totalMem / 1024 / 1024,totalMem / 1024 / 1024 / 1024);
    printf("Usable GPU Memory before execution: %zu B %zu KB %zu MB %zu GB\n", freeMem, freeMem / 1024,freeMem / 1024 / 1024,freeMem / 1024 / 1024 / 1024);
}

void malloc_GPU(size_t space)
{
    if(now_memory_GPU + space > freeMem)
    {
        printf("-----don't have enough GPU memory   %zu+%zu=%zu>%zu\n",now_memory_GPU,space,now_memory_GPU + space, freeMem);
        return ;
    }
    now_memory_GPU += space;
    if(now_memory_GPU > max_memory_GPU)
        max_memory_GPU = now_memory_GPU;
    // printf("-----now used GPU memory            %zu B\n",now_memory_GPU);
}

void free_GPU(size_t space)
{
    if(now_memory_GPU - space < 0)
    {
        printf("-----too much GPU memory is freed   %zu-%zu=%zu<0\n",now_memory_GPU,space,now_memory_GPU - space);
        return ;
    }
    now_memory_GPU -= space;
    if(now_memory_GPU > max_memory_GPU)
        max_memory_GPU = now_memory_GPU;
    // printf("-----now used GPU memory            %zu B\n",now_memory_GPU);
}

int length, blocksize;

__global__ void init_num(int *num, int length)
{
    int ii  = blockIdx.x * blockDim.x + threadIdx.x;

    if(ii < length) 
        num[ii] = 1;
}

void print_num(int *num, int length)
{
    // for(int i = 0;i < 2048 && i < length;i++)
    // {
    //     printf("%5d ",num[i]);
    //     if(i % 32 == 31) printf("\n");
    // }
    printf("end %d\n",num[length - 1]);
}

__global__ void print_num_device(int *num, int length)
{
    // for(int i = max(length - 1024,0);i < length;i++)
    // {
    //     printf("%5d ",num[i]);
    //     if(i % 32 == 31) printf("\n");
    // }
    printf("%d\n",num[length - 1]);
}

__global__ void firstprefixsum1(int *num, int *firstresult, int *num_out, int length, int blocknum)
{
    int begin = blockIdx.x * blockDim.x;
    int end   = begin + blockDim.x;
    if(end >= length) end = length;

    if(threadIdx.x == 0)
    {
        int prefixsum = 0;
        for(int i = begin;i < end;i++)
        {
            prefixsum += num[i];
            num_out[i] = prefixsum;
        }
        firstresult[blockIdx.x] = prefixsum;
    }
}

__global__ void secondprefixsum1(int *num, int length)
{
    int prefixsum = 0;
    for(int i = 0;i < length;i++)
    {
        prefixsum += num[i];
        num[i] = prefixsum;
    }
}

__global__ void addfirstsum1(int *firstresult, int *num_out, int length, int blocknum)
{
    if(blockIdx.x == 0) return ;

    int ii = blockIdx.x * blockDim.x + threadIdx.x;
    if(ii < length)
        num_out[ii] += firstresult[blockIdx.x - 1];
}

void prefixsum1(int *num, int *num_out, int length)
{
    size_t blocksize = 128;
    size_t blocknum = (length + blocksize - 1) / blocksize;

    int *firstresult;
    malloc_GPU(blocknum * sizeof(int));
    cudaMalloc((void**)&firstresult,blocknum * sizeof(int));

    firstprefixsum1<<<blocknum,blocksize>>>(num,firstresult,num_out,length,blocknum);

    secondprefixsum1<<<1,1>>>(firstresult,blocknum);

    addfirstsum1<<<blocknum,blocksize>>>(firstresult,num_out,length,blocknum);

    free_GPU(blocknum * sizeof(int));
    cudaFree(firstresult);
}

__device__ void prefixsum2_block(int *cache_num)
{
    if(threadIdx.x == 0)
    {
        int prefixsum = 0;
        for(int i = 0;i < blockDim.x;++i)
        {
            prefixsum   += cache_num[i];
            cache_num[i] = prefixsum;
        }
    }

    __syncthreads();
}

__global__ void firstprefixsum2(int *num, int *firstresult, int *num_out, int length, int blocknum)
{
    int ii  = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    extern __shared__ int cache_num[];
    if(ii < length) cache_num[tid] = num[ii];
    else cache_num[tid] = 0;
    __syncthreads();

    prefixsum2_block(cache_num);

    if(ii < length) 
        num_out[ii] = cache_num[tid];
    if(tid == blockDim.x - 1) 
        firstresult[blockIdx.x] = cache_num[tid];
}

// improve firstprefixsum1 from global to share
void prefixsum2(int *num, int *num_out, int length)
{
    size_t blocksize = 128;
    size_t blocknum = (length + blocksize - 1) / blocksize;

    int *firstresult;
    malloc_GPU(blocknum * sizeof(int));
    cudaMalloc((void**)&firstresult,blocknum * sizeof(int));

    firstprefixsum2<<<blocknum,blocksize>>>(num,firstresult,num_out,length,blocknum);

    secondprefixsum1<<<1,1>>>(firstresult,blocknum);

    addfirstsum1<<<blocknum,blocksize>>>(firstresult,num_out,length,blocknum);

    free_GPU(blocknum * sizeof(int));
    cudaFree(firstresult);
}

__device__ void prefixsum3_warp(int *cache_num, int lane)
{
    if(lane == 0)
    {
        int prefixsum = 0;
        for(int i = 0;i < 32;i++)
        {
            prefixsum   += cache_num[i];
            cache_num[i] = prefixsum;
        }
    }
}

__device__ void prefixsum3_block(int *cache_num)
{
    int warp_id = threadIdx.x >> 5;
    int lane    = threadIdx.x & 31;
    __shared__ int warp_sum[32];

    prefixsum3_warp(cache_num + warp_id * 32,lane);
    __syncthreads();

    if(lane == 31)
        warp_sum[warp_id] = cache_num[threadIdx.x];
    __syncthreads();

    if(warp_id == 0)
        prefixsum3_warp(warp_sum,lane);
    __syncthreads();

    if(warp_id > 0)
        cache_num[threadIdx.x] += warp_sum[warp_id - 1];
    __syncthreads();
}

__global__ void firstprefixsum3(int *num, int *firstresult, int *num_out, int length, int blocknum)
{
    int ii  = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    extern __shared__ int cache_num[];
    if(ii < length) cache_num[tid] = num[ii];
    else cache_num[tid] = 0;
    __syncthreads();

    prefixsum3_block(cache_num);

    if(ii < length) 
        num_out[ii] = cache_num[tid];
    if(tid == blockDim.x - 1) 
        firstresult[blockIdx.x] = cache_num[tid];
}

// improve firstprefixsum2 from block to warp
void prefixsum3(int *num, int *num_out, int length)
{
    size_t blocksize = 128;
    size_t blocknum = (length + blocksize - 1) / blocksize;

    int *firstresult;
    malloc_GPU(blocknum * sizeof(int));
    cudaMalloc((void**)&firstresult,blocknum * sizeof(int));

    firstprefixsum3<<<blocknum,blocksize>>>(num,firstresult,num_out,length,blocknum);

    secondprefixsum1<<<1,1>>>(firstresult,blocknum);

    addfirstsum1<<<blocknum,blocksize>>>(firstresult,num_out,length,blocknum);

    free_GPU(blocknum * sizeof(int));
    cudaFree(firstresult);
}

__device__ void prefixsum4_warp(int *cache_num, int lane)
{
    if(lane >= 1)
        cache_num[lane] += cache_num[lane - 1];
    __syncwarp();
    if(lane >= 2)
        cache_num[lane] += cache_num[lane - 2];
    __syncwarp();
    if(lane >= 4)
        cache_num[lane] += cache_num[lane - 4];
    __syncwarp();
    if(lane >= 8)
        cache_num[lane] += cache_num[lane - 8];
    __syncwarp();
    if(lane >= 16)
        cache_num[lane] += cache_num[lane - 16];
    // __syncwarp();
}

__device__ void prefixsum4_block(int *cache_num)
{
    int warp_id = threadIdx.x >> 5;
    int lane    = threadIdx.x & 31;
    __shared__ int warp_sum[32];

    prefixsum4_warp(cache_num + warp_id * 32,lane);
    __syncthreads();

    if(lane == 31)
        warp_sum[warp_id] = cache_num[threadIdx.x];
    __syncthreads();

    if(warp_id == 0)
        prefixsum4_warp(warp_sum,lane);
    __syncthreads();

    if(warp_id > 0)
        cache_num[threadIdx.x] += warp_sum[warp_id - 1];
    __syncthreads();
}

__global__ void firstprefixsum4(int *num, int *firstresult, int *num_out, int length, int blocknum)
{
    int ii  = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    extern __shared__ int cache_num[];
    if(ii < length) cache_num[tid] = num[ii];
    else cache_num[tid] = 0;
    __syncthreads();

    prefixsum4_block(cache_num);

    if(ii < length) 
        num_out[ii] = cache_num[tid];
    if(tid == blockDim.x - 1) 
        firstresult[blockIdx.x] = cache_num[tid];
}

// improve firstprefixsum3 from warp serial to parallel
void prefixsum4(int *num, int *num_out, int length)
{
    size_t blocksize = 128;
    size_t blocknum = (length + blocksize - 1) / blocksize;

    int *firstresult;
    malloc_GPU(blocknum * sizeof(int));
    cudaMalloc((void**)&firstresult,blocknum * sizeof(int));

    firstprefixsum4<<<blocknum,blocksize>>>(num,firstresult,num_out,length,blocknum);

    secondprefixsum1<<<1,1>>>(firstresult,blocknum);

    addfirstsum1<<<blocknum,blocksize>>>(firstresult,num_out,length,blocknum);

    free_GPU(blocknum * sizeof(int));
    cudaFree(firstresult);
}

__device__ void prefixsum5_warp(int *cache_num, int lane)
{
	volatile int *vcache_num = cache_num;
	vcache_num[0] += vcache_num[-1];
	vcache_num[0] += vcache_num[-2];
	vcache_num[0] += vcache_num[-4];
	vcache_num[0] += vcache_num[-8];
	vcache_num[0] += vcache_num[-16];
}

__device__ void prefixsum5_block(int *my_cache)
{
    int warp_id = threadIdx.x >> 5;
    int lane    = threadIdx.x & 31;
    extern __shared__ int warp_sum[];   // 16 zero padding

    prefixsum5_warp(my_cache,lane);
    __syncthreads();

    if(lane == 31)
        warp_sum[16 + warp_id] = my_cache[0];
    __syncthreads();

    if(warp_id == 0)
        prefixsum5_warp(warp_sum + 16 + lane,lane);
    __syncthreads();

    if(warp_id > 0)
        my_cache[0] += warp_sum[16 + warp_id - 1];
    __syncthreads();
}

__global__ void firstprefixsum5(int *num, int *firstresult, int *num_out, int length, int blocknum)
{
    int ii  = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int warp_id = threadIdx.x >> 5;
    int lane    = threadIdx.x & 31;

    // the first 16 + 32 is used to save warp sum
    extern __shared__ int cache_num[];

    if(tid < 16)
        cache_num[tid] = 0;
    if(lane < 16)
        cache_num[(16 + 32) + warp_id * (16 + 32) + lane] = 0;
    __syncthreads();

    int *my_cache = cache_num + (16 + 32) + warp_id * (16 + 32) + 16 + lane;
    if(ii < length)
        my_cache[0] = num[ii];
    else 
        my_cache[0] = 0;
    __syncthreads();

    prefixsum5_block(my_cache);
    __syncthreads();

    if(ii < length)
        num_out[ii] = my_cache[0];
    if(tid == blockDim.x - 1)
        firstresult[blockIdx.x] = my_cache[0];
}

// improve firstprefixsum4 on simplify warp
void prefixsum5(int *num, int *num_out, int length)
{
    size_t blocksize = 128;
    size_t blocknum  = (length + blocksize - 1) / blocksize;
    size_t warp_num  = blocksize / 32;
    size_t sharesize = (16 + 32 + warp_num * (16 + 32)) * sizeof(int);

    int *firstresult;
    malloc_GPU(blocknum * sizeof(int));
    cudaMalloc((void**)&firstresult,blocknum * sizeof(int));

    firstprefixsum5<<<blocknum,blocksize,sharesize>>>(num,firstresult,num_out,length,blocknum);

    secondprefixsum1<<<1,1>>>(firstresult,blocknum);

    addfirstsum1<<<blocknum,blocksize>>>(firstresult,num_out,length,blocknum);

    free_GPU(blocknum * sizeof(int));
    cudaFree(firstresult);
}

__device__ void prefixsum6_warp(int *cache_num, int lane)
{
	volatile int *vcache_num = cache_num;
	vcache_num[0] += vcache_num[-1];
	vcache_num[0] += vcache_num[-2];
	vcache_num[0] += vcache_num[-4];
	vcache_num[0] += vcache_num[-8];
	vcache_num[0] += vcache_num[-16];
}

__device__ void prefixsum6_block(int *my_cache)
{
    int warp_id = threadIdx.x >> 5;
    int lane    = threadIdx.x & 31;
    extern __shared__ int warp_sum[];   // 16 zero padding

    prefixsum6_warp(my_cache,lane);
    __syncthreads();

    if(lane == 31)
        warp_sum[16 + warp_id] = my_cache[0];
    __syncthreads();

    if(warp_id == 0)
        prefixsum6_warp(warp_sum + 16 + lane,lane);
    __syncthreads();

    if(warp_id > 0)
        my_cache[0] += warp_sum[16 + warp_id - 1];
    __syncthreads();
}

__global__ void firstprefixsum6(int *num, int *endresult, int *firstresult, int length, int blocknum)
{
    int ii  = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int warp_id = threadIdx.x >> 5;
    int lane    = threadIdx.x & 31;

    // the first 16 + 32 is used to save warp sum
    extern __shared__ int cache_num[];

    if(tid < 16)
        cache_num[tid] = 0;
    if(lane < 16)
        cache_num[(16 + 32) + warp_id * (16 + 32) + lane] = 0;
    __syncthreads();

    int *my_cache = cache_num + (16 + 32) + warp_id * (16 + 32) + 16 + lane;
    if(ii < length)
        my_cache[0] = num[ii];
    else 
        my_cache[0] = 0;
    __syncthreads();

    prefixsum6_block(my_cache);
    __syncthreads();

    if(ii < length)
        endresult[ii] = my_cache[0];
    if(tid == blockDim.x - 1)
        firstresult[blockIdx.x] = my_cache[0];
}

__global__ void prefixsum_communication_template(int *endresult, int *secondresult, int length)
{
	int ii  = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < length)
	{
		int block_num = blockIdx.x;
		if (block_num != 0)
            endresult[ii] += secondresult[block_num - 1];
	}
}

// improve firstprefixsum4 on simplify warp
void prefixsum6(int *num, int **num_out, int length)
{
    size_t blocksize = 128;
    size_t blocknum  = (length + blocksize - 1) / blocksize;
    size_t warp_num  = blocksize / 32;
    size_t sharesize = (16 + 32 + warp_num * (16 + 32)) * sizeof(int);

	int *endresult;
	malloc_GPU(length * sizeof(int));
    cudaMalloc((void**)&endresult,length * sizeof(int));
    // printf("malloc endresult length=%d\n",length);

    int *firstresult;
    malloc_GPU(blocknum * sizeof(int));
    cudaMalloc((void**)&firstresult,blocknum * sizeof(int));
    // printf("malloc firstresult length=%d\n",length);

    firstprefixsum6<<<blocknum,blocksize,sharesize>>>(num,endresult,firstresult,length,blocknum);

	if(blocknum == 1)
	{
		*num_out = endresult;
		free_GPU(blocknum * sizeof(int));
        cudaFree(firstresult);
        // printf("free firstresult length=%d\n",length);
        return ;
	}

	// do the second scan after the first scan
	int *secondresult;
	prefixsum6(firstresult,&secondresult,blocknum);

	// copy current result
    prefixsum_communication_template<<<blocknum, blocksize>>>(endresult, secondresult, length);
    cudaDeviceSynchronize();

	*num_out = endresult;

    free_GPU(blocknum * sizeof(int));
    cudaFree(secondresult);
    // printf("free secondresult length=%d\n",length);
    free_GPU(blocknum * sizeof(int));
    cudaFree(firstresult);
    // printf("free firstresult length=%d\n",length);
}

__device__ int prefixsum7_warp(int val, int lane)
{
	int temp = __shfl_up_sync(0xffffffff, val, 1);
	if(lane >= 1)
		val += temp;
	temp = __shfl_up_sync(0xffffffff, val, 2);
	if(lane >= 2)
		val += temp;
	temp = __shfl_up_sync(0xffffffff, val, 4);
	if(lane >= 4)
		val += temp;
	temp = __shfl_up_sync(0xffffffff, val, 8);
	if(lane >= 8)
		val += temp;
	temp = __shfl_up_sync(0xffffffff, val, 16);
	if(lane >= 16)
		val += temp;

	return val;
}

__device__ void prefixsum7_block(int *my_cache, int lane)
{
    int warp_id = threadIdx.x >> 5;
    // int lane    = threadIdx.x & 31;
    extern __shared__ int warp_sum[];   // 16 zero padding

    my_cache[0] = prefixsum7_warp(my_cache[0],lane);
    __syncthreads();

    if(lane == 31)
        warp_sum[16 + warp_id] = my_cache[0];
    __syncthreads();

    if(warp_id == 0)
        warp_sum[16 + lane] = prefixsum7_warp(warp_sum[16 + lane],lane);
    __syncthreads();

    if(warp_id > 0)
        my_cache[0] += warp_sum[16 + warp_id - 1];
    __syncthreads();
}

__global__ void firstprefixsum7(int *num, int *endresult, int *firstresult, int length, int blocknum)
{
    int ii  = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int warp_id = threadIdx.x >> 5;
    int lane    = threadIdx.x & 31;

    // the first 16 + 32 is used to save warp sum
    extern __shared__ int cache_num[];

    if(tid < 16)
        cache_num[tid] = 0;
    if(lane < 16)
        cache_num[(16 + 32) + warp_id * (16 + 32) + lane] = 0;
    __syncthreads();

    int *my_cache = cache_num + (16 + 32) + warp_id * (16 + 32) + 16 + lane;
    if(ii < length)
        my_cache[0] = num[ii];
    else 
        my_cache[0] = 0;
    __syncthreads();

    prefixsum7_block(my_cache,lane);
    __syncthreads();

    if(ii < length)
        endresult[ii] = my_cache[0];
    if(tid == blockDim.x - 1)
        firstresult[blockIdx.x] = my_cache[0];
}

// improve firstprefixsum4 on simplify warp
void prefixsum7(int *num, int **num_out, int length, size_t blocksize)
{
    // size_t blocksize = 128;
    size_t blocknum  = (length + blocksize - 1) / blocksize;
    size_t warp_num  = blocksize / 32;
    size_t sharesize = (16 + 32 + warp_num * (16 + 32)) * sizeof(int);

	int *endresult;
	malloc_GPU(length * sizeof(int));
    cudaMalloc((void**)&endresult,length * sizeof(int));

    int *firstresult;
    malloc_GPU(blocknum * sizeof(int));
    cudaMalloc((void**)&firstresult,blocknum * sizeof(int));

    firstprefixsum7<<<blocknum,blocksize,sharesize>>>(num,endresult,firstresult,length,blocknum);

	if(blocknum == 1)
	{
		*num_out = endresult;
		free_GPU(blocknum * sizeof(int));
        cudaFree(firstresult);
        return ;
	}

	// do the second scan after the first scan
	int *secondresult;
	prefixsum7(firstresult,&secondresult,blocknum,blocksize);

	// copy current result
    prefixsum_communication_template<<<blocknum, blocksize>>>(endresult, secondresult, length);
    cudaDeviceSynchronize();

	*num_out = endresult;

    free_GPU(blocknum * sizeof(int));
    cudaFree(secondresult);
    free_GPU(blocknum * sizeof(int));
    cudaFree(firstresult);
}

__device__ int prefixsum8_warp(int val, int lane)
{
	int temp = __shfl_up_sync(0xffffffff, val, 1);
	if(lane >= 1)
		val += temp;
	temp = __shfl_up_sync(0xffffffff, val, 2);
	if(lane >= 2)
		val += temp;
	temp = __shfl_up_sync(0xffffffff, val, 4);
	if(lane >= 4)
		val += temp;
	temp = __shfl_up_sync(0xffffffff, val, 8);
	if(lane >= 8)
		val += temp;
	temp = __shfl_up_sync(0xffffffff, val, 16);
	if(lane >= 16)
		val += temp;

	return val;
}

__device__ int prefixsum8_block(int val)
{
    int warp_id = threadIdx.x >> 5;
    int lane    = threadIdx.x & 31;
    extern __shared__ int warp_sum[];

    val = prefixsum8_warp(val,lane);
    __syncthreads();

    if(lane == 31)
        warp_sum[warp_id] = val;
    __syncthreads();

    if(warp_id == 0)
        warp_sum[lane] = prefixsum8_warp(warp_sum[lane],lane);
    __syncthreads();

    if(warp_id > 0)
        val += warp_sum[warp_id - 1];
    __syncthreads();

    return val;
}

__global__ void firstprefixsum8(int *num, int *endresult, int *firstresult, int length, int blocknum)
{
    int ii  = blockIdx.x * blockDim.x + threadIdx.x;

    int val;
    if(ii < length)
        val = num[ii];
    else 
        val = 0;

    val = prefixsum8_block(val);
    __syncthreads();

    if(ii < length)
        endresult[ii] = val;
    if(threadIdx.x == blockDim.x - 1)
        firstresult[blockIdx.x] = val;
}

// improve firstprefixsum4 on simplify warp
void prefixsum8(int *num, int **num_out, int length, size_t blocksize)
{
    // size_t blocksize = 128;
    size_t blocknum  = (length + blocksize - 1) / blocksize;
    size_t warp_num  = blocksize / 32;
    size_t sharesize = 32 * sizeof(int);

	int *endresult;
	malloc_GPU(length * sizeof(int));
    cudaMalloc((void**)&endresult,length * sizeof(int));

    int *firstresult;
    malloc_GPU(blocknum * sizeof(int));
    cudaMalloc((void**)&firstresult,blocknum * sizeof(int));

    firstprefixsum8<<<blocknum,blocksize,sharesize>>>(num,endresult,firstresult,length,blocknum);

	if(blocknum == 1)
	{
		*num_out = endresult;
		free_GPU(blocknum * sizeof(int));
        cudaFree(firstresult);
        return ;
	}

	// do the second scan after the first scan
	int *secondresult;
	prefixsum8(firstresult,&secondresult,blocknum,blocksize);

	// copy current result
    prefixsum_communication_template<<<blocknum, blocksize>>>(endresult, secondresult, length);
    cudaDeviceSynchronize();

	*num_out = endresult;

    free_GPU(blocknum * sizeof(int));
    cudaFree(secondresult);
    free_GPU(blocknum * sizeof(int));
    cudaFree(firstresult);
}

int main(int argc, char **argv)
{
    cudaSetDevice(0);

    GPU_Memory();

    length = atoi(argv[1]);

    int *num, *cuda_num, *cuda_num_out;
    // num = (int *)malloc(length * sizeof(int));
    malloc_GPU(length * sizeof(int));
    cudaMalloc((void**)&cuda_num,length * sizeof(int));
    malloc_GPU(length * sizeof(int));
    cudaMalloc((void**)&cuda_num_out,length * sizeof(int));

    // prefixsum1
    init_num<<<(length + 127) / 128,128>>>(cuda_num,length);
    init_num<<<(length + 127) / 128,128>>>(cuda_num_out,length);
    cudaDeviceSynchronize();
    gettimeofday(&prefixsum_begin,NULL);
    prefixsum1(cuda_num,cuda_num_out,length);
    cudaDeviceSynchronize();
    gettimeofday(&prefixsum_end,NULL);
    prefixsum1_time += (prefixsum_end.tv_sec - prefixsum_begin.tv_sec) * 1000 + (prefixsum_end.tv_usec - prefixsum_begin.tv_usec) / 1000.0;
    printf("prefixsum1     :");
    cudaDeviceSynchronize();
    print_num_device<<<1,1>>>(cuda_num_out, length);
    cudaDeviceSynchronize();

    // prefixsum2
    init_num<<<(length + 127) / 128,128>>>(cuda_num,length);
    init_num<<<(length + 127) / 128,128>>>(cuda_num_out,length);
    cudaDeviceSynchronize();
    gettimeofday(&prefixsum_begin,NULL);
    prefixsum2(cuda_num,cuda_num_out,length);
    cudaDeviceSynchronize();
    gettimeofday(&prefixsum_end,NULL);
    prefixsum2_time += (prefixsum_end.tv_sec - prefixsum_begin.tv_sec) * 1000 + (prefixsum_end.tv_usec - prefixsum_begin.tv_usec) / 1000.0;
    printf("prefixsum2     :");
    cudaDeviceSynchronize();
    print_num_device<<<1,1>>>(cuda_num_out, length);
    cudaDeviceSynchronize();

    // prefixsum3
    init_num<<<(length + 127) / 128,128>>>(cuda_num,length);
    init_num<<<(length + 127) / 128,128>>>(cuda_num_out,length);
    cudaDeviceSynchronize();
    gettimeofday(&prefixsum_begin,NULL);
    prefixsum3(cuda_num,cuda_num_out,length);
    cudaDeviceSynchronize();
    gettimeofday(&prefixsum_end,NULL);
    prefixsum3_time += (prefixsum_end.tv_sec - prefixsum_begin.tv_sec) * 1000 + (prefixsum_end.tv_usec - prefixsum_begin.tv_usec) / 1000.0;
    printf("prefixsum3     :");
    cudaDeviceSynchronize();
    print_num_device<<<1,1>>>(cuda_num_out, length);
    cudaDeviceSynchronize();

    // prefixsum4
    init_num<<<(length + 127) / 128,128>>>(cuda_num,length);
    init_num<<<(length + 127) / 128,128>>>(cuda_num_out,length);
    cudaDeviceSynchronize();
    gettimeofday(&prefixsum_begin,NULL);
    prefixsum4(cuda_num,cuda_num_out,length);
    cudaDeviceSynchronize();
    gettimeofday(&prefixsum_end,NULL);
    prefixsum4_time += (prefixsum_end.tv_sec - prefixsum_begin.tv_sec) * 1000 + (prefixsum_end.tv_usec - prefixsum_begin.tv_usec) / 1000.0;
    printf("prefixsum4     :");
    cudaDeviceSynchronize();
    print_num_device<<<1,1>>>(cuda_num_out, length);
    cudaDeviceSynchronize();

    // prefixsum5
    init_num<<<(length + 127) / 128,128>>>(cuda_num,length);
    init_num<<<(length + 127) / 128,128>>>(cuda_num_out,length);
    cudaDeviceSynchronize();
    gettimeofday(&prefixsum_begin,NULL);
    prefixsum5(cuda_num,cuda_num_out,length);
    cudaDeviceSynchronize();
    gettimeofday(&prefixsum_end,NULL);
    prefixsum5_time += (prefixsum_end.tv_sec - prefixsum_begin.tv_sec) * 1000 + (prefixsum_end.tv_usec - prefixsum_begin.tv_usec) / 1000.0;
    printf("prefixsum5     :");
    cudaDeviceSynchronize();
    print_num_device<<<1,1>>>(cuda_num_out, length);
    cudaDeviceSynchronize();

	// prefixsum6
    init_num<<<(length + 127) / 128,128>>>(cuda_num,length);
    free_GPU(length * sizeof(int));
    cudaFree(cuda_num_out);
    cudaDeviceSynchronize();
    gettimeofday(&prefixsum_begin,NULL);
    prefixsum6(cuda_num,&cuda_num_out,length);
    cudaDeviceSynchronize();
    gettimeofday(&prefixsum_end,NULL);
    prefixsum6_time += (prefixsum_end.tv_sec - prefixsum_begin.tv_sec) * 1000 + (prefixsum_end.tv_usec - prefixsum_begin.tv_usec) / 1000.0;
    printf("prefixsum6     :");
	cudaDeviceSynchronize();
    print_num_device<<<1,1>>>(cuda_num_out, length);
    cudaDeviceSynchronize();

	// prefixsum7
    init_num<<<(length + 127) / 128,128>>>(cuda_num,length);
    free_GPU(length * sizeof(int));
    cudaFree(cuda_num_out);
    cudaDeviceSynchronize();
	blocksize = 128;
    gettimeofday(&prefixsum_begin,NULL);
    prefixsum7(cuda_num,&cuda_num_out,length,blocksize);
    cudaDeviceSynchronize();
    gettimeofday(&prefixsum_end,NULL);
    prefixsum7_time += (prefixsum_end.tv_sec - prefixsum_begin.tv_sec) * 1000 + (prefixsum_end.tv_usec - prefixsum_begin.tv_usec) / 1000.0;
    printf("prefixsum7     :");
	cudaDeviceSynchronize();
    print_num_device<<<1,1>>>(cuda_num_out, length);
    cudaDeviceSynchronize();

    // prefixsum8 128
    init_num<<<(length + 127) / 128,128>>>(cuda_num,length);
    free_GPU(length * sizeof(int));
    cudaFree(cuda_num_out);
    cudaDeviceSynchronize();
	blocksize = 128;
    gettimeofday(&prefixsum_begin,NULL);
    prefixsum8(cuda_num,&cuda_num_out,length,blocksize);
    cudaDeviceSynchronize();
    gettimeofday(&prefixsum_end,NULL);
    prefixsum8_time_128 += (prefixsum_end.tv_sec - prefixsum_begin.tv_sec) * 1000 + (prefixsum_end.tv_usec - prefixsum_begin.tv_usec) / 1000.0;
    printf("prefixsum8_128 :");
	cudaDeviceSynchronize();
    print_num_device<<<1,1>>>(cuda_num_out, length);
    cudaDeviceSynchronize();

    // prefixsum8 256
    init_num<<<(length + 127) / 128,128>>>(cuda_num,length);
    free_GPU(length * sizeof(int));
    cudaFree(cuda_num_out);
    cudaDeviceSynchronize();
	blocksize = 256;
    gettimeofday(&prefixsum_begin,NULL);
    prefixsum8(cuda_num,&cuda_num_out,length,blocksize);
    cudaDeviceSynchronize();
    gettimeofday(&prefixsum_end,NULL);
    prefixsum8_time_256 += (prefixsum_end.tv_sec - prefixsum_begin.tv_sec) * 1000 + (prefixsum_end.tv_usec - prefixsum_begin.tv_usec) / 1000.0;
    printf("prefixsum8_256 :");
	cudaDeviceSynchronize();
    print_num_device<<<1,1>>>(cuda_num_out, length);
    cudaDeviceSynchronize();

    // prefixsum8 512
    init_num<<<(length + 127) / 128,128>>>(cuda_num,length);
    free_GPU(length * sizeof(int));
    cudaFree(cuda_num_out);
    cudaDeviceSynchronize();
	blocksize = 512;
    gettimeofday(&prefixsum_begin,NULL);
    prefixsum8(cuda_num,&cuda_num_out,length,blocksize);
    cudaDeviceSynchronize();
    gettimeofday(&prefixsum_end,NULL);
    prefixsum8_time_512 += (prefixsum_end.tv_sec - prefixsum_begin.tv_sec) * 1000 + (prefixsum_end.tv_usec - prefixsum_begin.tv_usec) / 1000.0;
    printf("prefixsum8_512 :");
	cudaDeviceSynchronize();
    print_num_device<<<1,1>>>(cuda_num_out, length);
    cudaDeviceSynchronize();

    // prefixsum8 1024
    init_num<<<(length + 127) / 128,128>>>(cuda_num,length);
    free_GPU(length * sizeof(int));
    cudaFree(cuda_num_out);
    cudaDeviceSynchronize();
	blocksize = 1024;
    gettimeofday(&prefixsum_begin,NULL);
    prefixsum8(cuda_num,&cuda_num_out,length,blocksize);
    cudaDeviceSynchronize();
    gettimeofday(&prefixsum_end,NULL);
    prefixsum8_time_1024 += (prefixsum_end.tv_sec - prefixsum_begin.tv_sec) * 1000 + (prefixsum_end.tv_usec - prefixsum_begin.tv_usec) / 1000.0;
    printf("prefixsum8_1024:");
	cudaDeviceSynchronize();
    print_num_device<<<1,1>>>(cuda_num_out, length);
    cudaDeviceSynchronize();

    // thrust
    init_num<<<(length + 127) / 128,128>>>(cuda_num,length);
    init_num<<<(length + 127) / 128,128>>>(cuda_num_out,length);
    cudaDeviceSynchronize();
    gettimeofday(&prefixsum_begin,NULL);
    thrust::inclusive_scan(thrust::device, cuda_num, cuda_num + length, cuda_num_out);
    cudaDeviceSynchronize();
    gettimeofday(&prefixsum_end,NULL);
    prefixsum_trust_time += (prefixsum_end.tv_sec - prefixsum_begin.tv_sec) * 1000 + (prefixsum_end.tv_usec - prefixsum_begin.tv_usec) / 1000.0;
    printf("thrust         :");
	cudaDeviceSynchronize();
    print_num_device<<<1,1>>>(cuda_num_out, length);
    cudaDeviceSynchronize();

    printf("\nTime\n");
    printf("the prefixsum1 time:      %10.3lf ms\n", prefixsum1_time);
    printf("the prefixsum2 time:      %10.3lf ms\n", prefixsum2_time);
    printf("the prefixsum3 time:      %10.3lf ms\n", prefixsum3_time);
    printf("the prefixsum4 time:      %10.3lf ms\n", prefixsum4_time);
    printf("the prefixsum5 time:      %10.3lf ms\n", prefixsum5_time);
	printf("the prefixsum6 time:      %10.3lf ms\n", prefixsum6_time);
	printf("the prefixsum7 time:      %10.3lf ms\n", prefixsum7_time);
    printf("the prefixsum8 time 128:  %10.3lf ms\n", prefixsum8_time_128);
    printf("the prefixsum8 time 256:  %10.3lf ms\n", prefixsum8_time_256);
    printf("the prefixsum8 time 512:  %10.3lf ms\n", prefixsum8_time_512);
    printf("the prefixsum8 time 1024: %10.3lf ms\n", prefixsum8_time_1024);
    printf("the trust time:           %10.3lf ms\n", prefixsum_trust_time);

    free_GPU(length * sizeof(int));
    cudaFree(cuda_num);
    free_GPU(length * sizeof(int));
    cudaFree(cuda_num_out);
    free(num);

    return 0;
}