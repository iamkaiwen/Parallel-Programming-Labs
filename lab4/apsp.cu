#define Block_size 32
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <device_functions.hpp>

const int INF = 1000000000;
void input(char *inFileName);
void output(char *outFileName);

void block_FW(int B);
int ceil(int a, int b);
__global__ void cal(int* Dist, int B, int Dist_width, int Round, int par_x, int par_y, int phase);

int n, m;	
int **Dist;
int *device_Dist = NULL;       // GPU image array

int main(int argc, char* argv[]) {
	input(argv[1]);
	int B = Block_size;
	block_FW(B);
	output(argv[2]);
	return 0;
}

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
	fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    int round = ceil(n, Block_size) * Block_size; //memset change to other method

	Dist = (int**)malloc(sizeof(int*) * round);
	for(int i = 0;i < round;i++){
		Dist[i] = (int*)malloc(sizeof(int) * round);
	}

    for (int i = 0; i < round; ++ i) {
        for (int j = 0; j < round; ++ j) {
            if (i == j) {
                Dist[i][j] = 0;
            } else {
                Dist[i][j] = INF;
            }
        }
    }

    int pair[3];
    for (int i = 0; i < m; ++ i) {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0]][pair[1]] = pair[2];
    }
    fclose(file);
}

void output(char *outFileName) {
	FILE *outfile = fopen(outFileName, "w");
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
            if (Dist[i][j] >= INF)
				Dist[i][j] = INF;
			// std:cout << i << " " << j << " " << Dist[i][j] << "\n";
		}
		fwrite(Dist[i], sizeof(int), n, outfile);
	}
    fclose(outfile);
}

int ceil(int a, int b) {
	return (a + b - 1) / b;
}

void block_FW(int B) {

	int round = ceil(n, B);

	cudaMalloc((void **)&device_Dist, (size_t)(round * B * round * B * sizeof(int)));

	for(int i = 0;i < round * B;i++){
        cudaMemcpy(device_Dist + (i * round * B), Dist[i], (size_t)(round * B * sizeof(int)), cudaMemcpyHostToDevice);
    }
    
	dim3 grid1(1, 1); dim3 grid20(1, round); dim3 grid21(round, 1); dim3 grid3(round, round);
	dim3 blk(Block_size , Block_size);
	
	for (int r = 0; r < round; ++r) {
        // printf("%d %d\n", r, round);

		// fflush(stdout);
		
		/* Phase 1*/
		cal<<<grid1 , blk>>>(device_Dist, B, round * B, r, r, r, 1);

		/* Phase 2*/
		cal<<<grid20 , blk>>>(device_Dist, B, round * B, r, r, 0, 20);
		cal<<<grid21 , blk>>>(device_Dist, B, round * B, r, 0, r, 21);
		
		/* Phase 3*/
		cal<<<grid3 , blk>>>(device_Dist, B, round * B, r, 0, 0, 3);
	}

	for(int i = 0;i < n;i++)
		cudaMemcpy(Dist[i], device_Dist + (i * round * B), (size_t)(n * sizeof(int)), cudaMemcpyDeviceToHost);
}

__global__ void cal(int* Dist, int B, int Dist_width, int Round, int par_x, int par_y, int phase) {
	
	__shared__ int i_k[Block_size][Block_size];
	__shared__ int k_j[Block_size][Block_size];
	__shared__ int i_j[Block_size][Block_size];
	
	int real_i = par_x * B + blockIdx.x * B;
	int real_j = par_y * B + blockIdx.y * B;

	if(phase == 20 && real_j == Round * B){
		return;
	}else if(phase == 21 && real_i == Round * B){
		return;
	}else if(phase == 3 && (real_i == Round * B || real_j == Round * B)){
		return;
	}
	
	real_i += threadIdx.y;
	real_j += threadIdx.x;

	int i = threadIdx.y , j = threadIdx.x , k;
	i_k[i][j] = Dist[real_i * Dist_width + Round * B + j];
	k_j[i][j] = Dist[(Round * B + i) * Dist_width + real_j];
	i_j[i][j] = Dist[real_i * Dist_width + real_j];
	__syncthreads();

	if(phase == 1){
		for(k = 0;k < B;k++){
			if (i_k[i][k] + k_j[k][j] < i_j[i][j]) {
				i_j[i][j] = i_k[i][k] + k_j[k][j];
				i_k[i][j] = i_j[i][j];
				k_j[i][j] = i_j[i][j];
			}
			__syncthreads();
		}
	}else if(phase == 20){
		for(k = 0;k < B;k++){
			if (i_k[i][k] + k_j[k][j] < i_j[i][j]) {
				i_j[i][j] = i_k[i][k] + k_j[k][j];
				k_j[i][j] = i_j[i][j];
			}
			__syncthreads();
		}
	}else if(phase == 21){
		for(k = 0;k < B;k++){
			if (i_k[i][k] + k_j[k][j] < i_j[i][j]) {
				i_j[i][j] = i_k[i][k] + k_j[k][j];
				i_k[i][j] = i_j[i][j];
			}
			__syncthreads();
		}
	}else{
		#pragma unroll
		for(k = 0; k < Block_size; k++)
			i_j[i][j] = min(i_j[i][j], i_k[i][k] + k_j[k][j]);
	}

	Dist[real_i * Dist_width + real_j] = i_j[i][j];
}

