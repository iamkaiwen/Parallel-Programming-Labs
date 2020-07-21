#define Block_size 32
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <mpi.h>

const int INF = 1000000000;
void input(char *inFileName);
void output(char *outFileName);

void block_FW(int B);
int ceil(int a, int b);
__global__ void cal(int* Dist, int B, int Dist_width, int Round, int par_x, int par_y, int phase);

int n, m;	
int **Dist;
int *device_Dist;	// GPU image array
int procs_rank , procs_size;
int split , leng;

int main(int argc, char* argv[]) {
	MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &procs_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procs_size);
	
	int B = Block_size;

	if(procs_rank == 0){
		input(argv[1]);
		split = (int)(ceil(n, B) / 2) + 1;
		MPI_Send(&n, 1, MPI_INT, 1, 0, MPI_COMM_WORLD); //send n , tag = 0
		for(int i = split * B;i < n;i++){
			MPI_Send(Dist[i], leng, MPI_INT, 1, 1, MPI_COMM_WORLD); //send data , tag = 1
		}
	}else{
		MPI_Recv(&n, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // recv n , tag = 0
		split = (int)(ceil(n, B) / 2) + 1;
		leng = ceil(n, Block_size) * Block_size; //memset change to other method
		Dist = (int**)malloc(sizeof(int*) * leng);
		for(int i = 0;i < leng;i++){
			Dist[i] = (int*)malloc(sizeof(int) * leng);
		}
		for(int i = split * B;i < n;i++){
			MPI_Recv(Dist[i], leng, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // recv data , tag = 1
		}
	}


	block_FW(B);
	cudaThreadSynchronize();

	if(procs_rank == 0){
		for(int i = split * B;i < n;i++){
			MPI_Recv(Dist[i], n, MPI_INT, 1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // recv output , tag = 2
		}
		output(argv[2]);
	}else{
		for(int i = split * B;i < n;i++){
			MPI_Send(Dist[i], n, MPI_INT, 0, 2, MPI_COMM_WORLD); //send output , tag = 2
		}
	}

	MPI_Finalize();
	return 0;
}

void input(char* infile) {
    FILE* file = fopen(infile, "rb");

    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);
    
	leng = ceil(n, Block_size) * Block_size; //memset change to other method

	Dist = (int**)malloc(sizeof(int*) * leng);
	for(int i = 0;i < leng;i++){
		Dist[i] = (int*)malloc(sizeof(int) * leng);
	}

    for (int i = 0; i < leng; ++ i) {
        for (int j = 0; j < leng; ++ j) {
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

	cudaMalloc((void **)&device_Dist, (size_t)(leng * leng * sizeof(int)));
	if(procs_rank == 0){
		for(int i = 0;i < split * B;i++){
			cudaMemcpy(device_Dist + (i * round * B), Dist[i], (size_t)(round * B * sizeof(int)), cudaMemcpyHostToDevice);
		}
	}else{
		for(int i = split * B;i < n;i++){
			cudaMemcpy(device_Dist + (i * round * B), Dist[i], (size_t)(round * B * sizeof(int)), cudaMemcpyHostToDevice);
		}
	}
    
	dim3 grid1(1, 1); dim3 grid20(1, round);
	dim3 grid21_0(split, 1); dim3 grid3_0(split, round);
	dim3 grid21_1(round - split, 1); dim3 grid3_1(round - split, round);
	dim3 block(Block_size , Block_size);
	
	for (int r = 0; r < round; ++r) {
		int x = (r < split)? 0 : 1;
		if(procs_rank == x){
			/* Phase 1*/
			cal<<<grid1 , block>>>(device_Dist, B, round * B, r, r, r, 1);
			/* Phase 2 row*/
			cal<<<grid20 , block>>>(device_Dist, B, round * B, r, r, 0, 20);
			for(int i = r * B;i < r * B + B;i++){
				cudaMemcpy(Dist[i], device_Dist + (i * round * B), (size_t)(round * B * sizeof(int)), cudaMemcpyDeviceToHost);
			}
			cudaThreadSynchronize();
			for(int i = 0;i < B;i++){
				MPI_Send(Dist[r * B + i], round * B, MPI_INT, 1 - x, 3, MPI_COMM_WORLD); //send phase , tag = 3
			}
		}else{
			cudaThreadSynchronize();
			for(int i = 0;i < B;i++){
				MPI_Recv(Dist[r * B + i], round * B,  MPI_INT, x, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE); //recv phase , tag = 3
			}
			for(int i = r * B;i < r * B + B;i++){
				cudaMemcpy(device_Dist + (i * round * B), Dist[i], (size_t)(round * B * sizeof(int)), cudaMemcpyHostToDevice);
			}
		}
		
		if(procs_rank == 0){
			/* Phase 2 col*/
			cal<<<grid21_0 , block>>>(device_Dist, B, round * B, r, 0, r, 21);
			/* Phase 3*/
			cal<<<grid3_0 , block>>>(device_Dist, B, round * B, r, 0, 0, 3);
		}else{
			/* Phase 2 col*/
			cal<<<grid21_1 , block>>>(device_Dist, B, round * B, r, split, r, 21);
			/* Phase 3*/
			cal<<<grid3_1 , block>>>(device_Dist, B, round * B, r, split, 0, 3);
		}
	}

	if(procs_rank == 0){
		for(int i = 0;i < split * B;i++){
			cudaMemcpy(Dist[i], device_Dist + (i * round * B), (size_t)(n * sizeof(int)), cudaMemcpyDeviceToHost);
		}
	}else{
		for(int i = split * B;i < n;i++){
			cudaMemcpy(Dist[i], device_Dist + (i * round * B), (size_t)(n * sizeof(int)), cudaMemcpyDeviceToHost);
		}
	}
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
