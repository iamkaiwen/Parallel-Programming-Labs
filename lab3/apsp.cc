#define CHUNK_SIZE 2000
#define INIT_POSITION *(edges_arr[i] + j)
#define INPUT_POSITION *(edges_arr[src - st] + dest)
#define POSITION *(edges_arr[i - st] + j)
#define ARR_POSITION *(edges_arr[i - st] + k)
#define BRR_POSITION brr[j]
#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <cassert>
#include <fstream>
#include <utility>
#include <cstdlib>
#include <cstring>
int procs_rank , procs_size;
int num_vertices , num_edges;
int brr[5005];
int *edges;
int **edges_arr;
int *ans;
int st , end;
const char* in_filename; const char* out_filename; const char* par_filename;
void read_input_file(){

    std::ifstream in_fs (in_filename, std::ifstream::binary);
    
    in_fs.read((char*)&num_vertices , sizeof(int));
    in_fs.read((char*)&num_edges , sizeof(int));

    int mod = num_vertices % procs_size;
    int block = (int)(num_vertices / procs_size);
    int buf_size = (mod == 0)? block :
                   (procs_rank < mod)? block + 1 : block;
    st = (mod == 0)? block * procs_rank :
         (procs_rank < mod)? (block + 1) * procs_rank : num_vertices - block * (procs_size - procs_rank);
    end = st + buf_size - 1;

    edges = (int *)malloc(buf_size * num_vertices * sizeof(int));
    edges_arr = (int **)malloc(buf_size * sizeof(int*));

    for(int i = 0;i < buf_size;i++){
        //edges_arr initialize
        edges_arr[i] = edges + i * num_vertices;
        for(int j = 0;j < num_vertices;j++){
            INIT_POSITION = (st + i == j)? 0 : -1;
        }
    }

    memset(brr , -1 , num_vertices * sizeof(int));
    brr[0] = 0;

    for(int i = 0;i < num_edges;i++){
        int src , dest , weight;
        in_fs.read((char*)&src , sizeof(int));
        in_fs.read((char*)&dest , sizeof(int));
        in_fs.read((char*)&weight , sizeof(int));
        if(st <= src && src <= end){
            INPUT_POSITION = weight;
        }
        if(src == 0){
            brr[dest] = weight;
        }
    }

    in_fs.close();
}
void write_output(){
    std::ofstream out_fs (out_filename , std::ofstream::binary);
    for(int i = 0;i < num_vertices * num_vertices;i++){
        out_fs.write((char*)&ans[i] , sizeof(int));
    }
    out_fs.close();
}
int main(int argc, char** argv){

    in_filename = argv[1];
    out_filename = argv[2];
    par_filename = (argc == 4)? argv[3] : "123";

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &procs_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procs_size);

    read_input_file();

    int rt = (num_vertices <= procs_size)? 1 : 0;

    for(int k = 0;k < num_vertices;k++){

        #pragma omp parallel
        {
            #pragma omp for schedule(guided , CHUNK_SIZE) nowait collapse(2) firstprivate(brr , edges , st , end , num_vertices) lastprivate(edges)
            for(int i = st;i <= end;i++){
                for(int j = 0;j < num_vertices;j++){
                    if(ARR_POSITION != -1 && BRR_POSITION != -1 && (POSITION == -1 || POSITION > ARR_POSITION + BRR_POSITION)){
                        POSITION = ARR_POSITION + BRR_POSITION;
                    }
                }
            }
        }

        if(st <= k + 1 && k + 1 <= end){
            int i = k + 1;
            for(int j = 0;j < num_vertices;j++){
                brr[j] = POSITION;
            }
            brr[num_vertices] = (k + 2 <= end)? procs_rank : procs_rank + 1;
        }

        if(procs_size > 1 && k + 1 < num_vertices){
            MPI_Bcast(brr, num_vertices + 1, MPI_INT, rt , MPI_COMM_WORLD);
            rt = brr[num_vertices];
        }
    }

    if(procs_rank != 0){
        MPI_Gatherv(edges , (end - st + 1) * num_vertices , MPI_INT , NULL , NULL , NULL , MPI_INT , 0 , MPI_COMM_WORLD);
    }else{
        ans = (int *)malloc(num_vertices * num_vertices * sizeof(int));
        int *recvcounts =  (int *)malloc(procs_size * sizeof(int)) , *displs = (int *)malloc(procs_size * sizeof(int));
        int p = 0 , mod = num_vertices % procs_size , block = (int)(num_vertices / procs_size);
        for(int k = 0;k < procs_size;k++){
            displs[k] = p;
            recvcounts[k] = (mod == 0)? block :
                            (k < mod)? block + 1 : block;
            recvcounts[k] *= num_vertices;
            p += recvcounts[k];
        }
        MPI_Gatherv(edges , (end - st + 1) * num_vertices , MPI_INT , ans , recvcounts , displs , MPI_INT , 0 , MPI_COMM_WORLD);
    }

    if(procs_rank == 0){
        write_output();
    }

    MPI_Finalize();

    return 0;
}

