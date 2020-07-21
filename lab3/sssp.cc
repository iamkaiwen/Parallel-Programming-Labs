#define MAX_THREADS 12
#define MAX_NODES 5005
#include <pthread.h>
#include <mpi.h>
#include <iostream>
#include <cassert>
#include <fstream>
#include <utility>
#include <vector>
#include <queue>
#include <algorithm>
#include <cstring>
int procs_rank , procs_size;
int num_vertices , num_edges;
int nodes[MAX_NODES] , ans[MAX_NODES] , recv_size[MAX_NODES];
bool tf[MAX_NODES];
std::vector< std::pair<int , int> > edges[MAX_NODES];
pthread_t threads[MAX_THREADS];
std::vector< std::pair<int , int> > vec;
pthread_mutex_t ans_mutex[MAX_NODES];
const char* in_filename;const char* out_filename;
int *print_ans;
int TID[MAX_THREADS];
void read_input_file(){
    std::ifstream in_fs (in_filename, std::ifstream::binary);
    
    in_fs.read((char*)&num_vertices , sizeof(int));
    in_fs.read((char*)&num_edges , sizeof(int));

    int tmp = 0;
    for(int i = procs_rank;i < num_vertices;i += procs_size){
        nodes[tmp] = i; ans[tmp] = -1; tf[tmp] = false;
        pthread_mutex_init(&ans_mutex[tmp] , NULL);
        tmp += 1;
    }

    for(int i = 0;i < MAX_THREADS;i++){
        TID[i] = i;
    }

    memset(recv_size , 0 , num_vertices * sizeof(int));

    for(int i = 0;i < num_edges;i++){
        int src , dest , weight;
        in_fs.read((char*)&src , sizeof(int));
        in_fs.read((char*)&dest , sizeof(int));
        in_fs.read((char*)&weight , sizeof(int));
        if(src % procs_size == procs_rank){
            edges[(int)(src / procs_size)].push_back(std::make_pair(dest , weight));
        }
        if(dest % procs_size == procs_rank){
            recv_size[src] += 1;
        }
    }

    in_fs.close();

    if(procs_rank == 0){
        ans[0] = 0;
    }
}
void write_output(){
    std::ofstream out_fs (out_filename , std::ofstream::binary);
    for(int i = 0;i < num_vertices;i++){
        int p = (i % procs_size) * ((int)(num_vertices / procs_size) + 1) + (int)(i / procs_size);
        out_fs.write((char*)&print_ans[p] , sizeof(int));
    }
    out_fs.close();
}
void Send_f(int x){
    for(auto item : edges[x]){
        int buf[2] = {item.first , item.second + ans[x]};
        if(buf[0] % procs_size == procs_rank){
            vec.push_back(std::make_pair(buf[0] , buf[1]));
        }else{
            MPI_Send(buf, 2, MPI_INT, buf[0] % procs_size , 2 , MPI_COMM_WORLD);
        }
    }
}
void Recv_f(int x){
    int buf[2];
    for(int i = 0;i < recv_size[x];i++){
        MPI_Recv(buf, 2, MPI_INT, x % procs_size, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        vec.push_back(std::make_pair(buf[0] , buf[1]));
    }
}
void* cal(void* threadid) {
    int* tid = (int*)threadid;
    for(int i = *tid;i < (int)(vec.size());i += (MAX_THREADS - 1)){
        int dest = vec[i].first , p = (int)(dest / procs_size) , val = vec[i].second;
        pthread_mutex_lock(&ans_mutex[p]);
            if(val < ans[p] || ans[p] == -1){
                ans[p] = val;
            }
        pthread_mutex_unlock(&ans_mutex[p]);
    }
    pthread_exit(NULL);
}
void get_mini_ss(int* buf , int* recv_buf){
    buf[0] = 0; buf[1] = -1;
    int tmp = 0;
    for(int i = procs_rank;i < num_vertices;i += procs_size){
        if(tf[tmp] == false && ans[tmp] != -1){
            if(buf[1] == -1 || ans[tmp] < buf[1]){
                buf[0] = i; buf[1] = ans[tmp];
            }
        }
        tmp += 1;
    }
    if(recv_buf == NULL || recv_buf[1] == -1){
        return;
    }else if(buf[1] == -1 || recv_buf[1] < buf[1]){
        buf[0] = recv_buf[0]; buf[1] = recv_buf[1];
    }
}
int main(int argc, char** argv){

    in_filename = argv[1];
    out_filename = argv[2];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &procs_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procs_size);

    read_input_file();
    
    int prev_proc = (procs_rank == 0)? procs_size - 1 : procs_rank - 1;
    int next_proc = (procs_rank == procs_size - 1)? 0 : procs_rank + 1;

    for(int i = 0;i < num_vertices;i++){
        int buf[2] , recv_buf[2];
        if(procs_size == 1){
            get_mini_ss(buf , NULL);
        }else{
            if(procs_rank == 0){
                get_mini_ss(buf , NULL);
                MPI_Send(buf, 2, MPI_INT, next_proc, 0 , MPI_COMM_WORLD);
                MPI_Recv(recv_buf, 2, MPI_INT, prev_proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }else{
                MPI_Recv(recv_buf, 2, MPI_INT, prev_proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                get_mini_ss(buf , recv_buf);
                MPI_Send(buf, 2, MPI_INT, next_proc, 0 , MPI_COMM_WORLD);
            }
            if(procs_rank == 0){
                MPI_Send(recv_buf, 2, MPI_INT, next_proc, 1 , MPI_COMM_WORLD);
                MPI_Recv(buf, 2, MPI_INT, prev_proc, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }else{
                MPI_Recv(buf, 2, MPI_INT, prev_proc, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(buf, 2, MPI_INT, next_proc, 1, MPI_COMM_WORLD);
            }
        }
        if(buf[0] % procs_size == procs_rank){
            tf[(int)(buf[0] / procs_size)] = true;
            Send_f((int)(buf[0] / procs_size));
        }else{
            Recv_f(buf[0]);
        }
        for(int j = 0;j < MAX_THREADS - 1;j++){
            pthread_create(&threads[j], NULL, cal, (void*)&TID[j]);
        }
        for(int j = 0;j < MAX_THREADS - 1;j++){
            pthread_join(threads[j], NULL);
        }
        vec.clear();
    }

    int recv_size = 0;
    if(procs_rank == 0){
        recv_size = ((int)(num_vertices / procs_size) + 1) * procs_size;
        print_ans = (int*)malloc(recv_size * sizeof(int));
    }
    MPI_Gather(ans , (int)(num_vertices / procs_size) + 1 , MPI_INT , print_ans , (int)(num_vertices / procs_size) + 1 , MPI_INT , 0 , MPI_COMM_WORLD);
    MPI_Finalize();

    if(procs_rank == 0){
        write_output();
    } 

    return 0;
}

