#include <mpi.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <algorithm> 
#include <utility>

int main(int argc, char *argv[]){
    int rank, size;
    int leng = atof(argv[1]);
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int buf_size = (leng % size == 0)? int(leng / size) :
                   (rank < leng % size)? int(leng / size) + 1 : int(leng / size);
    int st = (leng % size == 0)? int(leng / size) * rank :
             (rank < leng % size)? (int(leng / size) + 1) * rank : leng - int(leng / size) * (size - rank);
    int end = st + buf_size - 1;

    if(size > leng && rank >= leng % size){
        st = leng - 1 ; end = st - 1 ; buf_size = 0;
    }

    float *buf;
    buf = (float*)malloc(buf_size * sizeof(float));
    bool tf , all_tf[2] = {false , false};

    MPI_File input; MPI_Status input_status;
    MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_RDONLY , MPI_INFO_NULL, &input);
    MPI_File_read_at_all(input , st * sizeof(float) , buf , buf_size , MPI_FLOAT, &input_status);
    MPI_File_close(&input);

    MPI_Request st_request , st_request_1 , end_request , end_request_1; MPI_Status st_status , end_status;
    float recv_st_float , recv_end_float; bool recv_st_tf , recv_end_tf; int recv_st_p , recv_end_p;

    for(int j = 0; j < leng; j++){
        tf = true; all_tf[j % 2] = true;
        recv_st_tf = false , recv_end_tf = false;
        for(int p = st;p <= end;p++){
            int state = (buf_size == 1)? ((p % 2 == j % 2)? 2 : 1) :
                        (p == st)? ((p % 2 == j % 2)? 3 : 1) :
                        (p == end)? 2 : 3; //p == 1 st , p == 2 end , p == 3 mid
            
            if(state == 1){
                if(p != 0){
                    //send , recieve
                    float recv_float;
                    MPI_Isend(&buf[p - st] , 1 , MPI_FLOAT , rank - 1 , j , MPI_COMM_WORLD , &st_request_1);
                    MPI_Irecv(&recv_st_float , 1 , MPI_FLOAT , rank - 1 , j , MPI_COMM_WORLD , &st_request);
                    recv_st_tf = true; recv_st_p = p - st;
                }
            }else if(state == 2){
                if(p % 2 == j % 2 && p + 1 < leng){
                    //send , recieve
                    float recv_float;
                    MPI_Isend(&buf[p - st] , 1 , MPI_FLOAT , rank + 1 , j , MPI_COMM_WORLD , &end_request_1);
                    MPI_Irecv(&recv_end_float , 1 , MPI_FLOAT , rank + 1 , j , MPI_COMM_WORLD , &end_request);
                    recv_end_tf = true; recv_end_p = p - st;
                }
            }else if(p % 2 == j % 2){
                //swap
                if(buf[p - st] > buf[p + 1 - st]){
                    std::swap(buf[p - st] , buf[p + 1 - st]); tf = false;
                }
            }
        }

        if(recv_st_tf == true){
            MPI_Wait(&st_request , &st_status);
            if(recv_st_float > buf[recv_st_p]){
                buf[recv_st_p] = recv_st_float; tf = false;
            }
        }
        
        if(recv_end_tf == true){
            MPI_Wait(&end_request , &end_status);
            if(recv_end_float < buf[recv_end_p]){
                buf[recv_end_p] = recv_end_float; tf = false;
            }
        }

        MPI_Allreduce(&tf , &all_tf[j % 2] , 1 , MPI_C_BOOL , MPI_LAND , MPI_COMM_WORLD);

        if(all_tf[0] & all_tf[1] == true){
            break;
        }
    }
    
    MPI_File output; MPI_Status output_status;
    MPI_File_open(MPI_COMM_WORLD, argv[3], MPI_MODE_CREATE | MPI_MODE_WRONLY , MPI_INFO_NULL, &output);
    MPI_File_write_at_all(output , st * sizeof(float) , buf , buf_size , MPI_FLOAT, &output_status);
    MPI_File_close(&output);
    
    MPI_Finalize();

    return 0;
}