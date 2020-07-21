#include <mpi.h>
#include <algorithm> 

int main(int argc, char *argv[]){
    int rank, size;
    int leng = atof(argv[1]);
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    MPI_Group orig_group, new_group;
    MPI_Comm new_comm;
    MPI_Comm_group(MPI_COMM_WORLD, &orig_group);
    int ranks1[2] = {0 , 1} , ranks2[2] = {rank - 1 , rank} , ranks3[3] = {rank - 1 , rank , rank + 1};
    if(size == 1){
        orig_group = new_group;
    }else if(rank == 0){
        MPI_Group_incl(orig_group, 2, ranks1, &new_group);
    }else if(rank == size - 1){
        MPI_Group_incl(orig_group, 2, ranks2, &new_group);
    }else{
        MPI_Group_incl(orig_group, 3, ranks3, &new_group);
    }
    MPI_Comm_create(MPI_COMM_WORLD, new_group, &new_comm);

    int leng_size = int(leng / size);
    int buf_size , recv_buf_size[2] , st;
    if(leng % size == 0){
        buf_size = leng_size , recv_buf_size[0] = leng_size , recv_buf_size[1] = leng_size;
        st = leng_size * rank;
    }else{
        buf_size = (rank < leng % size)? leng_size + 1 : leng_size;
        recv_buf_size[0] = ((rank + 1) < leng % size)? leng_size + 1 : leng_size;
        recv_buf_size[1] = ((rank - 1) < leng % size)? leng_size + 1 : leng_size;
        st = (rank < leng % size)? (leng_size + 1) * rank : leng - leng_size * (size - rank);
    }

    if(size > leng && rank >= leng % size){
        st = leng - 1 ; buf_size = 0;
    }

    float *buf[2] , *recv_buf;
    buf[0] = (float*)malloc(buf_size * sizeof(float));
    buf[1] = (float*)malloc(buf_size * sizeof(float));
    recv_buf = (float*)malloc((leng_size + 1) * sizeof(float));

    bool tf , all_tf[2] = {false , false};

    MPI_File input;
    MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_RDONLY , MPI_INFO_NULL, &input);
    MPI_File_read_at(input , st * sizeof(float) , buf[0] , buf_size , MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&input);
    std::sort(buf[0] , buf[0] + buf_size);

    MPI_Request send_request , recv_request;
    int k = 1 , k1 = 0 , j , recv_state , j_mod_2; //recv_state == -1 nothing , == 0 with rank + 1 , == 1 with rank - 1

    for(j = 0;; j++){
        j_mod_2 = j % 2;
        tf = true; all_tf[j_mod_2] = true;
        recv_state = -1; 
        if(!(rank == 0 && j_mod_2 == 1) && !(rank == size - 1 && j_mod_2 == rank % 2)){
            k = 1 - k , k1 = 1 - k1;
        }
        if(j_mod_2 == rank % 2 && rank + 1 < size){
            // MPI_Isend(buf[k] , buf_size , MPI_FLOAT , rank + 1 , j , MPI_COMM_WORLD , &send_request);
            // MPI_Irecv(recv_buf , recv_buf_size[0] , MPI_FLOAT , rank + 1 , j , MPI_COMM_WORLD , &recv_request);
            MPI_Isend(buf[k] , buf_size , MPI_FLOAT , rank + 1 , j , new_comm , &send_request);
            MPI_Irecv(recv_buf , recv_buf_size[0] , MPI_FLOAT , rank + 1 , j , new_comm , &recv_request);
            recv_state = 0;
        }else if(j_mod_2 != rank % 2 && rank - 1 >= 0){
            // MPI_Isend(buf[k] , buf_size , MPI_FLOAT , rank - 1 , j , MPI_COMM_WORLD , &send_request);
            // MPI_Irecv(recv_buf , recv_buf_size[1] , MPI_FLOAT , rank - 1 , j , MPI_COMM_WORLD , &recv_request);
            MPI_Isend(buf[k] , buf_size , MPI_FLOAT , rank - 1 , j , new_comm , &send_request);
            MPI_Irecv(recv_buf , recv_buf_size[1] , MPI_FLOAT , rank - 1 , j , new_comm , &recv_request);
            recv_state = 1;            
        }

        if(recv_state >= 0){
            MPI_Wait(&recv_request , MPI_STATUS_IGNORE);

            int buf_p = 0 , recv_p = 0;

            if(recv_state == 0){
                for(int i = 0;i < buf_size;i++){
                    if(buf_p >= buf_size){ buf[k1][i] = recv_buf[recv_p++];
                    }else if(recv_p >= recv_buf_size[recv_state]){ buf[k1][i] = buf[k][buf_p++];
                    }else if(buf[k][buf_p] < recv_buf[recv_p]){ buf[k1][i] = buf[k][buf_p++];
                    }else{ buf[k1][i] = recv_buf[recv_p++]; }

                    if(buf[k1][i] != buf[k][i]){ tf = false; }
                }
            }else{
                for(int i = 0;i < recv_buf_size[recv_state];i++){
                    if(buf_p >= buf_size){ recv_p++;
                    }else if(recv_p >= recv_buf_size[recv_state]){ buf_p++;
                    }else if(buf[k][buf_p] < recv_buf[recv_p]){ buf_p++;
                    }else{ recv_p++; }
                }
                for(int i = 0;i < buf_size;i++){
                    if(buf_p >= buf_size){ buf[k1][i] = recv_buf[recv_p++];
                    }else if(recv_p >= recv_buf_size[recv_state]){ buf[k1][i] = buf[k][buf_p++];
                    }else if(buf[k][buf_p] < recv_buf[recv_p]){ buf[k1][i] = buf[k][buf_p++];
                    }else{ buf[k1][i] = recv_buf[recv_p++]; }

                    if(buf[k1][i] != buf[k][i]){ tf = false; }
                }
            }
        }

        MPI_Allreduce(&tf , &all_tf[j_mod_2] , 1 , MPI_C_BOOL , MPI_LAND , MPI_COMM_WORLD);

        if(all_tf[0] & all_tf[1] == true){
            break;
        }
    }
    
    MPI_File output;
    MPI_File_open(MPI_COMM_WORLD, argv[3], MPI_MODE_CREATE | MPI_MODE_WRONLY , MPI_INFO_NULL, &output);
    MPI_File_write_at(output , st * sizeof(float) , buf[k1] , buf_size , MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&output);
    
    MPI_Finalize();
    return 0;
}