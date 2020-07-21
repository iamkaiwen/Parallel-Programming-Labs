#define PNG_NO_SETJMP
#define BLOCKSIZE ((int)(area - ptr) / (size * 10)) + 1
#define MAKEDATABUF buf[0] = ptr , buf[1] = BLOCKSIZE

#include <mpi.h>
#include <omp.h>
#include <assert.h>
#include <png.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_ITER 10000

void write_png(const char* filename, const int width, const int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png_ptr, info_ptr);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        int var = height - 1 - y;
        #pragma omp parallel shared(row)
        {
            #pragma omp for schedule(static , 5) nowait firstprivate(buffer , width , row , var) lastprivate(row)
            for (int x = 0; x < width; ++x) {
                int p = buffer[var * width + x];
                png_bytep color = row + x * 3;
                if (p != MAX_ITER) {
                    if (p & 16) {
                        color[0] = 240;
                        color[1] = color[2] = p % 16 * 16;
                    } else {
                        color[0] = p % 16 * 16;
                    }
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

int main(int argc, char** argv) {
    //time
    time_t time_begin , time_end;
    time(&time_begin);
    long long int count = 0;

    /* argument parsing */
    assert(argc == 9);
    int num_threads = strtol(argv[1], 0, 10);
    double left = strtod(argv[2], 0);
    double right = strtod(argv[3], 0);
    double lower = strtod(argv[4], 0);
    double upper = strtod(argv[5], 0);
    int width = strtol(argv[6], 0, 10);
    int height = strtol(argv[7], 0, 10);
    const char* filename = argv[8];
    int area = width * height;

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* allocate memory for image */
    int* image = (int*)malloc(area * sizeof(int));
    int* all_image = (int*)malloc(area * sizeof(int));
    assert(image); assert(all_image);
    memset (image , 0 , sizeof(image));
    
    /* allocate memory for MPI COMM */
    int buf[2] , recv_buf[2];
    MPI_Status recv_status;
    //0 : data_tag 1 : result_tag , terminate_tag

    if(size == 0){
        /* mandelbrot set */
        for (int j = 0; j < height; ++j) {
            double y0 = j * ((upper - lower) / height) + lower;
            for (int i = 0; i < width; ++i) {
                double x0 = i * ((right - left) / width) + left;

                int repeats = 0;
                double x = 0;
                double y = 0;
                double length_squared = 0;
                while (repeats < MAX_ITER && length_squared < 4) {
                    double temp = x * x - y * y + x0;
                    y = 2 * x * y + y0;
                    x = temp;
                    length_squared = x * x + y * y;
                    ++repeats;
                }
                image[j * width + i] = repeats;
            }
        }
    }else if(rank == 0){
        int active_procs = 0 , ptr = 0;
        for(int i = 1;i < size;i++){
            MAKEDATABUF;
            MPI_Send(buf , 2 , MPI_INT , i , 0 , MPI_COMM_WORLD);
            ptr += BLOCKSIZE;
            active_procs++;
        }
        do{
            MPI_Recv(recv_buf , 2 , MPI_INT , MPI_ANY_SOURCE , 1 , MPI_COMM_WORLD , &recv_status);
            int src = recv_status.MPI_SOURCE;
            if(ptr < area){
                MAKEDATABUF;
                MPI_Send(buf , 2 , MPI_INT , src , 0 , MPI_COMM_WORLD);
		ptr += BLOCKSIZE;
            }else{
                MPI_Send(buf , 2 , MPI_INT , src , 2 , MPI_COMM_WORLD);
                active_procs--;
            }
         }while(active_procs > 0);
    }else{
        MPI_Recv(recv_buf , 2 , MPI_INT , 0 , MPI_ANY_TAG , MPI_COMM_WORLD , &recv_status);
        int src_tag = recv_status.MPI_TAG;
        while(src_tag == 0){
            /* mandelbrot set */
            int end = (recv_buf[1] + recv_buf[0] > area)? area - recv_buf[0] + 1 : recv_buf[1];
	    #pragma omp parallel shared(image)
            {
                #pragma omp for schedule(dynamic , (int)(end / (25 * num_threads)) + 1) nowait firstprivate(end , image , left , right , lower , upper , width , height) lastprivate(image)
                for(int k = 0;k < end;k++){
                    int i = (recv_buf[0] + k) % width , j = (int)((recv_buf[0] + k) / width);
		    double x0 = i * ((right - left) / width) + left;
                    double y0 = j * ((upper - lower) / height) + lower;
                    int repeats = 0;
                    double x = 0 , y = 0 , length_squared = 0;
                    while (repeats < MAX_ITER && length_squared < 4) {
                        double temp = x * x - y * y + x0;
                        y = 2 * x * y + y0;
                        x = temp;
                        length_squared = x * x + y * y;
                        ++repeats;
                    }
                    image[j * width + i] = repeats;
                }
            }
            MPI_Send(buf , 2 , MPI_INT , 0 , 1 , MPI_COMM_WORLD);
            MPI_Recv(recv_buf , 2 , MPI_INT , 0 , MPI_ANY_TAG , MPI_COMM_WORLD , &recv_status);
            src_tag = recv_status.MPI_TAG;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(image , all_image , area , MPI_INT , MPI_SUM , 0 , MPI_COMM_WORLD);    

    /* draw and cleanup */
    if(rank == 0){
         write_png(filename, width, height, all_image);
    }
    
    free(image);
    free(all_image);
    
    MPI_Finalize();

    //time
    time(&time_end);
    if(rank == 0){
        printf("%lf\n" , difftime(time_end , time_begin));
    }

    return 0;
}
