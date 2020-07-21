#define PNG_NO_SETJMP

#include <mpi.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

int main(int argc, char** argv) {
    //time start
    time_t time_begin , time_mid , time_end;
    time(&time_begin);
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

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* allocate memory for image */
    int* image = (int*)malloc(width * height * sizeof(int));
    int* all_image = (int*)malloc(width * height * sizeof(int));
    assert(image); assert(all_image);
    memset (image , 0 , sizeof(image));

    /* mandelbrot set */
    for (int j = rank; j < height; j = j + size) {
        for (int i = 0; i < width; ++i) {
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
    time(&time_mid);

    MPI_Reduce(image , all_image , width * height , MPI_INT , MPI_SUM , 0 , MPI_COMM_WORLD);    

    /* draw and cleanup */
    if(rank == 0){
        write_png(filename, width, height, all_image);
    }
    
    free(image);
    free(all_image);
    
    MPI_Finalize();
    //time end
    time(&time_end);
    double time_anal = difftime(time_end , time_begin);
    if(rank == 0)
        printf("total : %lf\n" , time_anal);
    printf("rank : %d time : %lf\n" , rank , difftime(time_mid , time_begin));
    return 0;
}


