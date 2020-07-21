#define PNG_NO_SETJMP

#include <omp.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_ITER 10000

int chunk = 50 , num_threads;

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
            #pragma omp for schedule(static , chunk) nowait firstprivate(buffer , width , row , var) lastprivate(row)
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
    num_threads = strtol(argv[1], 0, 10);
    double left = strtod(argv[2], 0);
    double right = strtod(argv[3], 0);
    double lower = strtod(argv[4], 0);
    double upper = strtod(argv[5], 0);
    int width = strtol(argv[6], 0, 10);
    int height = strtol(argv[7], 0, 10);
    const char* filename = argv[8];

    /* allocate memory for image */
    int* image = (int*)malloc(width * height * sizeof(int));
    assert(image);    

    chunk = (int)(width * height / num_threads / 10000) + 1;
    
    /* mandelbrot set */
    #pragma omp parallel shared(image)
    {
        #pragma omp for schedule(dynamic , chunk) nowait collapse(2) firstprivate(count , image , left , right , lower , upper , width , height) lastprivate(count , image)
        for (int j = 0; j < height; ++j) {
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
                count += repeats;
            }
        }
        printf("threads no. : %d count : %lld\n" , omp_get_thread_num() , count);
    }

    /* draw and cleanup */
    chunk = (int)(width / (num_threads)) + 1;
    omp_set_num_threads(num_threads);
    write_png(filename, width, height, image);
    free(image);
    //time
    time(&time_end);
    if(rank == 0){
        printf("%lf\n" , difftime(time_end , time_begin));
    }
    return 0;
}
