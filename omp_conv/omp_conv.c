#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <omp.h> // Include OpenMP header

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#ifdef _WIN32
#define FSEEKO _fseeki64
#else
#define FSEEKO fseeko
#endif

typedef enum
{
    RGB,
    GREY
} color_t;

// --- Function Declarations ---
void usage_omp(int argc, char **argv, char **image_filename, int *width, int *height, int *loops, color_t *imageType, int *num_threads);
uint8_t *offset_omp(uint8_t *array, int i, int j_byte_offset, int padded_width_bytes);
void convolute_pixel_grey(uint8_t *src, uint8_t *dst, int i, int j, int padded_width_bytes, float **h);
void convolute_pixel_rgb(uint8_t *src, uint8_t *dst, int i, int j_byte_start, int padded_width_bytes, float **h);

// --- Main Function with OpenMP ---
int main(int argc, char **argv)
{
    char *image_filename = NULL;
    int width = 0, height = 0, loops = 0, num_threads = 1;
    color_t imageType = GREY;
    int image_channels = 1;
    int i, j, t;

    double start_time_total, end_time_total;
    double start_time_conv, end_time_conv;
    double cpu_time_total, cpu_time_conv;

    start_time_total = omp_get_wtime(); // Using OpenMP timer

    // === Phase 1: Initialization ===
    usage_omp(argc, argv, &image_filename, &width, &height, &loops, &imageType, &num_threads);
    image_channels = (imageType == GREY) ? 1 : 3;
    printf("[OMP] Executing with %d threads: Image=%s, Size=%dx%d, Loops=%d, Type=%s\n",
           num_threads, image_filename, width, height, loops, (imageType == GREY ? "GREY" : "RGB"));

    // Set number of threads
    omp_set_num_threads(num_threads);

    // Initialize Filter (Kernel)
    float **h = malloc(3 * sizeof(float *));
    if (!h)
    {
        fprintf(stderr, "[OMP] Error allocating filter rows\n");
        return EXIT_FAILURE;
    }

#pragma omp parallel for
    for (i = 0; i < 3; i++)
    {
        h[i] = malloc(3 * sizeof(float));
        if (!h[i])
        {
            fprintf(stderr, "[OMP] Error allocating filter cols\n"); /* cleanup */
        }
    }

    float gaussian_blur[3][3] = {
        {1.f / 16, 2.f / 16, 1.f / 16},
        {2.f / 16, 4.f / 16, 2.f / 16},
        {1.f / 16, 2.f / 16, 1.f / 16}};

#pragma omp parallel for collapse(2)
    for (i = 0; i < 3; i++)
    {
        for (j = 0; j < 3; j++)
        {
            h[i][j] = gaussian_blur[i][j];
        }
    }

    // Calculate Padded Size
    int padded_rows = height + 2;
    int padded_cols = width + 2;
    int padded_width_bytes = padded_cols * image_channels;
    size_t padded_buffer_size = (size_t)padded_rows * padded_width_bytes;

    // Allocate Image Buffers (With Padding)
    uint8_t *src_padded = NULL, *dst_padded = NULL, *tmp_padded = NULL;
    src_padded = calloc(1, padded_buffer_size);
    dst_padded = calloc(1, padded_buffer_size);
    if (!src_padded || !dst_padded)
    {
        fprintf(stderr, "[OMP] Error allocating image buffers (size %zu)\n", padded_buffer_size);
        return EXIT_FAILURE;
    }
    printf("[OMP] Allocated padded buffers (%dx%d bytes each)\n", padded_rows, padded_width_bytes);

    // === Phase 2: Input ===
    FILE *f_in = fopen(image_filename, "rb");
    if (!f_in)
    {
        fprintf(stderr, "[OMP] Error opening input file: %s\n", image_filename);
        return EXIT_FAILURE;
    }
    printf("[OMP] Reading raw image data...\n");
    size_t row_bytes_to_read = (size_t)width * image_channels;

// Read image data in parallel by chunks
#pragma omp parallel for private(i)
    for (i = 0; i < height; ++i)
    {
        uint8_t *row_ptr_in_padded = offset_omp(src_padded, i + 1, image_channels, padded_width_bytes);
#pragma omp critical
        {
            FSEEKO(f_in, (long long)i * row_bytes_to_read, SEEK_SET);
            size_t bytes_read = fread(row_ptr_in_padded, 1, row_bytes_to_read, f_in);
            if (bytes_read != row_bytes_to_read)
            {
                fprintf(stderr, "[OMP] Error reading row %d. Read %zu, expected %zu\n",
                        i, bytes_read, row_bytes_to_read);
            }
        }
    }
    fclose(f_in);
    printf("[OMP] Finished reading.\n");

    // === Phase 3: Computation ===
    printf("[OMP] Starting convolution loops (%d)...\n", loops);
    start_time_conv = omp_get_wtime();

    for (t = 0; t < loops; ++t)
    {
        if (imageType == GREY)
        {
#pragma omp parallel for collapse(2) schedule(dynamic)
            for (i = 1; i <= height; ++i)
            {
                for (j = 1; j <= width; ++j)
                {
                    convolute_pixel_grey(src_padded, dst_padded, i, j, padded_width_bytes, h);
                }
            }
        }
        else
        { // RGB
#pragma omp parallel for collapse(2) schedule(dynamic)
            for (i = 1; i <= height; ++i)
            {
                for (j = 1; j <= width; ++j)
                {
                    int current_pixel_r_byte_offset = image_channels + j * image_channels;
                    convolute_pixel_rgb(src_padded, dst_padded, i, current_pixel_r_byte_offset,
                                        padded_width_bytes, h);
                }
            }
        }

        // Swap buffers for next iteration
        tmp_padded = src_padded;
        src_padded = dst_padded;
        dst_padded = tmp_padded;
    }

    end_time_conv = omp_get_wtime();
    cpu_time_conv = end_time_conv - start_time_conv;
    printf("[OMP] Convolution loops finished. Time elapsed: %f seconds\n", cpu_time_conv);

    // === Phase 4: Output Preparation ===
    printf("[OMP] Preparing final image buffer (removing padding)...\n");
    size_t final_image_bytes = (size_t)width * height * image_channels;
    uint8_t *final_image_buffer = malloc(final_image_bytes);
    if (!final_image_buffer)
    {
        fprintf(stderr, "[OMP] Error allocating final image buffer\n");
        return EXIT_FAILURE;
    }

// Copy data from src_padded to final_image_buffer in parallel
#pragma omp parallel for
    for (i = 0; i < height; ++i)
    {
        uint8_t *src_row_start = offset_omp(src_padded, i + 1, image_channels, padded_width_bytes);
        uint8_t *dst_row_start = final_image_buffer + (size_t)i * width * image_channels;
        memcpy(dst_row_start, src_row_start, row_bytes_to_read);
    }
    printf("[OMP] Final buffer prepared.\n");

    // === Phase 5: Output ===
    char png_filename[512];
    char *image_basename = strrchr(image_filename, '\\');
    if (!image_basename)
        image_basename = strrchr(image_filename, '/');
    if (!image_basename)
        image_basename = image_filename;
    else
        image_basename++;
    snprintf(png_filename, sizeof(png_filename), "blur_OMP_%s.png", image_basename);

    printf("[OMP] Writing PNG file: %s\n", png_filename);
    int stride_in_bytes = width * image_channels;
    int success = stbi_write_png(png_filename, width, height, image_channels,
                                 final_image_buffer, stride_in_bytes);
    if (!success)
    {
        fprintf(stderr, "[OMP] Error writing PNG file.\n");
    }
    else
    {
        printf("[OMP] Successfully wrote PNG file.\n");
    }

    // === Phase 6: Cleanup ===
    printf("[OMP] Cleaning up memory...\n");
#pragma omp parallel sections
    {
#pragma omp section
        {
            free(src_padded);
        }
#pragma omp section
        {
            free(dst_padded);
        }
#pragma omp section
        {
            free(final_image_buffer);
        }
    }

#pragma omp parallel for
    for (i = 0; i < 3; ++i)
    {
        if (h[i])
            free(h[i]);
    }
    free(h);
    free(image_filename);

    end_time_total = omp_get_wtime();
    cpu_time_total = end_time_total - start_time_total;
    printf("[OMP] Total execution time: %f seconds\n", cpu_time_total);

    printf("[OMP] OpenMP execution finished.\n");
    return EXIT_SUCCESS;
}

// --- Function Definitions ---

void usage_omp(int argc, char **argv, char **image_filename, int *width, int *height,
               int *loops, color_t *imageType, int *num_threads)
{
    if (argc == 7 && (!strcmp(argv[5], "grey") || !strcmp(argv[5], "rgb")))
    {
        *image_filename = malloc((strlen(argv[1]) + 1) * sizeof(char));
        if (!*image_filename)
        {
            fprintf(stderr, "Usage Error: Failed malloc image name.\n");
            exit(EXIT_FAILURE);
        }
        strcpy(*image_filename, argv[1]);
        *width = atoi(argv[2]);
        *height = atoi(argv[3]);
        *loops = atoi(argv[4]);
        *imageType = (!strcmp(argv[5], "grey")) ? GREY : RGB;
        *num_threads = atoi(argv[6]);

        if (*width <= 0 || *height <= 0 || *loops < 0 || *num_threads <= 0)
        {
            fprintf(stderr, "\nUsage Error: width/height/threads must be positive, loops non-negative.\n");
            free(*image_filename);
            exit(EXIT_FAILURE);
        }
    }
    else
    {
        fprintf(stderr, "\nInput Error!\nUsage: %s image_name width height loops [rgb|grey] num_threads\n\n",
                argv[0]);
        exit(EXIT_FAILURE);
    }
}

uint8_t *offset_omp(uint8_t *array, int i, int j_byte_offset, int padded_width_bytes)
{
    return &array[(size_t)padded_width_bytes * i + j_byte_offset];
}

// Note: These convolution functions remain unchanged as they operate on individual pixels
void convolute_pixel_grey(uint8_t *src, uint8_t *dst, int i, int j, int padded_width_bytes, float **h)
{
    float val = 0.0f;
    int k, l, ki, lj;
    for (k = -1, ki = 0; k <= 1; ++k, ++ki)
    {
        for (l = -1, lj = 0; l <= 1; ++l, ++lj)
        {
            int neighbor_row = i + k;
            int neighbor_col_byte = j + l;
            size_t index = (size_t)padded_width_bytes * neighbor_row + neighbor_col_byte;
            val += src[index] * h[ki][lj];
        }
    }
    size_t dst_index = (size_t)padded_width_bytes * i + j;
    if (val < 0.0f)
        dst[dst_index] = 0;
    else if (val > 255.0f)
        dst[dst_index] = 255;
    else
        dst[dst_index] = (uint8_t)(val + 0.5f);
}

void convolute_pixel_rgb(uint8_t *src, uint8_t *dst, int i, int j_byte_start_dst,
                         int padded_width_bytes, float **h)
{
    float r = 0.f, g = 0.f, b = 0.f;
    int k, l, ki, lj;
    for (k = -1, ki = 0; k <= 1; ++k, ++ki)
    {
        int neighbor_row = i + k;
        for (l = -1, lj = 0; l <= 1; ++l, ++lj)
        {
            int j_byte_offset_neighbor = j_byte_start_dst + (l * 3);
            size_t index_r = (size_t)padded_width_bytes * neighbor_row + j_byte_offset_neighbor;
            r += src[index_r] * h[ki][lj];
            g += src[index_r + 1] * h[ki][lj];
            b += src[index_r + 2] * h[ki][lj];
        }
    }
    if (r < 0.f)
        r = 0.f;
    else if (r > 255.f)
        r = 255.f;
    dst[j_byte_start_dst] = (uint8_t)(r + 0.5f);
    if (g < 0.f)
        g = 0.f;
    else if (g > 255.f)
        g = 255.f;
    dst[j_byte_start_dst + 1] = (uint8_t)(g + 0.5f);
    if (b < 0.f)
        b = 0.f;
    else if (b > 255.f)
        b = 255.f;
    dst[j_byte_start_dst + 2] = (uint8_t)(b + 0.5f);
}