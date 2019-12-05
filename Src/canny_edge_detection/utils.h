/*************************************************************************
/* ECE 285: GPU Programmming 2019 Fall quarter
/* Author: Group_G
/* Copyright 2019
/* University of California, San Diego
/*************************************************************************/

#include <time.h>
#include <windows.h>

#if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
#define DELTA_EPOCH_IN_MICROSECS  11644473600000000Ui64
#else
#define DELTA_EPOCH_IN_MICROSECS  11644473600000000ULL
#endif

#ifndef __CUSTOM_UTILS_H
#define __CUSTOM_UTILS_H

#include <cuda_fp16.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include <stdio.h>
#include <assert.h>
#define _USE_MATH_DEFINES // for M_PI
#include <math.h>

#include "imageLib/Image.h"
#include "imageLib/ImageIO.h"
#include "imageLib/Convert.h"

#include "gray_image.h"

// square shape
#define GAUSSIAN_KERNEL_SIZE 3
#define SOBEL_FILTER_SIZE 3

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}


struct timezone
{
    int  tz_minuteswest; /* minutes W of Greenwich */
    int  tz_dsttime;     /* type of dst correction */
};

/* convert CByteImage to a linearly stored image
*/
float *convert_CByteImage_to_array(CByteImage img) {
	CShape shape = img.Shape();
	float *array = (float *)malloc(sizeof(float) * shape.width * shape.height);
    assert(array != NULL);

	for (int x = 0; x < shape.width; x++) {
		for (int y = 0; y < shape.height; y++) {
			array[y * shape.width + x] = (img.Pixel(x,y,0))/255.0f;
		}
	}
	return array;
}

/* **************************************************************************************************** */

/* convert a linearly stored image to CByteImage
*/
CByteImage convert_array_to_CByteImage(float *img, int width, int height) {
	CByteImage cbimage(width, height, 1);

	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {
			cbimage.Pixel(x, y, 0) = (int)(img[y * width + x] * 255);
		}
	}
	return cbimage;
}

/* **************************************************************************************************** */

/* write image to file,
   image could either be in GPU or host memory
*/
void write_image_to_file(float *x_image, int width, int height, const char *file_name, bool from_device) {
	float *h_image;

	if (from_device) {
		h_image =  = (float *)malloc(sizeof(float) * width * height);
        assert(h_image != NULL);
		CHECK(cudaMemcpy(h_image, x_image, sizeof(float) * width * height, cudaMemcpyDeviceToHost));
	}
	else {
		h_image = x_image;
	}

	CByteImage cbimage = convert_array_to_CByteImage(h_image, width, height);
	WriteImage(cbimage, file_name);

	if (from_device)
        free(h_image);
}


/* **************************************************************************************************** */

/* get time of the day
*/
int gettimeofday(struct timeval *tv, struct timezone *tz)
{
    FILETIME ft;
    unsigned __int64 tmpres = 0;
    static int tzflag = 0;

    if (NULL != tv)
    {
        GetSystemTimeAsFileTime(&ft);

        tmpres |= ft.dwHighDateTime;
        tmpres <<= 32;
        tmpres |= ft.dwLowDateTime;

        tmpres /= 10;  /*convert into microseconds*/
        /*converting file time to unix epoch*/
        tmpres -= DELTA_EPOCH_IN_MICROSECS;
        tv->tv_sec = (long)(tmpres / 1000000UL);
        tv->tv_usec = (long)(tmpres % 1000000UL);
    }

    if (NULL != tz)
    {
        if (!tzflag)
        {
            _tzset();
            tzflag++;
        }
        tz->tz_minuteswest = _timezone / 60;
        tz->tz_dsttime = _daylight;
    }

    return 0;
}

/* **************************************************************************************************** */

/* get time in seconds
*/
inline double seconds()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}


#endif //__CUSTOM_UTILS_H