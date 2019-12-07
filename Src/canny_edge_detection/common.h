/*************************************************************************
/* ECE 285: GPU Programmming 2019 Fall quarter
/* Author: Group_G
/* Copyright 2019
/* University of California, San Diego
/*************************************************************************/

#ifndef _CUSTOM_COMMON_H
#define _CUSTOM_COMMON_H

#define _CRT_SECURE_NO_WARNINGS
#include <time.h>
#include <windows.h>

#if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
#define DELTA_EPOCH_IN_MICROSECS  11644473600000000Ui64
#else
#define DELTA_EPOCH_IN_MICROSECS  11644473600000000ULL
#endif

#include <stdlib.h>
#include <cstring>
#include <stdio.h>
#include <assert.h>
#define _USE_MATH_DEFINES // for M_PI
#include <math.h>

// square shape
#define GAUSSIAN_KERNEL_SIZE 3
#define SOBEL_FILTER_SIZE 5

#define STRONG_PIXEL_THRESHOLD 0.66f
#define STRONG_PIXEL_VALUE 1.0f

#define WEAK_PIXEL_THRESHOLD 0.33f
#define WEAK_PIXEL_VALUE 0.1f

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

/* **************************************************************************************************** */

/* get time of the day
*/
inline int gettimeofday(struct timeval *tv, struct timezone *tz)
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

        tmpres /= 10;  
        
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

/* **************************************************************************************************** */

/*  print a matrix given rows and columns to log file
*/
inline void print_log_matrix(FILE *f, float *image, int width, int height) {
	assert(f != NULL);

	assert(image != NULL);

	for (int r = 0; r < height; r++) {
		for (int c = 0; c < width; c++) {
			fprintf(f, "%.2f ", image[r * width + c]);
			printf("%.2f ", image[r * width + c]);
		}
		fprintf(f, "\n");
		printf("\n");
	}
}

/* **************************************************************************************************** */

/*  print a matrix given rows and columns
*/
inline void print_matrix(float *image, int width, int height){
	
	assert(image != NULL);

    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
			printf("%.2f ", image[r * width + c]);
        }
        printf("\n");
    }
}

/* **************************************************************************************************** */

/*  checks if an index is in bounds or not
*/
inline bool is_index_correct(int index, int max_elements){
    if (index >= 0 && index < max_elements)
        return true;
    else
        return false;
}

#endif //_CUSTOM_COMMON_H