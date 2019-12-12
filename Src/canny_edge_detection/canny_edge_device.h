/*************************************************************************
/* ECE 285: GPU Programmming 2019 Fall quarter
/* Author: Group_G
/* Copyright 2019
/* University of California, San Diego
/*************************************************************************/

#ifndef _CANNY_EDGE_DEVICE_H
#define _CANNY_EDGE_DEVICE_H

#include <cuda_fp16.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include "common.h"

#define TILE_WIDTH 32
#define SMEM_SIZE 128
#define MAX_THREADS_PER_BLOCK 1024

#define CLOCK_CUDA_INIT(start, stop)                                           \
{                                                                              \
    CHECK(cudaEventCreate(&start));                                            \
    CHECK(cudaEventCreate(&stop));                                             \
}
#define TIC_CUDA(start)                                                        \
{                                                                              \
	CHECK(cudaEventRecord(start));                                             \
}
#define TOC_CUDA(stop)                                                         \
{                                                                              \
	CHECK(cudaEventRecord(stop));                                              \
	CHECK(cudaEventSynchronize(stop));                                         \
}
#define TIME_DURATION_CUDA(miliseconds, start, stop)                           \
{                                                                              \
	CHECK(cudaEventElapsedTime(&miliseconds, start, stop));                    \
}

class canny_edge_device {
	// input image
	float *image;

	float *gaussian_kernel;
	// image with noise reduced
	float *gaussiated_image;

	float *sobel_filter_x;
	float *sobel_filter_y;
	// image with gradients in x (horizontal) direction
	float *sobeled_grad_x_image;
	// image with gradients in y (vertical) direction
	float *sobeled_grad_y_image;
	// image with gradient magnitude
	float *sobeled_mag_image;
	// image with gradient direction
	float *sobeled_dir_image;

	// image with non-maximal suppression
	float *non_max_suppressed_image;

	// image with double thresholds applied
	float *double_thresholded_image;

	// image with edge tracking using hysteresis applied
	float *edge_tracked_image;

	int width;
	int height;
	float strong_pixel_threshold;
	float weak_pixel_threshold;

	float total_time_taken;

	//__global__ void init_gaussian_kernel_cuda(float *gaussian_kernel);
	void init_gaussian_kernel();
	//__global__ void init_sobel_filters_cuda(float *sobel_filter_x, float *sobel_filter_y);
	void init_sobel_filters();
	
public:
	canny_edge_device(float *image, int width, int height);
	~canny_edge_device();

	int get_width();
	int get_height();
	float get_total_time_taken();
	float *get_gaussian_kernel();
	float *get_sobel_filter_x();
	float *get_sobel_filter_y();
	float *get_gaussiated_image();
	float *get_sobeled_grad_x_image();
	float *get_sobeled_grad_y_image();
	float *get_sobeled_mag_image();
	float *get_sobeled_dir_image();
	float *get_non_max_suppressed_image();
	float *get_double_thresholded_image();
	float *get_edge_tracked_image();

	void print_gaussian_kernel();
	void print_sobel_filters();
	void print_gaussiated_image();

	//__global__ void do_convolution(float *image, int image_width, int image_height, float *kernel, int kernel_size, float *result);
	void apply_gaussian_kernel();
	void compute_pixel_thresholds();
	void apply_sobel_filter_x();
	void apply_sobel_filter_y();
	void streamed_apply_sobel_filter_x_y();

	//__global__ void calculate_sobel_magnitude_cuda(float *sobeled_grad_x_image, float *sobeled_grad_y_image, float *sobeled_mag_image, int image_width, int image_height);
	void calculate_sobel_magnitude();
	//__global__ void calculate_sobel_direction_cuda(float *sobeled_grad_x_image, float *sobeled_grad_y_image, float *sobeled_mag_image, int image_width, int image_height);
	void calculate_sobel_direction();
	void streamed_calculate_sobel_magnitude_direction();
	
	void apply_non_max_suppression();
	void apply_double_thresholds();
	void apply_hysteresis_edge_tracking();
};

#endif //_CANNY_EDGE_DEVICE_H