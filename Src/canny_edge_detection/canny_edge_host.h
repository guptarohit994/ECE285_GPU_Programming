/*************************************************************************
/* ECE 285: GPU Programmming 2019 Fall quarter
/* Author: Group_G
/* Copyright 2019
/* University of California, San Diego
/*************************************************************************/

#ifndef _CANNY_EDGE_HOST_H
#define _CANNY_EDGE_HOST_H

#include "common.h"

class canny_edge_host {
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

	void init_gaussian_kernel();
	void init_sobel_filters();
	
public:
	canny_edge_host(float *image, int width, int height);
	~canny_edge_host();

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

	void do_convolution(float *image, int image_width, int image_height, float *kernel, int kernel_size, float *result);
	void apply_gaussian_kernel();
	void compute_pixel_thresholds();
	void apply_sobel_filter_x();
	void apply_sobel_filter_y();
	void calculate_sobel_magnitude();
	void calculate_sobel_direction();
	void apply_non_max_suppression();
	void apply_double_thresholds();
	void apply_hysteresis_edge_tracking();
};

#endif //_CANNY_EDGE_HOST_H