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
	float *image;

	float *gaussian_kernel;
	float *gaussiated_image;

	float *sobel_filter_x;
	float *sobel_filter_y;

	int width;
	int height;

	float total_time_taken;

	void init_gaussian_kernel();
	void init_sobel_filters();
	
public:
	canny_edge_host(float *image, int width, int height);
	~canny_edge_host();

	float get_total_time_taken();

	void print_gaussian_kernel();
	void print_sobel_filters();
	void print_gaussiated_image();

	void do_convolution(float *image, int image_width, int image_height, float *kernel, int kernel_size, float *result);
	void apply_gaussian_kernel();
};

#endif //_CANNY_EDGE_HOST_H