/*************************************************************************
/* ECE 285: GPU Programmming 2019 Fall quarter
/* Author: Group_G
/* Copyright 2019
/* University of California, San Diego
/*************************************************************************/

#ifndef __CANNY_EDGE_HOST_H
#define __CANNY_EDGE_HOST_H

#include "utils.h"

class canny_edge_host {
	float *image;

	float *gaussian_kernel;
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
};

#endif //__CANNY_EDGE_HOST_H