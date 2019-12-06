/*************************************************************************
/* ECE 285: GPU Programmming 2019 Fall quarter
/* Author: Group_G
/* Copyright 2019
/* University of California, San Diego
/*************************************************************************/

/*	implementation of class canny_edge_host
*/
#include "canny_edge_host.h"

/* 	constructor
*/
canny_edge_host::canny_edge_host(float *image, int width, int height) {
	assert(width > 0);
	assert(height > 0);

	this->image = (float*)malloc(sizeof(float) * width * height);
	assert(this->image != NULL);
	memcpy(this->image, image, sizeof(float) * width * height);

	this->width = width;
	this->height = height;
	this->total_time_taken = 0.0f;
	
	// allocate for gaussian_kernel
	this->gaussian_kernel = (float*)malloc(sizeof(float) * GAUSSIAN_KERNEL_SIZE * GAUSSIAN_KERNEL_SIZE);
	assert(this->gaussian_kernel != NULL);

	// allocate for sobel filters
	this->sobel_filter_x = (float*)malloc(sizeof(float) * SOBEL_FILTER_SIZE * SOBEL_FILTER_SIZE);
	assert(this->sobel_filter_x != NULL);
	this->sobel_filter_y = (float*)malloc(sizeof(float) * SOBEL_FILTER_SIZE * SOBEL_FILTER_SIZE);
	assert(this->sobel_filter_y!= NULL);
}

/* **************************************************************************************************** */

/* 	destructor
*/
canny_edge_host::~canny_edge_host() {
	if (this->image != NULL) 			free(this->image);
	if (this->gaussian_kernel != NULL) 	free(this->gaussian_kernel);
}

/* **************************************************************************************************** */

/* 	initialize gaussian kernel
*/
void canny_edge_host::init_gaussian_kernel() {
	float stddev = 1.0f;
	float denominator = 2 * pow(stddev, 2);
	float sum = 0.0f;

	for (int i = 0; i < (GAUSSIAN_KERNEL_SIZE * GAUSSIAN_KERNEL_SIZE); i++){
		int ix = i % GAUSSIAN_KERNEL_SIZE;
		int iy = i / GAUSSIAN_KERNEL_SIZE;
		
		float numerator = pow(ix - (GAUSSIAN_KERNEL_SIZE/2), 2);
		numerator += pow(iy - (GAUSSIAN_KERNEL_SIZE/2), 2);

		this->gaussian_kernel[i] = exp( (-1 * numerator)/ denominator) / (M_PI * denominator);
		sum += this->gaussian_kernel[i];
	}

	// normalize
	for (int i = 0; i < (GAUSSIAN_KERNEL_SIZE * GAUSSIAN_KERNEL_SIZE); i++){
		this->gaussian_kernel[i] /= sum;
	}
}

/* **************************************************************************************************** */

/* 	initialize sobel filters
	https://stackoverflow.com/a/41065243/253056
*/
void canny_edge_host::init_sobel_filters() {
	int weight = SOBEL_FILTER_SIZE / 2;

	for (int i = 0; i < (SOBEL_FILTER_SIZE * SOBEL_FILTER_SIZE); i++){
		int ix = i % SOBEL_FILTER_SIZE;
		int iy = i / SOBEL_FILTER_SIZE;

		float denominator = pow(ix, 2) + pow(iy, 2);

		if (denominator == 0.0f){
			this->sobel_filter_x[i] = 0.0f;
			this->sobel_filter_y[i] = 0.0f;
		}
		else {
			this->sobel_filter_x[i] = ((ix - weight) * weight) / denominator;
			this->sobel_filter_y[i] = ((iy - weight) * weight) / denominator;
		}
	}
}