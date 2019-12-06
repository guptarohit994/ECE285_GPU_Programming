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

	// allocate for gaussiated_image
	this->gaussiated_image = (float*)malloc(sizeof(float) * (width - GAUSSIAN_KERNEL_SIZE + 1) * (height - GAUSSIAN_KERNEL_SIZE + 1));

	// allocate for sobel filters
	this->sobel_filter_x = (float*)malloc(sizeof(float) * SOBEL_FILTER_SIZE * SOBEL_FILTER_SIZE);
	assert(this->sobel_filter_x != NULL);
	this->sobel_filter_y = (float*)malloc(sizeof(float) * SOBEL_FILTER_SIZE * SOBEL_FILTER_SIZE);
	assert(this->sobel_filter_y!= NULL);

	// initialize
	this->init_gaussian_kernel();
	this->init_sobel_filters();
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

/* **************************************************************************************************** */

/* 	print out the gaussian kernel
*/
void canny_edge_host::print_gaussian_kernel() {
	printf("host - print_gaussian_kernel:\n");
	for (int r = 0; r < (GAUSSIAN_KERNEL_SIZE); r++) {
		for (int c = 0; c < (GAUSSIAN_KERNEL_SIZE); c++) {
			printf("%.2f ", this->gaussian_kernel[r * GAUSSIAN_KERNEL_SIZE + c]);
		}
		printf("\n");
	}
}

/* **************************************************************************************************** */

/* 	print out the sobel filters
*/
void canny_edge_host::print_sobel_filters() {
	printf("host - print_sobel_filter X(horizontal):\n");
	for (int r = 0; r < (SOBEL_FILTER_SIZE); r++) {
		for (int c = 0; c < (SOBEL_FILTER_SIZE); c++) {
			printf("%.2f ", this->sobel_filter_x[r * SOBEL_FILTER_SIZE + c]);
		}
		printf("\n");
	}
	printf("host - print_sobel_filter Y(vertical):\n");
	for (int r = 0; r < (SOBEL_FILTER_SIZE); r++) {
		for (int c = 0; c < (SOBEL_FILTER_SIZE); c++) {
			printf("%.2f ", this->sobel_filter_y[r * SOBEL_FILTER_SIZE + c]);
		}
		printf("\n");
	}
}

/* **************************************************************************************************** */

/* 	perform convolution of a given image and kernel and store it in result
	assumes square kernel
*/
void canny_edge_host::do_convolution(float *image, int image_width, int image_height, float *kernel, int kernel_size, float *result) {
	for (int iy = 0; iy <= (image_height - kernel_size + 1); iy++) {
		for (int ix = 0; ix <= (image_width - kernel_size + 1); ix++) {
			float total = 0.0f;

			// now we selected a tile
			for (int tile_y = 0; tile_y < kernel_size; tile_y++) {
				for (int tile_x = 0; tile_x < kernel_size; tile_x++) {
					int image_index = (iy + tile_y) * image_width + (ix + tile_x);
					int kernel_index = tile_y * kernel_size + tile_x;

					total += (image[image_index] * kernel[kernel_index]);
				}
			}
			int result_index = iy * (image_width - kernel_size + 1) + ix;
			result[result_index] = total;
		}
	}
}