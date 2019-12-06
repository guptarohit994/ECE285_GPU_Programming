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

	// allocate for gaussiated_image (same size as image)
	this->gaussiated_image = (float*)malloc(sizeof(float) * width * height);

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
	print_matrix(this->gaussian_kernel, GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE);
}

/* **************************************************************************************************** */

/* 	print out the sobel filters
*/
void canny_edge_host::print_sobel_filters() {
	printf("host - print_sobel_filter X(horizontal):\n");
	print_matrix(this->sobel_filter_x, SOBEL_FILTER_SIZE, SOBEL_FILTER_SIZE);

	printf("host - print_sobel_filter Y(vertical):\n");
	print_matrix(this->sobel_filter_y, SOBEL_FILTER_SIZE, SOBEL_FILTER_SIZE);
}

/* **************************************************************************************************** */

/* 	print out the image after gaussian kernel has been applied
*/
void canny_edge_host::print_gaussiated_image() {
	printf("host - print_gaussiated_image:\n");
	print_matrix(this->gaussiated_image, this->width, this->height);
}

/* **************************************************************************************************** */

/* 	perform convolution (2D) of a given image (as if padded) and kernel and store it in result
	assumes square kernel
	output image will be of same size as input
*/
void canny_edge_host::do_convolution(float *image, int image_width, int image_height, float *kernel, int kernel_size, float *result) {
	for (int iy = 0; iy < image_height; iy++) {
		for (int ix = 0; ix < image_width; ix++) {
			float total = 0.0f;

			for (int m = 0; m < kernel_size; m++) {
				// row index of flipped kernel
				int flipped_ky = kernel_size - 1 - m;

				for (int n = 0; n < kernel_size; n++) {
					// column index of flipped kernel
					int flipped_kx = kernel_size - 1 - n;

					// index of input image, used for checking boundary
					int image_y = iy + ((kernel_size / 2) - flipped_ky);
					int image_x = ix + ((kernel_size / 2) - flipped_kx);

					// ignore input image pixels which are out of bound
					if ((image_x >= 0) && (image_x < image_width) && \
						(image_y >= 0) && (image_y < image_height)) {
						total += (image[image_y * image_width + image_x] * \
								  kernel[flipped_ky * kernel_size + flipped_kx]);
					}
				}
			}
			// keep total within bounds
			result[iy * image_width + ix] = fminf(fmaxf(total, 0.0f), 1.0f);
		}
	}
}

/* **************************************************************************************************** */

/*	apply the gaussian kernel to the image
*/
void canny_edge_host::apply_gaussian_kernel() {
	// convolution of image with gaussian kernel
	do_convolution(this->image, this->width, this->height, this->gaussian_kernel, GAUSSIAN_KERNEL_SIZE, this->gaussiated_image);
}