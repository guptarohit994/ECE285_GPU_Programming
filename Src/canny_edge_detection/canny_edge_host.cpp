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

	// allocate for sobeled images (same size as image)
	this->sobeled_grad_x_image = (float*)malloc(sizeof(float) * width * height);
	assert(this->sobeled_grad_x_image != NULL);
	this->sobeled_grad_y_image = (float*)malloc(sizeof(float) * width * height);
	assert(this->sobeled_grad_y_image != NULL);
	this->sobeled_mag_image = (float*)malloc(sizeof(float) * width * height);
	assert(this->sobeled_mag_image != NULL);
	this->sobeled_dir_image = (float*)malloc(sizeof(float) * width * height);
	assert(this->sobeled_dir_image != NULL);

	// allocate for image after non-maximal suppression (same size as image)
	this->non_max_suppressed_image = (float*)malloc(sizeof(float) * width * height);
	assert(this->non_max_suppressed_image != NULL);

	// allocate for image after double thresholds applied (same size as image)
	this->double_thresholded_image = (float*)malloc(sizeof(float) * width * height);
	assert(this->double_thresholded_image != NULL);

	// allocate for image after edge tracking has been applied (same size as image)
	this->edge_tracked_image = (float*)malloc(sizeof(float) * width * height);
	assert(this->edge_tracked_image != NULL);

	// initialize
	this->init_gaussian_kernel();
	this->init_sobel_filters();
}

/* **************************************************************************************************** */

/* 	destructor
*/
canny_edge_host::~canny_edge_host() {
	if (this->image != NULL) 					free(this->image);
	
	if (this->gaussian_kernel != NULL) 			free(this->gaussian_kernel);
	if (this->gaussiated_image != NULL)			free(this->gaussiated_image);
	
	if (this->sobel_filter_x != NULL)			free(this->sobel_filter_x);
	if (this->sobel_filter_y != NULL)			free(this->sobel_filter_y);
	if (this->sobeled_grad_x_image != NULL)		free(this->sobeled_grad_x_image);
	if (this->sobeled_grad_y_image != NULL)		free(this->sobeled_grad_y_image);
	if (this->sobeled_mag_image != NULL)		free(this->sobeled_mag_image);
	if (this->sobeled_dir_image != NULL)		free(this->sobeled_dir_image);

	if (this->non_max_suppressed_image != NULL)	free(this->non_max_suppressed_image);

	if (this->double_thresholded_image != NULL)	free(this->double_thresholded_image);

	if (this->edge_tracked_image != NULL)		free(this->edge_tracked_image);
}

/* **************************************************************************************************** */

/*	getters for private vars
*/
int canny_edge_host::get_width() {
	return this->width;
}

int canny_edge_host::get_height() {
	return this->height;
}

float canny_edge_host::get_total_time_taken() {
	return this->total_time_taken;
}

float* canny_edge_host::get_gaussian_kernel() {
	return this->gaussian_kernel;
}

float* canny_edge_host::get_sobel_filter_x() {
	return this->sobel_filter_x;
}

float* canny_edge_host::get_sobel_filter_y() {
	return this->sobel_filter_y;
}

float* canny_edge_host::get_gaussiated_image() {
	return this->gaussiated_image;
}

float* canny_edge_host::get_sobeled_grad_x_image() {
	return this->sobeled_grad_x_image;
}

float* canny_edge_host::get_sobeled_grad_y_image() {
	return this->sobeled_grad_y_image;
}

float* canny_edge_host::get_sobeled_mag_image() {
	return this->sobeled_mag_image;
}

float* canny_edge_host::get_sobeled_dir_image() {
	return this->sobeled_dir_image;
}

float* canny_edge_host::get_non_max_suppressed_image() {
	return this->non_max_suppressed_image;
}

float* canny_edge_host::get_double_thresholded_image() {
	return this->double_thresholded_image;
}

float* canny_edge_host::get_edge_tracked_image() {
	return this->edge_tracked_image;
}

/* **************************************************************************************************** */

/* 	initialize gaussian kernel
*/
void canny_edge_host::init_gaussian_kernel() {
	float stddev = 1.0f;
	float denominator = 2 * (float) pow(stddev, 2);
	float sum = 0.0f;

	for (int i = 0; i < (GAUSSIAN_KERNEL_SIZE * GAUSSIAN_KERNEL_SIZE); i++){
		int ix = i % GAUSSIAN_KERNEL_SIZE;
		int iy = i / GAUSSIAN_KERNEL_SIZE;
		
		float numerator = (float) pow(ix - (GAUSSIAN_KERNEL_SIZE/2), 2);
		numerator += (float) pow(iy - (GAUSSIAN_KERNEL_SIZE/2), 2);

		this->gaussian_kernel[i] = (float) (exp( (-1 * numerator)/ denominator) / (M_PI * denominator));
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

		float denominator = (float) (pow(ix - weight, 2) + pow(iy - weight, 2));

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
	this->do_convolution(this->image, this->width, this->height, this->gaussian_kernel, GAUSSIAN_KERNEL_SIZE, this->gaussiated_image);

	printf("canny_edge_host::apply_gaussian_kernel - done!\n");
}

/* **************************************************************************************************** */

/*	apply the sobel_filter_x to the image
*/
void canny_edge_host::apply_sobel_filter_x() {
	// convolution of image with sobel filter in horizontal direction
	this->do_convolution(this->gaussiated_image, this->width, this->height, this->sobel_filter_x, SOBEL_FILTER_SIZE, this->sobeled_grad_x_image);

	printf("canny_edge_host::apply_sobel_filter_x - done\n");
}
/* **************************************************************************************************** */

/*	apply the sobel_filter_y to the image
*/
void canny_edge_host::apply_sobel_filter_y() {
	// convolution of image with sobel filter in vertical direction
	this->do_convolution(this->gaussiated_image, this->width, this->height, this->sobel_filter_y, SOBEL_FILTER_SIZE, this->sobeled_grad_y_image);
	printf("canny_edge_host::apply_sobel_filter_y - done\n");
}

/* **************************************************************************************************** */

/*	calculate gradient magnitude after applying sobel filters to the image
*/
void canny_edge_host::calculate_sobel_magnitude() {
	
	for (int i = 0; i < (this->width * this->height); i++) {
		this->sobeled_mag_image[i] = (float) sqrt(pow(this->sobeled_grad_x_image[i], 2) + pow(this->sobeled_grad_y_image[i], 2));
	}

	printf("canny_edge_host::calculate_sobel_magnitude - done\n");
}

/* **************************************************************************************************** */

/*	calculate gradient direction after applying sobel filters to the image
	values in radians (normalized, 0 to 1)
*/
void canny_edge_host::calculate_sobel_direction() {
	
	for (int i = 0; i < (this->width * this->height); i++) {
		//printf("i:%d, y:%.2f, x:%.2f, atan():%.2f\n", i, this->sobeled_grad_y_image[i], this->sobeled_grad_x_image[i], (atan(this->sobeled_grad_y_image[i] / this->sobeled_grad_x_image[i])));
		if (this->sobeled_grad_x_image[i] == 0 && this->sobeled_grad_y_image[i] < 0)
            this->sobeled_dir_image[i] = (float) (((-1 * M_PI/2) + (M_PI/2)) / M_PI);
        else if (this->sobeled_grad_x_image[i] == 0)
            this->sobeled_dir_image[i] = (float) (((M_PI/2) + (M_PI/2)) / M_PI);
        else
            this->sobeled_dir_image[i] = (float) ((atan(this->sobeled_grad_y_image[i] / this->sobeled_grad_x_image[i]) + (M_PI/2)) / M_PI);	}

	printf("canny_edge_host::calculate_sobel_direction - done\n");
}

/* **************************************************************************************************** */

/*	performs non-maximal suppression on the input image and writes into result
	image should contain direction (sobeled_dir_image)
	Skips border values
*/
void canny_edge_host::apply_non_max_suppression() {
	
	float *mag_image = this->sobeled_mag_image;
	float *dir_image = this->sobeled_dir_image;
	int image_width = this->width;
	int image_height = this->height;
	float *result = this->non_max_suppressed_image;

	// TODO check if two for-loops are faster than single loop but more if-conditions
	// initialize so that unreachable points in next for loop are also 0
	for (int i = 0; i < (image_width * image_height); i++)
		result[i] = 0.0f;

	int window_size = 3; //square window
    for (int iy = 1; iy <= (image_height - window_size + 1); iy++) {
        for (int ix = 1; ix <= (image_width - window_size + 1); ix++) {

            // now we selected a tile
            int tile_center_index = iy * image_width + ix;

            float right_value;
            float left_value;

            // angle 0
            if ((dir_image[tile_center_index] < (float)(1/8)) || (dir_image[tile_center_index] >= (float)(7/8))) {
            	right_value = mag_image[tile_center_index + 1];
            	left_value = mag_image[tile_center_index - 1];
            }
            // angle 45
            else if ((dir_image[tile_center_index] >= (float)(1/8)) && (dir_image[tile_center_index] < (float)(3/8))) {
            	right_value = mag_image[tile_center_index - (image_width - 1)];
            	left_value = mag_image[tile_center_index + (image_width - 1)];
            }
            // angle 90
            else if ((dir_image[tile_center_index] >= (float)(3/8)) && (dir_image[tile_center_index] < (float)(5/8))) {
            	right_value = mag_image[tile_center_index - image_width];
            	left_value = mag_image[tile_center_index + image_width];
            }
            // angle 135
            else if ((dir_image[tile_center_index] >= (float)(5/8)) && (dir_image[tile_center_index] < (float)(7/8))) {
            	right_value = mag_image[tile_center_index - (image_width + 1)];
            	left_value = mag_image[tile_center_index + (image_width + 1)];
            }
            else{
            	// assert should not be reached
            	assert(0 > 1);
            }

            // suppress anything if not the maximum value
            if ((mag_image[tile_center_index] >= right_value) && (mag_image[tile_center_index] >= left_value))
            	result[tile_center_index] = mag_image[tile_center_index];
            else
            	result[tile_center_index] = 0.0f;

        }
    }

    printf("canny_edge_host::apply_non_max_suppression - done\n");
}

/* **************************************************************************************************** */

/*	applies the double thresholds to the provided image
*/
void canny_edge_host::apply_double_thresholds() {
	
	float *image = this->non_max_suppressed_image;
	int image_width = this->width;
	int image_height = this->height;
	float *result = this->double_thresholded_image;

	// compute the maximum
	float max_pixel_val = 0.0f;

	// figure out the max pixel value
	for (int i = 0; i < (image_width * image_height); i++) {
		if (image[i] > max_pixel_val)
			max_pixel_val = image[i]; 
	}

	float strong_pixel_val = max_pixel_val * STRONG_PIXEL_THRESHOLD;
	float weak_pixel_val = max_pixel_val * WEAK_PIXEL_THRESHOLD;

	// now, classify each pixel as strong, weak, zero
	for (int i = 0; i < (image_width * image_height); i++) {
		if (image[i] >= strong_pixel_val)
			result[i] = STRONG_PIXEL_VALUE;
		else if (image[i] >= weak_pixel_val)
			result[i] = WEAK_PIXEL_VALUE;
		else
			result[i] = 0.0f; 
	}

	printf("canny_edge_host::apply_double_thresholds - done\n");
}

/* **************************************************************************************************** */

/*	applies edge tracking by hysteresis to the provided image
	ignores the boundary pixels
*/
void canny_edge_host::apply_hysteresis_edge_tracking() {

	float *image = this->double_thresholded_image;
	int image_width = this->width;
	int image_height = this->height;
	float *result = this->edge_tracked_image;

	int window_size = 3; //square window
	
    for (int iy = 1; iy <= (image_height - window_size + 1); iy++) {
        for (int ix = 1; ix <= (image_width - window_size + 1); ix++) {

        	// selected a tile now
        	// check if the center is weak
        	int tile_center_index = iy * image_width + ix;
        	if (image[tile_center_index] == WEAK_PIXEL_VALUE) {
        		// check if any strong pixels are there in the vicinity
        		// start from 0deg, 45, 90, 135
        		if ((image[tile_center_index + 1] == STRONG_PIXEL_VALUE) 					||
        			(image[tile_center_index - 1] == STRONG_PIXEL_VALUE)					||
        			(image[tile_center_index - (image_width - 1)] == STRONG_PIXEL_VALUE)	||
        			(image[tile_center_index + (image_width - 1)] ==  STRONG_PIXEL_VALUE)	||
        			(image[tile_center_index - image_width] == STRONG_PIXEL_VALUE)			||
        			(image[tile_center_index + image_width] == STRONG_PIXEL_VALUE)			||
        			(image[tile_center_index - (image_width + 1)] == STRONG_PIXEL_VALUE)	||
        			(image[tile_center_index + (image_width + 1)] == STRONG_PIXEL_VALUE))
        			result[tile_center_index] = STRONG_PIXEL_VALUE;
        		else
        			result[tile_center_index] = 0.0f;
        	}
        }
    }

    printf("canny_edge_host::apply_hysteresis_edge_tracking - done\n");
}