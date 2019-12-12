/*************************************************************************
/* ECE 285: GPU Programmming 2019 Fall quarter
/* Author: Group_G
/* Copyright 2019
/* University of California, San Diego
/*************************************************************************/

/*	implementation of class canny_edge_device
*/
#include "canny_edge_device.h"

/* 	constructor
*/
canny_edge_device::canny_edge_device(float *image, int width, int height) {
	assert(width > 0);
	assert(height > 0);

	CHECK(cudaMalloc((void **)&this->image, sizeof(float) * width * height));
	assert(this->image != NULL);
	CHECK(cudaMemcpy(this->image, image, sizeof(float) * width * height, cudaMemcpyHostToDevice));

	this->width = width;
	this->height = height;
	this->total_time_taken = 0.0f;
	this->strong_pixel_threshold = 0.0f;
	this->weak_pixel_threshold = 0.0f;
	
	// allocate for gaussian_kernel
	CHECK(cudaMalloc((void **)&this->gaussian_kernel, sizeof(float) * GAUSSIAN_KERNEL_SIZE * GAUSSIAN_KERNEL_SIZE));
	assert(this->gaussian_kernel != NULL);

	// allocate for gaussiated_image (same size as image)
	CHECK(cudaMalloc((void **)&this->gaussiated_image, sizeof(float) * width * height));

	// allocate for sobel filters
	CHECK(cudaMalloc((void **)&this->sobel_filter_x, sizeof(float) * SOBEL_FILTER_SIZE * SOBEL_FILTER_SIZE));
	assert(this->sobel_filter_x != NULL);
	CHECK(cudaMalloc((void **)&this->sobel_filter_y, sizeof(float) * SOBEL_FILTER_SIZE * SOBEL_FILTER_SIZE));
	assert(this->sobel_filter_y!= NULL);

	// allocate for sobeled images (same size as image)
	CHECK(cudaMalloc((void **)&this->sobeled_grad_x_image, sizeof(float) * width * height));
	assert(this->sobeled_grad_x_image != NULL);
	CHECK(cudaMalloc((void **)&this->sobeled_grad_y_image, sizeof(float) * width * height));
	assert(this->sobeled_grad_y_image != NULL);
	
	CHECK(cudaMalloc((void **)&this->sobeled_mag_image, sizeof(float) * width * height));
	assert(this->sobeled_mag_image != NULL);
	
	CHECK(cudaMalloc((void **)&this->sobeled_dir_image, sizeof(float) * width * height));
	assert(this->sobeled_dir_image != NULL);

	// allocate for image after non-maximal suppression (same size as image)
	CHECK(cudaMalloc((void **)&this->non_max_suppressed_image, sizeof(float) * width * height));
	assert(this->non_max_suppressed_image != NULL);

	// allocate for image after double thresholds applied (same size as image)
	CHECK(cudaMalloc((void **)&this->double_thresholded_image, sizeof(float) * width * height));
	assert(this->double_thresholded_image != NULL);

	// allocate for image after edge tracking has been applied (same size as image)
	CHECK(cudaMalloc((void **)&this->edge_tracked_image, sizeof(float) * width * height));
	assert(this->edge_tracked_image != NULL);

	// initialize
	this->init_gaussian_kernel();
	this->init_sobel_filters();
}

/* **************************************************************************************************** */

/* 	destructor
*/
canny_edge_device::~canny_edge_device() {
	if (this->image != NULL) 					CHECK(cudaFree(this->image));
	
	if (this->gaussian_kernel != NULL) 			CHECK(cudaFree(this->gaussian_kernel));
	if (this->gaussiated_image != NULL)			CHECK(cudaFree(this->gaussiated_image));
	
	if (this->sobel_filter_x != NULL)			CHECK(cudaFree(this->sobel_filter_x));
	if (this->sobel_filter_y != NULL)			CHECK(cudaFree(this->sobel_filter_y));
	if (this->sobeled_grad_x_image != NULL)		CHECK(cudaFree(this->sobeled_grad_x_image));
	if (this->sobeled_grad_y_image != NULL)		CHECK(cudaFree(this->sobeled_grad_y_image));
	if (this->sobeled_mag_image != NULL)		CHECK(cudaFree(this->sobeled_mag_image));
	if (this->sobeled_dir_image != NULL)		CHECK(cudaFree(this->sobeled_dir_image))

	if (this->non_max_suppressed_image != NULL)	CHECK(cudaFree(this->non_max_suppressed_image));

	if (this->double_thresholded_image != NULL)	CHECK(cudaFree(this->double_thresholded_image));

	if (this->edge_tracked_image != NULL)		CHECK(cudaFree(this->edge_tracked_image));
}

/* **************************************************************************************************** */

/*	getters for private vars
*/
int canny_edge_device::get_width() {
	return this->width;
}

int canny_edge_device::get_height() {
	return this->height;
}

float canny_edge_device::get_total_time_taken() {
	return this->total_time_taken;
}

float* canny_edge_device::get_gaussian_kernel() {
	return this->gaussian_kernel;
}

float* canny_edge_device::get_sobel_filter_x() {
	return this->sobel_filter_x;
}

float* canny_edge_device::get_sobel_filter_y() {
	return this->sobel_filter_y;
}

float* canny_edge_device::get_gaussiated_image() {
	return this->gaussiated_image;
}

float* canny_edge_device::get_sobeled_grad_x_image() {
	return this->sobeled_grad_x_image;
}

float* canny_edge_device::get_sobeled_grad_y_image() {
	return this->sobeled_grad_y_image;
}

float* canny_edge_device::get_sobeled_mag_image() {
	return this->sobeled_mag_image;
}

float* canny_edge_device::get_sobeled_dir_image() {
	return this->sobeled_dir_image;
}

float* canny_edge_device::get_non_max_suppressed_image() {
	return this->non_max_suppressed_image;
}

float* canny_edge_device::get_double_thresholded_image() {
	return this->double_thresholded_image;
}

float* canny_edge_device::get_edge_tracked_image() {
	return this->edge_tracked_image;
}

/* **************************************************************************************************** */

/* 	CUDA kernel to initialize gaussian kernel
*/
__global__
void init_gaussian_kernel_cuda(float *gaussian_kernel) {
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int index = iy * GAUSSIAN_KERNEL_SIZE + ix;

	int weight = GAUSSIAN_KERNEL_SIZE / 2;

	float stddev = 1.0f;
	float denominator = 2.0f * (float)powf(stddev, 2);

	float numerator = (float)(powf(fabsf(ix - weight), 2.0f) + powf(fabsf(iy - weight), 2.0f));

	gaussian_kernel[index] = (float)(expf((-1 * numerator) / denominator) / (M_PI * denominator));
}

/* **************************************************************************************************** */

void canny_edge_device::init_gaussian_kernel() {
	dim3 block(GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE);
	init_gaussian_kernel_cuda << < 1, block >> >(this->gaussian_kernel);
}

/* **************************************************************************************************** */

/* 	CUDA kernel to initialize sobel filters
	https://stackoverflow.com/a/41065243/253056
*/
__global__
void init_sobel_filters_cuda(float *sobel_filter_x, float *sobel_filter_y) {
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int index = iy * SOBEL_FILTER_SIZE + ix;

	int weight = SOBEL_FILTER_SIZE / 2;

	float denominator = (float)(powf(fabsf(ix - weight), 2) + powf(fabsf(iy - weight), 2));

	if (denominator == 0.0f) {
		sobel_filter_x[index] = 0.0f;
		sobel_filter_y[index] = 0.0f;
	}
	else {
		sobel_filter_x[index] = ((ix - weight) * weight) / denominator;
		sobel_filter_y[index] = ((iy - weight) * weight) / denominator;
	}
}
/* **************************************************************************************************** */

/* 	initialize sobel filters
*/
void canny_edge_device::init_sobel_filters() {
	dim3 block(SOBEL_FILTER_SIZE, SOBEL_FILTER_SIZE);
	init_sobel_filters_cuda << < 1, block >> >(this->sobel_filter_x, this->sobel_filter_y);
}

/* **************************************************************************************************** */

/* 	perform convolution (2D) of a given image (as if padded) and kernel and store it in result
	assumes square kernel
	output image will be of same size as input
*/
__global__
void do_convolution(float *image, int image_width, int image_height, float *kernel, int kernel_size, float *result) {
	const int shared_mem_width = TILE_WIDTH + MAX(SOBEL_FILTER_SIZE, GAUSSIAN_KERNEL_SIZE) - 1;
	__shared__ float shared_mem[shared_mem_width][shared_mem_width];

	int weight = (kernel_size / 2);
	int dest = threadIdx.y * TILE_WIDTH + threadIdx.x;
	int destY = dest / shared_mem_width;
	int destX = dest % shared_mem_width;
	int srcY = blockIdx.y * TILE_WIDTH + destY - weight;
	int srcX = blockIdx.x * TILE_WIDTH + destX - weight;
	int src = (srcY * image_width + srcX);
	if (srcY >= 0 && srcY < image_height && srcX >= 0 && srcX < image_width)
		shared_mem[destY][destX] = image[src];
	else
		shared_mem[destY][destX] = 0;

	dest = threadIdx.y * TILE_WIDTH + threadIdx.x + TILE_WIDTH * TILE_WIDTH;
	destY = dest / shared_mem_width;
	destX = dest % shared_mem_width;
	srcY = blockIdx.y * TILE_WIDTH + destY - weight;
	srcX = blockIdx.x * TILE_WIDTH + destX - weight;
	src = (srcY * image_width + srcX);
	if (destY < shared_mem_width) {
		if (srcY >= 0 && srcY < image_height && srcX >= 0 && srcX < image_width)
			shared_mem[destY][destX] = image[src];
		else
			shared_mem[destY][destX] = 0;
	}
	__syncthreads();

	float accum = 0;
	for (int j = 0; j < kernel_size; j++)
		for (int i = 0; i < kernel_size; i++)
			accum += shared_mem[threadIdx.y + j][threadIdx.x + i] * kernel[j * kernel_size + i];
	int x = blockIdx.x * TILE_WIDTH + threadIdx.x;
	int y = blockIdx.y * TILE_WIDTH + threadIdx.y;
	if (x < image_width && y < image_height)
		result[y * image_width + x] = (fminf(fmaxf((accum), 0.0), 1.0));
	__syncthreads();
}

/* **************************************************************************************************** */

/*	apply the gaussian kernel to the image
*/
void canny_edge_device::apply_gaussian_kernel() {
	cudaEvent_t start, stop;
	CLOCK_CUDA_INIT(start, stop);

	TIC_CUDA(start);

	dim3 grid(((this->width  + TILE_WIDTH - 1) / TILE_WIDTH), ((this->height  + TILE_WIDTH - 1) / TILE_WIDTH));
	dim3 block(TILE_WIDTH, TILE_WIDTH);

	// convolution of image with gaussian kernel
	do_convolution <<< grid, block >>> (this->image, this->width, this->height, this->gaussian_kernel, GAUSSIAN_KERNEL_SIZE, this->gaussiated_image);

	TOC_CUDA(stop);
	
	float miliseconds = 0;
	TIME_DURATION_CUDA(miliseconds, start, stop);
	this->total_time_taken += miliseconds;

	printf("canny_edge_device::apply_gaussian_kernel - done in %.5f ms\n", miliseconds);
}

/* **************************************************************************************************** */

/*	calculate the thresholds using the gaussiated image.
	Constants are tuned
	TODO would it benefit from a kernel
*/

void canny_edge_device::compute_pixel_thresholds() {

	int image_width = this->width;
	int image_height = this->height;
	float *image = (float*)malloc(sizeof(float) * image_width * image_height);
	CHECK(cudaMemcpy(image, this->gaussiated_image, sizeof(float) * image_width * image_height, cudaMemcpyDeviceToHost));

	TIC;

	// compute the sum of all pixel values
	float sum_pixel_val = 0.0f;

	// figure out the max pixel value and also compute thresholds
	for (int i = 0; i < (image_width * image_height); i++) {
		sum_pixel_val += image[i];
	}

	this->strong_pixel_threshold = (0.66f * sum_pixel_val) / (image_width * image_height);
	this->weak_pixel_threshold = (0.33f * sum_pixel_val) / (image_width * image_height);

	TOC;

	TIME_DURATION;
	this->total_time_taken += time_taken.count() * 1000; // convert to ms
	printf("canny_edge_device::compute_pixel_thresholds - weak_pixel_threshold:%.2f, strong_pixel_threshold:%.2f\n", this->weak_pixel_threshold, this->strong_pixel_threshold);
	free(image);
}

/* **************************************************************************************************** */

/*	apply the sobel_filter_x to the image
*/
void canny_edge_device::apply_sobel_filter_x() {
	cudaEvent_t start, stop;
	CLOCK_CUDA_INIT(start, stop);

	TIC_CUDA(start);

	dim3 grid(((this->width  + TILE_WIDTH - 1) / TILE_WIDTH), ((this->height + TILE_WIDTH - 1) / TILE_WIDTH));
	dim3 block(TILE_WIDTH, TILE_WIDTH);

	// convolution of image with sobel filter in horizontal direction
	do_convolution <<< grid, block >>> (this->gaussiated_image, this->width, this->height, this->sobel_filter_x, SOBEL_FILTER_SIZE, this->sobeled_grad_x_image);

	TOC_CUDA(stop);
	
	float miliseconds = 0;
	TIME_DURATION_CUDA(miliseconds, start, stop);
	this->total_time_taken += miliseconds;
	printf("canny_edge_device::apply_sobel_filter_x - done in %.5f ms\n", miliseconds);
}

/* **************************************************************************************************** */

/*	apply the sobel_filter_y to the image
*/
void canny_edge_device::apply_sobel_filter_y() {
	cudaEvent_t start, stop;
	CLOCK_CUDA_INIT(start, stop);

	TIC_CUDA(start);

	dim3 grid(((this->width  + TILE_WIDTH - 1) / TILE_WIDTH), ((this->width  + TILE_WIDTH - 1) / TILE_WIDTH));
	dim3 block(TILE_WIDTH, TILE_WIDTH);

	// convolution of image with sobel filter in vertical direction
	do_convolution <<< grid, block >>> (this->gaussiated_image, this->width, this->height, this->sobel_filter_y, SOBEL_FILTER_SIZE, this->sobeled_grad_y_image);

	TOC_CUDA(stop);
	
	float miliseconds = 0;
	TIME_DURATION_CUDA(miliseconds, start, stop);
	this->total_time_taken += miliseconds;
	printf("canny_edge_device::apply_sobel_filter_y - done in %.5f ms\n", miliseconds);
}

/* **************************************************************************************************** */

/*	CUDA helper kernel to calculate sobel magnitude
*/
__global__
void calculate_sobel_magnitude_cuda(float *sobeled_grad_x_image, float *sobeled_grad_y_image, float *sobeled_mag_image, int image_width, int image_height) {
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	int index = iy * image_width + ix;
	if (index < (image_width * image_height))
		sobeled_mag_image[index] = (float) sqrtf(powf(sobeled_grad_x_image[index], 2) + powf(sobeled_grad_y_image[index], 2));
}

/* **************************************************************************************************** */

/*	calculate gradient magnitude after applying sobel filters to the image
*/
void canny_edge_device::calculate_sobel_magnitude() {
	cudaEvent_t start, stop;
	CLOCK_CUDA_INIT(start, stop);

	TIC_CUDA(start);

	int total_pixels = (this->width * this->height);
	dim3 block(MIN(256, total_pixels));
	dim3 grid((total_pixels + block.x - 1) / block.x);
	printf("grid.x:%d, grid.y:%d, block.x:%d, block.y:%d\n", grid.x, grid.y, block.x, block.y);
	calculate_sobel_magnitude_cuda <<< grid, block >>> (this->sobeled_grad_x_image, this->sobeled_grad_y_image, this->sobeled_mag_image, this->width, this->height);

	TOC_CUDA(stop);
	
	float miliseconds = 0;
	TIME_DURATION_CUDA(miliseconds, start, stop);
	this->total_time_taken += miliseconds;
	printf("canny_edge_device::calculate_sobel_magnitude - done in %.5f ms\n", miliseconds);
}

/* **************************************************************************************************** */

/*	CUDA helper kernel to compute sobel direction image
*/
__global__
void calculate_sobel_direction_cuda(float *sobeled_grad_x_image, float *sobeled_grad_y_image, float *sobeled_dir_image, int image_width, int image_height) {
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	int index = iy * image_width + ix;
	
	if (index < (image_width * image_height)) {
		float pix_x = sobeled_grad_x_image[index];
		float pix_y = sobeled_grad_y_image[index];

		if ((pix_x * pix_y) < 0)
			sobeled_dir_image[index] = (float)((atanf((float)pix_y / pix_x) + M_PI) / M_PI);
		else {
			// need to handle this case specifically. atanf gives nan
			if (pix_x == 0)
				sobeled_dir_image[index] = 0.0f;
			else
				sobeled_dir_image[index] = (float)(atanf((float)pix_y / pix_x) / M_PI);
		}
	}
}

/* **************************************************************************************************** */

/*	calculate gradient direction after applying sobel filters to the image
	values in radians (normalized, 0 to 1)
*/
void canny_edge_device::calculate_sobel_direction() {
	cudaEvent_t start, stop;
	CLOCK_CUDA_INIT(start, stop);

	TIC_CUDA(start);

	int total_pixels = (this->width * this->height);
	dim3 block(MIN(256, total_pixels));
	dim3 grid((total_pixels + block.x - 1) / block.x);

	calculate_sobel_direction_cuda <<< grid, block >>> (this->sobeled_grad_x_image, this->sobeled_grad_y_image, this->sobeled_dir_image, this->width, this->height);

	TOC_CUDA(stop);
	
	float miliseconds = 0;
	TIME_DURATION_CUDA(miliseconds, start, stop);
	this->total_time_taken += miliseconds;
	printf("canny_edge_device::calculate_sobel_direction - done in %.5f ms\n", miliseconds);
}

/* **************************************************************************************************** */
__global__
void apply_non_max_suppression_cuda(float *dir_image, float *mag_image, int image_width, int image_height, float *result) {
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	int window_size = 3;
	int total_windows_x = (image_width - window_size + 1);
	int total_windows_y = (image_height - window_size + 1);

	int tile_top_left_index = (int)(ix / total_windows_x) * image_width + (ix % total_windows_x);
	int tile_center_index = tile_top_left_index + (image_width + 1);

	if (ix < (total_windows_x * total_windows_y)) {
		//printf("ix:%d, iy:%d, tile_top_left_index:%d, tile_center_index:%d\n", ix, iy, tile_top_left_index, tile_center_index);

		float right_value;
		float left_value;

		// angle 0
		if ((dir_image[tile_center_index] < (float)(1 / 8)) || (dir_image[tile_center_index] >= (float)(7 / 8))) {
			right_value = mag_image[tile_center_index + 1];
			left_value = mag_image[tile_center_index - 1];
		}
		// angle 45
		else if ((dir_image[tile_center_index] >= (float)(1 / 8)) && (dir_image[tile_center_index] < (float)(3 / 8))) {
			right_value = mag_image[tile_center_index - (image_width - 1)];
			left_value = mag_image[tile_center_index + (image_width - 1)];
		}
		// angle 90
		else if ((dir_image[tile_center_index] >= (float)(3 / 8)) && (dir_image[tile_center_index] < (float)(5 / 8))) {
			right_value = mag_image[tile_center_index - image_width];
			left_value = mag_image[tile_center_index + image_width];
		}
		// angle 135
		else if ((dir_image[tile_center_index] >= (float)(5 / 8)) && (dir_image[tile_center_index] < (float)(7 / 8))) {
			right_value = mag_image[tile_center_index - (image_width + 1)];
			left_value = mag_image[tile_center_index + (image_width + 1)];
		}
		else {
			// assert should not be reached
			assert(0 > 1);
		}

		// suppress anything if not the maximum value
		if ((mag_image[tile_center_index] >= right_value) && (mag_image[tile_center_index] >= left_value)) {
			result[tile_center_index] = mag_image[tile_center_index];
			//printf("ix:%d, iy:%d, tile_top_left_index:%d, tile_center_index:%d, value:%.2f\n", ix, iy, tile_top_left_index, tile_center_index, mag_image[tile_center_index]);
		}
		else
			result[tile_center_index] = 0.0f;
	}
}

/* **************************************************************************************************** */

/*	performs non-maximal suppression on the input image and writes into result
	image should contain direction (sobeled_dir_image)
	Skips border values
*/
void canny_edge_device::apply_non_max_suppression() {
	cudaEvent_t start, stop;
	CLOCK_CUDA_INIT(start, stop);

	TIC_CUDA(start);

	float *mag_image = this->sobeled_mag_image;
	float *dir_image = this->sobeled_dir_image;
	int image_width = this->width;
	int image_height = this->height;
	float *result = this->non_max_suppressed_image;

	int window_size = 3; //square window
	int total_windows_x = (image_width - window_size + 1);
	int total_windows_y = (image_height - window_size + 1);
	int total_windows = total_windows_x * total_windows_y;


	dim3 block(MIN(256, total_windows));
	dim3 grid((total_windows + block.x - 1) / block.x);
    apply_non_max_suppression_cuda <<< grid, block >>> (dir_image, mag_image, image_width, image_height, result);

	TOC_CUDA(stop);
	
	float miliseconds = 0;
	TIME_DURATION_CUDA(miliseconds, start, stop);
	this->total_time_taken += miliseconds;
    printf("canny_edge_device::apply_non_max_suppression - done in %.5f ms\n", miliseconds);
}

/* **************************************************************************************************** */

__global__
void apply_double_thresholds_cuda(float *image, int image_width, int image_height, float strong_pixel_threshold, float weak_pixel_threshold, float *result) {
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	int index = iy * image_width + ix;

	if (index < (image_width * image_height)) {
		if (image[index] >= strong_pixel_threshold)
			result[index] = STRONG_PIXEL_VALUE;
		else if (image[index] >= weak_pixel_threshold)
			result[index] = WEAK_PIXEL_VALUE;
		else
			result[index] = 0.0f;
	}
}

/* **************************************************************************************************** */

/*	applies the double thresholds to the provided image
*/
void canny_edge_device::apply_double_thresholds() {
	cudaEvent_t start, stop;
	CLOCK_CUDA_INIT(start, stop);

	TIC_CUDA(start);

	int total_pixels = (this->width * this->height);
	dim3 block(MIN(256, total_pixels));
	dim3 grid((total_pixels + block.x - 1) / block.x);

	apply_double_thresholds_cuda << < grid, block >> > (this->non_max_suppressed_image, this->width, this->height, this->strong_pixel_threshold, this->weak_pixel_threshold, this->double_thresholded_image);

	TOC_CUDA(stop);

	float miliseconds = 0;
	TIME_DURATION_CUDA(miliseconds, start, stop);
	this->total_time_taken += miliseconds;
	printf("canny_edge_device::apply_double_thresholds - done in %.5f ms\n", miliseconds);
}

/* **************************************************************************************************** */

__global__
void apply_hysteresis_edge_tracking_cuda(float *image, int image_width, int image_height, float *result) {
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	int window_size = 3;
	int total_windows_x = (image_width - window_size + 1);
	int total_windows_y = (image_height - window_size + 1);

	int tile_top_left_index = (int)(ix / total_windows_x) * image_width + (ix % total_windows_x);
	int tile_center_index = tile_top_left_index + (image_width + 1);

	if (ix < (total_windows_x * total_windows_y)) {
		if (image[tile_center_index] == WEAK_PIXEL_VALUE) {
			// check if any strong pixels are there in the vicinity
			// start from 0deg, 45, 90, 135
			if ((image[tile_center_index + 1] == STRONG_PIXEL_VALUE) ||
				(image[tile_center_index - 1] == STRONG_PIXEL_VALUE) ||
				(image[tile_center_index - (image_width - 1)] == STRONG_PIXEL_VALUE) ||
				(image[tile_center_index + (image_width - 1)] == STRONG_PIXEL_VALUE) ||
				(image[tile_center_index - image_width] == STRONG_PIXEL_VALUE) ||
				(image[tile_center_index + image_width] == STRONG_PIXEL_VALUE) ||
				(image[tile_center_index - (image_width + 1)] == STRONG_PIXEL_VALUE) ||
				(image[tile_center_index + (image_width + 1)] == STRONG_PIXEL_VALUE)) {
				result[tile_center_index] = STRONG_PIXEL_VALUE;
			}
			else
				result[tile_center_index] = 0.0f;
		}
	}

}

/* **************************************************************************************************** */

/*	applies edge tracking by hysteresis to the provided image
ignores the boundary pixels
*/
void canny_edge_device::apply_hysteresis_edge_tracking() {
	cudaEvent_t start, stop;
	CLOCK_CUDA_INIT(start, stop);

	TIC_CUDA(start);

	float *image = this->double_thresholded_image;
	int image_width = this->width;
	int image_height = this->height;
	float *result = this->edge_tracked_image;

	// init as double thresholds
	CHECK(cudaMemcpy(this->edge_tracked_image, this->double_thresholded_image, sizeof(float) * this->width * this->height, cudaMemcpyDeviceToDevice));

	int window_size = 3; //square window
	int total_windows_x = (image_width - window_size + 1);
	int total_windows_y = (image_height - window_size + 1);
	int total_windows = total_windows_x * total_windows_y;


	dim3 block(MIN(256, total_windows));
	dim3 grid((total_windows + block.x - 1) / block.x);
	apply_hysteresis_edge_tracking_cuda << < grid, block >> > (this->double_thresholded_image, this->width, this->height, this->edge_tracked_image);
	
	TOC_CUDA(stop);

	float miliseconds = 0;
	TIME_DURATION_CUDA(miliseconds, start, stop);
	this->total_time_taken += miliseconds;
	printf("canny_edge_device::apply_hysteresis_edge_tracking - done in %.5f ms\n", miliseconds);
}