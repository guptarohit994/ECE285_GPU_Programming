/*************************************************************************
/* ECE 285: GPU Programmming 2019 Fall quarter
/* Author: Group_G
/* Copyright 2019
/* University of California, San Diego
/*************************************************************************/

#include "utils.h"
#include "canny_edge_host.h"

#define INPUT_FILE_NAME "C:/Users/r4gupta/Downloads/final_exam/ECE285_GPU_Programming/Input_images/person_mountain_cliff.png"

int main(int argc, char **argv) {

	// read image from disk
	// SILENTLY FAILS WITHOUT ERROR!
	CByteImage cbimage;
	ReadImage(cbimage, INPUT_FILE_NAME);

	// convert to grayscale
	CByteImage cbimage_gray = ConvertToGray(cbimage);
	CShape cbimage_gray_shape = cbimage_gray.Shape();

	// convert the image to a regular array
	gray_image gimage = gray_image(convert_CByteImage_to_array(cbimage_gray), \
								   cbimage_gray_shape.width, 				  \
								   cbimage_gray_shape.height);

	printf("Successfully loaded %s of height:%d, width:%d\n", INPUT_FILE_NAME, gimage.get_height(), gimage.get_width());

	canny_edge_host from_host = canny_edge_host(gimage.get_host_gimage(), gimage.get_width(), gimage.get_height());

	from_host.print_gaussian_kernel();
	from_host.print_sobel_filters();

	float *img = (float*)malloc(25 * sizeof(float));
	float *kernel = (float*)malloc(9 * sizeof(float));
	float *result = (float*)malloc(9 * sizeof(float));

	for (int i = 0; i < 25; i++) {
		img[i] = 1.0f;
	}
	for (int i = 0; i < 9; i++) {
		kernel[i] = 1.0f;
	}
	from_host.do_convolution(img, 5, 5, kernel, 3, result);
	
	printf("result:\n");
	for (int r = 0; r < (3); r++) {
		for (int c = 0; c < (3); c++) {
			printf("%.2f ", result[r * 3 + c]);
		}
		printf("\n");
	}
}