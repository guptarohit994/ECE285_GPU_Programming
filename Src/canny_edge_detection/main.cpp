/*************************************************************************
/* ECE 285: GPU Programmming 2019 Fall quarter
/* Author: Group_G
/* Copyright 2019
/* University of California, San Diego
/*************************************************************************/

#include "utils.h"
#include "canny_edge_host.h"

#define INPUT_FILE_NAME "C:/Users/r4gupta/Downloads/final_project/ECE285_GPU_Programming/Input_images/tree.png"

int main(int argc, char **argv) {

	// read image from disk
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

	return 0;
}