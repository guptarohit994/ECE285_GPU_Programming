/*************************************************************************
/* ECE 285: GPU Programmming 2019 Fall quarter
/* Author: Group_G
/* Copyright 2019
/* University of California, San Diego
/*************************************************************************/

#ifndef _CUSTOM_UTILS_H
#define _CUSTOM_UTILS_H

#include <cuda_fp16.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include "common.h"

#include "imageLib/Image.h"
#include "imageLib/ImageIO.h"
#include "imageLib/Convert.h"

#include "gray_image.h"

/* convert CByteImage to a linearly stored image
*/
float *convert_CByteImage_to_array(CByteImage img) {
	CShape shape = img.Shape();
	float *array = (float *)malloc(sizeof(float) * shape.width * shape.height);
    assert(array != NULL);

	for (int x = 0; x < shape.width; x++) {
		for (int y = 0; y < shape.height; y++) {
			array[y * shape.width + x] = (img.Pixel(x,y,0))/255.0f;
		}
	}
	return array;
}

/* **************************************************************************************************** */

/* convert a linearly stored image to CByteImage
*/
CByteImage convert_array_to_CByteImage(float *img, int width, int height) {
	CByteImage cbimage(width, height, 1);

	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {
			cbimage.Pixel(x, y, 0) = (int)(img[y * width + x] * 255);
		}
	}
	return cbimage;
}

/* **************************************************************************************************** */

/* write image to file,
   image could either be in GPU or host memory
*/
void write_image_to_file(float *x_image, int width, int height, const char *file_name, bool from_device) {
	float *h_image;

	if (from_device) {
		h_image = (float *)malloc(sizeof(float) * width * height);
        assert(h_image != NULL);
		CHECK(cudaMemcpy(h_image, x_image, sizeof(float) * width * height, cudaMemcpyDeviceToHost));
	}
	else {
		h_image = x_image;
	}

	CByteImage cbimage = convert_array_to_CByteImage(h_image, width, height);
	WriteImage(cbimage, file_name);

	if (from_device)
        free(h_image);
}

#endif //_CUSTOM_UTILS_H