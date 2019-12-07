/*************************************************************************
/* ECE 285: GPU Programmming 2019 Fall quarter
/* Author: Group_G
/* Copyright 2019
/* University of California, San Diego
/*************************************************************************/

/* 	creates a basic handler for a grayscale image
*/

#ifndef __GRAY_IMAGE_H
#define __GRAY_IMAGE_H

#include "common.h"

class gray_image {
	private:
		float *h_gimage;
		int width;
		int height;

	public:
		gray_image(float *h_gimage, int width, int height);
		~gray_image();

		float *get_host_gimage();
		int get_width();
		int get_height();
		float get_pixel(int x, int y);
};

#endif //__GRAY_IMAGE_H