/*************************************************************************
/* ECE 285: GPU Programmming 2019 Fall quarter
/* Author: Group_G
/* Copyright 2019
/* University of California, San Diego
/*************************************************************************/

#include "utils.h"
#include "canny_edge_host.h"

#define INPUT_FILE_NAME "C:/Users/r4gupta/Downloads/final_exam/ECE285_GPU_Programming/Input_images/range_rover_1920_1080.png"
#define OUTPUT_GAUSSIATED_FILE_NAME "C:/Users/r4gupta/Downloads/final_exam/ECE285_GPU_Programming/Output_images/tree_gaussiated.png"
#define OUTPUT_SOBELED_GRAD_X_FILE_NAME "C:/Users/r4gupta/Downloads/final_exam/ECE285_GPU_Programming/Output_images/tree_sobeled_grad_x.png"
#define OUTPUT_SOBELED_GRAD_Y_FILE_NAME "C:/Users/r4gupta/Downloads/final_exam/ECE285_GPU_Programming/Output_images/tree_sobeled_grad_y.png"
//#define OUTPUT_GAUSSIATED_FILE_NAME "C:/Users/r4gupta/Downloads/final_exam/ECE285_GPU_Programming/Output_images/tree_gaussiated.png"
#define OUTPUT_NON_MAX_SUPPRESSED_FILE_NAME "C:/Users/r4gupta/Downloads/final_exam/ECE285_GPU_Programming/Output_images/tree_nms.png"
#define OUTPUT_DOUBLE_THRESHOLDED_FILE_NAME "C:/Users/r4gupta/Downloads/final_exam/ECE285_GPU_Programming/Output_images/tree_double_thresholded.png"
#define OUTPUT_EDGE_TRACKED_FILE_NAME "C:/Users/r4gupta/Downloads/final_exam/ECE285_GPU_Programming/Output_images/tree_edge_tracked.png"

//#define DEBUG

int main(int argc, char **argv) {

#ifdef DEBUG
	char file_name[50];
	sprintf(file_name, "log_ours.txt");

	FILE *f = fopen(file_name, "w");
#endif //DEBUG

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

#ifdef DEBUG
	fprintf(f, "image %s\n", INPUT_FILE_NAME);
	print_log_matrix(f, gimage.get_host_gimage(), from_host.get_width(), from_host.get_height());

	fprintf(f, "gaussian_kernel\n");
	print_log_matrix(f, from_host.get_gaussian_kernel(), GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE);

	fprintf(f, "sobel_filter_x\n");
	print_log_matrix(f, from_host.get_sobel_filter_x(), SOBEL_FILTER_SIZE, SOBEL_FILTER_SIZE);
	fprintf(f, "sobel_filter_y\n");
	print_log_matrix(f, from_host.get_sobel_filter_y(), SOBEL_FILTER_SIZE, SOBEL_FILTER_SIZE);
#else
	from_host.print_gaussian_kernel();
	from_host.print_sobel_filters();
#endif //DEBUG

	from_host.apply_gaussian_kernel();
#ifdef DEBUG
	fprintf(f, "gaussiated_image\n");
	print_log_matrix(f, from_host.get_gaussiated_image(), from_host.get_width(), from_host.get_height());
#endif //DEBUG
	write_image_to_file(from_host.get_gaussiated_image(), from_host.get_width(), from_host.get_height(), OUTPUT_GAUSSIATED_FILE_NAME, false);

	from_host.compute_pixel_thresholds();

	from_host.apply_sobel_filter_x();
#ifdef DEBUG
	fprintf(f, "sobeled_grad_x_image\n");
	print_log_matrix(f, from_host.get_sobeled_grad_x_image(), from_host.get_width(), from_host.get_height());
#endif //DEBUG
	write_image_to_file(from_host.get_sobeled_grad_x_image(), from_host.get_width(), from_host.get_height(), OUTPUT_SOBELED_GRAD_X_FILE_NAME, false);

	from_host.apply_sobel_filter_y();
#ifdef DEBUG
	fprintf(f, "sobeled_grad_y_image\n");
	print_log_matrix(f, from_host.get_sobeled_grad_y_image(), from_host.get_width(), from_host.get_height());
#endif //DEBUG
	write_image_to_file(from_host.get_sobeled_grad_y_image(), from_host.get_width(), from_host.get_height(), OUTPUT_SOBELED_GRAD_Y_FILE_NAME, false);

	from_host.calculate_sobel_magnitude();
#ifdef DEBUG
	fprintf(f, "sobeled_mag_image\n");
	print_log_matrix(f, from_host.get_sobeled_mag_image(), from_host.get_width(), from_host.get_height());
#endif //DEBUG
	//write_image_to_file(from_host.get_gaussiated_image(), from_host.get_width(), from_host.get_height(), OUTPUT_GAUSSIATED_FILE_NAME, false);

	from_host.calculate_sobel_direction();
#ifdef DEBUG
	fprintf(f, "sobeled_dir_image\n");
	print_log_matrix(f, from_host.get_sobeled_dir_image(), from_host.get_width(), from_host.get_height());
#endif //DEBUG

	from_host.apply_non_max_suppression();
#ifdef DEBUG
	fprintf(f, "non_max_suppressed_image\n");
	print_log_matrix(f, from_host.get_non_max_suppressed_image(), from_host.get_width(), from_host.get_height());
#endif //DEBUG
	write_image_to_file(from_host.get_non_max_suppressed_image(), from_host.get_width(), from_host.get_height(), OUTPUT_NON_MAX_SUPPRESSED_FILE_NAME, false);

	from_host.apply_double_thresholds();
#ifdef DEBUG
	fprintf(f, "double_thresholded_image\n");
	print_log_matrix(f, from_host.get_double_thresholded_image(), from_host.get_width(), from_host.get_height());
#endif //DEBUG
	write_image_to_file(from_host.get_double_thresholded_image(), from_host.get_width(), from_host.get_height(), OUTPUT_DOUBLE_THRESHOLDED_FILE_NAME, false);

	from_host.apply_hysteresis_edge_tracking();
#ifdef DEBUG
	fprintf(f, "edge_tracked_image\n");
	print_log_matrix(f, from_host.get_edge_tracked_image(), from_host.get_width(), from_host.get_height());
#endif //DEBUG
	write_image_to_file(from_host.get_edge_tracked_image(), from_host.get_width(), from_host.get_height(), OUTPUT_EDGE_TRACKED_FILE_NAME, false);

#ifdef DEBUG
	fclose(f);
#endif //DEBUG

	printf("CPU took %.2fms\n", from_host.get_total_time_taken());
}