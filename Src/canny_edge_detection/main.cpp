#include "utils.h"

#define INPUT_FILE_NAME ""

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