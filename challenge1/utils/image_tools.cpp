#include <Eigen/Dense>
#include <iostream>
#include <cstdlib>

#define STB_IMAGE_IMPLEMENTATION
#include "../libs/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../libs/stb_image_write.h"

using namespace Eigen;

// Load an RGB image from filesystem
int loadImage2Matrix(MatrixXd& red, MatrixXd& green, MatrixXd& blue, const std::string& input_path){
    // Load the image using stb_image
    int width, height, channels;
    unsigned char* image_data = stbi_load(input_path.c_str(), &width, &height, &channels, 3); // Force load as RGB

    if (!image_data) {
        std::cerr << "Error: Could not load image " << input_path << std::endl;
        return 1;
    }
    
    red.resize(height, width);
    green.resize(height, width);
    blue.resize(height, width);

    // Fill the matrices with image data
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
        int index = (i * width + j) * 3; // 3 channels stride
            red(i, j) = static_cast<double>(image_data[index]) / 255.0;
            green(i, j) = static_cast<double>(image_data[index + 1]) / 255.0;
            blue(i, j) = static_cast<double>(image_data[index + 2]) / 255.0;
        }
    }

    // Free memory!!!
    stbi_image_free(image_data);
    
    std::cout << "Image loaded: " << width << "x" << height << " with " << channels << " channels." << std::endl;
    return 0;
}

// Load greyscale image and optionally applies a custom pixel transformer
int loadImage2Matrix(Eigen::MatrixXd& output,
                     const std::string& input_path,
                     std::function<void(Eigen::MatrixXd& output, unsigned char* image_data, int width, int height)> map =
                       [](MatrixXd& output, unsigned char* image_data, int width, int height) {
                           output.resize(height, width);
                           for (int i = 0; i < height; ++i) {
                               for (int j = 0; j < width; ++j) {
                                   int index = i * width + j; // Since forced to 1 channel, stride is width
                                   double val = static_cast<double>(image_data[index]) / 255.0;
                                   output(i, j) = val;
                               }
                           }
                        }
                    ){
    int width, height, channels;
    // Force loading 1 channel (grayscale)
    unsigned char* image_data = stbi_load(input_path.c_str(), &width, &height, &channels, 1);

    if (!image_data) {
        std::cerr << "Error: Could not load image " << input_path << std::endl;
        return 1;
    }

    map(output, image_data, width, height);

    stbi_image_free(image_data);
    std::cout << "Image loaded: " << width << "x" << height << std::endl;
    return 0;
}


// Save greyscale matrix to filesystem
int greyMatrix2File(const MatrixXd& grey, const std::string& output_path){
    int width = grey.cols();
    int height = grey.rows();
    // Allocate 2D matrix for 8bit usigned int 
    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> greyscale_image(width, height);
    
    // Use Eigen's unaryExpr to map the greyscale values (0.0 to 1.0) to 0 to 255
    greyscale_image = grey.unaryExpr([](double val) -> unsigned char {
        return static_cast<unsigned char>(val * 255.0);
    });

    // Save the greyscale image using stb_image_write
    if (stbi_write_png(output_path.c_str(), width, height, 1, greyscale_image.data(), width) == 0) {
        std::cerr << "Error: Could not save greyscale image" << std::endl;
        return 1;
    }

    std::cout << "greyscale image saved to " << output_path << std::endl;
    return 0;
}


// Convert RGB matrices to greyscale matrix
MatrixXd convertTogreyscale(const MatrixXd& red, const MatrixXd& green, const MatrixXd& blue) {
    return 0.299 * red + 0.587 * green + 0.114 * blue;
}
