#ifndef IMAGE_TOOLS_H
#define IMAGE_TOOLS_H

#include <Eigen/Dense>
#include <functional>
#include <string>

// Load an RGB image from filesystem
int loadImage2Matrix(Eigen::MatrixXd& red,
                     Eigen::MatrixXd& green,
                     Eigen::MatrixXd& blue,
                     const std::string& input_path);


// Load a greyscal image from filesystem and optionally apply a custom resizing and pixel mapping function
int loadImage2Matrix(Eigen::MatrixXd& output,
                     const std::string& input_path,
                     std::function<void(Eigen::MatrixXd& output, unsigned char* image_data, int width, int height)> map =
                        [](Eigen::MatrixXd& output, unsigned char* image_data, int width, int height) {
                            output.resize(height, width);
                            for (int i = 0; i < height; ++i) {
                                for (int j = 0; j < width; ++j) {
                                    int index = i * width + j; // Since forced to 1 channel, stride is width
                                    double val = static_cast<double>(image_data[index]) / 255.0;
                                    output(i, j) = val;
                                }
                            }
                        }
                    );


// Save greyscale matrix to filesystem
int greyMatrix2File(const Eigen::MatrixXd& grey,
                    const std::string& output_path);


// Convert RGB matrices to greyscale matrix
Eigen::MatrixXd convertTogreyscale(const Eigen::MatrixXd& red,
                                   const Eigen::MatrixXd& green,
                                   const Eigen::MatrixXd& blue);

#endif // IMAGE_TOOLS_H
