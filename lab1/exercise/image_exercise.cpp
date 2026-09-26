#include <iostream>
#include <cstdlib>
#include "image_tools.h"

using namespace Eigen;

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <image_path>" << " <output_path> "<< std::endl;
    return 1;
  }
  const char* input_path = argv[1];
  const char* output_path = argv[2];

  int result;
  MatrixXd grey;
  
  auto rotate = [](MatrixXd& output, unsigned char* image_data, int width, int height) {
    output.resize(width, height);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int index = i * width + j; // Since forced to 1 channel, stride is width
            double val = static_cast<double>(image_data[index]) / 255.0;
            output(j, height - 1 - i) = val;
        }
    }
  };


  double shift = 50.0 / 255.0;
  auto darken = [shift](MatrixXd& output, unsigned char* image_data, int width, int height) {
    output.resize(height, width);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int index = i * width + j; // Since forced to 1 channel, stride is width
            double val = static_cast<double>(image_data[index]) / 255.0;
            output(i, j) = std::max(0.0, val - shift);
        }
    }
  };
  
  result = loadImage2Matrix(grey, input_path, rotate);

  if (result == 1){
    return 1;
  }

  result = greyMatrix2File(grey, output_path);
  return result;
}
