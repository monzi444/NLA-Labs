#include <iostream>
#include <Eigen/Dense>
#include <cstdlib>
#include "utils/image_tools.h"



using namespace Eigen;

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <image_path>" << " <output_path> "<< std::endl;
    return 1;
  }
  const char* input_path = argv[1];
  const char* output_path = argv[2];

  int result;
  MatrixXd image;
  

  auto darken = [](MatrixXd& output, unsigned char* image_data, int width, int height) {
    output.resize(height, width);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int index = i * width + j; // Since forced to 1 channel, stride is width
            double val = static_cast<double>(image_data[index]) / 255.0;
            float shift = ((rand()%101) -50) / 255.0;
            output(i, j) = std::max(0.0, val - shift);
        }
    }
  };
  
  result = loadImage2Matrix(image, input_path, darken);

  if (result == 1){
    return 1;
  }

  result = greyMatrix2File(image, output_path);
  return result;
}