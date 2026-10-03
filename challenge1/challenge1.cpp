#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
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
  MatrixXd original_image;
  MatrixXd noisy_image;
  
  // Task 2
  auto randomize = [](MatrixXd& output, unsigned char* image_data, int width, int height) {
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
  
  // Task 1, 2
  // result = loadImage2Matrix(noisy_image, input_path, randomize);
  result = loadImage2Matrix(original_image, input_path);

  if (result == 1){
    return 1;
  }

  // Task 2
  /*
  result = greyMatrix2File(noisy_image, output_path);
  if (result == 1){
    return 1;
  }
  */
  
  // Task 3
  VectorXd vector_original = VectorXd::Zero(original_image.rows() * original_image.cols());
  VectorXd vector_noisy = VectorXd::Zero(noisy_image.rows() * noisy_image.cols());
  
  // a11 a12 ... a1j a21 a22 ... a2j
  for(int i = 0; i < original_image.rows() -1; ++i){
    vector_original.segment(i* original_image.cols(), original_image.cols()) = original_image.row(i);
    vector_noisy.segment(i* noisy_image.cols(), noisy_image.cols()) = noisy_image.row(i);
  }

  //std::cout << "size of vector: " << vector_original.size() << " n*m: " << original_image.rows() * original_image.cols() << std::endl;
  //std::cout << "euclidean norm of v: " << vector_original.norm() << std::endl;
  
  // Task 4
  // matrix nn x nn
  MatrixXd kernel = MatrixXd(3,3);
  kernel << 1.0/12.0, 1.0/12.0, 1.0/12.0, 
            1.0/12.0, 4.0/12.0, 1.0/12.0, 
            1.0/12.0, 1.0/12.0, 1.0/12.0;

  int n = original_image.rows();

  SparseMatrix<double> h1(original_image.rows() * n, n * n);
  for (int k = 0; k < vector_original.size(); ++k){
    for(int i = 0; i < kernel.rows(); ++i){
      for(int j = 0; j < kernel.cols(); ++j){
        int original_matrix_row_index = k / n;
        int original_matrix_col_index = k - original_matrix_row_index * n;
        int row = (original_matrix_row_index - 1 + i)*n + (original_matrix_col_index - 1) + j;
        if(row < original_matrix_row_index * n){
          h1.insert(row, k) = kernel(i,j);
        }
      }
    }
  }

  VectorXd vector_blurred = vector_original * h1;
  MatrixXd matrix_blurred = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(vector_blurred.data(), n, n);

  return greyMatrix2File(matrix_blurred, output_path);

}