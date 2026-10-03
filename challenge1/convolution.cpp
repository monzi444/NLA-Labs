#include <iostream>
#include <Eigen/Dense>
#include <cstdlib>
#include "utils/image_tools.h"



using namespace Eigen;


double compute_convolution(const MatrixXd& image, int i, int j){
    Matrix3d convolution_matrix = MatrixXd(3,3);
    /*
    for(int i1 = 0; i1 < 3; ++i1){
        for(int j1 = 0; j1 < 3; ++j1){
            if(i1 == 1 && j1 == 1){
                convolution_matrix(i1,j1) = 4.0/12.0;
            }
            else{
                convolution_matrix(i1,j1) = 1.0/12.0;
            }
        }
    }
    */
    
    convolution_matrix << -1.0, -1.0, -1.0,
          -1.0,  9.0, -1.0,
          -1.0, -1.0, -1.0;
    
    double sum = 0.0;
    int n = 3, m = 3;
    for(int k = 0; k < n; ++k){
        for(int l = 0; l < m; ++l){
            // k,l,i,j = 0
            // 0 + 0 - (2)/2
          
            int r = i + k - (n-1)/2;
            int c =  j + l - (m-1)/2;

            if(r >= 0 && c >= 0 && r < image.rows() && c < image.cols()){
                sum += convolution_matrix(k,l) * image(r, c);
            }
        }
    }
    return sum;
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <image_path>" << " <output_path> "<< std::endl;
    return 1;
  }
  const char* input_path = argv[1];
  const char* output_path = argv[2];

  
  int result;

  MatrixXd image;
  result = loadImage2Matrix(image, input_path);
  double convolution = 0.0;
  MatrixXd result_image = MatrixXd(image.rows(), image.cols());
  for (int row = 0; row < image.rows(); ++row){
    for (int col = 0; col < image.cols(); ++col){
        std::cout << "i, j " << row << col << std::endl;
        convolution = compute_convolution(image, row, col);
        if(convolution < 0){
            convolution = 0;
        }
        else if (convolution > 255.0){
            convolution = 255.0;
        }
        result_image(row,col) = convolution;
    }
  }

  result = greyMatrix2File(result_image, output_path);
}