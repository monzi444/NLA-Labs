#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;

float euclidean_norm(MatrixXd A){
    int n = A.rows();
    double result = 0;
    for (int i = 0; i < n; ++i){
        for (int j = 0; j < n; ++j){
            result += A(i,j) * A(i,j);
        }
    }
    return std::sqrt(result);
}


int main(int argc, char* argv[]){
    int n = 10;
    MatrixXd A = MatrixXd::Zero(n,n);

    for (int i = 0; i < n; ++i){
        for (int j = 0; j < n; ++j){
            if(i==j){
                A(i,j) = 2;
            }
            else if (i == j-1){
                A(i,j) = 1;
            }
            else if (i == j+1){
                A(i,j) = -1;
            }
        }
    }
    
    std::cout << "A:" << std::endl << A << std::endl; 
    std::cout << std::endl;

    std::cout << "Euclidean norm of A: " << euclidean_norm(A) << std::endl;
    std::cout << std::endl;

    std::cout << "Symetric part of A: " << std::endl << (A + A.transpose())/2 << std::endl;
    std::cout << std::endl;

    VectorXd b = VectorXd::Ones(n/2);
    std::cout << "n/2 x n/2 bottom right block of A times (1,1,...,1): " << std::endl << A.bottomRightCorner(n/2, n/2) * b << std::endl;
    std::cout << std::endl;
    
    VectorXd a = A.row(0).head(n/2);
    std::cout << "Dot product between the first half of the first row of A and (1,1,...,1): " << std::endl << a.dot(b) << std::endl;
}