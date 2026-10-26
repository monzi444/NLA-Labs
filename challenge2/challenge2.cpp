#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cstdlib>
#include <cstdio>
#include "utils/image_tools.hpp"
#include <unsupported/Eigen/SparseExtra>

using namespace Eigen;

void randomize(MatrixXd& input, MatrixXd& output){
    for (int i = 0; i < input.rows(); ++i) {
        for (int j = 0; j < input.cols(); ++j) {
            double val = input(i,j);
            float shift = ((rand()%121) -60);
            output(i, j) = std::max(0.0, val - shift);
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return 1;
    }
    const char* input_path = argv[1];
    
    int result;
    MatrixXd A;

    // Task 1
    std::cout << "Task 1: " << std::endl;
    result = loadImage2Matrix(A, input_path);
    if(result == 1){
        return 1;
    }
    MatrixXd AtA = A.transpose() * A;
    std::cout << "Norm of AtA = " << AtA.norm() << std::endl;

    // Task 2
    
    std::cout << "Task 2: " << std::endl;
    SelfAdjointEigenSolver<MatrixXd> saeigensolver(AtA);
    if (saeigensolver.info() != Eigen::Success) abort();
    VectorXd eigenValues = saeigensolver.eigenvalues();
    std::cout << "The two largest eigenvalues of AtA are: " << std::endl << eigenValues.tail(2) << std::endl;


    // Task 3, 4
    
    std::cout << "Task 3: " << std::endl;
    std::remove("resources/AtA.mtx");
    saveMarket(AtA, "resources/AtA.mtx");
    std::cout << "Exmported matrix to: resources/AtA.mtx" << std::endl;
    
    // execute lis solver
    const char* lis_cmd = "mpirun -n 4 ./utils/eigen1 resources/AtA.mtx resources/eigvec.txt resources/hist.txt -e ii -tol 1.0e-10 -shift 1.09864e+08";
    std::cout << "Executing: " << lis_cmd << std::endl;
    int sys_res = std::system(lis_cmd);
    if (sys_res != 0) {
        std::cerr << "Error: LIS solver failed with exit code " << sys_res << std::endl;
        return 1;
    }
    

    // Task 4
    
    std::cout << "Task 4: " << std::endl;
    double u = 1.09864e+08;
    SparseMatrix<double> I (AtA.rows(),AtA.cols());
    I.setIdentity();
    MatrixXd AtA_u = AtA - u * I;
    SelfAdjointEigenSolver<MatrixXd> saeigensolver_u(AtA_u);
    if (saeigensolver_u.info() != Eigen::Success) abort();
    eigenValues = saeigensolver_u.eigenvalues();
    std::cout << "The two largest eigenvalues of AtA-u are: " << std::endl << eigenValues.tail(2) << std::endl;
    std::cout << "Ratio between first and second: " << eigenValues.tail(2)[0] / eigenValues.tail(2)[1] << std::endl;
    
    
    // Task 5
    // compute Eigen SVD
    std::cout << "Task 5: " << std::endl;

    Eigen::BDCSVD<Eigen::MatrixXd> svd (A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    VectorXd W = svd.singularValues();
    // std::cout << "singular values: " << W << "\n";
    MatrixXd sigma = svd.singularValues().asDiagonal();
    std::cout << "norm of Sigma: " << sigma.norm() << std::endl;

    // Task 6
    std::cout << "Task 6: " << std::endl;

    int k = 90;
    int m = A.rows();
    int n = A.cols();

    MatrixXd C = MatrixXd(m, k);
    C = svd.matrixU().topLeftCorner(m, k);

    MatrixXd D = MatrixXd(n, k);
    //nk =  nk * kk = nk
    D = svd.matrixV().topLeftCorner(n, k) * sigma.topLeftCorner(k,k);

    std::cout << "nonzeros of C: " << C.nonZeros() << std::endl;
    std::cout << "nonzeros of D: " << D.nonZeros() << std::endl;

    // Task7
    std::cout << "Task 7: " << std::endl;

    MatrixXd compressed = C * D.transpose();
    compressed = compressed.cwiseMax(0.0).cwiseMin(255.0);
    result = greyMatrix2File(compressed, "resources/compressed.png");
    if (result == 1){
        return 1;
    }

    // Task 8
    std::cout << "Task 8: " << std::endl;

    n = 400;
    int check_size = 40;
    MatrixXd checked = MatrixXd(n,n);
    // -1 = black, 1 = white
    int col = -1;
    for (int i=0; i < n; ++i){
        if (i%check_size == 0){
            col = col * -1;
        }
        for (int j=0; j < n; ++j){
            if (j%check_size == 0){
                col = col * -1;
            }

            if(col == -1){
                checked(i,j) = 0.0;
            }
            else{
                checked(i,j) = 255.0;
            }
        }
    }
    std::cout << "norm of checked: " << checked.norm() << std::endl;
    //checked = checked.cwiseMax(0.0).cwiseMin(255.0);

    result = greyMatrix2File(checked, "resources/checked.png");
    if (result == 1){
        return 1;
    }

    // Task 9
    std::cout << "Task 9: " << std::endl;

    MatrixXd noisy = MatrixXd(n,n);
    randomize(checked, noisy);
    noisy = noisy.cwiseMax(0.0).cwiseMin(255.0);
    result = greyMatrix2File(noisy, "resources/noisy.png");
    if (result == 1){
        return 1;
    }

    // Task 10
    std::cout << "Task 10: " << std::endl;

    Eigen::BDCSVD<Eigen::MatrixXd> svd_noisy (noisy, Eigen::ComputeThinU | Eigen::ComputeThinV);
    W = svd_noisy.singularValues();
    std::cout << "largest two singular values: " << W[0] << " " << W[1] << std::endl;
    
    k = 6;
    m = noisy.rows();
    n = noisy.cols();

    sigma = W.asDiagonal();
    C = MatrixXd(m, k);
    C = svd_noisy.matrixU().leftCols(k);

    D = MatrixXd(n, k);
    //nk =  nk * kk = nk
    D = svd_noisy.matrixV().leftCols(k) * sigma.topLeftCorner(k,k);

    std::cout << "size of C: " << C.rows() << "X" << C.cols() << std::endl;
    std::cout << "size of D: " << D.rows() << "X" << D.cols() << std::endl;

    // Task 11
    std::cout << "Task 11: " << std::endl;

    MatrixXd noisy_compressed = C * D.transpose();
    noisy_compressed = noisy_compressed.cwiseMax(0.0).cwiseMin(255.0);
    noisy = noisy.cwiseMax(0.0).cwiseMin(255.0);
    result = greyMatrix2File(noisy_compressed, "resources/noisy_compressed.png");
    if (result == 1){
        return 1;
    }

    return 0;

    // Task 12
    // comment the results:
    /*
        - the compressed image is not anymore black and white, but its checks are of two different greys, one lighter one darker
          due to the information loss of the compression.
        - the "amount" of noise increases in the compressed image.
        - in the compressed image some "row pattern" seems to appear in the noise.
    */
}
