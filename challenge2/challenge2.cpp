#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cstdlib>
#include <cstdio>
#include "utils/image_tools.hpp"
#include <unsupported/Eigen/SparseExtra>

using namespace Eigen;

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
    /*
    std::cout << "Task 3: " << std::endl;
    std::remove("resources/AtA.mtx");
    saveMarket(AtA, "resources/AtA.mtx");
    std::cout << "Exmported matrix to: resources/AtA.mtx" << std::endl;
    
    // execute lis solver
    const char* lis_cmd = "mpirun -n 4 ./utils/eigen1 resources/AtA.mtx resources/eigvec.txt resources/hist.txt -e pi -tol 1.0e-10 -shift 0.03e8";
    std::cout << "Executing: " << lis_cmd << std::endl;
    int sys_res = std::system(lis_cmd);
    if (sys_res != 0) {
        std::cerr << "Error: LIS solver failed with exit code " << sys_res << std::endl;
        return 1;
    }
    */

    // Task 4
    /*
    std::cout << "Task 4: " << std::endl;
    double u = 0.03e8;
    SparseMatrix<double> I (AtA.rows(),AtA.cols());
    I.setIdentity();
    MatrixXd AtA_u = AtA - u * I;
    SelfAdjointEigenSolver<MatrixXd> saeigensolver_u(AtA_u);
    if (saeigensolver_u.info() != Eigen::Success) abort();
    eigenValues = saeigensolver_u.eigenvalues();
    std::cout << "The two largest eigenvalues of AtA-u are: " << std::endl << eigenValues.tail(2) << std::endl;
    std::cout << "Ratio between first and second: " << eigenValues.tail(2)[0] / eigenValues.tail(2)[1] << std::endl;
    */
    
    // Task 5
    // compute Eigen SVD
    std::cout << "Task 5: " << std::endl;

    Eigen::BDCSVD<Eigen::MatrixXd> svd (A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    VectorXd W = svd.singularValues();
    std::cout << "singular values: " << W << "\n";
    MatrixXd sigma = svd.singularValues().asDiagonal();
    std::cout << "norm of Sigma: " << sigma.norm() << std::endl;

    // Task 6
    std::cout << "Task 6: " << std::endl;

    int k = 30;
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

    return 0;
}

