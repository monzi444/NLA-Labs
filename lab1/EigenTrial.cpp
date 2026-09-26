#include <iostream>
#include <Eigen/Dense>
 
using namespace std;
using Eigen::VectorXd;
using Eigen::MatrixXd;
 


void test1(){
    MatrixXd A = MatrixXd::Random(3,3);
    std::cout << "A =" << std::endl << A << std::endl;
    A = (A + MatrixXd::Constant(3,3,1.0)) * 10;
    std::cout << "A =" << std::endl << A << std::endl;
    VectorXd v(3);
    v << 1, 0, 0;
    std::cout << "A * v =" << std::endl << A * v << std::endl;
}

void test2(){
  VectorXd v(6);
  v << 1, 2, 3, 4, 5, 6;
  cout << "v.head(3) =" << endl << v.head(3) << endl << endl;
  cout << "v.tail<3>() = " << endl << v.tail<3>() << endl << endl;
  v.segment(1,4) *= 2;
  cout << "after 'v.segment(1,4) *= 2', v =" << endl << v << endl;

  MatrixXd A = MatrixXd::Random(9,9);
  MatrixXd B = A.topLeftCorner(3,6);
  VectorXd w = B*v;
  cout << "norm of B*v = " << w.norm() << endl;
}

void test3(){
    float x = 33554432.0;
    cout << x << endl;
}

void exercise1(){
    //define empty matrix
    int n = 100;
    MatrixXd A = MatrixXd::Zero(n,n);
    
    //fill diagonal with 2s
    A.diagonal().array() = 2;

    //fill elements left of the diagonal with 1 and right with -1
    for (int i = 0; i < n; i++) {
        // A(i,i) = 2;
        if (i>0) A(i, i-1) = 1;
        if (i < n-1) A(i, i+1) = -1;
    }
    cout << A << endl;
}

int main(){
    test3();
    return 0;
}