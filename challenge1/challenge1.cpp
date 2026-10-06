#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cstdlib>
#include <cstdio>
#include "utils/image_tools.h"
#include <unsupported/Eigen/SparseExtra>


using namespace Eigen;

SparseMatrix<double> build (MatrixXd& kernel, int n){
  SparseMatrix<double> h(n * n, n * n);
  for (int k = 0; k < n*n; ++k){
    int original_matrix_row_index = k / n;
    int original_matrix_col_index = k - original_matrix_row_index * n;
    for(int i = 0; i < kernel.rows(); ++i){
      for(int j = 0; j < kernel.cols(); ++j){
        int a_i = (original_matrix_row_index - 1 + i);
        int a_j = (original_matrix_col_index - 1) + j;
        int row = a_i * n + a_j;
        if(a_i >= 0 && a_j >= 0 && a_i < n && a_j < n){
          h.insert(row, k) = kernel(i,j);
        }
      }
    }
  }
  h.makeCompressed();
  return h.transpose();
}

VectorXd import_vector_market(const std::string& input_path) {
    std::ifstream input(input_path);
    if (!input.is_open()) {
        std::cerr << "Errore: impossibile aprire il file " << input_path << std::endl;
        return VectorXd();
    }

    std::vector<double> values;
    std::string row;
    bool is_first_data_line = true;

    while (std::getline(input, row)) {
        // Ignora i commenti standard di Matrix Market
        if (row.empty() || row[0] == '%') continue;

        std::istringstream iss(row);
        std::string first_col, second_col;
        
        if (iss >> first_col) {
            // Se esiste una seconda colonna, estraiamo quella (logica originale)
            if (iss >> second_col) {
                // Se è la primissima riga di dati, è l'header delle dimensioni (es: "M 1" o "M N nnz") -> la saltiamo
                if (is_first_data_line) {
                    is_first_data_line = false; 
                    continue;
                }
                try { 
                    values.push_back(std::stod(second_col)); 
                } catch(...) {}
            } else {
                // Se c'è una sola colonna, estraiamo la prima (logica originale)
                if (is_first_data_line) {
                    is_first_data_line = false; 
                    continue;
                }
                try { 
                    values.push_back(std::stod(first_col)); 
                } catch(...) {}
            }
        }
    }
    
    input.close();

    // Map converte il std::vector nativo in un Eigen::VectorXd dinamicamente, senza passaggi su disco
    return Eigen::Map<VectorXd>(values.data(), values.size());
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
    return 1;
  }
  const char* input_path = argv[1];

  
  int result;
  MatrixXd original_image;
  MatrixXd noisy_image;
  
  // Task 2
  auto randomize = [](MatrixXd& output, unsigned char* image_data, int width, int height) {
    output.resize(height, width);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int index = i * width + j; // Since forced to 1 channel, stride is width
            double val = static_cast<double>(image_data[index]);
            float shift = ((rand()%101) -50);
            output(i, j) = std::max(0.0, val - shift);
        }
    }
  };
  
  // Task 1
  std::cout << std::endl;
  std::cout << "Task 1: " << std::endl;
  result = loadImage2Matrix(original_image, input_path);
  
  // Task 2
  std::cout << std::endl;
  std::cout << "Task 2: " << std::endl;
  result = loadImage2Matrix(noisy_image, input_path, randomize);
  if (result == 1){
    return 1;
  }
  noisy_image = noisy_image.cwiseMax(0.0).cwiseMin(255.0);

  // Task 2
  
  result = greyMatrix2File(noisy_image, "resources/noisy.png");
  if (result == 1){
    return 1;
  }
  
  
  // Task 3
  std::cout << std::endl;
  std::cout << "Task 3: " << std::endl;
  VectorXd vector_original = VectorXd::Zero(original_image.rows() * original_image.cols());
  VectorXd vector_noisy = VectorXd::Zero(noisy_image.rows() * noisy_image.cols());
  
  // a11 a12 ... a1j a21 a22 ... a2j
  for(int i = 0; i < original_image.rows(); ++i){
    vector_original.segment(i* original_image.cols(), original_image.cols()) = original_image.row(i);
    vector_noisy.segment(i* noisy_image.cols(), noisy_image.cols()) = noisy_image.row(i);
  }

  std::cout << "size of vector: " << vector_original.size() << " n*m: " << original_image.rows() * original_image.cols() << std::endl;
  std::cout << "euclidean norm of v: " << vector_original.norm() << std::endl;
  
  // Task 4
  std::cout << std::endl;
  std::cout << "Task 4: " << std::endl;
  MatrixXd kernel = MatrixXd(3,3);
  kernel << 1.0/12.0, 1.0/12.0, 1.0/12.0, 
            1.0/12.0, 4.0/12.0, 1.0/12.0, 
            1.0/12.0, 1.0/12.0, 1.0/12.0;
  
  int n = original_image.rows();
  
  SparseMatrix h1 = build(kernel, n);
  std::cout << "nonzero entries: " << h1.nonZeros() << std::endl;
  VectorXd vector_blurred = h1 * vector_original;
  MatrixXd matrix_blurred = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(vector_blurred.data(), n, n);
  matrix_blurred = matrix_blurred.cwiseMax(0.0).cwiseMin(255.0);
  
  result = greyMatrix2File(matrix_blurred, "resources/blurred.png");
  if (result == 1){
    return 1;
  }


  // Task 5
  std::cout << std::endl;
  std::cout << "Task 5: " << std::endl;
  VectorXd vector_noisy_blurred = h1 * vector_noisy;
  MatrixXd matrix_noisy_blurred = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(vector_noisy_blurred.data(), n, n);
  matrix_noisy_blurred = matrix_noisy_blurred.cwiseMax(0.0).cwiseMin(255.0);
  result = greyMatrix2File(matrix_noisy_blurred, "resources/noisy_blurred.png");
  if (result == 1){
    return 1;
  }
  

  // Task 6, 7
  MatrixXd kernel2 = MatrixXd(3,3);
  kernel2 << 0.0, -3.0, 0.0,
            -1.0, 9.0, -3.0,
            0.0, -1.0, 0.0;
  SparseMatrix h2 = build(kernel2, n);
  VectorXd vector_sharpened = h2 * vector_original;
  MatrixXd matrix_sharpened = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(vector_sharpened.data(), n, n);
  matrix_sharpened = matrix_sharpened.cwiseMax(0.0).cwiseMin(255.0);
  
  std::cout << std::endl;
  std::cout << "Task 6: " << std::endl;
  std::cout << "nonzero entries: " << h2.nonZeros() << std::endl;
  double norm = (SparseMatrix<double>(h2.transpose()) -h2).norm();
  if( norm == 0.0){
    std::cout << "h2 is symetric! norm of symetric part: " << norm << std::endl;
  }
  else{
    std::cout << "h2 is not symetric! norm of symetric part: " << norm << std::endl;
  }

  std::cout << std::endl;
  std::cout << "Task 7: " << std::endl;
  result = greyMatrix2File(matrix_sharpened, "resources/sharpened.png");
  if (result == 1){
    return 1;
  }
  

  // Task 8
  std::cout << std::endl;
  std::cout << "Task 8: " << std::endl;
  std::remove("resources/h2.mtx");
  saveMarket(h2, "resources/h2.mtx");
  std::cout << "matrix saved to resources/h2.mtx" << std::endl;


  int m = vector_noisy.size();
  std::remove("resources/w.mtx");
  FILE* out = fopen("resources/w.mtx", "w");
  fprintf(out, "%%%%MatrixMarket vector coordinate real general\n");
  fprintf(out, "%d\n", m);
  for( int i=0; i<m; i++){
    fprintf(out, "%d %.16e\n", i+1, vector_noisy(i));
  }
  fclose(out);

  std::cout << "vector saved to resources/w.mtx" << std::endl;

  // Task 8 (LIS solver execution)
  const char* lis_cmd = "mpirun -n 4 ./test1 resources/h2.mtx resources/w.mtx resources/sol.mtx resources/hist.txt -tol 1.0e-12 -i bicgstab -p ilu";
  std::cout << "Executing: " << lis_cmd << std::endl;
  int sys_res = std::system(lis_cmd);
  if (sys_res != 0) {
      std::cerr << "Error: LIS solver failed with exit code " << sys_res << std::endl;
      return 1;
  }
  
  // Task 9
  std::cout << std::endl;
  std::cout << "Task 9: " << std::endl;
  VectorXd solution = import_vector_market("resources/sol.mtx");
  MatrixXd solution_matrix = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(solution.data(), n, n);
  solution_matrix = solution_matrix.cwiseMax(0.0).cwiseMin(255.0);

  result = greyMatrix2File(solution_matrix, "resources/system_solution.png");
  if (result == 1){
    return 1;
  }

  // Task 10, 11
  MatrixXd kernel3 = MatrixXd(3,3);
  kernel3 << -1,0,1,
             -2,0,2,
             -1,0,1;
  SparseMatrix h3 = build(kernel3,n);
  VectorXd vector_detectioned = h3 * vector_original;
  MatrixXd matrix_detectioned = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(vector_detectioned.data(), n, n);
  matrix_detectioned = matrix_detectioned.cwiseMax(0.0).cwiseMin(255.0);

  // Task 10
  std::cout << std::endl;
  std::cout << "Task 10: " << std::endl;
  std::cout << "nonzero entries: " << h3.nonZeros() << std::endl;
  norm = (SparseMatrix<double>(h3.transpose()) -h3).norm();
  if( norm == 0.0){
    std::cout << "h3 is symetric! norm of symetric part: " << norm << std::endl;
  }
  else{
    std::cout << "h3 is not symetric! norm of symetric part: " << norm << std::endl;
  }

  // Task 11
  std::cout << std::endl;
  std::cout << "Task 11: " << std::endl;
  result = greyMatrix2File(matrix_detectioned,"resources/dectioned.png");
  if(result==1){
    return 1;
  }

  // Task 12
  std::cout << std::endl;
  std::cout << "Task 12: " << std::endl;
  SparseMatrix<double> I (n*n,n*n);
  I.setIdentity();
  SparseMatrix<double> mat = 4 * I + h3;
  double tol = 1.e-10;                  // tollerance
  int maxit = 1000;                     // max iterations

  //BiCGSTAB<SparseMatrix<double>, IncompleteLUT<double>> BiCG;
  BiCGSTAB<SparseMatrix<double>> BiCG;

  BiCG.setMaxIterations(maxit);
  BiCG.setTolerance(tol);
  BiCG.compute(mat); 

  solution = BiCG.solve(vector_noisy);

  std::cout << "Eigen native BiCGSTAB" << std::endl; // Ho corretto "CG" in "BiCGSTAB"
  std::cout << "# iterations:      " << BiCG.iterations() << std::endl;
  std::cout << "relative residual: " << BiCG.error()      << std::endl;

  // Task 13
  std::cout << std::endl;
  std::cout << "Task 13: " << std::endl;
  MatrixXd matrix_solution = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(solution.data(), n, n);
  matrix_solution = matrix_solution.cwiseMax(0.0).cwiseMin(255.0);

  result = greyMatrix2File(matrix_solution,"resources/system_solution_2.png");
  if(result==1){
    return 1;
  }
  return 0;
}