# Basic linear algebra with LIS

Read the documentation available at

- [LIS documentation](https://www.ssisc.org/lis/index.en.html)

Lis (Library of Iterative Solvers for linear systems) is a parallel software library for solving discretized linear equations and eigenvalue problems that arise in the numerical solution of partial differential equations using iterative methods. 

As a first example of the usage of LIS we aim to run the first test reported in the user guide. To do so, follow the next steps:

1. To enable `mk` modules, you should type each time you open the terminal 
```bash
source /u/sw/etc/bash.bashrc
``` 

2. Once enabled, they provide the command module. Load the LIS module with:
```bash
module load gcc-glibc
module load lis
```

3. Download the [latest version](https://www.ssisc.org/lis/dl/lis-2.1.13.zip) of LIS and copy the unzipped folder into you shared folder (only the `lis-2.1.13/test/` folder is needed);

4. Rename and move into the test folder, e.g. `cd lis2.1.13_test/` (you can provide another name to the folder if you prefer)

5. Compile the file `test1.c` by typing 

```
mpicc -DUSE_MPI -I${mkLisInc} -L${mkLisLib} -llis test1.c -o test1
```

6. Run the test using the commands

```
./test1 testmat0.mtx 1 sol.txt hist.txt

./test1 testmat0.mtx testvec0.mtx sol.txt hist.txt
```

6. (b) To run the code with multiprocessors using mpi, type 

```
mpirun -n 4 ./test1 testmat0.mtx 1 sol.txt hist.txt
```

The option `-n` sets the number of processes (the default value is 4).

- Exercise:  Following the instructions available on the Lis user-guide, compile and run test2.c