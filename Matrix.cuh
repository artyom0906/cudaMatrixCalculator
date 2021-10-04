//
// Created by artyom on 9/8/21.
//

#ifndef CUDA_MATRIX_CALCULATOR_MATRIX_CUH
#define CUDA_MATRIX_CALCULATOR_MATRIX_CUH
#define TILE_DIM 16                     // Tile dimension
#include <iostream>
#include <cstdarg>
using namespace std;

template<typename T>
class Matrix {
public:
    explicit Matrix(int width, int height);
    explicit Matrix(int width, int height, T arg...);

    Matrix operator*(Matrix m1);

    Matrix operator>>(Matrix m1);

    Matrix operator+(Matrix m1);

    int operator()();

    __attribute__((unused)) Matrix multiplyCpu(Matrix m1);

    /*
     * @param results - vector of SLAE results
     *
     * @return
     */
    Matrix solve(Matrix results);


    void randomize();
    T* data();

    void print() const;
    int getHeight() const {
        return height;
    }

    int getWidth() const {
        return width;
    }


private:
    T* matrix;
    int height{};
    int width{};

protected:
    Matrix<T> to_triangle();
};


#endif //CUDAMATRIXCALCULATOR_MATRIX_CUH
