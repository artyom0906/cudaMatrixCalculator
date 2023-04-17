//
// Created by artyom on 9/8/21.
//

#ifndef CUDA_MATRIX_CALCULATOR_MATRIX_CUH
#define CUDA_MATRIX_CALCULATOR_MATRIX_CUH
#define TILE_DIM 16                     // Tile dimension
#include <iostream>
#include <cstdarg>
#include <chrono>
#include <iomanip>
#include <type_traits> // enable_if, conjuction

#include <Gaus.cuh>
#include <utils.cuh>
#include <PLUfactorisation.cuh>


using namespace std::chrono;
using namespace std;



template<class Head, class... Tail>
using are_same = std::conjunction<std::is_same<Head, Tail>...>;

template<typename T>
class Matrix {
public:
    explicit Matrix(int width, int height);

    template<typename ... Args, typename = std::enable_if_t<are_same<T, Args...>::value, void>>
    explicit Matrix(int width, int height, Args... args);

    Matrix operator*(Matrix m1);

    Matrix operator>>(Matrix m1);

    Matrix operator+(Matrix m1);

    int operator()();

    Matrix multiplyCpu(Matrix m1);

    Matrix<T> forward_substitution(Matrix<T> b);
    Matrix<T> backward_substitution(Matrix<T> b);

    std::tuple<Matrix, Matrix, Matrix> PLU_factorisation();

    /*
     * @param results - vector of SLAE results
     *
     * @return
     */
    std::tuple<Matrix<T>, Matrix<T>> Matrix<T>::solve(Matrix b);


    void randomize();
    void eye();
    T* data();

    void print() const;
    int getHeight() const {
        return height;
    }

    int getWidth() const {
        return width;
    }

    Matrix<T> to_triangle();

  private:
    T* matrix;
    int height{};
    int width{};
};


#endif //CUDAMATRIXCALCULATOR_MATRIX_CUH
