#include<cuda_runtime.h>
#include "Matrix.cuh"
#include "Matrix.cu"

using namespace std;

void testDet(){
//    Matrix<double> m_1(2, 3,
//                       1.f, 2.f, 1.f,
//                           0.f, 1.f, 2.f);
//    Matrix<double> m_2(3, 2,
//                       1.f, 0.f,
//                           0.f, 1.f,
//                           1.f, 1.f);
    //Matrix<double> m_3 = m_1 * m_2;
    //m_3.print();
    //printf("\n\n");

//    Matrix<double> m(3, 3,
//                     1., 0., 0.,
//                          0., 5., 0.,
//                          0., 0., 7.);
//    //m.randomize();
//    m.print();
//    cout<<m();
//    printf("\n\n");
//    Matrix<double> m1(1, 3,
//              -2.,
//                   -2.,
//                    2.);
//    //m1.randomize();
//    m1.print();
//    m.solve(m1);
    //cout<<m();
    //m.print();



//    Matrix<float> m1(3, 3,
//                    -1.f,  1.f,  6.f,
//                    -4.f, -8.f,  6.f,
//                     2.f, 16.f, 23.f);

//Matrix<float> A(4, 4,
//                7.f, 3.f, -1.f,  2.f,
//                3.f, 8.f,  1.f, -4.f ,
//                -1.f, 1.f,  4.f, -1.f,
//                2.f,-4.f, -1.f,  6.f);

//2 x1 + 7 x2 - 5 x3 + 2 x4 = -91
//2 x1 - 3 x3 - 10 x4 = -84
//8 x1 - 10 x2 - 4 x3 + x4 = 64
//-6 x1 - 3 x2 - 8 x3 + 4 x4 = -13


    Matrix<float> A(4, 4,
          2.f,  7.f,-5.f, 2.f,
          2.f, -3.f, 0.f,-10.f,
         -8.f,-10.f,-4.f, 1.f,
         -6.f, -3.f,-8.f, 4.f);



    Matrix<float> B(4, 1, -91.f, -84.f, -64.f, -13.f);

    //A.solve(B);
    auto t1 = high_resolution_clock::now();
    auto [P, L, U] = A.PLU_factorisation();
    auto t2 = high_resolution_clock::now();
    auto time = duration_cast<microseconds>(t2 - t1);
    printf("Time to calculate on gpu with memory copy: %d us %1.5f s \n", time.count(), time.count()*1e-6);

    std::cout<<"A = ******\n";
    A.print();
    std::cout<<"P = ******\n";
    P.print();
    std::cout<<"L = ******\n";
    L.print();
    std::cout<<"U = ******\n";
    U.print();
    std::cout<<"******\n\n\n\n";
    Matrix<float> Bs = P*B;
    Bs = L.forward_substitution(Bs);
    Bs = U.backward_substitution(Bs);
    std::cout<<"Solution PLU = ******\n";
    Bs.print();

    std::cout<<"\n\n*************************GAUS*************************\n\n";

    auto [A1, B1] = A.solve(B);

    std::cout<<"A1 = ******\n";
    A1.print();
    std::cout<<"B1 = ******\n";
    B1.print();
    Bs = A1.backward_substitution(B1);
    std::cout<<"Solution Gaus = ******\n";
    Bs.print();






    //m1.print();
    //Matrix<float> m2 = m1.solve(Matrix<float>(1, 3, 0.f, 13.f, 14.f));
    //m2.print();
//    Matrix<float> m2(20'000, 20'000);
//    m2.randomize();
//
//    const int size = 20'000;
//    Matrix<float> m5(size, size);
//    m5.randomize();
//    Matrix<float> m6(size, size);
//    m6.randomize();
//
//    cout<<"####****####"<<endl;
//    Matrix<float> m3 = m2*m1;
//    Matrix<float> m4 = m2.multiplyCpu(m1);
//
//    cout<<"####++++####"<<endl;
//    Matrix<float> m7 = m5 + m6;
//
//    float error = 0.f;
//    for(int i = 0; i < m3.getHeight(); i++){
//        for(int j = 0; j < m3.getWidth(); j++){
//            error = max(error, m3.data()[i*m3.getWidth()+j]-m4.data()[i*m3.getWidth()+j]);
//            //cout<<m3.data()[i*m3.getWidth()+j]<<" "<<m4.data()[i*m3.getWidth()+j]<<endl;
//        }
//    }
//    cout<<"error: "<<error;
    //m3.print();sh


}

int main() {
    testDet();
    return 0;
}