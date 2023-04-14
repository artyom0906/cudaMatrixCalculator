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



    Matrix<float> m1(3, 3,
                     2.f, -1.f, 0.f,
                    -1.f,  1.f, 4.f,
                     1.f,  2.f, 3.f);
    //m1.print();
    Matrix<float> m2 = m1.solve(Matrix<float>(1, 3, 0.f, 13.f, 14.f));
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