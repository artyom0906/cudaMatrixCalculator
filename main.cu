#include<cuda_runtime.h>
#include <sys/time.h>
#include "Matrix.cuh"
#include "Matrix.cu"

using namespace std;

void testDet(){
    Matrix<double> m_1(2, 3,
                       1.f, 2.f, 1.f,
                       0.f, 1.f, 2.f);
    Matrix<double> m_2(3, 2,
                       1.f, 0.f,
                       0.f, 1.f,
                       1.f, 1.f);
    /*Matrix m = m_1*m_2;
    m.print();*/

    Matrix<double> m(3, 3,
                     1., 0., 0.,
                          0., 5., 0.,
                          0., 0., 7.);
    //m.randomize();
    m.print();
    cout<<m();
    //printf("\n\n");
    /*Matrix<double> m1(1, 3,
              -2.,
                   -2.,
                    2.);
    //m1.randomize();
    m1.print();
    m.solve(m1);*/
    //cout<<m();
    //m.print();


/*
    Matrix m1(1000, 10000);
    m1.randomize();
    Matrix m2(10000, 1000);
    m2.randomize();


    Matrix m5(1000, 1000);
    m1.randomize();
    Matrix m6(1000, 1000);
    m2.randomize();
    cout<<"####++++####"<<endl;
    Matrix m7 = m5+m6;

    cout<<"####****####"<<endl;
    Matrix m3 = m2*m1;
    Matrix m4 = m2.multiplyCpu(m1);

    float error = 0.f;
    for(int i = 0; i < m3.getHeight(); i++){
        for(int j = 0; j < m3.getWidth(); j++){
            error = max(error, m3.data()[i*m3.getWidth()+j]-m4.data()[i*m3.getWidth()+j]);
            //cout<<m3.data()[i*m3.getWidth()+j]<<" "<<m4.data()[i*m3.getWidth()+j]<<endl;
        }
    }
    cout<<"error: "<<error;*/
    //m3.print();sh


}

int main() {

    return 0;
}