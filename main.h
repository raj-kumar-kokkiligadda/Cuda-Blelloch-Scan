#ifndef MAIN_H
#define MAIN_H
template <class T>
void RunKernel0(T* pDevInput, T* pDevOutput, const unsigned int& roundSize);

template <class T>
void RunKernel1(T* pDevInput, T* pDevOutput, const unsigned int& roundSize);
#endif