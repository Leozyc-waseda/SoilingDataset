


#include "bejewelled-board.H"
#include <vector>
#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <sstream>
#include <time.h>

using namespace std ;

template <class T> Matrix<T>::Matrix(int row , int col , T** m)
{
        mems =         *m ;
        r = row ;
        c = col ;
}

// ######################################################################
template <class T> Matrix<T>::~Matrix()

{
        delete[] mems;
}
// ######################################################################
template <class T> int Matrix<T>::getNumOfRows()
{
        return r ;
}
// ######################################################################
template <class T> int Matrix<T>::getNumOfColumns()
{
        return c ;
}
// ######################################################################
template <class T> void Matrix<T>::setNumOfRows(int nr)
{
        r = nr ;
}
// ######################################################################
template <class T> void Matrix<T>::setNumOfColumns(int nc)
{
        c = nc ;
}
// ######################################################################
template <class T> string Matrix<T>::toFormattedString()
{
        string s("") ;
        for (int i = 0 ; i < r ; i++){
                s += "<" ;
                for(int j = 0 ; j < c ; j++){
                        int index = i*c + j;
                        T m = *(mems + index);
                        s+=stringify(m)+" " ;
                }
                s += ">" ;
        }
        return s ;
}
// ######################################################################
template <class T> string Matrix<T>::toString()
{
        string s("") ;
        for (int i = 0 ; i < r ; i++){
                for(int j = 0 ; j < c ; j++){
                        int index = i*c + j;
                        T m = *(mems + index);
                        s+=stringify(m)+" " ;
                }
        }
        return s ;
}
// ######################################################################
template <class T> string Matrix<T>::stringify(T i)
{
        ostringstream o ;
        o << i ;
        return o.str();
}

 template <class T> T* Matrix<T>::operator[](int row){
     return &mems[row*c];
 }
// ######################################################################
Cell::Cell(const uint row , const uint col) {
        r = row ;
        c = col ;
}
// ######################################################################


Board::Board(Matrix<uint>& m){
        cstate = &m ;
}

Board::~Board(){
        delete cstate;
}

Board::getCurrentState(){
        cstate ;
}

Board::getObjectAtCell(Cell& cell){
        return (*cstate)[cell.r][cell.c];
}
// ######################################################################


int main(){
        float* a = new float[6] ;
        for (int i = 0 ; i < 6 ; i++){
                a[i] = i ;
        }
        Matrix<float> m(3,2,&a);
        //cout<<"hi : "<< m.get(0,1)<<endl ;
        m[0][0] = 21.45 ;
        m[0][1] = 0.00001 ;
        m[1][0] = 13 ;
        for(int i = 0 ; i < 2 ; i++ ){
                for( int j = 0 ; j < 3  ; j++ ){
                        cout<< m[i][j]<<endl ;
                }
        }
        cout<<m.toFormattedString()<<endl ;
        cout<<m.toString()<<endl;
        return 0 ;
}


