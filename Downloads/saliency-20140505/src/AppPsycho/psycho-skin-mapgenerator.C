// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2001 by the //
// University of Southern California (USC) and the iLab at USC.         //
// See http://iLab.usc.edu for information about this project.          //
// //////////////////////////////////////////////////////////////////// //
// Major portions of the iLab Neuromorphic Vision Toolkit are protected //
// under the U.S. patent ``Computation of Intrinsic Perceptual Saliency //
// in Visual Environments, and Applications'' by Christof Koch and      //
// Laurent Itti, California Institute of Technology, 2001 (patent       //
// pending; application number 09/912,225 filed July 23, 2001; see      //
// http://pair.uspto.gov/cgi-bin/final/home.pl for current status).     //
// //////////////////////////////////////////////////////////////////// //
// This file is part of the iLab Neuromorphic Vision C++ Toolkit.       //
//                                                                      //
// The iLab Neuromorphic Vision C++ Toolkit is free software; you can   //
// redistribute it and/or modify it under the terms of the GNU General  //
// Public License as published by the Free Software Foundation; either  //
// version 2 of the License, or (at your option) any later version.     //
//                                                                      //
// The iLab Neuromorphic Vision C++ Toolkit is distributed in the hope  //
// that it will be useful, but WITHOUT ANY WARRANTY; without even the   //
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR      //
// PURPOSE.  See the GNU General Public License for more details.       //
//                                                                      //
// You should have received a copy of the GNU General Public License    //
// along with the iLab Neuromorphic Vision C++ Toolkit; if not, write   //
// to the Free Software Foundation, Inc., 59 Temple Place, Suite 330,   //
// Boston, MA 02111-1307 USA.                                           //
// //////////////////////////////////////////////////////////////////// //
//written by Nader Noori, March 2008

#include "psycho-skin-mapgenerator.h"
#include <vector>
#include <iostream>
#include <string>
#include "psycho-skin-mapgenerator.h"
#include <stdio.h>
#include <stdlib.h>
#include <sstream>
#include <time.h>

using namespace std ;

Matrix::Matrix(int row , int col , int** m)
{
        mems =         *m ;
        r = row ;
        c = col ;
}

// ######################################################################
Matrix::~Matrix()

{
        delete[] mems;
}
// ######################################################################
int Matrix::getNumOfRows()
{
        return r ;
}
// ######################################################################
int Matrix::getNumOfColumns()
{
        return c ;
}
// ######################################################################
int Matrix::get(int row , int col)
{
        int index = row*c + col ;
        return *(mems + index);
}
// ######################################################################
void Matrix::set(int row , int col , int val)
{
        int index = row*c+col ;
        *(mems + index) = val ;
}
// ######################################################################
void Matrix::setNumOfRows(int nr)
{
        r = nr ;
}
// ######################################################################
void Matrix::setNumOfColumns(int nc)
{
        c = nc ;
}
// ######################################################################
string Matrix::toFormattedString()
{
        string s("") ;
        for (int i = 0 ; i < r ; i++){
                s += "<" ;
                for(int j = 0 ; j < c ; j++){
                        int index = i*c + j;
                        int m = *(mems + index);
                        s+=stringify(m)+" " ;
                }
                s += ">" ;
        }
        return s ;
}
// ######################################################################
string Matrix::toString()
{
        string s("") ;
        for (int i = 0 ; i < r ; i++){
                for(int j = 0 ; j < c ; j++){
                        int index = i*c + j;
                        int m = *(mems + index);
                        s+=stringify(m)+" " ;
                }
        }
        return s ;
}
// ######################################################################
string Matrix::stringify(int i)
{
        ostringstream o ;
        o << i ;
        return o.str();
}

// ######################################################################
Matrix* getARandomMap(int row , int col , int channel)
{

        int* a = new int[row*col] ;
        srand ( time(NULL) );
        for (int i = 0 ; i < row*col ; i++){
                a[i] = rand()%channel +1 ;
        }
        Matrix* matrix= new Matrix(row,col,&a) ;
        //delete[] a ;
        return matrix ;
}
// ######################################################################
Matrix* getAnEmptyMap(int row , int col){
        int* a = new int[row*col] ;
        for (int i = 0 ; i < row*col ; i++){
                a[i] = 0 ;
        }
        Matrix* matrix= new Matrix(row,col,&a) ;
        //delete[] a ;
        return matrix ;
}
// ######################################################################

Matrix* getASeededMap(int row , int col , int numOfChannels , Matrix& p , int ch)
{

        srand ( time(NULL) );
        Matrix* m = getAnEmptyMap(row,col) ;
        int ix = rand()%(row - p.getNumOfRows()) ;
        int iy = rand()%(col - p.getNumOfColumns()) ;

        for(int x = 0 ; x < p.getNumOfRows() ; x++){
                for(int y = 0 ; y < p.getNumOfColumns() ; y++){
                        int f = p.get(x,y);
                        if( f != 0 ){
                                m->set(ix+x,iy+y,ch) ;
                        } else {
                                int temp = rand()%numOfChannels+1;
                                while( temp == ch){
                                        temp = rand()%numOfChannels+1;
                                }
                                m->set(ix+x,iy+y,temp) ;
                        }

                }
        }

        for(int i = 0; i < row ; i++){
                for(int j = 0 ; j < col ; j++){
                        if (m->get(i,j) == 0) m->set(i,j,rand()%numOfChannels +1) ;
                }
        }

        return m ;
}

// ######################################################################
Matrix* getTheChannelMap(Matrix& map , int ch)
{
        Matrix *e = getAnEmptyMap(map.getNumOfRows(),map.getNumOfColumns()) ;
        for(int i  = 0 ; i < map.getNumOfRows() ; i ++){
                for (int j = 0 ;  j< map.getNumOfColumns() ; j++){
                        if(map.get(i,j)==ch) e->set(i,j,1) ;
                }
        }
        return e ;
}
// ######################################################################
int getNumberOfMatches(Matrix& ormap , Matrix& pattern , int channel)
{
        int mapRows = ormap.getNumOfRows();
        int mapCols = ormap.getNumOfColumns();
        int patternRows = pattern.getNumOfRows();
        int patternCols = pattern.getNumOfColumns() ;
        Matrix* map = getTheChannelMap(ormap , channel);
        int result = 0 ;
        for(int i = 0 ; i < mapRows ; i++ ){
                for(int j = 0 ;  j < mapCols ; j++){

                        bool matchFlag = true ;
                        if(i+ patternRows <= mapRows && j+ patternCols <= mapCols){
                                for(int i1 = 0 ; i1 < patternRows ; i1++){
                                        for (int j1 = 0 ; j1 < patternCols ; j1++){
                                                if(pattern.get(i1,j1)> map->get(i+i1,j+j1)) {
                                                        matchFlag = false ;
                                                        break;
                                                }
                                                if(!matchFlag) break;
                                        }
                                        if(!matchFlag) break;
                                }

                        }else{

                                matchFlag = false ;
                        }

                        if(matchFlag){
                                result++ ;
                        }

                }
        }

        delete map ;
        return result ;
}
// ######################################################################
Matrix* getMatchMatrix(Matrix& ormap , Matrix& pattern , int channel){
        int mapRows = ormap.getNumOfRows();
        int mapCols = ormap.getNumOfColumns();
        int patternRows = pattern.getNumOfRows();
        int patternCols = pattern.getNumOfColumns() ;
        Matrix* result = getAnEmptyMap(mapRows,mapCols) ;
        Matrix* map = getTheChannelMap(ormap , channel);
        for(int i = 0 ; i < mapRows ; i++ ){
                for(int j = 0 ;  j < mapCols ; j++){

                        bool matchFlag = true ;
                        if(i+ patternRows <= mapRows && j+ patternCols <= mapCols){
                                for(int i1 = 0 ; i1 < patternRows ; i1++){
                                        for (int j1 = 0 ; j1 < patternCols ; j1++){

                                                if(pattern.get(i1,j1)> map->get(i+i1,j+j1) ){
                                                        matchFlag = false ;
                                                        break;
                                                }
                                                if(!matchFlag) break ;
                                        }
                                        if(!matchFlag) break ;
                                }

                        }else{

                                matchFlag = false ;
                        }

                        if(matchFlag){
                                result->set(i,j,1) ;
                        }else {result->set(i,j,0) ;}

                }
        }
        delete map ;
        return result;

}
// ######################################################################
Matrix* getPattenByString(std::string patStr){

        int rn = 1 ;
        int cn = 1 ;
        for(int i = 0 ; i< (int)patStr.size() ; i++) if( patStr[i]==char('_')) rn++ ;
        if(rn == 1)
        {
                cn = patStr.size();
        }else
        {
                string::size_type position = patStr.find("_");
                cn = int(position) ;
        }
        Matrix* t = getAnEmptyMap(rn,cn);
        for(int i = 0 ; i < rn ; i++)
        {
                for (int j = 0 ; j < cn ; j++)
                {
                        if(patStr.at( i*(cn+1) + j )=='1') t->set(i,j,1) ;
                }

        }

        return t ;
}

// ######################################################################

Matrix* getMapByString(std::string mapStr){
        int rn = 1 ;
        int cn = 1 ;
        for(int i = 0 ; i< (int)mapStr.size() ; i++) if( mapStr[i]==char('_')) rn++ ;
        if(rn == 1)
        {
                cn = mapStr.size();
        }else
        {
                string::size_type position = mapStr.find("_");
                cn = int(position) ;
        }
        Matrix* t = getAnEmptyMap(rn,cn);
        for(int i = 0 ; i < rn ; i++)
        {
                for (int j = 0 ; j < cn ; j++)
                {
                        char rf[2] ;
                        strcpy(rf,mapStr.substr(i*(cn+1) + j ,1 ).c_str());
                        t->set(i,j,atoi(rf)) ;
                }

        }

        return t ;
}

// ######################################################################

Matrix* getTranspose(Matrix& m , bool replace)
{
        int mRs = m.getNumOfRows() ;
        int mCs = m.getNumOfColumns() ;
        Matrix* nM = getAnEmptyMap(mCs,mRs);
        for(int i = 0 ; i < mRs ; i++){
                for(int j = 0 ; j < mCs ; j++){
                        nM->set(j,i,m.get(i,j)) ;
                }
        }
        if(replace) {
                m.setNumOfRows(mCs);
                m.setNumOfColumns(mRs) ;
                for(int i = 0 ; i < nM->getNumOfRows() ; i++){
                        for(int j = 0 ; j < nM->getNumOfColumns() ; j++){
                                m.set(i,j,nM->get(i,j));

                        }
                }
                nM->Matrix::~Matrix() ;
                return &m ;
        }
        return nM ;
}
// ######################################################################
Matrix* getHorizontalReverse(Matrix& m , bool replace)
{
        int mRs = m.getNumOfRows() ;
        int mCs = m.getNumOfColumns() ;
        Matrix* nM = getAnEmptyMap(mRs,mCs);
        for(int i = 0 ; i < mRs ; i++){
                for(int j = 0 ; j < mCs ; j++){

                        nM->set(i,j,m.get(mRs-i-1,j)) ;
                }
        }

        if(replace) {
                for(int i = 0 ; i < mRs ; i++){
                        for(int j = 0 ; j < mCs ; j++){
                                m.set(i,j,nM->get(i,j));

                        }
                }
                nM->Matrix::~Matrix() ;
                return &m ;
        }
        return nM ;
}
// ######################################################################
Matrix* getVerticalReverse(Matrix& m , bool  replace)
{
        int mRs = m.getNumOfRows() ;
        int mCs = m.getNumOfColumns() ;
        Matrix* nM = getAnEmptyMap(mRs,mCs);
        for(int i = 0 ; i < mRs ; i++){
                for(int j = 0 ; j < mCs ; j++){

                        nM->set(i,j,m.get(i,mCs - j -1)) ;
                }
        }
        if(replace) {
                for(int i = 0 ; i < mRs ; i++){
                        for(int j = 0 ; j < mCs ; j++){
                                m.set(i,j,nM->get(i,j));

                        }
                }
                nM->Matrix::~Matrix() ;
                return &m ;
        }
        return nM ;
}
// ######################################################################
/*
THis function is used for purification of the maps, this is the way it works: it
sees which row or column has the most number of non-zero elements alignment
then replicates that row or columns and returns it.
*/
Matrix* getDominantLinedupFeaturePattern(Matrix& p){

        int* colSum = new int[p.getNumOfColumns()] ;
        int* rowSum = new int[p.getNumOfRows()] ;
        int colSumMax = -1 ;
        int rowSumMax = -1 ;
        int mRow = 0 ;
        int mCol = 0 ;

        for( int r = 0 ; r < p.getNumOfRows(); r++ ){
                rowSum[r] = 0;
                for( int c = 0 ; c < p.getNumOfColumns() ; c++){
                        rowSum[r] += p.get(r,c);
                }
                rowSumMax = max(rowSumMax,rowSum[r]);
                if(rowSumMax == rowSum[r]) mRow = r ;
        }

        for( int c = 0 ; c < p.getNumOfColumns() ;c++ ){
                colSum[c] = 0 ;
                for( int r = 0 ; r < p.getNumOfRows() ; r++ ){
                        colSum[c] += p.get(r,c) ;
                }
                colSumMax = max(colSumMax,colSum[c]) ;
                if(colSumMax == colSum[c]) mCol = c ;
        }

        if(colSumMax > rowSumMax){
                Matrix* d = getAnEmptyMap(colSumMax , 1) ;
                for( int r = 0 ; r < p.getNumOfRows() ; r++){
                        d->set(r,0,1);
                }
                delete[] colSum ;
                delete[] rowSum ;
                return d ;

        }else{
                Matrix* d = getAnEmptyMap(1,rowSumMax);
                for(int c = 0 ; c < p.getNumOfColumns() ; c++ ){
                        d->set(0,c,1);
                }
                delete[] colSum ;
                delete[] rowSum ;
                return d ;
        }

}

// ######################################################################
bool areTheyEqual(Matrix& a , Matrix& b){
        int aR = a.getNumOfRows() ;
        int aC = a.getNumOfColumns() ;
        int bR = b.getNumOfRows() ;
        int bC = b.getNumOfColumns() ;
        if(aC != bC || aR!=bR ) return false ;
        for(int i = 0 ; i < aR ; i++)
        {
                for(int j = 0 ; j < aC ; j++)
                {
                        if(a.get(i,j)!=b.get(i,j)) return false ;
                }
        }
        return true ;
}
// ######################################################################
vector<Matrix*> *getAllVariations(Matrix& p)
{

        vector<Matrix*>* v = new vector<Matrix*>();
        v->push_back(&p) ;

        Matrix* p1 = getTranspose(p) ;
        if(!isInTheBag(*p1 ,*v)){
                v->push_back(p1) ;
        }else{
                p1->Matrix::~Matrix();
        }

        Matrix* p2 = getHorizontalReverse(p) ;
        if(!isInTheBag(*p2 ,*v)){
                v->push_back(p2) ;
        }else{
                p2->Matrix::~Matrix();
        }

        Matrix* p3 = getVerticalReverse(p) ;
        if(!isInTheBag(*p3 ,*v)){
                v->push_back(p3) ;
        }else{
                p3->Matrix::~Matrix();
        }

        Matrix* p4 = getVerticalReverse(*(getTranspose(p)),true) ;
        if(!isInTheBag(*p4 ,*v)){
                v->push_back(p4) ;
        }else{
                p4->Matrix::~Matrix();
        }

        Matrix* p5 = getHorizontalReverse(*(getTranspose(p)),true) ;
        if(!isInTheBag(*p5 ,*v)){
                v->push_back(p5) ;
        }else{
                p5->Matrix::~Matrix();
        }

        Matrix* p6 = getVerticalReverse(*(getHorizontalReverse(p)),true) ;
        if(!isInTheBag(*p6 ,*v)){
                v->push_back(p6) ;
        }else{
                p6->Matrix::~Matrix();
        }

        Matrix* p7 = getHorizontalReverse(*(getVerticalReverse(p)),true) ;
        if(!isInTheBag(*p7 ,*v)){
                v->push_back(p7) ;
        }else{
                p7->Matrix::~Matrix();
        }


        Matrix* p8 = getVerticalReverse(*getHorizontalReverse(*(getTranspose(p)),true),true) ;
        if(!isInTheBag(*p8 ,*v)){
                v->push_back(p8) ;
        }else{
                p8->Matrix::~Matrix();
        }


        return v ;
}
// ######################################################################
bool isInTheBag(Matrix& p , vector<Matrix*>& v)
{
        bool flag = false ;
        for ( vector<Matrix*>::iterator it=v.begin() ; it < v.end(); it++ )
                if(areTheyEqual(*(*it),p))
        {
                flag = true ;
                break;
        }

        return flag ;
}
// ######################################################################
int getNumberOfMatchesAgainstAllVariationsForAllChannels(Matrix& map ,  Matrix& pattern , int numOfChannels)
{
        int n= 0 ;
        vector<Matrix*> * v = getAllVariations(pattern) ;
        for(int c = 1 ;  c <= numOfChannels ; c++ ){
                for(vector<Matrix*>::iterator it=v->begin() ; it < v->end(); it++){
                        n += getNumberOfMatches(map ,*(*it) , c);
                }

        }
        return n;
}
// ######################################################################
vector<Matrix*> *getMapsWithExactPattenAndExactChannel(int rs , int cs , int numOfch , int ch , Matrix& pattern , int numOfMatches , int n )
{
        vector<Matrix*>* v = new vector<Matrix*>();
        //Matrix* p = getPattenByString(pS) ;
        //cout<< p->toFormattedString() ;

        while( (int)(v->size())!=n )
        {
                Matrix *m ;
                if( numOfMatches != 0 ){
                        m = getASeededMap(rs , cs , numOfch , pattern , ch) ;
                }else{
                        m = getARandomMap(rs , cs , numOfch ) ;

                }

                if(getNumberOfMatches(*m , pattern , ch) == numOfMatches && !isInTheBag(*m,*v))
                {
                        v->push_back(m) ;
                }else{
                        delete m;

                }

        }

        return v ;

}

// ######################################################################
vector<Matrix*> *getMapsWithExactPattenAndExactChannelWithExclusionList(int rs , int cs , int numOfch , int ch , Matrix& pattern ,vector<string>& exList, int numOfMatches , int n )
{
        vector<Matrix*>* v = new vector<Matrix*>();
        vector<Matrix*>* exvector = new vector<Matrix*>() ;
        vector<int>::size_type sz = exList.size();
        for( uint i = 0 ; i < sz ; i++){
                vector<Matrix*>* vars = getAllVariations(*getPattenByString(exList[i]));
                vector<int>::size_type vs = vars->size() ;
                for( uint j = 0 ; j < vs ; j++ ){
                        exvector->push_back((*vars)[j]) ;
                }
        }

        while( (int)(v->size())!=n )
        {
                Matrix *m = getASeededMap(rs , cs , numOfch , pattern , ch) ;
                if(getNumberOfMatches(*m , pattern , ch) == numOfMatches && !isInTheBag(*m,*v))
                {
                        bool exFlag = true ;
                        for( uint i = 0 ; i < exvector->size() ; i++){
                                for( int j = 0 ; j < numOfch  ; j++){
                                        if(getNumberOfMatches(*m,*((*exvector)[i]),j+1)!= 0 )
                                                exFlag = false ;
                                }
                        }
                        if(exFlag) v->push_back(m) ;
                }

        }
        exvector->~vector<Matrix*>() ;
        return v ;

}

// #########################################################################
vector<Matrix*> *getPureMapsWithExactPatternAndExactChannel(int rs , int cs , int numOfch , int ch , Matrix& pattern , int numOfMatches , int n){

        vector<Matrix*>* v = new vector<Matrix*>();
        Matrix *m ;
        Matrix* dominantPattern ;
        Matrix* tm = 0;
        while( (int)(v->size())!=n )
        {

                if( numOfMatches != 0 ){
                        tm = getASeededMap(rs , cs , numOfch , pattern , ch) ;
                }else{
                        tm = getARandomMap(rs , cs , numOfch ) ;
                }
                dominantPattern = getDominantLinedupFeaturePattern(pattern) ;
                if(getNumberOfMatches(*tm , pattern , ch) == numOfMatches && getNumberOfMatches(*tm,*dominantPattern,ch)==numOfMatches && !isInTheBag(*tm,*v))
                {
                        m = getAnEmptyMap(rs,cs) ;
                        for(int r = 0 ; r < rs ; r++){
                                for(int  c = 0 ; c <cs ; c++){
                                        m->set(r,c,tm->get(r,c));
                                }
                        }
                        v->push_back(m) ;
                }
                delete dominantPattern ;
                delete tm;
        }
        return v ;

}

// ######################################################################
// Matrix *getAMapWithExactPatternAndExactChannel(int rs , int cs , int numOfch, int ch , Matrix& pattern , int numOfMatches= 1 , std::vector<std::string> exList){
//         bool flag = true
//         Matrix* m = getASeededMap(rs,cs, numOfch, pattern , ch );
//         do{
//                 if(getNumberOfMatches(*m,pattern,ch) != 1 ) flag = false;
//
//         }while( !flag );
//
//         return m;
// }

//int main(){
//        cout<<"Hi there!"<<endl ;
//        string pS = "101_010";
//        vector<Matrix*>* v = getMapsWithExactPattenAndExactChannel(9 , 9 , 6 , 3 , pS , 1 , 1 );
//        for ( vector<Matrix*>::iterator it=v->begin() ; it < v->end(); it++ )
//                cout<< (*it)->toFormattedString()<<endl ;
        /*
        Matrix *t = getARandomMap(9,9,6) ;
        cout<<t->toFormattedString()<<endl ;
        string s = (*t).toString();
        string p1 = string("110");
        string p2 = string("exit");
        while (p1.compare(p2)!=0){
                cout<<"enter the patten string"<<endl;
                cin >> p1 ;
                Matrix *ps1 = getPattenByString(p1) ;
                vector<Matrix*>* v = getAllVariations(*ps1);
                for ( vector<Matrix*>::iterator it=v->begin() ; it < v->end(); it++ )
                cout<< (*it)->toFormattedString()<<endl ;
                cout<<"matches for all channels : " << getNumberOfMatchesAgainstAllVariationsForAllChannels(*t,*ps1)<<endl;
}
        */

        //cout<< ps1->toFormattedString()<<endl ;
        //getTranspose(*ps1,false) ;
        //cout<< ps1->toFormattedString()<<endl ;
        //getVerticalReverse(*ps1 ,true) ;
        //cout<< ps1->toFormattedString()<<endl ;
/*        getVerticalReverse
        //Matrix *ps2 = getPattenByString(p2) ;
        Matrix *ps2 = getHorizontalReverse(*ps1) ;
        cout<< ps1->toFormattedString()<<endl;
        cout<< ps2->toFormattedString()<<endl;
        //cout << getNumberOfMatches(*t , *ps1 , 9)<< endl;
        int matches1 = 0 ;
        int matches2 = 0 ;
        for (int i = 1 ; i <=9 ; i++ ) matches1 += getNumberOfMatches(*t , *ps1 , i);
        for (int i = 1 ; i <=9 ; i++ ) matches2 += getNumberOfMatches(*t , *ps2 , i);
        cout << t->toFormattedString()<<endl ;
        cout<< matches1 << endl ;
        cout<< matches2 << endl ;
        Matrix* match1 = getMatchMatrix(*t,*ps1,1);
        Matrix* match2 = getMatchMatrix(*t,*ps2,1);
        cout<< match1->toFormattedString()<< endl;
        cout<< match2->toFormattedString()<< endl;
*/
//        return 0 ;
//}

