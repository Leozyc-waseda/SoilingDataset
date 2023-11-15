#include "psycho-skin-accuracyEstimator.h"
#include <vector>
#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <sstream>
#include <time.h>
#include <math.h>

using namespace std ;

IntString::IntString( int* m , int s)
{
        mems = new int[s] ;
        for(int i = 0 ; i < s ; i++ ) mems[i] = *(m + i) ;
        theSize = s ;
        degree = -1 ;
}

// ######################################################################
IntString::~IntString()

{
        delete[] mems;
}

// ######################################################################

inline int IntString::getDegree(){
        if (degree == -1 ){
                degree = 0 ;
                for(int i = 0 ;  i < theSize ; i++)
                        degree += *(mems + i) ;
        }
        return degree ;
}
// ######################################################################
inline int IntString::size(){
        return theSize ;
}
// ######################################################################
inline int IntString::getElementAt(int i) {

        return *(mems + i) ;
}
// ######################################################################

inline void IntString::set(int pos , int val){
        *(mems + pos) = val ;
}
// ######################################################################
std::string  IntString::toString(){
        string s("") ;
        for(int i = 0 ; i < theSize ; i++) {
                int m = *(mems + i) ;
                s += stringify(m) ;
        }
        return s ;
}
// ######################################################################
string IntString::stringify(int i)
{
        ostringstream o ;
        o << i ;
        return o.str();
}
// ######################################################################

IntString* getTPS(IntString& testIntStr , IntString& repIntStr){
        int s  =testIntStr.size() ;
        int* a = new int[s] ;
        IntString* pps = new IntString(a , s);
        for(int i = 0 ; i < s ; i++){
                if(testIntStr.getElementAt(i)==1 && repIntStr.getElementAt(i) == 1)
                {
                        pps->set(i,1);
                }else{
                        pps->set(i,0) ;
                }
        }
        return pps ;
}
// ######################################################################
inline float getAccuracyOfPPS(int d , int p , int n) {

        return ((float)(n+d))/((float)(n+p)) ;
}
// ######################################################################
map<float,float>* getProbabilityDistribution(IntString& tstIntStr , IntString& repIntStr){
        map<float , float>* m = new map<float , float>();
        int p = tstIntStr.getDegree();
        int n = tstIntStr.size() - p ;
        IntString* tps = getTPS(tstIntStr,repIntStr) ;
        int d = tps->getDegree();
        int p_repStr = repIntStr.getDegree() ;
        int n_repStr = repIntStr.size() - p_repStr ;

        for(int dp = 0 ; dp <= d ; dp++){
                int powerOfPositiveProbability = p_repStr - dp ;
                float prbOfPositive = ((float)(p-dp))/((float)(p+n-dp)) ;
                float prbOfNegative = ((float)(n))/((float)(p+n-dp)) ;
                //float acc = ((float)(n+dp))/((float)(p+n)) ;
                float acc = ((float)(dp))/((float)(p)) ;
                int numOfPossibledps = getBinomialCoef(d,dp) ;
                float prb = 1.0f ;
                if(prbOfPositive != 0) {
                        prb = numOfPossibledps * pow(prbOfPositive,powerOfPositiveProbability)*pow(prbOfNegative,n_repStr) ;
                }else{
                        if(powerOfPositiveProbability!=0){prb = 0.0f;}
                }
                (*m)[acc] = prb;
        }
        normalizeThePDF(*m) ;
        return m ;
}

map<float,float>* getCummulativeProbabilityDistribution(IntString& tstIntStr , IntString& repIntStr){
        map<float , float>* pdf = getProbabilityDistribution(tstIntStr,repIntStr) ;
        map<float , float>* cdf = new map<float , float>();

        for (map<float,float>::iterator fit = pdf->begin(); fit != pdf->end(); ++fit) {
                float cp = 0.0f ;
                for (map<float,float>::iterator sit = pdf->begin(); sit != pdf->end(); ++sit) {
                        if ((sit->first) <= (fit->first)) cp+= sit->second ;
                }
                (*cdf)[fit->first] = cp ;
        }
        pdf->map::~map() ;
        return cdf ;
}

void normalizeThePDF(map<float,float>& pdf){

        float sum = 0 ;
        for (map<float,float>::iterator it = pdf.begin(); it != pdf.end(); ++it) {
                sum += pdf[it->first];
            }

        for (map<float,float>::iterator it = pdf.begin(); it != pdf.end(); ++it) {
                pdf[it->first] = pdf[it->first]/sum;
            }
}

inline int getBinomialCoef(int n , int r) {
        return factorial(n)/(factorial(r)*factorial(n-r));
}

inline int factorial(int n) {
        if(n==0 || n==1){return 1 ;}else{return n*factorial(n-1);}
}

float getExpectedValue(map<float , float>& dist){
        float avrg = 0 ;
        for(map<float , float>::iterator it = dist.begin() ; it != dist.end() ; ++it){
                avrg += (it->first) * (it->second);
        }
        return avrg ;
}

float getSTD(std::map<float , float>& dist){
        float avrg = getExpectedValue(dist) ;
        float std = 0 ;
        for(map<float , float>::iterator it = dist.begin() ; it != dist.end() ; ++it){
                std += pow(it->first-avrg,2.0f) * (it->second);
        }

        return sqrt(std) ;
}
// ######################################################################
extern "C" int main( int argc, char* args[] ){
        cout<<"Hi it\'s working!"<<endl ;
        int* a = new int [24] ;
        int* b = new int[24] ;
        for(int i = 0 ; i < 24 ; i++){
                if (i<7){a[i]=1;}else{a[i]=0;}
                if (i<6){b[i]=1;}else{b[i]=0;}
        }
        b[11]= 1 ;
        b[12] = 1 ;
        b[13] = 1;
        b[14] = 1 ;
        b[15]= 0 ;
        IntString* ts = new IntString(a,24) ;
        IntString* is = new IntString(b,24) ;
        IntString* tps = getTPS(*ts,*is) ;

        cout<<tps->toString()<< " size: "<< tps->size()<<"  degree:"<< tps->getDegree()<<endl ;

        map<float,float>* pdf = getProbabilityDistribution(*ts,*is);
        for(map<float,float>::iterator it= pdf->begin() ; it!= pdf->end(); ++it){
                cout<<"a: " <<it->first << "  p = " << it->second <<endl;
        }

        map<float,float>* cdf = getCummulativeProbabilityDistribution(*ts,*is) ;
        for(map<float,float>::iterator it= cdf->begin() ; it!= cdf->end(); ++it){
                cout<<"a: " <<it->first << "  p = " << it->second <<endl;
        }
        cout << "expected accuracy = "<<getExpectedValue(*pdf)<< " with std = "<< getSTD(*pdf)<<endl;
        return 0 ;
}

