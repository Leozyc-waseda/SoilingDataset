//
// C++ Interface: psycho-skin-accracyEstimator
//
// Description:
//
//
// Author: Nader Noori <nnoori@usc.edu>, (C) 2008
//
// Copyright: See COPYING file that comes with this distribution
//
//
#ifndef _PSYCHO_SKIN_ACCURACYESTIMATOR_H_
#define _PSYCHO_SKIN_ACCURACYESTIMATOR_H_
#include <map>
#include <string>


class IntString{

        public :
        IntString(int* mem , int s) ;
        ~IntString() ;
        int getDegree() ;
        int size() ;
        int getElementAt(int i) ;
        std::string  toString();
        void set(int pos , int val) ;


        private :
        std::string stringify(int i) ;
        int * mems ;
        int theSize ;
        int degree ;
} ;

IntString* getTPS(IntString& testIntStr , IntString& repIntStr);

std::map<float,float>* getProbabilityDistribution(IntString& tstIntStr , IntString& repIntStr) ;

std::map<float,float>* getCummulativeProbabilityDistribution(IntString& tstIntStr , IntString& repIntStr) ;

float getExpectedValue(std::map<float , float>& dist) ;

float getSTD(std::map<float , float>& dist) ;

float getAccuracyOfPPS(int d , int p , int n) ;

int getBinomialCoef(int n , int r) ;

void normalizeThePDF(std::map<float,float>& pdf);

int factorial(int n);

#endif

