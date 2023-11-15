/*!@file AppPsycho/psycho-skin-mapgenerator.h Psychophysics supports psycho-skin-bsindexing.h and psycho-aindexing.h */

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
//
// Primary maintainer for this file: Laurent Itti <itti@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/psycho-skin-mapgenerator.h $
// $Id: psycho-skin-mapgenerator.h 10794 2009-02-08 06:21:09Z itti $
//
//written by Nader Noori March 2008

#ifndef _PSYCHO_SKIN_MAPGENERATOE_H_
#define _PSYCHO_SKIN_MAPGENERATOE_H_
#include <vector>
#include <string>



class Matrix
{
        public:
                Matrix(int row , int col , int** mems);
        ~Matrix() ;
        int getNumOfRows();
        int getNumOfColumns();
        int get(int row , int col );
        void set(int row , int col , int val);
        void setNumOfColumns(int nc) ;
        void setNumOfRows(int nr);
        std::string toFormattedString() ;
        std::string  toString();
        private:
        std::vector<int>* membs ;
        int* mems;
        int r ;
        int c ;
        std::string stringify(int i) ;
} ;

Matrix* getARandomMap(int row , int col , int channel);

Matrix* getAnEmptyMap(int row , int col );

Matrix* getASeededMap(int row , int col , int numOfChannels , Matrix& p , int ch);

Matrix* getTheChannelMap(Matrix& map , int ch);

Matrix* getPattenByString(std::string patStr);

Matrix* getMapByString(std::string mapStr) ;

int getNumberOfMatches(Matrix& map , Matrix& pattern , int channel = 1);

int getNumberOfMatchesAgainstAllVariationsForAllChannels(Matrix& map ,  Matrix& pattern , int numOfChannels=6);

Matrix* getMatchMatrix(Matrix& map , Matrix& pattern , int channel);

Matrix* getTranspose(Matrix& m , bool replace = false ) ;

Matrix* getHorizontalReverse(Matrix& m , bool replace = false) ;

Matrix* getVerticalReverse(Matrix& m , bool replace = false) ;

std::vector<Matrix*> *getAllVariations(Matrix& p);

bool isInTheBag(Matrix& p , std::vector<Matrix*>&);

bool areTheyEqual(Matrix& a , Matrix& b);

std::vector<Matrix*> *getMapsWithExactPattenAndExactChannel(int rs , int cs , int numOfch , int ch , Matrix& p , int numOfMatches=1 , int n=1) ;

std::vector<Matrix*> *getPureMapsWithExactPatternAndExactChannel(int rs , int cs , int numOfch , int ch , Matrix& p , int numOfMatches=1 , int n=1) ;

std::vector<Matrix*> *getMapsWithExactPattenAndExactChannelWithExclusionList(int rs , int cs , int numOfch , int ch , Matrix& pattern ,std::vector<std::string>& exList, int numOfMatches , int n );

#endif

