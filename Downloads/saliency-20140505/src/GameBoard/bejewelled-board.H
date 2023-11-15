/*!@file GameBoard/bejewelled-board.H classes and functions for generating and manipulating bejewelled gameboard*/
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
//written by Nader Noori, April 2008
#ifndef BEJEWELLED_GAMEBOARD_H_DEFINED
#define BEJEWELLED_GAMEBOARD_H_DEFINED

template <typename T>
class Matrix
{
        public:
        Matrix(int row , int col , T** mems);
        ~Matrix() ;
        int getNumOfRows();
        int getNumOfColumns();
        T* operator[](int row) ;
        void setNumOfColumns(int nc) ;
        void setNumOfRows(int nr);
        std::string toFormattedString() ;
        std::string  toString();
        private:
        T* mems;
        int r ;
        int c ;
        std::string stringify(T t) ;
} ;

class Cell
{
        public:
        Cell(const uint row, const uint col);
        uint r ;
        uint c ;
};

class Board
{
        public:
        Board(const uint row , const uint col , const uint nc);
        Board(Matrix<uint>& m) ;
        ~Board() ;
        Matrix<uint>* getCurrentState();
        uint getObjectAtCell(Cell& cell);
        private:
        Matrix<uint>* cstate ;
        uint numOfChannels ;
};

#endif


