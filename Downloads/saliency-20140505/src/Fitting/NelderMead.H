//Nelder-Mead minimization algorithm
//////////////////////////////////////////////////////////////////////////
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2001 by the //
// University of Southern California (USC) and the iLab at USC.         //
// See http://iLab.usc.edu for information about this project.          //
//////////////////////////////////////////////////////////////////////////
// Major portions of the iLab Neuromorphic Vision Toolkit are protected //
// under the U.S. patent ``Computation of Intrinsic Perceptual Saliency //
// in Visual Environments, and Applications'' by Christof Koch and      //
// Laurent Itti, California Institute of Technology, 2001 (patent       //
// pending; application number 09/912,225 filed July 23, 2001; see      //
// http://pair.uspto.gov/cgi-bin/final/home.pl for current status).     //
//////////////////////////////////////////////////////////////////////////
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
//////////////////////////////////////////////////////////////////////////

//primary maintainer: David J. Berg

#ifndef FIT_SIMPLEX_H_DEFINED
#define FIT_SIMPLEX_H_DEFINED

#ifdef INVT_USE_CPP11//we need c++ 0X features for this to work

#include <vector>
#include <functional>

// ######################################################################
// Simpliex
// Nelder-Mead minimization algorithm
// ######################################################################
class NelderMead  
{
  public:
    enum FinishCode {NONE, MAXITERATIONS, ERRORTOLERANCE, DELTAERRORTOLERANCE};
    typedef std::pair<std::vector<double>, double> VertexPair;

    struct Result
    {
        VertexPair params;
        FinishCode code;
        uint iterations;
    };

    //!construct a Nelder-Mead algorithm
    /*
      The first parameters is the starting point.
      The second parameter controls the size of the initial simplex
      The third parameter is your function object which takes a N-D point and returns the error. 
      The fourth parameter controls how close we need to be to our target
      The fifth parameter is the maximum number of iterations
      The sixth parameter toggles multithreading in iteration 0 where all simplex points need to be evaluated. Be
      careful, as your functor needs to be copyable, and actually make a deep copy of any state data.
      The seventh parameter toggles display messages
     */    
	NelderMead(std::vector<double> const & start, 
               std::vector<double> const & min_param_val, std::vector<double> const & max_param_val,
               double const & simplex_size, 
               std::function<double const (std::vector<double> const &)> func, 
               double const & error_tol, uint const maxiter, double const & errorchange_tol, 
               bool const multithread = false, bool const display = false);
    
    //!minimize the function of interest and return the parameters and the error as a VertexPair
    Result minimize();

    //!find the amount of shift in each parameter to cause a delta_error change in error
    void findConstraints(double const & delta_error, double const & delta_param, std::vector<double> & min, std::vector<double> & max);

    //convert a finish code to string
    static std::string finishCodeToString(FinishCode const code);
    
  private:

    double const itsErrorTolerance;
    double const itsDeltaErrorTolerance;
    uint const itsMaxIter;
    bool const itsMultiThread;
    bool const itsDisplay;
    std::function<double const (std::vector<double> const &)> itsFunc; 
    std::vector<VertexPair> itsSimplex;
    std::vector<double> itsRange;
};

#endif 
#endif
