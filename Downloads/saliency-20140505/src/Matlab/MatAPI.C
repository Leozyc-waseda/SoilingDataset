/*!
   @file Matlab/MatAPI.C some basic wrappers for the Matlab API
*/

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
// Primary maintainer for this file: Manu Viswanathan <mviswana at usc dot edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Matlab/MatAPI.C $
// $Id: MatAPI.C 10794 2009-02-08 06:21:09Z itti $
//

//------------------------------ HEADERS --------------------------------

// Matlab API header
#include "Matlab/MatAPI.H"

// INVT headers

// Standard C++ headers
#include <sstream>
#include <algorithm>
#include <iterator>

//----------------------- NAMESPACE DEFINITION --------------------------

// Encapsulate this API within its own namespace
namespace Matlab {

//--------------------------- MATLAB ARRAYS -----------------------------

// Matlab stores matrices internally in flat 1-dimensional arrays using
// column major ordering. Thus, to copy an entire row into an mxArray, we
// start off at the appropriate offset from the start of the internal
// data array and increment by the number of rows.
//
// Here is an illustrative example: let us say we want to append rows to
// a 4x3 matrix. The first row will be in array indices 0, 4, 8; the
// next row will occupy indices 1, 5, 9; and so on.
//
// The Matrix::mx_data_ptr inner class implements an iterator (for use
// with STL algorithms) that starts off pointing at the first array
// element of the "current" row in the mxArray, i.e., the row number the
// push_back() function will fill, and then increments this pointer by
// the number of rows in the matrix to get at the array element that will
// store the next value of the matrix's row.
Matrix::mx_data_ptr::mx_data_ptr(const Matrix& M)
   : p(mxGetPr(M.mat_array) + M.current_row), n(M.num_rows)
{}

// Create an mxn matrix of real numbers with all elements initialized to
// zero.
Matrix::Matrix(int m, int n)
   : num_rows(m), num_cols(n),
     mat_array(mxCreateDoubleMatrix(m, n, mxREAL)),
     current_row(0),
     ref_count(1)
{
   if (! mat_array)
      throw std::runtime_error("unable to create matrix") ;
}

// Wrap a preexisting mxArray pointer into a Matrix object. This
// pointer must actually reference an already initialized mxArray (i.e.,
// not an empty matrix).
//
// WARNING: Once the pointer is handed over to an instance of this class,
// it should not be used directly thereafter (and especially not deleted
// or destroyed with the mxDestroyArray() function).
Matrix::Matrix(mxArray* A)
   : num_rows(mxGetM(A)), num_cols(mxGetN(A)),
     mat_array(A),
     current_row(num_rows), // cannot modify matrix!
     ref_count(1)
{}

// Object copy
//
// WARNING: Only meant for use with temporaries (i.e., function
// parameters and return values). Any use more sophisticated than this
// will mess up the ref-counting. See ref-counting notes in comment
// preceding class definition.
Matrix::Matrix(const Matrix& M)
   : num_rows(M.num_rows), num_cols(M.num_cols),
     mat_array(M.mat_array),
     current_row(M.current_row),
     ref_count(M.ref_count + 1)
{}

// Clean-up
Matrix::~Matrix()
{
   if (--ref_count == 0)
      mxDestroyArray(mat_array) ;
}

// Append a given row of numbers to the matrix
void Matrix::push_back(const std::vector<double>& row)
{
   if (row.size() != num_cols)
      throw std::runtime_error("cannot append row of different size") ;
   if (current_row >= num_rows)
      throw std::runtime_error("cannot append any more rows") ;

   std::copy(row.begin(), row.end(), mx_data_ptr(*this)) ;
   ++current_row ;
}

//--------------------------- MATLAB ENGINE -----------------------------

// Matlab engine initialization and setup for Netlab toolbox
Engine::Engine(const std::string& netlab_path)
   : mat_engine(engOpen("matlab -nosplash -nojvm"))
{
   if (! mat_engine)
      throw std::runtime_error("unable to start Matlab engine") ;

   exec("addpath '" + netlab_path + "'") ;
}

// Getting and sending matrices from/to Matlab engine
Matrix Engine::get(const std::string& var_name)
{
   mxArray* M = engGetVariable(mat_engine, var_name.c_str()) ;
   if (! M)
      throw std::runtime_error("unable to retrieve variable "
                               + var_name + " from Matlab engine") ;
   return M ;
}

void Engine::put(const std::string& var_name, const Matrix& M)
{
   if (engPutVariable(mat_engine, var_name.c_str(), M))
      throw std::runtime_error("unable to send variable " + var_name +
                               " to Matlab engine") ;
}

// Getting the Matlab engine to do our bidding
void Engine::exec(const std::string& cmd)
{
   if (engEvalString(mat_engine, cmd.c_str()))
      throw std::runtime_error("Matlab engine could not eval: " + cmd) ;
}

// Shutting down the Matlab engine
Engine::~Engine()
{
   engClose(mat_engine) ;
}

//---------------------------- MATLAB FILE ------------------------------

// Initialization
File::File(const std::string& file_name, const std::string& mode)
   : mat_file(matOpen(file_name.c_str(), mode.c_str()))
{
   if (! mat_file)
      throw std::runtime_error("unable to open " + file_name) ;
}

// Reading data from the .mat file
Matrix File::get(const std::string& var_name)
{
   mxArray* M = matGetVariable(mat_file, var_name.c_str()) ;
   if (! M)
      throw std::runtime_error("no variable named " + var_name) ;
   return M ;
}

// Writing data to the .mat file
void File::put(const std::string& var_name, const Matrix& matrix)
{
   matPutVariable(mat_file, var_name.c_str(), matrix) ;
}

// Clean-up
File::~File()
{
   matClose(mat_file) ;
}

// Output operator for writing textons to a File instance
File& operator<<(File& f, const FileData& d)
{
   f.put(d.first, d.second) ;
   return f ;
}

//-----------------------------------------------------------------------

} // end of namespace encapsulating this file

/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
