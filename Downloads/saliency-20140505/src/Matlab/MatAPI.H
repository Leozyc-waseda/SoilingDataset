/*!
   @file Matlab/MatAPI.H some basic wrappers for the Matlab API
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Matlab/MatAPI.H $
// $Id: MatAPI.H 10794 2009-02-08 06:21:09Z itti $
//

#ifndef MATLAB_MATAPI_H_DEFINED
#define MATLAB_MATAPI_H_DEFINED

//------------------------------ HEADERS --------------------------------

// INVT headers
#include "Image/Image.H"

// Matlab headers
#ifdef MATLAB_LIBS_AVAILABLE // should be made available by configure system

#include <engine.h>
#include <mat.h>

#else // use dummies to NOP out Matlab types and functions

// Use dummies for various Matlab types and functions. Otherwise, builds
// will break!
//
// DEVNOTE: Although this module and others dependent on it will build
// okay with these dummies, runtime errors will prevent any dependent
// programs from executing smoothly. thus, it may be a good idea to check
// the MATLAB_LIBS_AVAILABLE preprocessor symbol in those modules as well
// and do something appropriate in case the Matlab headers and libraries
// are not available (e.g., have main() print a message stating that it
// needs this API rather than allow it to blithely proceed and then
// crash shortly thereafter with some inscrutable complaint about null
// pointers or some such thing).
#define mxREAL 0

typedef double mxArray ;
typedef double Engine ;
typedef double MATFile ;

inline mxArray* mxCreateDoubleMatrix(int, int, int) {return 0 ;}
inline void     mxDestroyArray(mxArray*) {}
inline double*  mxGetPr(mxArray*) {return 0;}
inline unsigned mxGetM(mxArray*)  {return 0;}
inline unsigned mxGetN(mxArray*)  {return 0;}

inline Engine*  engOpen(const char*) {return 0 ;}
inline mxArray* engGetVariable(Engine*, const char*) {return 0 ;}
inline int  engPutVariable(Engine*, const char*, const mxArray*) {return 1 ;}
inline int  engEvalString(Engine*, const char*) {return 1 ;}
inline void engClose(Engine*) {}

inline MATFile* matOpen(const char*, const char*) {return 0 ;}
inline mxArray* matGetVariable(MATFile*, const char*) {return 0 ;}
inline void matPutVariable(MATFile*, const char*, const mxArray*) {}
inline void matClose(MATFile*) {}

#endif // MATLAB_LIBS_AVAILABLE

// Standard C++ headers
#include <stdexcept>
#include <vector>
#include <iterator>
#include <utility>

//----------------------- NAMESPACE DEFINITION --------------------------

// Encapsulate this API within its own namespace
namespace Matlab {

//--------------------------- MATLAB ARRAYS -----------------------------

// Quick helper to encapsulate the necessary portions of the Matlab
// mxArray API.
//
// DEVNOTE: REFERENCE COUNTING HACK
//
// This class implements a VERY BAD reference counting mechanism. It is
// only meant to be used for taking care of copying of temporaries, e.g.,
// when passing parameters to functions or getting back return values.
// Taking advantage of this mechanism for anything more sophisticated can
// only lead to trouble... To help prevent some of the pathologies
// associated with this hacked up ref-counting, assignment and creation
// of MatArrays on the heap are disabled.
//
// We could (and probably should) have used nub::ref_counted over here.
// But it seemed a bit heavy-duty. But more importantly, the author was
// unable to quickly locate documentation with a tutorial nature that
// could have helped figure out how to use it in less than ten minutes.
// Thus this monstrosity. When time permits, this class should be
// revamped and made more generally useful (to ease interfacing with
// Matlab from inside of C++).
class Matrix {
   unsigned int num_rows, num_cols ;
   mxArray* mat_array;
   unsigned int current_row ;
   int ref_count ; // WARNING: MAJOR HACK! SEE COMMENT ABOVE.

   // Disable assignment and creation on the heap as they can be
   // hazardous to the hackish reference counting mechanism implemented
   // herein.
   Matrix& operator=(const Matrix&) ;
   void* operator new(size_t) ;

   // Quick-and-dirty output iterator for accessing mxArray in
   // column-major order (that's how Matlab arrays are stored).
   class mx_data_ptr {
      mutable double* p ; // current element being pointed to
      int n ; // increment value for p (equals num_rows in matrix)
   public :
      mx_data_ptr(const Matrix&) ;
      mx_data_ptr& operator++() throw() {
         p += n ; return *this ;
      }
      const mx_data_ptr& operator++() const throw() {
         p += n ; return *this ;
      }
      mx_data_ptr operator++(int) throw() { // post-increment
         mx_data_ptr prev(*this) ; ++*this ; return prev ;
      }
      const mx_data_ptr operator++(int) const throw() {
         mx_data_ptr prev(*this) ; ++*this ; return prev ;
      }
      double& operator*() throw() {return *p ;}
      const double& operator*() const throw() {return *p ;}

      // Need to define these types for STL compatibility
      typedef std::output_iterator_tag iterator_category ;
      typedef double value_type ;
      typedef int difference_type ;
      typedef double* pointer ;
      typedef double& reference ;
   } ; // end of inner class mx_data_ptr

public :
   Matrix(int m, int n) ;
   Matrix(mxArray* M) ;
   template<typename T> Matrix(const Image<T>&) ;
   Matrix(const Matrix& M) ; // copy okay, but see ref counting notes above
   ~Matrix() ;

   void push_back(const std::vector<double>& row) ;

   operator const mxArray*() const {return mat_array ;} // conversion operator
   template<typename T> Image<T> to_image() const ; // explicit "conv.op"
} ;

// Initialize an mxArray using an Image<T>
template<typename T>
Matrix::Matrix(const Image<T>& I)
   : num_rows(I.getHeight()), num_cols(I.getWidth()),
     mat_array(mxCreateDoubleMatrix(num_rows, num_cols, mxREAL)),
     current_row(num_rows), // cannot modify this particular matrix!
     ref_count(1)
{
   if (! mat_array)
      throw std::runtime_error("unable to create matrix") ;

   double* p = mxGetPr(mat_array) ;
   for (uint x = 0; x < num_cols; ++x)
      for (uint y = 0; y < num_rows; ++y, ++p)
         *p = I.getVal(x, y) ; // column major order
}

// Convert an mxArray to an Image<T>
template<typename T>
Image<T> Matrix::to_image() const
{
   Image<T> I(num_cols, num_rows, NO_INIT) ;

   const double* p = mxGetPr(mat_array) ;
   for (uint x = 0; x < num_cols; ++x)
      for (uint y = 0; y < num_rows; ++y, ++p)
          I.setVal(x, y, static_cast<T>(*p)) ; // column major order

   return I ;
}

//--------------------------- MATLAB ENGINE -----------------------------

// Renninger-Malik used the Netlab toolbox's K-means implementation to
// compute the universal textons of the training set. We do the same
// thing. But rather than fire up Matlab separately and hand it the
// training textons via a .mat file, we simply start the Matlab engine
// from inside of C++ and pass it the training textons matrix directly.
// Once the K-means is done, we obtain the universal textons matrix from
// the engine before shutting it down.
//
// The following class provides a quick wrapper around the Matlab engine
// API.
typedef Engine MatEngine ;
class Engine {
   MatEngine* mat_engine ;
public :
   Engine(const std::string& netlab_path) ;
   Matrix get(const std::string& variable_name) ;
   void put(const std::string& variable_name, const Matrix&) ;
   void exec(const std::string& command) ;
   ~Engine() ;
} ;

//---------------------------- MATLAB FILE ------------------------------

// The following class encapsulates a small portion of the Matlab file
// API to help with the task of writing the training set's universal
// textons to a .mat file.
class File {
   MATFile* mat_file ;
public :
   File(const std::string& file_name, const std::string& open_mode) ;
   Matrix get(const std::string& variable_name) ;
   void put(const std::string& variable_name, const Matrix& matrix) ;
   ~File() ;
} ;

// To put some data into a Matlab file, we need a variable name and an
// mxArray containing the data.
typedef std::pair<std::string, Matrix> FileData ;

// Output operator for writing matrices to a Matlab file
File& operator<<(File& f, const FileData& d) ;

//-----------------------------------------------------------------------

} // end of namespace encapsulating this file

/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif
