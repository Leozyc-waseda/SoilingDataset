/*!@file Matlab/MexModelManager.H
 */
// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2002   //
// by the University of Southern California (USC) and the iLab at USC.  //
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
// Primary maintainer for this file: Dirk Walther <walther@caltech.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Matlab/MexModelManager.H $
// $Id: MexModelManager.H 6022 2005-12-08 02:40:53Z rjpeters $
//

#ifndef MEXMODELMANAGER_H_DEFINED
#define MEXMODELMANAGER_H_DEFINED

#include "Component/ModelManager.H"
#include <mex.h>
#include <vector>

//! A special form of ModelManager used in mex files
class MexModelManager : public ModelManager
{
public:
  // ######################################################################

  //! Default constructor
  /*! @param descrName descriptive name
    @param tagName tag name (see ModelComponent.H)*/
  MexModelManager(const std::string& descrName = "Mex Model",
                  const std::string& tagName = "mexModel");

  //! Virtual destructor ensures proper destruction of derived classes
  virtual ~MexModelManager();

  //! Parse all input arguments
  /*! All strings that start with '-' are interpreted as command-line
    options, all other arguments will be available though
    getExtraMexArg(). Use '--help' for a list of available options.
    Returns true upon fully successful parse.
    @param nrhs the number of command-line args
    @param prhs the input args
    @param usage a textual description of the expected usage format
    @param minarg the minimum number of non-optional arguments
    @param maxarg the maximum number of non-optional args, or -1 if
      unbounded */
  bool parseMexCommandLine(const int nrhs, const mxArray *prhs[],
                           const char *usage, const int minarg, const int maxarg);

  //! Get the number of available extra args
  uint numExtraMexArgs() const;

  //! Get an extra arg, by number
  const mxArray* getExtraMexArg(const uint num) const;

  //! Called to leave the program, e.g. when printing the Help messages
  /*!This implementation calls mexErrMsgTxt.*/
  void exitMexFunction(int return_code = 0);

protected:

  //! non-option command-line args
  std::vector<const mxArray*> extraMexArgs;

};


#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
