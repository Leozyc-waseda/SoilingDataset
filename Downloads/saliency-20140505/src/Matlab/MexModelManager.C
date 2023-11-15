/*!@file Matlab/MexModelManager.C
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Matlab/MexModelManager.C $
// $Id: MexModelManager.C 15310 2012-06-01 02:29:24Z itti $
//

#include "Matlab/MexModelManager.H"

#include "Matlab/mexLog.H"
#include "Neuro/NeuroOpts.H"
#include "Util/fpe.H"
#include "Util/sformat.H"

namespace
{
  void mexPrintUsageAndExit(const char *usage)
  {
    mexFatal(sformat("USAGE: %s\nTry: %s('--help'); "
                     "for additional information.",
                     usage,mexFunctionName()));
  }
}

// ######################################################################
MexModelManager::MexModelManager(const std::string& descrName,
                                 const std::string& tagName)
  : ModelManager(descrName,tagName,false,false,false)
{
  this->setFPE(false);
  this->unRequestTestMode();
  this->unRequestUsingFPE();
  fpExceptionsUnlock();
  fpExceptionsOff();
  fpExceptionsLock();
}

MexModelManager::~MexModelManager()
{}

// ######################################################################
bool MexModelManager::parseMexCommandLine(const int nrhs, const mxArray *prhs[],
                                          const char *usage, const int minarg,
                                          const int maxarg)
{
  // let's export options if it hasn't been done already (note that
  // exportOptions() will just return without doing anything if
  // somebody had already called it)
  exportOptions(MC_RECURSE);

  // do a few resets:
  clearExtraArgs();
  extraMexArgs.clear();

  // initialize fields
  std::vector<char*> buffers;
  std::vector<const char*> args;
  args.push_back(mexFunctionName());

  // convert the aguments we get from Matlab to something like argc and argv
  for (int n = 0; n < nrhs; n++)
    {
      // Is not a string -> store in itsExtraArgs and go on
      if (!mxIsChar(prhs[n])) {
        extraMexArgs.push_back(prhs[n]); continue; }

      // It's a string. But if it's empty, store it in itsExtraArgs.
      int numelem = mxGetNumberOfElements(prhs[n]);
      if (numelem == 0) {
        extraMexArgs.push_back(prhs[n]); continue; }

      // get the string into a buffer
      int buflen = numelem * sizeof(mxChar) + 1;
      char *buf = new char[buflen];
      mxGetString(prhs[n], buf, buflen);
      buffers.push_back(buf);  // keep track of buffers to destroy them later

      // if the first character isn't a '-', store the string in itsExtraArgs
      if (buf[0] != '-') {
        extraMexArgs.push_back(prhs[n]); continue; }

      // Now separate out the options
      for (char *ptr = buf; ptr < (buf+numelem); ++ptr)
        {
          args.push_back(ptr);  //store the address of this argument
          // find the next white space
          while(!isspace(*ptr) && (ptr < (buf+numelem))) ++ptr;
          *ptr = '\0'; // blank this one out
        }
    }

  // do the core of the parsing:
  bool ret = parseCommandLineCore(args.size(), &args[0]);

  // delete all temporary buffers
  for (uint i = 0; i < buffers.size(); ++i)
    delete [] buffers[i];

  // Were we successful?
  if (ret == false) return false;

  // check if the number of extraMexArgs is within the limits we want
  if ((int(extraMexArgs.size()) < minarg) ||
      (maxarg >= 0 && int(extraMexArgs.size()) > maxarg))
    {
      if (maxarg == -1)
        mexError(sformat("Incorrect number of (non-opt) arg: %" ZU " [%d..Inf]",
                         extraMexArgs.size(), minarg));
      else
        mexError(sformat("Incorrect number of (non-opt) arg: %" ZU " [%d..%d]",
                         extraMexArgs.size(), minarg, maxarg));
      mexPrintUsageAndExit(usage);
    }

  // there shouldn't be any leftover text args here
  if (this->numExtraArgs() > 0)
    {
      for (uint i = 0; i < this->numExtraArgs(); ++i)
        mexError(sformat("Unknown command line option: %s",
                         this->getExtraArg(i).c_str()));

      mexError("There were unknown command line options.");
      mexPrintUsageAndExit(usage);
    }

  // all went well
  return true;
}

// ######################################################################
uint MexModelManager::numExtraMexArgs() const
{ return extraMexArgs.size(); }

// ######################################################################
const mxArray* MexModelManager::getExtraMexArg(const uint num) const
{
  if (num >= extraMexArgs.size()) {
    mexFatal(sformat("Invalid arg number %d (0..%" ZU ") -- IGNORED",
                     num, extraMexArgs.size()));
    return 0;
  }
  return extraMexArgs[num];
}

// ######################################################################
void MexModelManager::exitMexFunction(int return_code)
{
  if (return_code == 0)
    mexFatal(sformat("%s finished.",mexFunctionName()));
  else
    mexFatal(sformat("%s exited with error code %d.",
                     mexFunctionName(),return_code));
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
