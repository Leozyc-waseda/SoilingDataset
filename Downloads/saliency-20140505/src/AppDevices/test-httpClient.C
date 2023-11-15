/*!@file AppDevices/test-httpClient.C Test the http client  */

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
// Primary maintainer for this file: Lior Elazary
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppDevices/test-httpClient.C $
// $Id: test-httpClient.C 12962 2010-03-06 02:13:53Z irock $
//

#include "Component/ModelManager.H"
#include "Devices/KeyBoard.H"
#include "Devices/HTTPClient.H"
#include "Util/log.H"

#include <stdio.h>

int main(int argc, const char **argv)
{
  // Instantiate a ModelManager:
  ModelManager manager("Test Model for httpClient Class");

  // Instantiate our various ModelComponents:
  nub::soft_ref<HTTPClient> httpClient(new HTTPClient(manager));
  manager.addSubComponent(httpClient);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "<data>", 1, 1) == false) return(1);

  std::string data = manager.getExtraArg(0);

  // let's get all our ModelComponent instances started:
  manager.start();

  httpClient->sendGetRequest(data);

  std::string retData = httpClient->readData();

  LINFO("GOT: %s", retData.c_str());

  // stop all our ModelComponents
  manager.stop();

  // all done!
  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
