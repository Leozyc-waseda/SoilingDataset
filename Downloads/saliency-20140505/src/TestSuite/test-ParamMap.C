/*!@file TestSuite/test-ParamMap.C */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2005   //
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
// Primary maintainer for this file: Rob Peters <rjpeters at usc dot edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/TestSuite/test-ParamMap.C $
// $Id: test-ParamMap.C 8386 2007-05-13 22:01:11Z rjpeters $
//

#ifndef TESTSUITE_TEST_PARAMMAP_C_DEFINED
#define TESTSUITE_TEST_PARAMMAP_C_DEFINED

#include "Component/ParamMap.H"
#include "Util/sformat.H"

#include <iostream>
#include <vector>

int main()
{
  {
    // generate a ParamMap
    std::cout << "\n\n\n==== generating A ParamMap...\n\n";

    rutz::shared_ptr<ParamMap> pmap1(new ParamMap);

    pmap1->putStringParam("stringparam", "string param value");
    pmap1->putIntParam("intparam", 42);
    pmap1->putDoubleParam("doubleparam", 5.56);

    rutz::shared_ptr<ParamMap> pmap2(new ParamMap);

    pmap2->putStringParam("submapstringparam", "submap string param value");
    pmap2->putIntParam("submapintparam", 93);

    std::vector<int> values;
    values.push_back(3);
    values.push_back(10);
    values.push_back(-9);

    pmap2->putIntParam("num_array", int(values.size()));
    for (size_t i = 0; i < values.size(); ++i)
      {
        pmap2->putIntParam(sformat("array%d", int(i)), values[i]);
      }

    pmap1->putSubpmap("submap", pmap2);

    std::cout << "\n\n\n==== printing the ParamMap to stdout...\n\n";

    pmap1->format(std::cout);

    // save it to disk
    std::cout << "\n\n\n==== saving the ParamMap to disk...\n\n";
    pmap1->format("parammap-test.pmap");
  }

  // load it back from disk
  std::cout << "\n\n\n==== loading the ParamMap back in from disk...\n\n";
  rutz::shared_ptr<ParamMap> pmap3 =
    ParamMap::loadPmapFile("parammap-test.pmap");

  std::cout << "\n\n\n==== unpacking the loaded ParamMap...\n\n";

  // now parse parameters back out
  std::string s1 = pmap3->getStringParam("stringparam");
  std::cout << "got stringparam = '" << s1 << "'\n";

  int i1 = pmap3->getIntParam("intparam");
  std::cout << "got intparam = '" << i1 << "'\n";

  double d1 = pmap3->getDoubleParam("doubleparam");
  std::cout << "got doubleparam = '" << d1 << "'\n";

  rutz::shared_ptr<ParamMap> pmap4 = pmap3->getSubpmap("submap");

  std::string s2 = pmap4->getStringParam("submapstringparam");
  std::cout << "got submap.submapstringparam = '" << s2 << "'\n";

  int i2 = pmap4->getIntParam("submapintparam");
  std::cout << "got submap.submapintparam = '" << i2 << "'\n";

  int sz = pmap4->getIntParam("num_array");
  std::cout << "got submap.num_array = '" << sz << "'\n";
  for (int i = 0; i < sz; ++i)
    {
      int v = pmap4->getIntParam(sformat("array%d", i));
      std::cout << "got submap.array" << i << " = '" << v << "'\n";
    }
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // TESTSUITE_TEST_PARAMMAP_C_DEFINED
