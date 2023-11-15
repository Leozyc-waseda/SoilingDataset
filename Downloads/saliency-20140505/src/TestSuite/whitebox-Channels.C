/*!@file TestSuite/whitebox-Channels.C */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/TestSuite/whitebox-Channels.C $
// $Id: whitebox-Channels.C 9992 2008-07-28 19:10:59Z lior $
//

#ifndef TESTSUITE_WHITEBOX_CHANNELS_C_DEFINED
#define TESTSUITE_WHITEBOX_CHANNELS_C_DEFINED

#include "Channels/ChannelFacet.H"
#include "Channels/ComplexChannel.H"
#include "Channels/SingleChannel.H"
#include "Component/ModelManager.H"
#include "Component/ParamMap.H"
#include "TestSuite/TestSuite.H"

#include <sstream>

namespace
{
  class DummyFacet : public ChannelFacet
  {
  public:
    DummyFacet(int m) : magic(m) {}

    virtual ~DummyFacet() {}

    virtual void writeTo(ParamMap& pmap) const
    {
      pmap.putIntParam("magic", magic);
    }

    virtual void readFrom(const ParamMap& pmap)
    {
      pmap.queryIntParam("magic", magic);
    }

    int magic;
  };
}

static void Channels_xx_facets_xx_1(TestSuite& suite)
{
  ModelManager m("");

  nub::ref<SingleChannel> c
    (new SingleChannel
     (m, "dummy", "dummy", UNKNOWN,
      rutz::make_shared(new GaussianPyrBuilder<float>(5))));

  m.addSubComponent(c);
  m.exportOptions(MC_RECURSE);

  m.start();

  REQUIRE_EQ(c->hasFacet<DummyFacet>(), false);
  REQUIRE_EQ(c->hasFacet<int>(), false);

  rutz::shared_ptr<DummyFacet> f(new DummyFacet(42));

  c->setFacet(f);

  REQUIRE_EQ(c->hasFacet<DummyFacet>(), true);
  REQUIRE_EQ(c->hasFacet<int>(), false);
  REQUIRE_EQ(c->getFacet<DummyFacet>()->magic, 42);

  ParamMap pmap;
  c->writeTo(pmap);
  std::ostringstream oss;
  pmap.format(oss);

  REQUIRE_EQ(oss.str(),
             std::string("(anonymous\\ namespace)::DummyFacet  {\n"
                         "\tmagic  42\n"
                         "}\n"
                         "descriptivename  dummy\n"));

  f->magic = 39;
  REQUIRE_EQ(f->magic, 39);

  ParamMap pmap2;
  std::istringstream iss(oss.str());
  pmap2.load(iss);
  c->readFrom(pmap2);
  REQUIRE_EQ(f->magic, 42);
}

///////////////////////////////////////////////////////////////////////
//
// main
//
///////////////////////////////////////////////////////////////////////

int main(int argc, const char** argv)
{
  TestSuite suite;

  suite.ADD_TEST(Channels_xx_facets_xx_1);

  suite.parseAndRun(argc, argv);

  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // TESTSUITE_WHITEBOX_CHANNELS_C_DEFINED
