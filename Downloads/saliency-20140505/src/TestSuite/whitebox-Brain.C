/*!@file TestSuite/whitebox-Brain.C */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/TestSuite/whitebox-Brain.C $
// $Id: whitebox-Brain.C 10827 2009-02-11 09:40:02Z itti $
//

#ifndef TESTSUITE_WHITEBOX_BRAIN_C_DEFINED
#define TESTSUITE_WHITEBOX_BRAIN_C_DEFINED

#include "Component/ModelManager.H"
#include "Neuro/AttentionGuidanceMap.H"
#include "Neuro/GistEstimator.H"
#include "Neuro/InferoTemporal.H"
#include "Neuro/Retina.H"
#include "Neuro/EyeHeadController.H"
#include "Neuro/SaliencyMap.H"
#include "Neuro/ShapeEstimator.H"
#include "Neuro/SimulationViewer.H"
#include "Neuro/StdBrain.H"
#include "Neuro/TargetChecker.H"
#include "Neuro/TaskRelevanceMap.H"
#include "Neuro/VisualCortex.H"
#include "Neuro/WinnerTakeAll.H"
#include "TestSuite/TestSuite.H"


#include "rutz/demangle.h"

namespace
{
  // these classes represent a potential future implementation of a
  // Brain base class that basically just represents the
  // collection-of-modules interface

  class Visitor
  {
  public:
    virtual ~Visitor() {}

    virtual void visit(ModelComponent* p) = 0;
  };

  template <class T>
  class TypeVisitor : public Visitor
  {
  public:
    TypeVisitor(bool strict = true) : itsResult(0), itsStrict(strict) {}

    virtual void visit(ModelComponent* p)
    {
      T* t = dynamic_cast<T*>(p);
      if (t != 0)
        {
          if (itsResult != 0 && itsStrict)
            LFATAL("ambiguous type match while searching for a %s:\n"
                   "\tfirst match was a %s, second match was a %s",
                   rutz::demangled_name(typeid(T)),
                   rutz::demangled_name(typeid(*itsResult)),
                   rutz::demangled_name(typeid(*t)));
          else
            itsResult = t;
        }
    }

    T* result() const { return itsResult; }

  private:
    T* itsResult;
    bool itsStrict;
  };

  class Aggregate
  {
  public:

    virtual ~Aggregate() {}

    virtual void accept(Visitor& c) = 0;

    template <class T>
    T* findModuleSoft()
    {
      TypeVisitor<T> checker(false);
      this->accept(checker);
      return checker.result();
    }

    template <class T>
    T* findModule()
    {
      TypeVisitor<T> checker(true);
      this->accept(checker);
      if (checker.result() == 0)
        LFATAL("no module found of type %s",
               rutz::demangled_name(typeid(T)));
      return checker.result();
    }
  };

  class ProtoBrain : public Aggregate
  {
  public:

    virtual void accept(Visitor& c)
    {
      /*
      c.visit(itsBrain->getRET().get());
      c.visit(itsBrain->getVC().get());
      c.visit(itsBrain->getEHC().get());
      c.visit(itsBrain->getSV().get());
      c.visit(itsBrain->getSE().get());
      c.visit(itsBrain->getSM().get());
      c.visit(itsBrain->getTRM().get());
      c.visit(itsBrain->getAGM().get());
      c.visit(itsBrain->getWTA().get());
      c.visit(itsBrain->getIT().get());
      c.visit(itsBrain->getGE().get());
      c.visit(itsBrain->getTC().get());
      */
    }

    nub::soft_ref<Brain> itsBrain;
  };

}

static void Brain_xx_modules_xx_1(TestSuite& suite)
{
  ModelManager mm("dummy test model manager");

  nub::ref<StdBrain> brain(new StdBrain(mm));
  mm.addSubComponent(brain);

  nub::ref<SimEventQueue> seq(new SimEventQueue(mm));
  mm.addSubComponent(seq);

  const char* args[] =
    { __FILE__, "--ehc-type=Simple", "--esc-type=Trivial",
      "--hsc-type=None", "--it-type=Std", "--ge-type=Std", 0 };

  REQUIRE(mm.parseCommandLine(4, args, "", 0, 0) == true);

  // here we're testing the local "ProtoBrain" class, which is a
  // prototype implementation of a Brain::findModule<ModuleType>()
  // interface
  ProtoBrain pbrain;
  pbrain.itsBrain = brain;

  /*

    the visiting should recurse

  REQUIRE(pbrain.findModule<Retina>() != 0);
  REQUIRE(pbrain.findModule<VisualCortex>() != 0);
  REQUIRE(pbrain.findModule<EyeHeadController>() != 0);
  REQUIRE(pbrain.findModule<SimulationViewer>() != 0);
  REQUIRE(pbrain.findModule<ShapeEstimator>() != 0);
  REQUIRE(pbrain.findModule<SaliencyMap>() != 0);
  REQUIRE(pbrain.findModule<TaskRelevanceMap>() != 0);
  REQUIRE(pbrain.findModule<AttentionGuidanceMap>() != 0);
  REQUIRE(pbrain.findModule<WinnerTakeAll>() != 0);
  REQUIRE(pbrain.findModule<InferoTemporal>() != 0);
  REQUIRE(pbrain.findModule<GistEstimator>() != 0);
  REQUIRE(pbrain.findModule<TargetChecker>() != 0);

  REQUIRE(pbrain.findModuleSoft<Brain>() == 0);

  REQUIRE(pbrain.findModuleSoft<ModelComponent>() != 0);

  bool caught = false;

  try {
    // we expect this to fail, because the findModule() call is
    // ambiguous -- ProtoBrain has more than one module of type
    // (derived from) ModelComponent
    (void) pbrain.findModule<ModelComponent>();
  } catch (lfatal_exception& e) {
    caught = true;
  }
  REQUIRE(caught);
  */
  mm.start();
}

///////////////////////////////////////////////////////////////////////
//
// main
//
///////////////////////////////////////////////////////////////////////

int main(int argc, const char** argv)
{
  TestSuite suite;

  suite.ADD_TEST(Brain_xx_modules_xx_1);

  suite.parseAndRun(argc, argv);

  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif // TESTSUITE_WHITEBOX_BRAIN_C_DEFINED
