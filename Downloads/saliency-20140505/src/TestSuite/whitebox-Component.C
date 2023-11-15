/*!@file TestSuite/whitebox-Component.C */

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
// Primary maintainer for this file:
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/TestSuite/whitebox-Component.C $
// $Id: whitebox-Component.C 14376 2011-01-11 02:44:34Z pez $
//

#ifndef TESTSUITE_WHITEBOX_COMPONENT_C_DEFINED
#define TESTSUITE_WHITEBOX_COMPONENT_C_DEFINED

#include "Component/ModelManager.H"
#include "Component/ModelOptionDef.H"
#include "Component/ModelParam.H"
#include "Component/OptionManager.H"
#include "Component/ParamClient.H"
#include "Component/ParamMap.H"
#include "TestSuite/TestSuite.H"
#include "Util/sformat.H"

#include <exception>
#include <vector>

namespace
{
  // Dummy implementation of ParamClient that we can use to make sure
  // our model params are behaving as expected
  class TestParamClient : public ParamClient
  {
  public:
    bool param_has_registered,
      param_has_unregistered,
      param_has_changed,
      should_reject_changes;

    TestParamClient() :
      param_has_registered(false),
      param_has_unregistered(false),
      param_has_changed(false),
      should_reject_changes(false) {}

    virtual ~TestParamClient() {}

    virtual void registerParam(ModelParamBase*)
    { param_has_registered = true; }

    virtual void registerOptionedParam(OptionedModelParam*, int)
    { param_has_registered = true; }

    virtual void unregisterParam(const ModelParamBase*)
    { param_has_unregistered = true; }

    virtual void paramChanged(ModelParamBase* param,
                              const bool valueChanged,
                              ParamClient::ChangeStatus* status)
    {
      param_has_changed = true;
      if (should_reject_changes)
        *status = ParamClient::CHANGE_REJECTED;
    }
  };

  const ModelOptionDef OPT_testOption =
    {
      MODOPT_ARG(int),
      "testOption",
      &MOC_GENERAL,
      OPTEXP_CORE,
      "this is a test option",
      "call-it-like-this",
      's',
      NULL,
      "1"
    };

  class TestComponent : public ModelComponent
  {
  public:
    TestComponent(OptionManager& m) :
      ModelComponent(m, "test", "test") {}
  };

  // Like TestComponent, but with virtual inheritance
  class TestComponent2 : public virtual ModelComponent
  {
  public:
    TestComponent2(OptionManager& m) :
      ModelComponent(m, "test2", "test2") {}
  };

  class SlaveComponent : public ModelComponent
  {
  public:
    SlaveComponent(OptionManager& m) :
      ModelComponent(m, "slave", "slave"),
      itsParam(&OPT_testOption, this)
    {}

    OModelParam<int> itsParam;
  };

  class MasterComponent : public ModelComponent
  {
  public:
    MasterComponent(OptionManager& m) :
      ModelComponent(m, "master", "master"),
      itsSub(new SlaveComponent(m)),
      itsParam(&OPT_testOption, this)
    {
      this->addSubComponent(itsSub);
    }

    virtual void paramChanged(ModelParamBase* const param,
                              const bool valueChanged,
                              ParamClient::ChangeStatus* status)
    {
      if (param == &itsParam)
        {
          itsSub->setModelParamString("testOption", "42");
        }
    }

    nub::ref<SlaveComponent> itsSub;

    OModelParam<int> itsParam;
  };
}

struct WeirdParamType
{
  int x;

  bool operator==(const WeirdParamType& that) { return this->x == that.x; }
};

void convertFromString(const std::string& str, WeirdParamType& p)
{
  // this converter function always throws an exception, but it messes
  // with p's value beforehand; we do this so that we can make sure
  // that calling code is careful so that exceptions don't
  // unnecessarily garble param values
  p.x = -1;
  conversion_error::raise<WeirdParamType>(str);
}

std::string convertToString(const WeirdParamType& p)
{
  return convertToString(p.x);
}

static void modelparam_xx_getset_xx_1(TestSuite& suite)
{
  // do some idiot checks to make sure trivial things are working as
  // expected (e.g. set a value, then read it back... this tests that
  // the string conversions are working properly)

  TestParamClient client;

  NModelParam<int> p1("foo", &client, 1);

  p1.setVal(2);
  REQUIRE_EQ(p1.getVal(), 2);

  p1.setValString("13579");
  REQUIRE_EQ(p1.getVal(), 13579);

  p1.setVal(-45);
  REQUIRE_EQ(p1.getValString(), "-45");

  // now check that we correctly throw an exception for bogus
  // conversions:

  bool caught;

  try                         { caught = false; p1.setValString("abcde"); }
  catch (conversion_error& e) { caught = true; }
  REQUIRE(caught);

  try                         { caught = false; p1.setValString(""); }
  catch (conversion_error& e) { caught = true; }
  REQUIRE(caught);
}

static void modelparam_xx_getset_xx_2(TestSuite& suite)
{
  TestParamClient client;

  WeirdParamType v; v.x = 42;
  NModelParam<WeirdParamType> p1("foo", &client, v);

  REQUIRE_EQ(p1.getValString(), std::string("42"));

  bool caught;
  try                         { caught = false; p1.setValString(""); }
  catch (conversion_error& e) { caught = true; }
  REQUIRE(caught);

  // make sure that p1's value is still the same even though an
  // exception was thrown during the setValString() call
  REQUIRE_EQ(p1.getValString(), std::string("42"));
}

static void modelparam_xx_getset_xx_3(TestSuite& suite)
{
  // Here we are testing the ParamClient::ChangeStatus system where
  // implementors of paramChanged() can reject a particular parameter
  // change by setting *status to ParamClient::CHANGE_REJECTED

  TestParamClient client;

  NModelParam<int> p1("foo", &client, 0);
  REQUIRE_EQ(p1.getVal(), 0);

  p1.setVal(1);
  REQUIRE_EQ(p1.getVal(), 1);

  client.should_reject_changes = true;
  p1.setVal(2);
  // the change should be rejected in paramChanged() which should
  // cause NModelParam to restore the original value in p1
  REQUIRE_EQ(p1.getVal(), 1);

  client.should_reject_changes = false;
  p1.setVal(3);
  // now we should be allowing changes to get through again:
  REQUIRE_EQ(p1.getVal(), 3);
}

static void modelparam_xx_readfrom_xx_1(TestSuite& suite)
{
  // we want to ensure that any value change happening in readFrom()
  // gets passed on to the proper ParamClient

  TestParamClient client;

  REQUIRE(!client.param_has_registered);
  REQUIRE(!client.param_has_unregistered);
  REQUIRE(!client.param_has_changed);

  {
    NModelParam<int> param("foo", &client, 42);

    // make sure that the param registered with us
    REQUIRE(client.param_has_registered);
    REQUIRE(!client.param_has_unregistered);
    REQUIRE(!client.param_has_changed);

    // make the param get its value from a ParamMap that won't
    // change its existing value
    ParamMap pmap1;
    pmap1.putIntParam("foo", 42);
    param.readFrom(pmap1, false);

    REQUIRE(client.param_has_registered);
    REQUIRE(!client.param_has_unregistered);
    REQUIRE(!client.param_has_changed);

    // now make the param get its value from a ParamMap that WILL change
    // its existing value
    ParamMap pmap2;
    pmap2.putIntParam("foo", 49);

    param.readFrom(pmap2, false);

    REQUIRE(client.param_has_registered);
    REQUIRE(!client.param_has_unregistered);
    REQUIRE(client.param_has_changed);
  }

  // now the model param has gone out of scope and has been destroyed,
  // so it should have un-registered:

  REQUIRE(client.param_has_registered);
  REQUIRE(client.param_has_unregistered);
  REQUIRE(client.param_has_changed);
}

static void modelparam_xx_writeto_xx_1(TestSuite& suite)
{
  TestParamClient client;

  NModelParam<double> param("foo", &client, 3.5);

  ParamMap pmap;
  param.writeTo(pmap);

  REQUIRE_EQ(pmap.getDoubleParam("foo", 0.0), 3.5);
}

static void modelmanager_xx_findoptiondef_xx_1(TestSuite& suite)
{
  ModelManager m;
  m.setOptionValString(&OPT_testOption, "59");
  std::string s = std::string("test") + "Option";
  REQUIRE(m.findOptionDef(s.c_str()) == &OPT_testOption);
  REQUIRE_EQ(m.getOptionValString(&OPT_testOption), "59");
}

static void modelmanager_xx_defaultvalue_xx_1(TestSuite& suite)
{
  ModelManager m;

  ModelComponent c1(m, "c1", "c1"), c2(m, "c2", "c2");

  // set up an OModelParam that doesn't force its value as default
  OModelParam<int> p1(&OPT_testOption, &c1);
  REQUIRE_EQ(p1.getVal(), 1);
  p1.setVal(5);
  REQUIRE_EQ(p1.getVal(), 5);
  m.requestOption(p1, false);
  REQUIRE_EQ(p1.getVal(), 1);

  // set up an OModelParam that DOES force its value as default
  OModelParam<int> p2(&OPT_testOption, &c2, 6, USE_MY_VAL);
  REQUIRE_EQ(p1.getVal(), 1);
  REQUIRE_EQ(p2.getVal(), 6);
  m.requestOption(p2, true);
  REQUIRE_EQ(p1.getVal(), 6);
  REQUIRE_EQ(p2.getVal(), 6);

  m.setOptionValString(&OPT_testOption, "42");
  REQUIRE_EQ(p1.getVal(), 42);
  REQUIRE_EQ(p1.getVal(), 42);
}

static void modelcomponent_xx_forbidsharedptr_xx_1(TestSuite& suite)
{
  ModelManager m;

  bool got_exception = false;
  try { rutz::shared_ptr<ModelComponent> p(new TestComponent(m)); }
  catch (std::exception& e) { got_exception = true; }

  REQUIRE(got_exception);
}

static void modelcomponent_xx_forbidsharedptr_xx_2(TestSuite& suite)
{
  ModelManager m;

  try { rutz::shared_ptr<ModelComponent> p(new TestComponent2(m)); }
  catch (std::exception& e) {}

  // FIXME: We don't have a good way to make this test actually work:
  /* REQUIRE(got_exception); */

  // The problem is that ModelComponent's pointer-checking mechanisms
  // runs into trouble when we have virtual inheritance. When somebody
  // inherits virtually from ModelComponent, then ModelComponent's
  // constructor has no way of knowing where the address of the full
  // object actually begins.
}

static void modelcomponent_xx_root_object_xx_1(TestSuite& suite)
{
  ModelManager m;

  REQUIRE(m.getRootObject() == &m);
  REQUIRE(m.getParent() == 0);

  nub::soft_ref<ModelComponent> c1(new TestComponent(m));

  REQUIRE(c1->getRootObject() == c1.get());
  REQUIRE(c1->getParent() == 0);

  m.addSubComponent(c1);

  REQUIRE(c1->getRootObject() == &m);
  REQUIRE(c1->getParent() == &m);

  nub::soft_ref<ModelComponent> c2(new TestComponent2(m));

  c1->addSubComponent(c2);

  REQUIRE(c2->getRootObject() == &m);
  REQUIRE(c2->getParent() == c1.get());
}

#include "Util/Assert.H"

namespace
{
  class ReentrantTester : public ModelComponent
  {
  public:
    ReentrantTester(OptionManager& mgr)
      :
      ModelComponent(mgr, "Reentrant Tester", "ReentrantTester"),
      itsOption(&OPT_testOption, this)
    {}

    virtual void paramChanged(ModelParamBase* const param,
                              const bool valueChanged,
                              ParamClient::ChangeStatus* status)
    {
      if (param == &itsOption)
        {
          for (int i = 0; i < 10000; ++i)
            {
              itsNewParams.push_back
                (NModelParam<int>::make(sformat("foo%d", i), this, i));
            }
        }
    }

    OModelParam<int> itsOption;
    std::vector<rutz::shared_ptr<NModelParam<int> > > itsNewParams;
  };
}

static void modelcomponent_xx_reentrant_pinfos_xx_1(TestSuite& suite)
{
  // Here we are testing that it is safe to create new model params on
  // the fly during a paramChanged() callback triggered by
  // exportOptions(); previously this had the potential to cause a
  // crash because ModelComponent's rep->pinfos would be appended to
  // (by the creation of a new NModelParam) thus invalidating
  // iterators, while we were in the midst of iterating over
  // rep->pinfos in exportOptions(). The new implementation of
  // ModelComponent::exportOptions() eschews iterators in favor of
  // simple iteration using array indexing, so that it is safe to
  // change rep->pinfos while we are iterating over it.

  ModelManager m;

  nub::ref<ReentrantTester> r(new ReentrantTester(m));

  m.addSubComponent(r);

  m.exportOptions(MC_RECURSE);

  REQUIRE_EQ(r->getNumModelParams(), size_t(10001));
}

static void modelcomponent_xx_master_slave_export_xx_1(TestSuite& suite)
{
  ModelManager m;
  nub::ref<MasterComponent> master(new MasterComponent(m));
  m.addSubComponent(master);
  m.exportOptions(MC_RECURSE);

  // when exportOptions() is called on MasterComponent, it sets
  // SlaveComponent::itsParam to 42, and we are testing that
  // SlaveComponent::itsParam should not be overwritten by a later
  // exportOptions() calls -- that is, we are testing that the
  // implementation of exportOptions() first propagates to
  // subcomponents, and then exports the options of the calling
  // ModelComponent

  REQUIRE_EQ(master->itsSub->itsParam.getVal(), 42);
}

///////////////////////////////////////////////////////////////////////
//
// main
//
///////////////////////////////////////////////////////////////////////

int main(int argc, const char** argv)
{
  TestSuite suite;

  suite.ADD_TEST(modelparam_xx_getset_xx_1);
  suite.ADD_TEST(modelparam_xx_getset_xx_2);
  suite.ADD_TEST(modelparam_xx_getset_xx_3);
  suite.ADD_TEST(modelparam_xx_readfrom_xx_1);
  suite.ADD_TEST(modelparam_xx_writeto_xx_1);
  suite.ADD_TEST(modelmanager_xx_findoptiondef_xx_1);
  suite.ADD_TEST(modelmanager_xx_defaultvalue_xx_1);
#if defined(GVX_MEM_DEBUG)
  suite.ADD_TEST(modelcomponent_xx_forbidsharedptr_xx_1);
  suite.ADD_TEST(modelcomponent_xx_forbidsharedptr_xx_2);
#else
  // prevent "unused static function" warnings:
  (void) &modelcomponent_xx_forbidsharedptr_xx_1;
  (void) &modelcomponent_xx_forbidsharedptr_xx_2;
#endif
  suite.ADD_TEST(modelcomponent_xx_root_object_xx_1);

  suite.ADD_TEST(modelcomponent_xx_reentrant_pinfos_xx_1);

  suite.ADD_TEST(modelcomponent_xx_master_slave_export_xx_1);

  suite.parseAndRun(argc, argv);

  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif // TESTSUITE_WHITEBOX_COMPONENT_C_DEFINED
