/*!@file TestSuite/TestSuite.C Class to manage a suite of tests */

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
// Primary maintainer for this file: Rob Peters <rjpeters@klab.caltech.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/TestSuite/TestSuite.C $
// $Id: TestSuite.C 10854 2009-02-14 05:09:44Z mundhenk $
//

#ifndef TESTSUITE_C_DEFINED
#define TESTSUITE_C_DEFINED

#include "TestSuite/TestSuite.H"

#include "Image/IO.H"
#include "Image/Image.H"
#include "Image/MathOps.H"
#include "Image/Pixels.H"
#include "rutz/prof.h"

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////
/*
 *
 * TestData helper struct just pairs a TestFunc with its name.
 *
 */
///////////////////////////////////////////////////////////////////////

struct TestData
{
  TestData(const std::string& n, TestFunc f) : name(n), func(f) {}

  std::string name;
  TestFunc func;
};

///////////////////////////////////////////////////////////////////////
/*
 *
 * TestSuite::Impl holds all the member variables that TestSuite
 * needs. However, by putting them in a private struct and having TestSuite
 * just store a pointer to that struct, the header file just needs to see a
 * forward declaration of the struct, and does NOT have to #include <vector>,
 * <sstream>, etc.
 *
 */
///////////////////////////////////////////////////////////////////////

struct TestSuite::Impl
{
  Impl() : itsTests(), itsSuccess(true), itsOutput() {}

  std::vector<TestData> itsTests;
  bool                  itsSuccess;
  std::ostringstream    itsOutput;

  void printResult()
  {
    std::cout << itsSuccess << "\n"
              << itsOutput.str() << "\n";
  }

  template <class T>
  bool requireScalarEq(const T& expr, const T& expected,
                       const char* srcfile, int line, const char* expr_str)
  {
    const bool ok = (expr == expected);
    if ( !ok )
      {
        itsSuccess = false;
        itsOutput << "\tFAILED @ " << srcfile << ":" << line << ": "
                  << expr_str << ", "
                  << "expected '"
                  << std::setprecision(30) << std::showpoint << expected
                  << "', got '"
                  << std::setprecision(30) << std::showpoint << expr
                  << "'\n";
      }
    return ok;
  }

  template <class T>
  bool requireImgEq(const Image<T>& expr, const Image<T>& expected,
                    const char* srcfile, int line, const char* expr_str)
  {
    const bool ok = (expr == expected);
    if ( !ok )
      {
        itsSuccess = false;
        itsOutput << "\tFAILED @ " << srcfile << ":" << line << ": "
                  << expr_str << ", "
                  << "expected (" << convertToString(expected.getDims()) << ") "
                  << expected << ", "
                  << "got (" << convertToString(expr.getDims()) << ") "
                  << expr << "\n";
      }
    return ok;
  }

  bool requireImgEqFp(const Image<float>& expr, const Image<float>& expected,
                      const float prec, const char* srcfile, int line,
                      const char* expr_str)
  {
    bool ok = false;
    std::ostringstream extrainfo;

    if (expr.getDims() == expected.getDims())
      {
        Image<float> diff = abs(expr - expected);
        float ma; Point2D<int> loc;
        findMax(diff, loc, ma);

        ok = (ma <= prec);

        extrainfo << "(largest difference of " << ma << " at [x="
                  << loc.i << ",y=" << loc.j << "])\n";
      }

    if ( !ok )
      {
        itsSuccess = false;
        itsOutput << "\tFAILED @ " << srcfile << ":" << line << ": "
                  << expr_str << ", "
                  << "expected (" << convertToString(expected.getDims()) << ") "
                  << expected << ", "
                  << "got (" << convertToString(expr.getDims()) << ") "
                  << expr << "\n"
                  << extrainfo.str();
      }
    return ok;
  }

  bool requireEqUserTypeImpl(const bool ok,
                             const std::string& expr,
                             const std::string& expected,
                             const char* srcfile, int line,
                             const char* expr_str)

  {
    if ( !ok )
      {
        itsSuccess = false;
        itsOutput << "\tFAILED @ " << srcfile << ":" << line << ": "
                  << expr_str << ", "
                  << "expected '"
                  << expected
                  << "', got '"
                  << expr
                  << "'\n";
      }
    return ok;
  }

  template <class T>
  static bool compare(const T& lhs, Op op, const T& rhs)
  {
    switch (op)
      {
      case EQ:  return lhs == rhs;
      case NEQ: return lhs != rhs;
      case LT:  return lhs <  rhs;
      case LTE: return lhs <= rhs;
      case GT:  return lhs >  rhs;
      case GTE: return lhs >= rhs;
      }
    LFATAL("invalid Op '%d'", int(op));
    return false; // "can't happen"
  }

  static const char* opname(Op op)
  {
    switch (op)
      {
      case EQ:  return " == ";
      case NEQ: return " != ";
      case LT:  return " < ";
      case LTE: return " <= ";
      case GT:  return " > ";
      case GTE: return " >= ";
      }
    LFATAL("invalid Op '%d'", int(op));
    return ""; // "can't happen"
  }

  template <class T>
  bool require(const T& lhs, Op op, const T& rhs,
               const char* srcfile, int line,
               const char* lhs_str, const char* rhs_str)
  {
    const bool ok = compare(lhs, op, rhs);
    if ( !ok )
      {
        itsSuccess = false;
        itsOutput << "\tFAILED @ " << srcfile << ":" << line << ":\n"
                  << "\t expected " << lhs_str << opname(op) << rhs_str << ",\n "
                  << "\t got " << lhs_str << "==" << lhs
                  << " and " << rhs_str << "==" << rhs << '\n';
      }
    return ok;
  }
};

///////////////////////////////////////////////////////////////////////
//
// TestSuite member function definitions
//
///////////////////////////////////////////////////////////////////////


// ######################################################################
TestSuite::TestSuite() :
  rep(new Impl)
{}

// ######################################################################
TestSuite::~TestSuite()
{
  delete rep;
}

// ######################################################################
void TestSuite::addTest(const char* name, TestFunc func)
{
  rep->itsTests.push_back(TestData(name, func));
}

// ######################################################################
void TestSuite::printAvailableTests() const
{
  for (unsigned int i = 0; i < rep->itsTests.size(); ++i)
    std::cout << '{' << i << '\t' << rep->itsTests[i].name << "}\n";
}

// ######################################################################
void TestSuite::printAvailableTestsForPerl() const
{
  for (unsigned int i = 0; i < rep->itsTests.size(); ++i)
    std::cout << i << '\t' << rep->itsTests[i].name << "\n";
}

// ######################################################################
void TestSuite::runTest(int test_n, int repeat_n)
{
  if (test_n < 0 || (unsigned int) test_n >= rep->itsTests.size())
    return;

  for (int i = 0; i < repeat_n; ++i)
    rep->itsTests.at(test_n).func(*this);

  rep->printResult();
}

// ######################################################################
namespace
{
  void showUsageAndExit(const char* argv0)
  {
    std::cerr << "usage: " << argv0 << " <options>\n"
              << "available options:\n"
              << "\t--query        print test names and number (for tcl)\n"
              << "\t--perlquery    print test names and number (for perl)\n"
              << "\t--run <n>      run the test number <n>\n"
              << "\t--repeat <n>   repeat the test <n> times (for profiling, etc.)\n"
              << "\t--dump-prof    print a profiling summary before exit\n";

    exit(1);
  }
}

// ######################################################################
void TestSuite::parseAndRun(int argc, const char** argv)
{
  if (argc == 1)
    {
      showUsageAndExit(argv[0]);
    }

  int test_n = -1;
  int repeat_n = 1;
  bool dump_prof = false;

  for (int i = 1; i < argc; ++i)
    {
      if (strcmp(argv[i], "--run") == 0)
        {
          if (++i < argc) test_n = atoi(argv[i]);
        }

      else if (strcmp(argv[i], "--repeat") == 0)
        {
          if (++i < argc) repeat_n = atoi(argv[i]);
        }

      else if (strcmp(argv[i], "--query") == 0)
        {
          printAvailableTests();
          return;
        }

      else if (strcmp(argv[i], "--perlquery") == 0)
        {
          printAvailableTestsForPerl();
          return;
        }

      else if (strcmp(argv[i], "--dump-prof") == 0)
        {
          dump_prof = true;
        }

      else
        {
          showUsageAndExit(argv[0]);
        }
    }

  runTest(test_n, repeat_n);

  if (dump_prof)
    rutz::prof::print_all_prof_data(std::cerr);
}

// ######################################################################
bool TestSuite::require(bool expr, const char* srcfile, int line, const char* expr_str)
{
  if (!expr)
    {
      rep->itsSuccess = false;
      rep->itsOutput << "\tFAILED @ " << srcfile << ":" << line << ": "
                     << expr_str << "\n";
    }
  return expr;
}

// ######################################################################
bool TestSuite::requireEq(int expr, int expected,
                          const char* srcfile, int line, const char* expr_str)
{ return rep->requireScalarEq(expr, expected, srcfile, line, expr_str); }

// ######################################################################
bool TestSuite::requireEq(uint expr, uint expected,
                          const char* srcfile, int line, const char* expr_str)
{ return rep->requireScalarEq(expr, expected, srcfile, line, expr_str); }

// ######################################################################
bool TestSuite::requireEq(long expr, long expected,
                          const char* srcfile, int line, const char* expr_str)
{ return rep->requireScalarEq(expr, expected, srcfile, line, expr_str); }

// ######################################################################
bool TestSuite::requireEq(unsigned long expr, unsigned long expected,
                          const char* srcfile, int line, const char* expr_str)
{ return rep->requireScalarEq(expr, expected, srcfile, line, expr_str); }

// ######################################################################
bool TestSuite::requireEq(double expr, double expected,
                          const char* srcfile, int line, const char* expr_str)
{ return rep->requireScalarEq(expr, expected, srcfile, line, expr_str); }

// ######################################################################
bool TestSuite::requireEq(const std::string& expr, const std::string& expected,
                          const char* srcfile, int line, const char* expr_str)
{ return rep->requireScalarEq(expr, expected, srcfile, line, expr_str); }

// ######################################################################
template <class T>
bool TestSuite::requireEq(const Image<T>& expr, const Image<T>& expected,
                          const char* srcfile, int line, const char* expr_str)
{ return rep->requireImgEq(expr, expected, srcfile, line, expr_str); }

// ######################################################################
bool TestSuite::requireEq(const Image<float>& expr,
                          const Image<float>& expected,
                          const float prec, const char* srcfile, int line,
                          const char* expr_str)
{ return rep->requireImgEqFp(expr, expected, prec, srcfile, line, expr_str); }

// ######################################################################
bool TestSuite::require(int lhs, Op op, int rhs,
                        const char* srcfile, int line,
                        const char* lhs_str, const char* rhs_str)
{ return rep->require(lhs, op, rhs, srcfile, line, lhs_str, rhs_str); }

// ######################################################################
bool TestSuite::require(long lhs, Op op, long rhs,
                        const char* srcfile, int line,
                        const char* lhs_str, const char* rhs_str)
{ return rep->require(lhs, op, rhs, srcfile, line, lhs_str, rhs_str); }

// ######################################################################
bool TestSuite::require(double lhs, Op op, double rhs,
                        const char* srcfile, int line,
                        const char* lhs_str, const char* rhs_str)
{ return rep->require(lhs, op, rhs, srcfile, line, lhs_str, rhs_str); }

// ######################################################################
bool TestSuite::requireEqUserTypeImpl(const bool ok,
                                      const std::string& expr,
                                      const std::string& expected,
                                      const char* srcfile, int line,
                                      const char* expr_str)
{ return rep->requireEqUserTypeImpl(ok, expr, expected, srcfile, line, expr_str); }

// Include the explicit instantiations, and make sure that they go in the
// TestSuite:: namespace
#define INST_CLASS TestSuite::
#include "inst/TestSuite/TestSuite.I"

template bool INST_CLASS requireEq(const Image<int>& expr, const Image<int>& expected, const char* srcfile, int line, const char* expr_str);

/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif // !TESTSUITE_C_DEFINED
