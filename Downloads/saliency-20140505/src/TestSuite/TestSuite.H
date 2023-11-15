/*!@file TestSuite/TestSuite.H Class to manage a suite of tests */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/TestSuite/TestSuite.H $
// $Id: TestSuite.H 12962 2010-03-06 02:13:53Z irock $
//

#ifndef TESTSUITE_H_DEFINED
#define TESTSUITE_H_DEFINED

#include "Util/StringConversions.H"
#include "Util/Types.H" // for uint

#include <string>

class TestSuite;

typedef void (*TestFunc)(TestSuite&);

template <class T> class Image;

//! TestSuite manages a set of tests.
/*! The tests can be run one at a time or all at once. Each "test" is a
    function whose signature matches the TestFunc typedef. Such functions
    receive a TestSuite as a reference parameter. This allows the
    TestFunc's to use TestSuite's diagnostic functions (such as require()
    and requireEq()) to verify whatever the test is testing. */
class TestSuite
{
  TestSuite(const TestSuite&); // not allowed
  TestSuite& operator=(const TestSuite&); // not allowed

public:
  //! Default constructor builds an empty test suite.
  TestSuite();

  //! Destructor.
  ~TestSuite();

  //! Add a test to the test suite.
  /*! (Note that the ADD_TEST macro below makes it easier to have the string
      name of the function be the same as the C++ function name, without
      having to type it twice.) */
  void addTest(const char* name, TestFunc func);

  //! Print all the available tests to stdout.
  void printAvailableTests() const;

  //! Print all the available tests to stdout.
  void printAvailableTestsForPerl() const;

  //! Run a single test, optionally repeating it multiple times.
  /*! (Multiple repeats may be useful for profiling.) The output of the
      test is printed to standard output: on the first line is a boolean
      code indicating whether the test succeeded (code==1) or failed
      (code==0), and on subsequent lines are any diagnostic messages that
      may relate to the success or failure of the test (e.g., require() and
      requireEq() will generate a diagnostic message on failure). */
  void runTest(int test_n, int repeat_n = 1);

  //! Parse command-line args and take appropriate action.
  /*! This may involve either printing a list of available tests, or
      running a specific test. */
  void parseAndRun(int argc, const char** argv);

  //! Require that the expression evaluates to true.
  /** Return the truth value of the expression. */
  bool require(bool expr, const char* srcfile, int line, const char* expr_str);

  //! Different types of arithmetic conditions that can be required.
  enum Op
    {
      EQ,   // ==
      NEQ,  // !=
      LT,   // <
      LTE,  // <=
      GT,   // >
      GTE   // >=
    };

  //! Check that the given "int" expressions compare equal.
  /** Return the truth value of the comparison. */
  bool requireEq(int expr, int expected,
                 const char* srcfile, int line, const char* expr_str);

  //! Check that the given "unsigned int" expressions compare equal.
  /** Return the truth value of the comparison. */
  bool requireEq(uint expr, uint expected,
                 const char* srcfile, int line, const char* expr_str);

  //! Check that the given "long" expressions compare equal.
  /** Return the truth value of the comparison. */
  bool requireEq(long expr, long expected,
                 const char* srcfile, int line, const char* expr_str);

  //! Check that the given "unsigned long" expressions compare equal.
  /** Return the truth value of the comparison. */
  bool requireEq(unsigned long expr, unsigned long expected,
                 const char* srcfile, int line, const char* expr_str);

  //! Check that the given std::string expressions compare equal (as strings).
  /** Return the truth value of the comparison. */
  bool requireEq(const std::string& expr, const std::string& expected,
                 const char* srcfile, int line, const char* expr_str);

  //! Check that the given "double" expressions compare equal.
  /** Return the truth value of the comparison. */
  bool requireEq(double expr, double expected,
                 const char* srcfile, int line, const char* expr_str);

  //! Check that the expression "lhs OP rhs" is true.
  /** Return the truth value of the comparison. */
  bool require(int lhs, Op op, int rhs,
               const char* srcfile, int line,
               const char* lhs_str, const char* rhs_str);

  //! Check that the expression "lhs OP rhs" is true.
  /** Return the truth value of the comparison. */
  bool require(long lhs, Op op, long rhs,
               const char* srcfile, int line,
               const char* lhs_str, const char* rhs_str);

  //! Check that the expression "lhs OP rhs" is true.
  /** Return the truth value of the comparison. */
  bool require(double lhs, Op op, double rhs,
               const char* srcfile, int line,
               const char* lhs_str, const char* rhs_str);

  //! Check that the given Image objects compare equal.
  template <class T_or_RGB>
  /** Return the truth value of the comparison. */
  bool requireEq(const Image<T_or_RGB>& expr, const Image<T_or_RGB>& expected,
                 const char* srcfile, int line, const char* expr_str);

  //! Check that the given Image objects compare equal to given precision
  /** Return the truth value of the comparison. */
  bool requireEq(const Image<float>& expr, const Image<float>& expected,
                 const float prec, const char* srcfile, int line,
                 const char* expr_str);

  //! Check that the given user-defined objects compare equal to given precision
  /** Return the truth value of the comparison. */
  template <class T>
  bool requireEqUserType(const T& expr, const T& expected,
                         const char* srcfile, int line,
                         const char* expr_str)
  {
    return this->requireEqUserTypeImpl
      (expr == expected,
       convertToString(expr),
       convertToString(expected),
       srcfile, line, expr_str);
  }

private:
  bool requireEqUserTypeImpl(const bool ok,
                             const std::string& expr,
                             const std::string& expected,
                             const char* srcfile, int line,
                             const char* expr_str);

  struct Impl;
  Impl* const rep;
};

#define ADD_TEST(x) addTest(#x, &x);

// These macros hackily assumes that "suite" is already declared as a local
// variable (or function parameter, as it should be in the test functions)

#define REQUIRE_EQ(expr, targ) \
suite.requireEq((expr), (targ), __FILE__, __LINE__, #expr)

#define REQUIRE_EQFP(expr, targ, prec) \
suite.requireEq((expr), (targ), prec, __FILE__, __LINE__, #expr)

#define REQUIRE_EQ_USERTYPE(expr, targ) \
suite.requireEqUserType((expr), (targ), __FILE__, __LINE__, #expr)

#define REQUIRE(expr) \
suite.require((expr), __FILE__, __LINE__, #expr)

#define REQUIRE_NEQ(lhs, rhs) \
suite.require((lhs), TestSuite::NEQ, (rhs), __FILE__, __LINE__, #lhs, #rhs)

#define REQUIRE_LT(lhs, rhs) \
suite.require((lhs), TestSuite::LT, (rhs), __FILE__, __LINE__, #lhs, #rhs)

#define REQUIRE_LTE(lhs, rhs) \
suite.require((lhs), TestSuite::LTE, (rhs), __FILE__, __LINE__, #lhs, #rhs)

#define REQUIRE_GT(lhs, rhs) \
suite.require((lhs), TestSuite::GT, (rhs), __FILE__, __LINE__, #lhs, #rhs)

#define REQUIRE_GTE(lhs, rhs) \
suite.require((lhs), TestSuite::GTE, (rhs), __FILE__, __LINE__, #lhs, #rhs)


/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif // !TESTSUITE_H_DEFINED
