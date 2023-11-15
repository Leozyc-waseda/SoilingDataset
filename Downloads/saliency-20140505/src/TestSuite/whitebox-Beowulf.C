/*!@file TestSuite/whitebox-Beowulf.C Whitebox tests of Beowulf and
  related classes. */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/TestSuite/whitebox-Beowulf.C $
// $Id: whitebox-Beowulf.C 9061 2007-11-29 18:31:14Z rjpeters $
//

#ifndef TESTSUITE_WHITEBOX_BEOWULF_C_DEFINED
#define TESTSUITE_WHITEBOX_BEOWULF_C_DEFINED

#include "Beowulf/Beowulf.H"
#include "Beowulf/TCPmessage.H"
#include "Component/ModelManager.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "TestSuite/TestSuite.H"
#include "Util/Timer.H"
#include "Util/csignals.H"

#include <limits>
#include <signal.h> // for kill()
#include <sys/types.h> // for pid_t
#include <sys/wait.h> // for waitpid()
#include <unistd.h>

namespace
{
  int run_master(TCPmessage* rmsg)
  {
    try
      {
        MYLOGPREFIX = "master";

        ModelManager manager("test-master");

        nub::soft_ref<Beowulf> beomaster
          (new Beowulf(manager, "Beowulf Master", "BeowulfMaster", true));
        manager.addSubComponent(beomaster);

        const char* argv[] = { "test-master", "--ip-addr=127.0.0.1",
                               "--ip-port=9788",
                               "--beowulf-slaves=localhost:9790",
                               "--beowulf-init-timeout=5.0", 0 };

        if (manager.parseCommandLine(5, argv, "", 0, 0) == false)
          return 1;

        MYLOGVERB = LOG_DEBUG;

        LINFO("starting...");

        manager.start();

        LINFO("... done starting");

        int rnode = 0;
        int32 rframe = 0, raction = 0;

        LINFO("receiving...");

        bool gotit = false;

        Timer t;

        while (t.getSecs() < 10.0)
          {
            gotit = beomaster->receive(rnode, *rmsg, rframe, raction, 50);
            if (gotit) break;
          }

        LINFO("gotit=%d", int(gotit));

        LINFO("exiting");

        return 0;
      }
    catch (...)
      {
      }

    return 1;
  }

  int run_slave(TCPmessage* msg)
  {
    try
      {
        MYLOGPREFIX = "slave";

        ModelManager manager("test-slave");

        nub::soft_ref<Beowulf> beoslave
          (new Beowulf(manager, "Beowulf Slave", "BeowulfSlave", false));

        manager.addSubComponent(beoslave);

        const char* argv[] = { "test-slave", "--ip-addr=127.0.0.1",
                               "--ip-port=9790", 0 };

        if (manager.parseCommandLine(3, argv, "", 0, 0) == false)
          return 1;

        MYLOGVERB = LOG_DEBUG;

        LINFO("starting...");

        manager.start();

        LINFO("... done starting");

        sleep(2);

        LINFO("sending...");
        beoslave->send(-1, *msg);
        LINFO("... done sending");

        volatile int signum = 0;
        catchsignals(&signum);

        LINFO("waiting for a signal...");

        while (signum == 0)
          usleep(10000);

        LINFO("... got signal %d", signum);

        LINFO("exiting");

        return 0;
      }
    catch (...)
      {
      }

    return 1;
  }
}

static void beowulf_xx_basic_tcpmessage_xx_1(TestSuite& suite)
{
  // ok, the basic idea here is that we spawn a beowulf slave process
  // and have it send a TCPmessage back to us in the parent process
  // where we try to receive that TCPmessage through a beowulf master
  // object, and then verify that the message has the expected
  // contents

  Image<PixRGB<byte> > orig_colbyteima(4, 4, NO_INIT);
  Image<byte> orig_byteima(4, 4, NO_INIT);
  Image<float> orig_floatima(4, 4, NO_INIT);
  for (int i = 0; i < orig_colbyteima.getSize(); ++i)
    {
      orig_colbyteima.setVal(i, PixRGB<byte>(i, 2*i, 3*i));
      orig_byteima.setVal(i, 4*i);
      orig_floatima.setVal(i, 0.5f*i);
    }
  const std::string orig_string("Hello, World!");
  const int32 orig_int32 = 42;
  const int64 orig_int64 = std::numeric_limits<int64>::max();
  const float orig_float = 1234.5678f;
  const double orig_double = 1.234567e8;

  pid_t pid = fork();

  if (pid == 0)
    {
      // child

      TCPmessage smsg;
      smsg.reset(0, 0);
      smsg.addImage(orig_colbyteima);
      smsg.addImage(orig_byteima);
      smsg.addImage(orig_floatima);
      smsg.addString(orig_string.c_str());
      smsg.addInt32(orig_int32);
      smsg.addInt64(orig_int64);
      smsg.addFloat(orig_float);
      smsg.addDouble(orig_double);

      exit(run_slave(&smsg));
    }
  else
    {
      // parent

      sleep(1);

      TCPmessage rmsg;

      int parentcode = run_master(&rmsg);
      REQUIRE_EQ(parentcode, 0);

      kill(pid, SIGINT);

      int childcode = -1;
      int c = 0;
      while (waitpid(pid, &childcode, WNOHANG) == 0)
        {
          usleep(10000);
          if (++c % 100 == 0)
            LINFO("waiting for child to exit");
        }

      bool no_exceptions = true;

      try
        {
          const Image<PixRGB<byte> > cbi = rmsg.getElementColByteIma();
          const Image<byte> bi = rmsg.getElementByteIma();
          const Image<float> fi = rmsg.getElementFloatIma();
          const std::string str = rmsg.getElementString();
          const int32 i32 = rmsg.getElementInt32();
          const int64 i64 = rmsg.getElementInt64();
          const float f = rmsg.getElementFloat();
          const double d = rmsg.getElementDouble();

          REQUIRE_EQ(cbi, orig_colbyteima);
          REQUIRE_EQ(bi, orig_byteima);
          REQUIRE_EQ(fi, orig_floatima);
          REQUIRE_EQ(str, orig_string);
          REQUIRE_EQ(i32, orig_int32);
          REQUIRE(i64 == orig_int64);
          REQUIRE_EQ(f, orig_float);
          REQUIRE_EQ(d, orig_double);
        }
      catch (...)
        {
          no_exceptions = false;
        }

      REQUIRE(no_exceptions);
    }
}

///////////////////////////////////////////////////////////////////////
//
// main
//
///////////////////////////////////////////////////////////////////////

int main(int argc, const char** argv)
{
  TestSuite suite;

  suite.ADD_TEST(beowulf_xx_basic_tcpmessage_xx_1);

  suite.parseAndRun(argc, argv);

  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // TESTSUITE_WHITEBOX_BEOWULF_C_DEFINED
