/*!@file BeoSub/test-JAUS.C test the JAUS AUV control protocol */

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

#include "Util/log.H"
#include <signal.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <netdb.h>
#include <fcntl.h>
#include <arpa/inet.h>
#include <stdlib.h>
#include <ctype.h>

// JAUS command & control port
#define JAUSCC_PORT 3794

// Max UDP packet length
#define BUFLEN 9500

//! Trivial tester for JAUS remote commands over UDP
/*! This just turns the motors on/off in response to commands received
  over UDP in the JAUS protocol. */
int main(const int argc, const char **argv)
{
  // ########## block signals HUP and PIPE:
  sigset_t sset; sigemptyset(&sset);
  sigaddset(&sset, SIGHUP); sigaddset(&sset, SIGPIPE);
  int s = sigprocmask(SIG_BLOCK, &sset, NULL);
  if (s != 0) PLERROR("Sigprocmask failed");

  while(1)
    {
      // find out our port:
      struct servent *sv = getservbyname("jauscc", "udp"); int po;
      if (sv == NULL)
        {
          LERROR("jauscc/udp not configured in /etc/services");
          po = JAUSCC_PORT; LERROR("Using default port %d", po);
        }
      else
        po = ntohs(sv->s_port);
      LINFO("Start. Listening to port %d", po);

      // setup server:
      int sock = socket(AF_INET, SOCK_DGRAM, 0);
      if (sock == -1)
        { PLERROR("Cannot create server socket"); sleep(30); continue; }

      struct sockaddr_in addr;
      addr.sin_family = AF_INET;
      addr.sin_addr.s_addr = htonl(INADDR_ANY); // accept from any host
      addr.sin_port = htons(po);
      if (bind(sock, (struct sockaddr*)(&addr), sizeof(addr)) == -1)
        {
          PLERROR("Cannot bind server socket"); close(sock);
          sleep(30); continue;
        }

      char buf[BUFLEN]; bool running = true;
      while(running)
        {
          // receive a message:
          struct sockaddr_in fromaddr; unsigned int fromlen = sizeof(fromaddr);
          int ret = recvfrom(sock, buf, BUFLEN, 0,
                             (struct sockaddr *)(&fromaddr), &fromlen);
          LINFO("Received message %d bytes from %s:%d", ret,
                inet_ntoa(fromaddr.sin_addr), ntohs(fromaddr.sin_port));

          // Check message format:
          if (strncmp(buf, "JAUS01.0", 8))
            { LERROR("Missing JAUS01.0 header -- SKIPPING"); continue; }

          // and so on...




        }
      // hum, we got messed-up...
      close(sock);
      LERROR("Cooling off for 10 seconds..."); sleep(10);
    }
  exit(1);
}
