/*!@file NeovisionII/nv2-test-client.c */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/NeovisionII/nv2-test-client.c $
// $Id: nv2-test-client.c 8362 2007-05-12 16:34:40Z rjpeters $
//

#ifndef NEOVISIONII_NV2_TEST_CLIENT_C_DEFINED
#define NEOVISIONII_NV2_TEST_CLIENT_C_DEFINED

#include "NeovisionII/nv2_common.h"
#include "NeovisionII/nv2_label_reader.h"

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

int main (int argc, char* const argv[])
{
        if (argc < 4)
        {
                fprintf(stderr, "usage: %s dest-ip-addr imagesize nimages\n",
                        argv[0]);
                return 0;
        }

        const char* const dest_ip_addr = argv[1];
        const int imagesize = atoi(argv[2]);
        const int nimages = atoi(argv[3]);

        if (imagesize <= 0)
        {
                fprintf(stderr, "expected imagesize>0, "
                        "but got imagesize=%d\n", imagesize);
        }

        srand(time(0) + getpid() * 345);

        struct nv2_label_reader* reader =
                nv2_label_reader_create(NV2_LABEL_READER_PORT,
                                        dest_ip_addr,
                                        NV2_PATCH_READER_PORT);

        for (int i = 0; i < nimages; ++i)
        {
                struct nv2_image_patch p;
                p.protocol_version = NV2_PATCH_PROTOCOL_VERSION;
                p.width = imagesize;
                p.height = imagesize;
                p.id = (uint32_t) i;
                p.type = NV2_PIXEL_TYPE_GRAY8;
                p.data = (unsigned char*) malloc(p.width * p.height
                                                 * sizeof(unsigned char));

                for (uint32_t i = 0; i < p.width * p.height; ++i)
                        p.data[i] = rand() % 256;

                nv2_label_reader_send_patch(reader, &p);

                usleep(100000);

                struct nv2_patch_label l;
                if (nv2_label_reader_get_current_label(reader, &l))
                {
                        fprintf(stderr,
                                "got patch label: id=%u, "
                                "source=%s, name=%s, extra_info=%s\n",
                                (unsigned int) l.patch_id,
                                l.source, l.name, l.extra_info);
                }
        }

        nv2_label_reader_send_patch(reader, 0);

        nv2_label_reader_destroy(reader);

        return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* c-file-style: "linux" */
/* End: */

#endif // NEOVISIONII_NV2_TEST_CLIENT_C_DEFINED
