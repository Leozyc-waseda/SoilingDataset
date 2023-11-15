/*!@file NeovisionII/nv2-test-server.c */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/NeovisionII/nv2-test-server.c $
// $Id: nv2-test-server.c 8661 2007-08-06 23:45:24Z rjpeters $
//

#ifndef NEOVISIONII_NV2_TEST_SERVER_C_DEFINED
#define NEOVISIONII_NV2_TEST_SERVER_C_DEFINED

#include "NeovisionII/nv2_common.h"
#include "NeovisionII/nv2_label_server.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static char** load_words(const size_t wordlen, size_t* nwords_ret)
{
        const size_t nwords = *nwords_ret;

        char** const result =
                malloc(sizeof(char*) * nwords + nwords * (wordlen + 1));

        char* const wordarea = (char*)(result + nwords);

        for (size_t i = 0; i < nwords; ++i)
        {
                result[i] = wordarea + i * (wordlen + 1);
                result[i][0] = '\0';
        }

        if (result == 0)
        {
                fprintf(stderr, "malloc() failed\n");
                exit(-1);
        }

        FILE* f = fopen("/usr/share/dict/words", "r");

        if (f == 0)
        {
                fprintf(stderr, "couldn't open /usr/share/dict/words\n");
                exit(-1);
        }

        size_t i = 0;

        size_t p = 0;

        while (1)
        {
                int c = getc_unlocked(f);
                if (c == EOF)
                {
                        result[i][p] = '\0';
                        break;
                }
                else if (c == '\n')
                {
                        if (p == wordlen)
                        {
                                result[i][p] = '\0';
                                if (++i >= nwords) break;
                        }

                        p = 0;
                }
                else
                {
                        if (p < wordlen)
                        {
                                result[i][p] = c;
                        }

                        ++p;
                }
        }

        fclose(f);

        *nwords_ret = i;

        fprintf(stderr, "loaded %d words of length %d "
                "from /usr/share/dict/words\n",
                (int) *nwords_ret, (int) wordlen);

        return result;
}

int main (int argc, char* const argv[])
{
        if (argc < 2 || argc > 6)
        {
                fprintf(stderr,
                        "usage: %s ident patch-reader-port=%d "
                        "label-reader-ip-addr=127.0.0.1 "
                        "label-reader-port=%d "
                        "send-interval=1\n",
                        argv[0], NV2_PATCH_READER_PORT,
                        NV2_LABEL_READER_PORT);
                return 0;
        }

        const char* const ident = argv[1];
        const int patch_reader_port =
                argc >= 3 ? atoi(argv[2]) : NV2_PATCH_READER_PORT;
        const char* const label_reader_ip_addr =
                argc >= 4 ? argv[3] : "127.0.0.1";
        const int label_reader_port =
                argc >= 5 ? atoi(argv[4]) : NV2_LABEL_READER_PORT;
        const int send_interval =
                argc >= 6 ? atoi(argv[5]) : 1;

        struct nv2_label_server* s =
                nv2_label_server_create(patch_reader_port,
                                        label_reader_ip_addr,
                                        label_reader_port);


        const size_t wordlen = 13;
        size_t nwords = 40000;

        char** const words = load_words(wordlen, &nwords);

        double confidence = 0.0;

        while (1)
        {
                struct nv2_image_patch p;
                const enum nv2_image_patch_result res =
                        nv2_label_server_get_current_patch(s, &p);

                if (res == NV2_IMAGE_PATCH_END)
                {
                        fprintf(stdout, "ok, quitting\n");
                        break;
                }
                else if (res == NV2_IMAGE_PATCH_NONE)
                {
                        usleep(10000);
                        continue;
                }

                // else... res == NV2_IMAGE_PATCH_VALID

                confidence += 0.05 * drand48();
                if (confidence > 1.0) confidence = 0.0;

                struct nv2_patch_label l;
                l.protocol_version = NV2_LABEL_PROTOCOL_VERSION;
                l.patch_id = p.id;
                l.confidence = (uint32_t)(0.5 + confidence
                                          * (double)(NV2_MAX_LABEL_CONFIDENCE));
                snprintf(l.source, sizeof(l.source), "%s",
                         ident);
                snprintf(l.name, sizeof(l.name), "%s (%ux%u #%u)",
                         words[rand() % nwords],
                         (unsigned int) p.width,
                         (unsigned int) p.height,
                         (unsigned int) p.id);
                snprintf(l.extra_info, sizeof(l.extra_info),
                         "auxiliary information");

                if (l.patch_id % send_interval == 0)
                {
                        nv2_label_server_send_label(s, &l);

                        fprintf(stdout, "sent label '%s (%s)'\n",
                                l.name, l.extra_info);
                }
                else
                {
                        fprintf(stdout, "DROPPED label '%s (%s)'\n",
                                l.name, l.extra_info);
                }

                nv2_image_patch_destroy(&p);
        }

        nv2_label_server_destroy(s);

        return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* c-file-style: "linux" */
/* End: */

#endif // NEOVISIONII_NV2_TEST_SERVER_C_DEFINED
