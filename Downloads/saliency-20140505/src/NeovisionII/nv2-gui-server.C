/*!@file NeovisionII/nv2-gui-server.C sample neovision2 label server that sends its image patches to an OutputFrameSeries */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/NeovisionII/nv2-gui-server.C $
// $Id: nv2-gui-server.C 15310 2012-06-01 02:29:24Z itti $
//

#ifndef NEOVISIONII_NV2_GUI_SERVER_C_DEFINED
#define NEOVISIONII_NV2_GUI_SERVER_C_DEFINED

#include "Component/ModelManager.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Media/FrameSeries.H"
#include "NeovisionII/nv2_common.h"
#include "NeovisionII/nv2_label_server.h"
#include "Util/sformat.H"

#include <cstdio>
#include <fstream>
#include <string>
#include <unistd.h> // for usleep()
#include <vector>

void load_words2(const size_t wordlen, std::vector<std::string>& words)
{
  std::ifstream ifs("/usr/share/dict/words");

  if (!ifs.is_open())
    LFATAL("couldn't open /usr/share/dict/words");

  int c = 0;
  std::string line;
  while (std::getline(ifs, line))
    if (line.length() == wordlen)
      {
        words.push_back(line);
        ++c;
      }

  LINFO("loaded %d words of length %" ZU " from /usr/share/dict/words\n",
        c, wordlen);
}

int main (int argc, char* const argv[])
{
  ModelManager manager("nv2-gui-server");

  nub::ref<OutputFrameSeries> ofs(new OutputFrameSeries(manager));
  manager.addSubComponent(ofs);

  if (manager.parseCommandLine(argc, argv,
                               sformat("ident patch-reader-port=%d "
                                       "label-reader-ip-addr=127.0.0.1 "
                                       "label-reader-port=%d "
                                       "send-interval=1 "
                                       "do-send-labels=1",
                                       NV2_PATCH_READER_PORT,
                                       NV2_LABEL_READER_PORT).c_str(),
                               1, 6) == false)
    return 1;

  const std::string ident = manager.getExtraArg(0);
  const int patch_reader_port =
    manager.numExtraArgs() >= 2 ? manager.getExtraArgAs<int>(1) : NV2_PATCH_READER_PORT;
  const std::string label_reader_ip_addr =
    manager.numExtraArgs() >= 3 ? manager.getExtraArg(2) : std::string("127.0.0.1");
  const int label_reader_port =
    manager.numExtraArgs() >= 4 ? manager.getExtraArgAs<int>(3) : NV2_LABEL_READER_PORT;
  const int send_interval =
    manager.numExtraArgs() >= 5 ? manager.getExtraArgAs<int>(4) : 1;
  const bool do_send_labels =
    manager.numExtraArgs() >= 6 ? manager.getExtraArgAs<bool>(5) : true;

  manager.start();

  struct nv2_label_server* s =
    nv2_label_server_create(patch_reader_port,
                            label_reader_ip_addr.c_str(),
                            label_reader_port);


  const size_t wordlen = 13;
  std::vector<std::string> words;

  if (do_send_labels)
    load_words2(wordlen, words);

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

      ofs->updateNext();

      if (p.type == NV2_PIXEL_TYPE_GRAY8)
        {
          const Image<byte> im((const byte*) p.data,
                               p.width, p.height);

          ofs->writeGray(im, "gray8-on-label-server");
        }
      else if (p.type == NV2_PIXEL_TYPE_RGB24)
        {
          const Image<PixRGB<byte> > im((const PixRGB<byte>*) p.data,
                                        p.width, p.height);

          ofs->writeRGB(im, "rgb24-on-label-server");
        }

      if (do_send_labels)
        {
          confidence += 0.05 * drand48();
          if (confidence > 1.0) confidence = 0.0;

          struct nv2_patch_label l;
          l.protocol_version = NV2_LABEL_PROTOCOL_VERSION;
          l.patch_id = p.id;
          l.confidence = (uint32_t)(0.5 + confidence
                                    * double(NV2_MAX_LABEL_CONFIDENCE));
          snprintf(l.source, sizeof(l.source), "%s",
                   ident.c_str());
          snprintf(l.name, sizeof(l.name), "%s (%ux%u #%u)",
                   words.at(rand() % words.size()).c_str(),
                   (unsigned int) p.width,
                   (unsigned int) p.height,
                   (unsigned int) p.id);
          snprintf(l.extra_info, sizeof(l.extra_info),
                   "auxiliary information");

          if (l.patch_id % send_interval == 0)
            {
              nv2_label_server_send_label(s, &l);
              LINFO("sent label '%s (%s)'\n", l.name, l.extra_info);
            }
          else
            {
              LINFO("DROPPED label '%s (%s)'\n", l.name, l.extra_info);
            }
        }

      nv2_image_patch_destroy(&p);
    }

  nv2_label_server_destroy(s);

  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // NEOVISIONII_NV2_GUI_SERVER_C_DEFINED
