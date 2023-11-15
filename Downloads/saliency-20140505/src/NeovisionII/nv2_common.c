/*!@file NeovisionII/nv2_common.c */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/NeovisionII/nv2_common.c $
// $Id: nv2_common.c 9496 2008-03-19 04:15:58Z rjpeters $
//

#ifndef NEOVISIONII_NV2_COMMON_C_DEFINED
#define NEOVISIONII_NV2_COMMON_C_DEFINED

#include "NeovisionII/nv2_common.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h> // for strerror()
#include <unistd.h>

uint32_t nv2_pixel_type_bytes_per_pixel(enum nv2_pixel_type typ)
{
        switch (typ)
        {
        case NV2_PIXEL_TYPE_NONE: return 0;
        case NV2_PIXEL_TYPE_GRAY8: return 1;
        case NV2_PIXEL_TYPE_RGB24: return 3;
        default: break;
        }

        errno = 0;
        nv2_fatal("invalid pixel type value");
        /* can't happen */ return ((uint32_t) -1);
}

void nv2_image_patch_init_empty(struct nv2_image_patch* p)
{
        p->protocol_version = NV2_PATCH_PROTOCOL_VERSION;
        p->width = 0;
        p->height = 0;
        p->id = 0;
        p->is_training_image = 0;
        p->type = NV2_PIXEL_TYPE_NONE;
        p->training_label[sizeof(p->training_label)-1] = '\0';
        p->remote_command[sizeof(p->remote_command)-1] = '\0';
        p->data = 0;
}

void nv2_image_patch_destroy(struct nv2_image_patch* p)
{
        if (p->data)
                free(p->data);

        nv2_image_patch_init_empty(p);
}

void nv2_image_patch_set_training_label(
        struct nv2_image_patch* p,
        const char* label)
{
        strncpy(&p->training_label[0], label,
                sizeof(p->training_label));
        p->training_label[sizeof(p->training_label)-1] = '\0';
}

void nv2_image_patch_set_remote_command(
        struct nv2_image_patch* p,
        const char* command)
{
        strncpy(&p->remote_command[0], command,
                sizeof(p->remote_command));
        p->remote_command[sizeof(p->remote_command)-1] = '\0';
}

void nv2_patch_label_init_empty(struct nv2_patch_label* l)
{
        l->protocol_version = NV2_LABEL_PROTOCOL_VERSION;
        l->patch_id = 0;
        l->confidence = 0;
        l->source[0] = '\0';
        l->name[0] = '\0';
        l->extra_info[0] = '\0';
}

void nv2_fatal_impl(const char* file, int line, const char* function,
                    const char* what)
{
        if (errno != 0)
                fprintf(stderr, "%s:%d(%s): error: %s (%s)\n",
                        file, line, function, what, strerror(errno));
        else
                fprintf(stderr, "%s:%d(%s): error: %s\n",
                        file, line, function, what);
        exit(-1);
}

void nv2_warn_impl(const char* file, int line, const char* function,
                   const char* what)
{
        if (errno != 0)
                fprintf(stderr, "%s:%d(%s): warning: %s (%s)\n",
                        file, line, function, what, strerror(errno));
        else
                fprintf(stderr, "%s:%d(%s): warning: %s\n",
                        file, line, function, what);
}

size_t nv2_robust_write(int fd, const void* const data,
                        size_t const nbytes)
{
        size_t offset = 0;

        while (offset < nbytes)
        {
                errno = 0;
                const ssize_t n =
                        write(fd, data + offset,
                              nbytes - offset);

                if (errno == EINTR || errno == EAGAIN)
                {
                        // just try the same write() call again
                        continue;
                }
                else if (n < 0)
                {
                        // there was some error, so just return the
                        // current offset value; the caller can tell
                        // that an error occurred because offset will
                        // be less than the requested nbytes
                        break;
                }
                else
                {
                        offset += n;
                }
        }

        return offset;
}

size_t nv2_robust_read(int fd, void* buf, size_t nbytes,
                       int* nchunks_return)
{
        int nchunks = 0;

        size_t offset = 0;
        while (offset < nbytes)
        {
                errno = 0;
                const ssize_t n =
                        read(fd, buf + offset,
                             nbytes - offset);

                if (errno == EINTR || errno == EAGAIN)
                {
                        // just try the same read() call again
                        continue;
                }
                else if (n < 0)
                {
                        // there was some error, so just return the
                        // current offset value; the caller can tell
                        // that an error occurred because offset will
                        // be less than the requested nbytes
                        break;
                }
                else
                {
                        offset += n;
                        ++nchunks;
                }
        }

        if (nchunks_return != NULL)
                *nchunks_return = nchunks;

        return offset;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* c-file-style: "linux" */
/* End: */

#endif // NEOVISIONII_NV2_COMMON_C_DEFINED
