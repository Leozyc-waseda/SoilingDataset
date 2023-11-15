/*!@file NeovisionII/nv2_label_server.h */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/NeovisionII/nv2_label_server.h $
// $Id: nv2_label_server.h 8376 2007-05-13 03:37:47Z lior $
//

#ifndef NEOVISIONII_NV2_LABEL_SERVER_H_DEFINED
#define NEOVISIONII_NV2_LABEL_SERVER_H_DEFINED

struct nv2_label_server;
struct nv2_image_patch;
struct nv2_patch_label;

#ifdef __cplusplus
#  define NV2_EXTERN_C extern "C"
#else
#  define NV2_EXTERN_C
#endif

/// Create a label server (which will also be a patch reader)
/** @param patch_reader_port tcp port on which the patch reader should
    run on this host machine (by default you should use
    NV2_PATCH_READER_PORT)

    @param remote_label_reader_addr ip address to which we should
    connect to find the label reader

    @param remote_label_reader_port tcp port on which the label reader
    is running on remote_label_reader_addr
*/
NV2_EXTERN_C
struct nv2_label_server* nv2_label_server_create(
        const int patch_reader_port,
        const char* remote_label_reader_addr,
        const int remote_label_reader_port);

/// Destroy a label server and release all resources associated with it
NV2_EXTERN_C
void nv2_label_server_destroy(struct nv2_label_server* p);

/// Result type of nv2_label_server_get_current_patch()
enum nv2_image_patch_result
{
        NV2_IMAGE_PATCH_VALID = 10001, ///< patch is a normal, valid patch
        NV2_IMAGE_PATCH_NONE,          ///< patch is empty, should be ignored
        NV2_IMAGE_PATCH_END            ///< patch stream has ended, program should quit
};

/// Get the most recently received patch
/** @param p handle to the label server from which to retrieve the
    patch

    @param ret the returned image patch; NOTE that ownership of the
    nv2_image_patch object is transferred to the caller here, so that
    the caller is responsible for freeing resources associated with
    this image patch by calling nv2_image_patch_destroy() after use
*/
NV2_EXTERN_C
enum nv2_image_patch_result
nv2_label_server_get_current_patch(struct nv2_label_server* p,
                                   struct nv2_image_patch* ret);

/// Result type of nv2_label_server_send_label()
enum nv2_label_send_result
{
        NV2_LABEL_SEND_OK = 12345, ///< label was sent succesfully
        NV2_LABEL_SEND_FAIL,       ///< label was not sent (maybe server is down?)
};

/// Send a patch label
/** @param p handle to the label server from which to send the patch
    label

    @param l patch label to be sent

    @return non-zero if the label was sent succesfully, or zero if
    there was an error
*/
NV2_EXTERN_C
enum nv2_label_send_result
nv2_label_server_send_label(struct nv2_label_server* p,
                            const struct nv2_patch_label* l);

/// Set the verbosity level for the label server
/** @param verbosity if 0, then don't print anything except fatal
    error messages; if 1, then also print warnings (but not
    informational messages) */
NV2_EXTERN_C
void nv2_label_server_set_verbosity(struct nv2_label_server* p,
                                    const int verbosity);

#undef NV2_EXTERN_C

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* c-file-style: "linux" */
/* End: */

#endif // NEOVISIONII_NV2_LABEL_SERVER_H_DEFINED
