/*!@file NeovisionII/nv2_label_reader.h */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/NeovisionII/nv2_label_reader.h $
// $Id: nv2_label_reader.h 8125 2007-03-16 04:33:29Z rjpeters $
//

#ifndef NEOVISIONII_NV2_LABEL_READER_H_DEFINED
#define NEOVISIONII_NV2_LABEL_READER_H_DEFINED

#include <stddef.h>
#include <stdint.h>

struct nv2_label_reader;
struct nv2_image_patch;
struct nv2_patch_label;

#ifdef __cplusplus
#  define NV2_EXTERN_C extern "C"
#else
#  define NV2_EXTERN_C
#endif

/// Create a label reader (which will also be a patch server)
/** @param label_reader_port tcp port on which the label reader should
    run on this host machine (by default, you should use
    NV2_LABEL_READER_PORT)

    @param remote_patch_reader_addr ip address to which we should
    connect to find the patch reader

    @param remote_patch_reader_port tcp port on which the patch reader
    is running on dest_addr
 */
NV2_EXTERN_C
struct nv2_label_reader* nv2_label_reader_create(
        const int label_reader_port,
        const char* remote_patch_reader_addr,
        const int remote_patch_reader_port);

/// Destroy a label reader and release all resources associated with it
NV2_EXTERN_C
void nv2_label_reader_destroy(struct nv2_label_reader* p);

/// Get the most recently received patch label, if any
/** @param l if new label is found, it will be copied here

    @return non-zero if a label is found, or zero if no label has been
    received since the last time this function was called
 */
NV2_EXTERN_C
int nv2_label_reader_get_current_label(struct nv2_label_reader* p,
                                       struct nv2_patch_label* l);

/// Result type of nv2_label_reader_send_patch()
enum nv2_patch_send_result
{
        NV2_PATCH_SEND_OK = 23456, ///< patch was sent succesfully
        NV2_PATCH_SEND_FAIL,       ///< patch was not sent (maybe server is down?)
};

/// Send an image patch from the patch server asynchronously
/** @param r handle to the patch server from which to send the patch

    @param p image patch to be sent; note the patch server assumes
    ownership of this image patch and will free memory associated with
    it, so (1) the memory in p->data should have be allocated with
    malloc() so that it can be released with free(), and (2) the patch
    pointed to by p will be an empty patch on return from this
    function (reflecting the fact that ownership of the original patch
    is now transferred to the patch server)
 */
NV2_EXTERN_C
void
nv2_label_reader_send_patch(struct nv2_label_reader* r,
                            struct nv2_image_patch* p);

/// Send an image patch from the patch server synchronously
/** This is the work function that is called from a background thread
    to implement the asynchronous nv2_label_reader_send_patch().

    @param show_warnings whether or not to print warnings if the patch
    send fails
*/
NV2_EXTERN_C
enum nv2_patch_send_result
nv2_label_reader_send_patch_sync(struct nv2_label_reader* r,
                                 struct nv2_image_patch* p,
                                 const int show_warnings);

#undef NV2_EXTERN_C

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* c-file-style: "linux" */
/* End: */

#endif // NEOVISIONII_NV2_LABEL_READER_H_DEFINED
