/*!@file NeovisionII/nv2_common.h */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/NeovisionII/nv2_common.h $
// $Id: nv2_common.h 9742 2008-05-11 00:41:01Z lior $
//

#ifndef NEOVISIONII_NV2_COMMON_H_DEFINED
#define NEOVISIONII_NV2_COMMON_H_DEFINED

#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
#  define NV2_EXTERN_C extern "C"
#else
#  define NV2_EXTERN_C
#endif

/// Different pixel types that can be stored in an nv2_image_patch
enum nv2_pixel_type
{
        NV2_PIXEL_TYPE_NONE  = 5000,
        NV2_PIXEL_TYPE_GRAY8 = 5001,
        NV2_PIXEL_TYPE_RGB24 = 5002
};

/// Get the number of bytes used per pixel of a given type
uint32_t nv2_pixel_type_bytes_per_pixel(enum nv2_pixel_type typ);

/// Represents the current version of struct nv2_image_patch
/** Make sure that any struct nv2_image_patch objects to be sent
    across the network have this value assigned to their
    protocol_version field, and make sure that any struct
    nv2_image_patch objects received from the network have the proper
    protocol_version before using the other data fields.
*/
#define NV2_PATCH_PROTOCOL_VERSION ((uint32_t) 1004)

/// Represents an image patch, with image data stored in row-major fashion
struct nv2_image_patch
{
        uint32_t protocol_version;
        uint32_t width;
        uint32_t height;
        uint32_t id;
        uint32_t is_training_image;
        uint32_t type;
        uint32_t fix_x;
        uint32_t fix_y;
        char training_label[128];
        char remote_command[128];
        unsigned char* data; // this field must remain last
};

#define NV2_IMAGE_PATCH_HEADER_SIZE \
        (sizeof(struct nv2_image_patch) - sizeof(unsigned char*))

/// Initialize with the current protocol version, and zeros elsewhere
/** This function does NOT free any data associated with p->data; it
    is up to the caller to do that (if necessary) before calling this
    function. */
NV2_EXTERN_C
void nv2_image_patch_init_empty(struct nv2_image_patch* p);

/// Free the data array associated with p, and reinitialize to an empty state
NV2_EXTERN_C
void nv2_image_patch_destroy(struct nv2_image_patch* p);

/// Copy a training label into the image patch
NV2_EXTERN_C
void nv2_image_patch_set_training_label(
        struct nv2_image_patch* p,
        const char* label);

/// Copy a remote command into the image patch
NV2_EXTERN_C
void nv2_image_patch_set_remote_command(
        struct nv2_image_patch* p,
        const char* command);

/// Represents the current version of struct nv2_patch_label
/** Be sure to assign the proper protocol_version before sending any
    nv2_patch_label objects across the network, and be sure to check
    the protocol_version field of any nv2_patch_label objects received
    from the network before using any of the other data fields.
*/
#define NV2_LABEL_PROTOCOL_VERSION ((uint32_t) 2004)

/// Represents a label for an image patch
/** The patch_id indicates which nv2_image_patch this label
    corresponds to, and the name field contains a human-readable label
    for the patch.
*/
struct nv2_patch_label
{
        uint32_t protocol_version;
        uint32_t patch_id;
        uint32_t confidence; ///< min = 0; max = NV2_MAX_LABEL_CONFIDENCE
        char source[16];
        char name[64];
        char extra_info[64];
};

/// The scaling factor used to represent confidence levels in nv2_patch_label
/** A raw confidence value in [0,1] should be encoded as
    uint32_t(0.5 + raw*NV2_MAX_LABEL_CONFIDENCE) */
#define NV2_MAX_LABEL_CONFIDENCE 10000u

/// Initialize with the current protocol version, and zeros elsewhere
NV2_EXTERN_C
void nv2_patch_label_init_empty(struct nv2_patch_label* l);

/// Helper for nv2_fatal
NV2_EXTERN_C
void nv2_fatal_impl(const char* file, int line, const char* function,
                    const char* what);

/// Print an error message to stderr and exit
#define nv2_fatal(what) nv2_fatal_impl(__FILE__, __LINE__, __FUNCTION__, what)

/// Helper for nv2_warn
NV2_EXTERN_C
void nv2_warn_impl(const char* file, int line, const char* function,
                   const char* what);

/// Print a warning message to stderr
#define nv2_warn(what) nv2_warn_impl(__FILE__, __LINE__, __FUNCTION__, what)

/// Keep doing write() calls until either all the bytes are written or an error occurs
/** Returns the number of bytes written; if this number is less than
    the number of requested bytes (nbytes), then it means that some
    error occurred */
size_t nv2_robust_write(int fd, const void* data, size_t nbytes);

/// Keep doing read() calls until either all the bytes are read or an error occurs
/** Returns the number of bytes read; if this number is less than the
    number of requested bytes (nbytes), then it means that some error
    occurred

    @param nchunks if not null, then the total number of read() system
    calls performed will be returned through this pointer
*/
size_t nv2_robust_read(int fd, void* buf, size_t nbytes,
                       int* nchunks);

/// Default tcp port number for the patch reader
#define NV2_PATCH_READER_PORT 9930

/// Default tcp port number for the label reader
#define NV2_LABEL_READER_PORT 9931

#undef NV2_EXTERN_C

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* c-file-style: "linux" */
/* End: */

#endif // NEOVISIONII_NV2_COMMON_H_DEFINED
