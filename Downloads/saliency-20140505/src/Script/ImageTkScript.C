/*!@file Script/ImageTkScript.C */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Script/ImageTkScript.C $
// $Id: ImageTkScript.C 11876 2009-10-22 15:53:06Z icore $
//

#ifndef SCRIPT_IMAGETKSCRIPT_C_DEFINED
#define SCRIPT_IMAGETKSCRIPT_C_DEFINED

#include "Script/ImageTkScript.H"

#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Script/ImageScript.H"
#include "Util/Assert.H"
#include "Util/Types.H"
#include "Util/log.H"
#include "tcl/interp.h"
#include "tcl/pkg.h"

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/extensions/XShm.h>
#include <stdlib.h>
#include <sys/shm.h>
#include <tk.h>

namespace
{
  int fourByteAlign(int v)
  {
    const int remainder = v % 4;
    return (remainder == 0) ? v : v + (4-remainder);
  }
}

template <class T>
struct ImageTk
{
  // typedef struct Tk_ImageType {
  //     char *name;
  //     Tk_ImageCreateProc *createProc;
  //     Tk_ImageGetProc *getProc;
  //     Tk_ImageDisplayProc *displayProc;
  //     Tk_ImageFreeProc *freeProc;
  //     Tk_ImageDeleteProc *deleteProc;
  // } Tk_ImageType;

  static Tk_ImageType imageType;

  struct Master
  {
    Tk_ImageMaster token;
    Image<T>       image;
  };

  struct Instance
  {
    Master*         const master;
    Tk_Window       const tkwin;
    GC                    gc;   // current graphic context
    XImage*               image; // pointer to image data
    XShmSegmentInfo       shminfo; // info on shared memory segment
    char*                 buffer; // shared memory buffer
    bool                  shm_buffer; // true if we can use shared memory

    Instance(Tk_Window tkw, Master* m)
      :
      master(m),
      tkwin(tkw),
      gc(0),
      image(0)
    {
      const Image<PixRGB<byte> > img = m->image;

      const Dims idims = img.getDims();

      // get default screen:
      int byt_per_pix = 4;
      if (Tk_Depth(tkwin) == 16) byt_per_pix = 2;

      // alloc XImage in shared memory if posible (for faster display):
      int Shm_OK = XShmQueryExtension(Tk_Display(tkwin));

      // check if remote display, and disable shared memory if so:
      char* disp = getenv("DISPLAY");
      if (disp != NULL && disp[0] != ':')
        {
          Shm_OK = 0;
          LDEBUG("Not using shared memory for remote display");
        }

      // Get a padded row width that is 4-byte aligned
      const int padded_width = fourByteAlign(idims.w());

      // allocate an image buffer of the size of the window:
      if (Shm_OK)
        {
          shminfo.shmid = shmget(IPC_PRIVATE,
                                 padded_width*idims.h()*byt_per_pix,
                                 IPC_CREAT | 0777);
          buffer = (char *)shmat(shminfo.shmid, NULL, 0);  // attach shared memory
          if (buffer == 0) LFATAL("Cannot get shared memory");
          shminfo.shmaddr = buffer;        // link buffer to shminfo structure
          shminfo.readOnly = False;
          XShmAttach(Tk_Display(tkwin), &shminfo);   // now attach to X
          shm_buffer = true;
        }
      else
        {
          const int bufsiz = padded_width * idims.h() * byt_per_pix;
          buffer = (char *)malloc(bufsiz);
          LDEBUG("buffer: %p", buffer);
          if (buffer == NULL) LFATAL("Cannot allocate image buffer");
          shm_buffer = false;
        }

      uint idepth;
      int pad;

      // make sure we use the correct depth values
      if (byt_per_pix == 2)
        { idepth = 16; pad = 16;}
      else
        { idepth = 24; pad = 32;}

      if (shm_buffer)
        // ok for shared memory; X will allocate the buffer:
        image = XShmCreateImage(Tk_Display(tkwin), Tk_Visual(tkwin), idepth, ZPixmap,
                                buffer, &shminfo, idims.w(), idims.h());
      else
        // cannot alloc in shared memory... do conventional alloc
        image = XCreateImage(Tk_Display(tkwin), Tk_Visual(tkwin), idepth,
                             ZPixmap , 0, buffer,
                             idims.w(), idims.h(), pad,
                             fourByteAlign(byt_per_pix * idims.w()));

      int w = idims.w(); int h = idims.h();

      const byte* im = reinterpret_cast<const byte*>(img.getArrayPtr());
      byte* bu = reinterpret_cast<byte*>(buffer);

      if (byt_per_pix == 2)
        {
          const int bytes_per_row = fourByteAlign(byt_per_pix * w);

          // 16 bit format: 565, lowest byte first
          for (int j = 0; j < h; ++j)
            {
              byte* bu_row = bu + j*bytes_per_row;
              for (int i = 0; i < w; ++i)
                {
                  *bu_row++ = ((im[1] & 0x1C)<<3) | ((im[2] & 0xF8)>>3);
                  *bu_row++ = ((im[1] & 0xE0)>>5) | (im[0] & 0xF8);
                  im += 3;
                }
            }
        }
      else
        {
          // 24 bit format with extra byte for padding
          for (int i = 0; i < w * h; i ++)
            { *bu++ = im[2]; *bu++ = im[1]; *bu++ = *im; *bu++ = 255; im += 3; }
        }
    }

    ~Instance()
    {
      if (this->shm_buffer)
        {
          XShmDetach(Tk_Display(this->tkwin), &this->shminfo);
        }

      if (this->image)
        {
          // We have to set the XImage's data to zero before
          // destroying the XImage, because otherwise the XImage will
          // want to free the data as it is destroyed, which means
          // that our 'buffer' would become a dangling pointer, and
          // that we'd be double-free'ing it when we try to free() it
          // ourselves later.
          this->image->data = 0;
          XDestroyImage(this->image);
          this->image = 0;
        }

      if (this->shm_buffer)
        {
          shmdt(this->shminfo.shmaddr);
          shmctl(this->shminfo.shmid, IPC_RMID, NULL);
        }
      else
        {
          free(this->buffer);
        }

      if (this->gc)
        XFreeGC(Tk_Display(this->tkwin), this->gc);
    }

    void draw(Display* display,
              Drawable drawable,
              int imageX, int imageY,
              int width, int height,
              int drawableX, int drawableY)
    {
      if (this->gc == 0)
        this->gc = XCreateGC(display, drawable, 0, NULL);

      if (this->shm_buffer)
        XShmPutImage(display, drawable, this->gc, this->image,
                     imageX, imageY, drawableX, drawableY,
                     width, height, 0);
      else
        XPutImage(display, drawable, this->gc, this->image,
                  imageX, imageY, drawableX, drawableY,
                  width, height);
      XFlush(display);
    }
  };

#ifdef INVT_NEWER_TCL_VERSION
  static int createMaster(Tcl_Interp* intp,
                          const char* name,
                          int objc,
                          Tcl_Obj* CONST objv[],
                          const Tk_ImageType* typePtr,
                          Tk_ImageMaster token,
                          ClientData* masterDataPtr)
#else
  static int createMaster(Tcl_Interp* intp,
                          /* should be: const */ char* name,
                          int objc,
                          Tcl_Obj* CONST objv[],
                          /* should be: const */ Tk_ImageType* typePtr,
                          Tk_ImageMaster token,
                          ClientData* masterDataPtr)
#endif
  {
    tcl::interpreter interp(intp);

    try
      {
        if (objc != 1)
          {
            Tcl_AppendResult
              (intp, "wrong # args: expected\n"
               "\timage create Image<type> name img", 0);
            return TCL_ERROR;
          }

        const Image<T> image = tcl::convert_to<Image<T> >(objv[0]);

        Master* master = new Master;

        master->token = token;
        master->image = image;

        *masterDataPtr = static_cast<void*>(master);

        Tk_ImageChanged(token, 0, 0,
                        master->image.getWidth(),
                        master->image.getHeight(),
                        master->image.getWidth(),
                        master->image.getHeight());

        return TCL_OK;
      }
    catch (...)
      {
        interp.handle_live_exception(imageType.name, SRC_POS);
      }
    return TCL_ERROR;
  }

  static ClientData getInstance(Tk_Window tkwin,
                                ClientData masterData)
  {
    return static_cast<void*>
      (new Instance(tkwin, static_cast<Master*>(masterData)));
  }

  static void displayInstance(ClientData instanceData,
                              Display* display,
                              Drawable drawable,
                              int imageX,
                              int imageY,
                              int width,
                              int height,
                              int drawableX,
                              int drawableY)
  {
    LDEBUG("image x=%d, y=%d", imageX, imageY);
    LDEBUG("image w=%d, h=%d", width, height);
    LDEBUG("drawable x=%d, y=%d", drawableX, drawableY);

    Instance* instance = static_cast<Instance*>(instanceData);

    instance->draw(display, drawable, imageX, imageY,
                   width, height, drawableX, drawableY);
  }

  static void freeInstance(ClientData instanceData,
                           Display* display)
  {
    delete static_cast<Instance*>(instanceData);
  }

  static void deleteMaster(ClientData masterData)
  {
    delete static_cast<Master*>(masterData);
  }
};

#ifdef INVT_NEWER_TCL_VERSION

#define INST_IMG_TK_TYPE(T)                     \
                                                \
template <>                                     \
Tk_ImageType ImageTk< T >::imageType =          \
  {                                             \
    "Image<" #T ">",                            \
    &ImageTk< T >::createMaster,                \
    &ImageTk< T >::getInstance,                 \
    &ImageTk< T >::displayInstance,             \
    &ImageTk< T >::freeInstance,                \
    &ImageTk< T >::deleteMaster                 \
  }
#else
#define INST_IMG_TK_TYPE(T)                     \
                                                \
template <>                                     \
Tk_ImageType ImageTk< T >::imageType =          \
  {                                             \
    /*FIXME*/ const_cast<char*>("Image<" #T ">"),       \
    &ImageTk< T >::createMaster,                \
    &ImageTk< T >::getInstance,                 \
    &ImageTk< T >::displayInstance,             \
    &ImageTk< T >::freeInstance,                \
    &ImageTk< T >::deleteMaster                 \
  }
#endif

INST_IMG_TK_TYPE(byte);
INST_IMG_TK_TYPE(float);
INST_IMG_TK_TYPE(PixRGB<byte>);
INST_IMG_TK_TYPE(PixRGB<float>);


extern "C"
int Imagetk_Init(Tcl_Interp* interp)
{
  GVX_PKG_CREATE(pkg, interp, "ImageTk", "4.$Revision: 1$");

  Tk_CreateImageType(&ImageTk<byte>::imageType);
  Tk_CreateImageType(&ImageTk<float>::imageType);
  Tk_CreateImageType(&ImageTk<PixRGB<byte> >::imageType);
  Tk_CreateImageType(&ImageTk<PixRGB<float> >::imageType);

  GVX_PKG_RETURN(pkg);
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif // SCRIPT_IMAGETKSCRIPT_C_DEFINED
