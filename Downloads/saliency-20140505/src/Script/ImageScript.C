/*!@file Script/ImageScript.C */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Script/ImageScript.C $
// $Id: ImageScript.C 11876 2009-10-22 15:53:06Z icore $
//

#ifndef SCRIPT_IMAGESCRIPT_C_DEFINED
#define SCRIPT_IMAGESCRIPT_C_DEFINED

#include "Script/ImageScript.H"

#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Image/ShapeOps.H"
#include "Raster/Raster.H"
#include "Util/log.H"

#include "rutz/error.h"
#include "rutz/fstring.h"
#include "rutz/sfmt.h"

#include "tcl/conversions.h"
#include "tcl/list.h"
#include "tcl/obj.h"
#include "tcl/pkg.h"

#include <cstdio>
#include <cstdlib>
#include <tcl.h>

template <class T>
struct ImageObj
{
  static Tcl_ObjType objType;

  static Tcl_ObjType* objTypePtr() { return &objType; }

  static Tcl_Obj* make(const Image<T>& img);

  static void freeInternalRep(Tcl_Obj* objPtr);

  static void dupInternalRep(Tcl_Obj* srcPtr, Tcl_Obj* dupPtr);

  static void updateString(Tcl_Obj* objPtr);

  static int setFromAny(Tcl_Interp* interp, Tcl_Obj* objPtr);
};

template <class T>
Tcl_Obj* ImageObj<T>::make(const Image<T>& img)
{
  Tcl_Obj* objPtr = Tcl_NewObj();
  objPtr->typePtr = &ImageObj<T>::objType;
  objPtr->internalRep.otherValuePtr = new Image<T>(img);

  LDEBUG("new Image: %p", objPtr->internalRep.otherValuePtr);

  Tcl_InvalidateStringRep(objPtr);

  return objPtr;
}

template <class T>
void ImageObj<T>::freeInternalRep(Tcl_Obj* objPtr)
{
  Image<T>* iobj =
    static_cast<Image<T>*>(objPtr->internalRep.otherValuePtr);

  LDEBUG("delete Image: %p", iobj);

  delete iobj;

  objPtr->internalRep.otherValuePtr = 0;
}

template <class T>
void ImageObj<T>::dupInternalRep(Tcl_Obj* srcPtr, Tcl_Obj* dupPtr)
{
  if (dupPtr->typePtr != 0 && dupPtr->typePtr->freeIntRepProc != 0)
    {
      dupPtr->typePtr->freeIntRepProc(dupPtr);
    }

  Image<T>* iobj =
    static_cast<Image<T>*>(srcPtr->internalRep.otherValuePtr);

  LDEBUG("dup Image: %p (src)", iobj);

  dupPtr->internalRep.otherValuePtr =
    static_cast<void*>(new Image<T>(*iobj));

  LDEBUG("dup Image: %p (dst)", dupPtr->internalRep.otherValuePtr);

  dupPtr->typePtr = &ImageObj<T>::objType;
}

template <class T>
void ImageObj<T>::updateString(Tcl_Obj* objPtr)
{
  Image<T>* iobj =
    static_cast<Image<T>*>(objPtr->internalRep.otherValuePtr);

  ASSERT(iobj != 0);

  ASSERT(objPtr->bytes == 0);

  rutz::fstring s =
    rutz::sfmt("{{%s:%dx%d}",
               ImageObj<T>::objType.name,
               iobj->getWidth(), iobj->getHeight());

  LDEBUG("string Image: %p", iobj);

  objPtr->bytes = Tcl_Alloc(s.length()+1);;
  strcpy(objPtr->bytes, s.c_str());
  objPtr->length = s.length();
}

template <class T>
int ImageObj<T>::setFromAny(Tcl_Interp* interp, Tcl_Obj* objPtr)
{
  Tcl_AppendResult(interp, "can't convert to image type");
  return TCL_ERROR;
}

namespace
{
  template <class T>
  tcl::obj image2tcl(Image<T> img) { return ImageObj<T>::make(img); }

  template <class T>
  Image<T> tcl2image(Tcl_Obj* obj)
  {
    if (obj->typePtr == &ImageObj<T>::objType)
      {
        Image<T>* iobj =
          static_cast<Image<T>*>(obj->internalRep.otherValuePtr);
        return *iobj;
      }

    throw rutz::error
      (rutz::sfmt
       ("wrong object type:\n"
        "\t     got: %s\n"
        "\texpected: %s",
        obj->typePtr ? obj->typePtr->name : "(unknown)",
        ImageObj<T>::objType.name),
       SRC_POS);

    /* can't happen */ return Image<T>();
  }
}

tcl::obj tcl::aux_convert_from(Dims d)
{
  tcl::list result;
  result.append(d.w());
  result.append(d.h());
  return result.as_obj();
}

Dims tcl::aux_convert_to(Tcl_Obj* obj, Dims*)
{
  tcl::list l(obj);
  return Dims(l.get<int>(0), l.get<int>(1));
}

#define INST_IMG_OBJ_TYPE(T)                            \
tcl::obj tcl::aux_convert_from(Image< T > img)          \
{ return image2tcl(img); }                              \
                                                        \
Image< T > tcl::aux_convert_to(Tcl_Obj* obj, Image<T>*) \
{ return tcl2image< T >(obj); }                         \
                                                        \
template <>                                             \
Tcl_ObjType ImageObj< T >::objType =                    \
  {                                                     \
    const_cast<char*>("Image<" #T ">"),                 \
    &ImageObj< T >::freeInternalRep,                    \
    &ImageObj< T >::dupInternalRep,                     \
    &ImageObj< T >::updateString,                       \
    &ImageObj< T >::setFromAny                          \
  }


INST_IMG_OBJ_TYPE(byte);
INST_IMG_OBJ_TYPE(float);
INST_IMG_OBJ_TYPE(PixRGB<byte>);
INST_IMG_OBJ_TYPE(PixRGB<float>);

namespace
{
  template <class T>
  Image<T> makeImage(unsigned int w, unsigned int h)
  {
    return Image<T>(w, h, ZEROS);
  }

  template <class T>
  rutz::fstring describeImage(const Image<T>& img)
  {
    return rutz::sfmt("here it is: {%dx%d}",
                      img.getWidth(), img.getHeight());
  }

  template <class T>
  bool imageInitialized(const Image<T>& img)
  {
    return img.initialized();
  }

  Image<byte> readGray(const char* fname)
  { return Raster::ReadGray(fname); }

  Image<float> readFloat(const char* fname)
  { return Raster::ReadFloat(fname); }

  Image<PixRGB<byte> > readRGB(const char* fname)
  { return Raster::ReadRGB(fname); }


  void writeGray(Image<byte> img, const char* fname)
  { Raster::WriteGray(img, fname); }

  void writeFloat(Image<float> img, int flags, const char* fname)
  { Raster::WriteFloat(img, flags, fname); }

  void writeRGB(Image<PixRGB<byte> > img, const char* fname)
  { Raster::WriteRGB(img, fname); }

  template <class T>
  int imgInit(Tcl_Interp* interp, const char* pkgname)
  {
    GVX_PKG_CREATE(pkg, interp, pkgname, "4.$Revision: 1$");

    Tcl_RegisterObjType(&ImageObj<T>::objType);

    pkg->def("make", "w h", &makeImage<T>, SRC_POS);
    pkg->def("describe", "img", &describeImage<T>, SRC_POS);
    pkg->def("initialized", "img", &imageInitialized<T>, SRC_POS);
    pkg->def("zoom", "img xzoom yzoom", &zoomXY<T>, SRC_POS);

    GVX_PKG_RETURN(pkg);
  }
}

extern "C"
int Bimage_Init(Tcl_Interp* interp)
{
  return imgInit<byte>(interp, "BImage");
}

extern "C"
int Fimage_Init(Tcl_Interp* interp)
{
  return imgInit<float>(interp, "FImage");
}

extern "C"
int Cbimage_Init(Tcl_Interp* interp)
{
  return imgInit<PixRGB<byte> >(interp, "CBImage");
}

extern "C"
int Cfimage_Init(Tcl_Interp* interp)
{
  return imgInit<PixRGB<float> >(interp, "CFImage");
}

extern "C"
int Raster_Init(Tcl_Interp* interp)
{
  GVX_PKG_CREATE(pkg, interp, "Raster", "4.$Revision: 1$");

  pkg->def("readGray", "fname", &readGray, SRC_POS);
  pkg->def("readFloat", "fname", &readFloat, SRC_POS);
  pkg->def("readRGB", "fname", &readRGB, SRC_POS);

  pkg->def("writeGray", "img fname", &writeGray, SRC_POS);
  pkg->def("writeFloat", "img flags fname", &writeFloat, SRC_POS);
  pkg->def("writeRGB", "img fname", &writeRGB, SRC_POS);

  GVX_PKG_RETURN(pkg);
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif // SCRIPT_IMAGESCRIPT_C_DEFINED
