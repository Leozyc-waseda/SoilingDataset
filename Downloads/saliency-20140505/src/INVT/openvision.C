/*!@file INVT/openvision.C  version of ezvision.C that uses on-file color
  filters */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2001 by the //
// University of Southern California (USC) and the iLab at USC.         //
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
// Primary maintainer for this file: Laurent Itti <itti@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/INVT/openvision.C $
// $Id: openvision.C 10845 2009-02-13 08:49:12Z itti $
//

#include "Channels/RGBConvolveChannel.H"
#include "Component/ModelManager.H"
#include "Image/ColorOps.H"
#include "Image/MathOps.H"
#include "Image/MatrixOps.H"
#include "Image/Pixels.H"
#include "Image/Transforms.H"
#include "Media/FrameSeries.H"
#include "Neuro/NeuroOpts.H"
#include "Channels/RawVisualCortex.H"
#include "Raster/Raster.H"
#include "Util/sformat.H"

#include <fstream>

#define NB_FILTERS  3
#define NB_COEFFS 8

int main(const int argc, const char **argv)
{
  int n = NB_COEFFS;
  int m = NB_COEFFS * NB_COEFFS * NB_FILTERS * 3;

  MYLOGVERB = LOG_INFO;  // Suppress debug messages

  // Generate the haar transform matrix:
  Image<float> hmat(n, n, ZEROS);
  for(int i = 0; i < n; i++)
    {
      hmat.setVal(i, 0, 1.0f);
    }
  for(int i = 0; i < n / 2; i++)
    {
      hmat.setVal(i, 1, 1.0f);
      hmat.setVal(i + n / 2, 1, -1.0f);
      if (i - 2 < 0)
        {
          hmat.setVal(i, 2, 1.0f);
          hmat.setVal(i + 2, 2, -1.0f);
        }
      else
        {
          hmat.setVal(i + 2, 3, 1.0f);
          hmat.setVal(i + 4, 3, -1.0f);
        }
      hmat.setVal(2 * i, i + n / 2, 1.0f);
      hmat.setVal(2 * i + 1, i + n / 2, -1.0f);
    }

  // Instantiate a ModelManager:
  ModelManager manager("Open Attention Model");

  // Instantiate our various ModelComponents:
  nub::soft_ref<RawVisualCortex> vcx(new RawVisualCortex(manager));
  manager.addSubComponent(vcx);

  // let's make one dummy RGBConvolveChannel so that we get the
  // command-line options for it:
  nub::soft_ref<RGBConvolveChannel> channel(new RGBConvolveChannel(manager));
  vcx->addSubChan(channel);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "<data.txt> <image.ppm>",
                               2, -1) == false)
    return(1);

  // Ok, get rid of our placeholder channel; the manager will keep a
  // trace of its configured options:
  vcx->removeAllSubChans();

  // Get the input image name:
  char framename[1024];
  strncpy(framename, manager.getExtraArg(1).c_str(), 1023);

  // Load data:
  float data[m];
  char dataname[1024];
  strncpy(dataname, manager.getExtraArg(0).c_str(), 1023);
  std::ifstream inputfile (dataname);
  if (inputfile.is_open())
    {
      for (int j = 0; j < m; j++)
        inputfile >> data[j];
      inputfile.close();
    }
  else
    {
      LERROR("*** Cannot open input file !");
      return 1;
    }

  // Convert data into filters:
  ImageSet<float> trans(NB_FILTERS * 3);
  for (int i = 0; i < NB_FILTERS * 3; i++)
    trans[i] = Image<float>(data + (n * n * i), n, n);
  ImageSet<float> filter(NB_FILTERS * 3);
  Dims filterdim(8, 8);
  for (int i = 0; i < NB_FILTERS * 3; i++)
    filter[i] = scaleBlock(matrixMult(transpose(hmat),
                                      matrixMult(trans[i], hmat)),
                           filterdim);

  for (int i = 0; i < NB_FILTERS; i++)
    {
      Image<float> rf = filter[3 * i];
      Image<float> gf = filter[3 * i + 1];
      Image<float> bf = filter[3 * i + 2];
      float min, max, fmax;
      getMinMax(rf, min, max);
      if (fabs(min) > fabs(max))
        fmax = min;
      else
        fmax = max;
      getMinMax(gf, min, max);
      if (fabs(min) > fmax)
        fmax = min;
      if (fabs(max) > fmax)
        fmax = max;
      getMinMax(bf, min, max);
      if (fabs(min) > fmax)
        fmax = min;
      if (fabs(max) > fmax)
        fmax = max;
      if (fmax < 1.0e-10F)
        fmax = 1; // images are uniform
      float scale = 128.0F / fmax;
      Image<float>::iterator rptr = rf.beginw();
      Image<float>::iterator gptr = gf.beginw();
      Image<float>::iterator bptr = bf.beginw();
      Image<float>::iterator stop = rf.endw();
      while (rptr != stop)
        {
          *rptr = (float)(float(*rptr) * scale);
          *gptr = (float)(float(*gptr) * scale);
          *bptr = (float)(float(*bptr) * scale);
          ++rptr;
          ++gptr;
          ++bptr;
        }
      Image< PixRGB<byte> > color_filter = makeRGB(rf + 128.0F,
                                                   gf + 128.0F,
                                                   bf + 128.0F);
      Raster::WriteRGB(color_filter, sformat("filter%i.ppm", i));
    }

  for (int i = 0; i < NB_FILTERS; i++)
    {
      // Create a channel attached to each filter:
      nub::soft_ref<RGBConvolveChannel> channel(new RGBConvolveChannel(manager));

      channel->setDescriptiveName(sformat("RGBConvolve%d", i));
      channel->setTagName(sformat("rgbconv%d", i));

      channel->exportOptions(MC_RECURSE);  // Get our configs

//       const char *filtername = manager.getExtraArg(i).c_str();
//       FILE *f = fopen(filtername, "r");
//       if (f == NULL) LFATAL("Cannot open %s", filtername);

//       // Scan the filter file to get the 3 kernels:
//       int w, h;
//       if (fscanf(f, "%d %d\n", &w, &h) != 2)
//         LFATAL("Bogus first line in %s", filtername);
//       LINFO("Building %dx%d RGB kernel from '%s'", w, h, filtername);
//       Image<float> rker(w, h, NO_INIT);
//       for (int j = 0; j < h; j ++)
//         for (int i = 0; i < w; i ++) {
//           float coeff;
//           if (fscanf(f, "%f\n", &coeff) != 1)
//             LFATAL("Bogus coeff in %s at red (%d, %d)",
//                    filtername, i, j);
//           rker.setVal(i, j, coeff);
//         }
//       Image<float> gker(w, h, NO_INIT);
//       for (int j = 0; j < h; j ++)
//         for (int i = 0; i < w; i ++) {
//           float coeff;
//           if (fscanf(f, "%f\n", &coeff) != 1)
//             LFATAL("Bogus coeff in %s at green (%d, %d)",
//                    filtername, i, j);
//           gker.setVal(i, j, coeff);
//         }
//       Image<float> bker(w, h, NO_INIT);
//       for (int j = 0; j < h; j ++)
//         for (int i = 0; i < w; i ++) {
//           float coeff;
//           if (fscanf(f, "%f\n", &coeff) != 1)
//             LFATAL("Bogus coeff in %s at blue (%d, %d)",
//                    filtername, i, j);
//           bker.setVal(i, j, coeff);
//         }

      // Assign the 3 filters to the channel:
      channel->setFilters(filter[3 * i], filter[3 * i + 1],
                          filter[3 * i + 2],
                          CONV_BOUNDARY_ZERO);

      // Attach the channel to our visual cortex:
      vcx->addSubChan(channel);
    }

  // Let's get all our ModelComponent instances started:
  manager.start();

  // ####################################################################
  // Main processing:

  // Read the input image:
  LINFO("*** Loading image %s", framename);
  Image< PixRGB<byte> > picture = Raster::ReadRGB(framename, RASFMT_PNM);

  // Process the image through the visual cortex:
  vcx->input(InputFrame::fromRgb(&picture));

  // Get the resulting saliency map:
  Image<float> sm = vcx->getOutput();

  // Normalize the saliency map:
  inplaceNormalize(sm, 0.0f, 255.0f);
  Image<byte> smb = sm;

  // Save the normalized saliency map:
  int i = strlen(framename) - 1; while(i > 0 && framename[i] != '.') i--;
  framename[i] = '\0'; // Remove input file extension
  LINFO("*** Saving '%s-SM.pgm'...", framename);
  Raster::WriteGray(smb, sformat("%s-SM.pgm", framename));

//   // Chamfer the binary mask, rescale it to the saliency map's size,
//   // and invert it:
//   Dims dim = smb.getDims();
//   Image<byte> blur_mask = binaryReverse(chamfer34(mask, (byte) 255),
//                                         (byte) 255);
//   inplaceLowThresh(blur_mask, (byte) 200);
//   Image<byte> mask_in = scaleBlock(blur_mask, dim);
//   Image<byte> mask_out = binaryReverse(mask_in, (byte) 255);

//   // Weight the saliency map using the in and out masks:
//   Image<float> smb_in = (mask_in * (1.0f / 255.0f)) * smb;
//   Image<float> smb_out = (mask_out * (1.0f / 255.0f)) * smb;

//   // Get the max_in and max_out values:
//   float max_in, max_out, min;
//   getMinMax(smb_in, min, max_in);
//   getMinMax(smb_out, min, max_out);

//   // Compute the error:
//   float detect_coeff = 1.0f - ((max_in - max_out) / 255.0f);
//   float error_val = detect_coeff * detect_coeff;
//   // Display the error value:
//   std::cout << error_val << std::endl;

  // Save the result:
  // LINFO("*** Saving 'max_in_out.txt'...", max_in, max_out);
  // FILE *result = fopen("max_in_out.txt", "w");
  //fprintf(result, "%f %f", max_in, max_out);

  // Stop all our ModelComponents
  manager.stop();

  // Convolve the picture with the filters and save the results
  for (int i = 0; i < NB_FILTERS; i++)
    {
      RGBConvolvePyrBuilder<float> rgbcpb(filter[3 * i],
                                          filter[3 * i + 1],
                                          filter[3 * i + 2],
                                          CONV_BOUNDARY_ZERO);
      ImageSet< PixRGB<float> > rgbpyr = rgbcpb.build2(picture, 0, 4);
      for (int j = 0; j < 4; j++)
        {
          Image<float> rc, gc, bc;
          getComponents(rgbpyr[j], rc, gc, bc);
          float min, max, fmax;
          getMinMax(rc, min, max);
          if (fabs(min) > fabs(max))
            fmax = min;
          else
            fmax = max;
          getMinMax(gc, min, max);
          if (fabs(min) > fmax)
            fmax = min;
          if (fabs(max) > fmax)
            fmax = max;
          getMinMax(bc, min, max);
          if (fabs(min) > fmax)
            fmax = min;
          if (fabs(max) > fmax)
            fmax = max;
          if (fmax < 1.0e-10F)
            fmax = 1; // images are uniform
          float scale = 255.0F / fmax;
          Image<float>::iterator rptr = rc.beginw();
          Image<float>::iterator gptr = gc.beginw();
          Image<float>::iterator bptr = bc.beginw();
          Image<float>::iterator stop = rc.endw();
          while (rptr != stop)
            {
              *rptr = (float)(float(*rptr) * scale);
              *gptr = (float)(float(*gptr) * scale);
              *bptr = (float)(float(*bptr) * scale);
              ++rptr;
              ++gptr;
              ++bptr;
            }
          Image<float> prc, nrc, pgc, ngc, pbc, nbc;
          splitPosNeg(rc, prc, nrc);
          splitPosNeg(gc, pgc, ngc);
          splitPosNeg(bc, pbc, nbc);
          rgbpyr[j] = makeRGB((rc / 2.0F) + 128.0F,
                              (gc / 2.0F) + 128.0F,
                              (bc / 2.0F) + 128.0F);
          Image< PixRGB<byte> > rgbpyrb = rgbpyr[j];
          Raster::WriteRGB(rgbpyrb, sformat("%s/conv-f%i-l%i.ppm",
                                            framename, i, j));
          rgbpyr[j] = makeRGB(prc, pgc, pbc);
          rgbpyrb = rgbpyr[j];
          Raster::WriteRGB(rgbpyrb, sformat("%s/conv-f%i-l%i-pos.ppm",
                                            framename, i, j));
          rgbpyr[j] = makeRGB(nrc, ngc, nbc);
          rgbpyrb = rgbpyr[j];
          Raster::WriteRGB(rgbpyrb, sformat("%s/conv-f%i-l%i-neg.ppm",
                                            framename, i, j));
        }
    }

  // All done!
  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
