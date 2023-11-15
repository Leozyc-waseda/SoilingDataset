/*!@file INVT/openvision3.C version of ezvision.C that uses on-file gray
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/INVT/openvision3.C $
// $Id: openvision3.C 10845 2009-02-13 08:49:12Z itti $
//

#include "Channels/ChannelOpts.H"
#include "Channels/ConvolveChannel.H"
#include "Component/ModelManager.H"
#include "Image/ColorOps.H"
#include "Image/DrawOps.H"
#include "Image/MathOps.H"
#include "Image/MatrixOps.H"
#include "Image/Pixels.H"
#include "Image/ShapeOps.H"
#include "Media/FrameSeries.H"
#include "Channels/RawVisualCortex.H"
#include "Raster/Raster.H"
#include "Util/sformat.H"

#include <fstream>

#define NB_FILTERS 4
#define NB_COEFFS 8
#define RAD 64
#define NB_CONV 1

int main(const int argc, const char **argv)
{
  int n = NB_COEFFS;
  int m = NB_COEFFS * NB_COEFFS * NB_FILTERS;

  MYLOGVERB = LOG_INFO;  // Suppress debug messages

  // Generate the haar transform matrix:
  Image<float> hmat(n, n, ZEROS);
  for(int i = 0; i < n; i++)
    hmat.setVal(i, 0, 1.0f);
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
  manager.setOptionValString(&OPT_MaxNormType, "Maxnorm");

  // let's make one dummy RGBConvolveChannel so that we get the
  // command-line options for it:
  nub::soft_ref<ConvolveChannel> cchannel(new ConvolveChannel(manager));
  vcx->addSubChan(cchannel);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "<data.txt> <image.ppm>",
                               2, -1) == false)
    return(1);

  // Ok, get rid of our placeholder channel; the manager will keep a
  // trace of its configured options:
  vcx->removeAllSubChans();

  // Get the input image name:
  const std::string framename = manager.getExtraArg(1);

  // Load Scanpaths:
  Point2D<int> ch[20];
  Point2D<int> kp[20];
  Point2D<int> pw[20];
  Point2D<int> wkm[20];
  int nch, nkp, npw, nwkm;
  int count = 0;
  std::ifstream chfile(sformat("%s.ch.dat", framename.c_str()).c_str());
  bool eofile = false;
  while (!chfile.eof())
    {
      int px, py;
      chfile >> px >> py;
      if (!chfile.eof())
        {
          ch[count].i = px;
          ch[count].j = py;
          count++;
        }
      else
        eofile = true;
    }
  chfile.close();
  nch = count;
  count = 0;
  std::ifstream kpfile(sformat("%s.kp.dat", framename.c_str()).c_str());
  eofile = false;
  while (!eofile)
    {
      int px, py;
      kpfile >> px >>py;
      if (!kpfile.eof())
        {
          kp[count].i = px;
          kp[count].j = py;
          count++;
        }
      else
        eofile = true;
    }
  kpfile.close();
  nkp = count;
  count = 0;
  std::ifstream pwfile(sformat("%s.pw.dat", framename.c_str()).c_str());
  eofile = false;
  while (!eofile)
    {
      int px, py;
      pwfile >> px >> py;
      if (!pwfile.eof())
        {
          pw[count].i = px;
          pw[count].j = py;
          count++;
        }
      else
        eofile = true;
    }
  pwfile.close();
  npw = count;
  count = 0;
  std::ifstream wkmfile(sformat("%s.wkm.dat", framename.c_str()).c_str());
  eofile = false;
  while (!eofile)
    {
      int px, py;
      wkmfile >> px >> py;
      if (!wkmfile.eof())
        {
          wkm[count].i = px;
          wkm[count].j = py;
          count++;
        }
      else
        eofile = true;
    }
  wkmfile.close();
  nwkm = count;

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
  ImageSet<float> trans(NB_FILTERS);
  for (int i = 0; i < NB_FILTERS; i++)
    trans[i] = Image<float>(data + (n * n * i), n, n);
  ImageSet<float> filter(NB_FILTERS);
  for (int i = 0; i < NB_FILTERS; i++)
    filter[i] = matrixMult(transpose(hmat),
                           matrixMult(trans[i], hmat));

  // Compute the dot product of the filters:
  int nb_prod = (NB_FILTERS * (NB_FILTERS - 1)) / 2;
  ImageSet<float> prod(nb_prod);
  int k = 0;
  for (int j = 0; j < NB_FILTERS - 1; j++)
    for (int i = j + 1; i < NB_FILTERS; i++)
      {
        prod[k] = filter[j] * filter[i];
        k++;
      }
  float dotprod = 0.0f;
  for (int i = 0; i < nb_prod; i++)
    dotprod += sum(prod[i]);

  // Inject filterx in Visual Cortex:
  for (int i = 0; i < NB_FILTERS; i++)
    {
      // Create a channel attached to each filter:
      nub::soft_ref<ConvolveChannel> channel(new ConvolveChannel(manager));

      channel->setDescriptiveName(sformat("Convolve%d", i));
      channel->setTagName(sformat("conv%d", i));

      channel->exportOptions(MC_RECURSE);  // Get our configs

      // Assign the 3 filters to the channel:
      channel->setFilter(filter[i], CONV_BOUNDARY_ZERO);

      // Attach the channel to our visual cortex:
      vcx->addSubChan(channel);
    }

  // Let's get all our ModelComponent instances started:
  manager.start();

  // ####################################################################
  // Main processing:

  // Read the input image:
  LINFO("*** Loading image %s", framename.c_str());
  Image< PixRGB<byte> > picture =
    Raster::ReadRGB(sformat("%s.ppm", framename.c_str()));

  // Process the image through the visual cortex:
  vcx->input(InputFrame::fromRgb(&picture));

  // Get the resulting saliency map:
  Image<float> sm = vcx->getOutput();

  // Normalize the saliency map:
  inplaceNormalize(sm, 0.0f, 255.0f);
  Image<byte> smb = sm;

  // Save the normalized saliency map:
  Raster::WriteGray(smb, sformat("%s-SM.pgm", framename.c_str()));

  // Rescale saliency map:
  Dims dim = picture.getDims();
  Image< PixRGB<byte> > smr = rescale(smb, dim);

  // Get the average saliency:
  double avgsal = mean(sm);

  // Get the map level to scale things down:
  const LevelSpec lspec = vcx->getModelParamVal<LevelSpec>("LevelSpec");
  int sml = lspec.mapLevel();

  // Scale the radius
  int radius = RAD >> sml;

  // Get the saliency on each end of saccade and draw circles on big sm:
  float chsal = 0;
  int sc = 1 << sml;
  PixRGB<byte> red(200, 0, 0);
  PixRGB<byte> green(0, 200, 0);
  PixRGB<byte> blue(0, 0, 200);
  PixRGB<byte> yellow(200, 200, 0);
  for (int i = 0; i < nch; i++)
    {
      chsal += getLocalMax(sm, ch[i] / sc, radius);
      drawCircle(smr, ch[i], RAD, red);
    }
  chsal /= nch;
  float kpsal = 0;
  for (int i = 0; i < nkp; i++)
    {
      kpsal += getLocalMax(sm, kp[i] / sc, radius);
      drawCircle(smr, kp[i], RAD, green);
    }
  kpsal /= nkp;
  float pwsal = 0;
  for (int i = 0; i < npw; i++)
    {
      pwsal += getLocalMax(sm, pw[i] / sc, radius);
      drawCircle(smr, pw[i], RAD, blue);
    }
  pwsal /= npw;
  float wkmsal = 0;
  for (int i = 0; i < nwkm; i++)
    {
      wkmsal += getLocalMax(sm, wkm[i] / sc, radius);
      drawCircle(smr, wkm[i], RAD, yellow);
    }
  wkmsal /= nwkm;

  float goodsal = (chsal + kpsal + pwsal + wkmsal) / 4;

  // Save saliency map with circles:
  Raster::WriteRGB(smr, sformat("%s-SM_circles.ppm", framename.c_str()));

  // Compute the error:
  float error = (1 + avgsal) / (1 + goodsal);
  float result = error + 0.00001f * fabs(dotprod);

  // Save results:
  const std::string resname = sformat("%s-score.txt", framename.c_str());
  std::ofstream resultfile(resname.c_str());
  resultfile << "*** " << framename << " ***\n";
  resultfile << "Filters product : " << fabs(dotprod) << "\n";
  resultfile << "Saliency score : " << error << "\n";
  resultfile << "Total score : " << result << "\n";
  resultfile.close();

  {
    float fmax = 0.0f;
    for (int i = 0; i < NB_FILTERS; i++)
      {
        float min, max;
        getMinMax(filter[i], min, max);
        if (fabs(min) > fmax)
          fmax = min;
        if (fabs(max) > fmax)
          fmax = max;
      }
    if (fmax < 1.0e-10F)
      fmax = 1; // images are uniform
    float scale = 128.0F / fmax;
    for (int i = 0; i < NB_FILTERS; i++)
      {
        Image<float>::iterator fptr = filter[i].beginw();
        Image<float>::iterator stop = filter[i].endw();
        while (fptr != stop)
          {
            *fptr = (float)(float(*fptr) * scale);
            ++fptr;
          }
        Image<byte> filterb = filter[i] + 128.0f;
        Raster::WriteGray(filterb, sformat("filter%i.pgm", i));
      }
  }

  // Stop all our ModelComponents
  manager.stop();

  // Convolve the picture with the filters and save the results
  for (int i = 0; i < NB_FILTERS; i++)
    {
      ConvolvePyrBuilder<float> pyrb(filter[i], CONV_BOUNDARY_ZERO);
      Image<float> pictureg = luminance(picture);
      ImageSet<float> pyr = pyrb.build(pictureg, 0, NB_CONV);
      for (int j = 0; j < NB_CONV; j++)
        {
          float min, max, fmax;
          getMinMax(pyr[j], min, max);
          if (fabs(min) > fabs(max))
            fmax = min;
          else
            fmax = max;
          if (fmax < 1.0e-10F)
            fmax = 1; // convolution is uniform
          float scale = 128.0F / fmax;
          Image<float>::iterator pyrptr = pyr[j].beginw();
          Image<float>::iterator stop = pyr[j].endw();
          while (pyrptr != stop)
            {
              *pyrptr = (float)(float(*pyrptr) * scale);
              ++pyrptr;
            }
          Image<byte> conv = pyr[j] + 128.0f;
          Raster::WriteGray(conv,
                            sformat("%s_conv%i_filt%i.pgm",
                                    framename.c_str(), j, i));
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
