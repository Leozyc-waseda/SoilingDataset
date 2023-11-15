/*! @file Landmark/density.C [put description here] */

// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Landmark/density.C $
// $Id: density.C 6410 2006-04-01 22:12:24Z rjpeters $

// find the density plot: treat each object as 1 pt in 2d space
// not considering sigma yet

#include "Landmark/density.H"

#include "Image/FilterOps.H"
#include "Image/Image.H"
#include "Raster/Raster.H"

#include <fstream>
#include <math.h>

Image<float> density(const char* filename, std::map<int, Object*>& list)
{

  // read features into image
  std::ifstream file(filename);
  LDEBUG("opening file %s", filename);
  if (file == 0) LFATAL("Couldn't open object file: '%s'", filename);
  std::string line;
  int count = 0;

  while(getline(file, line))
    {
      double mu1, sigma1;
      char name[256];
      sscanf(line.c_str(), "%s\t%lf %lf",
             name, &mu1, &sigma1);
      //LDEBUG("read %s\t%lf %lf\t%lf %lf\n",
      //             name, mu1, sigma1, mu2, sigma2);

      Object* o = new Object(name, count, mu1, sigma1);
      list.insert(std::pair<int, Object*>(count, o));
      count++;

    }

  // compute the inter-object distances
  int num = list.size();
  double sum = 0.0;
  double min_w = -1.0, max_w = 0.0;
  double distance[num][num];
  for(int i = 0; i < num; i++)
    {
      double min = 0.0;
      for(int j = i + 1; j < num; j++)
        {
          if(i == j)
            distance[i][j] = 0.0;
          else
            {
              // find the distance
              distance[i][j] = std::abs(list[i]->mu1 - list[j]->mu1);

              // find the minimum of all distances to this object
              if(min == 0.0)
                min = distance[i][j];
              else if(min > distance[i][j])
                 min = distance[i][j];

              // find the image width and height
              if(min_w == -1.0)
                min_w = list[i]->mu1;
              else if(min_w > list[i]->mu1)
                min_w = list[i]->mu1;

              if(max_w == 0.0)
                max_w = list[i]->mu1;
              else if(max_w < list[i]->mu1)
                max_w = list[i]->mu1;

            }
        }
      sum += min;
    }

  // consider the last object in the list to find the image width and height
  if(min_w == -1.0)
    min_w = list[num-1]->mu1;
  else if(min_w > list[num-1]->mu1)
    min_w = list[num-1]->mu1;

  if(max_w == 0.0)
    max_w = list[num-1]->mu1;
  else if(max_w < list[num-1]->mu1)
    max_w = list[num-1]->mu1;

  // find the average min inter-object distance
  double avg = sum / num;
  LDEBUG(" average inter-object min. distance = %lf", avg);

  // estimate image width and height
  LDEBUG(" width -- (%lf, %lf)",
         min_w, max_w);
  double w = max_w - min_w;

  // scale the image size to 128 * 1
  Image<float> image(128, 1, NO_INIT);
  image.clear(0.0f);
  double scale_w = 125.0 / w;

  // first plot the objects in the 1d space
  for(int i = 0; i < num; i++)
    {
      list[i]->mu1 = (list[i]->mu1 - min_w) * scale_w;
      int mu = (int)list[i]->mu1;
      // one more object yields the same feature response
      image.setVal(mu, image.getVal(mu)+1.0f);
    }

  // use avg to estimate the gaussian convolution kernel
  //double sigma = avg * std::max(scale_w, scale_h);
  double sigma = avg * scale_w; // / 5;
  LDEBUG("sigma of gaussian = %lf", sigma);

  Image<float> density = convGauss(image, sigma, 0, 1);
  LDEBUG("convolution over");

  Raster::WriteGray( image, "input.pgm");
  LDEBUG("written input");
  //Raster::WriteGray( density, "density.pgm");

  return density;

}
