/*!@file GA/GAPopulation.C A population class for genetic algorithm. */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2002   //
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
// Primary maintainer for this file: Laurent Itti <itti@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/GA/GAPopulation.C $
// $Id: GAPopulation.C 6005 2005-11-29 18:49:14Z rjpeters $
//

#include "GA/GAPopulation.H"

#include "GA/GAChromosome.H"
#include "Util/Assert.H"
#include "Util/Types.H"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <istream>
#include <ostream>

#define MSG_DATA 1
#define MSG_RESULT 2

GAPopulation::GAPopulation() :
  psize(0), csize(0), total_fitness(0.0f), total_linear_fitness(0.0f),
  mean_fitness(0.0f), sigma(0.0f), chromosomes(NULL), offspring(NULL)
{ }

GAPopulation::GAPopulation(const int N, const int a) :
  psize(0), csize(0), total_fitness(0.0f), total_linear_fitness(0.0f),
  mean_fitness(0.0f), sigma(0.0f), chromosomes(NULL), offspring(NULL)
{
  init(N, a);
}

void GAPopulation::resize(const int N, const int a)
{
  if (psize)
    {
      delete [] chromosomes;
      delete [] offspring;
    }
  chromosomes = new GAChromosome[N];
  offspring = new GAChromosome[N];
  psize = N;
  csize = a;
}

void GAPopulation::init(const int N, const int a)
{
  resize(N, a);
  for (int i = 0; i < N; i++)
    {
      chromosomes[i].init(a);
      offspring[i].init(a);
    }
}

GAPopulation::~GAPopulation()
{
  if (psize)
    {
      delete [] chromosomes;
      delete [] offspring;
    }
}

void GAPopulation::set_chromosome(const int i, const GAChromosome& c)
{
  ASSERT(c.get_size() == csize);
  chromosomes[i] = c;
}

GAChromosome GAPopulation::get_chromosome(const int i) const
{
  return chromosomes[i];
}

float GAPopulation::get_mean_fitness() const
{
  return mean_fitness;
}

float GAPopulation::get_sigma() const
{
  return sigma;
}

void GAPopulation::update()
{
  for (int i = 0; i < psize; i++)
    chromosomes[i] = offspring[i];
}

void GAPopulation::compute_pop_fitness()
{
  total_fitness = 0.0f;
  for (int i = 0; i < psize; i++)
    total_fitness += chromosomes[i].get_fitness();
  mean_fitness = total_fitness / psize;
}

void GAPopulation::compute_sigma()
{
  sigma = 0.0f;
  for (int i = 0; i < psize; i++)
    {
      float x = chromosomes[i].get_fitness() - mean_fitness;
      sigma += x * x;
    }
  sigma /= psize - 1;
  sigma = sqrt(sigma);
}

void GAPopulation::linear_scaling()
{
  total_linear_fitness = 0.0f;
  for (int i = 0; i < psize; i++)
    {
      float lf;
      if (sigma == 0.0f)
        lf = 1.0f;
      else
        {
          float f = chromosomes[i].get_fitness();
          if (mean_fitness - f > sigma)
            lf = 0.0f;
          else
            lf = 1.0f + (f - mean_fitness) / (2.0f * sigma);
        }
      chromosomes[i].set_linear_fitness(lf);
      total_linear_fitness += lf;
    }
}

void GAPopulation::selection()
{
  int x = rand() % 100;
  float spin = (((float) x) / 100.0f) / psize;
  int c = -1;
  float f = 0.0f;
  for (int i = 0; i < psize; i++)
    {
      while (f < spin)
        {
          c++;
          f += chromosomes[c].get_linear_fitness() / total_linear_fitness;
        }
      chromosomes[c].add_breeding();
      f -= spin;
    }
}

void GAPopulation::crossover()
{
  std::sort(chromosomes, chromosomes + psize);
  int father = psize - 1;
  int mother = psize - 2;
  int breed = 0;
  while (breed < psize)
    {
      int cross = (rand() % (csize - 1)) + 1;
      for (int i = 0; i < cross; i++)
        {
          offspring[breed].set_gene(i, chromosomes[father].get_gene(i));
          offspring[breed + 1].set_gene(i, chromosomes[mother].get_gene(i));
        }
      for (int i = cross; i < csize; i++)
        {
          offspring[breed].set_gene(i, chromosomes[mother].get_gene(i));
          offspring[breed + 1].set_gene(i, chromosomes[father].get_gene(i));
        }
      chromosomes[father].use_breeding();
      chromosomes[mother].use_breeding();
      breed = breed + 2;
      float fb = chromosomes[father].get_breedings();
      float mb = chromosomes[mother].get_breedings();
      for (int i = mother;
           i > 0 && mb < chromosomes[i - 1].get_breedings();
           i--)
        std::swap(chromosomes[i], chromosomes[i - 1]);
      for (int i = father;
           i > 0 && fb < chromosomes[i - 1].get_breedings();
           i--)
        std::swap(chromosomes[i], chromosomes[i - 1]);
    }
}

void GAPopulation::mutate()
{
  for (int i = 0; i < psize; i++)
    {
      int lucky_shot = rand() % csize;
      if (lucky_shot == 7)
        chromosomes[i].mutation();
    }
}

std::istream& operator>>(std::istream& in, GAPopulation& pop)
{
  int ps, cs;
  in >> ps >> cs;
  pop.resize(ps, cs);
  in >> pop.total_fitness >> pop.mean_fitness >> pop.sigma;
  in >> pop.total_linear_fitness;
  for (int i = 0; i < pop.psize; i++)
    {
      GAChromosome c;
      in >> c;
      ASSERT(c.get_size() == pop.csize);
      pop.chromosomes[i] = c;
    }
  return in;
}

std::ostream& operator<<(std::ostream& out, GAPopulation& pop)
{
  out << pop.psize << ' ' << pop.csize << '\n';
  out << pop.total_fitness << ' ' << pop.mean_fitness << ' ';
  out << pop.sigma << ' ' << pop.total_linear_fitness << '\n';
  for (int i = 0; i < pop.psize; i++)
    out << pop.chromosomes[i];
  return out;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
