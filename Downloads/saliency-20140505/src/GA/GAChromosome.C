/*!@file GA/GAChromosome.C A chromosome class for genetic algorithm. */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/GA/GAChromosome.C $
// $Id: GAChromosome.C 6005 2005-11-29 18:49:14Z rjpeters $
//

#include "GA/GAChromosome.H"

#include "Util/Assert.H"
#include "Util/Types.H"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <istream>
#include <ostream>

GAChromosome::GAChromosome() :
  fitness(0.0f), linear_fitness(0.0f), breedings(0), size(0), genes(NULL)
{ }

GAChromosome::GAChromosome(const int N) :
  fitness(0.0f), linear_fitness(0.0f), breedings(0), size(0), genes(NULL)
{
  init(N);
}

GAChromosome::GAChromosome(const int N, const int *a) :
  fitness(0.0f), linear_fitness(0.0f), breedings(0), size(0), genes(NULL)
{
  init(N, a);
}

GAChromosome::GAChromosome(const GAChromosome& c) :
  fitness(c.fitness), linear_fitness(c.linear_fitness),
  breedings(c.breedings), size(0), genes(NULL)
{
  init(c.size, c.genes);
}

void GAChromosome::resize(const int N)
{
  if (size) delete [] genes;
  genes = new int[N];
  size = N;
}

void GAChromosome::init(const int N, const int *a)
{
  resize(N);
  memcpy(genes, a, N * sizeof(int));
}

void GAChromosome::init(const int N)
{
  resize(N);
  for (int i = 0; i < N; i++)
    genes[i] = (rand() % 3) - 1;
}

GAChromosome::~GAChromosome()
{
  if (size) delete [] genes;
}

int GAChromosome::get_size() const
{
  return size;
}

void GAChromosome::set_gene(const int i, const int a)
{
  ASSERT(i >= 0 && i < size && abs(a) < 2);
  genes[i] = a;
}

int GAChromosome::get_gene(const int i) const
{
  ASSERT(i >= 0 && i < size);
  return genes[i];
}

void GAChromosome::set_fitness(const float a)
{
  ASSERT(a >= 0);
  fitness = a;
}

float GAChromosome::get_fitness() const
{
  return fitness;
}

void GAChromosome::set_linear_fitness(const float a)
{
  ASSERT(a >= 0);
  linear_fitness = a;
}

float GAChromosome::get_linear_fitness() const
{
  return linear_fitness;
}

void GAChromosome::set_breedings(const int a)
{
  ASSERT(a >= 0);
  breedings = a;
}

int GAChromosome::get_breedings() const
{
  return breedings;
}

GAChromosome& GAChromosome::operator=(const GAChromosome& c)
{
  init(c.size, c.genes);
  fitness = c.fitness;
  linear_fitness = c.linear_fitness;
  breedings = c.breedings;
  return *this;
}

bool GAChromosome::operator<(const GAChromosome& c) const
{
  return breedings < c.breedings;
}

void GAChromosome::mutation()
{
  int i = rand() % size;
  genes[i] = ((genes[i] + 2) % 3) - 1;
}

void GAChromosome::add_breeding()
{
  breedings++;
}

void GAChromosome::use_breeding()
{
  ASSERT(breedings > 0);
  breedings--;
}

std::istream& operator>> (std::istream& in, GAChromosome& c)
{
  int s, f, l, b;
  in >> s >> f >> l >> b;
  ASSERT(f >=0 && l >= 0 && b >= 0);
  c.resize(s);
  c.fitness = f;
  c.linear_fitness = l;
  c.breedings = b;
  for (int i = 0; i < c.size; i++)
    {
      int g;
      in >> g;
      ASSERT(abs(g) < 2);
      c.genes[i] = g;
    }
  return in;
}

std::ostream& operator<< (std::ostream& out, GAChromosome& c)
{
  out << c.size << '\n';
  out << c.fitness << ' ' << c.linear_fitness << ' ';
  out << c.breedings << '\n';
  for (int i = 0; i < c.size; i++)
    {
      out << c.genes[i] << '\n';
    }
  return out;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
