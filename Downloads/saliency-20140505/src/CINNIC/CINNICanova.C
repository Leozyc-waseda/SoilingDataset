/*!@file CINNIC/CINNICanova.C do ANOVA tests on CINNIC outputs */

// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CINNIC/CINNICanova.C $
// $Id: CINNICanova.C 6003 2005-11-29 17:22:45Z rjpeters $

// ############################################################
// ############################################################
// ##### ---CINNIC---
// ##### Contour Integration:
// ##### T. Nathan Mundhenk nathan@mundhenk.com
// ############################################################
// ############################################################

#include "Image/Image.H"
#include "Util/Assert.H"
#include "Util/log.H"
#include "Util/readConfig.H"
#include "Util/stats.H"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

//! the file to save stats out to
char *savefilename;
//! two data sets to open
readConfig ReadConfigA(25);
//! two data sets to open
readConfig ReadConfigB(25);
//! Files containing data sets
char* dataSet1;
char* dataSet2;
//! Vector for regresion data set 1
std::vector<float> rSet1;
//! Vector for regresion data set 2
std::vector<float> rSet2;
//! Vector for euclidian data set 1
std::vector<float> eSet1;
//! Vector for euclidian data set 2
std::vector<float> eSet2;
//! Stats objects
stats<float> rStats;
//! Stats objects
stats<float> eStats;
//! F from anova
float rF,eF;

int main(int argc, char* argv[])
{
  ASSERT(argc >= 2);
  dataSet1 = argv[1];
  dataSet2 = argv[2];
  //open and parse data sets
  ReadConfigA.openFile(dataSet1);
  ReadConfigB.openFile(dataSet2);
  //Make sure data sets are the same size
  ASSERT(ReadConfigA.itemCount() == ReadConfigB.itemCount());
  //resize data set vectors
  rSet1.resize(ReadConfigA.itemCount(),0.0F);
  eSet1.resize(ReadConfigA.itemCount(),0.0F);
  rSet2.resize(ReadConfigB.itemCount(),0.0F);
  eSet2.resize(ReadConfigB.itemCount(),0.0F);
  //load values from config to a vector<float>
  for(int i = 0; ReadConfigA.readFileTrue(i); i++)
  {
    rSet1[i] = ReadConfigA.readFileValueNameF(i);
    eSet1[i] = ReadConfigA.readFileValueF(i);
    rSet2[i] = ReadConfigB.readFileValueNameF(i);
    eSet2[i] = ReadConfigB.readFileValueF(i);
  }
  rF = rStats.simpleANOVA(rSet1,rSet2);
  eF = eStats.simpleANOVA(eSet1,eSet2);

  LINFO("DATA sets %s : %s",dataSet1,dataSet2);
  LINFO("Regression Stats on ANOVA");

  using std::cout;

  cout << "between\t";
  cout << rStats.SSbetween << "\t" << rStats.DFbetween << "\t"
       << rStats.MSbetween << "\t" << rStats.F << "\n";
  cout << "within\t";
  cout << rStats.SSwithin << "\t" << rStats.DFwithin << "\t"
       << rStats.MSwithin << "\n";
  cout << "total\t";
  cout << rStats.SStotal << "\n\n";
  LINFO("Euclidian Stats on ANOVA");
  cout << "between\t";
  cout << eStats.SSbetween << "\t" << eStats.DFbetween << "\t"
       << eStats.MSbetween << "\t" << eStats.F << "\n";
  cout << "within\t";
  cout << eStats.SSwithin << "\t" << eStats.DFwithin << "\t"
       << eStats.MSwithin << "\n";
  cout << "total\t";
  cout << eStats.SStotal << "\n\n";
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
