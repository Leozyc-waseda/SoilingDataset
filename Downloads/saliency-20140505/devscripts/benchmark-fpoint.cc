// modified from http://www.cs.brandeis.edu/~cs155/YUV_to_RGB.c

// $Id: benchmark-fpoint.cc 4329 2005-06-07 18:52:19Z rjpeters $

/*

Run the benchmark-fpoint.sh script in your machine, and add the
results here!

hostname         cpu           cpu-MHz gcc   opt   double    float      int

iLab1.usc.edu    Intel Xeon       2801 3.4.1 -O0   1.000s   0.830s   0.650s
iLab1.usc.edu    Intel Xeon       2801 3.4.1 -O1   0.670s   0.680s   0.440s
iLab1.usc.edu    Intel Xeon       2801 3.4.1 -O2   0.660s   0.670s   0.440s
iLab1.usc.edu    Intel Xeon       2801 3.4.1 -O3   0.550s   0.560s   0.400s

iLab4.usc.edu    Pentium III       862 3.4.1 -O0   3.860s   3.160s   1.970s
iLab4.usc.edu    Pentium III       862 3.4.1 -O1   2.670s   2.550s   0.920s
iLab4.usc.edu    Pentium III       862 3.4.1 -O2   2.420s   2.170s   0.910s
iLab4.usc.edu    Pentium III       862 3.4.1 -O3   1.960s   1.970s   0.790s

iLab5.usc.edu    AMD Athlon       1397 3.4.1 -O0   2.030s   1.860s   1.580s
iLab5.usc.edu    AMD Athlon       1397 3.4.1 -O1   1.230s   1.140s   0.670s
iLab5.usc.edu    AMD Athlon       1397 3.4.1 -O2   1.210s   1.080s   0.630s
iLab5.usc.edu    AMD Athlon       1397 3.4.1 -O3   1.000s   0.960s   0.520s

iLab6.usc.edu    AMD Athlon       1666 3.4.1 -O0   1.670s   1.540s   1.280s
iLab6.usc.edu    AMD Athlon       1666 3.4.1 -O1   0.970s   0.860s   0.540s
iLab6.usc.edu    AMD Athlon       1666 3.4.1 -O2   0.920s   0.870s   0.510s
iLab6.usc.edu    AMD Athlon       1666 3.4.1 -O3   0.760s   0.770s   0.430s

iLab7.usc.edu    AMD Athlon       1095 3.3.1 -O0   2.550s   2.270s   1.860s
iLab7.usc.edu    AMD Athlon       1095 3.3.1 -O1   1.330s   1.310s   0.810s
iLab7.usc.edu    AMD Athlon       1095 3.3.1 -O2   1.270s   1.260s   0.790s
iLab7.usc.edu    AMD Athlon       1095 3.3.1 -O3   1.210s   1.180s   0.660s

iLab8.usc.edu    Pentium III       733 3.3.2 -O0   4.110s   3.710s   2.300s
iLab8.usc.edu    Pentium III       733 3.3.2 -O1   2.930s   2.910s   1.100s
iLab8.usc.edu    Pentium III       733 3.3.2 -O2   2.660s   2.650s   1.140s
iLab8.usc.edu    Pentium III       733 3.3.2 -O3   2.370s   2.360s   0.940s

iLab9.usc.edu    Intel Xeon       2801 3.4.3 -O0   1.000s   0.830s   0.670s
iLab9.usc.edu    Intel Xeon       2801 3.4.3 -O1   0.670s   0.680s   0.450s
iLab9.usc.edu    Intel Xeon       2801 3.4.3 -O2   0.800s   0.680s   0.450s
iLab9.usc.edu    Intel Xeon       2801 3.4.3 -O3   0.530s   0.540s   0.410s

iLab11.usc.edu   AMD Athlon       1749 3.4.1 -O0   1.520s   1.380s   1.160s
iLab11.usc.edu   AMD Athlon       1749 3.4.1 -O1   0.870s   0.830s   0.490s
iLab11.usc.edu   AMD Athlon       1749 3.4.1 -O2   0.830s   0.770s   0.490s
iLab11.usc.edu   AMD Athlon       1749 3.4.1 -O3   0.750s   0.720s   0.410s

sideswipe        Pentium 4        1199 3.4.3 -O0   2.780s   1.160s   0.950s
sideswipe        Pentium 4        1798 3.4.3 -O1   1.830s   0.960s   0.990s
sideswipe        Pentium 4        1798 3.4.3 -O2   0.960s   0.960s   1.070s
sideswipe        Pentium 4        1798 3.4.3 -O3   0.750s   0.760s   0.590s

*/

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <time.h>
#include <unistd.h> // for gethostname()

//
//      Floating point defines
//
const double DBL_C_1  = -1.40200;
const double DBL_C_2  = -0.34414;
const double DBL_C_3  =  0.71414;
const double DBL_C_4  =  1.77200;
const double DBL_HALF =  0.50000;

void DBL_YUV_to_RGB(double Y, double U, double V,
                    volatile int *pR, volatile int *pG, volatile int *pB)
{
  U -= 127;
  V -= 127;
  *pR = (int) (Y + DBL_C_1 * V + DBL_HALF);
  *pG = (int) (Y + DBL_C_2 * U + DBL_C_3 * V + DBL_HALF);
  *pB = (int) (Y + DBL_C_4 * U + DBL_HALF);
}

//
//      Floating point defines
//
const float FLT_C_1  = -1.40200f;
const float FLT_C_2  = -0.34414f;
const float FLT_C_3  =  0.71414f;
const float FLT_C_4  =  1.77200f;
const float FLT_HALF =  0.50000f;

void FLT_YUV_to_RGB(float Y, float U, float V,
                    volatile int *pR, volatile int *pG, volatile int *pB)
{
  U -= 127;
  V -= 127;
  *pR = (int) (Y + FLT_C_1 * V + FLT_HALF);
  *pG = (int) (Y + FLT_C_2 * U + FLT_C_3 * V + FLT_HALF);
  *pB = (int) (Y + FLT_C_4 * U + FLT_HALF);
}

//
//      Fixed point defines
//
#define YUV_SHIFT       16
#define YUV_ONE         (1 << YUV_SHIFT)
#define YUV_HALF        (1 << (YUV_SHIFT - 1))

const int INT_C_1 =   ((int)(-1.40200 * YUV_ONE));
const int INT_C_2 =   ((int)(-0.34414 * YUV_ONE));
const int INT_C_3 =   ((int)( 0.71414 * YUV_ONE));
const int INT_C_4 =   ((int)( 1.77200 * YUV_ONE));

void YUV_to_RGB(int Y, int U, int V,
                volatile int *pR, volatile int *pG, volatile int *pB)
{
  U -= 127;
  V -= 127;
  *pR = Y + ((INT_C_1 * V + YUV_HALF) >> YUV_SHIFT);
  *pG = Y + ((INT_C_2 * U + INT_C_3 * V + YUV_HALF) >> YUV_SHIFT);
  *pB = Y + ((INT_C_4 * U + YUV_HALF ) >> YUV_SHIFT);
}

std::string extract_modelname(const std::string& line)
{
  std::string modelname = line.substr(line.find(':')+2);
  if (modelname.compare(0, 13, "Intel(R) Xeon") == 0)
    {
      modelname = "Intel Xeon";
    }
  else if (modelname.compare(0, 21, "Intel(R) Pentium(R) 4") == 0)
    {
      modelname = "Pentium 4";
    }
  else
    {
      modelname = modelname.substr(0, modelname.find('('));
    }
  return modelname.substr(0, 12);
}

//
//      Test program
//
int main(int argc, char** argv)
{
  volatile int R, G, B;   // Prevents code elimination with -O3

  clock_t start, finish;

  const int nframes =
    (argc > 1)
    ? atoi(argv[1])
    : 3000;

  const int nloop = (640 * 480 * nframes);

  const int nrand =
    (argc > 2)
    ? atoi(argv[2])
    : 1024;

  const char* optlevel =
    (argc > 3)
    ? argv[3]
    : "";

  char hostname[256];
  gethostname(&hostname[0], 256);
  hostname[255] = '\0';

  int Y[nrand];
  int U[nrand];
  int V[nrand];

  for (int i = 0; i < nrand; ++i)
    {
      Y[i] = rand() % 256;
      U[i] = rand() % 256;
      V[i] = rand() % 256;
    }

  std::string modelname = "";
  int cpumhz = 0;

  std::ifstream f("/proc/cpuinfo");
  while (f)
    {
      std::string line;
      std::getline(f, line);
      if (line.compare(0, 10, "model name") == 0)
        {
          modelname = extract_modelname(line);
        }
      else if (line.compare(0, 7, "cpu MHz") == 0)
        {
          std::string x = line.substr(line.find(':')+2);
          cpumhz = int(atof(x.c_str()));
        }
    }

  char gccversion[64];
  snprintf(&gccversion[0], 64, "%d.%d.%d",
           __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);

  printf("%-16s %-12s %8s %-5s %3s %8s %8s %8s\n",
         "hostname", "cpu", "cpu-MHz", "gcc", "opt", "double", "float", "int");
  printf("%-16s %-12s %8d %-5s %3s ",
         hostname, modelname.c_str(), cpumhz, gccversion, optlevel);

  // Floating point math
  start = clock();
  for (int i = 0; i < nloop; ++i)
    {
      DBL_YUV_to_RGB(Y[i%nrand], U[i%nrand], V[i%nrand], &R, &G, &B);
    }
  finish = clock();
  printf("%7.3fs ", double(finish - start) / CLOCKS_PER_SEC);

  // Floating point math
  start = clock();
  for (int i = 0; i < nloop; ++i)
    {
      FLT_YUV_to_RGB(Y[i%nrand], U[i%nrand], V[i%nrand], &R, &G, &B);
    }
  finish = clock();
  printf("%7.3fs ", double(finish - start) / CLOCKS_PER_SEC);

  // Fixed point math
  start = clock();
  for (int i = 0; i < nloop; ++i)
    {
      YUV_to_RGB(Y[i%nrand], U[i%nrand], V[i%nrand], &R, &G, &B);
    }
  finish = clock();
  printf("%7.3fs\n", double(finish - start) / CLOCKS_PER_SEC);
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
