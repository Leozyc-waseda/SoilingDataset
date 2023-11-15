////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////
extern "C" void ConvRowCPU(float *h_Result, float *h_Data, float *h_Kernel,
                           int dataW, int dataH, int kernelR)
{
  int x, y, k, d;
  float sum;
  for (y=0;y<dataH;y++)
    for (x=0;x<dataW;x++) {
      sum = 0;
      for (k=-kernelR;k<=kernelR;k++) {
        d = x + k;
        d = (d<0 ? 0 : d);
        d = (d>=dataW ? dataW-1 : d);
        sum += h_Data[y*dataW + d] * h_Kernel[kernelR - k];
      }
      h_Result[y*dataW + x] = sum;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Reference column convolution filter
////////////////////////////////////////////////////////////////////////////////
extern "C" void ConvColCPU(float *h_Result, float *h_Data, float *h_Kernel,
                           int dataW, int dataH, int kernelR)
{
  int x, y, k, d;
  float sum;
  for (y=0;y<dataH;y++)
    for (x=0;x<dataW;x++) {
      sum = 0;
      for (k=-kernelR;k<=kernelR;k++) {
        d = y + k;
        d = (d<0 ? 0 : d);
        d = (d>=dataH ? dataH-1 : d);
        sum += h_Data[d*dataW + x] * h_Kernel[kernelR - k];
      }
      h_Result[y*dataW + x] = sum;
    }
}
