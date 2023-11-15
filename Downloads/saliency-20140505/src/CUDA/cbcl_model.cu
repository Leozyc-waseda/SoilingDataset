/*
 *  Modified from MIT CBCL SVN repository
 *   Author: Sharat Chikkerur
 */


#include <cuda.h>
#include "CUDA/cutil.h"
#include <stdio.h>
#include <assert.h>
#include "cbcl_model.h"

#define BLOCK_SIZE 16 


const float* min_element(const float* start,const float* end)
{

	const float* minptr=start;
	float  minval=*start;
	float  val   = minval;
	for(const float* ptr=start;ptr!=end;ptr++)
	{
		val=*ptr;
		if(val<minval)
		{
			minval=val;
			minptr=ptr;
		}
	}
	return minptr;
}


const float* max_element(const float* start,const float* end)
{
	const float* maxptr=start;
	float  maxval=*start;
	float  val   = maxval;
	for(const float* ptr=start;ptr!=end;ptr++)
	{
		val=*ptr;
		if(val>maxval)
		{
			maxval=val;
			maxptr=ptr;
		}
	}
	return maxptr;
}


void cpu_create_filters(band_info** ppfilt,int nfilts, int width, int height, int depth)
{
  *ppfilt = new band_info[nfilts];
  assert(*ppfilt !=NULL);
  for(int i=0;i<nfilts;i++)
    {
      band_info* pfilt = *ppfilt+i;
      pfilt->depth = depth;
      pfilt->height = height;
      pfilt->width = width;
      pfilt->where = ONHOST;
      /*allocate memory for the image*/
      pfilt->pitch=pfilt->width*sizeof(float);
      pfilt->ptr  =new float[pfilt->depth*pfilt->height*pfilt->width];
      assert(pfilt->ptr);
    }
}

int cpu_get_width(band_info** ppfilt, int nfilts, int index)
{
  band_info* pfilt = *ppfilt+index;
  assert(index < nfilts);
  return pfilt->width;
}

int cpu_get_height(band_info** ppfilt, int nfilts, int index)
{
  band_info* pfilt = *ppfilt+index;
  assert(index < nfilts);
  return pfilt->height;
}

int cpu_get_depth(band_info** ppfilt, int nfilts, int index)
{
  band_info* pfilt = *ppfilt+index;
  assert(index < nfilts);
  return pfilt->depth;
}

void cpu_copy_filter(band_info** ppfilt, int nfilts, int index, int x1, int y1, band_info **ppfiltin, int indexin, int x2, int y2)
{
  band_info* pfilt = *ppfilt+index;
  band_info* pfiltin = *ppfiltin+indexin;
  assert(index < nfilts);
  assert(x1 < pfilt->width);
  assert(y1 < pfilt->height);
  assert(x2 < pfiltin->width);
  assert(y2 < pfiltin->height);
  assert(pfilt->width > 0);
  assert(pfilt->height > 0);
  assert(pfilt->depth > 0);
  assert(pfiltin->width > 0);
  assert(pfiltin->height > 0);
  assert(pfiltin->depth > 0);

  assert(pfiltin->depth == pfilt->depth);

  for(int d=0;d<pfilt->depth;d++)
    {
      float* ptr=pfilt->ptr+d*pfilt->height*pfilt->width;  
      float* ptrin=pfiltin->ptr+d*pfiltin->height*pfiltin->width;  
      ptr[y1*pfilt->width+x1] = ptrin[y2*pfiltin->width+x2];
    }
}

void cpu_read_filters(const char *fname, band_info** ppfilt,int* pnfilts)
{
  /*read number of filters*/
  int num_filters=0;

  FILE *fp;
 
  fp = fopen(fname, "r");
  if(!fp) exit(0); 
  fscanf(fp,"%d",&num_filters);

  printf("Loading %d filters\n",num_filters);
  assert(num_filters >= 1);
  *pnfilts= num_filters;
  *ppfilt = new band_info[num_filters];
  assert(*ppfilt !=NULL);

  for(int i=0;i<num_filters;i++)
    {
      band_info* pfilt = *ppfilt+i;
      fscanf(fp,"%d",&(pfilt->depth));
      fscanf(fp,"%d",&(pfilt->height));
      fscanf(fp,"%d",&(pfilt->width));
      /*allocate memory for the image*/
      pfilt->pitch=pfilt->width*sizeof(float);
      pfilt->ptr  =new float[pfilt->depth*pfilt->height*pfilt->width];
      pfilt->where=ONHOST;
      assert(pfilt->ptr);
      for(int d=0;d<pfilt->depth;d++)
	{
	  float* ptr=pfilt->ptr+d*pfilt->height*pfilt->width;
	  for(int y=0;y<pfilt->height;y++)
	    {
	      for(int x=0;x<pfilt->width;x++)
		fscanf(fp,"%f",&(ptr[y*pfilt->width+x]));
	    }
	}
    }
  fclose(fp);

}

void cpu_write_filters(band_info* pfilt,int nfilts,const char* filename)
{
  FILE *fp;
  fp = fopen(filename,"w");
  assert(nfilts>=1);
  assert(pfilt!=NULL);

  fprintf(fp,"%d\n",nfilts);

  for(int i=0;i<nfilts;i++,pfilt++)
    {
      fprintf(fp,"%d\n",pfilt->depth);
      fprintf(fp,"%d\n",pfilt->height);
      fprintf(fp,"%d\n",pfilt->width);

      for(int d=0;d<pfilt->depth;d++)
        {
	  //float* ptr=pfilt->ptr+d*pfilt->height*pfilt->width;
	  for(int y=0;y<pfilt->height;y++)
            {
	      for(int x=0;x<pfilt->width;x++)
		fprintf(fp,"%.8f ",*elptr(pfilt->ptr,d,y,x,pfilt->height,pfilt->pitch));
	      fprintf(fp,"\n");
            }
        }
    }
  fclose(fp);
}

/*
  image texture
*/


texture<float,2,cudaReadModeElementType> teximg;
texture<float,2,cudaReadModeElementType> texfilt;

__host__ __device__ float* elptr(float* base,int depth,int row,int col,int height,int pitch)
{
	return (float*)((char*)base+depth*height*pitch+row*pitch)+col;
}

__global__ void kernel_c_local(float* dest,int pitch,int depth,float wt,float ht,int poolxy,int stepxy,int srcwidth,int srcheight,float xscale,float yscale)
{
  int col       = blockIdx.x*BLOCK_SIZE+threadIdx.x;
  int row       = blockIdx.y*BLOCK_SIZE+threadIdx.y; 
  int x         = col*stepxy;
  int y         = row*stepxy;
  if(row>=ht) return;
  if(col>=wt) return;
  for(int d=0;d<depth;d++)
    {
      float maxval  = 0;
      float pixval  = 0;
      int   istride = d*srcheight;
      for(int v=0;v<poolxy;v++)
	for(int u=0;u<poolxy;u++)
	  {
	    pixval=tex2D(teximg,(x+u)*xscale,istride+(y+v)*yscale);
	    maxval=maxval>pixval?maxval:pixval;
	  }
      float* ptr = elptr(dest,d,row,col,ht,pitch);
      if(*ptr<maxval)
	*ptr = maxval;
    }
}

void gpu_c_local(
		 IN  band_info* sin,     /*pointer to DEVICE storage*/
		 IN  int in_bands,     /*number of input bands*/
		 IN  int pool_xy,      /*spatial pooling: subsampling by pool_xy/2*/
		 IN  int step_xy,      /*spatial subsampling factor*/
		 IN  int pool_scale,   /*scale wise pooling: out_bands=in_bands/pool_scale*/
		 IN  int step_scale,   /*scale incremenet step*/
		 OUT band_info** c,   /*pointer to DEVICE storage*/
		 OUT int* pout_bands,   /*number of output bands*/
		 IN  bool copy
		 )
{
  cudaArray*				imgarray;
  band_info*				h_outbands;
  float*                   d_ptr;
  cudaMemcpyKind           copydir=cudaMemcpyHostToDevice;
  int i,o,b;
  size_t pitch;
  int out_bands = (in_bands-pool_scale)/step_scale+1;
  /*stage output*/
  h_outbands = new band_info[out_bands];
  assert(h_outbands!=NULL);
  for(i=0,o=0;i<=in_bands-pool_scale;i+=step_scale,o++)
    {
      h_outbands[o].height = (sin[i].height-pool_xy)/step_xy+1;
      h_outbands[o].width  = (sin[i].width-pool_xy)/step_xy+1;
      h_outbands[o].depth  = sin[i].depth;
      CUDA_SAFE_CALL(cudaMallocPitch((void**)&d_ptr,&pitch,h_outbands[o].width*sizeof(float),h_outbands[o].depth*h_outbands[o].height));
      CUDA_SAFE_CALL(cudaMemset(d_ptr,0,pitch*h_outbands[o].depth*h_outbands[o].height));
      h_outbands[o].pitch = pitch;
      h_outbands[o].ptr   = d_ptr;
      h_outbands[o].where = ONDEVICE;
    }
  *pout_bands   = out_bands;
  *c            = h_outbands;

  cudaChannelFormatDesc	imgdesc=cudaCreateChannelDesc<float>();
  /*bind the texture*/
  teximg.addressMode[0] = cudaAddressModeClamp;
  teximg.addressMode[1] = cudaAddressModeClamp;
  teximg.filterMode     = cudaFilterModePoint; 
  teximg.normalized     = false;
		
  /*copy image*/ 
  for(i=0,o=0;i<=in_bands-pool_scale;i+=step_scale,o++)
    {
      for(b=0;b<pool_scale;b++)
	{
	  copydir = cudaMemcpyHostToDevice;
	  if(sin[i+b].where==ONDEVICE)
	    copydir=cudaMemcpyDeviceToDevice;
	  CUDA_SAFE_CALL(cudaMallocArray(&imgarray,&imgdesc,sin[i+b].width,sin[i+b].height*sin[i+b].depth));
	  assert(imgarray!=NULL);
	  CUDA_SAFE_CALL(cudaMemcpy2DToArray(imgarray,0,0,
					     sin[i+b].ptr,sin[i+b].pitch,
					     sin[i+b].width*sizeof(float),sin[i+b].height*sin[i+b].depth,
					     copydir));
	  CUDA_SAFE_CALL(cudaBindTextureToArray(teximg,imgarray));
	  /*call the kernel*/
	  uint3 gridsz	 = make_uint3(ceilf((float)h_outbands[o].width/BLOCK_SIZE),ceilf((float)h_outbands[o].height/BLOCK_SIZE),1);
	  uint3 blocksz	 = make_uint3(BLOCK_SIZE,BLOCK_SIZE,1);
	  kernel_c_local<<<gridsz,blocksz>>>(h_outbands[o].ptr,h_outbands[o].pitch,
					     h_outbands[o].depth,h_outbands[o].width,h_outbands[o].height,
					     pool_xy,step_xy,
					     sin[i+b].width,sin[i+b].height,
					     (float)sin[i+b].width/sin[i].width,(float)sin[i+b].height/sin[i].height);
	  CUDA_SAFE_CALL(cudaThreadSynchronize());
	  CUDA_SAFE_CALL(cudaUnbindTexture(teximg));
	  CUDA_SAFE_CALL(cudaFreeArray(imgarray));
	}
    }
  /*copy image back to host*/
  for(b=0;copy && b<out_bands;b++)
    {
      int    sz  = h_outbands[b].height*h_outbands[b].width*h_outbands[b].depth;
      float* ptr = new float[sz];
      assert(ptr!=NULL);
      CUDA_SAFE_CALL(cudaMemcpy2D(ptr,h_outbands[b].width*sizeof(float),
				  h_outbands[b].ptr,h_outbands[b].pitch,
				  h_outbands[b].width*sizeof(float),h_outbands[b].height*h_outbands[b].depth,
				  cudaMemcpyDeviceToHost));
      CUDA_SAFE_CALL(cudaFree(h_outbands[b].ptr));
      h_outbands[b].ptr   =ptr;
      h_outbands[b].pitch =h_outbands[b].width*sizeof(float);
      h_outbands[b].where =ONHOST;
    }
  CUDA_SAFE_CALL(cudaThreadSynchronize());
}


void cpu_release_images(band_info** ppbands,int num_bands)
{
  for(int i=0;i<num_bands;i++)
    {
      band_info* pb=*ppbands+i;
      if(pb->where==ONHOST)
	delete[] pb->ptr;
      else
	CUDA_SAFE_CALL(cudaFree(pb->ptr));
    }
  delete [] *ppbands;
  *ppbands = NULL;
}

__global__ void kernel_resize(float *dest,int pitch,int wt,int ht,float xscale,float yscale)
{
  int row=blockIdx.y*BLOCK_SIZE+threadIdx.y;
  int col=blockIdx.x*BLOCK_SIZE+threadIdx.x;
  if(row>=ht) return;
  if(col>=wt) return;
  float* ptr      = elptr(dest,0,row,col,ht,pitch);
  *ptr            = tex2D(teximg,(float)col*xscale,(float)row*yscale);
}

void gpu_create_c0(const float* pimg,int width,int height,band_info** ppc,int* pbands,float scale,int num_scales,bool copy)
{
  size_t  pitch          = 0;
  *ppc				   = new band_info[num_scales];
  *pbands				   = num_scales;
  assert(*ppc!=NULL);
  assert(*pbands>=1);
  float* ptr               = NULL;
  CUDA_SAFE_CALL(cudaMallocPitch((void**)&ptr,&pitch,width*sizeof(float),height));
  assert(ptr);
  CUDA_SAFE_CALL(cudaMemcpy2D(ptr,pitch,pimg,width*sizeof(float),width*sizeof(float),height,cudaMemcpyHostToDevice));
  /*create first band*/
  (*ppc)->height           = height;
  (*ppc)->width            = width;
  (*ppc)->depth            = 1;
  (*ppc)->pitch            = pitch;
  (*ppc)->ptr              = ptr;
  cudaArray*               pdimg;
  cudaChannelFormatDesc	 imgdesc=cudaCreateChannelDesc<float>();
  /*bind the texture*/
  teximg.addressMode[0] = cudaAddressModeClamp;
  teximg.addressMode[1] = cudaAddressModeClamp;
  teximg.filterMode     = cudaFilterModeLinear;
  teximg.normalized     = false;
  /*copy to array*/
  for(int b=1;b<num_scales;b++)
    {
      band_info* prev          = *ppc+b-1;
      band_info* pc            = *ppc+b;
      int bht			         = roundf(prev->height/scale);
      int bwt			         = roundf(prev->width/scale);
      /*map to texture*/
      CUDA_SAFE_CALL(cudaMallocArray(&pdimg,&imgdesc,prev->width,prev->height));
      CUDA_SAFE_CALL(cudaMemcpy2DToArray(pdimg,0,0,
					 prev->ptr,prev->pitch,
					 prev->width*sizeof(float),prev->height,
					 cudaMemcpyDeviceToDevice));
      CUDA_SAFE_CALL(cudaBindTextureToArray(teximg,pdimg));
      /*allocate the memory*/
      float* pdout             = NULL;
      CUDA_SAFE_CALL(cudaMallocPitch((void**)&pdout,&pitch,bwt*sizeof(float),bht));
      CUDA_SAFE_CALL(cudaMemset(pdout,0,bht*pitch));
      pc->height		         = bht;
      pc->width		         = bwt;
      pc->pitch		         = pitch;
      pc->depth		         = 1;
      pc->ptr                  = pdout;
      pc->where                = ONDEVICE;
      /*call the kernel*/
      uint3 gridsz	 = make_uint3(ceilf((float)bwt/BLOCK_SIZE),ceilf((float)bht/BLOCK_SIZE),1);
      uint3 blocksz	 = make_uint3(BLOCK_SIZE,BLOCK_SIZE,1);
      kernel_resize<<<gridsz,blocksz>>>(pdout,pitch,bwt,bht,(float)prev->width/bwt,(float)prev->height/bht);
      CUDA_SAFE_CALL(cudaThreadSynchronize());
      CUDA_SAFE_CALL(cudaFreeArray(pdimg));
      CUDA_SAFE_CALL(cudaUnbindTexture(teximg));						   
    }
  for(int b=0;copy && b<num_scales;b++)
    {
      band_info* pc =*ppc+b;
      float*     ptr=new float[pc->height*pc->width];
      CUDA_SAFE_CALL(cudaMemcpy2D(ptr,pc->width*sizeof(float),
				  pc->ptr,pc->pitch,
				  pc->width*sizeof(float),pc->height,
				  cudaMemcpyDeviceToHost));
      CUDA_SAFE_CALL(cudaFree(pc->ptr));
      pc->ptr       = ptr;
      pc->pitch     = pc->width*sizeof(float);
      pc->where     = ONHOST;
    }
}

void cpu_create_c0(const float* pimg,int width,int height,band_info** ppc,int* pbands,float scale,int num_scales)
{
  *ppc				   = new band_info[num_scales];
  *pbands				   = num_scales;
  assert(*ppc!=NULL);
  assert(*pbands>=1);
  /*create first band*/
  band_info*  prev         = *ppc;
  prev->height           = height;
  prev->width            = width;
  prev->depth            = 1;
  prev->pitch            = width*sizeof(float);
  prev->ptr              = new float[height*width];
  prev->where            = ONHOST;
  memcpy(prev->ptr,pimg,height*width*sizeof(float));
  assert(prev->ptr);
  /*determine dynamic range*/
  float minval           = *min_element(pimg,&pimg[height*width]);
  float maxval           = *max_element(pimg,&pimg[height*width]);
  float range            = maxval-minval+1e-6;

  /*create the other bands recursively*/
  for(int b=1;b<num_scales;b++,prev++)
    {
      pimg            = prev->ptr;
      width           = prev->width;
      height          = prev->height;
        
      band_info* pc	= *ppc+b;
      int bht			= roundf(height/scale);
      int bwt			= roundf(width/scale);
      pc->height		= bht;
      pc->width		= bwt;
      pc->pitch		= bwt*sizeof(float);
      pc->depth		= 1;
      pc->ptr			= new float[bht*bwt];
      pc->where       = ONHOST;
      assert(pc->ptr!=NULL);
      float cmin      = 1e6; /*current min*/
      float cmax      = 0;   /*current max*/
      for(int x=0;x<bwt;x++)
	{
	  for(int y=0;y<bht;y++)
	    {
	      float sx = x*scale;
	      float sy = y*scale;
	      int   fx = floorf(sx); int  cx = ceilf(sx);cx=(cx>=width)?(width-1):cx;
	      int   fy = floorf(sy); int  cy = ceilf(sy);cy=(cy>=height)?(height-1):cy;
	      float xalpha=sx-fx;
	      float yalpha=sy-fy;
	      float val   =pimg[fx+fy*width]*(1-xalpha)*(1-yalpha)+
		pimg[cx+fy*width]*(xalpha)*(1-yalpha)+
		pimg[fx+cy*width]*(1-xalpha)*(yalpha)+
		pimg[cx+cy*width]*(xalpha)*(yalpha);
	      pc->ptr[y*bwt+x]=val;
	      if(val<cmin) cmin=val;
	      if(val>cmax) cmax=val;
	    }
	}
      float crange = cmax-cmin+1e-6; 
      float factor = range/crange;
      for(int i=0;i<bht*bwt;i++)
	pc->ptr[i]=(pc->ptr[i]-cmin)*factor+minval;
    }
}

template<int u>
__device__ void colNorm(int irow,int frow,int col,float* pnum,float* pden)
{
    float pval=0;
    float fval=0;
    pval = tex2D(teximg,col+u-1,irow);
    fval = tex2D(texfilt,u-1,frow);
    *pnum += pval*fval;
    *pden += pval*pval;
    colNorm<u-1>(irow,frow,col,pnum,pden);
}

template<>
__device__ void colNorm<0>(int irow,int frow,int col,float *pnum,float *pden)
{
    return;
}


__global__  void kernel_s_norm_filter(float* dest,int pitch,int wt,int ht,int srcwidth,int srcheight,int depth,int fsz,int num_filt)
{
    int         row             = blockIdx.y*BLOCK_SIZE+threadIdx.y;
    int         col             = blockIdx.x*BLOCK_SIZE+threadIdx.x;
    int         fstride         = fsz*depth;
	int u,v,d,l,f,iy,fy;
    float       den             = 0;
    float       num             = 0;
    float       pixval          = 0;
    float       filtval         = 0;
    int         nloops          = fsz/4;
    int         residue         = fsz%4;
    if(row>=ht) return;
    if(col>=wt) return;
    for(f=0;f<num_filt;f++)
    {
        den = 1e-6;
        num = 0;
        for(d=0;d<depth;d++)
        {
            iy = d*srcheight+row;
            fy = d*fsz+f*fstride;
            switch(fsz)
            {
                case 5:
                    v = 0;
                    colNorm<5>(iy+v,fy+v,col,&num,&den);v++;
                    colNorm<5>(iy+v,fy+v,col,&num,&den);v++;
                    colNorm<5>(iy+v,fy+v,col,&num,&den);v++;
                    colNorm<5>(iy+v,fy+v,col,&num,&den);v++;
                    colNorm<5>(iy+v,fy+v,col,&num,&den);v++;
                 case 7:
                    v = 0;
                    colNorm<7>(iy+v,fy+v,col,&num,&den);v++;
                    colNorm<7>(iy+v,fy+v,col,&num,&den);v++;
                    colNorm<7>(iy+v,fy+v,col,&num,&den);v++;
                    colNorm<7>(iy+v,fy+v,col,&num,&den);v++;
                    colNorm<7>(iy+v,fy+v,col,&num,&den);v++;
                    colNorm<7>(iy+v,fy+v,col,&num,&den);v++;
                    colNorm<7>(iy+v,fy+v,col,&num,&den);v++;
                    break;
                default:
                    for(v=0;v<fsz;v++,iy++,fy++)
                    {
                        u =0;
                        for(l=0;l<nloops;l++)
                        {
                            //1
                            pixval =tex2D(teximg,col+u,iy);
                            filtval=tex2D(texfilt,u,fy);
                            num    +=filtval*pixval;
                            den    +=pixval*pixval;
                            u++;

                            //2
                            pixval =tex2D(teximg,col+u,iy);
                            filtval=tex2D(texfilt,u,fy);
                            num    +=filtval*pixval;
                            den    +=pixval*pixval;
                            u++;

                            //3
                            pixval =tex2D(teximg,col+u,iy);
                            filtval=tex2D(texfilt,u,fy);
                            num    +=filtval*pixval;
                            den    +=pixval*pixval;
                            u++;

                            //4
                            pixval =tex2D(teximg,col+u,iy);
                            filtval=tex2D(texfilt,u,fy);
                            num    +=filtval*pixval;
                            den    +=pixval*pixval;
                            u++;
                        }
                        switch(residue)
                        {
                            case 3:
                            pixval =tex2D(teximg,col+u,iy);
                            filtval=tex2D(texfilt,u,fy);
                            num    +=filtval*pixval;
                            den    +=pixval*pixval;
                            u++;
                            case 2:
                            pixval =tex2D(teximg,col+u,iy);
                            filtval=tex2D(texfilt,u,fy);
                            num    +=filtval*pixval;
                            den    +=pixval*pixval;
                            u++;
                            case 1:
                            pixval =tex2D(teximg,col+u,iy);
                            filtval=tex2D(texfilt,u,fy);
                            num    +=filtval*pixval;
                            den    +=pixval*pixval;
                            u++;
                            case 0:
                            break;
                        }
                    }//v
                    break;
           }//switch
       }//d
        *elptr(dest,f,row,col,ht,pitch)=fabs(num)/sqrtf(den);
    }
}

/*
put the image into texture memory
put the filter into global memory
call the kernel for each band of the input (maybe change later)
*/
void gpu_s_norm_filter(band_info* cin,int in_bands,band_info* filt,int num_filt, band_info** pps, int *out_bands,bool copy)
{
   cudaArray*				imgarray;
   cudaArray*               filtarray;
   band_info*				h_outbands;
   float*					d_ptr;
   cudaMemcpyKind           copydir;
   size_t                   pitch;
   /*channel description*/
   
   /*determine largest filter size*/
   int fsz             = filt[0].width;
   int fdepth          = filt[0].depth;
   /*stage output*/
   h_outbands = new band_info[in_bands];
   for(int b=0;b<in_bands;b++)
   {
		h_outbands[b].height = cin[b].height-fsz+1;
		h_outbands[b].width  = cin[b].width-fsz+1;
		h_outbands[b].depth  = num_filt;
        CUDA_SAFE_CALL(cudaMallocPitch((void**)&d_ptr,&pitch,h_outbands[b].width*sizeof(float),num_filt*h_outbands[b].height));
        assert(d_ptr!=NULL);
		CUDA_SAFE_CALL(cudaMemset(d_ptr,0,pitch*num_filt*h_outbands[b].height));
		h_outbands[b].pitch = pitch;
        h_outbands[b].where = ONDEVICE;
		h_outbands[b].ptr   = d_ptr;
   }
   *pps      = h_outbands;
   *out_bands= in_bands;
	   
   /*copy image*/ 
   cudaChannelFormatDesc	imgdesc=cudaCreateChannelDesc<float>();
   /*fix address modes*/
    teximg.addressMode[0] = cudaAddressModeClamp;
    teximg.addressMode[1] = cudaAddressModeClamp;
    teximg.filterMode     = cudaFilterModePoint;
    teximg.normalized     = false;
    cudaChannelFormatDesc    filtdesc=cudaCreateChannelDesc<float>();
    CUDA_SAFE_CALL(cudaMallocArray(&filtarray,&filtdesc,fsz,num_filt*fsz*fdepth));
    texfilt.addressMode[0] = cudaAddressModeClamp;
    texfilt.addressMode[1] = cudaAddressModeClamp;
    texfilt.filterMode     = cudaFilterModePoint;
    texfilt.normalized     = false;
    /*copy filters*/
    for(int f=0;f<num_filt;f++)
    {
      CUDA_SAFE_CALL(cudaMemcpy2DToArray(filtarray,0,f*fsz*fdepth,
                                         filt[f].ptr,filt[f].width*sizeof(float),
                                         filt[f].width*sizeof(float),filt[f].height*filt[f].depth,
                                         cudaMemcpyHostToDevice));
    }
	CUDA_SAFE_CALL(cudaBindTextureToArray(texfilt,filtarray));
		
   /*call the kernel*/
   for(int b=0;b<in_bands;b++)
   {
	    /*copy to array*/
        copydir     = cudaMemcpyHostToDevice;
        if(cin[b].where==ONDEVICE)
            copydir = cudaMemcpyDeviceToDevice;
        CUDA_SAFE_CALL(cudaMallocArray(&imgarray,&imgdesc,cin[b].width,cin[b].height*cin[b].depth));
        assert(imgarray!=NULL);
		CUDA_SAFE_CALL(cudaMemcpy2DToArray(imgarray,0,0,
										   cin[b].ptr,cin[b].pitch,
										   cin[b].width*sizeof(float),cin[b].height*cin[b].depth,
									       copydir));
	    CUDA_SAFE_CALL(cudaBindTextureToArray(teximg,imgarray));
        uint3 gridsz	 = make_uint3(ceilf((float)h_outbands[b].width/BLOCK_SIZE),ceilf((float)h_outbands[b].height/BLOCK_SIZE),1);
		uint3 blocksz	 = make_uint3(BLOCK_SIZE,BLOCK_SIZE,1);
		kernel_s_norm_filter<<<gridsz,blocksz>>>(h_outbands[b].ptr,
                                                 h_outbands[b].pitch,h_outbands[b].width,h_outbands[b].height,
                                                 cin[b].width,cin[b].height,
                                                 fdepth,fsz,num_filt);
        CUDA_SAFE_CALL(cudaThreadSynchronize());
        CUDA_SAFE_CALL(cudaUnbindTexture(teximg));
        CUDA_SAFE_CALL(cudaFreeArray(imgarray));
   }
   CUDA_SAFE_CALL(cudaUnbindTexture(texfilt));
   for(int b=0;copy && b<in_bands;b++)
   {
       int    sz  = h_outbands[b].height*h_outbands[b].width*num_filt;
       float* ptr = new float[sz];
       assert(ptr!=NULL);
       CUDA_SAFE_CALL(cudaMemcpy2D(ptr,h_outbands[b].width*sizeof(float),
                                   h_outbands[b].ptr,h_outbands[b].pitch,
                                   h_outbands[b].width*sizeof(float),h_outbands[b].height*h_outbands[b].depth,
                                   cudaMemcpyDeviceToHost));
       CUDA_SAFE_CALL(cudaFree(h_outbands[b].ptr));
       h_outbands[b].ptr   =ptr;
       h_outbands[b].pitch =h_outbands[b].width*sizeof(float);
       h_outbands[b].where =ONHOST;
   }
   CUDA_SAFE_CALL(cudaThreadSynchronize());
   /*copy image to output*/   
   CUDA_SAFE_CALL(cudaFreeArray(filtarray));
}

template<int u>
__device__ float colDist(int irow,int frow,int col)
{
    float pval=0;
    float fval=0;
    pval = tex2D(teximg,col+u-1,irow);
    fval = tex2D(texfilt,u-1,frow);
    return (pval-fval)*(pval-fval)+colDist<u-1>(irow,frow,col);
}

template<>
__device__ float colDist<0>(int irow,int frow,int col)
{
    return 0;
}

__global__  void kernel_s_rbf(float* dest,int pitch,int wt,int ht,int srcwidth,int srcheight,int depth,int fsz,int num_filt,float sigma)
{
    int         row             = blockIdx.y*BLOCK_SIZE+threadIdx.y;
    int         col             = blockIdx.x*BLOCK_SIZE+threadIdx.x;
    int         fstride         = fsz*depth;
	int u,v,d,l,iy,fy;
    float       num             = 0;
    float       pixval          = 0;
    float       filtval         = 0;
    int         nloops          = fsz/4;
    int         residue         = fsz%4;
    if(row>=ht) return;
    if(col>=wt) return;
    for(int f=0;f<num_filt;f++)
    {
        num = 0;
        for(d=0;d<depth;d++)
        {
            iy = d*srcheight+row;
            fy = d*fsz+f*fstride;
            switch(fsz)
            {
                case 4:
                    v    = 0;
                    num += colDist<4>(iy+v,fy+v,col);v++;
                    num += colDist<4>(iy+v,fy+v,col);v++;
                    num += colDist<4>(iy+v,fy+v,col);v++;
                    num += colDist<4>(iy+v,fy+v,col);v++;
                    break;
                case 6:
                    v    = 0;
                    num += colDist<6>(iy+v,fy+v,col);v++;
                    num += colDist<6>(iy+v,fy+v,col);v++;
                    num += colDist<6>(iy+v,fy+v,col);v++;
                    num += colDist<6>(iy+v,fy+v,col);v++;
                    num += colDist<6>(iy+v,fy+v,col);v++;
                    num += colDist<6>(iy+v,fy+v,col);v++;
                    break;
                case 8:
                    v    = 0;
                    num += colDist<8>(iy+v,fy+v,col);v++;
                    num += colDist<8>(iy+v,fy+v,col);v++;
                    num += colDist<8>(iy+v,fy+v,col);v++;
                    num += colDist<8>(iy+v,fy+v,col);v++;
                    num += colDist<8>(iy+v,fy+v,col);v++;
                    num += colDist<8>(iy+v,fy+v,col);v++;
                    num += colDist<8>(iy+v,fy+v,col);v++;
                    num += colDist<8>(iy+v,fy+v,col);v++;
                 default:
                    for(v=0;v<fsz;v++,iy++,fy++)
                    {
                        for(l=0;l<nloops;l++)
                        {
                            pixval =tex2D(teximg,col+u,iy);
                            filtval=tex2D(texfilt,u,fy);
                            num    +=(filtval-pixval)*(filtval-pixval);
                            u++;

                            pixval =tex2D(teximg,col+u,iy);
                            filtval=tex2D(texfilt,u,fy);
                            num    +=(filtval-pixval)*(filtval-pixval);
                            u++;

                            pixval =tex2D(teximg,col+u,iy);
                            filtval=tex2D(texfilt,u,fy);
                            num    +=(filtval-pixval)*(filtval-pixval);
                            u++;

                            pixval =tex2D(teximg,col+u,iy);
                            filtval=tex2D(texfilt,u,fy);
                            num    +=(filtval-pixval)*(filtval-pixval);
                            u++;
                         }
                         switch(residue)
                         {
                            case 3:
                            pixval =tex2D(teximg,col+u,iy);
                            filtval=tex2D(texfilt,u,fy);
                            num    +=(filtval-pixval)*(filtval-pixval);
                            u++;
                            case 2:
                            pixval =tex2D(teximg,col+u,iy);
                            filtval=tex2D(texfilt,u,fy);
                            num    +=(filtval-pixval)*(filtval-pixval);
                            u++;
                            case 1:
                            pixval =tex2D(teximg,col+u,iy);
                            filtval=tex2D(texfilt,u,fy);
                            num    +=(filtval-pixval)*(filtval-pixval);
                            u++;
                            case 0:
                            break;
                          }
                    }//v
                    break;
            }//switch
        }//d
        *elptr(dest,f,row,col,ht,pitch)=exp(-num/sigma);
    }//f
}

/*
put the image into texture memory
put the filter into global memory
call the kernel for each band of the input (maybe change later)
*/
void gpu_s_rbf(band_info* cin,int in_bands,band_info* filt,int num_filt, float sigma,band_info** pps, int *out_bands,bool copy)
{
   cudaArray*				imgarray;
   cudaArray*               filtarray;
   band_info*				h_outbands;
   float*					d_ptr;
   cudaMemcpyKind           copydir=cudaMemcpyHostToDevice;
   size_t                   pitch;
   /*determine largest filter size*/
   int fsz             = filt[0].width;
   int fdepth          = filt[0].depth;
 
   /*stage output*/
   h_outbands = new band_info[in_bands];
   for(int b=0;b<in_bands;b++)
   {
		h_outbands[b].height = cin[b].height-fsz+1;
		h_outbands[b].width  = cin[b].width-fsz+1;
		h_outbands[b].depth  = num_filt;
		CUDA_SAFE_CALL(cudaMallocPitch((void**)&d_ptr,&pitch,h_outbands[b].width*sizeof(float),num_filt*h_outbands[b].height));
        assert(d_ptr!=NULL);
		CUDA_SAFE_CALL(cudaMemset(d_ptr,0,pitch*num_filt*h_outbands[b].height));
		h_outbands[b].pitch = pitch; 
		h_outbands[b].ptr   = d_ptr;
        h_outbands[b].where = ONDEVICE;
   }
   *pps      = h_outbands;
   *out_bands= in_bands;
	   
   /*copy image*/ 
   cudaChannelFormatDesc	imgdesc=cudaCreateChannelDesc<float>();
   /*fix address modes*/
    teximg.addressMode[0] = cudaAddressModeClamp;
    teximg.addressMode[1] = cudaAddressModeClamp;
    teximg.filterMode     = cudaFilterModePoint;
    teximg.normalized     = false;
    
    cudaChannelFormatDesc    filtdesc=cudaCreateChannelDesc<float>();
    CUDA_SAFE_CALL(cudaMallocArray(&filtarray,&filtdesc,fsz,num_filt*fsz*fdepth));
    texfilt.addressMode[0] = cudaAddressModeClamp;
    texfilt.addressMode[1] = cudaAddressModeClamp;
    texfilt.filterMode     = cudaFilterModePoint;
    texfilt.normalized     = false;

    sigma                  = 2*sigma*sigma;
    /*copy filters*/
    for(int f=0;f<num_filt;f++)
    {
      CUDA_SAFE_CALL(cudaMemcpy2DToArray(filtarray,0,f*fsz*fdepth,
                                         filt[f].ptr,filt[f].width*sizeof(float),
                                         filt[f].width*sizeof(float),filt[f].height*filt[f].depth,
                                         cudaMemcpyHostToDevice));
    }
	CUDA_SAFE_CALL(cudaBindTextureToArray(texfilt,filtarray));
	/*call the kernel*/
   for(int b=0;b<in_bands;b++)
   {
        copydir = cudaMemcpyHostToDevice;
        if(cin[b].where==ONDEVICE)
            copydir=cudaMemcpyDeviceToDevice;
        CUDA_SAFE_CALL(cudaMallocArray(&imgarray,&imgdesc,cin[b].width,cin[b].height*cin[b].depth));
        assert(imgarray!=NULL);
	    /*copy to array*/
		CUDA_SAFE_CALL(cudaMemcpy2DToArray(imgarray,0,0,
										   cin[b].ptr,cin[b].pitch,
										   cin[b].width*sizeof(float),cin[b].height*cin[b].depth,
									       copydir));
	    CUDA_SAFE_CALL(cudaBindTextureToArray(teximg,imgarray));
		uint3 gridsz	 = make_uint3(ceilf((float)h_outbands[b].width/BLOCK_SIZE),ceilf((float)h_outbands[b].height/BLOCK_SIZE),1);
		uint3 blocksz	 = make_uint3(BLOCK_SIZE,BLOCK_SIZE,1);
		kernel_s_rbf<<<gridsz,blocksz>>>(h_outbands[b].ptr,
                                         h_outbands[b].pitch,h_outbands[b].width,h_outbands[b].height,
                                         cin[b].width,cin[b].height,
                                         fdepth,fsz,num_filt,sigma);
        CUDA_SAFE_CALL(cudaThreadSynchronize());
        CUDA_SAFE_CALL(cudaUnbindTexture(teximg));
        CUDA_SAFE_CALL(cudaFreeArray(imgarray));
   }
   CUDA_SAFE_CALL(cudaUnbindTexture(texfilt));
   for(int b=0;copy && b<in_bands;b++)
   {
       int    sz  = h_outbands[b].height*h_outbands[b].width*num_filt;
       float* ptr = new float[sz];
       assert(ptr!=NULL);
       CUDA_SAFE_CALL(cudaMemcpy2D(ptr,h_outbands[b].width*sizeof(float),
                                   h_outbands[b].ptr,h_outbands[b].pitch,
                                   h_outbands[b].width*sizeof(float),h_outbands[b].height*h_outbands[b].depth,
                                   cudaMemcpyDeviceToHost));
       CUDA_SAFE_CALL(cudaFree(h_outbands[b].ptr));
       h_outbands[b].ptr   =ptr;
       h_outbands[b].pitch =h_outbands[b].width*sizeof(float);
       h_outbands[b].where =ONHOST;
   }
   CUDA_SAFE_CALL(cudaThreadSynchronize());
   /*copy image to output*/   
   CUDA_SAFE_CALL(cudaFreeArray(filtarray));
}

void cpu_c_global(
	IN band_info* s,      /*pointer to device storage*/
	IN int in_bands,      /*number of input bands*/
	OUT float** ppc,          /*pointer to DEVICE storage*/
	OUT int* out_units   /*=input depth*/	
)
{
	*out_units = s[0].depth;
	*ppc       = new float[*out_units];
	assert(*ppc);
	
	float* pc  = *ppc;
	memset(pc,0,sizeof(float)*(*out_units));

	for(int d=0;d<s[0].depth;d++)
	{
		for(int b=0;b<in_bands;b++)
		{
			int    numel  = s[b].height*s[b].width;
			float* ptr    = s[b].ptr+d*numel;
			const float* pmaxval= max_element(ptr,ptr+numel);
			pc[d]         = max(*pmaxval,pc[d]);
		}
	}
}


