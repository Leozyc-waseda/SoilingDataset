/*
Copyright 2010, Ming-Yu Liu

All Rights Reserved 

Permission to use, copy, modify, and distribute this software and 
its documentation for any non-commercial purpose is hereby granted 
without fee, provided that the above copyright notice appear in 
all copies and that both that copyright notice and this permission 
notice appear in supporting documentation, and that the name of 
the author not be used in advertising or publicity pertaining to 
distribution of the software without specific, written prior 
permission. 

THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, 
INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR 
ANY PARTICULAR PURPOSE. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR 
ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES 
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN 
AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING 
OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE. 

Lior Elazary: Changed to work with the INVT
*/



#include "LFLineFitter.h"


LFLineFitter::LFLineFitter()
{
	localWindSize_ = 50;
	smallLocalWindowSize_ = max(localWindSize_/10,5);
	nMinEdges_ = 2;
	nMaxWindPoints_ = 4*(localWindSize_+1)*(localWindSize_+1);
	minLength_ = 2;

	nLinesToFitInStage_[0] = 300;
	nLinesToFitInStage_[1] = 2000;
	nTrialsPerLineInStage_[0] = 300;
	nTrialsPerLineInStage_[1] = 1;
	sigmaFitALine_ = 0.75;
	sigmaFindSupport_ = 0.75;
	maxGap_ = 1.5;

	outEdgeMap_ = NULL;
	rpoints_ = NULL;
	rProjection_ = NULL;
	absRProjection_ = NULL;
	idx_ = NULL;
}

void LFLineFitter::Configure(double sigmaFitALine,double sigmaFindSupport, double maxGap,
	int nLayer,int *nLinesToFitInStage,int *nTrialsPerLineInStage)
{
	sigmaFitALine_ = sigmaFitALine;
	sigmaFindSupport_ = sigmaFindSupport;
	maxGap_ = maxGap;
	for(int i=0;i<nLayer;i++)
	{
		nLinesToFitInStage_[i] = nLinesToFitInStage[i];
		nTrialsPerLineInStage_[i] = nTrialsPerLineInStage[i];
	}
}

void LFLineFitter::Configure(const char *fileName)
{

	for(int i=0;i<50;i++)
		cout<<"*";
	cout<<endl;

	cout<<" LFLineFitting"<<endl;

	for(int i=0;i<50;i++)
		cout<<"*";
	cout<<endl;


	char str[256];
	FILE *fp=NULL;
	fp = fopen(fileName,"rt");
	if(fp==NULL)
	{
		cerr<<"[ERROR] Cannot read file "<<fileName<<"\n!!!";
		exit(0);
	}
	
	cout<<"Load Configuration from "<<fileName<<endl;
	
  int ret=0;
	ret=fscanf(fp,"%s\n",str);
	cout<<str<<" = ";
	ret=fscanf(fp,"%s\n",str);
	sigmaFitALine_ = atof(str);
	cout<<sigmaFitALine_<<endl;

	ret=fscanf(fp,"%s\n",str);
	cout<<str<<" = ";
	ret=fscanf(fp,"%s\n",str);
	sigmaFindSupport_ = atof(str);
	cout<<sigmaFindSupport_<<endl;

	ret=fscanf(fp,"%s\n",str);
	cout<<str<<" = ";
	ret=fscanf(fp,"%s\n",str);
	maxGap_ = atof(str);
	cout<<maxGap_<<endl;

	for(int i=0;i<2;i++)
	{
		ret=fscanf(fp,"%s\n",str);
		cout<<str<<" = ";
		ret=fscanf(fp,"%s\n",str);
		nLinesToFitInStage_[i] = (int)atof(str);
		cout<<nLinesToFitInStage_[i]<<endl;

		ret=fscanf(fp,"%s\n",str);
		cout<<str<<" = ";
		ret=fscanf(fp,"%s\n",str);
		nTrialsPerLineInStage_[i] = (int)atof(str);
		cout<<nTrialsPerLineInStage_[i]<<endl;
	}

	fclose(fp);
	cout<<endl<<endl;
}


LFLineFitter::~LFLineFitter()
{
	SafeRelease();
}

void LFLineFitter::SafeRelease()
{
	if(outEdgeMap_)
		delete [] outEdgeMap_;
	if(rpoints_)
		delete [] rpoints_;
	if(rProjection_)
		delete [] rProjection_;
	if(absRProjection_)
		delete [] absRProjection_;
	if(idx_)
		delete [] idx_;
	outEdgeMap_ = NULL;
	rpoints_ = NULL;
	rProjection_ = NULL;
	absRProjection_ = NULL;
	idx_ = NULL;


}

void LFLineFitter::Init()
{
	outEdgeMap_ = new LFLineSegment[nLinesToFitInStage_[0]+nLinesToFitInStage_[1]];
	rpoints_ = new CvPoint [nMaxWindPoints_];
	rProjection_ = new double [nMaxWindPoints_];
	absRProjection_ = new double [nMaxWindPoints_];
	idx_ = new int [nMaxWindPoints_];
	
}

void LFLineFitter::FitLine(IplImage *inputImage)
{
	//long int t1, t2, f;
	//QueryPerformanceFrequency(&f);
	//QueryPerformanceCounter(&t1);


  //cvNamedWindow("debug",1);
  //cvShowImage("debug",inputImage);
  //cvWaitKey(0);

	width_ = inputImage->width;
	height_ = inputImage->height;

	map<int,CvPoint> edgeMap;

	int i,j,k;
	int x0,y0;
	int width,height;
	int index=0;
	int nPixels=0;
	int nEdges=0;
	int maxSupport=0;	
	LFLineSegment tmpLs,bestLs;
	CvPoint2D64f lnormal;
	int nWindPoints=0,nWaitingKillingList=0,nProposedKillingList=0;
	CvPoint *windPoints,*waitingKillingList,*proposedKillingList;
	windPoints = new CvPoint [nMaxWindPoints_];
	waitingKillingList = new CvPoint[nMaxWindPoints_];
	proposedKillingList = new CvPoint[nMaxWindPoints_];

	width = inputImage->width;
	height = inputImage->height;
	nPixels = width*height;

	for(int y=0;y<height;y++)
	{
		for(int x=0;x<width;x++)
		{
			i = x + y*width;
			if(cvGetReal1D(inputImage,i)!=0)
			{
				edgeMap.insert(pair<int,CvPoint>(i,cvPoint(x,y)));
				nEdges++;
			}
		}
	}


	nInputEdges_ = nEdges;
	nLineSegments_=0;
	for(k=0;k<2;k++)
	{
		if(nEdges<nMinEdges_)
			break;
		for(i=0;i<nLinesToFitInStage_[k];i++)
		{
			maxSupport = 0;
			for(j=0;j<nTrialsPerLineInStage_[k];j++)
			{
				// Sample a point
				index = SampleAPixel(&edgeMap,inputImage,nPixels);
				y0 = index/width;
				x0 = index - y0*width;
				
				// Locate the subwindow
				Find(x0,y0,windPoints,nWindPoints,inputImage,smallLocalWindowSize_);				
				
				// Infer a line direction
				FitALine(nWindPoints,windPoints,sigmaFitALine_,lnormal);

				// Locate the subwindow			
				Find(&edgeMap,x0,y0,windPoints,nWindPoints,inputImage,localWindSize_);

				// Find the support			
				FindSupport(nWindPoints,windPoints,lnormal,sigmaFindSupport_,maxGap_,
					tmpLs,proposedKillingList,nProposedKillingList,x0,y0);
				
				// Check if need to update
				if(tmpLs.nSupport_ > maxSupport)
				{
					maxSupport = tmpLs.nSupport_;
					nWaitingKillingList = nProposedKillingList;
					memcpy(waitingKillingList,proposedKillingList,sizeof(CvPoint)*nWaitingKillingList);
					bestLs = tmpLs;
				}
			}

			// Remove points
			for(j=0;j<maxSupport;j++)
			{
				cvSetReal2D(inputImage,waitingKillingList[j].y,waitingKillingList[j].x,0.0);

				edgeMap.erase(waitingKillingList[j].y*width+waitingKillingList[j].x);
			}
			nEdges -= bestLs.nSupport_;
			bestLs.len_ = sqrt( (bestLs.sx_-bestLs.ex_)*(bestLs.sx_-bestLs.ex_) + (bestLs.sy_-bestLs.ey_)*(bestLs.sy_-bestLs.ey_));
			outEdgeMap_[nLineSegments_] = bestLs;
			nLineSegments_++;

			if(nEdges<nMinEdges_)
				break;
		}
	}


	qsort(outEdgeMap_,nLineSegments_,sizeof(LFLineSegment),LFLineSegment::Compare);
	std::cout<<"Num. lines in test image = "<<nLineSegments_<<std::endl;


	delete [] windPoints;
	delete [] waitingKillingList;
	delete [] proposedKillingList;
	edgeMap.clear();

  std::cout<<"Line Ftting Done"<<std::endl;
	//QueryPerformanceCounter(&t2);
	//std::cout<<"Line Fitting Time "<<setiosflags(ios::fixed)<<setprecision(6)<<(t2.QuadPart - t1.QuadPart)/(1.0*f.QuadPart)<<std::endl;
}

void LFLineFitter::SaveEdgeMap(const char *filename)
{
	FILE *fp;
	qsort(outEdgeMap_,nLineSegments_,sizeof(LFLineSegment),LFLineSegment::Compare);
	fp = fopen(filename,"wt");

	fprintf(fp,"%d %d\n",width_,height_);
	fprintf(fp,"%d\n",nLineSegments_);

	double ratio=0;
	double count=0;
	for(int i=0;i<nLineSegments_;i++)
	{
		count += (double)outEdgeMap_[i].nSupport_;
		ratio = count/nInputEdges_;
		fprintf(fp,"%d %d %d %d\n",(int)outEdgeMap_[i].sx_,(int)outEdgeMap_[i].sy_,(int)outEdgeMap_[i].ex_,(int)outEdgeMap_[i].ey_);
	}

	fclose(fp);


}


void LFLineFitter::DisplayEdgeMap(IplImage *inputImage,const char *outputImageName)
{
	// debug
	IplImage *debugImage = cvCreateImage( cvSize(inputImage->width,inputImage->height), IPL_DEPTH_8U,1);
	cvZero(debugImage);

	for(int i=0;i<nLineSegments_;i++)
	{
		cvLine(debugImage,cvPoint((int)outEdgeMap_[i].sx_,(int)outEdgeMap_[i].sy_),
			cvPoint((int)outEdgeMap_[i].ex_,(int)outEdgeMap_[i].ey_),cvScalar(255));
	}

	//if(outputImageName!=NULL)
	//{
	//	printf("Save Image %s\n\n",outputImageName);
	//	cvSaveImage(outputImageName,debugImage);
	//}
	//else
	//{		
	//	cvNamedWindow("debug",1);
	//	cvShowImage("debug",debugImage);
	//	cvWaitKey(0);
	//}

	cvReleaseImage(&debugImage);
}

void LFLineFitter::FindSupport(const int nWindPoints,CvPoint *windPoints,CvPoint2D64f &lnormal,
							 double sigmaFindSupport,double maxGap,LFLineSegment &ls,
							 CvPoint *proposedKillingList,int &nProposedKillingList,int x0,int y0)
{
	int i,j;	
	int nRindices = 0;
	int zeroIndex=0;
	double residuals;
	CvPoint2D64f ldirection;
	
	// Find the point within the threshold by taking dot product
	for(i=0;i<nWindPoints;i++)
	{
		residuals = fabs( windPoints[i].x*lnormal.x + windPoints[i].y*lnormal.y );
		if(residuals<sigmaFindSupport)
		{
			rpoints_[nRindices] = windPoints[i];
			nRindices++;
		}
	}
	ldirection.x = -lnormal.y;
	ldirection.y = lnormal.x;


	if(nRindices<minLength_)
	{
		ls.nSupport_ = -1;
		return;
	}


	// Project to the line
	for(i=0;i<nRindices;i++)
	{
		rProjection_[i] = rpoints_[i].x*ldirection.x+ rpoints_[i].y*ldirection.y;		
		idx_[i] = i;
	}
	
	// Sort the projection and find the starting and ending points	
	ISort(rProjection_,nRindices,idx_);

	for(i=0;i<nRindices;i++)
		absRProjection_[i] = fabs(rProjection_[i]);

	for(i=0;i<nRindices;i++)
	{
		if(absRProjection_[i]==0)
		{
			zeroIndex = i;
			break;
		}
	}
	
	int maxIndex = nRindices-1;
	for( i=zeroIndex; i<(nRindices-1); i++)
	{	
		if((rProjection_[i+1]-rProjection_[i])>maxGap)
		{	
			maxIndex = i;
			break;
		}
	}

	int minIndex = 0;
	for( i=zeroIndex; i>0; i--)
	{	
		if((rProjection_[i]-rProjection_[i-1])>maxGap)
		{	
			minIndex = i;
			break;
		}
	}

	ls.nSupport_ = maxIndex-minIndex+1;
	ls.sx_ = (double)rpoints_[ idx_[minIndex] ].x+x0;
	ls.sy_ = (double)rpoints_[ idx_[minIndex] ].y+y0;
	ls.ex_ = (double)rpoints_[ idx_[maxIndex] ].x+x0;
	ls.ey_ = (double)rpoints_[ idx_[maxIndex] ].y+y0;
	
	j = 0;
	for(i=minIndex;i<=maxIndex;i++)
	{
		proposedKillingList[j].x = rpoints_[ idx_[i] ].x + x0;
		proposedKillingList[j].y = rpoints_[ idx_[i] ].y + y0;
		j++;
	}
	nProposedKillingList = j;

	ls.normal_ = lnormal;
}


int LFLineFitter::FitALine(const int nWindPoints,CvPoint *windPoints,const double sigmaFitALine,CvPoint2D64f &lnormal)
{
	double inlierRatio = 0.9;	
	double outlierRatio = 0.9;
	double gamma = 0.05;	
	int nMaxTry = 29; //ceil(log(0.05)/log(0.9))

	int i=0,j=0,index=0;
	int cscore;
	double tmpScore;
	double norm;
	int bestscore = -1;	
	CvPoint2D64f cdirection,cnormal;

	while(i<nMaxTry)
	{
		index = (int)floor(rand()/(double)RAND_MAX*(double)nWindPoints);
		norm = sqrt( 1.0*windPoints[index].x*windPoints[index].x + windPoints[index].y*windPoints[index].y );
		
		if(norm>0)
		{
			cdirection.x = windPoints[index].x/norm;
			cdirection.y = windPoints[index].y/norm;

			cnormal.x = -cdirection.y;
			cnormal.y = cdirection.x;
			
			cscore = 0;
			for(j=0;j<nWindPoints;j++)
			{
				tmpScore = fabs(windPoints[j].x*cnormal.x+windPoints[j].y*cnormal.y);
				if(tmpScore<sigmaFitALine)
					cscore++;
			}

			if( ((double)cscore)/nWindPoints > inlierRatio)
			{
				bestscore = cscore;
				lnormal = cnormal;
				return bestscore;
			}

			if( (1.0-((double)cscore)/nWindPoints) < outlierRatio)
			{
				outlierRatio = 1.0-((double)cscore)/nWindPoints;
				nMaxTry = (int)ceil(log(gamma)/log(outlierRatio));
			}

			if(cscore>bestscore)
			{
				bestscore = cscore;
				lnormal = cnormal;
			}
		}
		i = i+1;
	}

	return bestscore;
}




void LFLineFitter::Find(int x0,int y0,CvPoint *windPoints,int &nWindPoints,IplImage *inputImage,int localWindSize)
{
	int x,y;
	nWindPoints = 0;

	for(y=max(y0-localWindSize,0);y<min(y0+localWindSize,inputImage->height);y++)
		for(x=max(x0-localWindSize,0);x<min(x0+localWindSize,inputImage->width);x++)
		{
			if(cvGetReal2D(inputImage,y,x)!=0)
			{
				windPoints[nWindPoints].x = x - x0;
				windPoints[nWindPoints].y = y - y0;
				nWindPoints++;
			}
		}
}


void LFLineFitter::Find(map<int,CvPoint> *edgeMap,int x0,int y0,CvPoint *windPoints,int &nWindPoints,IplImage *inputImage,int localWindSize)
{
	nWindPoints = 0;
	map<int,CvPoint>::iterator it;

	int left = max(x0-localWindSize,0);
	int right = min(x0+localWindSize,inputImage->width);
	int top = max(y0-localWindSize,0);
	int bottom = min(y0+localWindSize,inputImage->height);

	for(it = edgeMap->begin();it!=edgeMap->end();++it)
	{
		if( it->second.x > left && it->second.x < right && 
			it->second.y > top && it->second.y < bottom )
		{
			windPoints[nWindPoints].x = it->second.x - x0;
			windPoints[nWindPoints].y = it->second.y - y0;
			nWindPoints++;
		}
	}
}


int LFLineFitter::SampleAPixel(map<int,CvPoint> *edgeMap,IplImage *inputImage,int nPixels)
{
	int index = (int)(floor(rand()/(double)RAND_MAX*(double)(edgeMap->size()-1)));
	map<int,CvPoint>::iterator it;
	it=edgeMap->begin();
	for(int i=0;i<index;i++)
		it++;
	return (*it).first;
}

void LFLineFitter::ISort(double* ra, int nVec, int* ira)
{
   unsigned long n, l, ir, i, j;
   n = nVec;
   double rra;
   int irra;
   
   if (n<2)
      return;
   l = (n>>1)+1;
   ir = n;
   for (;;)
   {
      if (l>1)
      {
         irra = ira[(--l)-1];
         rra = ra[l-1];
      }
      else
      {
         irra = ira[ir-1];
         rra = ra[ir-1];

         ira[ir-1] = ira[1-1];
         ra[ir-1] = ra[1-1];

         if (--ir==1)
         {
            ira[1-1] = irra;
            ra[1-1] = rra;
            break;
         }
      }
      i = l;
      j = l+l;
      while (j<=ir)
      {
         if (j<ir && ra[j-1]<ra[j+1-1])
            j++;
         if (rra<ra[j-1])
         {
            ira[i-1] = ira[j-1];
            ra[i-1] = ra[j-1];

            i = j;
            j <<= 1;
         }
         else
            j = ir+1;
      }
      ira[i-1] = irra;
      ra[i-1] = rra;
   }

}
