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
*/



#include "LMLineMatcher.h"


LMLineMatcher::LMLineMatcher()
{
	dbImages_ = NULL;
	distances_ = new double[5000000];
	indices_ = new int[5000000];
	xIndices_ = new int[5000000];
	yIndices_ = new int[5000000];
	sIndices_ = new double[5000000];
	dIndices_ = new double[5000000];
	iindices_ = new int[5000000];
	wIndices_ = new int[5000000];
	hIndices_ = new int[5000000];
};

LMLineMatcher::~LMLineMatcher()
{
	SafeRelease();
};

void LMLineMatcher::SafeRelease()
{
	delete []	distances_;
	delete []	indices_;
	delete []	xIndices_;
	delete []	yIndices_;
	delete []	dIndices_;
	delete []	iindices_;
	delete []	sIndices_;
	delete []	wIndices_;
	delete []	hIndices_;

	if (dbImages_)
		delete[] dbImages_;
	dbImages_ = NULL;

};

void LMLineMatcher::Configure(const char *fileName)
{

	for(int i=0;i<50;i++)
		cout<<"*";
	cout<<endl;

	cout<<" LMLineMatching"<<endl;

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
	// nDirections
	ret=fscanf(fp,"%s\n",str);
	cout<<str<<" = ";
	ret=fscanf(fp,"%s\n",str);
	nDirections_ = atoi(str);
	cout<<nDirections_<<endl;

	// direction cost
	ret=fscanf(fp,"%s\n",str);
	cout<<str<<" = ";
	ret=fscanf(fp,"%s\n",str);
	directionCost_ = (float)atof(str);
	cout<<directionCost_<<endl;

	// maximum cost
	ret=fscanf(fp,"%s\n",str);
	cout<<str<<" = ";
	ret=fscanf(fp,"%s\n",str);
	maxCost_ = atof(str);
	cout<<maxCost_<<endl;

	// matching Scale
	ret=fscanf(fp,"%s\n",str);
	cout<<str<<" = ";
	ret=fscanf(fp,"%s\n",str);
	scale_ = atof(str);
	cout<<scale_<<endl;


	// db Scale
	ret=fscanf(fp,"%s\n",str);
	cout<<str<<" = ";
	ret=fscanf(fp,"%s\n",str);
	db_scale_ = atof(str);
	cout<<db_scale_<<endl;

	// Base Search Scale
	ret=fscanf(fp,"%s\n",str);
	cout<<str<<" = ";
	ret=fscanf(fp,"%s\n",str);
	baseSearchScale_ = atof(str);
	cout<<baseSearchScale_<<endl;

	// Min Search Scale
	ret=fscanf(fp,"%s\n",str);
	cout<<str<<" = ";
	ret=fscanf(fp,"%s\n",str);
	minSearchScale_ = atof(str);
	cout<<minSearchScale_<<endl;

	// Max Search Scale
	ret=fscanf(fp,"%s\n",str);
	cout<<str<<" = ";
	ret=fscanf(fp,"%s\n",str);
	maxSearchScale_ = atof(str);
	cout<<maxSearchScale_<<endl;


	// Search Step Size
	ret=fscanf(fp,"%s\n",str);
	cout<<str<<" = ";
	ret=fscanf(fp,"%s\n",str);
	searchStepSize_ = atoi(str);
	cout<<searchStepSize_<<endl;

	// Search Boundary Size
	ret=fscanf(fp,"%s\n",str);
	cout<<str<<" = ";
	ret=fscanf(fp,"%s\n",str);
	searchBoundarySize_ = atoi(str);
	cout<<searchBoundarySize_<<endl;

	// Min Cost Ratio
	ret=fscanf(fp,"%s\n",str);
	cout<<str<<" = ";
	ret=fscanf(fp,"%s\n",str);
	minCostRatio_ = atof(str);
	cout<<minCostRatio_<<endl;


	fclose(fp);
	cout<<endl<<endl;
}


void LMLineMatcher::Init(const char* fileName,double db_scale)
{
	char imageName[512];

	db_scale_ = db_scale;
	
	FILE* fin=NULL;

	fin = fopen(fileName, "r");
	if(fileName==NULL)
	{
		cerr<<"[ERROR] Cannot read file "<<fileName<<"\n!!!";
		exit(0);
	}
  int ret = 0;
	ret=fscanf(fin, "%d", &ndbImages_);
	dbImages_ = new EIEdgeImage [ndbImages_];

	std::cout<<"Num. templates = "<<ndbImages_<<std::endl;

	for (int i=0;i<ndbImages_ ;i++)
	{
		ret=fscanf(fin, "%s\n", (char*)imageName);
		dbImages_[i].SetNumDirections(nDirections_);
		dbImages_[i].Read(imageName);		
		dbImages_[i].Scale(scale_*db_scale_);
	}

	fclose(fin);
}


void LMLineMatcher::computeIDT3(int width, int height, int nLines, LFLineSegment* linesSegment)
{

	queryImage_.SetNumDirections(nDirections_);
	queryImage_.Read(width, height, nLines, linesSegment);
	queryImage_.Scale(scale_);
	queryDistanceImage_.Configure(directionCost_,maxCost_);
	queryDistanceImage_.SetImage(queryImage_);

	std::cout<<"IDT3 Computation done " << std::endl;

}



std::vector<LMLineMatcher::Rect> LMLineMatcher::Match(int width, int height, int nLines, LFLineSegment* linesSegment)
{
	//LARGE_INTEGER t1, t2, f;
	//QueryPerformanceFrequency(&f);
	//QueryPerformanceCounter(&t1);



	queryImage_.SetNumDirections(nDirections_);
	queryImage_.Read(width, height, nLines, linesSegment);
	queryImage_.Scale(scale_);
	queryDistanceImage_.Configure(directionCost_,maxCost_);
	queryDistanceImage_.SetImage(queryImage_);

	std::cout<<"IDT3 Computation done " << std::endl;


	double minCost = 1e+10;
	int counter = 0;
	for (int i=0 ; i<ndbImages_ ; i++)
	{		
		MatchBruteForce(dbImages_[i], i, iindices_,
        indices_, xIndices_, yIndices_, dIndices_, sIndices_, distances_,
        counter, minCost);		
	}

	//QueryPerformanceCounter(&t2);
	//std::cout<<"Fast Directional Chamfer Matching Time "<<setiosflags(ios::fixed)<<setprecision(6)<<(t2.QuadPart - t1.QuadPart)/(1.0*f.QuadPart)<<std::endl;	
	std::cout<<"Fast Directional Chamfer Matching DOne "<<std::endl;	

	ISort(distances_, counter, iindices_);


	// Display best matcher in edge map
  std::vector<Rect> matches;
  for(int i=0; i<counter; i++)
  {
    int x = (int)ceil(xIndices_[iindices_[i]]/scale_-0.5);
    int y = (int)ceil(yIndices_[iindices_[i]]/scale_-0.5);
    double scale = sIndices_[iindices_[i]];
    int detWindWidth= (int)(dbImages_[indices_[iindices_[i]]].width_*scale/scale_);
    int detWindHeight= (int)(dbImages_[indices_[iindices_[i]]].height_*scale/scale_);
    //printf("MATCH: %i %f %i %i %i %i\n", i, distances_[i], 
    //    x, y, detWindWidth, detWindHeight);
    matches.push_back(Rect(x,y,detWindWidth, detWindHeight, distances_[i]));


    if (imageName_.size() > 0)
    {
      //IplImage *debugImage = cvLoadImage(imageName_.c_str(),1);
      //DrawDetWind(debugImage,x,y,detWindWidth,detWindHeight);
      //cvNamedWindow("output",0);
      //cvShowImage("output",debugImage);
      //cvWaitKey(0);
      //cvReleaseImage(&debugImage);
    }
  }
  return matches;
}

void LMLineMatcher::DrawMatchTemplate(IplImage *image,EIEdgeImage &ei,int x,int y,double scale,CvScalar scalar,int thickness)
{
	EIEdgeImage tdbImage;
	tdbImage = ei;
	tdbImage.Scale(scale);
	LFLineSegment line;
	double ltrans[2];
	ltrans[0] = 1.0*x;
	ltrans[1] = 1.0*y;
	for (int k=0 ; k<tdbImage.nLines_ ; k++)
	{
		line = tdbImage.lines_[k];
		line.Translate(ltrans);	
		cvLine(image,cvPoint((int)line.sx_,(int)line.sy_),cvPoint((int)line.ex_,(int)line.ey_),scalar,thickness);
	}
}

void LMLineMatcher::DrawDetWind(IplImage *image,int x,int y,int detWindWidth,int detWindHeight,CvScalar scalar,int thickness)
{
	cvLine(image,cvPoint( x,y),cvPoint( x+detWindWidth,y),scalar,thickness);
	cvLine(image,cvPoint( x+detWindWidth,y),cvPoint( x+detWindWidth, y+detWindHeight),scalar,thickness);
	cvLine(image,cvPoint( x+detWindWidth,y+detWindHeight),cvPoint( x, y+detWindHeight),scalar,thickness);
	cvLine(image,cvPoint( x, y+detWindHeight),cvPoint( x, y),scalar,thickness);

}



double LMLineMatcher::getCost(EIEdgeImage& tdbImage, double ltrans[2], double factor, int& count)
{

	LFLineSegment line;
	double minCost = 1e+10;
  
  double cost = 0;				
  for (int k=0 ; k<tdbImage.nLines_ ; k++)
  {
    line = tdbImage.lines_[k];
    line.Translate(ltrans);	

    //printf("Count: %f %f %i %i %i %i\n",
    //    ltrans[0], ltrans[1],
    //    (int)line.sx_, (int) line.sy_,
    //    (int)line.ex_,(int) line.ey_);


    int ccount = 0;
    double sum = queryDistanceImage_.idtImages_[tdbImage.directionIndices_[k]].Sum((int)line.sx_,(int) line.sy_,(int) line.ex_,(int) line.ey_, ccount);
    count += ccount;

    //printf("Done\n");
    cost+=sum*factor;
    if (cost > minCost*minCostRatio_)
    {
      cost = 1e+10;
      break;
    }
  }

  return cost;
}

double LMLineMatcher::getCost(int directionIdx, int sx, int sy, int ex, int ey, int count) const
{
  return queryDistanceImage_.idtImages_[directionIdx].Sum(sx,sy,ex,ey,count);
}

double LMLineMatcher::MatchBruteForce(EIEdgeImage& dbImage,
    int index, int* iindices,
    int* indices, int* xIndex, int* yIndex,
    double* dIndex, double* sIndex,
    double* distances, int& counter, double& minCost)
{
	
	int count =0;
	int i = 0;
	//int k;

	LFLineSegment dbLine, queryLine;
	

	double ltrans[2];
	
	double factor = 1.0;
	double cost; //, sum;
	double minx, miny, maxx, maxy;
	double scale;

	LFLineSegment line;
	
	//int currentcount;

	for(double s = minSearchScale_  ; s< maxSearchScale_ ; s++)	
	{		
		scale = pow(baseSearchScale_,s);
    printf("Scale %f\n", scale);
		EIEdgeImage tdbImage;
		tdbImage = dbImage;
		tdbImage.Scale(scale);
		factor = 1.0/dbImage.Length();
		tdbImage.Boundary(minx, miny, maxx, maxy);
		tdbImage.SetDirectionIndices();

    printf("W %i H %i\n", queryImage_.width_, queryImage_.height_);
		for (int x=-(int)minx ; x<queryImage_.width_-(int)minx ; x += searchStepSize_)
		{
			for (int y=-(int)miny; y<queryImage_.height_-(int)miny; y += searchStepSize_)
			{
		
				ltrans[0] = (double)x;
				ltrans[1] = (double)y;				
				cost = 0;				

				if (minx + ltrans[0] <=searchBoundarySize_ ||
            minx + ltrans[0] >=queryImage_.width_-searchBoundarySize_ || 
            maxx + ltrans[0] <=searchBoundarySize_ ||
            maxx + ltrans[0] >=queryImage_.width_-searchBoundarySize_ ||
            miny + ltrans[1] <=searchBoundarySize_ ||
            miny + ltrans[1] >=queryImage_.height_-searchBoundarySize_ || 
            maxy + ltrans[1] <=searchBoundarySize_ ||
            maxy + ltrans[1] >=queryImage_.height_-searchBoundarySize_ )
				{
					cost = 1e+10;
					continue;
				}
				else
				{
					count++;

          int ccount;
          cost = getCost(tdbImage, ltrans, factor, ccount);
					//for (k=0 ; k<tdbImage.nLines_ ; k++)
					//{
					//	line = tdbImage.lines_[k];
					//	line.Translate(ltrans);	

					//	sum = queryDistanceImage_.idtImages_[tdbImage.directionIndices_[k]].Sum((int)line.sx_,(int) line.sy_,(int) line.ex_,(int) line.ey_, currentcount);
		
					//	cost+=sum*factor;
					//	if (cost > minCost*minCostRatio_)
					//	{
					//		cost = 1e+10;
					//		break;
					//	}
					//}
				}


				if (cost<minCost*minCostRatio_)
				{
					xIndex[counter] = (int)ltrans[0];
					yIndex[counter] = (int)ltrans[1];
					dIndex[counter] = (i*M_PI)/nDirections_;
					sIndex[counter] = scale;
					distances[counter] = cost;
					iindices[counter] = counter;
					indices[counter++] = index;

					if (cost<minCost)
						minCost = cost;

				}


			}
		}
		
	}
	return minCost;
}


void LMLineMatcher::ISort(double* ra, int nVec, int* ira)
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
