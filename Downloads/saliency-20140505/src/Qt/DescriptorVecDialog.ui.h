/****************************************************************************
** ui.h extension file, included from the uic-generated form implementation.
**
** If you want to add, delete, or rename functions or slots, use
** Qt Designer to update this file, preserving your code.
**
** You should not define a constructor or destructor in this file.
** Instead, write your code in functions called init() and destroy().
** These will automatically be called by the form's constructor and
** destructor.
*****************************************************************************/


void DescriptorVecDialog::init( DescriptorVec &dv )
{
    itsDV = &dv;
    Image<PixRGB<byte> > img = itsDV->getFoveaImage();
    Image<PixRGB<byte> > histImg = itsDV->getHistogramImage();
    if(img.initialized()){
       imgDisp->setImage(img);
       histDisp->setImage(histImg);
    }

}


void DescriptorVecDialog::update()
{
    Image<PixRGB<byte> > img = itsDV->getFoveaImage();
    Image<PixRGB<byte> > histImg = itsDV->getHistogramImage();
    imgDisp->setImage(img);
    histDisp->setImage(histImg);

    std::vector<double> FV = itsDV->getFV();
    LINFO("Get desription\n");
    FVtable->setNumRows(FV.size());
    FVtable->setNumCols(1);

    for(uint i=0; i<FV.size(); i++)
    {
      QString stat = QString("%L1").arg(FV[i]);
      FVtable->setText(i-1,1, stat);
    }
    FVtable->updateContents();
}
