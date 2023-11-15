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

#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Util/Timer.H"
#include "Util/Types.H"
#include "Util/log.H"

void BeoSubMappingQtMainForm::init()
{
    prefix = "bottom";        // defalut
    setSave = false;
    startTimer(100);
}
void BeoSubMappingQtMainForm::change_bottom()
{
    prefix = "bottom";
}
void BeoSubMappingQtMainForm::change_front()
{
    prefix = "front";
}
void BeoSubMappingQtMainForm::change_top()
{
    prefix = "top";
}
void BeoSubMappingQtMainForm::camera_return()
{

//  char *a = camera->text().ascii();
//  prefix = a;

}

void BeoSubMappingQtMainForm::displayFunc()
{
    float fx, fy;
    if(List->currentItem()!= -1)
        poolImage->getRealPosition(List->currentItem(), fx, fy);
    else
        fx= fy = 0.0F;
    if(List->currentItem() == 0)
       displayCoord->setText(QString(sformat("x:%f y:%f", 0.0F, 0.0F)));
   else
       displayCoord->setText(QString(sformat("x:%f y:%f", fx, fy)));
}

void BeoSubMappingQtMainForm::timerEvent( QTimerEvent *e )
{
  displayFunc();
}

void BeoSubMappingQtMainForm::displayLog()
{
    // open the selected log file
    FILE *f;
    char stemp[20];
    std::string s;
    QString qs;

    poolImage->currentItem(List->currentItem());
    f = fopen(List->currentText().ascii(), "r");
    //can't open the file
    if (f == NULL)
        Log->setText(QString("can't find the file"));
    else
    {
      if (fscanf(f, "%s\t", stemp) != 1) LFATAL("fscanf() failed");
        // change to the number of lines in the LOg
        for(int i=0 ; i<8 ; i++)
        {
	  if (fscanf( f, "%s", stemp) != 1) LFATAL("fscanf() failed");
            if(i%2 == 0)
                qs += QString(sformat( "%s\t", stemp));
            else
                qs += QString(sformat( "%s\n", stemp));
        }

        Log->setText(qs);
    }
    fclose(f);

    // show the image
    QString qs2 = List->currentText().replace(".txt", ".png");
    img = Raster::ReadRGB( qs2 );
    QPixmap qpixm = convertToQPixmap( img );

    QWMatrix m;
    m.rotate( -90 );                          // rotate coordinate system
    QPixmap rpm =qpixm.xForm( m );
    test->setPixmap(rpm);
    displayImage->setPixmap(qpixm);
}

// read the file and parse the coordinate
void BeoSubMappingQtMainForm::loadList()
{
    int i, j;
    //int tempx, tempy;
    float tempx, tempy, glx, gly;
    char stemp[10][20];
    FILE *f;
    // start to load the total txt file in the dir
    refreshImages();
    // create the object in opengl
    poolImage->reset();
    createIcon();
    // setting the postition for each one

    for(i=0 ; i<numItem ; i++)
    {
        f = fopen(List->text(i).ascii(), "r");
        // add the coordinate for the icon on the panel and the real coordinate
        for(j=0 ; j<9 ; j++)
	  if (fscanf( f, "%s\t", stemp[j]) != 1) LFATAL("fscanf() failed");

        if (fscanf(f, "%f %f", &tempx, &tempy) != 2) LFATAL("fscanf() failed");
        if(tempx != -1000 && tempy!=-1000)
        {
            // FIXME it should be the real coord
            //poolImage->setCoord(i, tempx, tempy);
            poolImage->getGLPosition(tempx, tempy, glx, gly);
            poolImage->setCoordByGL(i, glx, gly);
        }
        fclose(f);
    }

}


// insert items into the list
void BeoSubMappingQtMainForm::refreshImages()
{
    DIR *dir;
    struct dirent *entry;
    int size = 50;
    char buffer[size], fileName[20];

    numItem = 0;
    List->clear();
    if (getcwd(buffer, size) == NULL) LFATAL("cannot getcwd");
    if((dir = opendir(buffer)) == NULL)
        printf("open dir error");
    else
    {

        while((entry = readdir(dir)) !=NULL)
        {
            if(entry->d_name[0] != '.')
            {
                strcpy(fileName, entry->d_name);
                QString qs(fileName);
        if(prefix =="bottom")
        {
                if(qs.contains("taskGdown.txt") > 0)
                {
                    numItem++;
                    List->insertItem(qs, 0);
                }
                if(qs.contains("taskAdown.txt") > 0)
                {
                    numItem++;
                    List->insertItem(qs,1);
                }
                if(qs.contains("taskBdown.txt") > 0)
                {
                    numItem++;
                    List->insertItem(qs,2);
                }
                if(qs.contains("taskCdown.txt") > 0)
                {
                    numItem++;
                    List->insertItem(qs, 3);
                }

        }
        else if(prefix == "front")
        {
                 if(qs.contains("taskGfront.txt") > 0)
                {
                    numItem++;
                    List->insertItem(qs, 0);
                }
                if(qs.contains("taskAfront.txt") > 0)
                {
                    numItem++;
                    List->insertItem(qs,1);
                }
                if(qs.contains("taskBfront.txt") > 0)
                {
                    numItem++;
                    List->insertItem(qs,2);
                }
                if(prefix == "top")
                {
                    numItem++;
                    List->insertItem(qs, 3);
                }
        }
        else if(prefix =="top")
        {
                 if(qs.contains("taskGup.txt") > 0)
                {
                    numItem++;
                    List->insertItem(qs, 0);
                }
                if(qs.contains("taskSup.txt") > 0)
                {
                    numItem++;
                    List->insertItem(qs,1);
                }
                if(qs.contains("taskCup.txt") > 0)
                {
                    numItem++;
                    List->insertItem(qs, 2);
                }

        }

            }
        }
        if((dir = opendir(buffer)) == NULL)
            printf("open dir error");
        else
        {
            while((entry = readdir(dir)) !=NULL)
            {
                if(entry->d_name[0] != '.')
                {
                    strcpy(fileName, entry->d_name);
                    QString qs(fileName);
                    if(qs.contains(".txt") > 0 &&qs.contains(prefix) > 0&& qs.contains("task") <= 0)
                    {
                        numItem++;
                        List->insertItem(qs);  // fix me
                    }
                }
            }
        }
    }
}

// recreate or create the item
void BeoSubMappingQtMainForm::createIcon()
{
    if(numItem != 0)
        poolImage->createIcon(numItem);
}

void BeoSubMappingQtMainForm::saveiconList()
{
   // iconImage *list = poolImage->getList();
    FILE *f;
    int i, j;
    //int tempx, tempy;
    float tempx, tempy, rx, ry;
    char stemp[10][20];

    // get every thing
    for(i=0 ; i<numItem ; i++)
    {
        f = fopen(List->text(i).ascii(), "r");
        // add the coordinate for the icon on the panel and the real coordinate
        for(j=0 ; j<9 ; j++)
	  if (fscanf( f, "%s\t", stemp[j]) != 1) LFATAL("fscanf() failed");
        //fscanf(f, "%d %d", &tempx, &tempy);
        if (fscanf(f, "%f %f", &tempx, &tempy) != 2) LFATAL("fscanf() failed");
        fclose(f);
        f = fopen(List->text(i).ascii(), "w");
            // delete the last 2 lines and then add the new one
        for(j = 0 ; j<9 ; j++)
            fprintf(f, "%s\t", stemp[j]);

        poolImage->getRealPosition(i, rx, ry);
        fprintf(f, "%f %f", rx, ry);
        //fprintf(f, "%d %d", list[i].x, list[i].y);
        fclose(f);
    }
}

void BeoSubMappingQtMainForm::LoadAngleScale()
{
    FILE *f;
    //char stemp[10];
    float angle;
    float scale;
    f = fopen("anglescale.as", "r");
    if (fscanf(f,"%f %f", &angle, &scale) != 2) LFATAL("fscanf() failed");
    poolImage->angle = angle;
    poolImage->scale = scale;

    fclose(f);
}

void BeoSubMappingQtMainForm::resetAllHeading()
{
    // read the file and reset the information about heading to t-1000
    DIR *dir;
    FILE *f;
    struct dirent *entry;
    int i, size = 50;
    char buffer[size], fileName[20];
    char stemp[12][20];
    if (getcwd(buffer, size) == NULL) LFATAL("cannot getcwd");
    if((dir = opendir(buffer)) == NULL)
      printf("open dir error");
    else
    {

        while((entry = readdir(dir)) !=NULL)
        {
            if(entry->d_name[0] != '.')
            {
                strcpy(fileName, entry->d_name);
                QString qs(fileName);
                if(qs.contains(".txt") > 0)
                {
            f = fopen(qs.ascii(), "r");
            for(i=0 ; i<12 ;i++)
            {
                if(i<9 || i==10)
                  if (fscanf( f, "%s\t", stemp[i]) != 1) LFATAL("fscanf() failed");
                if(i == 9)
		  if (fscanf(f, "%s ",stemp[i]) != 1) LFATAL("fscanf() failed");
                if(i==11)
		  if (fscanf(f, "%s", stemp[i]) != 1) LFATAL("fscanf() failed");
            }

            fclose(f);
            f = fopen(qs.ascii(), "w");
            for(i= 0 ; i<12 ; i++)
            {
                if(i!=1 && (i<9 ||i==10))
                fprintf(f, "%s\t", stemp[i]);
                else if(i==1)
                    fprintf(f, "%s\t", sformat("%f", -1000.0).c_str());
                else if(i==11)
                    fprintf(f, "%s",stemp[i]);
            }
            fclose(f);
        }
            }
        }
    }
}

void BeoSubMappingQtMainForm::changeTheHeading()
{
    FILE *f;
    int i;
    char stemp[12][20];
            f = fopen(List->currentText().ascii(), "r");
            for(i=0 ; i<12 ;i++)
            {
                if(i<9 || i==10)
                  if (fscanf( f, "%s\t", stemp[i]) != 1) LFATAL("fscanf() failed");
                if(i == 9)
		  if (fscanf(f, "%s ",stemp[i]) != 1) LFATAL("fscanf() failed");
                if(i==11)
		  if (fscanf(f, "%s", stemp[i]) != 1) LFATAL("fscanf() failed");
            }

            fclose(f);
            f = fopen(List->currentText().ascii(), "w");
            for(i= 0 ; i<12 ; i++)
            {
                if(i!=1 && (i<9 ||i==10))
                fprintf(f, "%s\t", stemp[i]);
                else if(i==1)
                    fprintf(f, "%s\t", changeHeading->text().ascii());
                else if(i==11)
                    fprintf(f, "%s",stemp[i]);
            }
            fclose(f);
            displayLog();
}
