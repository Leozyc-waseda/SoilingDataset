#include <Qt/qapplication.h>
#include <QtGui/qgraphicsscene.h>
#include <QtGui/qgraphicsview.h>

#include "Component/ModelManager.H"
#include "Component/ModelComponent.H"
#include "Media/FrameSeries.H"
#include "Image/Image.H"
#include "Image/PixelsTypes.H"
#include "QtUtil/ImageConvert4.H"
#include "Qt4/ImageGraphicsItem.qt.H"
#include "NeovisionII/NeoAnnotate/MainWindow.qt.H"


int main(int argc, char* argv[])
{
  QApplication app(argc, argv);


  MainWindow mainWindow;

  mainWindow.setGeometry(100,100,900,800);
  mainWindow.show();

  return app.exec();
}
