/****************************************************************************
** Form interface generated from reading ui file 'Qt/BiasSettingsDialog.ui'
**
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#ifndef BIASSETTINGSDIALOG_H
#define BIASSETTINGSDIALOG_H

#include <qvariant.h>
#include <qpixmap.h>
#include <qdialog.h>
#include "Component/ModelManager.H"
#include "Neuro/StdBrain.H"
#include "Neuro/VisualCortex.H"
#include "Neuro/SaliencyMap.H"
#include "Image/ColorOps.H"
#include "Channels/ComplexChannel.H"
#include "Channels/SingleChannel.H"
#include "Image/ImageSet.H"
#include "Image/MathOps.H"
#include "Qt/BiasValImage.h"

class QVBoxLayout;
class QHBoxLayout;
class QGridLayout;
class QSpacerItem;
class ImageCanvas;
class QTabWidget;
class QWidget;
class QSpinBox;
class QLabel;
class QCheckBox;
class QPushButton;

class BiasSettingsDialog : public QDialog
{
    Q_OBJECT

public:
    BiasSettingsDialog( QWidget* parent = 0, const char* name = 0, bool modal = FALSE, WFlags fl = 0 );
    ~BiasSettingsDialog();

    QTabWidget* tabDisp;
    QWidget* tab;
    QSpinBox* spinBox43;
    QLabel* textLabel10;
    ImageCanvas* imageCanvas50;
    QCheckBox* chkBoxShowRaw;
    QCheckBox* chkBoxResizeToSLevel;
    QPushButton* updateValButton;
    QPushButton* buttonOk;
    QPushButton* buttonCancel;

    virtual void init( ModelManager & manager );
    virtual void showFeatures();
    virtual void setupTab( ComplexChannel & cc, SingleChannel & sc);

public slots:
    virtual void biasFeature( int value );
    virtual void update();

protected:
    std::vector<BiasValImage*> itsBiasValImage;

    QVBoxLayout* BiasSettingsDialogLayout;
    QHBoxLayout* layout4;
    QSpacerItem* spacer2;

protected slots:
    virtual void languageChange();

private:
    ModelManager *itsMgr;

    QPixmap image0;

};

#endif // BIASSETTINGSDIALOG_H
