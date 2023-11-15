/****************************************************************************
** Form interface generated from reading ui file 'Qt/SSCMainForm.ui'
**
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#ifndef SSCMAINFORM_H
#define SSCMAINFORM_H

#include <qvariant.h>
#include <qdialog.h>
#include "Component/ModelManager.H"
#include "Devices/ssc.H"

class QVBoxLayout;
class QHBoxLayout;
class QGridLayout;
class QSpacerItem;
class QLabel;
class QLineEdit;
class QRadioButton;
class QPushButton;
class QGroupBox;
class QSlider;
class QLCDNumber;

class SSCMainForm : public QDialog
{
    Q_OBJECT

public:
    SSCMainForm( QWidget* parent = 0, const char* name = 0, bool modal = FALSE, WFlags fl = 0 );
    ~SSCMainForm();

    QLabel* labelSerDev;
    QLineEdit* lineEditSerDev;
    QLabel* labelBaudrate;
    QLineEdit* lineEditBaudrate;
    QRadioButton* radioButtonDec;
    QRadioButton* radioButtonHex;
    QPushButton* pushButtonQuit;
    QGroupBox* groupBoxAllSSC;
    QLabel* textLabelSSC1;
    QSlider* sliderSSC1;
    QLCDNumber* lCDNumberSSC1;
    QLabel* textLabelSSC2;
    QSlider* sliderSSC2;
    QLCDNumber* lCDNumberSSC2;
    QLabel* textLabelSSC3;
    QSlider* sliderSSC3;
    QLCDNumber* lCDNumberSSC3;
    QLabel* textLabelSSC4;
    QSlider* sliderSSC4;
    QLCDNumber* lCDNumberSSC4;
    QLabel* textLabelSSC5;
    QSlider* sliderSSC5;
    QLCDNumber* lCDNumberSSC5;
    QLabel* textLabelSSC6;
    QSlider* sliderSSC6;
    QLCDNumber* lCDNumberSSC6;
    QLabel* textLabelSSC7;
    QSlider* sliderSSC7;
    QLCDNumber* lCDNumberSSC7;
    QLabel* textLabelSSC8;
    QSlider* sliderSSC8;
    QLCDNumber* lCDNumberSSC8;

    virtual void init( ModelManager * mgr, nub::soft_ref<SSC> thessc );

public slots:
    virtual void lineEditSerDev_textChanged( const QString & newdevname );
    virtual void lineEditBaudrate_textChanged( const QString & newbaud );
    virtual void sliderSSC1_valueChanged( int val );
    virtual void sliderSSC2_valueChanged( int val );
    virtual void sliderSSC3_valueChanged( int val );
    virtual void sliderSSC4_valueChanged( int val );
    virtual void sliderSSC5_valueChanged( int val );
    virtual void sliderSSC6_valueChanged( int val );
    virtual void sliderSSC7_valueChanged( int val );
    virtual void sliderSSC8_valueChanged( int val );
    virtual void radioButtonDec_clicked();
    virtual void radioButtonHex_clicked();
    virtual void lineEditSerDev_returnPressed();
    virtual void lineEditBaudrate_returnPressed();

protected:
    nub::soft_ref<SSC> ssc;
    ModelManager *manager;

    QHBoxLayout* layoutBottomBar;
    QSpacerItem* spacerSerDev;
    QSpacerItem* spacerBaud;
    QSpacerItem* spacerHex;
    QVBoxLayout* layoutAllSSC;
    QSpacerItem* spacer1_2;
    QSpacerItem* spacer2_3;
    QSpacerItem* spacer3_4;
    QSpacerItem* spacer4_5;
    QSpacerItem* spacer5_6;
    QSpacerItem* spacer6_7;
    QSpacerItem* spacer7_8;
    QHBoxLayout* layoutSSC1;
    QHBoxLayout* layoutSSC2;
    QHBoxLayout* layoutSSC3;
    QHBoxLayout* layoutSSC4;
    QHBoxLayout* layoutSSC5;
    QHBoxLayout* layoutSSC6;
    QHBoxLayout* layoutSSC7;
    QHBoxLayout* layoutSSC8;

protected slots:
    virtual void languageChange();

private:
    QString devname;
    QString baud;

};

#endif // SSCMAINFORM_H
