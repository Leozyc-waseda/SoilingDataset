/********************************************************************************
** Form generated from reading ui file 'BeoChipMainForm.ui'
**
**
** WARNING! All changes made in this file will be lost when recompiling ui file!
********************************************************************************/

#ifndef BEOCHIPMAINFORM_H
#define BEOCHIPMAINFORM_H

#include <Qt3Support/Q3Frame>
#include <Qt3Support/Q3GroupBox>
#include <Qt3Support/Q3MimeSourceFactory>
#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QCheckBox>
#include <QtGui/QDialog>
#include <QtGui/QHBoxLayout>
#include <QtGui/QLCDNumber>
#include <QtGui/QLabel>
#include <QtGui/QPushButton>
#include <QtGui/QRadioButton>
#include <QtGui/QSlider>
#include <QtGui/QSpacerItem>
#include <QtGui/QVBoxLayout>

class Ui_BeoChipMainForm
{
public:
    QVBoxLayout *vboxLayout;
    Q3GroupBox *groupBoxAllSSC;
    QVBoxLayout *vboxLayout1;
    QSpacerItem *spacerItem;
    Q3Frame *frame5_2;
    QHBoxLayout *hboxLayout;
    QLabel *textLabelSSC0;
    QSlider *sliderSSC0;
    QLCDNumber *lCDNumberSSC0;
    QSpacerItem *spacerItem1;
    Q3Frame *frame6;
    QHBoxLayout *hboxLayout1;
    QLabel *textLabelSSC1;
    QSlider *sliderSSC1;
    QLCDNumber *lCDNumberSSC1;
    QSpacerItem *spacerItem2;
    Q3Frame *frame7;
    QHBoxLayout *hboxLayout2;
    QLabel *textLabelSSC2;
    QSlider *sliderSSC2;
    QLCDNumber *lCDNumberSSC2;
    QSpacerItem *spacerItem3;
    Q3Frame *frame8;
    QHBoxLayout *hboxLayout3;
    QLabel *textLabelSSC3;
    QSlider *sliderSSC3;
    QLCDNumber *lCDNumberSSC3;
    QSpacerItem *spacerItem4;
    Q3Frame *frame9;
    QHBoxLayout *hboxLayout4;
    QLabel *textLabelSSC4;
    QSlider *sliderSSC4;
    QLCDNumber *lCDNumberSSC4;
    QSpacerItem *spacerItem5;
    Q3Frame *frame10;
    QHBoxLayout *hboxLayout5;
    QLabel *textLabelSSC5;
    QSlider *sliderSSC5;
    QLCDNumber *lCDNumberSSC5;
    QSpacerItem *spacerItem6;
    Q3Frame *frame11;
    QHBoxLayout *hboxLayout6;
    QLabel *textLabelSSC6;
    QSlider *sliderSSC6;
    QLCDNumber *lCDNumberSSC6;
    QSpacerItem *spacerItem7;
    Q3Frame *frame12;
    QHBoxLayout *hboxLayout7;
    QLabel *textLabelSSC7;
    QSlider *sliderSSC7;
    QLCDNumber *lCDNumberSSC7;
    QSpacerItem *spacerItem8;
    Q3Frame *frame4;
    QHBoxLayout *hboxLayout8;
    QLabel *textLabelKBD;
    QLCDNumber *digin4;
    QLCDNumber *digin3;
    QLCDNumber *digin2;
    QLCDNumber *digin1;
    QLCDNumber *digin0;
    QSpacerItem *spacerItem9;
    QLabel *textLabelDOUT;
    QCheckBox *checkBoxOut3;
    QCheckBox *checkBoxOut2;
    QCheckBox *checkBoxOut1;
    QCheckBox *checkBoxOut0;
    QSpacerItem *spacerItem10;
    QLabel *textLabelAin;
    QLCDNumber *lCDNumberA0;
    QLCDNumber *lCDNumberA1;
    QSpacerItem *spacerItem11;
    QLabel *textLabelPWM;
    QLCDNumber *lCDNumberPWM0;
    QLCDNumber *lCDNumberPWM1;
    Q3Frame *frame5;
    QHBoxLayout *hboxLayout9;
    QLabel *labelSerDev;
    QSpacerItem *spacerItem12;
    QRadioButton *radioButtonDec;
    QRadioButton *radioButtonHex;
    QSpacerItem *spacerItem13;
    QCheckBox *checkDebounce;
    QSpacerItem *spacerItem14;
    QPushButton *pushButtonReset;
    QSpacerItem *spacerItem15;
    QPushButton *pushButtonQuit;

    void setupUi(QDialog *BeoChipMainForm)
    {
    if (BeoChipMainForm->objectName().isEmpty())
        BeoChipMainForm->setObjectName(QString::fromUtf8("BeoChipMainForm"));
    BeoChipMainForm->resize(750, 602);
    vboxLayout = new QVBoxLayout(BeoChipMainForm);
    vboxLayout->setSpacing(3);
    vboxLayout->setMargin(11);
    vboxLayout->setObjectName(QString::fromUtf8("vboxLayout"));
    groupBoxAllSSC = new Q3GroupBox(BeoChipMainForm);
    groupBoxAllSSC->setObjectName(QString::fromUtf8("groupBoxAllSSC"));
    QSizePolicy sizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    sizePolicy.setHorizontalStretch(0);
    sizePolicy.setVerticalStretch(0);
    sizePolicy.setHeightForWidth(groupBoxAllSSC->sizePolicy().hasHeightForWidth());
    groupBoxAllSSC->setSizePolicy(sizePolicy);
    groupBoxAllSSC->setColumnLayout(0, Qt::Vertical);
    groupBoxAllSSC->layout()->setSpacing(3);
    groupBoxAllSSC->layout()->setMargin(11);
    vboxLayout1 = new QVBoxLayout();
    QBoxLayout *boxlayout = qobject_cast<QBoxLayout *>(groupBoxAllSSC->layout());
    if (boxlayout)
        boxlayout->addLayout(vboxLayout1);
    vboxLayout1->setAlignment(Qt::AlignTop);
    vboxLayout1->setObjectName(QString::fromUtf8("vboxLayout1"));
    spacerItem = new QSpacerItem(20, 16, QSizePolicy::Minimum, QSizePolicy::Expanding);

    vboxLayout1->addItem(spacerItem);

    frame5_2 = new Q3Frame(groupBoxAllSSC);
    frame5_2->setObjectName(QString::fromUtf8("frame5_2"));
    frame5_2->setFrameShape(QFrame::NoFrame);
    frame5_2->setFrameShadow(QFrame::Plain);
    hboxLayout = new QHBoxLayout(frame5_2);
    hboxLayout->setSpacing(3);
    hboxLayout->setMargin(11);
    hboxLayout->setObjectName(QString::fromUtf8("hboxLayout"));
    textLabelSSC0 = new QLabel(frame5_2);
    textLabelSSC0->setObjectName(QString::fromUtf8("textLabelSSC0"));
    QSizePolicy sizePolicy1(QSizePolicy::Fixed, QSizePolicy::Preferred);
    sizePolicy1.setHorizontalStretch(0);
    sizePolicy1.setVerticalStretch(0);
    sizePolicy1.setHeightForWidth(textLabelSSC0->sizePolicy().hasHeightForWidth());
    textLabelSSC0->setSizePolicy(sizePolicy1);
    QFont font;
    font.setBold(true);
    font.setWeight(75);
    textLabelSSC0->setFont(font);
    textLabelSSC0->setWordWrap(false);

    hboxLayout->addWidget(textLabelSSC0);

    sliderSSC0 = new QSlider(frame5_2);
    sliderSSC0->setObjectName(QString::fromUtf8("sliderSSC0"));
    sliderSSC0->setMaximum(255);
    sliderSSC0->setOrientation(Qt::Horizontal);
    sliderSSC0->setTickInterval(5);

    hboxLayout->addWidget(sliderSSC0);

    lCDNumberSSC0 = new QLCDNumber(frame5_2);
    lCDNumberSSC0->setObjectName(QString::fromUtf8("lCDNumberSSC0"));
    lCDNumberSSC0->setLineWidth(0);
    lCDNumberSSC0->setNumDigits(3);
    lCDNumberSSC0->setSegmentStyle(QLCDNumber::Flat);

    hboxLayout->addWidget(lCDNumberSSC0);


    vboxLayout1->addWidget(frame5_2);

    spacerItem1 = new QSpacerItem(20, 20, QSizePolicy::Minimum, QSizePolicy::Expanding);

    vboxLayout1->addItem(spacerItem1);

    frame6 = new Q3Frame(groupBoxAllSSC);
    frame6->setObjectName(QString::fromUtf8("frame6"));
    frame6->setFrameShape(QFrame::NoFrame);
    frame6->setFrameShadow(QFrame::Plain);
    hboxLayout1 = new QHBoxLayout(frame6);
    hboxLayout1->setSpacing(3);
    hboxLayout1->setMargin(11);
    hboxLayout1->setObjectName(QString::fromUtf8("hboxLayout1"));
    textLabelSSC1 = new QLabel(frame6);
    textLabelSSC1->setObjectName(QString::fromUtf8("textLabelSSC1"));
    sizePolicy1.setHeightForWidth(textLabelSSC1->sizePolicy().hasHeightForWidth());
    textLabelSSC1->setSizePolicy(sizePolicy1);
    textLabelSSC1->setFont(font);
    textLabelSSC1->setWordWrap(false);

    hboxLayout1->addWidget(textLabelSSC1);

    sliderSSC1 = new QSlider(frame6);
    sliderSSC1->setObjectName(QString::fromUtf8("sliderSSC1"));
    sliderSSC1->setMaximum(255);
    sliderSSC1->setOrientation(Qt::Horizontal);
    sliderSSC1->setTickInterval(5);

    hboxLayout1->addWidget(sliderSSC1);

    lCDNumberSSC1 = new QLCDNumber(frame6);
    lCDNumberSSC1->setObjectName(QString::fromUtf8("lCDNumberSSC1"));
    lCDNumberSSC1->setLineWidth(0);
    lCDNumberSSC1->setNumDigits(3);
    lCDNumberSSC1->setSegmentStyle(QLCDNumber::Flat);

    hboxLayout1->addWidget(lCDNumberSSC1);


    vboxLayout1->addWidget(frame6);

    spacerItem2 = new QSpacerItem(20, 20, QSizePolicy::Minimum, QSizePolicy::Expanding);

    vboxLayout1->addItem(spacerItem2);

    frame7 = new Q3Frame(groupBoxAllSSC);
    frame7->setObjectName(QString::fromUtf8("frame7"));
    frame7->setFrameShape(QFrame::NoFrame);
    frame7->setFrameShadow(QFrame::Plain);
    hboxLayout2 = new QHBoxLayout(frame7);
    hboxLayout2->setSpacing(3);
    hboxLayout2->setMargin(11);
    hboxLayout2->setObjectName(QString::fromUtf8("hboxLayout2"));
    textLabelSSC2 = new QLabel(frame7);
    textLabelSSC2->setObjectName(QString::fromUtf8("textLabelSSC2"));
    sizePolicy1.setHeightForWidth(textLabelSSC2->sizePolicy().hasHeightForWidth());
    textLabelSSC2->setSizePolicy(sizePolicy1);
    textLabelSSC2->setFont(font);
    textLabelSSC2->setWordWrap(false);

    hboxLayout2->addWidget(textLabelSSC2);

    sliderSSC2 = new QSlider(frame7);
    sliderSSC2->setObjectName(QString::fromUtf8("sliderSSC2"));
    sliderSSC2->setMaximum(255);
    sliderSSC2->setOrientation(Qt::Horizontal);
    sliderSSC2->setTickInterval(5);

    hboxLayout2->addWidget(sliderSSC2);

    lCDNumberSSC2 = new QLCDNumber(frame7);
    lCDNumberSSC2->setObjectName(QString::fromUtf8("lCDNumberSSC2"));
    QSizePolicy sizePolicy2(QSizePolicy::Fixed, QSizePolicy::Minimum);
    sizePolicy2.setHorizontalStretch(0);
    sizePolicy2.setVerticalStretch(0);
    sizePolicy2.setHeightForWidth(lCDNumberSSC2->sizePolicy().hasHeightForWidth());
    lCDNumberSSC2->setSizePolicy(sizePolicy2);
    lCDNumberSSC2->setLineWidth(0);
    lCDNumberSSC2->setNumDigits(3);
    lCDNumberSSC2->setSegmentStyle(QLCDNumber::Flat);

    hboxLayout2->addWidget(lCDNumberSSC2);


    vboxLayout1->addWidget(frame7);

    spacerItem3 = new QSpacerItem(20, 20, QSizePolicy::Minimum, QSizePolicy::Expanding);

    vboxLayout1->addItem(spacerItem3);

    frame8 = new Q3Frame(groupBoxAllSSC);
    frame8->setObjectName(QString::fromUtf8("frame8"));
    frame8->setFrameShape(QFrame::NoFrame);
    frame8->setFrameShadow(QFrame::Plain);
    hboxLayout3 = new QHBoxLayout(frame8);
    hboxLayout3->setSpacing(3);
    hboxLayout3->setMargin(11);
    hboxLayout3->setObjectName(QString::fromUtf8("hboxLayout3"));
    textLabelSSC3 = new QLabel(frame8);
    textLabelSSC3->setObjectName(QString::fromUtf8("textLabelSSC3"));
    sizePolicy1.setHeightForWidth(textLabelSSC3->sizePolicy().hasHeightForWidth());
    textLabelSSC3->setSizePolicy(sizePolicy1);
    textLabelSSC3->setFont(font);
    textLabelSSC3->setWordWrap(false);

    hboxLayout3->addWidget(textLabelSSC3);

    sliderSSC3 = new QSlider(frame8);
    sliderSSC3->setObjectName(QString::fromUtf8("sliderSSC3"));
    sliderSSC3->setMaximum(255);
    sliderSSC3->setOrientation(Qt::Horizontal);
    sliderSSC3->setTickInterval(5);

    hboxLayout3->addWidget(sliderSSC3);

    lCDNumberSSC3 = new QLCDNumber(frame8);
    lCDNumberSSC3->setObjectName(QString::fromUtf8("lCDNumberSSC3"));
    sizePolicy2.setHeightForWidth(lCDNumberSSC3->sizePolicy().hasHeightForWidth());
    lCDNumberSSC3->setSizePolicy(sizePolicy2);
    lCDNumberSSC3->setLineWidth(0);
    lCDNumberSSC3->setNumDigits(3);
    lCDNumberSSC3->setSegmentStyle(QLCDNumber::Flat);

    hboxLayout3->addWidget(lCDNumberSSC3);


    vboxLayout1->addWidget(frame8);

    spacerItem4 = new QSpacerItem(20, 20, QSizePolicy::Minimum, QSizePolicy::Expanding);

    vboxLayout1->addItem(spacerItem4);

    frame9 = new Q3Frame(groupBoxAllSSC);
    frame9->setObjectName(QString::fromUtf8("frame9"));
    frame9->setFrameShape(QFrame::NoFrame);
    frame9->setFrameShadow(QFrame::Plain);
    hboxLayout4 = new QHBoxLayout(frame9);
    hboxLayout4->setSpacing(3);
    hboxLayout4->setMargin(11);
    hboxLayout4->setObjectName(QString::fromUtf8("hboxLayout4"));
    textLabelSSC4 = new QLabel(frame9);
    textLabelSSC4->setObjectName(QString::fromUtf8("textLabelSSC4"));
    sizePolicy1.setHeightForWidth(textLabelSSC4->sizePolicy().hasHeightForWidth());
    textLabelSSC4->setSizePolicy(sizePolicy1);
    textLabelSSC4->setFont(font);
    textLabelSSC4->setWordWrap(false);

    hboxLayout4->addWidget(textLabelSSC4);

    sliderSSC4 = new QSlider(frame9);
    sliderSSC4->setObjectName(QString::fromUtf8("sliderSSC4"));
    sliderSSC4->setMaximum(255);
    sliderSSC4->setOrientation(Qt::Horizontal);
    sliderSSC4->setTickInterval(5);

    hboxLayout4->addWidget(sliderSSC4);

    lCDNumberSSC4 = new QLCDNumber(frame9);
    lCDNumberSSC4->setObjectName(QString::fromUtf8("lCDNumberSSC4"));
    sizePolicy2.setHeightForWidth(lCDNumberSSC4->sizePolicy().hasHeightForWidth());
    lCDNumberSSC4->setSizePolicy(sizePolicy2);
    lCDNumberSSC4->setLineWidth(0);
    lCDNumberSSC4->setNumDigits(3);
    lCDNumberSSC4->setSegmentStyle(QLCDNumber::Flat);

    hboxLayout4->addWidget(lCDNumberSSC4);


    vboxLayout1->addWidget(frame9);

    spacerItem5 = new QSpacerItem(20, 16, QSizePolicy::Minimum, QSizePolicy::Expanding);

    vboxLayout1->addItem(spacerItem5);

    frame10 = new Q3Frame(groupBoxAllSSC);
    frame10->setObjectName(QString::fromUtf8("frame10"));
    frame10->setFrameShape(QFrame::NoFrame);
    frame10->setFrameShadow(QFrame::Plain);
    hboxLayout5 = new QHBoxLayout(frame10);
    hboxLayout5->setSpacing(3);
    hboxLayout5->setMargin(11);
    hboxLayout5->setObjectName(QString::fromUtf8("hboxLayout5"));
    textLabelSSC5 = new QLabel(frame10);
    textLabelSSC5->setObjectName(QString::fromUtf8("textLabelSSC5"));
    sizePolicy1.setHeightForWidth(textLabelSSC5->sizePolicy().hasHeightForWidth());
    textLabelSSC5->setSizePolicy(sizePolicy1);
    textLabelSSC5->setFont(font);
    textLabelSSC5->setWordWrap(false);

    hboxLayout5->addWidget(textLabelSSC5);

    sliderSSC5 = new QSlider(frame10);
    sliderSSC5->setObjectName(QString::fromUtf8("sliderSSC5"));
    sliderSSC5->setMaximum(255);
    sliderSSC5->setOrientation(Qt::Horizontal);
    sliderSSC5->setTickInterval(5);

    hboxLayout5->addWidget(sliderSSC5);

    lCDNumberSSC5 = new QLCDNumber(frame10);
    lCDNumberSSC5->setObjectName(QString::fromUtf8("lCDNumberSSC5"));
    sizePolicy2.setHeightForWidth(lCDNumberSSC5->sizePolicy().hasHeightForWidth());
    lCDNumberSSC5->setSizePolicy(sizePolicy2);
    lCDNumberSSC5->setLineWidth(0);
    lCDNumberSSC5->setNumDigits(3);
    lCDNumberSSC5->setSegmentStyle(QLCDNumber::Flat);

    hboxLayout5->addWidget(lCDNumberSSC5);


    vboxLayout1->addWidget(frame10);

    spacerItem6 = new QSpacerItem(20, 20, QSizePolicy::Minimum, QSizePolicy::Expanding);

    vboxLayout1->addItem(spacerItem6);

    frame11 = new Q3Frame(groupBoxAllSSC);
    frame11->setObjectName(QString::fromUtf8("frame11"));
    frame11->setFrameShape(QFrame::NoFrame);
    frame11->setFrameShadow(QFrame::Plain);
    hboxLayout6 = new QHBoxLayout(frame11);
    hboxLayout6->setSpacing(3);
    hboxLayout6->setMargin(11);
    hboxLayout6->setObjectName(QString::fromUtf8("hboxLayout6"));
    textLabelSSC6 = new QLabel(frame11);
    textLabelSSC6->setObjectName(QString::fromUtf8("textLabelSSC6"));
    sizePolicy1.setHeightForWidth(textLabelSSC6->sizePolicy().hasHeightForWidth());
    textLabelSSC6->setSizePolicy(sizePolicy1);
    textLabelSSC6->setFont(font);
    textLabelSSC6->setWordWrap(false);

    hboxLayout6->addWidget(textLabelSSC6);

    sliderSSC6 = new QSlider(frame11);
    sliderSSC6->setObjectName(QString::fromUtf8("sliderSSC6"));
    sliderSSC6->setMaximum(255);
    sliderSSC6->setOrientation(Qt::Horizontal);
    sliderSSC6->setTickInterval(5);

    hboxLayout6->addWidget(sliderSSC6);

    lCDNumberSSC6 = new QLCDNumber(frame11);
    lCDNumberSSC6->setObjectName(QString::fromUtf8("lCDNumberSSC6"));
    sizePolicy2.setHeightForWidth(lCDNumberSSC6->sizePolicy().hasHeightForWidth());
    lCDNumberSSC6->setSizePolicy(sizePolicy2);
    lCDNumberSSC6->setLineWidth(0);
    lCDNumberSSC6->setNumDigits(3);
    lCDNumberSSC6->setSegmentStyle(QLCDNumber::Flat);

    hboxLayout6->addWidget(lCDNumberSSC6);


    vboxLayout1->addWidget(frame11);

    spacerItem7 = new QSpacerItem(20, 20, QSizePolicy::Minimum, QSizePolicy::Expanding);

    vboxLayout1->addItem(spacerItem7);

    frame12 = new Q3Frame(groupBoxAllSSC);
    frame12->setObjectName(QString::fromUtf8("frame12"));
    frame12->setFrameShape(QFrame::NoFrame);
    frame12->setFrameShadow(QFrame::Plain);
    hboxLayout7 = new QHBoxLayout(frame12);
    hboxLayout7->setSpacing(3);
    hboxLayout7->setMargin(11);
    hboxLayout7->setObjectName(QString::fromUtf8("hboxLayout7"));
    textLabelSSC7 = new QLabel(frame12);
    textLabelSSC7->setObjectName(QString::fromUtf8("textLabelSSC7"));
    sizePolicy1.setHeightForWidth(textLabelSSC7->sizePolicy().hasHeightForWidth());
    textLabelSSC7->setSizePolicy(sizePolicy1);
    textLabelSSC7->setFont(font);
    textLabelSSC7->setWordWrap(false);

    hboxLayout7->addWidget(textLabelSSC7);

    sliderSSC7 = new QSlider(frame12);
    sliderSSC7->setObjectName(QString::fromUtf8("sliderSSC7"));
    sliderSSC7->setMaximum(255);
    sliderSSC7->setOrientation(Qt::Horizontal);
    sliderSSC7->setTickInterval(5);

    hboxLayout7->addWidget(sliderSSC7);

    lCDNumberSSC7 = new QLCDNumber(frame12);
    lCDNumberSSC7->setObjectName(QString::fromUtf8("lCDNumberSSC7"));
    sizePolicy2.setHeightForWidth(lCDNumberSSC7->sizePolicy().hasHeightForWidth());
    lCDNumberSSC7->setSizePolicy(sizePolicy2);
    lCDNumberSSC7->setLineWidth(0);
    lCDNumberSSC7->setNumDigits(3);
    lCDNumberSSC7->setSegmentStyle(QLCDNumber::Flat);

    hboxLayout7->addWidget(lCDNumberSSC7);


    vboxLayout1->addWidget(frame12);

    spacerItem8 = new QSpacerItem(20, 20, QSizePolicy::Minimum, QSizePolicy::Expanding);

    vboxLayout1->addItem(spacerItem8);


    vboxLayout->addWidget(groupBoxAllSSC);

    frame4 = new Q3Frame(BeoChipMainForm);
    frame4->setObjectName(QString::fromUtf8("frame4"));
    QSizePolicy sizePolicy3(QSizePolicy::Preferred, QSizePolicy::Minimum);
    sizePolicy3.setHorizontalStretch(0);
    sizePolicy3.setVerticalStretch(0);
    sizePolicy3.setHeightForWidth(frame4->sizePolicy().hasHeightForWidth());
    frame4->setSizePolicy(sizePolicy3);
    frame4->setMinimumSize(QSize(0, 40));
    frame4->setFrameShape(QFrame::NoFrame);
    frame4->setFrameShadow(QFrame::Plain);
    hboxLayout8 = new QHBoxLayout(frame4);
    hboxLayout8->setSpacing(3);
    hboxLayout8->setMargin(11);
    hboxLayout8->setObjectName(QString::fromUtf8("hboxLayout8"));
    textLabelKBD = new QLabel(frame4);
    textLabelKBD->setObjectName(QString::fromUtf8("textLabelKBD"));
    textLabelKBD->setWordWrap(false);

    hboxLayout8->addWidget(textLabelKBD);

    digin4 = new QLCDNumber(frame4);
    digin4->setObjectName(QString::fromUtf8("digin4"));
    digin4->setCursor(QCursor(Qt::CrossCursor));
    digin4->setLineWidth(0);
    digin4->setNumDigits(1);
    digin4->setSegmentStyle(QLCDNumber::Flat);

    hboxLayout8->addWidget(digin4);

    digin3 = new QLCDNumber(frame4);
    digin3->setObjectName(QString::fromUtf8("digin3"));
    digin3->setCursor(QCursor(Qt::CrossCursor));
    digin3->setLineWidth(0);
    digin3->setNumDigits(1);
    digin3->setSegmentStyle(QLCDNumber::Flat);

    hboxLayout8->addWidget(digin3);

    digin2 = new QLCDNumber(frame4);
    digin2->setObjectName(QString::fromUtf8("digin2"));
    digin2->setCursor(QCursor(Qt::CrossCursor));
    digin2->setLineWidth(0);
    digin2->setNumDigits(1);
    digin2->setSegmentStyle(QLCDNumber::Flat);

    hboxLayout8->addWidget(digin2);

    digin1 = new QLCDNumber(frame4);
    digin1->setObjectName(QString::fromUtf8("digin1"));
    digin1->setCursor(QCursor(Qt::CrossCursor));
    digin1->setLineWidth(0);
    digin1->setNumDigits(1);
    digin1->setSegmentStyle(QLCDNumber::Flat);

    hboxLayout8->addWidget(digin1);

    digin0 = new QLCDNumber(frame4);
    digin0->setObjectName(QString::fromUtf8("digin0"));
    digin0->setCursor(QCursor(Qt::CrossCursor));
    digin0->setLineWidth(0);
    digin0->setNumDigits(1);
    digin0->setSegmentStyle(QLCDNumber::Flat);

    hboxLayout8->addWidget(digin0);

    spacerItem9 = new QSpacerItem(16, 18, QSizePolicy::Expanding, QSizePolicy::Minimum);

    hboxLayout8->addItem(spacerItem9);

    textLabelDOUT = new QLabel(frame4);
    textLabelDOUT->setObjectName(QString::fromUtf8("textLabelDOUT"));
    textLabelDOUT->setWordWrap(false);

    hboxLayout8->addWidget(textLabelDOUT);

    checkBoxOut3 = new QCheckBox(frame4);
    checkBoxOut3->setObjectName(QString::fromUtf8("checkBoxOut3"));

    hboxLayout8->addWidget(checkBoxOut3);

    checkBoxOut2 = new QCheckBox(frame4);
    checkBoxOut2->setObjectName(QString::fromUtf8("checkBoxOut2"));

    hboxLayout8->addWidget(checkBoxOut2);

    checkBoxOut1 = new QCheckBox(frame4);
    checkBoxOut1->setObjectName(QString::fromUtf8("checkBoxOut1"));

    hboxLayout8->addWidget(checkBoxOut1);

    checkBoxOut0 = new QCheckBox(frame4);
    checkBoxOut0->setObjectName(QString::fromUtf8("checkBoxOut0"));

    hboxLayout8->addWidget(checkBoxOut0);

    spacerItem10 = new QSpacerItem(53, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

    hboxLayout8->addItem(spacerItem10);

    textLabelAin = new QLabel(frame4);
    textLabelAin->setObjectName(QString::fromUtf8("textLabelAin"));
    textLabelAin->setWordWrap(false);

    hboxLayout8->addWidget(textLabelAin);

    lCDNumberA0 = new QLCDNumber(frame4);
    lCDNumberA0->setObjectName(QString::fromUtf8("lCDNumberA0"));
    lCDNumberA0->setLineWidth(0);
    lCDNumberA0->setNumDigits(3);
    lCDNumberA0->setSegmentStyle(QLCDNumber::Flat);

    hboxLayout8->addWidget(lCDNumberA0);

    lCDNumberA1 = new QLCDNumber(frame4);
    lCDNumberA1->setObjectName(QString::fromUtf8("lCDNumberA1"));
    lCDNumberA1->setLineWidth(0);
    lCDNumberA1->setNumDigits(3);
    lCDNumberA1->setSegmentStyle(QLCDNumber::Flat);

    hboxLayout8->addWidget(lCDNumberA1);

    spacerItem11 = new QSpacerItem(50, 18, QSizePolicy::Expanding, QSizePolicy::Minimum);

    hboxLayout8->addItem(spacerItem11);

    textLabelPWM = new QLabel(frame4);
    textLabelPWM->setObjectName(QString::fromUtf8("textLabelPWM"));
    textLabelPWM->setWordWrap(false);

    hboxLayout8->addWidget(textLabelPWM);

    lCDNumberPWM0 = new QLCDNumber(frame4);
    lCDNumberPWM0->setObjectName(QString::fromUtf8("lCDNumberPWM0"));
    lCDNumberPWM0->setLineWidth(0);
    lCDNumberPWM0->setNumDigits(4);
    lCDNumberPWM0->setSegmentStyle(QLCDNumber::Flat);

    hboxLayout8->addWidget(lCDNumberPWM0);

    lCDNumberPWM1 = new QLCDNumber(frame4);
    lCDNumberPWM1->setObjectName(QString::fromUtf8("lCDNumberPWM1"));
    lCDNumberPWM1->setLineWidth(0);
    lCDNumberPWM1->setNumDigits(4);
    lCDNumberPWM1->setSegmentStyle(QLCDNumber::Flat);

    hboxLayout8->addWidget(lCDNumberPWM1);


    vboxLayout->addWidget(frame4);

    frame5 = new Q3Frame(BeoChipMainForm);
    frame5->setObjectName(QString::fromUtf8("frame5"));
    sizePolicy3.setHeightForWidth(frame5->sizePolicy().hasHeightForWidth());
    frame5->setSizePolicy(sizePolicy3);
    frame5->setMinimumSize(QSize(0, 40));
    frame5->setFrameShape(QFrame::NoFrame);
    frame5->setFrameShadow(QFrame::Plain);
    hboxLayout9 = new QHBoxLayout(frame5);
    hboxLayout9->setSpacing(3);
    hboxLayout9->setMargin(11);
    hboxLayout9->setObjectName(QString::fromUtf8("hboxLayout9"));
    labelSerDev = new QLabel(frame5);
    labelSerDev->setObjectName(QString::fromUtf8("labelSerDev"));
    labelSerDev->setWordWrap(false);

    hboxLayout9->addWidget(labelSerDev);

    spacerItem12 = new QSpacerItem(16, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

    hboxLayout9->addItem(spacerItem12);

    radioButtonDec = new QRadioButton(frame5);
    radioButtonDec->setObjectName(QString::fromUtf8("radioButtonDec"));
    radioButtonDec->setChecked(true);

    hboxLayout9->addWidget(radioButtonDec);

    radioButtonHex = new QRadioButton(frame5);
    radioButtonHex->setObjectName(QString::fromUtf8("radioButtonHex"));

    hboxLayout9->addWidget(radioButtonHex);

    spacerItem13 = new QSpacerItem(16, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

    hboxLayout9->addItem(spacerItem13);

    checkDebounce = new QCheckBox(frame5);
    checkDebounce->setObjectName(QString::fromUtf8("checkDebounce"));
    checkDebounce->setChecked(true);

    hboxLayout9->addWidget(checkDebounce);

    spacerItem14 = new QSpacerItem(16, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

    hboxLayout9->addItem(spacerItem14);

    pushButtonReset = new QPushButton(frame5);
    pushButtonReset->setObjectName(QString::fromUtf8("pushButtonReset"));

    hboxLayout9->addWidget(pushButtonReset);

    spacerItem15 = new QSpacerItem(16, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

    hboxLayout9->addItem(spacerItem15);

    pushButtonQuit = new QPushButton(frame5);
    pushButtonQuit->setObjectName(QString::fromUtf8("pushButtonQuit"));
    pushButtonQuit->setCursor(QCursor(Qt::PointingHandCursor));
    pushButtonQuit->setFocusPolicy(Qt::ClickFocus);
    pushButtonQuit->setAutoDefault(false);
    pushButtonQuit->setDefault(false);

    hboxLayout9->addWidget(pushButtonQuit);


    vboxLayout->addWidget(frame5);


    retranslateUi(BeoChipMainForm);
    QObject::connect(sliderSSC0, SIGNAL(valueChanged(int)), lCDNumberSSC0, SLOT(display(int)));
    QObject::connect(sliderSSC1, SIGNAL(valueChanged(int)), lCDNumberSSC1, SLOT(display(int)));
    QObject::connect(sliderSSC2, SIGNAL(valueChanged(int)), lCDNumberSSC2, SLOT(display(int)));
    QObject::connect(sliderSSC3, SIGNAL(valueChanged(int)), lCDNumberSSC3, SLOT(display(int)));
    QObject::connect(sliderSSC4, SIGNAL(valueChanged(int)), lCDNumberSSC4, SLOT(display(int)));
    QObject::connect(sliderSSC5, SIGNAL(valueChanged(int)), lCDNumberSSC5, SLOT(display(int)));
    QObject::connect(sliderSSC6, SIGNAL(valueChanged(int)), lCDNumberSSC6, SLOT(display(int)));
    QObject::connect(sliderSSC7, SIGNAL(valueChanged(int)), lCDNumberSSC7, SLOT(display(int)));
    QObject::connect(pushButtonQuit, SIGNAL(clicked()), BeoChipMainForm, SLOT(close()));
    QObject::connect(radioButtonDec, SIGNAL(clicked()), BeoChipMainForm, SLOT(radioButtonDec_clicked()));
    QObject::connect(radioButtonHex, SIGNAL(clicked()), BeoChipMainForm, SLOT(radioButtonHex_clicked()));
    QObject::connect(checkDebounce, SIGNAL(stateChanged(int)), BeoChipMainForm, SLOT(checkDebounce_stateChanged(int)));
    QObject::connect(checkBoxOut3, SIGNAL(stateChanged(int)), BeoChipMainForm, SLOT(checkBoxOut3_stateChanged(int)));
    QObject::connect(checkBoxOut2, SIGNAL(stateChanged(int)), BeoChipMainForm, SLOT(checkBoxOut2_stateChanged(int)));
    QObject::connect(checkBoxOut1, SIGNAL(stateChanged(int)), BeoChipMainForm, SLOT(checkBoxOut1_stateChanged(int)));
    QObject::connect(checkBoxOut0, SIGNAL(stateChanged(int)), BeoChipMainForm, SLOT(checkBoxOut0_stateChanged(int)));
    QObject::connect(sliderSSC0, SIGNAL(valueChanged(int)), BeoChipMainForm, SLOT(sliderSSC0_valueChanged(int)));
    QObject::connect(sliderSSC1, SIGNAL(valueChanged(int)), BeoChipMainForm, SLOT(sliderSSC1_valueChanged(int)));
    QObject::connect(sliderSSC2, SIGNAL(valueChanged(int)), BeoChipMainForm, SLOT(sliderSSC2_valueChanged(int)));
    QObject::connect(sliderSSC3, SIGNAL(valueChanged(int)), BeoChipMainForm, SLOT(sliderSSC3_valueChanged(int)));
    QObject::connect(sliderSSC4, SIGNAL(valueChanged(int)), BeoChipMainForm, SLOT(sliderSSC4_valueChanged(int)));
    QObject::connect(sliderSSC5, SIGNAL(valueChanged(int)), BeoChipMainForm, SLOT(sliderSSC5_valueChanged(int)));
    QObject::connect(sliderSSC6, SIGNAL(valueChanged(int)), BeoChipMainForm, SLOT(sliderSSC6_valueChanged(int)));
    QObject::connect(sliderSSC7, SIGNAL(valueChanged(int)), BeoChipMainForm, SLOT(sliderSSC7_valueChanged(int)));
    QObject::connect(pushButtonReset, SIGNAL(clicked()), BeoChipMainForm, SLOT(pushButtonReset_clicked()));

    QMetaObject::connectSlotsByName(BeoChipMainForm);
    } // setupUi

    void retranslateUi(QDialog *BeoChipMainForm)
    {
    BeoChipMainForm->setWindowTitle(QApplication::translate("BeoChipMainForm", "BeoChip Interface", 0, QApplication::UnicodeUTF8));
    groupBoxAllSSC->setTitle(QApplication::translate("BeoChipMainForm", "BeoChip Servo Controls", 0, QApplication::UnicodeUTF8));
    textLabelSSC0->setText(QApplication::translate("BeoChipMainForm", "Servo 0", 0, QApplication::UnicodeUTF8));
    textLabelSSC1->setText(QApplication::translate("BeoChipMainForm", "Servo 1", 0, QApplication::UnicodeUTF8));
    textLabelSSC2->setText(QApplication::translate("BeoChipMainForm", "Servo 2", 0, QApplication::UnicodeUTF8));
    textLabelSSC3->setText(QApplication::translate("BeoChipMainForm", "Servo 3", 0, QApplication::UnicodeUTF8));
    textLabelSSC4->setText(QApplication::translate("BeoChipMainForm", "Servo 4", 0, QApplication::UnicodeUTF8));
    textLabelSSC5->setText(QApplication::translate("BeoChipMainForm", "Servo 5", 0, QApplication::UnicodeUTF8));
    textLabelSSC6->setText(QApplication::translate("BeoChipMainForm", "Servo 6", 0, QApplication::UnicodeUTF8));
    textLabelSSC7->setText(QApplication::translate("BeoChipMainForm", "Servo 7", 0, QApplication::UnicodeUTF8));
    textLabelKBD->setText(QApplication::translate("BeoChipMainForm", "Digital In", 0, QApplication::UnicodeUTF8));
    digin4->setToolTip(QApplication::translate("BeoChipMainForm", "Digital In 4", 0, QApplication::UnicodeUTF8));
    digin3->setToolTip(QApplication::translate("BeoChipMainForm", "Digital In 3", 0, QApplication::UnicodeUTF8));
    digin2->setToolTip(QApplication::translate("BeoChipMainForm", "Digital In 2", 0, QApplication::UnicodeUTF8));
    digin1->setToolTip(QApplication::translate("BeoChipMainForm", "Digital In 1", 0, QApplication::UnicodeUTF8));
    digin0->setToolTip(QApplication::translate("BeoChipMainForm", "Digital In 0", 0, QApplication::UnicodeUTF8));
    textLabelDOUT->setText(QApplication::translate("BeoChipMainForm", "Digital Out", 0, QApplication::UnicodeUTF8));
    checkBoxOut3->setToolTip(QApplication::translate("BeoChipMainForm", "Digital Out 3", 0, QApplication::UnicodeUTF8));
    checkBoxOut3->setText(QString());
    checkBoxOut2->setToolTip(QApplication::translate("BeoChipMainForm", "Digital Out 2", 0, QApplication::UnicodeUTF8));
    checkBoxOut2->setText(QString());
    checkBoxOut1->setToolTip(QApplication::translate("BeoChipMainForm", "Digital Out 1", 0, QApplication::UnicodeUTF8));
    checkBoxOut1->setText(QString());
    checkBoxOut0->setToolTip(QApplication::translate("BeoChipMainForm", "Digital Out 0", 0, QApplication::UnicodeUTF8));
    checkBoxOut0->setText(QString());
    textLabelAin->setText(QApplication::translate("BeoChipMainForm", "Analog", 0, QApplication::UnicodeUTF8));
    lCDNumberA0->setToolTip(QApplication::translate("BeoChipMainForm", "Analog In 0", 0, QApplication::UnicodeUTF8));
    lCDNumberA1->setToolTip(QApplication::translate("BeoChipMainForm", "Analog In 1", 0, QApplication::UnicodeUTF8));
    textLabelPWM->setText(QApplication::translate("BeoChipMainForm", "PWM", 0, QApplication::UnicodeUTF8));
    lCDNumberPWM0->setToolTip(QApplication::translate("BeoChipMainForm", "PWM In 0", 0, QApplication::UnicodeUTF8));
    lCDNumberPWM1->setToolTip(QApplication::translate("BeoChipMainForm", "PWM In 1", 0, QApplication::UnicodeUTF8));
    labelSerDev->setText(QApplication::translate("BeoChipMainForm", "Serial Dev: /dev/ttyS0", 0, QApplication::UnicodeUTF8));
    radioButtonDec->setToolTip(QApplication::translate("BeoChipMainForm", "Click for decimal displays", 0, QApplication::UnicodeUTF8));
    radioButtonDec->setText(QApplication::translate("BeoChipMainForm", "Dec", 0, QApplication::UnicodeUTF8));
    radioButtonHex->setToolTip(QApplication::translate("BeoChipMainForm", "Click for hexadecimal displays", 0, QApplication::UnicodeUTF8));
    radioButtonHex->setText(QApplication::translate("BeoChipMainForm", "Hex", 0, QApplication::UnicodeUTF8));
    checkDebounce->setToolTip(QApplication::translate("BeoChipMainForm", "Check to debounce all digital inputs", 0, QApplication::UnicodeUTF8));
    checkDebounce->setText(QApplication::translate("BeoChipMainForm", "Debounce", 0, QApplication::UnicodeUTF8));
    pushButtonReset->setToolTip(QApplication::translate("BeoChipMainForm", "Click to reset BeoChip", 0, QApplication::UnicodeUTF8));
    pushButtonReset->setText(QApplication::translate("BeoChipMainForm", "Reset", 0, QApplication::UnicodeUTF8));
    pushButtonQuit->setToolTip(QApplication::translate("BeoChipMainForm", "Click to quit!", 0, QApplication::UnicodeUTF8));
    pushButtonQuit->setText(QApplication::translate("BeoChipMainForm", "Quit", 0, QApplication::UnicodeUTF8));
    Q_UNUSED(BeoChipMainForm);
    } // retranslateUi

};

namespace Ui {
    class BeoChipMainForm: public Ui_BeoChipMainForm {};
} // namespace Ui

#endif // BEOCHIPMAINFORM_H
