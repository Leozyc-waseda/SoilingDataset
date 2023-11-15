/****************************************************************************
** Form interface generated from reading ui file 'Qt/DescriptorVecDialog.ui'
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#ifndef DESCRIPTORVECDIALOG_H
#define DESCRIPTORVECDIALOG_H

#include <qvariant.h>
#include <qpixmap.h>
#include <qdialog.h>
#include "Channels/DescriptorVec.H"
#include "Qt/ExtraIncludes.H"

class QVBoxLayout;
class QHBoxLayout;
class QGridLayout;
class QSpacerItem;
class ImageCanvas;
class QTabWidget;
class QWidget;
class QTable;
class QPushButton;

class DescriptorVecDialog : public QDialog
{
    Q_OBJECT

public:
    DescriptorVecDialog( QWidget* parent = 0, const char* name = 0, bool modal = FALSE, WFlags fl = 0 );
    ~DescriptorVecDialog();

    QTabWidget* FVtab;
    QWidget* tab;
    ImageCanvas* imgDisp;
    QWidget* TabPage;
    ImageCanvas* histDisp;
    QWidget* TabPage_2;
    QTable* FVtable;
    QPushButton* buttonOk;
    QPushButton* buttonCancel;

    virtual void init( DescriptorVec & dv );
    virtual void update();

protected:
    QVBoxLayout* DescriptorVecDialogLayout;
    QHBoxLayout* tabLayout;
    QVBoxLayout* TabPageLayout;
    QHBoxLayout* TabPageLayout_2;
    QHBoxLayout* layout3;
    QSpacerItem* spacer3;

protected slots:
    virtual void languageChange();

private:
    DescriptorVec *itsDV;

    QPixmap image0;

};

#endif // DESCRIPTORVECDIALOG_H
