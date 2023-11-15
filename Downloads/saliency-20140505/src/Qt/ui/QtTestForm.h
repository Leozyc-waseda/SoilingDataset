/****************************************************************************
** Form interface generated from reading ui file 'Qt/QtTestForm.ui'
**
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#ifndef QTTESTFORM_H
#define QTTESTFORM_H

#include <qvariant.h>
#include <qpixmap.h>
#include <qdialog.h>

class QVBoxLayout;
class QHBoxLayout;
class QGridLayout;
class QSpacerItem;
class QPushButton;
class QLabel;

class QtTestForm : public QDialog
{
    Q_OBJECT

public:
    QtTestForm( QWidget* parent = 0, const char* name = 0, bool modal = FALSE, WFlags fl = 0 );
    ~QtTestForm();

    QPushButton* AboutQtButton;
    QLabel* iLabLogo;
    QPushButton* AboutButton;
    QPushButton* ExitButton;

public slots:
    virtual void showAboutQt();
    virtual void showAboutiLab();

protected:

protected slots:
    virtual void languageChange();

private:
    QPixmap image0;

};

#endif // QTTESTFORM_H
