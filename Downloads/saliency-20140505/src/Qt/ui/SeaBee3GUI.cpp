/****************************************************************************
** Form implementation generated from reading ui file 'Qt/SeaBee3GUI.ui'
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#include "Qt/ui/SeaBee3GUI.h"

#include <qvariant.h>
#include <qgroupbox.h>
#include <qframe.h>
#include <qlabel.h>
#include <qlineedit.h>
#include <qcheckbox.h>
#include <qspinbox.h>
#include <qbuttongroup.h>
#include <qradiobutton.h>
#include <qtabwidget.h>
#include <qwidget.h>
#include <qpushbutton.h>
#include <qlcdnumber.h>
#include <qcombobox.h>
#include <qlayout.h>
#include <qtooltip.h>
#include <qwhatsthis.h>
#include <qaction.h>
#include <qmenubar.h>
#include <qpopupmenu.h>
#include <qtoolbar.h>
#include <qimage.h>
#include <qpixmap.h>

#include "Qt/ImageCanvas.h"
#include "Qt/SeaBee3GUI.ui.h"
static const char* const image0_data[] = {
"19 38 75 2",
"#b c #342e11",
".b c #343d46",
".g c #404040",
".c c #414141",
".V c #42390f",
".f c #474747",
".e c #4a4a4a",
".d c #4b4b4b",
".a c #526477",
"#h c #584b0e",
".# c #5d7085",
".X c #5e5b4e",
".h c #5e5e5e",
"#f c #64550d",
"Qt c #657a90",
"#g c #66560e",
"#e c #66580f",
"#c c #665810",
"#d c #675911",
".7 c #67614a",
".l c #766d33",
".R c #786b23",
".Y c #796d24",
".O c #807327",
".x c #807328",
".E c #817427",
".K c #817428",
"#i c #86847c",
".i c #868686",
".j c #888888",
".5 c #9f9d95",
".U c #a9a48f",
"#. c #aaa48e",
"#a c #b3ae99",
".D c #b4ae99",
".N c #b4af99",
".w c #b4af9f",
".v c #c0a529",
".4 c #c6a923",
".6 c #c8aa23",
".k c #c8c8c8",
".W c #ceaf25",
".C c #d3b426",
".9 c #d5b527",
".J c #d5b627",
".p c #d5b728",
".M c #d6b627",
".L c #d6b727",
".q c #d7b828",
".s c #d7b928",
".r c #d8b929",
".t c #d8ba29",
".o c #d8bb2b",
".n c #d8bb2c",
".Q c #d9ba28",
".m c #d9bc2c",
".2 c #daba28",
".u c #dcbe2c",
".Z c #e0bf29",
".T c #e3c22a",
".3 c #f4d12d",
".S c #f5d22e",
".B c #f6d22e",
".8 c #f6d32e",
".y c #f7d22e",
"## c #f7d22f",
".z c #f7d32e",
".A c #f7d32f",
".1 c #f7d42e",
".P c #f7d42f",
".0 c #f8d32f",
".I c #f8d42e",
".G c #f8d42f",
".H c #f9d42f",
".F c #f9d52f",
"Qt.#.#.#.#.#.#.a.b.c.d.d.e.f.g.h.i.j.k",
".l.m.m.n.o.m.m.o.p.q.r.s.s.q.q.t.u.v.w",
".x.y.z.z.z.y.A.B.z.B.B.y.B.y.y.y.z.C.D",
".E.F.G.H.I.G.I.G.I.F.H.I.I.I.H.G.G.J.D",
".K.H.H.H.F.F.F.H.I.G.I.F.I.G.F.I.I.L.D",
".K.I.H.I.I.I.I.F.G.G.I.H.I.H.I.I.G.L.D",
".E.I.I.H.H.I.H.I.I.G.G.H.I.I.H.I.G.M.D",
".K.H.H.H.I.G.H.I.H.H.I.I.I.I.G.I.I.J.N",
".E.H.I.I.G.G.I.I.F.G.G.I.I.G.I.I.G.M.D",
".O.G.G.G.H.I.I.I.H.H.I.I.H.F.H.H.G.M.D",
".E.F.G.H.I.H.I.G.I.I.H.G.I.I.I.I.H.M.D",
".K.H.I.I.I.I.I.I.H.I.G.G.I.H.I.P.I.Q.D",
".R.y.I.H.z.F.G.z.I.S.y.z.G.H.I.H.I.T.U",
".V.z.G.H.I.I.I.G.z.I.I.G.G.I.I.G.I.W.X",
".Y.G.G.I.I.I.H.I.z.H.G.H.I.I.I.G.H.Z.U",
".O.I.F.I.I.I.F.F.F.F.H.H.H.I.G.I.I.L.D",
".E.I.I.I.I.I.I.I.I.I.I.F.G.G.I.H.I.L.D",
".K.H.I.H.I.I.H.H.I.I.z.H.G.I.I.G.z.L.D",
".O.G.H.G.F.G.0.I.G.I.A.I.z.I.G.H.I.L.D",
".K.I.H.G.I.F.1.G.G.I.1.H.I.I.I.I.I.J.D",
".K.G.I.H.G.H.G.H.I.G.z.I.z.I.I.1.F.2.D",
".K.I.F.F.I.I.H.I.H.I.A.z.I.3.I.z.I.4.5",
".K.G.G.G.H.G.I.I.I.H.z.H.I.z.z.H.H.6.7",
".E.I.I.G.H.I.H.I.F.0.z.I.8.F.I.0.I.9#.",
".K.G.G.G.I.H.I.I.I.I.G.z.I.I.H.I.I.J.N",
".K.F.I.G.G.H.G.I.I.G.I.z.G.H.I.I.I.J.D",
".E.I.I.H.G.I.G.I.G.I.F.G.G.H.H.I.I.L.D",
".K.H.I.H.I.I.I.H.G.G.I.I.G.I.I.I.G.J.D",
".O.I.F.G.I.H.I.G.H.I.H.I.A.G.I.H.I.J.D",
".E.I.I.I.G.H.I.I.G.z.z##.z.F.I.F.I.J.N",
".K.G.G.I.I.I.I.G.I.I.I.B.H.I.G.G.F.J#a",
".E.I.H.I.G.H.F.I.F.I.I.I.I.H.I.I.I.J.D",
".O.I.I.I.I.I.F.I.I.H.G.I.I.F.H.H.G.J.N",
".K.F.I.I.G.G.G.I.I.I.H.I.I.I.G.G.G.L.D",
".K.I.I.G.I.F.G.H.F.I.G.I.H.I.G.G.I.L.D",
".K.F.F.I.F.H.G.I.H.I.G.H.G.I.H.I.H.J.D",
".K.G.I.I.I.G.I.G.I.G.G.F.G.I.G.I.I.J.D",
"#b#c#c#c#d#c#c#d#c#d#e#f#g#g#g#g#g#h#i"};

static const char* const image1_data[] = {
"39 16 264 2",
"#5 c #5c656f",
".p c #5c738d",
".q c #5c748e",
"bO c #5d6977",
"#Q c #626e7e",
"#q c #646f7b",
"#b c #657688",
"ak c #687786",
"#r c #69727f",
"bs c #6a7788",
".6 c #6a7887",
"aG c #6a7889",
"az c #6a7d92",
".r c #6a7e96",
"bf c #6b7885",
"#k c #6e7e90",
"bP c #708296",
"bN c #717a86",
"aA c #71849b",
"al c #7289a4",
".w c #728dac",
"#F c #748fae",
"bo c #768ea9",
"aO c #7b7a7a",
"ap c #7b8fa8",
"#G c #7b92ab",
"#6 c #7b94b1",
"#E c #7b98b7",
"a# c #7c8289",
"bz c #7d858e",
"bA c #7d8a98",
"#U c #7d95b2",
"#c c #7e8895",
"aq c #7f91a2",
"aP c #80858b",
"ao c #8096b1",
"bZ c #8191a3",
"aX c #8193a8",
"#V c #8196af",
".H c #8397af",
".I c #8498ae",
"be c #849ebd",
"a6 c #858b92",
".v c #8695a5",
"b0 c #8699af",
"bp c #86a2c1",
".1 c #888989",
"bB c #88a3c0",
"a7 c #8a95a3",
".0 c #8a99ab",
"bU c #8a9cb1",
"bC c #8aa5c6",
"c# c #8b8d8f",
".Z c #8b9eb5",
"#p c #8ba5c5",
"aF c #8ba8c8",
"#H c #8c97a3",
"aB c #8ca6c5",
"ar c #8d9092",
"ca c #8d949c",
".J c #8d9cad",
"bn c #8e9cac",
"a. c #8ea2b9",
"aQ c #8ea2ba",
".S c #8ea7c3",
"#W c #8f9caa",
".2 c #90908f",
"aR c #90a6bf",
"#a c #90aaca",
"aW c #90adcd",
"ay c #9198a1",
".R c #919ca9",
"bt c #929292",
"#s c #9298a2",
"#X c #929ea9",
"cb c #92a5ba",
"a8 c #92a9c3",
"b6 c #92aac5",
"bI c #93979e",
"aa c #93989c",
".L c #939ba3",
"cc c #93a8c2",
"#9 c #93aac6",
".d c #93aac9",
".Y c #93aece",
"#B c #93afce",
"#P c #949494",
".K c #949aa3",
".s c #94a1ad",
"bH c #94afcc",
".o c #94b5d9",
"#A c #959ba2",
"a9 c #95aecc",
"aS c #95afcc",
".G c #95b3d6",
".X c #95b4d8",
"bY c #96989a",
"aj c #979fa9",
"aY c #97a0ab",
"bm c #97a1ad",
"bD c #97b5d7",
".e c #98b2d2",
"#o c #98b7dc",
"b1 c #99b5d6",
"cd c #99b5d7",
"#I c #9aa3ad",
"#4 c #9b9c9d",
"bl c #9ba4ae",
"#8 c #9bbadd",
".F c #9cbce2",
"ce c #9dbce1",
".u c #9ea7b2",
"b2 c #9ebde0",
"br c #9ebde1",
".c c #9fbadc",
"bX c #a09e9c",
"#7 c #a0c0e6",
"c. c #a1a2a2",
"an c #a1bfe3",
"#T c #a1c3e9",
".M c #a2a8ae",
"#Y c #a2a9b1",
".t c #a2aab5",
"aT c #a2c1e6",
"ax c #a3a5a9",
"b. c #a3c3e7",
"aU c #a3c3e8",
"b3 c #a3c4e7",
"b# c #a4c4ea",
"b7 c #a5a9ae",
"#R c #a5c7eb",
"bQ c #a6c6ed",
"## c #a7c7ed",
"as c #a8a9ab",
".f c #a8c8ec",
"ba c #a8caf0",
"bg c #a9adb2",
"bR c #a9caef",
".W c #a9cbf3",
"aw c #aaacae",
"aH c #abb0b6",
"#. c #abccf1",
"b5 c #abccf4",
"bd c #accef6",
"am c #accff6",
"#S c #accff9",
"aC c #adcff5",
"#D c #adcff9",
"aD c #add0f8",
".b c #aecff7",
"#n c #aed0f7",
"aE c #aed1f6",
"bS c #aed1f9",
".h c #afd1f8",
".9 c #afd1f9",
"bT c #afd3fa",
".g c #b0d1f8",
"aV c #b0d3fb",
".E c #b0d4fd",
"#d c #b1b2b4",
".8 c #b1d3fb",
"b4 c #b1d4fc",
"bF c #b1d5fd",
"bE c #b1d5fe",
"bM c #b2b6b9",
".i c #b2d5fc",
".a c #b2d5fd",
"cf c #b2d5fe",
"bc c #b2d6fd",
".x c #b2d6fe",
"bq c #b2d6ff",
".# c #b3d5fd",
".7 c #b3d5fe",
"Qt c #b3d6fe",
"bb c #b3d6ff",
".j c #b3d7ff",
".z c #b3d8ff",
".A c #b3d9ff",
"bk c #b4b4b5",
"aI c #b4b9be",
"#l c #b4d6fe",
"bG c #b4d6ff",
".B c #b4d7ff",
".y c #b4d8ff",
".n c #b4d9ff",
".T c #b5d7ff",
".U c #b5d8ff",
".C c #b5d9ff",
".V c #b5daff",
"#m c #b6daff",
".l c #b7dbff",
".k c #b8dbff",
".m c #b9dfff",
"ai c #babdc2",
".D c #badeff",
"#C c #bce2ff",
"bj c #bdbbb7",
".3 c #bdbcbc",
"a5 c #bdbdbe",
"by c #bdc1c5",
".Q c #bfc1c4",
"ah c #c0c4c8",
"#J c #c1c3c4",
"#e c #c3c3c5",
"aJ c #c3c7ca",
"#Z c #c5c5c4",
"#K c #c5c6c7",
"aZ c #c7c9cd",
"bV c #c9c9cb",
"bL c #cacbcd",
".N c #cfd0d0",
"bx c #d0d2d4",
"bh c #d2d2d0",
"#t c #d2d4d8",
"a4 c #d4d3d0",
"#j c #d4d5d7",
"#0 c #d5d4d3",
"bw c #d7d8d9",
"#3 c #d8d7d6",
".P c #d9d8d8",
"#L c #d9d9d9",
"a3 c #dad8d6",
"#u c #dadcdf",
"ab c #dbdbdb",
"ag c #dcdcdc",
".5 c #dcddde",
".O c #dddcdc",
"aN c #e0dfde",
"av c #e3e3e4",
"#z c #e5e5e5",
"ac c #e6e6e5",
"a0 c #e6e6e6",
"aK c #e7e7e7",
"af c #e8e7e6",
"ae c #ebeae9",
"b9 c #ebebea",
"a1 c #ececec",
"aL c #ededee",
"ad c #eeeded",
"#v c #eeeeef",
"bv c #efeeee",
"bW c #f0efee",
"a2 c #f2f1f1",
"#w c #f3f3f3",
"at c #f3f4f3",
"bJ c #f4f4f4",
"bK c #f6f6f6",
"bu c #f7f6f5",
"#f c #f7f7f7",
"#2 c #f8f6f4",
"b8 c #f8f6f5",
"#O c #f8f7f7",
"#g c #f8f8f7",
"au c #f9f8f8",
"#M c #fafafa",
"bi c #fbfaf8",
"#h c #fbfbfa",
"#N c #fcfcfc",
"#1 c #fefdfb",
"#i c #fefefd",
"#x c #fefefe",
"#y c #fffefe",
"aM c #fffffe",
".4 c #ffffff",
"QtQt.#Qt.a.#Qt.a.aQtQt.aQt.a.b.c.d.e.f.g.h.iQt.j.k.l.m.n.o.p.q.r.s.t.u.v.w.a.a",
"QtQt.#Qt.aQtQt.aQtQtQt.#Qt.a.x.y.z.A.j.B.B.C.D.E.F.G.H.I.J.K.L.M.N.O.P.Q.R.S.#",
".#.#.#.a.aQtQt.#QtQtQt.#.a.a.a.j.T.C.U.V.C.W.X.Y.Z.0.1.2.3.4.4.4.4.4.4.4.5.6.#",
".#Qt.a.a.7Qt.#QtQt.a.aQt.a.a.j.x.#.8.9#.###a#b#c#d#e#f#g#h.4.4.4.4.4.4#i#j#k.a",
".#Qt.a.a.a.aQt.a.aQt.a.aQt#l#m#m#n#o#p#q#r#s#t#u#v#w#x#y.4.4.4.4.4.4.4#z#A#B.a",
".a.aQt.aQt.#Qt.#.aQt.B#m#C#D#E#F#G#H#I#J#K#L#M#N.4.4.4.4.4.4.4.4.4.4#O#P#Q#RQt",
".aQtQt.aQt.#.B.l.k.y#S#T#U#V#W#X#Y#Z#0.4.4.4.4.4.4.4.4.4.4.4.4#1#2#3#4#5#6.hQt",
".aQt.#.a.a.T.a#T#7#8#9a.a#aaabacad.4.4.4.4.4.4.4.4.4.4.4aeafagahaiajakalam.7Qt",
"Qt.aQt.a.a.hanaoapaqarasatau#i#x#x.4.4.4.4.4.4.4.4#N#wavawaxayazaAaBaCaD.7.aQt",
"Qt.##l.BaEaFaGaHaIaJaKaL#x.4.4.4.4.4.4.4.4.4.4.4aMaNaOaPaQaRaSaTaU#..i.aQt.a.a",
".a.BaVaWaXaYaZa0a1a2.4.4.4.4.4.4.4.4.4.4.4a3a4a5a6a7a8a9b.b#babbbbbcQt.x.aQtQt",
".#bdbebfbgbh.4.4.4.4.4.4.4.4.4.4.4.4bibjbkblbmbnbobpbbbq.B.j.j.B.jQt.#.a.#QtQt",
".#brbsbta2.4.4.4.4.4.4.4.4.4bubvbwbxbybzbAbBbCbDbEbFbGbbbb.BQt.#.aQt.a.a.#Qt.a",
"#nbHbIbJ.4.4.4.4.4.4.4#xbKa1bLbMbNbObPbQbRbSbTaVQtQt.a.a.aQt.a.#.a.#.a.a.a.aQt",
"b#bUbV.4.4.4.4.4.4.4.4bWbXbYbZb0b1b2b3aVb4QtQt.a.a.a.a.a.aQt.a.a.a.a.a.a.#.aQt",
"b5b6b7b8.4.4.4.4b9c.c#cacbcccdce.hcfQtbbQtQtQt.a.a.a.a.aQt.aQtQt.a.aQt.a.a.a.#"};

static const char* const image2_data[] = {
"22 17 136 2",
".5 c #585d6c",
"#d c #65788d",
"aa c #686d7e",
".j c #6a6168",
".L c #6c7382",
"#5 c #6e3e2e",
".d c #6e849d",
".K c #6f6670",
"#K c #6f6b77",
".Q c #717687",
"#i c #735855",
".k c #73707c",
"#p c #748ba6",
"a# c #77696f",
".x c #788da8",
".B c #794735",
"#c c #7a686d",
"#1 c #7c5850",
".h c #7d442d",
".0 c #7e5c5a",
".q c #7e7884",
".e c #7f99b5",
"#2 c #80452f",
"#Q c #80757d",
".1 c #80899d",
"#F c #816b6e",
"a. c #837f8d",
"#j c #845547",
"#R c #856764",
".2 c #85a0bf",
"#t c #867b84",
".w c #875f56",
"#u c #884b30",
".v c #8b594a",
"#U c #8c492d",
"#y c #8d5542",
".Z c #8d5845",
"#h c #8e94a8",
"#e c #8eabca",
"#G c #904e35",
"#P c #90a0ba",
".g c #90a4c0",
"#z c #90aece",
"#0 c #9194a9",
"#v c #923d10",
"#s c #95acc9",
"#E c #96a1b9",
"#V c #96b0d1",
"#9 c #98acca",
".M c #99b4d4",
"#r c #99b7db",
"#g c #9bb8da",
".l c #9bbadc",
"#D c #9caeca",
".6 c #9d4617",
".c c #9dbddf",
"#Z c #9eb9d9",
".m c #9ebde0",
"#8 c #9fb9da",
"#O c #9fbbdf",
"ad c #9fbee2",
"#o c #a05737",
"#b c #a05c42",
"ae c #a0c0e6",
"#S c #a14613",
"#Y c #a1c1e6",
".A c #a2bcde",
".3 c #a2c1e6",
"#C c #a3c2e7",
"#6 c #a4c4e9",
"ac c #a4c5ea",
".N c #a6c7ec",
".J c #a75d3f",
".4 c #a7c8ee",
".i c #a85e41",
".P c #a9cbf1",
"ab c #aacbf0",
".p c #abcbf1",
"#q c #abccf3",
".y c #accef3",
"#L c #accef5",
"#X c #aecff6",
".O c #afd1f8",
".f c #afd1f9",
"#a c #b05c33",
"#B c #b0d1fa",
"#f c #b0d2fa",
"#N c #b0d3fc",
".I c #b15e38",
"af c #b1d3fa",
".b c #b1d3fb",
"#7 c #b1d3fc",
".z c #b1d4fc",
"#A c #b1d5fd",
"#W c #b2d4fd",
".n c #b2d5fb",
".# c #b2d5fd",
"#M c #b2d6fe",
".a c #b3d5fd",
".o c #b3d5fe",
"Qt c #b3d6fe",
".r c #b65017",
"#3 c #bb5620",
"#k c #be5218",
".u c #be5927",
".Y c #c15d2b",
"#4 c #c6571a",
".R c #c6591e",
".t c #c75a20",
"#H c #c8581c",
"#J c #c95c25",
".X c #c95d24",
"#n c #ca5e27",
"## c #cc5b1f",
"#w c #cf5b1c",
".H c #cf5d1f",
"#I c #d05b1c",
".G c #d45d1d",
"#. c #d55e1e",
"#T c #db611e",
"#m c #dd6220",
"#l c #df621f",
".s c #df6320",
".W c #e06320",
"#x c #e06422",
".C c #e1641f",
".V c #e26420",
".8 c #e66521",
".7 c #e76621",
".9 c #e86621",
".F c #e86721",
".U c #e96621",
".T c #ea6721",
".D c #ea6821",
".E c #eb6821",
".S c #eb6822",
"Qt.#.#Qt.a.a.#.#QtQt.#.#QtQt.#Qt.#.a.#Qt.#.#",
"Qt.#Qt.#Qt.aQt.#.#Qt.#.#Qt.#.aQt.#.aQt.#Qt.#",
"Qt.b.c.d.e.a.#Qt.a.#.#QtQtQt.#.#.#.a.aQt.#.#",
"Qt.f.g.h.i.j.k.l.m.n.#.#.#.#.#.o.#.#.#Qt.a.#",
".#.p.q.r.s.t.u.v.w.x.e.yQt.#.#.#.#.#.#.#.#.a",
".z.A.B.C.D.E.F.G.H.I.J.K.L.M.N.O.#Qt.a.#QtQt",
".P.Q.R.S.T.D.D.T.U.V.W.X.Y.Z.0.1.2.3QtQtQt.#",
".4.5.6.7.8.T.T.S.S.S.D.S.9#.###a#b#c#d#eQt.#",
"#f#g#h#i#j#k#l.D.S.D.T.D.E.S.T.V#m#n#o#p.a.a",
".#.z#q#r#s#t#u#v#w.S.T.S.D.T.S.D.T#x#y#z.a.#",
".#QtQt#A#B#C#D#E#F#G#H#I.T.E.T.S.9#J#K#LQt.a",
"QtQt.#Qt.a#M#N.O#O#P#Q#R#S#T.D.T#l#U#V.a.#.#",
"Qt.#.a.#.aQt.#Qt#W#X#Y#Z#0#1#2#3#4#5#6QtQt.#",
"QtQt.#.aQtQt.#Qt.#Qt.##7#q#8#9a.a#aaabQt.a.#",
".#Qt.#Qt.#.#Qt.#.#Qt.a.#Qt.#.zacadaeafQt.aQt",
".#QtQtQt.#Qt.a.#Qt.#.#.#.#QtQt.#.#.aQtQt.#.a",
".#.#.#Qt.#Qt.a.#.aQt.#.#.a.#Qt.a.#QtQt.#.#.#"};

static const char* const image3_data[] = {
"44 44 1014 2",
"#g c #3c4d60",
"#f c #3d4f62",
"#k c #3e4f62",
"#j c #3e5063",
"#h c #3f5064",
"#i c #405266",
"#l c #415366",
"bv c #44414b",
"#m c #44576b",
"mY c #462a30",
"b5 c #464755",
"#e c #46596e",
"mX c #472c32",
"b4 c #482b2e",
"m0 c #482e33",
"cB c #484856",
"mZ c #492e34",
"#n c #495d73",
"bu c #4a2624",
"cA c #4a3036",
"m1 c #4a3037",
"m2 c #4b343c",
"mW c #4e3740",
"aX c #4e4853",
"nm c #4e576d",
"m3 c #4f3b43",
"nl c #4f586f",
"#d c #4f657b",
"nq c #50596f",
"np c #505b70",
"c7 c #513134",
"nn c #515b71",
"c8 c #524e5e",
"no c #525c73",
"bw c #52657a",
"aY c #52677c",
"nr c #536078",
"gL c #546981",
"mV c #55444e",
"b6 c #55677c",
"cC c #556b82",
"gZ c #556b83",
"nk c #56627b",
".O c #567c9c",
"ns c #57667f",
"gu c #576d86",
".N c #577e9d",
".S c #587e9d",
".R c #587f9e",
"aW c #593637",
"#o c #597089",
".P c #597f9f",
"#I c #5a3734",
"#H c #5a3836",
".Q c #5a80a0",
".T c #5a81a0",
"cz c #5b211a",
"#M c #5b3836",
"#L c #5b3937",
"m4 c #5b4f5b",
"gK c #5b5361",
"b3 c #5c1c12",
"#J c #5c3937",
"#K c #5c3b39",
"gY c #5c5461",
"#N c #5d3b39",
"nj c #5d6d88",
"hb c #5d738e",
".U c #5d84a4",
"gt c #5e5665",
"gc c #5e758f",
"bt c #5f1b0e",
"#O c #5f3f3d",
"iM c #5f5462",
"fF c #5f5766",
"ha c #5f5865",
"#c c #5f7790",
".M c #5f86a6",
"c6 c #60231b",
"#G c #60403f",
"mU c #605866",
"fo c #615867",
"gb c #615b6a",
".V c #6189aa",
"dC c #62424e",
"#P c #624444",
"iq c #625867",
"fW c #625c6c",
"fG c #627a96",
"hl c #635d6b",
"nt c #647793",
"gM c #647c97",
"#F c #654848",
"c9 c #657892",
"iN c #657c99",
"fp c #657f9b",
"h7 c #665f6f",
"hU c #666070",
"hm c #667e9b",
"hr c #676170",
"hD c #676171",
"ir c #67809d",
"g0 c #67819c",
".L c #678fb1",
"i4 c #686172",
"gv c #68829e",
"aZ c #688ca9",
"ni c #697f9d",
"hK c #6a3a40",
"hy c #6a3c43",
"#Q c #6a5257",
"dB c #6b2c28",
"gh c #6b3d43",
"f2 c #6b3e45",
"m5 c #6b697c",
"ap c #6b7991",
"g5 c #6c414a",
"e8 c #6c6a7c",
"h8 c #6c86a3",
"gA c #6d4148",
"gR c #6d424a",
"#E c #6d575a",
"dD c #6d6a82",
"hV c #6d87a4",
"hs c #6d87a5",
"b7 c #6d8ca8",
"cD c #6d92b1",
"h0 c #6e454e",
"lp c #6e5964",
"i5 c #6e87a5",
"bx c #6e89a4",
".W c #6e97ba",
"hJ c #6f6174",
"ao c #6f697a",
"hE c #6f88a7",
"fL c #704953",
"hx c #706377",
"gg c #706476",
"f1 c #706578",
"mT c #707286",
"#p c #708ba8",
"aV c #712a20",
"kR c #715860",
"gz c #71677a",
"lo c #717891",
"gd c #718caa",
"lq c #723e3a",
"g4 c #72677d",
"gQ c #72687c",
"kQ c #726f83",
"aq c #7292b0",
"mx c #732415",
"mw c #732516",
"mA c #732617",
"e9 c #7390af",
".K c #739dc0",
"my c #742617",
"mz c #742819",
"mB c #74281a",
"mC c #742b1d",
"hZ c #746b80",
"hc c #748fae",
"mv c #752c20",
"fv c #755866",
"kh c #755961",
"fK c #756e84",
"kg c #75768b",
"nu c #758fb0",
"mD c #763023",
"ic c #765966",
"#R c #766670",
"lT c #766774",
"#b c #7692af",
"mu c #77352b",
"an c #775860",
"gN c #778ea9",
"kS c #783c35",
"lS c #78819a",
"ln c #798dac",
"fX c #7996b6",
"lE c #7a3c3c",
"mE c #7a3d35",
"gX c #7a4543",
"gJ c #7a4644",
"lU c #7a5050",
"lF c #7a5764",
"ma c #7a6473",
"nh c #7a97b8",
"c5 c #7b1d09",
"b2 c #7b1e08",
"cy c #7b1e0b",
"iL c #7b4039",
"mt c #7b433c",
"fn c #7b4540",
"h# c #7b4745",
"gs c #7b4846",
"m# c #7b545b",
"jW c #7b5f68",
"#D c #7b707c",
"fu c #7b7a93",
"jp c #7b8097",
"kP c #7b88a2",
"d2 c #7c3d42",
"ip c #7c433f",
"hk c #7c4947",
"d3 c #7c6074",
"ib c #7c7c94",
"jV c #7c829b",
"eO c #7c8399",
"gw c #7c94b0",
"g1 c #7c94b1",
"ki c #7d423a",
"hT c #7d4947",
"fV c #7d4a47",
"ga c #7d4a48",
"mF c #7d4d4b",
"fd c #7d6a7e",
"aD c #7d7893",
"m6 c #7d849d",
"nL c #7d94b4",
"nK c #7d95b5",
"lr c #7e2a17",
"i3 c #7e4c4a",
"e7 c #7e5052",
"j8 c #7e5a67",
"nP c #7e96b6",
"nO c #7e97b6",
"kf c #7e98ba",
"lD c #7f2f24",
"ms c #7f5453",
"b. c #7f6675",
"ix c #7f7184",
"kG c #7f7a94",
"mb c #7f7d96",
"mS c #7f89a1",
"aC c #7f8cac",
"nM c #7f97b7",
"nN c #7f98b7",
"lR c #7f9bbf",
"fH c #7f9dbe",
"dN c #803c31",
"dh c #805c62",
"#S c #807989",
"nQ c #8098b8",
"nR c #8099ba",
"bs c #811700",
"am c #814c49",
"mG c #815d63",
"mr c #815f62",
"j9 c #817d98",
"nJ c #819abb",
".X c #81abd2",
"dA c #822816",
"jx c #82646c",
"bI c #826775",
"a9 c #827789",
"lG c #827a92",
"nS c #829dbe",
"hn c #82a0c2",
"lV c #833f31",
"jX c #83443c",
"m. c #834742",
"bJ c #834f51",
"b# c #835761",
"dM c #836167",
"eN c #83626c",
"jo c #83656e",
"jq c #83a1c3",
"kF c #845d68",
"mq c #846d75",
"fc c #848aa7",
"ar c #84add0",
"hL c #852a1d",
"hz c #852b1f",
"g6 c #852f25",
"ea c #854a45",
"ce c #856470",
"mH c #857380",
"d. c #85a0c0",
"nI c #85a0c2",
"eP c #85a7c8",
"b8 c #85add0",
"kT c #862b16",
"gi c #862c1f",
"f3 c #862d21",
"gB c #862f23",
"gS c #863025",
"h1 c #863329",
"lb c #865d66",
"lc c #867a8e",
"#C c #86889c",
"iw c #8690ae",
"lm c #86a0be",
"#q c #86a5c6",
"fq c #86a6c8",
"nv c #86a7cc",
".J c #86b0d5",
"fM c #87362d",
"id c #87403b",
"cf c #874445",
"aE c #875e6a",
"jO c #87616c",
"#8 c #877e96",
"jw c #878296",
"eU c #87829b",
"hI c #878faa",
"gf c #8790ac",
"hw c #8791ac",
"dE c #8797b5",
"aB c #879fc0",
"ge c #87a1c0",
"iO c #87a7ca",
"fw c #88403c",
"cL c #886067",
"eq c #887689",
"dg c #887f91",
"mp c #888395",
"cK c #888398",
"bH c #88849c",
"a8 c #8890a9",
"f0 c #8891ad",
"#7 c #8899bd",
"jU c #889ebf",
"kO c #88a6c7",
"is c #88a9cc",
"a0 c #88acce",
"eb c #892710",
"bK c #894642",
"fe c #894f51",
"iy c #895152",
"jP c #898197",
"gy c #8992af",
"gP c #8993b0",
"g3 c #8994b0",
"kH c #899dc1",
"nT c #89a6c9",
"ng c #89acd0",
"lQ c #89acd2",
"di c #8a4037",
"la c #8a413d",
"jy c #8a463e",
"#9 c #8a6970",
"mI c #8a869b",
"d4 c #8a8ca7",
"hY c #8a95b1",
"#a c #8aa9ca",
"ep c #8b525a",
"eV c #8b6169",
"j# c #8b686e",
"ey c #8b6c77",
"cd c #8b869e",
"iS c #8b8ea9",
"#T c #8b90a7",
"fJ c #8b97b4",
"mc c #8b9cbd",
"h9 c #8baccf",
"d1 c #8c2f23",
"ez c #8c463c",
"al c #8c483e",
"iT c #8c6c76",
"e# c #8c7886",
"mo c #8c96af",
"mR c #8c9eb9",
"nH c #8caace",
"i6 c #8cadd0",
"hW c #8caed1",
".m c #8cafd4",
".l c #8cb0d4",
".p c #8cb0d5",
"cE c #8cb1d4",
"aU c #8d2913",
"mJ c #8d93ac",
"ld c #8d9bb9",
"er c #8d9db9",
"hd c #8da7c8",
"ht c #8dadd2",
".n c #8db0d5",
".q c #8db1d5",
".o c #8db1d6",
"ke c #8db1d8",
"lW c #8e3d2a",
"j7 c #8e4440",
"cM c #8e453f",
"ja c #8e4b42",
"m7 c #8ea1bf",
"by c #8eaccd",
"#6 c #8eacd3",
".r c #8eb2d6",
"lC c #8f2710",
"kj c #8f2f17",
"l9 c #8f3e2f",
"a. c #8f554e",
"ex c #8f93b0",
"lH c #8f9bb9",
"ft c #8f9ebd",
"eT c #8f9ebf",
"hF c #8fb0d5",
".k c #8fb3d8",
"cg c #903022",
"j. c #90889c",
"ia c #909fbe",
"mn c #90a7c7",
".s c #90b5da",
"ls c #912000",
"l# c #912c1b",
"jN c #914541",
"cJ c #919fbc",
"#B c #919fbd",
"#U c #91a2bf",
"f. c #91b3d7",
"dO c #92270f",
"dL c #928ea4",
"bG c #92a2c1",
"a7 c #92a4c3",
"cc c #92a9cb",
"nU c #92b3d8",
"aA c #92b5d9",
".j c #92b8dd",
"jY c #933219",
"kE c #934b46",
"iU c #934e45",
"k. c #93a2c3",
"iR c #93a8cb",
"#V c #93accc",
"fY c #93afd1",
"a# c #944330",
"ba c #94453f",
"jv c #949fba",
"mK c #94a4c3",
".Y c #94c0e8",
"bL c #954339",
"fb c #95a8c9",
"ll c #95b5d7",
"nG c #95b7dc",
"lP c #95b8db",
"kN c #95b8de",
"#5 c #95bae1",
"ak c #964534",
"eM c #964945",
"aF c #964d45",
"jn c #96504b",
"e. c #96a3c3",
".t c #96bbe1",
"es c #96bde1",
"eA c #972a0e",
"eW c #97473b",
"df c #97a3be",
"i9 c #97a8c9",
"iv c #97accd",
"ew c #97aed3",
"nf c #97bee6",
".I c #97c0e7",
"kU c #982100",
"aa c #983e26",
"#W c #98b9dd",
"mm c #98b9df",
"jr c #98bcdf",
".i c #98bde3",
"nw c #98c0e8",
"l. c #99240a",
"jz c #99351d",
"eo c #99362f",
"jT c #99b6d8",
"md c #99b7db",
"le c #99b7dd",
"cN c #9a301f",
"ab c #9a3d26",
"mQ c #9ab5d6",
"b9 c #9ac1e7",
"lX c #9b402a",
"m8 c #9bb5d7",
"hH c #9bb9dc",
"fZ c #9bbadd",
"jQ c #9bbfe3",
"nV c #9bbfe6",
"eQ c #9bc0e6",
"ag c #9c3c23",
"ah c #9c3d25",
"ai c #9c3e27",
"aj c #9c402a",
"#A c #9cb7dc",
"hv c #9cbadd",
"gx c #9cbbde",
"gO c #9cbbdf",
"cb c #9cbde3",
"## c #9cbee2",
"bF c #9cbee5",
"as c #9cc2e8",
"ch c #9d2f19",
"dj c #9d3520",
"jb c #9d361d",
"ac c #9d3c23",
"ae c #9d3d24",
"af c #9d3d25",
"l8 c #9d3d28",
"iz c #9d3e29",
"eS c #9db4d8",
"d5 c #9db5d6",
"a6 c #9db6d9",
"mL c #9db8dc",
"lI c #9dbbdd",
"g2 c #9dbcdf",
"fI c #9dbce0",
"#r c #9dbfe4",
"az c #9dc2e8",
"nF c #9dc2e9",
"c4 c #9e2202",
"cx c #9e2b0d",
"ad c #9e3c23",
"cI c #9eb5d6",
"kI c #9ebbe2",
"ml c #9ec3ea",
".u c #9ec4eb",
"#4 c #9ec9f2",
"dz c #9f2809",
"e6 c #9f4334",
"dK c #9fb6da",
"d9 c #9fbce2",
"ho c #9fbde2",
"lk c #9fc1e6",
"kd c #9fc3e9",
"ju c #a0bade",
"iQ c #a0badf",
"fs c #a0bfe4",
"i# c #a0c0e4",
".h c #a0c6ec",
"ec c #a11a00",
"b1 c #a12302",
"iV c #a1371c",
"m9 c #a1bee0",
"lO c #a1c5ea",
"#X c #a1c6ec",
"gI c #a23f2c",
"ff c #a24230",
"dF c #a2bfe2",
"#z c #a2c3ea",
"bz c #a2c6eb",
"h. c #a3402c",
"gr c #a3402d",
"hj c #a3412d",
"i2 c #a3422f",
"ev c #a3bee4",
"mP c #a3c1e5",
"kM c #a3c5eb",
"lt c #a42500",
"kD c #a43b27",
"hS c #a4402d",
"hq c #a4412d",
"g# c #a4412e",
"fa c #a4c3e9",
"d# c #a4c4e8",
"mk c #a4c8ee",
"et c #a4cbf2",
"nE c #a4ccf4",
"br c #a51c00",
"kk c #a52d0b",
"jM c #a53622",
"fU c #a5402d",
"de c #a5c2e8",
"iu c #a5c6eb",
"bE c #a5cbf4",
"nW c #a5ccf5",
".H c #a5cef6",
"ne c #a5d1fb",
"lB c #a62806",
"d0 c #a63317",
"io c #a63c26",
"fm c #a63e28",
"k# c #a6c3e8",
"mM c #a6c7ed",
"ca c #a6c7ee",
"lf c #a6c8f0",
".v c #a6cbf3",
"jR c #a6ccf1",
"nx c #a6d0f9",
"j6 c #a73c28",
"fE c #a73e28",
"i8 c #a7c6ed",
"fr c #a7c8ee",
"a5 c #a7c8ef",
"a1 c #a7c9ee",
"#y c #a7c9f0",
".g c #a7cdf4",
"#3 c #a7d1fa",
"ie c #a83921",
"iK c #a83a21",
"lY c #a83e24",
"n. c #a8c7eb",
"lj c #a8c9ef",
"jt c #a8c9f0",
"ay c #a8caf1",
"dJ c #a8caf3",
"lN c #a8ccf2",
"me c #a8cef5",
"lJ c #a8cff6",
"d6 c #a8d0f7",
".Z c #a8d3fd",
"eX c #a93617",
"bb c #a93b23",
"d8 c #a9c7ee",
"eR c #a9c9f0",
"cH c #a9caf1",
"cF c #a9cbf1",
"k9 c #aa2300",
"cO c #aa2304",
"aT c #aa3314",
"fx c #aa3c25",
"bM c #aa3c26",
"aG c #aa4027",
"iP c #aaccf3",
"#Y c #aacef5",
"bA c #aacff5",
"l7 c #ab3d22",
"it c #abcdf3",
"i. c #abcdf4",
"mj c #abcef5",
"ny c #abd6fe",
"#s c #accff7",
"bD c #accff8",
"#. c #acd0f8",
"#x c #acd0f9",
"dd c #acd0fa",
"nX c #acd4fe",
"nD c #acd5ff",
"jZ c #ad2901",
"jm c #ad3f2a",
"js c #adcef5",
"eu c #adcef6",
"lM c #adcff6",
"hX c #adcff7",
"c. c #add0f7",
".f c #add2fa",
"kc c #add3fb",
"en c #ae2f17",
"eL c #ae3622",
"ci c #ae371a",
"lZ c #ae3b1d",
"hu c #aecef7",
"dI c #aecff8",
"n# c #aed0f6",
"kL c #aed1f9",
"c# c #aed2fa",
"lK c #aed3fb",
"#2 c #aed3fd",
".w c #aed4fd",
"d7 c #aed5fd",
"nd c #aed9ff",
"eB c #af1f00",
"jA c #af2e08",
"fN c #af381e",
"li c #afd0f8",
"i7 c #afd1f8",
"f# c #afd1f9",
"mi c #afd2f9",
"a4 c #afd2fa",
"cG c #afd3fc",
"jS c #afd4fc",
"nY c #afd7ff",
"dP c #b02e0b",
"bB c #b0d2fa",
"hG c #b0d2fb",
"mh c #b0d3fa",
"ax c #b0d3fb",
"#w c #b0d4fc",
"bC c #b0d5fe",
"ka c #b0d8ff",
"jc c #b1330f",
"h2 c #b1381d",
"lL c #b1d2fb",
"na c #b1d3fa",
"kJ c #b1d3fb",
"dc c #b1d3fc",
"at c #b1d4fb",
"#Z c #b1d4fc",
"#t c #b1d4fd",
"mO c #b1d5fd",
"a3 c #b1d5fe",
"dG c #b1d8fe",
"nC c #b1d9ff",
"nz c #b1daff",
"mf c #b1dbff",
"kC c #b23011",
"gT c #b2361a",
"lh c #b2d4fc",
"#v c #b2d4fd",
"Qt c #b2d5fd",
"aw c #b2d5fe",
".6 c #b2d6fe",
".9 c #b2d6ff",
"kb c #b2d9ff",
".G c #b2daff",
"gC c #b33619",
"l0 c #b33819",
"l6 c #b33a1b",
"lg c #b3d4fc",
".a c #b3d5fd",
"#u c #b3d5fe",
".# c #b3d6fe",
"av c #b3d6ff",
".A c #b3d7ff",
".e c #b3d8ff",
"dH c #b3d9ff",
"mN c #b3daff",
"mg c #b3dbff",
"gj c #b43417",
"f4 c #b43518",
"l5 c #b43918",
".B c #b4d6fe",
".b c #b4d6ff",
".5 c #b4d7ff",
"#1 c #b4d8ff",
".d c #b4d9ff",
"nZ c #b4daff",
"dk c #b53314",
"hM c #b53517",
"l4 c #b53817",
".7 c #b5d6ff",
".C c #b5d7ff",
".c c #b5d8ff",
"#0 c #b5d9ff",
"da c #b5dafe",
"db c #b5daff",
".0 c #b5deff",
"l1 c #b63715",
"l3 c #b63716",
".8 c #b6d8ff",
"a2 c #b6d9ff",
".x c #b6daff",
"nB c #b6dbff",
"nA c #b6ddff",
"iW c #b72d03",
"iA c #b7310b",
"l2 c #b73616",
".z c #b7daff",
".y c #b7dbff",
".E c #b7dcff",
".F c #b7ddff",
"cP c #b82600",
".D c #b8dbff",
".4 c #b8dcff",
"nc c #b8deff",
"lu c #b9320b",
"nb c #b9deff",
"kK c #b9e1ff",
"au c #badeff",
".1 c #bae1ff",
"dy c #bb2d04",
"kl c #bb3008",
"kB c #bc300b",
"c3 c #bd320a",
".3 c #bde0ff",
".2 c #bde1ff",
"lA c #be330c",
"jL c #be3615",
"aH c #be380f",
"bN c #be3917",
"fg c #be3c18",
"kV c #c02500",
"k8 c #c02700",
"b0 c #c02a00",
"eY c #c02d01",
"e5 c #c0381a",
"dZ c #c13710",
"j5 c #c13814",
"jl c #c13816",
"ed c #c22900",
"bc c #c2340d",
"cj c #c23a15",
"cw c #c33710",
"bq c #c42b00",
"eK c #c43311",
"aS c #c53e19",
"cQ c #c63105",
"j0 c #c63205",
"i1 c #c63b18",
"em c #c73913",
"jd c #c8370c",
"hC c #c83916",
"g. c #c83a16",
"hi c #c83a17",
"gq c #c83b17",
"gH c #c83b18",
"jB c #c92c00",
"dl c #c93007",
"kA c #c93309",
"fT c #c93a15",
"if c #ca3910",
"eC c #cb2d00",
"iX c #cc3406",
"in c #cd3711",
"fl c #cd3913",
"aI c #cd3a0e",
"fy c #cd3f18",
"fD c #ce3912",
"kW c #d03101",
"km c #d03304",
"iJ c #d0360d",
"dx c #d0380c",
"lv c #d03c10",
"dQ c #d03d14",
"iB c #d12d00",
"dm c #d13205",
"jk c #d13b13",
"ck c #d13e14",
"bO c #d2380d",
"eZ c #d33506",
"j4 c #d3370b",
"dY c #d3390b",
"e4 c #d4370f",
"k7 c #d53203",
"bZ c #d53404",
"fh c #d53707",
"kz c #d63607",
"jK c #d6380b",
"eJ c #d63c12",
"lz c #d63e12",
"aJ c #d64116",
"aR c #d64318",
"cR c #d73808",
"dn c #d73809",
"fO c #d74218",
"kn c #d93606",
"bd c #da3200",
"i0 c #da3a0e",
"cv c #da3e10",
"h3 c #da4218",
"aQ c #da4318",
"hp c #db390d",
"hh c #db390e",
"aP c #db4419",
"jC c #dc3704",
"f9 c #dc3a0d",
"g9 c #dc3a0f",
"j3 c #dc3d0f",
"j1 c #dc3f11",
"c2 c #dc4013",
"je c #dd3909",
"fS c #dd3a0c",
"gp c #dd3b0f",
"gG c #dd3b10",
"dX c #dd3e10",
"el c #dd4012",
"gU c #dd4318",
"aN c #dd4519",
"ko c #de3b0b",
"e3 c #de3f11",
"cl c #de4014",
"gD c #de4318",
"aK c #de4519",
"aM c #de4619",
"fi c #df3c0b",
"iY c #df4112",
"aO c #df4519",
"bp c #e03907",
"ky c #e03b0c",
"do c #e03c0c",
"jj c #e03e10",
"f5 c #e04418",
"aL c #e0461a",
"iC c #e13703",
"im c #e1390a",
"fk c #e13a0c",
"cS c #e13e0e",
"gk c #e14418",
"fC c #e23a0b",
"hg c #e23f11",
"hR c #e24011",
"iZ c #e24012",
"cu c #e24112",
"kX c #e33b08",
"ee c #e3400f",
"f8 c #e34012",
"e0 c #e34112",
"dw c #e34314",
"hN c #e34418",
"ig c #e43905",
"iI c #e43908",
"fz c #e43e0c",
"fR c #e44012",
"go c #e44113",
"dT c #e44213",
"dR c #e44214",
"j2 c #e44314",
"bP c #e53905",
"dU c #e54213",
"ly c #e54313",
"dW c #e54315",
"jJ c #e63b09",
"jf c #e63e0d",
"kp c #e6400f",
"dp c #e64010",
"eD c #e6410f",
"eI c #e64313",
"bY c #e73d0b",
"fj c #e74111",
"dS c #e74213",
"dV c #e74314",
"ct c #e74414",
"hQ c #e74415",
"k6 c #e83e0b",
"ek c #e8420f",
"lx c #e84413",
"dr c #e84415",
"ds c #e84416",
"cW c #e84515",
"gn c #e84516",
"iH c #e9400f",
"fA c #e9410f",
"kx c #e94110",
"ej c #e94312",
"cm c #e94313",
"dt c #e94415",
"eG c #e94416",
"cV c #e94515",
"cX c #e94516",
"be c #ea3800",
"ji c #ea400f",
"dq c #ea4414",
"h6 c #ea4415",
"g8 c #ea4514",
"cq c #ea4515",
"du c #ea4516",
"fB c #ea4616",
"ih c #eb3e0a",
"jD c #eb4311",
"jg c #eb4413",
"hB c #eb4415",
"co c #eb4514",
"cp c #eb4515",
"ei c #eb4516",
"c1 c #eb4615",
"cY c #eb4616",
"jI c #ec410f",
"kq c #ec4311",
"iD c #ec4312",
"jh c #ec4412",
"ii c #ec4414",
"cn c #ec4513",
"e2 c #ec4514",
"hf c #ec4515",
"cr c #ec4615",
"cZ c #ec4616",
"il c #ec4716",
"cs c #ed4515",
"c0 c #ed4615",
"dv c #ed4616",
"cU c #ed4716",
"eH c #ee4512",
"lw c #ee4516",
"cT c #ee4616",
"eh c #ee4715",
"e1 c #ee4716",
"ik c #ee4717",
"bQ c #ef3e07",
"kY c #ef410c",
"fP c #ef4310",
"fQ c #ef4412",
"ij c #ef4717",
"iG c #ef4817",
"kr c #f04512",
"jH c #f04715",
"h5 c #f14411",
"ku c #f14513",
"kw c #f14613",
"jG c #f14716",
"iF c #f14818",
"h4 c #f24511",
"kv c #f24613",
"bR c #f3420d",
"he c #f34512",
"gW c #f34612",
"ks c #f34613",
"iE c #f34918",
"bf c #f44009",
"bo c #f4410b",
"bX c #f4430e",
"gF c #f44612",
"kt c #f44614",
"eF c #f44917",
"jF c #f44918",
"k5 c #f5440e",
"g7 c #f54611",
"gV c #f54612",
"f7 c #f54714",
"eE c #f54916",
"eg c #f54a17",
"bW c #f6440f",
"bV c #f64410",
"hA c #f64613",
"gE c #f64712",
"gm c #f64714",
"ef c #f64a16",
"jE c #f64a18",
"bU c #f74510",
"hP c #f74713",
"bn c #f8420d",
"bm c #f8430e",
"bS c #f84510",
"k4 c #f8460f",
"f6 c #f84713",
"bT c #f94610",
"k3 c #f94611",
"gl c #f94813",
"bj c #fa440e",
"k2 c #fa4611",
"bk c #fb450e",
"bi c #fb460e",
"k1 c #fb4611",
"hO c #fb4914",
"bl c #fc450e",
"bg c #fc460e",
"kZ c #fc4711",
"bh c #fe470f",
"k0 c #fe4812",
"QtQtQtQtQt.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#QtQtQt.#.#.#.a.a.a.#.#.#.#.#.#.#.#.#.#.#",
"QtQtQtQtQt.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#QtQtQt.#.#.#.a.a.a.#.#.#.#.#.#.#.#.#.#.#",
"QtQtQtQtQt.#.#.b.c.c.c.d.e.f.g.h.i.j.k.l.m.n.o.p.m.q.r.s.t.u.v.w.d.x.y.z.c.A.#.#.#.#.#.#",
".#.#QtQtQtQt.B.C.z.D.E.F.G.H.I.J.K.L.M.N.O.P.Q.R.S.T.U.V.W.X.Y.Z.0.1.2.3.4.5.6.#.#.#QtQt",
".#.#QtQtQtQt.a.7.8.c.5.9#.###a#b#c#d#e#f#g#h#i#j#k#l#m#n#o#p#q#r#s#t.c.z.c.#.6.#.#.#QtQt",
".#.#QtQt#u.#.##v#w#x#y#z#A#B#C#D#E#F#G#H#I#J#K#L#M#N#O#P#Q#R#S#T#U#V#W#X#Y#Z#0.c.#.#QtQt",
"QtQt.#.#.b#1.A#2#3#4#5#6#7#8#9a.a#aaabacadaeafagacahaiajakalamanaoapaqarasatau.D.Bav.#.#",
"QtQt.#aw.a.6axayazaAaBaCaDaEaFaGaHaIaJaKaLaMaNaKaOaPaQaRaSaTaUaVaWaXaYaZa0a1a2#0.a.5.5.#",
"QtQtava3a4#.a5a6a7a8a9b.b#babbbcbdbebfbgbhbibjbkblbmbnbobpbqbrbsbtbubvbwbxbybzbAbB#0.c.#",
".a.#.AbCbDbEbFbGbHbIbJbKbLbMbNbObPbQbRbSbTbUbUbUbSbVbWbXbYbZb0b1b2b3b4b5b6b7b8b9c..D.z.#",
".a.6.9c#cacbcccdcecfcgchcicjckclcmcncococpcpcpcpcpcqcrcsctcucvcwcxcyczcAcBcCcDcEcF.4.zQt",
".a#ZcGcHcIcJcKcLcMcNcOcPcQcRcScmcTcUcqcVcWcXcXcXcXcXcYcZcYc0c1c2c3c4c5c6c7c8c9d.d#dadbQt",
".5dcdddedfdgdhdidjdkdldmdndodpdqcZcYcVdrdrdsdsdsdrdtducYcYdvdvdwdxdydzdAdBdCdDdEdFdGdHQt",
".#dIdJdKdLdMdNdOdPdQdRdSdTdUdVdrdtdtdrdrdrdrdrdrdrdrdtducXdududWdXdYdZd0d1d2d3d4d5d6d7Qt",
"axd8d9e.e#eaebecedeeefegeheicVdrdrdrdrdrdrdrdrdrdrdrdtcXcXduducVejekelemeneoepeqereset#Z",
"euevewexeyezeAeBeCeDeEeFehcYcVdrdrdrdrdrdrcWcWcWcXcXeGdreGcXducqcneHeIeJeKeLeMeNeOePeQ#Z",
"eReSeTeUeVeWeXeYeZe0cTe1cqcWcWdrdrdtcVcVcVcWcWcWcXcXcXdrdrdrdrdrcqe2cte3e4e5e6e7e8e9f.f#",
"fafbfcfdfefffgfhfieIeicYcVcWcWdrdrdtcVcVcVcWcWcWcXcXcXdrdrdrdrdrcpe1c1fjfkflfmfnfofpfqfr",
"fsftfufvfwfxfyfzfAdrdufBcVcVcWdrdrdtcXcXcXcWcWcWcXcXcXdrdrdrdrcVc1e1crfjfCfDfEfnfFfGfHfI",
"fIfJfKfLfMfNfOfPfQduducqcVcVdrdrdrdscXcXcXdrdrdrdrdrdrdrdrdrcVcVc1c0cqfRfSfTfUfVfWfGfXfY",
"fZf0f1f2f3f4f5f6f7eicXcVcVcVdrdrdrdscXcXcXdrdrdrdrdrdrdrdrdrcVcVc1c0cVf8f9g.g#gagbgcgdge",
"fZgfggghgigjgkglgmeigndrcVcVdscXcXgncXcXcXdrdrdrdrdrdrdrdrdrcVcVc1c0cXgogpgqgrgsgtgugvgw",
"gxgygzgAgBgCgDgEgFcqcWdrdrdrdscXcXcXcVcVcVdrdrdrdrdrdrdrdrdrdrdrcqc0cXgogGgHgIgJgKgLgMgN",
"gOgPgQgRgSgTgUgVgWcqcWdrdrdrdscXcXcXcVcVcVdrdrdrdrdrdrdrdrdrdrdrcqc0cXgogGgHgIgXgYgZg0g1",
"g2g3g4g5g6gTgUg7gWg8cWdrdrdrdscXcXcXcVcVcVcXcXcXdrdrdrcVcVcVdrdrcqcseGf8g9gqh.h#hahbhchd",
"g2g3g4g5g6gTgUg7hedqdrdrdrdrdrdrdrdrdrdrcXcXcXcXdrdrdrcVcVcVcVcVcphfdrhghhhihjhkhlhmhnho",
"g2g3g4g5g6gTgUg7hedqdrdrdrdrdrdrdrdrdrdrcXcXcXcXdrdrdrcVcVcVcVcVcphfdrhghpg.hqgahrhshthu",
"hvhwhxhyhzf4gkglhAhBdrdrdrdrdrdrdrdrdrdrcXcXcXcXdrdrdrcVcVcVcVcVcphfdrhghphChqgahDhEhFhG",
"hHhIhJhKhLhMhNhOhPhBhQgncXcXcXdrdrdrdrdrdrdrdrdrcWcWcWdrdrdrcVcVc1crcWhRhphChShThUhVhWhX",
"g2hYhZh0h1h2h3h4h5h6drcXcXcXcXdrdrdrdrdrdrdrdrdrcWcWcWdrdrdrcVcVc1crcWhRhphChShTh7h8h9i.",
"i#iaibicidieifigihiiijikfBgngndrdrdrdrdrdrdrdrdrcWcWcWdrdrdrcXcXile1cYdpiminioipiqirisit",
"iuiviwixiyiziAiBiCiDiEiFfBcWcWdrdrdrdrdrdrdrdrdrcVcVcVcWcWcWcXcXcUiGdviHiIiJiKiLiMiNiOiP",
"itiQiRiSiTiUiViWiXiYcsdvcVcWcWdrdrdrdrdrdrdrdrdrcVcVcVcWcWcWcXcXcYcZgniZi0i1i2i3i4i5i6i.",
"i7i8hoi9j.j#jajbjcjdjejfjgiGijeih6cVcVdrdrdrdrdrcVcVcVcWcVc1cTdvjhjijjjkjljmjnjojpjqjrc.",
"#ZjsjtjujvjwjxjyjzjAjBjCjDjEjFcTcpcqcVcVcWdrdrdrcWcWcVdtcqcTjGjHjIjJjKjLjMjNjOjPbGjQjRax",
".##ZjSayjTjUjVjWjXjYjZj0j1cZcTdtdrcVcVcVcVdrdrdrcWcWcWdrdrdtdqj2j3j4j5j6j7j8j9k.k#kakb.6",
".5.5dHkckdkekfkgkhkikjkkklkmknkokpkqkrksktkukukukvkwkrfQkxkykzkAkBkCkDkEkFkGkHkIkJkK.F.A",
".B#uavkLkMkNkOkPkQkRkSkTkUeBkVkWkXkYbWkZk0k1k2k1kZk3k4k5k6k7k8k9l.l#lalblcldlelf#Z.Edb.6",
".alglhliljlklllmlnlolplqlrlsltlulveldUhflwhBh6hBhfdtlxlylzlAlBlClDlElFlGlHlIlJlK#Z.5.AQt",
".a.a.alLlMlNlOlPlQlRlSlTlUlVlWlXlYlZl0l1l2l3l4l3l1l4l5l6l7l8l9m.m#mambmcmdmemfmg.A.5.AQt",
".#.a.alhmhmimjmkmlmmmnmompmqmrmsmtmumvmwmxmymzmAmwmBmCmDmEmFmGmHmImJmKmLmMlKmN.dav.B.#.#",
".#.#.#.#.#.#QtlhmOf#ljmPmQmRmSmTmUmVmWmXmYmZcAm0mXm1m2m3m4m5m6m7m8m9n.n#naQt.5.5.#.#.#.#",
".#.#.#.#.5a2a2.4nbncmgndnenfngnhninjnknlnmnnnonpnqnonrnsntnunvnwnxnynznAnB#0.C.5.#.#.#.#",
"Qt.6.6.#.#.5#1#0nBnBmNnCnDnEnFnGnHnInJnKnLnMnNnOnPnQnRnSnTnUnVnWnXnYkbnZ.d.5.B.5.#.#.a.a"};

static const char* const image4_data[] = {
"100 44 87 2",
".4 c #000000",
".n c #010101",
"Qt c #020202",
".j c #050505",
".b c #060606",
".f c #070707",
".H c #090d11",
".x c #090d12",
".B c #090e12",
".h c #0a0e12",
".k c #0a0e13",
".g c #0d0d0d",
".# c #121212",
".a c #131313",
"## c #141414",
".e c #192028",
".i c #1a2028",
".z c #1a2128",
".O c #1a2129",
".t c #1a212a",
".L c #1b2128",
".R c #1b2129",
"#m c #1b212a",
".u c #1b2228",
".T c #1b2229",
".c c #1b222a",
"#k c #1c222a",
".M c #1c232a",
".o c #1c232b",
".m c #35414f",
".D c #35424f",
".Q c #36424f",
".S c #364250",
".1 c #364350",
"#j c #374350",
"#. c #374351",
"#b c #374451",
"#u c #4b4b4b",
"#n c #4c4c4c",
".6 c #4f4f4f",
".N c #505050",
".p c #515151",
".F c #525252",
"#q c #898989",
"#s c #8a8a8a",
"#p c #8c8c8c",
".J c #909090",
".E c #919191",
".s c #929292",
".v c #939393",
".K c #949494",
"#h c #999999",
"#c c #a0a0a0",
".5 c #a9cbf1",
".C c #aacbf1",
".I c #aacbf2",
".l c #aaccf2",
".G c #b2d5fd",
".d c #b3d5fd",
"#l c #b3d5fe",
".y c #b3d6fe",
"#f c #b5b5b5",
"#e c #b6b6b6",
"#r c #c2c2c2",
"#t c #c3c3c3",
".8 c #c9c9c9",
".A c #cacaca",
".w c #cbcbcb",
".P c #cccccc",
".9 c #cdcdcd",
".X c #cecece",
"#d c #d7d7d7",
".V c #d8d8d8",
"#g c #e4e4e4",
".2 c #e5e5e5",
".0 c #e6e6e6",
".3 c #e7e7e7",
"#a c #e8e8e8",
"#i c #f1f1f1",
"#o c #f2f2f2",
".Y c #f3f3f3",
".7 c #f4f4f4",
".U c #f9f9f9",
".Z c #fafafa",
".W c #fdfdfd",
".r c #fefefe",
".q c #ffffff",
"Qt.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.a.#.#.#.b.c.d.e.f.#.#.#.#.#.#.a.a.#.#.#.#.#.#.#.#.#.#.#.#.#.g.h.d.i.j.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.g.k.l.m.n.#.#.#.#.#.a.#.#.#.#.#.#.#.#.#.#.#.#.#.#.j.o",
".p.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.r.s.t.d.u.v.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.w.x.y.z.s.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.A.B.C.D.p.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.r.E.c",
".F.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.r.E.t.G.z.v.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.A.H.G.z.v.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.A.x.I.m.F.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.r.J.c",
".F.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.r.E.t.G.z.K.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.w.B.y.L.s.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.A.H.C.D.p.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.r.J.c",
".p.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.r.E.t.G.z.v.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.w.x.y.z.v.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.w.x.l.m.p.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.r.J.M",
".p.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.r.E.t.d.z.K.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.A.x.y.z.v.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.w.x.l.m.p.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.r.E.c",
".p.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.r.E.t.G.z.v.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.w.B.y.z.v.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.A.x.C.m.p.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.r.J.c",
".N.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.s.O.y.O.s.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.w.x.y.O.s.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.P.x.l.Q.N.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.E.R",
".N.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.v.O.y.O.s.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.w.x.G.O.E.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.w.x.C.S.p.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.E.O",
".N.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.v.O.y.R.s.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.P.x.y.O.s.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.w.x.C.D.N.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.E.O",
".N.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.s.O.d.O.s.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.w.x.y.O.E.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.P.x.I.D.N.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.E.O",
".N.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.s.T.G.O.s.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.P.B.G.O.E.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.w.x.I.Q.N.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.s.O",
".p.q.q.q.q.U.P.w.w.w.w.w.w.P.P.w.w.V.q.q.q.v.O.G.O.s.q.q.q.q.q.W.X.P.w.w.w.w.w.w.P.P.Y.q.q.q.q.w.H.G.O.s.q.q.q.q.Z.w.w.w.w.w.w.w.w.P.w.P.0.q.q.q.q.w.x.I.1.N.q.q.q.q.U.P.w.P.w.P.w.w.P.P.P.P.2.q.q.q.E.O",
".N.q.q.q.q.3.4.4.4.4.4.4.4.4.4.4.4.N.q.q.q.s.O.G.O.s.q.q.q.q.q.Y.#.4.4.4.4.4.4.4.4.4.w.q.q.q.q.w.x.y.O.E.q.q.q.q.0.4.4.4.4.4.4.4.4.4.4.4.s.q.q.q.q.P.x.C.D.N.q.q.q.q.0.4.4.4.4.4.4.4.4.4.4.4.E.q.q.q.E.O",
".N.q.q.q.q.0.4.4.4.4.4.4.4.4.4.4.4.N.q.q.q.s.O.G.O.s.q.q.q.q.q.Y.#.4.4.4.4.4.4.4.4.4.w.q.q.q.q.w.B.y.O.s.q.q.q.q.0.4.4.4.4.4.4.4.4.4.4.4.s.q.q.q.q.w.x.C.Q.p.q.q.q.q.0.4.4.4.4.4.4.4.4.4.4.4.s.q.q.q.E.O",
".p.q.q.q.q.0.4.4.4.4.4.4.4.4.4.4.4.N.q.q.q.v.R.G.O.s.q.q.q.q.q.Y.#.4.4.4.4.4.4.4.4.4.P.q.q.q.q.w.x.d.R.s.q.q.q.q.0.4.4.4.4.4.4.4.4.4.4.4.s.q.q.q.q.w.x.I.S.N.q.q.q.q.0.4.4.4.4.4.4.4.4.4.4.4.s.q.q.q.s.O",
".N.q.q.q.q.0.4.4.4.4.4.4.4.4.4.4.4.N.q.q.q.s.O.d.O.v.q.q.q.q.q.Y.#.4.4.4.4.4.4.4.4.4.w.q.q.q.q.w.x.G.O.E.q.q.q.q.0.4.4.4.4.4.4.4.4.4.4.4.s.q.q.q.q.w.x.l.1.p.q.q.q.q.0.4.4.4.4.4.4.4.4.4.4.4.E.q.q.q.s.O",
".N.q.q.q.q.0.4.4.4.4.4.4.4.4.4.4.4.N.q.q.q.v.O.y.O.s.q.q.q.q.q.Y.#.4.4.4.4.4.4.4.4.4.P.q.q.q.q.w.x.y.O.E.q.q.q.q.0.4.4.4.4.4.4.4.4.4.4.4.v.q.q.q.q.P.x.I.S.N.q.q.q.q.0.4.4.4.4.4.4.4.4.4.4.4.E.q.q.q.E.O",
".N.q.q.q.q.0.4.4.4.4.4.4.4.4.4.4.4.N.q.q.q.s.O.y.O.s.q.q.q.q.q.Y.#.4.4.4.4.4.4.4.4.4.w.q.q.q.q.w.x.G.O.E.q.q.q.q.0.4.4.4.4.4.4.4.4.4.4.4.s.q.q.q.q.P.x.l.S.N.q.q.q.q.0.4.4.4.4.4.4.4.4.4.4.4.s.q.q.q.E.O",
".N.q.q.q.q.3.4.4.4.4.4.4.4.4.4.4.4.N.q.q.q.v.O.y.O.s.q.q.q.q.q.Y.#.4.4.4.4.4.4.4.4.4.P.q.q.q.q.w.x.G.O.s.q.q.q.q.0.4.4.4.4.4.4.4.4.4.4.4.s.q.q.q.q.P.x.I.Q.N.q.q.q.q.0.4.4.4.4.4.4.4.4.4.4.4.E.q.q.q.s.O",
".N.q.q.q.q.0.4.4.4.4.4.4.4.4.4.4.4.N.q.q.q.s.O.d.O.s.q.q.q.q.q.Y.#.4.4.4.4.4.4.4.4.4.w.q.q.q.q.w.x.y.O.E.q.q.q.q.0.4.4.4.4.4.4.4.4.4.4.4.s.q.q.q.q.P.H.C.S.N.q.q.q.q.0.4.4.4.4.4.4.4.4.4.4.4.E.q.q.q.E.O",
".N.q.q.q.q.0.4.4.4.4.4.4.4.4.4.4.4.N.q.q.q.v.R.G.O.s.q.q.q.q.q.Y.#.4.4.4.4.4.4.4.4.4.w.q.q.q.q.P.x.G.O.E.q.q.q.q.0.4.4.4.4.4.4.4.4.4.4.4.v.q.q.q.q.w.B.C.Q.N.q.q.q.q.0.4.4.4.4.4.4.4.4.4.4.4.E.q.q.q.s.O",
".N.q.q.q.q.0.4.4.4.4.4.4.4.4.4.4.4.N.q.q.q.v.O.y.O.s.q.q.q.q.q.Y.#.4.4.4.4.4.4.4.4.4.P.q.q.q.q.P.x.y.O.E.q.q.q.q.0.4.4.4.4.4.4.4.4.4.4.4.s.q.q.q.q.P.x.I.1.N.q.q.q.q.0.4.4.4.4.4.4.4.4.4.4.4.E.q.q.q.E.O",
".N.q.q.q.q.0.4.4.4.4.4.4.4.4.4.4.4.N.q.q.q.s.O.d.O.s.q.q.q.q.q.Y.#.4.4.4.4.4.4.4.4.4.w.q.q.q.q.w.x.G.O.s.q.q.q.q.0.4.4.4.4.4.4.4.4.4.4.4.v.q.q.q.q.w.x.I.Q.N.q.q.q.q.0.4.4.4.4.4.4.4.4.4.4.4.E.q.q.q.E.O",
".N.q.q.q.q.0.4.4.4.4.4.4.4.4.4.4.4.p.q.q.q.v.O.y.O.v.q.q.q.q.q.Y.#.4.4.4.4.4.4.4.4.4.w.q.q.q.q.P.H.d.O.E.q.q.q.q.0.4.4.4.4.4.4.4.4.4.4.4.v.q.q.q.q.P.H.C.S.N.q.q.q.q.0.4.4.4.4.4.4.4.4.4.4.4.s.q.q.q.E.O",
".N.q.q.q.q.0.4.4.4.4.4.4.4.4.4.4.4.p.q.q.q.s.O.d.O.s.q.q.q.q.q.Y.#.4.4.4.4.4.4.4.4.4.w.q.q.q.q.w.x.G.O.s.q.q.q.q.0.4.4.4.4.4.4.4.4.4.4.4.s.q.q.q.q.P.x.I.S.N.q.q.q.q.0.4.4.4.4.4.4.4.4.4.4.4.E.q.q.q.s.O",
".N.q.q.q.q.0.4.4.4.4.4.4.4.4.4.4.4.N.q.q.q.v.O.y.O.s.q.q.q.q.q.Y.#.4.4.4.4.4.4.4.4.4.P.q.q.q.q.w.H.G.O.E.q.q.q.q.0.4.4.4.4.4.4.4.4.4.4.4.s.q.q.q.q.P.x.5.S.N.q.q.q.q.0.4.4.4.4.4.4.4.4.4.4.4.E.q.q.q.E.O",
".p.r.q.q.q.3.n.4.4.4.4.4.4.4.4.4.4.6.r.q.q.v.z.y.t.E.r.q.q.q.q.7.a.4.4.4.4.4.4.4.4.4.8.q.q.q.q.9.x.d.t.J.r.q.q.q.3.n.4.4.4.4.4.4.4.4.4.4.E.r.q.q.q.P.B.I#..N.r.q.q.q.3.n.4.4.4.4.4.4.4.4.4.4.J.r.q.q.v.c",
".p.r.q.q.q.3.n.4.4.4.4.4.4.4.4.4.4.6.r.q.q.K.z.G.t.s.r.q.q.q.q.7##.4.4.4.4.4.4.4.4.4.8.q.q.q.q.P.x.y.t.J.r.q.q.q.3.n.4.4.4.4.4.4.4.4.4.4.E.r.q.q.q.P.x.l#..6.r.q.q.q#a.n.4.4.4.4.4.4.4.4.4.4.E.r.q.q.s.c",
".N.r.q.q.q#a.n.4.4.4.4.4.4.4.4.4.4.6.r.q.q.v.z.y.t.s.r.q.q.q.q.7.a.4.4.4.4.4.4.4.4.4.A.q.q.q.q.P.x.G.t.E.r.q.q.q.3.n.4.4.4.4.4.4.4.4.4.4.s.r.q.q.q.9.x.C#b.6.r.q.q.q.3.n.4.4.4.4.4.4.4.4.4.4.E.r.q.q.s.c",
".N.r.q.q.q.3.n.4.4.4.4.4.4.4.4.4.4.6.r.q.q.K.z.G.t.E.r.q.q.q.q.7.a.4.4.4.4.4.4.4.4.4.8.q.q.q.q.9.x.d.t.E.r.q.q.q.3.n.4.4.4.4.4.4.4.4.4.4.E.r.q.q.q.P.x.I#..6.r.q.q.q.3.n.4.4.4.4.4.4.4.4.4.4.J.r.q.q.v.c",
".N.r.q.q.q.3.n.4.4.4.4.4.4.4.4.4.4.6.r.q.q.v.z.G.t.E.r.q.q.q.q.7.a.4.4.4.4.4.4.4.4.4.8.q.q.q.q.9.x.G.t.J.r.q.q.q.3.n.4.4.4.4.4.4.4.4.4.4.E.r.q.q.q.P.x.l#..6.r.q.q.q.3.n.4.4.4.4.4.4.4.4.4.4.J.r.q.q.s.c",
".p.r.q.q.q.3.n.4.4.4.4.4.4.4.4.4#c#d.q.q.q.K.z.G.t.E.r.q.q.q.q.7.a.4.4.4.4.4.4.4.4#e.Y.q.q.q.q.9.H.G.t.J.r.q.q.q.3.n.4.4.4.4.4.4.4.4.4#f.0.q.q.q.q.9.x.C#..6.r.q.q.q.3.n.4.4.4.4.4.4.4.4.4#c.2.q.q.q.s.c",
".N.r.q.q.q.7.v.E.E.s.s.E.E.s.s.s#g.q.q.q.q.v.z.y.t.s.r.q.q.q.q.Z#h.E.s.E.s.s.E.s.E#i.q.q.q.q.q.P.x.d.t.E.r.q.q.q.7.s.s.s.E.s.E.E.s.E.E#i.q.q.q.q.q.9.x.l#..6.r.q.q.q.Y.v.E.E.E.s.E.s.s.E.E#g.q.q.q.q.s.c",
".N.r.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.v.z.d.t.E.r.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.9.x.G.t.J.r.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.9.x.C.1.6.r.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.v.c",
".N.r.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.v.z.y.t.E.r.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.9.H.G.t.E.r.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.P.x.C#j.6.r.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.s.c",
".N.r.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.v.u.G.c.E.r.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.P.x.y.t.E.r.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.P.x.C#..N.r.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.s#k",
".N.r.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.v.z.y.t.E.r.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.P.x#l.t.J.r.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.9.x.I.1.6.r.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.s.c",
".N.r.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.v.L.G#m.E.r.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.9.x.y.t.E.r.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.9.B.I.1.6.r.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.v.c",
".p.r.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.v.z.G.t.s.r.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.P.B.G.t.J.r.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.9.x.l#..N.r.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.v.c",
".N.r.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.v.z.y.t.E.r.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.P.H.G.t.J.r.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.P.x.I.1.6.r.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.s.c",
".N.r.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.v.z.d.t.E.r.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.9.x.y.t.E.r.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.P.H.C#..6.r.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.v.c",
".p.r.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.v.L.y.t.s.r.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.P.x.y.t.J.r.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.P.x.C#..6.r.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.q.v.M",
"#n#o.Y.Y.Y.Y.Y.Y.Y.Y.Y.Y.Y.Y.Y.Y.Y.Y.Y.Y.7#p.z.y.t#q#o.Y.Y.Y.Y.Y.Y.Y.Y.Y.Y.Y.Y.Y.Y.Y.Y.Y.Y.Y.Y#r.x.G.t#s#o.Y.Y.Y.Y.Y.Y.Y.Y.Y.Y.Y.Y.Y.Y.Y.Y.Y.Y.Y.Y#t.x.I#.#u#o.Y.Y.Y.Y.Y.Y.Y.Y.Y.Y.Y.Y.Y.Y.Y.Y.Y.Y.7#p.c"};

static const char* const image5_data[] = {
"36 41 301 2",
"#A c #69cd3c",
"al c #6fab7b",
"#U c #72ce5e",
"au c #74d44b",
"#K c #74e539",
"#p c #75e92f",
"a4 c #76e63c",
"#J c #77ac98",
".4 c #77ec30",
"aV c #77ed38",
"#o c #79dc64",
"#2 c #79ea3e",
".5 c #79ec37",
"ad c #79ee35",
"b5 c #79ef37",
"cc c #79ef39",
"bW c #79ef3a",
".Y c #79f039",
"#d c #79f131",
"#u c #79f132",
"#P c #79f133",
"bM c #79f134",
"#3 c #79f136",
"#e c #79f230",
"#c c #7aed41",
"bG c #7aed44",
"bk c #7aed46",
"bR c #7aed48",
".q c #7aed49",
"#Q c #7aee41",
"aD c #7aef37",
"ay c #7aef3a",
"bi c #7aef3c",
"bs c #7aef3d",
"bw c #7aef3e",
"bx c #7aef3f",
"#v c #7aef41",
".S c #7aef42",
"#L c #7aef43",
"cd c #7af037",
"ci c #7af038",
"aP c #7af03a",
".p c #7af03c",
"cB c #7af132",
"ax c #7af133",
".D c #7af134",
".y c #7af135",
"a. c #7af136",
"#k c #7af137",
".F c #7af138",
"a9 c #7af139",
"bH c #7af13b",
".9 c #7af231",
"aq c #7af232",
".o c #7af234",
".x c #7af330",
".w c #7af331",
"#E c #7af332",
".T c #7be955",
"a1 c #7beb4f",
"bb c #7bec4d",
"#j c #7bec4e",
"cp c #7bec4f",
".O c #7bec50",
"cF c #7bec51",
"#q c #7bed46",
"ca c #7bed49",
"aU c #7bed4e",
"#D c #7bee47",
"#8 c #7bef40",
"b2 c #7bef45",
"am c #7bf133",
"a8 c #7bf13a",
".3 c #7bf232",
".E c #7bf331",
"aO c #7bf332",
"a0 c #7bf335",
"#9 c #7bf431",
"bg c #7cea55",
"ch c #7cea56",
"cr c #7ceb55",
"bo c #7ceb56",
"cL c #7cec51",
"bp c #7cec52",
"cm c #7cec53",
"ah c #7ced4f",
"aJ c #7de657",
"bA c #7de861",
".v c #7de95f",
"bZ c #7de962",
"ae c #7dea57",
".N c #7dea59",
"ap c #7dea5e",
".2 c #7dea60",
"#V c #7deb54",
"aE c #7deb56",
"#. c #7deb59",
"b0 c #7deb5a",
"#Z c #7dec51",
".U c #7edb6b",
"ar c #7ede70",
"aI c #7fe769",
"bX c #7fe76e",
"bl c #7fe866",
"#F c #7fe966",
"aN c #7fea50",
"b8 c #80e869",
"bc c #81e572",
".i c #81e676",
"aW c #82c793",
".8 c #82e57a",
".R c #82e67b",
"c# c #82e777",
".z c #83e480",
"#B c #83e579",
"#Y c #83e57c",
"cw c #83e57d",
"cs c #83e57e",
"an c #83e679",
"bN c #83e67c",
"cx c #84e67e",
"#l c #86e28b",
"bC c #86e488",
"cl c #86e489",
"aZ c #86e581",
"b6 c #86e586",
"b. c #87e38b",
"cj c #87e38e",
"bQ c #87e48a",
".n c #87e48d",
"cK c #88e291",
"ce c #88e38e",
".r c #88e48d",
"aC c #89ca98",
"az c #89cea2",
".Z c #8ae196",
"#t c #8ae199",
".G c #8ae392",
"br c #8ae395",
"a# c #8be390",
"ac c #8cb5bc",
"ag c #8ce19b",
"bI c #8ce19d",
"cC c #8ce19e",
"bf c #8ce29b",
"#1 c #8dbcb7",
"#f c #8ddf98",
"at c #8eabc7",
"#4 c #8ee0a1",
"bj c #8ee0a2",
"cA c #8ee0a5",
".X c #8ee1a2",
".j c #8ee1a3",
"#b c #8fe0a3",
"cI c #8fe1a3",
"bt c #8fe1a7",
"by c #90dfaa",
"cJ c #90e0aa",
".6 c #91d7a0",
"bT c #91dfac",
"aQ c #91dfad",
"a5 c #93c4c2",
"b4 c #93dfb2",
"aw c #94deb5",
".C c #94dfb3",
"cv c #94dfb5",
"#w c #94e0b0",
"#z c #95cbc4",
"b1 c #95dcb9",
"#7 c #95deb7",
".I c #95dfb6",
"a7 c #96ddba",
"bS c #96deb9",
".k c #96deba",
"aM c #97cec4",
".h c #97ddbb",
"bF c #97debc",
"b3 c #98dcbf",
"bv c #98dcc0",
"bL c #98ddbe",
"bU c #98debd",
"#T c #99c2d4",
"#R c #99dcc1",
"#O c #9adcc5",
"cb c #9addc4",
"ak c #9bbadc",
"co c #9bdbc7",
"#i c #9bdcc7",
"bD c #9bddc5",
"cn c #9bddc6",
"cH c #9cdcc9",
"bV c #9ddccd",
"ai c #9ed4d1",
"ba c #9edbcd",
"## c #9edbce",
"aT c #9edbcf",
"#r c #9fdbce",
"cq c #9fdbcf",
"a2 c #a0dad1",
"cP c #a1d9d9",
"#M c #a1dad4",
"#C c #a1dad6",
"bh c #a1dbd6",
".1 c #a2d8db",
"#n c #a2d9d7",
".u c #a2dbd5",
"#0 c #a3d2e2",
"bn c #a3d9da",
"ao c #a3d9db",
".M c #a3d9dc",
"cE c #a3dadb",
".V c #a4cce0",
".s c #a4dada",
"bz c #a5d8e0",
"b7 c #a5d8e1",
"bY c #a5d9de",
".P c #a5d9df",
"aH c #a5d9e1",
"aK c #a6cde9",
"cQ c #a6d7e3",
"c. c #a6d8e4",
".H c #a6d9e0",
"as c #a7caec",
".Q c #a7d8e4",
"#X c #a7d8e5",
"bq c #a7d9e3",
"b9 c #a7d9e4",
"bP c #a8d7e6",
".7 c #a8d8e6",
"cD c #a8d8e7",
"cy c #a8d9e4",
"#I c #a9ccf1",
"aa c #a9cdf0",
"aY c #a9d2ea",
"cM c #a9d8e7",
"bB c #a9d8e9",
"ck c #a9d8ea",
"aB c #aacdf0",
"a3 c #aad7ed",
".f c #aad8e7",
".m c #aad8ea",
"cO c #aad8eb",
"bO c #abd7eb",
"#s c #abd7ed",
"af c #abd7ee",
"#a c #acd6ef",
"cz c #acd6f1",
".W c #acd7f0",
"#W c #acd8ee",
"aF c #add6ef",
".K c #add6f0",
"ct c #add6f2",
"cN c #add6f4",
".J c #add7ee",
"be c #add7f1",
".A c #add7f2",
"aL c #aed0f6",
"bd c #aed3f4",
"b# c #aed6f3",
".g c #aed6f4",
"#6 c #aed6f5",
"#G c #aed7f3",
"#m c #aed7f5",
"aA c #afd2f9",
"#x c #afd2fa",
"bK c #afd5f6",
".e c #afd6f6",
"av c #afd6f7",
"cf c #afd7f4",
"bE c #afd7f6",
"#S c #b0d3fa",
"a6 c #b0d4f7",
"cg c #b0d5f9",
"cG c #b0d5fa",
"bJ c #b0d6f7",
"bu c #b0d6f8",
"bm c #b0d6f9",
"aX c #b1d2fa",
".L c #b1d3fa",
"#g c #b1d4fa",
"aj c #b1d4fb",
"#y c #b1d4fc",
".t c #b1d5fb",
"cu c #b1d5fc",
"#h c #b1d6f8",
"#5 c #b1d6fa",
"#N c #b1d6fb",
"#H c #b2d4fb",
"Qt c #b2d4fc",
".B c #b2d5fc",
".a c #b2d5fd",
".0 c #b2d6fa",
".l c #b2d6fb",
"aS c #b2d6fc",
"aR c #b2d6fd",
"aG c #b3d4fb",
"ab c #b3d4fc",
".c c #b3d5fd",
".# c #b3d5fe",
".b c #b3d6fe",
".d c #b4d4fa",
"Qt.#.a.b.cQt.d.a.a.a.a.#.c.b.a.b.a.a.b.b.a.a.b.c.a.b.b.c.b.a.a.a.a.b.a.a",
".a.a.c.a.b.c.a.a.a.a.b.c.a.a.b.b.b.a.e.f.c.b.b.b.b.a.a.c.a.a.c.a.a.b.a.a",
".a.b.b.a.b.a.a.a.b.a.b.b.b.a.a.a.a.g.h.i.j.k.l.b.a.a.a.a.a.a.a.b.a.a.bQt",
".b.a.c.b.b.#.a.b.c.a.b.a.a.b.a.c.a.m.n.o.p.q.r.s.c.b.b.a.b.a.a.a.c.a.a.a",
".a.b.c.a.a.a.c.a.b.a.b.a.b.a.b.a.t.u.v.w.x.y.z.A.a.a.a.c.a.a.c.a.a.c.B.b",
".a.a.b.a.a.b.b.a.b.b.a.c.c.b.a.b.g.C.D.E.w.F.G.b.b.b.a.t.H.I.J.K.L.a.b.a",
".a.b.a.a.a.a.c.b.bQt.B.a.a.a.b.a.M.N.x.E.w.O.P.c.b.a.a.Q.R.S.T.U.V.a.a.a",
".b.c.a.a.a.b.c.a.a.c.a.a.c.b.a.W.X.y.w.w.Y.Z.0.a.a.b.b.1.2.3.4.5.6.L.B.b",
".a.b.c.c.a.c.b.a.a.a.a.c.a.b.b.7.8.w.w.9#.##.a.b.a.a#a#b#c#d#e.D#f#g.c.c",
".c.b.c.b.b.a.a.a.a.#.a.#.a.b#h#i#j.x.w#k#l#m.a.a.c.a#n#o#p.E.E#q#r.b.c.b",
".b.c.c.c.c.b.a.b.a.a.b.b.b.a#s#t#u.E.w#v#w#x#y.b.b.t#z#A#d.w.E#B.e.b.a.c",
".a.a.a.b.a.b.b.a.a.c.a.b.b.a#C#D.E.x#E#F#G.a.b.a#H#I#J#K.w.w#L#M.c.b.b.c",
".c.b.a.a.a.a.c.a.b.a.c.a.b#N#O#P.E.w#Q#R#S.a.a.bQt#T#U.9.x.w#V#W.b.b.a.b",
".b.c.b.a.b.c.a.b.a.a.b.a.b#X#Y.w.x.x#Z#0Qt.c.a.a#y#1#2.w.E#3#4#5.b.a.a.a",
".a.b.b.a.b.a.a.a.a.a.a.b#6#7#8#9.wa.a#aa.a.b.b.aabacad.w.xae.H.a.c.a.b.b",
".a.a.a.a.b.a.b.b.a.b.a.aafag#E.w.wahaiaj.a.#.a#Sakalam#9.wan#5.b.a.c.a.b",
".c.a.a.b.a.c.c.c.c.a.b.baoap.E.xaqaras.b.a.a.c.batau.w.x#k.h.b.b.a.a.a.b",
".c.c.a.a.b.a.a.b.a.a.aavawax.E.EayazaA.a.a.bQtaBaCaD.x.xaEaF.a.a.b.a.a.a",
"aG.c.c.c.a.a.c.a.b.c.baHaI.E.E.waJaK.a.b.b.aaLaMaNaO.EaPaQaR.a.b.a.a.b.c",
".c.B.B.a.a.a.b.a.a.caSaTaU.w.waVaWaX.a.a.aQtaYaZa0.E.wa1a2.b.b.a.a.a.c.c",
"Qt.a.a.#.a.a.c.b.a.ba3#tax.w.xa4a5.b.c.c.aa6a7a8.w#Ea9b.b#.c.b.a.b.a.a.c",
".a.b.a.a.caj.b.b.a#5babb.w.xaqbcbd.b.c.b.bbebf.E.x.Ebgbh.b.b.b.b.a.c.c.a",
".a.a.a.b.a#y.b.a.cafagaO.E.Ebibj#5.c.c.b.a#Cbk.x.x.Eblbm.a.a.b.a.a.a.a.b",
".a.b.b.a.c.b.a.b.abnbo.w.w#9bpbq.b.a.c.ba3br.9.E.Ebsbt.c.b.b.b.a.c.a.a.c",
".a.a.a.b.c.a.c.bbubvbw.w.wbxby#N.a.b.a.abzbl.w.x#EbAbn.c.a.a.a.a.a.b.a.b",
".c.c.a.c.a.c.a.abBbC.9.x.w#qbD.a.a.a.abEbFbG.w.EbHbIbJ.b.b.b.b.a.a.a.#.c",
".c.b.a.a.a.a.cbKbLbx.w.EbMbNbO.b.a.a.bbPbQ.9.x.EbRbS.a.b.a.b.a.a.a.b.c.a",
".a.a.a.a.b.b.a.gbT.w.w.xbkbU.B.a.a.b.tbVbW#9.w#EbXaf.a.b.a.a.a.b.a.b.b.b",
".b.a.b.c.b.c.abYbZ.x#9.wb0.A.b.a.c.b#hb1.E.x.Eb2b3.c.a.a.a.a.b.b.a.b.a.a",
".c.a.b.a.a.b.gb4b5.w.x.yb6.c.a.b.a.ab7b8.x.x.xbpb9.c.b.b.b.a.a.a.a.b.a.b",
".a.b.a.c.a.ac.c#.w.x.xcacb.a.c.b.c.Abycc.x.wcdcecf.a.a.b.c.c.b.a.b.a.c.c",
".b.b.c.c.acgaTch.E.Ecicj#G.c.b.a.ackcl.9.E.wcmcn.t.a.a.c.c.b.b.a.a.b.a.a",
".b.c.c.c.abmcocp#9.w.p.j.a.a.c.a.tcqcr.E.x.ocsct.a.b.a.a.b.c.b.c.c.a.a.c",
".c.b.#.c.ccu.WaTcvcwcxcy.a.a.c.bczcAcB#9.waycC.b.b.a.b.a.c.b.a.a.a.a.a.b",
".a.b.c.a.a.b.ccubKcD.Q.B.a.a.a.acEcr.w.E.EcFbq.b.a.b.b.a.a.a.a.a.#.a.b.c",
".a.a.a.a.a.b.a.a.c.a.b.b.c.b.acGcHbi.w.wbscI.t.a.b.a.b.a.b.a.b.a.a.c.a.b",
".a.b.a.c.a.a.c.a.a.a.a.a.b.b.cctcJ#k.E#9#Dcn.a.a.a.c.a.a.a.a.a.a.b.b.a.b",
".b.c.c.a.a.a.a.b.b.b.a.b.a.a.c.t.HaQcKcLbNcM.c.a.b.b.b.a.b.a.b.b.a.c.b.b",
".b.b.c.a.a.b.b.b.c.a.b.a.a.b.a.b.tcNcOcPcQ.l.c.c.a.a.b.b.c.a.a.a.b.c.a.a",
".a.b.a.a.b.a.a.b.b.a.b.a.a.a.a.c.b.a.c.c.c.c.b.c.c.c.b.a.b.c.c.a.c.b.a.a",
".a.a.b.b.b.a.a.b.b.b.a.a.a.b.b.a.b.a.a.a.b.a.b.b.a.a.b.c.b.c.b.b.a.a.a.a"};

static const char* const image6_data[] = {
"18 23 125 2",
"#Y c #79ef3c",
".1 c #79f037",
".H c #79f132",
".7 c #79f133",
".Q c #7aec4a",
"#x c #7aed46",
"#e c #7aee3e",
"#K c #7aee42",
".M c #7aee43",
"#C c #7aef3f",
"#B c #7af038",
".z c #7af039",
"#a c #7af03a",
"#f c #7af03b",
"#n c #7af134",
"#J c #7af135",
".t c #7af137",
".G c #7af231",
"#T c #7af232",
".B c #7af234",
".u c #7af330",
".A c #7af331",
".v c #7af332",
".W c #7bec4f",
"#t c #7bec51",
".0 c #7bed4b",
"#O c #7bed4c",
"#o c #7bee48",
".R c #7bf331",
".n c #7bf332",
"## c #7bf431",
"#j c #7ce95d",
".o c #7ceb58",
".S c #7de960",
"#G c #7dea5d",
"#U c #7dea5e",
".8 c #7deb5b",
"#S c #7dec56",
".I c #7ee866",
".m c #7ee961",
".6 c #7fe967",
"#F c #80e76f",
"#X c #80e86b",
"#P c #82e773",
".C c #83e57c",
"#u c #83e57d",
".s c #84e482",
"#w c #84e580",
".F c #84e67e",
"#4 c #85e585",
"#I c #86e488",
"#5 c #87e38c",
".2 c #88e391",
".V c #89e293",
".g c #8ae397",
"#Z c #8be19b",
"#g c #8be298",
"#b c #8ce29c",
".L c #8de0a1",
"#D c #8de1a0",
"#. c #8ee0a2",
"#d c #8ee1a2",
"#3 c #92deb2",
".N c #92dfaf",
".p c #92e0ae",
"#m c #93dfb1",
".X c #93e0b1",
".y c #95ddb8",
"#A c #95deb6",
"#L c #95deb7",
".f c #97dcbd",
"#y c #97debb",
"#s c #98ddbe",
".h c #98ddbf",
"#N c #99ddc1",
"#p c #9adbc6",
".Z c #9adcc4",
".J c #9ddbcc",
".P c #9fdbd2",
"#k c #a0dbd3",
".c c #a1c0e4",
"#R c #a1d9d6",
".l c #a1dad5",
".w c #a1dad6",
"#i c #a1dbd5",
"#V c #a2d9d7",
"#2 c #a2d9d9",
".q c #a2d9da",
".d c #a3c2e7",
".5 c #a3dadb",
"#W c #a4d9db",
".E c #a6d9e2",
"#6 c #a7d7e4",
".T c #a7d8e5",
"#H c #a8d7e8",
".r c #a8d8e6",
".b c #abccf2",
".U c #abd7ed",
".K c #abd7ee",
"#h c #abd8ec",
"#E c #acd6f1",
"#c c #acd7ef",
".9 c #add7f1",
".e c #add7f2",
"#l c #aed6f5",
".3 c #aed7f2",
".i c #aed7f4",
"#r c #afd6f5",
".x c #afd6f6",
"#1 c #afd6f7",
".Y c #afd6f8",
"#v c #afd7f4",
"#q c #afd7f6",
"#M c #b0d6f8",
".k c #b1d5fb",
".D c #b1d6f9",
"#0 c #b1d6fa",
".a c #b2d4fd",
".4 c #b2d5fc",
"Qt c #b2d5fd",
".O c #b2d6fa",
"#z c #b2d6fd",
".j c #b3d5fd",
"#Q c #b3d5fe",
".# c #b3d6fe",
"Qt.#.a.b.c.d.bQtQtQtQtQt.e.f.g.h.i.j",
".#.#Qt.#.#.#Qt.#.#Qt.#.k.l.m.n.o.p.q",
".j.j.#QtQtQtQt.#Qt.#Qt.r.s.t.u.v.m.w",
".#Qt.jQt.jQtQt.#.#.j.x.y.z.A.A.B.C.D",
"QtQt.#.#QtQt.#.jQtQt.E.F.G.A.H.I.J.j",
".#.#.#QtQtQt.#Qt.#.K.L.M.u.A.M.N.OQt",
"Qt.#Qt.#QtQt.#.jQt.P.Q.R.A.A.S.TQt.#",
".j.j.#Qt.#QtQt.#.U.V.A.A.G.W.XQtQtQt",
".#.#Qt.#QtQt.j.Y.Z.0.A.u.1.2.3.#.jQt",
".#.#.#.#Qt.j.4.5.6.7.A.n.8.w.#.j.#Qt",
"QtQt.#Qt.#Qt.9#..A.u###a#bQt.jQt.#Qt",
"QtQtQtQt.##c#d#e.u.u#f#g#h.#Qt.#QtQt",
".jQtQtQt.k#i.S##.A.G#j#kQtQtQtQtQt.#",
"QtQt.#.j#l#m#n.R.R#o#p#q.#.#QtQtQtQt",
"Qt.#.##r#s#t.u.A#n#u#vQt.jQt.#QtQt.#",
".#.jQt.r#w.G.R###x#y#z.#Qt.#.jQtQt.j",
".#.##l#A#B.u.R#C#D#EQtQt.#QtQt.#.#Qt",
"QtQt.E#F.R.R.R#G.KQtQtQt.#.#.#Qt.#Qt",
"Qt#H#I#J.A.R#K#L.#Qt.#QtQt.#QtQtQt.j",
"#M#N#O.R##.B#P.rQt#Q.#Qt.#Qt.#.#Qt.#",
"#R#S#T##.G#U#V.DQtQt.#Qt.#Qt.#Qt.jQt",
"#W#X#Y.u#f#Z#0Qt.j.#QtQt.j.#QtQtQt.#",
"#1#2#3#4#5#6.#Qt.#Qt.#.#Qt.jQt.#Qt.j"};

static const char* const image7_data[] = {
"51 43 256 2",
"#4 c #394541",
"#V c #4c5d5c",
"aT c #506148",
"aR c #50615f",
"aS c #51644b",
"aO c #526275",
"#U c #536557",
"aq c #536579",
"#T c #556758",
"#3 c #55682a",
"#S c #586b5e",
"af c #5c6f85",
"ae c #5f7536",
"aA c #657991",
"#Z c #667b5c",
"#2 c #677d2f",
"aJ c #677d38",
"aF c #677d87",
"aG c #687e4e",
"bQ c #687e96",
"aK c #697e79",
"az c #69803d",
"aU c #69807e",
"bT c #6b8199",
"#1 c #6b8337",
"#e c #6e849d",
"a# c #6e8562",
"bU c #6e859e",
"#0 c #6f8741",
"#R c #71878f",
"bW c #7188a2",
"ap c #71893e",
"au c #738977",
"ad c #738c37",
"#t c #758da8",
"ak c #768d73",
"bN c #768ea8",
"ay c #779039",
"aI c #77903c",
"a9 c #7891ac",
"aD c #7891ad",
"ac c #789239",
"an c #78923a",
"#B c #7991ac",
"bM c #7992ac",
"#G c #7992ae",
"ao c #79933a",
"#O c #7a92ae",
"ax c #7a933c",
"a5 c #7a93af",
"ab c #7a943e",
"aw c #7b9442",
"am c #7b9541",
"aH c #7c9647",
"bc c #7d96b2",
"aC c #7e98b5",
"aa c #7e994d",
"bg c #7f98b5",
".I c #7f99b7",
"bh c #7f9ab6",
".L c #809ab8",
"al c #809b53",
"av c #809b58",
"## c #819ab8",
"bv c #819cb9",
".x c #819cba",
".7 c #819cbb",
"#5 c #839ebb",
"bL c #839ebc",
"bE c #849ebb",
".D c #849ebe",
"#m c #85a0be",
"#f c #85a1bf",
"#y c #86a1bf",
"a4 c #87a1bf",
"aX c #87a2c1",
".A c #88a1c0",
"aY c #88a3c2",
"bO c #88a4c2",
".F c #89a3c4",
"bK c #89a5c5",
"aQ c #8aa6c0",
"bI c #8ca8c7",
".B c #8da9c9",
"#7 c #8eaaca",
"bf c #8eabcb",
"bA c #8facce",
"#b c #90accc",
"#g c #90adcd",
"bk c #91accd",
"bp c #91adcc",
".N c #91adcd",
"bG c #91adce",
".C c #91adcf",
".q c #91aecd",
"bH c #91aed0",
".e c #92afcf",
"a. c #93b0ce",
"as c #93b0d0",
"#r c #93b0d1",
"bY c #93b1d2",
"aj c #94b1d0",
"#. c #94b1d2",
".m c #94b1d3",
"bz c #95b2d5",
"bR c #96b3d4",
".E c #96b4d5",
".z c #96b4d6",
".h c #97b4d5",
".n c #97b5d6",
".G c #97b5d7",
"bi c #97b5d8",
"#k c #98b6d8",
"#I c #98b6d9",
"bs c #98b6da",
"bj c #98b8db",
"b. c #99b7d9",
"bB c #99b7db",
"#Y c #99b8d7",
"#A c #99b8db",
".j c #99b9da",
"#c c #9ab6da",
"bn c #9ab8db",
"a1 c #9ab9d8",
"a2 c #9ab9d9",
"#p c #9ab9dc",
"bo c #9abadd",
".u c #9bb9da",
"#N c #9bb9db",
"b3 c #9bb9dd",
"#E c #9bbadc",
"bb c #9bbadd",
"bS c #9bbade",
"aB c #9bbbdd",
".v c #9bbbde",
"a0 c #9cbad9",
"b7 c #9cbade",
"aV c #9cbbdc",
".k c #9dbbde",
"bC c #9dbbdf",
"bV c #9ebde0",
"#P c #9ebde1",
".M c #9fbde1",
"b2 c #9fbde3",
".J c #9fbee3",
".g c #9fbfe2",
"aW c #9fbfe3",
"bd c #9fbfe4",
"at c #9fc0e2",
"b1 c #a0bfe3",
".w c #a0bfe4",
"#C c #a0c0e4",
".o c #a1c0e3",
"br c #a1c0e5",
".8 c #a1c1e4",
".K c #a1c1e5",
"#8 c #a1c1e6",
".P c #a1c2e5",
"aL c #a2c1e6",
"a3 c #a2c2e5",
"#6 c #a2c2e6",
"be c #a2c2e7",
"bF c #a3c3e7",
"b0 c #a3c3e8",
"#X c #a4c3e8",
"bu c #a4c4e8",
"#Q c #a4c5e7",
"#q c #a4c5ea",
"ar c #a5c5e9",
"#v c #a5c5ea",
"#d c #a5c5eb",
"#w c #a5c6ea",
".3 c #a6c6ec",
".d c #a6c7ec",
".6 c #a6c7ed",
"aZ c #a7c7e7",
"ah c #a7c7ed",
".r c #a7c8ed",
".Z c #a7c8ee",
".Q c #a7c9ee",
"b8 c #a7c9ef",
".V c #a8c8ee",
"#D c #a8c8ef",
"bm c #a8c9ee",
".1 c #a8c9ef",
"b6 c #a8c9f0",
"aE c #a8caef",
"#L c #a9c9ef",
"b4 c #a9c9f0",
"#K c #a9caef",
".W c #a9caf0",
"bP c #a9caf1",
".y c #a9cbf1",
"#M c #aacbf0",
".R c #aacbf1",
"#s c #aacbf2",
"bX c #aaccf1",
"#F c #aaccf2",
"b5 c #aaccf3",
"by c #abccf2",
".X c #abccf3",
".0 c #abccf4",
".H c #abcdf3",
".U c #abcdf4",
"#W c #abcdf5",
"ai c #accdf4",
"b9 c #accdf5",
"#l c #accef4",
"#x c #accef5",
"aM c #adcef5",
"#n c #adcff5",
".f c #adcff6",
".4 c #adcff7",
"a8 c #add0f6",
"#H c #add0f8",
".s c #aecff5",
"ag c #aecff6",
"bt c #aecff7",
".S c #aed0f8",
".p c #aed1f7",
".l c #aed1f8",
".9 c #aed1f9",
".5 c #afd0f8",
"a7 c #afd1f5",
"a6 c #afd1f6",
"bw c #afd1f7",
"aP c #afd1f8",
"#o c #afd1f9",
".t c #afd2f9",
"#z c #afd2fa",
"bq c #afd2fb",
"#j c #b0d2f8",
".i c #b0d2f9",
".T c #b0d2fa",
"bZ c #b0d2fb",
".O c #b0d3fa",
".Y c #b0d3fb",
"bJ c #b1d2fa",
"#J c #b1d3fa",
"#h c #b1d3fb",
".2 c #b1d3fc",
"b# c #b1d4fa",
"#u c #b1d4fb",
".c c #b1d4fc",
"#i c #b2d4fc",
"#a c #b2d4fd",
"#9 c #b2d5fc",
"Qt c #b2d5fd",
"bD c #b2d5fe",
"bx c #b2d6fe",
"bl c #b3d4fc",
"ba c #b3d5fc",
".a c #b3d5fd",
".b c #b3d5fe",
"aN c #b3d6fd",
".# c #b3d6fe",
"QtQtQtQtQtQtQtQtQtQtQt.#Qt.aQt.#.a.#QtQtQtQt.b.#.#.#.#.aQt.#.#.#QtQt.a.aQt.aQt.#.#Qt.a.#.bQt.aQt.a.a.a",
".#.#.aQt.#Qt.a.#Qt.a.#Qt.aQtQtQtQtQtQt.#Qt.aQtQt.#.aQtQt.#.#.aQt.aQtQt.#.#QtQt.#QtQt.#QtQt.#.#.#Qt.#Qt",
".#.a.#QtQtQt.a.#Qt.aQt.aQt.#QtQt.aQt.#.#QtQt.#Qt.aQt.#Qt.#QtQt.aQtQt.a.a.#Qt.a.aQtQtQt.#.#Qt.#.#.#.#Qt",
"Qt.a.a.#.a.#.#.aQt.a.aQt.#.#.c.d.e.f.g.h.i.j.k.l.m.g.n.o.p.q.d.r.e.s.#Qt.#QtQtQtQtQt.#.#QtQt.#.#.aQt.#",
"QtQt.#.#.a.#.#.#Qt.aQt.b.t.u.v.w.x.y.z.A.p.B.C.f.D.E.F.G.H.I.J.K.L.M.NQt.#.#Qt.a.#.#.a.aQtQtQtQt.aQtQt",
".#Qt.#.#QtQt.#.#.a.bQt.O.P.Q.R.S.d.T.U.V.c.W.X.Y.Z.0.1.X.2.3.4.5.3.6.7.8.#.a.#QtQtQtQt.#.#Qt.#QtQt.#Qt",
".#Qt.#.#.bQt.a.a.#Qt.9#.##.c#a.a.#.aQt.a.#QtQt.#.#QtQtQtQtQt.#.#.a.2.d.3#bQt.#.#.#.#QtQt.#.#Qt.#.bQt.#",
".#.#Qt.#QtQtQtQtQtQt.s#c.o.bQt.a.bQt.#QtQtQtQt.#.#.b.#QtQt.#Qt.a.#QtQt#d#e.#.a.#.aQtQtQt.#Qt.#Qt.a.#.#",
".aQtQtQtQt.#.#QtQt.f#f#g#h.#Qt.a.#.#.#.#.#QtQtQtQtQtQtQtQtQt.#Qt.#QtQt.S.K.S#.QtQt.a.#.aQt.aQtQt.a.bQt",
"Qt.a.#QtQt.#Qt#i#j.S.m.1.#QtQt.a.a.#.#QtQtQtQtQtQt.#.#QtQtQtQtQtQtQt.aQt.#.t#kQtQt.#.#Qt.#QtQt.aQtQt.#",
".aQt.#.bQt.#.##l#mQt#n.#.#.#QtQtQtQt.#QtQt.a.aQt.#.#QtQt.a.#.a.#QtQt.#Qt.#Qt.f#o#p.#.#Qt.#QtQt.a.#.#.a",
"Qt.#.#.#QtQt.#.O#qQtQt.a.#.a.#Qt.#.#.#QtQt.#QtQtQt.aQtQt.aQtQt.#.a.a.#QtQt.#.#.l#r.#Qt.#.#QtQtQt.#.#Qt",
".#.a.a.#.#.a#s#t#nQt.aQt.#.aQtQt.#Qt.#QtQtQt.#.#.#.#.#.#.#.#.#.#Qt.#Qt.#.aQt.a#u#v#wQtQt.aQtQt.#.#Qt.#",
".a.a.#QtQtQt#x#y.5QtQtQtQtQtQtQtQt.#.#QtQtQtQt.#.#.#QtQt.a.#Qt.#QtQtQt.aQt.#.a#z#A#B.aQtQt.aQtQt.#Qt.#",
".aQt.a.a.aQt.i#C#u.a.#.c.#.#.#Qt.#Qt.#QtQt.aQt.a.a.aQt.#QtQtQtQt.bQtQtQt.#.#QtQt#D#EQtQt.#.#Qt.#Qt.#Qt",
".aQtQtQt.#.##F#G.5.aQtQt.#Qt.#QtQt.aQt.#Qt.aQt.a.#.#Qt.#.#Qt.#Qt.aQtQt.#Qt.#Qt.c.r.GQtQt.#.#Qt.#.#Qt.a",
"QtQt.#Qt.#.##H#I#JQtQtQtQt.#.#QtQtQt.#.#QtQtQt.U.1#K#L#M#j.#Qt.#.#QtQt.#.aQt.a#z#N#OQt.#Qt.a.#Qt.#.#.#",
".#.#.#.#.#Qt.9#P#uQt.a.#.#Qt.#Qt.#.#.#QtQtQt#Q#R#S#T#U#V#O.aQt.#QtQt.#Qt.a.#.aQt#W#XQt.#.aQt.#Qt.aQtQt",
"QtQt.#.#.aQt#o.w#iQtQtQtQt.aQtQt.#Qt.#QtQt.Y#Y#Z#0#1#2#3#4#5.#.aQt.#.#QtQtQt.b.Y#6#7Qt.#QtQtQt.aQtQt.a",
".#.aQt.#.#.#.i#8.c#u#9QtQt.#QtQt.#.#.#Qt.c#xa.a#aaabacadaeafQtQtQt.#.#QtQtQtQtQtagahQtQtQt.#QtQt.#QtQt",
".a.#.aQtQtQt.X#5.T#i.#.aQtQt.#Qt.#Qt.#Qt#uaiajakalamanaoapaq.#.aQt.#.#.#QtQt.a.Yaras.#QtQtQt.#QtQt.#.#",
".#.#.#Qt.#Qt#s#k#iQt.#.aQt.#QtQt.#.aQt.#Qt#JatauavawaxayazaAQtQt.#.#QtQtQt.#Qt.YaBaC.#.#QtQt.#.#Qt.#.#",
".#.#.#Qt.#.b.taD.r#9#i.YQtQt#9.#.#Qt.aQtQtQtaEaFaGaHaIaJaKaLQt.#QtQt.#QtQtQtQt#iaM#dQt.#QtQt.#Qt.#.a.#",
"Qt.#.##uaN.##uaOaP.Y.#.#.#Qt#9Qt.#QtQt.#.#.c#KaQaRaSaTaUaV.#.#.#QtQt.#QtQtQt.##uaWaXQt.#.#QtQt.#QtQt.#",
".i.#.##h.a.#.caY.XQt.aQt.#.c#h.aQt.#.#.#QtQt#xaZa0a1a2a3#h.#QtQtQtQtQt.#.#.a.##h.Ja4.a.#.#QtQtQtQtQt.#",
"#iQt#9.c.#.##ha5.gQtQt.#.#QtQtQtQt.#Qt.#.c#9.ia6a7#J#9.aQt.#Qt.#QtQt.#.#QtQt.a.ca8ahQtQtQtQt.aQt.a.#Qt",
"QtQtQtQt.#.##ha9b..#.#QtaN.T.#.#QtQt.#Qt.cQt.a#9b##ubaQtQtQtQt.#QtQtQtQtQtQt.a.ObbbcQtQtQtQtQt.#QtQtQt",
".#Qt.#Qt.#QtQtbd.Z.#Qt.a#h.t.#QtQt.aQt.#Qt.#Qt.##u.Yba.#.#.#.#QtQtQt.#.#.#Qt.#.cbebfQtQtQt.#Qt.#Qt.#.#",
"Qt.a.a.a.aQt.cbg#F.#Qt.#QtQtQt.#.YQtQt.#QtQt.aQt#9.caN.#QtQt.#.#Qt.a.#.#QtQt.##i.R.gQt.#Qt.#.b.#.#.a.#",
".#.aaNQtQt.##u.maiQt#iQt#9.c.#.O.T.c.#.#QtQt.c.#aNQtQtQtaNQt.#QtQtQt.aQt.aQtQt.UbhbiQt.#.aQt.#Qt.bQtQt",
".aQt#i.#.#Qt.c.Zbjbk.a.##h.cQt.l#h.aQt.#Qt.#.#.#QtblQt#i#u.caN.aQtQtQt.#.a.c.pbmbn.XQt.#Qt.aQtQtQtQt.#",
"Qt.#.#Qt.aQt.#.TbobpQt.##u.cQt.O.cQt.#QtQt.aQt.#Qt.#.##i.cQt.#Qt.#QtQt.#.#bqbrbs.i.#.#QtQtQt.a.#QtQt.#",
".#Qt.#.#.aQt.a.cbtbubvbwQt.c#9.#.#Qt.a.#QtQt.#.#.a.##9#9QtQtQt.#.#.a.#bx.i#j.4byQtQt.#QtQt.#.#.#QtQt.#",
"Qt.#Qt.#.#.a.aQtQt.tbzbAQt#i.a.#.#.#QtQt.#Qt.#QtQtQtaNQt.#Qt.#Qt.#QtQt#zbBbCQt.#QtQtQt.#.#QtQt.#QtQt.#",
".#.a.#Qt.#.a.a.#.abD.S#8bEQtQt.#Qt.#Qt.#.#Qt.#QtQtQt.aQt.#QtQtQtQt.##xbE.Hai.#.aQt.aQt.aQtQtQtQt.#.#Qt",
"Qt.aQtQtQt.#.#Qt.#Qt.a#zbFQt.##i#u.Y#u.#QtQt.#QtQtQtQt.aQt.bQtQt.#.SbGbH.#QtQt.a.#QtQt.#.#.#Qt.a.aQt.#",
"QtQt.aQt.#.#Qt.#QtQt.#.##a.fbIbJ#i#9#9.aQtQtQt.#Qt.#.#.#.a.aQt.#.fbK.1aiQt.#.#QtQt.#.aQtQt.a.aQtQtQtQt",
"Qt.#.#.#QtQtQt.aQt.#Qt.#.a#z.o#xbLbMaibNbObPbQbRbSbT.EbUbV.nbWbXbIbY.aQtQtQt.#Qt.#Qt.a.#QtQt.#QtQt.#.#",
".#.a.#QtQt.#.#QtQtQt.#QtQtQt.TbZb0b1.Yb2#v#zb3b4b5b3b6b7.Ub8bC#o.db9Qt.#.#.#.#.#.#.#Qt.#QtQtQtQt.#.#Qt",
"Qt.a.a.#.#.#QtQt.#.a.#QtQtQtQtQtQt.#.#.#QtQt.#Qt.#.#Qt.aQt.#.a.#.#.a.#.#.#QtQtQtQt.a.#QtQtQtQtQt.#.#Qt",
"Qt.#.a.#.a.aQt.aQt.a.#.#QtQt.bQtQtQtQt.aQt.#Qt.a.#QtQt.aQt.#QtQtQt.a.a.#Qt.#QtQtQt.#.#.#QtQtQt.#.#.aQt",
"Qt.#Qt.#.#.a.#QtQtQtQtQt.#.#Qt.aQtQtQtQtQtQtQt.#.aQtQt.#Qt.b.a.aQtQtQt.#QtQt.#.#QtQtQt.#QtQt.aQt.#.a.#",
"Qt.#QtQt.#Qt.aQt.aQt.#.#Qt.#.aQtQtQtQt.#Qt.#.#Qt.#QtQt.#QtQtQt.aQt.#.#Qt.a.#.#.a.#Qt.aQtQtQt.#Qt.#.#Qt"};


/*
 *  Constructs a SeaBee3MainDisplayForm as a child of 'parent', with the
 *  name 'name' and widget flags set to 'f'.
 *
 */
SeaBee3MainDisplayForm::SeaBee3MainDisplayForm( QWidget* parent, const char* name, WFlags fl )
    : QMainWindow( parent, name, fl ),
      image0( (const char **) image0_data ),
      image1( (const char **) image1_data ),
      image2( (const char **) image2_data ),
      image3( (const char **) image3_data ),
      image4( (const char **) image4_data ),
      image5( (const char **) image5_data ),
      image6( (const char **) image6_data ),
      image7( (const char **) image7_data )
{
    (void)statusBar();
    if ( !name )
        setName( "SeaBee3MainDisplayForm" );
    setPaletteForegroundColor( QColor( 255, 255, 255 ) );
    setPaletteBackgroundColor( QColor( 0, 0, 0 ) );
    setCentralWidget( new QWidget( this, "qt_central_widget" ) );

    groupBox9_4 = new QGroupBox( centralWidget(), "groupBox9_4" );
    groupBox9_4->setEnabled( TRUE );
    groupBox9_4->setGeometry( QRect( 720, 530, 240, 190 ) );

    frame3 = new QFrame( groupBox9_4, "frame3" );
    frame3->setEnabled( TRUE );
    frame3->setGeometry( QRect( 10, 20, 220, 160 ) );
    frame3->setPaletteBackgroundColor( QColor( 0, 0, 0 ) );
    frame3->setFrameShape( QFrame::StyledPanel );
    frame3->setFrameShadow( QFrame::Raised );

    ThrusterCurrentMeterCanvas4 = new ImageCanvas( frame3, "ThrusterCurrentMeterCanvas4" );
    ThrusterCurrentMeterCanvas4->setEnabled( TRUE );
    ThrusterCurrentMeterCanvas4->setGeometry( QRect( 90, 10, 16, 120 ) );

    ThrusterCurrentMeterCanvas5 = new ImageCanvas( frame3, "ThrusterCurrentMeterCanvas5" );
    ThrusterCurrentMeterCanvas5->setEnabled( TRUE );
    ThrusterCurrentMeterCanvas5->setGeometry( QRect( 110, 10, 16, 120 ) );

    ThrusterCurrentMeterCanvas2 = new ImageCanvas( frame3, "ThrusterCurrentMeterCanvas2" );
    ThrusterCurrentMeterCanvas2->setEnabled( TRUE );
    ThrusterCurrentMeterCanvas2->setGeometry( QRect( 50, 10, 16, 120 ) );

    ThrusterCurrentMeterCanvas3 = new ImageCanvas( frame3, "ThrusterCurrentMeterCanvas3" );
    ThrusterCurrentMeterCanvas3->setEnabled( TRUE );
    ThrusterCurrentMeterCanvas3->setGeometry( QRect( 70, 10, 16, 120 ) );

    ThrusterCurrentMeterCanvas6 = new ImageCanvas( frame3, "ThrusterCurrentMeterCanvas6" );
    ThrusterCurrentMeterCanvas6->setEnabled( TRUE );
    ThrusterCurrentMeterCanvas6->setGeometry( QRect( 130, 10, 16, 120 ) );

    textLabel2 = new QLabel( frame3, "textLabel2" );
    textLabel2->setEnabled( TRUE );
    textLabel2->setGeometry( QRect( 10, 130, 16, 21 ) );
    textLabel2->setPaletteForegroundColor( QColor( 255, 255, 255 ) );

    textLabel2_2 = new QLabel( frame3, "textLabel2_2" );
    textLabel2_2->setEnabled( TRUE );
    textLabel2_2->setGeometry( QRect( 30, 130, 16, 21 ) );
    textLabel2_2->setPaletteForegroundColor( QColor( 255, 255, 255 ) );

    textLabel2_2_2 = new QLabel( frame3, "textLabel2_2_2" );
    textLabel2_2_2->setEnabled( TRUE );
    textLabel2_2_2->setGeometry( QRect( 50, 130, 16, 21 ) );
    textLabel2_2_2->setPaletteForegroundColor( QColor( 255, 255, 255 ) );

    textLabel2_2_3 = new QLabel( frame3, "textLabel2_2_3" );
    textLabel2_2_3->setEnabled( TRUE );
    textLabel2_2_3->setGeometry( QRect( 70, 130, 16, 21 ) );
    textLabel2_2_3->setPaletteForegroundColor( QColor( 255, 255, 255 ) );

    textLabel2_2_4 = new QLabel( frame3, "textLabel2_2_4" );
    textLabel2_2_4->setEnabled( TRUE );
    textLabel2_2_4->setGeometry( QRect( 90, 130, 16, 21 ) );
    textLabel2_2_4->setPaletteForegroundColor( QColor( 255, 255, 255 ) );

    textLabel2_2_5 = new QLabel( frame3, "textLabel2_2_5" );
    textLabel2_2_5->setEnabled( TRUE );
    textLabel2_2_5->setGeometry( QRect( 110, 130, 16, 21 ) );
    textLabel2_2_5->setPaletteForegroundColor( QColor( 255, 255, 255 ) );

    textLabel2_2_6 = new QLabel( frame3, "textLabel2_2_6" );
    textLabel2_2_6->setEnabled( TRUE );
    textLabel2_2_6->setGeometry( QRect( 130, 130, 16, 21 ) );
    textLabel2_2_6->setPaletteForegroundColor( QColor( 255, 255, 255 ) );

    textLabel2_2_7 = new QLabel( frame3, "textLabel2_2_7" );
    textLabel2_2_7->setEnabled( TRUE );
    textLabel2_2_7->setGeometry( QRect( 150, 130, 16, 21 ) );
    textLabel2_2_7->setPaletteForegroundColor( QColor( 255, 255, 255 ) );

    ThrusterCurrentMeterCanvas7 = new ImageCanvas( frame3, "ThrusterCurrentMeterCanvas7" );
    ThrusterCurrentMeterCanvas7->setEnabled( TRUE );
    ThrusterCurrentMeterCanvas7->setGeometry( QRect( 150, 10, 16, 120 ) );

    textLabel1_3 = new QLabel( frame3, "textLabel1_3" );
    textLabel1_3->setEnabled( TRUE );
    textLabel1_3->setGeometry( QRect( 175, 95, 25, 16 ) );
    textLabel1_3->setPaletteForegroundColor( QColor( 144, 144, 144 ) );
    QFont textLabel1_3_font(  textLabel1_3->font() );
    textLabel1_3_font.setFamily( "Helvetica [Adobe]" );
    textLabel1_3->setFont( textLabel1_3_font );

    textLabel1_2 = new QLabel( frame3, "textLabel1_2" );
    textLabel1_2->setEnabled( TRUE );
    textLabel1_2->setGeometry( QRect( 175, 50, 25, 16 ) );
    textLabel1_2->setPaletteForegroundColor( QColor( 144, 144, 144 ) );
    QFont textLabel1_2_font(  textLabel1_2->font() );
    textLabel1_2_font.setFamily( "Helvetica [Adobe]" );
    textLabel1_2->setFont( textLabel1_2_font );

    textLabel1 = new QLabel( frame3, "textLabel1" );
    textLabel1->setEnabled( TRUE );
    textLabel1->setGeometry( QRect( 175, 10, 25, 16 ) );
    textLabel1->setPaletteForegroundColor( QColor( 144, 144, 144 ) );
    QFont textLabel1_font(  textLabel1->font() );
    textLabel1_font.setFamily( "Helvetica [Adobe]" );
    textLabel1->setFont( textLabel1_font );

    ThrusterCurrentMeterCanvas1 = new ImageCanvas( frame3, "ThrusterCurrentMeterCanvas1" );
    ThrusterCurrentMeterCanvas1->setEnabled( TRUE );
    ThrusterCurrentMeterCanvas1->setGeometry( QRect( 30, 10, 16, 120 ) );

    ThrusterCurrentMeterCanvas0 = new ImageCanvas( frame3, "ThrusterCurrentMeterCanvas0" );
    ThrusterCurrentMeterCanvas0->setEnabled( TRUE );
    ThrusterCurrentMeterCanvas0->setGeometry( QRect( 10, 10, 16, 120 ) );

    groupBox1 = new QGroupBox( centralWidget(), "groupBox1" );
    groupBox1->setEnabled( TRUE );
    groupBox1->setGeometry( QRect( 720, 10, 240, 510 ) );

    groupBox9_3 = new QGroupBox( groupBox1, "groupBox9_3" );
    groupBox9_3->setEnabled( TRUE );
    groupBox9_3->setGeometry( QRect( 10, 180, 150, 160 ) );

    field_int_press = new QLineEdit( groupBox9_3, "field_int_press" );
    field_int_press->setGeometry( QRect( 10, 130, 130, 21 ) );
    field_int_press->setPaletteForegroundColor( QColor( 30, 253, 0 ) );
    field_int_press->setPaletteBackgroundColor( QColor( 0, 0, 0 ) );
    QFont field_int_press_font(  field_int_press->font() );
    field_int_press_font.setPointSize( 12 );
    field_int_press->setFont( field_int_press_font );
    field_int_press->setFrame( FALSE );
    field_int_press->setCursorPosition( 1 );
    field_int_press->setAlignment( int( QLineEdit::AlignHCenter ) );

    IPressureCanvas = new ImageCanvas( groupBox9_3, "IPressureCanvas" );
    IPressureCanvas->setEnabled( TRUE );
    IPressureCanvas->setGeometry( QRect( 10, 20, 130, 110 ) );

    ExtPressAuto = new QCheckBox( groupBox1, "ExtPressAuto" );
    ExtPressAuto->setEnabled( TRUE );
    ExtPressAuto->setGeometry( QRect( 160, 400, 60, 30 ) );
    QFont ExtPressAuto_font(  ExtPressAuto->font() );
    ExtPressAuto_font.setFamily( "Helvetica [Adobe]" );
    ExtPressAuto->setFont( ExtPressAuto_font );
    ExtPressAuto->setChecked( TRUE );

    IntPressMax = new QSpinBox( groupBox1, "IntPressMax" );
    IntPressMax->setEnabled( TRUE );
    IntPressMax->setGeometry( QRect( 160, 189, 50, 17 ) );
    IntPressMax->setPaletteForegroundColor( QColor( 255, 255, 255 ) );
    IntPressMax->setPaletteBackgroundColor( QColor( 71, 71, 71 ) );
    QFont IntPressMax_font(  IntPressMax->font() );
    IntPressMax_font.setFamily( "Helvetica [Adobe]" );
    IntPressMax->setFont( IntPressMax_font );
    IntPressMax->setMaxValue( 10000 );

    IntPressMin = new QSpinBox( groupBox1, "IntPressMin" );
    IntPressMin->setEnabled( TRUE );
    IntPressMin->setGeometry( QRect( 160, 310, 50, 20 ) );
    IntPressMin->setPaletteForegroundColor( QColor( 255, 255, 255 ) );
    IntPressMin->setPaletteBackgroundColor( QColor( 71, 71, 71 ) );
    QFont IntPressMin_font(  IntPressMin->font() );
    IntPressMin_font.setFamily( "Helvetica [Adobe]" );
    IntPressMin->setFont( IntPressMin_font );
    IntPressMin->setMaxValue( 10000 );

    ExtPressMax = new QSpinBox( groupBox1, "ExtPressMax" );
    ExtPressMax->setEnabled( TRUE );
    ExtPressMax->setGeometry( QRect( 160, 350, 50, 16 ) );
    ExtPressMax->setPaletteForegroundColor( QColor( 255, 255, 255 ) );
    ExtPressMax->setPaletteBackgroundColor( QColor( 71, 71, 71 ) );
    QFont ExtPressMax_font(  ExtPressMax->font() );
    ExtPressMax_font.setFamily( "Helvetica [Adobe]" );
    ExtPressMax->setFont( ExtPressMax_font );
    ExtPressMax->setMaxValue( 10000 );

    ExtPressMin = new QSpinBox( groupBox1, "ExtPressMin" );
    ExtPressMin->setEnabled( TRUE );
    ExtPressMin->setGeometry( QRect( 160, 470, 50, 20 ) );
    ExtPressMin->setPaletteForegroundColor( QColor( 255, 255, 255 ) );
    ExtPressMin->setPaletteBackgroundColor( QColor( 71, 71, 71 ) );
    QFont ExtPressMin_font(  ExtPressMin->font() );
    ExtPressMin_font.setFamily( "Helvetica [Adobe]" );
    ExtPressMin->setFont( ExtPressMin_font );
    ExtPressMin->setMaxValue( 10000 );

    groupBox6 = new QGroupBox( groupBox1, "groupBox6" );
    groupBox6->setEnabled( TRUE );
    groupBox6->setGeometry( QRect( 10, 20, 150, 160 ) );

    CompassCanvas = new ImageCanvas( groupBox6, "CompassCanvas" );
    CompassCanvas->setEnabled( TRUE );
    CompassCanvas->setGeometry( QRect( 10, 20, 130, 110 ) );

    field_heading = new QLineEdit( groupBox6, "field_heading" );
    field_heading->setGeometry( QRect( 10, 130, 130, 21 ) );
    field_heading->setPaletteForegroundColor( QColor( 30, 253, 0 ) );
    field_heading->setPaletteBackgroundColor( QColor( 0, 0, 0 ) );
    QFont field_heading_font(  field_heading->font() );
    field_heading_font.setPointSize( 12 );
    field_heading->setFont( field_heading_font );
    field_heading->setFrame( FALSE );
    field_heading->setAlignment( int( QLineEdit::AlignHCenter ) );

    IntPressAuto = new QCheckBox( groupBox1, "IntPressAuto" );
    IntPressAuto->setEnabled( TRUE );
    IntPressAuto->setGeometry( QRect( 160, 240, 60, 40 ) );
    QFont IntPressAuto_font(  IntPressAuto->font() );
    IntPressAuto_font.setFamily( "Helvetica [Adobe]" );
    IntPressAuto->setFont( IntPressAuto_font );
    IntPressAuto->setChecked( TRUE );

    groupBox10 = new QGroupBox( groupBox1, "groupBox10" );
    groupBox10->setEnabled( TRUE );
    groupBox10->setGeometry( QRect( 10, 340, 150, 160 ) );

    EPressureCanvas = new ImageCanvas( groupBox10, "EPressureCanvas" );
    EPressureCanvas->setEnabled( TRUE );
    EPressureCanvas->setGeometry( QRect( 10, 20, 130, 110 ) );

    field_ext_press = new QLineEdit( groupBox10, "field_ext_press" );
    field_ext_press->setGeometry( QRect( 10, 130, 130, 21 ) );
    field_ext_press->setPaletteForegroundColor( QColor( 30, 253, 0 ) );
    field_ext_press->setPaletteBackgroundColor( QColor( 0, 0, 0 ) );
    QFont field_ext_press_font(  field_ext_press->font() );
    field_ext_press_font.setPointSize( 12 );
    field_ext_press->setFont( field_ext_press_font );
    field_ext_press->setFrame( FALSE );
    field_ext_press->setCursorPosition( 1 );
    field_ext_press->setAlignment( int( QLineEdit::AlignHCenter ) );

    ImageDisplay0_2_2 = new ImageCanvas( centralWidget(), "ImageDisplay0_2_2" );
    ImageDisplay0_2_2->setEnabled( TRUE );
    ImageDisplay0_2_2->setGeometry( QRect( 25, 25, 290, 240 ) );

    desired_heading_field_2_3 = new QLineEdit( centralWidget(), "desired_heading_field_2_3" );
    desired_heading_field_2_3->setEnabled( TRUE );
    desired_heading_field_2_3->setGeometry( QRect( 430, 190, 60, 21 ) );

    desired_heading_field_2_2 = new QLineEdit( centralWidget(), "desired_heading_field_2_2" );
    desired_heading_field_2_2->setEnabled( TRUE );
    desired_heading_field_2_2->setGeometry( QRect( 430, 260, 60, 21 ) );

    groupBox10_2 = new QGroupBox( centralWidget(), "groupBox10_2" );
    groupBox10_2->setEnabled( TRUE );
    groupBox10_2->setGeometry( QRect( 10, 559, 701, 160 ) );

    groupBox2 = new QGroupBox( groupBox10_2, "groupBox2" );
    groupBox2->setEnabled( TRUE );
    groupBox2->setGeometry( QRect( 290, 30, 390, 100 ) );

    buttonGroup1 = new QButtonGroup( groupBox2, "buttonGroup1" );
    buttonGroup1->setEnabled( TRUE );
    buttonGroup1->setGeometry( QRect( 10, 20, 190, 70 ) );

    radio_manual = new QRadioButton( buttonGroup1, "radio_manual" );
    radio_manual->setEnabled( TRUE );
    radio_manual->setGeometry( QRect( 20, 21, 110, 20 ) );
    radio_manual->setChecked( TRUE );

    radio_auto = new QRadioButton( buttonGroup1, "radio_auto" );
    radio_auto->setEnabled( TRUE );
    radio_auto->setGeometry( QRect( 20, 41, 80, 16 ) );

    desired_speed_field = new QLineEdit( groupBox2, "desired_speed_field" );
    desired_speed_field->setEnabled( TRUE );
    desired_speed_field->setGeometry( QRect( 320, 60, 60, 20 ) );

    desired_depth_field = new QLineEdit( groupBox2, "desired_depth_field" );
    desired_depth_field->setEnabled( TRUE );
    desired_depth_field->setGeometry( QRect( 320, 40, 60, 20 ) );
    desired_depth_field->setPaletteForegroundColor( QColor( 0, 0, 0 ) );

    textLabel2_3_2_2 = new QLabel( groupBox2, "textLabel2_3_2_2" );
    textLabel2_3_2_2->setEnabled( TRUE );
    textLabel2_3_2_2->setGeometry( QRect( 210, 20, 105, 22 ) );
    QFont textLabel2_3_2_2_font(  textLabel2_3_2_2->font() );
    textLabel2_3_2_2->setFont( textLabel2_3_2_2_font );
    textLabel2_3_2_2->setAlignment( int( QLabel::AlignVCenter | QLabel::AlignRight ) );

    textLabel2_2_2_3_2_2 = new QLabel( groupBox2, "textLabel2_2_2_3_2_2" );
    textLabel2_2_2_3_2_2->setEnabled( TRUE );
    textLabel2_2_2_3_2_2->setGeometry( QRect( 210, 40, 92, 22 ) );
    QFont textLabel2_2_2_3_2_2_font(  textLabel2_2_2_3_2_2->font() );
    textLabel2_2_2_3_2_2->setFont( textLabel2_2_2_3_2_2_font );
    textLabel2_2_2_3_2_2->setAlignment( int( QLabel::AlignVCenter | QLabel::AlignRight ) );

    textLabel2_2_2_3_2_2_2 = new QLabel( groupBox2, "textLabel2_2_2_3_2_2_2" );
    textLabel2_2_2_3_2_2_2->setEnabled( TRUE );
    textLabel2_2_2_3_2_2_2->setGeometry( QRect( 210, 60, 93, 22 ) );
    QFont textLabel2_2_2_3_2_2_2_font(  textLabel2_2_2_3_2_2_2->font() );
    textLabel2_2_2_3_2_2_2->setFont( textLabel2_2_2_3_2_2_2_font );
    textLabel2_2_2_3_2_2_2->setAlignment( int( QLabel::AlignVCenter | QLabel::AlignRight ) );

    desired_heading_field = new QLineEdit( groupBox2, "desired_heading_field" );
    desired_heading_field->setEnabled( TRUE );
    desired_heading_field->setGeometry( QRect( 320, 20, 60, 21 ) );

    groupBox4 = new QGroupBox( groupBox10_2, "groupBox4" );
    groupBox4->setEnabled( TRUE );
    groupBox4->setGeometry( QRect( 20, 30, 230, 100 ) );

    textLabel2_3_2_2_2 = new QLabel( groupBox4, "textLabel2_3_2_2_2" );
    textLabel2_3_2_2_2->setEnabled( TRUE );
    textLabel2_3_2_2_2->setGeometry( QRect( 20, 25, 119, 22 ) );

    textLabel2_3_2_2_2_2 = new QLabel( groupBox4, "textLabel2_3_2_2_2_2" );
    textLabel2_3_2_2_2_2->setEnabled( TRUE );
    textLabel2_3_2_2_2_2->setGeometry( QRect( 20, 55, 119, 22 ) );

    heading_output_field = new QLineEdit( groupBox4, "heading_output_field" );
    heading_output_field->setEnabled( TRUE );
    heading_output_field->setGeometry( QRect( 130, 20, 70, 23 ) );
    heading_output_field->setPaletteForegroundColor( QColor( 38, 255, 0 ) );
    heading_output_field->setPaletteBackgroundColor( QColor( 0, 0, 0 ) );
    heading_output_field->setAlignment( int( QLineEdit::AlignHCenter ) );

    depth_output_field = new QLineEdit( groupBox4, "depth_output_field" );
    depth_output_field->setEnabled( TRUE );
    depth_output_field->setGeometry( QRect( 130, 51, 70, 23 ) );
    depth_output_field->setPaletteForegroundColor( QColor( 0, 255, 0 ) );
    depth_output_field->setPaletteBackgroundColor( QColor( 0, 0, 0 ) );
    depth_output_field->setAlignment( int( QLineEdit::AlignHCenter ) );

    tabWidget3 = new QTabWidget( centralWidget(), "tabWidget3" );
    tabWidget3->setEnabled( TRUE );
    tabWidget3->setGeometry( QRect( 20, 10, 700, 550 ) );
    tabWidget3->setPaletteForegroundColor( QColor( 255, 255, 255 ) );
    tabWidget3->setPaletteBackgroundColor( QColor( 0, 0, 0 ) );

    tab = new QWidget( tabWidget3, "tab" );

    frame4 = new QFrame( tab, "frame4" );
    frame4->setEnabled( TRUE );
    frame4->setGeometry( QRect( 55, 10, 595, 495 ) );
    frame4->setPaletteBackgroundColor( QColor( 48, 48, 48 ) );
    frame4->setFrameShape( QFrame::StyledPanel );
    frame4->setFrameShadow( QFrame::Raised );

    ImageDisplay1 = new ImageCanvas( frame4, "ImageDisplay1" );
    ImageDisplay1->setEnabled( TRUE );
    ImageDisplay1->setGeometry( QRect( 300, 5, 290, 240 ) );

    ImageDisplay2 = new ImageCanvas( frame4, "ImageDisplay2" );
    ImageDisplay2->setEnabled( TRUE );
    ImageDisplay2->setGeometry( QRect( 5, 250, 290, 240 ) );

    ImageDisplay3 = new ImageCanvas( frame4, "ImageDisplay3" );
    ImageDisplay3->setEnabled( TRUE );
    ImageDisplay3->setGeometry( QRect( 300, 250, 290, 240 ) );

    ImageDisplay0 = new ImageCanvas( frame4, "ImageDisplay0" );
    ImageDisplay0->setEnabled( TRUE );
    ImageDisplay0->setGeometry( QRect( 5, 5, 290, 240 ) );
    tabWidget3->insertTab( tab, QString::fromLatin1("") );

    TabPage = new QWidget( tabWidget3, "TabPage" );

    frame4_2 = new QFrame( TabPage, "frame4_2" );
    frame4_2->setEnabled( TRUE );
    frame4_2->setGeometry( QRect( 60, 10, 300, 495 ) );
    frame4_2->setPaletteBackgroundColor( QColor( 48, 48, 48 ) );
    frame4_2->setFrameShape( QFrame::StyledPanel );
    frame4_2->setFrameShadow( QFrame::Raised );

    MovementDisplay0 = new ImageCanvas( frame4_2, "MovementDisplay0" );
    MovementDisplay0->setEnabled( TRUE );
    MovementDisplay0->setGeometry( QRect( 5, 5, 290, 240 ) );

    MovementDisplay1 = new ImageCanvas( frame4_2, "MovementDisplay1" );
    MovementDisplay1->setEnabled( TRUE );
    MovementDisplay1->setGeometry( QRect( 5, 250, 290, 240 ) );
    tabWidget3->insertTab( TabPage, QString::fromLatin1("") );

    tab_2 = new QWidget( tabWidget3, "tab_2" );

    groupBox9_2 = new QGroupBox( tab_2, "groupBox9_2" );
    groupBox9_2->setEnabled( TRUE );
    groupBox9_2->setGeometry( QRect( 180, 10, 140, 171 ) );

    textLabel2_3_2_2_3_3_2 = new QLabel( groupBox9_2, "textLabel2_3_2_2_3_3_2" );
    textLabel2_3_2_2_3_3_2->setEnabled( TRUE );
    textLabel2_3_2_2_3_3_2->setGeometry( QRect( 20, 92, 20, 30 ) );

    textLabel2_3_2_2_3_2_2 = new QLabel( groupBox9_2, "textLabel2_3_2_2_3_2_2" );
    textLabel2_3_2_2_3_2_2->setEnabled( TRUE );
    textLabel2_3_2_2_3_2_2->setGeometry( QRect( 18, 65, 20, 30 ) );

    field_depth_i = new QLineEdit( groupBox9_2, "field_depth_i" );
    field_depth_i->setEnabled( TRUE );
    field_depth_i->setGeometry( QRect( 48, 95, 30, 23 ) );

    field_depth_d = new QLineEdit( groupBox9_2, "field_depth_d" );
    field_depth_d->setEnabled( TRUE );
    field_depth_d->setGeometry( QRect( 48, 125, 30, 23 ) );

    field_depth_k = new QLineEdit( groupBox9_2, "field_depth_k" );
    field_depth_k->setEnabled( TRUE );
    field_depth_k->setGeometry( QRect( 48, 36, 30, 23 ) );

    textLabel2_3_2_2_3_2_4_2_2 = new QLabel( groupBox9_2, "textLabel2_3_2_2_3_2_4_2_2" );
    textLabel2_3_2_2_3_2_4_2_2->setEnabled( TRUE );
    textLabel2_3_2_2_3_2_4_2_2->setGeometry( QRect( 20, 33, 20, 30 ) );

    textLabel2_3_2_2_3_4_2 = new QLabel( groupBox9_2, "textLabel2_3_2_2_3_4_2" );
    textLabel2_3_2_2_3_4_2->setEnabled( TRUE );
    textLabel2_3_2_2_3_4_2->setGeometry( QRect( 18, 125, 20, 30 ) );

    field_depth_p = new QLineEdit( groupBox9_2, "field_depth_p" );
    field_depth_p->setEnabled( TRUE );
    field_depth_p->setGeometry( QRect( 48, 65, 30, 23 ) );

    groupBox11 = new QGroupBox( tab_2, "groupBox11" );
    groupBox11->setEnabled( TRUE );
    groupBox11->setGeometry( QRect( 10, 400, 210, 110 ) );

    textLabel1_4_2 = new QLabel( groupBox11, "textLabel1_4_2" );
    textLabel1_4_2->setEnabled( TRUE );
    textLabel1_4_2->setGeometry( QRect( 60, 60, 34, 21 ) );
    QFont textLabel1_4_2_font(  textLabel1_4_2->font() );
    textLabel1_4_2_font.setPointSize( 8 );
    textLabel1_4_2->setFont( textLabel1_4_2_font );

    textLabel1_4_3 = new QLabel( groupBox11, "textLabel1_4_3" );
    textLabel1_4_3->setEnabled( TRUE );
    textLabel1_4_3->setGeometry( QRect( 110, 60, 34, 21 ) );
    QFont textLabel1_4_3_font(  textLabel1_4_3->font() );
    textLabel1_4_3_font.setPointSize( 8 );
    textLabel1_4_3->setFont( textLabel1_4_3_font );

    textLabel1_4_3_2 = new QLabel( groupBox11, "textLabel1_4_3_2" );
    textLabel1_4_3_2->setEnabled( TRUE );
    textLabel1_4_3_2->setGeometry( QRect( 160, 60, 34, 21 ) );
    QFont textLabel1_4_3_2_font(  textLabel1_4_3_2->font() );
    textLabel1_4_3_2_font.setPointSize( 8 );
    textLabel1_4_3_2->setFont( textLabel1_4_3_2_font );

    textLabel1_4 = new QLabel( groupBox11, "textLabel1_4" );
    textLabel1_4->setEnabled( TRUE );
    textLabel1_4->setGeometry( QRect( 10, 60, 34, 21 ) );
    QFont textLabel1_4_font(  textLabel1_4->font() );
    textLabel1_4_font.setPointSize( 8 );
    textLabel1_4->setFont( textLabel1_4_font );

    recordButton = new QPushButton( groupBox11, "recordButton" );
    recordButton->setEnabled( TRUE );
    recordButton->setGeometry( QRect( 10, 20, 40, 40 ) );
    recordButton->setPaletteForegroundColor( QColor( 255, 0, 0 ) );
    recordButton->setPaletteBackgroundColor( QColor( 112, 114, 116 ) );
    QFont recordButton_font(  recordButton->font() );
    recordButton_font.setPointSize( 20 );
    recordButton->setFont( recordButton_font );

    stopButton = new QPushButton( groupBox11, "stopButton" );
    stopButton->setEnabled( TRUE );
    stopButton->setGeometry( QRect( 60, 20, 40, 40 ) );
    stopButton->setPaletteForegroundColor( QColor( 255, 255, 0 ) );
    stopButton->setPaletteBackgroundColor( QColor( 112, 114, 116 ) );
    QFont stopButton_font(  stopButton->font() );
    stopButton_font.setPointSize( 20 );
    stopButton->setFont( stopButton_font );

    eraseButton = new QPushButton( groupBox11, "eraseButton" );
    eraseButton->setEnabled( TRUE );
    eraseButton->setGeometry( QRect( 110, 20, 40, 40 ) );
    eraseButton->setPaletteForegroundColor( QColor( 0, 26, 255 ) );
    eraseButton->setPaletteBackgroundColor( QColor( 112, 114, 116 ) );
    QFont eraseButton_font(  eraseButton->font() );
    eraseButton_font.setPointSize( 20 );
    eraseButton->setFont( eraseButton_font );

    saveButton = new QPushButton( groupBox11, "saveButton" );
    saveButton->setEnabled( TRUE );
    saveButton->setGeometry( QRect( 160, 20, 40, 40 ) );
    saveButton->setPaletteForegroundColor( QColor( 68, 255, 0 ) );
    saveButton->setPaletteBackgroundColor( QColor( 112, 114, 116 ) );
    QFont saveButton_font(  saveButton->font() );
    saveButton_font.setPointSize( 20 );
    saveButton->setFont( saveButton_font );

    textLabel1_4_4 = new QLabel( groupBox11, "textLabel1_4_4" );
    textLabel1_4_4->setEnabled( TRUE );
    textLabel1_4_4->setGeometry( QRect( 10, 80, 70, 21 ) );
    QFont textLabel1_4_4_font(  textLabel1_4_4->font() );
    textLabel1_4_4_font.setPointSize( 8 );
    textLabel1_4_4->setFont( textLabel1_4_4_font );

    dataLoggerLCD = new QLCDNumber( groupBox11, "dataLoggerLCD" );
    dataLoggerLCD->setEnabled( TRUE );
    dataLoggerLCD->setGeometry( QRect( 70, 80, 111, 21 ) );
    dataLoggerLCD->setPaletteForegroundColor( QColor( 110, 190, 52 ) );

    groupBox9 = new QGroupBox( tab_2, "groupBox9" );
    groupBox9->setEnabled( TRUE );
    groupBox9->setGeometry( QRect( 20, 10, 140, 171 ) );

    textLabel2_3_2_2_3_2_4_2 = new QLabel( groupBox9, "textLabel2_3_2_2_3_2_4_2" );
    textLabel2_3_2_2_3_2_4_2->setEnabled( TRUE );
    textLabel2_3_2_2_3_2_4_2->setGeometry( QRect( 22, 28, 20, 30 ) );

    textLabel2_3_2_2_3_2 = new QLabel( groupBox9, "textLabel2_3_2_2_3_2" );
    textLabel2_3_2_2_3_2->setEnabled( TRUE );
    textLabel2_3_2_2_3_2->setGeometry( QRect( 20, 60, 20, 30 ) );

    textLabel2_3_2_2_3_3 = new QLabel( groupBox9, "textLabel2_3_2_2_3_3" );
    textLabel2_3_2_2_3_3->setEnabled( TRUE );
    textLabel2_3_2_2_3_3->setGeometry( QRect( 22, 87, 20, 30 ) );

    textLabel2_3_2_2_3_4 = new QLabel( groupBox9, "textLabel2_3_2_2_3_4" );
    textLabel2_3_2_2_3_4->setEnabled( TRUE );
    textLabel2_3_2_2_3_4->setGeometry( QRect( 20, 120, 20, 30 ) );

    field_heading_i = new QLineEdit( groupBox9, "field_heading_i" );
    field_heading_i->setEnabled( TRUE );
    field_heading_i->setGeometry( QRect( 50, 90, 30, 23 ) );

    field_heading_k = new QLineEdit( groupBox9, "field_heading_k" );
    field_heading_k->setEnabled( TRUE );
    field_heading_k->setGeometry( QRect( 50, 31, 30, 23 ) );

    field_heading_d = new QLineEdit( groupBox9, "field_heading_d" );
    field_heading_d->setEnabled( TRUE );
    field_heading_d->setGeometry( QRect( 50, 120, 30, 23 ) );

    field_heading_p = new QLineEdit( groupBox9, "field_heading_p" );
    field_heading_p->setEnabled( TRUE );
    field_heading_p->setGeometry( QRect( 50, 60, 30, 23 ) );
    tabWidget3->insertTab( tab_2, QString::fromLatin1("") );

    TabPage_2 = new QWidget( tabWidget3, "TabPage_2" );

    MapCanvas = new ImageCanvas( TabPage_2, "MapCanvas" );
    MapCanvas->setEnabled( TRUE );
    MapCanvas->setGeometry( QRect( 10, 10, 400, 400 ) );

    groupBox15 = new QGroupBox( TabPage_2, "groupBox15" );
    groupBox15->setGeometry( QRect( 8, 414, 401, 101 ) );

    textLabel1_5 = new QLabel( groupBox15, "textLabel1_5" );
    textLabel1_5->setGeometry( QRect( 10, 60, 72, 31 ) );

    MapName = new QLineEdit( groupBox15, "MapName" );
    MapName->setGeometry( QRect( 90, 60, 131, 30 ) );

    SaveMapBtn = new QPushButton( groupBox15, "SaveMapBtn" );
    SaveMapBtn->setGeometry( QRect( 260, 60, 130, 30 ) );

    LoadMapBtn = new QPushButton( groupBox15, "LoadMapBtn" );
    LoadMapBtn->setGeometry( QRect( 260, 20, 131, 31 ) );

    groupBox12 = new QGroupBox( TabPage_2, "groupBox12" );
    groupBox12->setEnabled( TRUE );
    groupBox12->setGeometry( QRect( 420, 0, 270, 360 ) );

    PlatformBtn = new QPushButton( groupBox12, "PlatformBtn" );
    PlatformBtn->setEnabled( TRUE );
    PlatformBtn->setGeometry( QRect( 120, 30, 120, 21 ) );

    PlatformPic = new QLabel( groupBox12, "PlatformPic" );
    PlatformPic->setEnabled( TRUE );
    PlatformPic->setGeometry( QRect( 50, 20, 20, 38 ) );
    PlatformPic->setPixmap( image0 );

    GatePic = new QLabel( groupBox12, "GatePic" );
    GatePic->setEnabled( TRUE );
    GatePic->setGeometry( QRect( 40, 60, 41, 26 ) );
    GatePic->setPixmap( image1 );

    GateBtn = new QPushButton( groupBox12, "GateBtn" );
    GateBtn->setEnabled( TRUE );
    GateBtn->setGeometry( QRect( 120, 60, 120, 21 ) );

    PipePic = new QLabel( groupBox12, "PipePic" );
    PipePic->setEnabled( TRUE );
    PipePic->setGeometry( QRect( 40, 90, 40, 20 ) );
    PipePic->setPixmap( image2 );
    PipePic->setScaledContents( TRUE );

    PipeBtn = new QPushButton( groupBox12, "PipeBtn" );
    PipeBtn->setEnabled( TRUE );
    PipeBtn->setGeometry( QRect( 120, 90, 120, 21 ) );

    BuoyPic = new QLabel( groupBox12, "BuoyPic" );
    BuoyPic->setEnabled( TRUE );
    BuoyPic->setGeometry( QRect( 50, 120, 35, 35 ) );
    BuoyPic->setPixmap( image3 );
    BuoyPic->setScaledContents( TRUE );

    FlareBtn = new QPushButton( groupBox12, "FlareBtn" );
    FlareBtn->setEnabled( TRUE );
    FlareBtn->setGeometry( QRect( 120, 120, 120, 21 ) );

    BinPic = new QLabel( groupBox12, "BinPic" );
    BinPic->setEnabled( TRUE );
    BinPic->setGeometry( QRect( 10, 160, 100, 44 ) );
    BinPic->setPixmap( image4 );
    BinPic->setScaledContents( TRUE );

    BarbwirePic = new QLabel( groupBox12, "BarbwirePic" );
    BarbwirePic->setEnabled( TRUE );
    BarbwirePic->setGeometry( QRect( 50, 210, 37, 41 ) );
    BarbwirePic->setPixmap( image5 );
    BarbwirePic->setScaledContents( FALSE );

    MachineGunNestPic = new QLabel( groupBox12, "MachineGunNestPic" );
    MachineGunNestPic->setEnabled( TRUE );
    MachineGunNestPic->setGeometry( QRect( 50, 260, 33, 36 ) );
    MachineGunNestPic->setPixmap( image6 );
    MachineGunNestPic->setScaledContents( TRUE );

    OctagonSurfacePic = new QLabel( groupBox12, "OctagonSurfacePic" );
    OctagonSurfacePic->setEnabled( TRUE );
    OctagonSurfacePic->setGeometry( QRect( 40, 300, 51, 50 ) );
    OctagonSurfacePic->setPixmap( image7 );

    BinBtn = new QPushButton( groupBox12, "BinBtn" );
    BinBtn->setEnabled( TRUE );
    BinBtn->setGeometry( QRect( 120, 161, 120, 30 ) );

    OctagonSurfaceBtn = new QPushButton( groupBox12, "OctagonSurfaceBtn" );
    OctagonSurfaceBtn->setEnabled( TRUE );
    OctagonSurfaceBtn->setGeometry( QRect( 120, 311, 120, 30 ) );

    MachineGunNestBtn = new QPushButton( groupBox12, "MachineGunNestBtn" );
    MachineGunNestBtn->setEnabled( TRUE );
    MachineGunNestBtn->setGeometry( QRect( 120, 260, 120, 30 ) );

    BarbwireBtn = new QPushButton( groupBox12, "BarbwireBtn" );
    BarbwireBtn->setEnabled( TRUE );
    BarbwireBtn->setGeometry( QRect( 120, 210, 120, 30 ) );

    groupBox13 = new QGroupBox( TabPage_2, "groupBox13" );
    groupBox13->setEnabled( TRUE );
    groupBox13->setGeometry( QRect( 420, 360, 270, 150 ) );

    ObjectList = new QComboBox( FALSE, groupBox13, "ObjectList" );
    ObjectList->setEnabled( TRUE );
    ObjectList->setGeometry( QRect( 10, 20, 250, 20 ) );

    textLabel2_3 = new QLabel( groupBox13, "textLabel2_3" );
    textLabel2_3->setEnabled( TRUE );
    textLabel2_3->setGeometry( QRect( 10, 80, 50, 20 ) );

    MapObjectY = new QLineEdit( groupBox13, "MapObjectY" );
    MapObjectY->setEnabled( TRUE );
    MapObjectY->setGeometry( QRect( 70, 80, 56, 21 ) );

    textLabel1_6 = new QLabel( groupBox13, "textLabel1_6" );
    textLabel1_6->setEnabled( TRUE );
    textLabel1_6->setGeometry( QRect( 10, 50, 51, 20 ) );

    textLabel2_4 = new QLabel( groupBox13, "textLabel2_4" );
    textLabel2_4->setGeometry( QRect( 150, 80, 35, 21 ) );

    MapObjectX = new QLineEdit( groupBox13, "MapObjectX" );
    MapObjectX->setEnabled( TRUE );
    MapObjectX->setGeometry( QRect( 70, 50, 56, 21 ) );

    MapObjectVarX = new QLineEdit( groupBox13, "MapObjectVarX" );
    MapObjectVarX->setGeometry( QRect( 201, 50, 60, 21 ) );

    MapObjectVarY = new QLineEdit( groupBox13, "MapObjectVarY" );
    MapObjectVarY->setGeometry( QRect( 201, 80, 60, 21 ) );

    textLabel1_7 = new QLabel( groupBox13, "textLabel1_7" );
    textLabel1_7->setGeometry( QRect( 150, 50, 36, 21 ) );

    DeleteBtn = new QPushButton( groupBox13, "DeleteBtn" );
    DeleteBtn->setEnabled( TRUE );
    DeleteBtn->setGeometry( QRect( 171, 111, 90, 30 ) );

    PlaceBtn = new QPushButton( groupBox13, "PlaceBtn" );
    PlaceBtn->setEnabled( TRUE );
    PlaceBtn->setGeometry( QRect( 40, 110, 90, 30 ) );
    tabWidget3->insertTab( TabPage_2, QString::fromLatin1("") );

    TabPage_3 = new QWidget( tabWidget3, "TabPage_3" );

    BriefcaseFoundBtn = new QPushButton( TabPage_3, "BriefcaseFoundBtn" );
    BriefcaseFoundBtn->setEnabled( TRUE );
    BriefcaseFoundBtn->setGeometry( QRect( 18, 374, 181, 31 ) );

    BombingRunBtn = new QPushButton( TabPage_3, "BombingRunBtn" );
    BombingRunBtn->setEnabled( TRUE );
    BombingRunBtn->setGeometry( QRect( 18, 334, 181, 31 ) );

    ContourFoundBoxesBtn = new QPushButton( TabPage_3, "ContourFoundBoxesBtn" );
    ContourFoundBoxesBtn->setEnabled( TRUE );
    ContourFoundBoxesBtn->setGeometry( QRect( 18, 294, 181, 31 ) );

    BarbwireDoneBtn = new QPushButton( TabPage_3, "BarbwireDoneBtn" );
    BarbwireDoneBtn->setEnabled( TRUE );
    BarbwireDoneBtn->setGeometry( QRect( 18, 254, 181, 31 ) );

    ContourFoundBarbwireBtn = new QPushButton( TabPage_3, "ContourFoundBarbwireBtn" );
    ContourFoundBarbwireBtn->setEnabled( TRUE );
    ContourFoundBarbwireBtn->setGeometry( QRect( 18, 214, 181, 31 ) );

    FlareDoneBtn = new QPushButton( TabPage_3, "FlareDoneBtn" );
    FlareDoneBtn->setEnabled( TRUE );
    FlareDoneBtn->setGeometry( QRect( 18, 174, 181, 31 ) );

    GateFoundBtn = new QPushButton( TabPage_3, "GateFoundBtn" );
    GateFoundBtn->setEnabled( TRUE );
    GateFoundBtn->setGeometry( QRect( 18, 54, 181, 31 ) );

    InitDoneBtn = new QPushButton( TabPage_3, "InitDoneBtn" );
    InitDoneBtn->setEnabled( TRUE );
    InitDoneBtn->setGeometry( QRect( 18, 14, 181, 31 ) );

    GateDoneBtn = new QPushButton( TabPage_3, "GateDoneBtn" );
    GateDoneBtn->setEnabled( TRUE );
    GateDoneBtn->setGeometry( QRect( 18, 94, 181, 31 ) );

    ContourFoundFlareBtn = new QPushButton( TabPage_3, "ContourFoundFlareBtn" );
    ContourFoundFlareBtn->setEnabled( TRUE );
    ContourFoundFlareBtn->setGeometry( QRect( 18, 134, 181, 31 ) );
    tabWidget3->insertTab( TabPage_3, QString::fromLatin1("") );

    TabPage_4 = new QWidget( tabWidget3, "TabPage_4" );

    textLabel1_8 = new QLabel( TabPage_4, "textLabel1_8" );
    textLabel1_8->setGeometry( QRect( 70, 330, 53, 31 ) );

    colorFileLoadText = new QLineEdit( TabPage_4, "colorFileLoadText" );
    colorFileLoadText->setGeometry( QRect( 71, 360, 230, 31 ) );

    ColorPickLoadBut = new QPushButton( TabPage_4, "ColorPickLoadBut" );
    ColorPickLoadBut->setGeometry( QRect( 310, 360, 50, 31 ) );

    textLabel2_3_2_2_5 = new QLabel( TabPage_4, "textLabel2_3_2_2_5" );
    textLabel2_3_2_2_5->setEnabled( TRUE );
    textLabel2_3_2_2_5->setGeometry( QRect( 360, 120, 30, 22 ) );
    QFont textLabel2_3_2_2_5_font(  textLabel2_3_2_2_5->font() );
    textLabel2_3_2_2_5->setFont( textLabel2_3_2_2_5_font );
    textLabel2_3_2_2_5->setAlignment( int( QLabel::AlignVCenter | QLabel::AlignRight ) );

    textLabel2_3_2_2_4 = new QLabel( TabPage_4, "textLabel2_3_2_2_4" );
    textLabel2_3_2_2_4->setEnabled( TRUE );
    textLabel2_3_2_2_4->setGeometry( QRect( 360, 100, 30, 22 ) );
    QFont textLabel2_3_2_2_4_font(  textLabel2_3_2_2_4->font() );
    textLabel2_3_2_2_4->setFont( textLabel2_3_2_2_4_font );
    textLabel2_3_2_2_4->setAlignment( int( QLabel::WordBreak | QLabel::AlignVCenter | QLabel::AlignRight ) );

    textLabel2_3_2_2_4_3 = new QLabel( TabPage_4, "textLabel2_3_2_2_4_3" );
    textLabel2_3_2_2_4_3->setEnabled( TRUE );
    textLabel2_3_2_2_4_3->setGeometry( QRect( 370, 70, 80, 22 ) );
    QFont textLabel2_3_2_2_4_3_font(  textLabel2_3_2_2_4_3->font() );
    textLabel2_3_2_2_4_3->setFont( textLabel2_3_2_2_4_3_font );
    textLabel2_3_2_2_4_3->setAlignment( int( QLabel::WordBreak | QLabel::AlignVCenter | QLabel::AlignRight ) );

    textLabel2_3_2_2_7 = new QLabel( TabPage_4, "textLabel2_3_2_2_7" );
    textLabel2_3_2_2_7->setEnabled( TRUE );
    textLabel2_3_2_2_7->setGeometry( QRect( 370, 160, 20, 22 ) );
    QFont textLabel2_3_2_2_7_font(  textLabel2_3_2_2_7->font() );
    textLabel2_3_2_2_7->setFont( textLabel2_3_2_2_7_font );
    textLabel2_3_2_2_7->setAlignment( int( QLabel::AlignVCenter | QLabel::AlignRight ) );

    textLabel2_3_2_2_6 = new QLabel( TabPage_4, "textLabel2_3_2_2_6" );
    textLabel2_3_2_2_6->setEnabled( TRUE );
    textLabel2_3_2_2_6->setGeometry( QRect( 370, 140, 20, 22 ) );
    QFont textLabel2_3_2_2_6_font(  textLabel2_3_2_2_6->font() );
    textLabel2_3_2_2_6->setFont( textLabel2_3_2_2_6_font );
    textLabel2_3_2_2_6->setAlignment( int( QLabel::AlignVCenter | QLabel::AlignRight ) );

    v_mean_val = new QLineEdit( TabPage_4, "v_mean_val" );
    v_mean_val->setEnabled( TRUE );
    v_mean_val->setGeometry( QRect( 400, 160, 60, 21 ) );

    h1__mean_val = new QLineEdit( TabPage_4, "h1__mean_val" );
    h1__mean_val->setEnabled( TRUE );
    h1__mean_val->setGeometry( QRect( 400, 100, 60, 21 ) );

    h2_mean_val = new QLineEdit( TabPage_4, "h2_mean_val" );
    h2_mean_val->setEnabled( TRUE );
    h2_mean_val->setGeometry( QRect( 400, 120, 60, 21 ) );

    s_mean_val = new QLineEdit( TabPage_4, "s_mean_val" );
    s_mean_val->setEnabled( TRUE );
    s_mean_val->setGeometry( QRect( 400, 140, 60, 21 ) );

    AvgColorImg = new ImageCanvas( TabPage_4, "AvgColorImg" );
    AvgColorImg->setEnabled( TRUE );
    AvgColorImg->setGeometry( QRect( 400, 190, 130, 130 ) );

    textLabel2_3_2_2_4_3_2 = new QLabel( TabPage_4, "textLabel2_3_2_2_4_3_2" );
    textLabel2_3_2_2_4_3_2->setEnabled( TRUE );
    textLabel2_3_2_2_4_3_2->setGeometry( QRect( 470, 70, 60, 22 ) );
    QFont textLabel2_3_2_2_4_3_2_font(  textLabel2_3_2_2_4_3_2->font() );
    textLabel2_3_2_2_4_3_2->setFont( textLabel2_3_2_2_4_3_2_font );
    textLabel2_3_2_2_4_3_2->setAlignment( int( QLabel::WordBreak | QLabel::AlignVCenter | QLabel::AlignRight ) );

    h1_std_val = new QLineEdit( TabPage_4, "h1_std_val" );
    h1_std_val->setEnabled( TRUE );
    h1_std_val->setGeometry( QRect( 470, 100, 60, 21 ) );

    h2_std_val = new QLineEdit( TabPage_4, "h2_std_val" );
    h2_std_val->setEnabled( TRUE );
    h2_std_val->setGeometry( QRect( 470, 120, 60, 21 ) );

    v_std_val = new QLineEdit( TabPage_4, "v_std_val" );
    v_std_val->setEnabled( TRUE );
    v_std_val->setGeometry( QRect( 470, 160, 60, 21 ) );

    s_std_val = new QLineEdit( TabPage_4, "s_std_val" );
    s_std_val->setEnabled( TRUE );
    s_std_val->setGeometry( QRect( 470, 140, 60, 21 ) );

    ColorPickerImg = new ImageCanvas( TabPage_4, "ColorPickerImg" );
    ColorPickerImg->setEnabled( TRUE );
    ColorPickerImg->setGeometry( QRect( 70, 80, 290, 240 ) );
    tabWidget3->insertTab( TabPage_4, QString::fromLatin1("") );

    // toolbars

    languageChange();
    resize( QSize(970, 750).expandedTo(minimumSizeHint()) );
    clearWState( WState_Polished );

    // signals and slots connections
    connect( PlaceBtn, SIGNAL( pressed() ), this, SLOT( moveObject() ) );
    connect( GateFoundBtn, SIGNAL( pressed() ), this, SLOT( sendGateFound() ) );
    connect( GateDoneBtn, SIGNAL( pressed() ), this, SLOT( sendGateDone() ) );
    connect( desired_depth_field, SIGNAL( returnPressed() ), this, SLOT( updateDesiredDepth() ) );
    connect( ImageDisplay1, SIGNAL( mousePressed(int,int,int) ), this, SLOT( ToggleCamera1() ) );
    connect( PipeBtn, SIGNAL( pressed() ), this, SLOT( addPipe() ) );
    connect( GateBtn, SIGNAL( pressed() ), this, SLOT( addGate() ) );
    connect( PlatformBtn, SIGNAL( pressed() ), this, SLOT( addPlatform() ) );
    connect( field_depth_p, SIGNAL( returnPressed() ), this, SLOT( updateDepthPID() ) );
    connect( field_depth_d, SIGNAL( returnPressed() ), this, SLOT( updateDepthPID() ) );
    connect( field_depth_k, SIGNAL( returnPressed() ), this, SLOT( updateDepthPID() ) );
    connect( OctagonSurfaceBtn, SIGNAL( pressed() ), this, SLOT( addOctagonSurface() ) );
    connect( BinBtn, SIGNAL( pressed() ), this, SLOT( addBin() ) );
    connect( ImageDisplay0, SIGNAL( mousePressed(int,int,int) ), this, SLOT( Image1Click(int,int,int) ) );
    connect( ImageDisplay0, SIGNAL( mousePressed(int,int,int) ), this, SLOT( ToggleCamera0() ) );
    connect( field_depth_i, SIGNAL( returnPressed() ), this, SLOT( updateDepthPID() ) );
    connect( field_heading_d, SIGNAL( returnPressed() ), this, SLOT( updateHeadingPID() ) );
    connect( FlareBtn, SIGNAL( pressed() ), this, SLOT( addFlare() ) );
    connect( ContourFoundBoxesBtn, SIGNAL( pressed() ), this, SLOT( sendContourFoundBoxes() ) );
    connect( desired_heading_field, SIGNAL( returnPressed() ), this, SLOT( updateDesiredHeading() ) );
    connect( InitDoneBtn, SIGNAL( pressed() ), this, SLOT( sendInitDone() ) );
    connect( FlareDoneBtn, SIGNAL( pressed() ), this, SLOT( sendFlareDone() ) );
    connect( ContourFoundFlareBtn, SIGNAL( pressed() ), this, SLOT( sendContourFoundFlare() ) );
    connect( radio_auto, SIGNAL( clicked() ), this, SLOT( autoClicked() ) );
    connect( BriefcaseFoundBtn, SIGNAL( pressed() ), this, SLOT( sendBriefcaseFound() ) );
    connect( ContourFoundBarbwireBtn, SIGNAL( pressed() ), this, SLOT( sendContourFoundBarbwire() ) );
    connect( desired_speed_field, SIGNAL( returnPressed() ), this, SLOT( updateDesiredSpeed() ) );
    connect( field_heading_i, SIGNAL( returnPressed() ), this, SLOT( updateHeadingPID() ) );
    connect( ObjectList, SIGNAL( activated(int) ), this, SLOT( selectObject(int) ) );
    connect( MachineGunNestBtn, SIGNAL( pressed() ), this, SLOT( addMachineGunNest() ) );
    connect( radio_manual, SIGNAL( clicked() ), this, SLOT( manualClicked() ) );
    connect( field_heading_p, SIGNAL( returnPressed() ), this, SLOT( updateHeadingPID() ) );
    connect( DeleteBtn, SIGNAL( pressed() ), this, SLOT( deleteObject() ) );
    connect( BombingRunBtn, SIGNAL( pressed() ), this, SLOT( sendBombingRunDone() ) );
    connect( BarbwireDoneBtn, SIGNAL( pressed() ), this, SLOT( sendBarbwireDone() ) );
    connect( SaveMapBtn, SIGNAL( pressed() ), this, SLOT( saveMap() ) );
    connect( LoadMapBtn, SIGNAL( pressed() ), this, SLOT( loadMap() ) );
    connect( ColorPickerImg, SIGNAL( mousePressed(int,int,int) ), this, SLOT( clickColorPickerImg(int,int,int) ) );
}

/*
 *  Destroys the object and frees any allocated resources
 */
SeaBee3MainDisplayForm::~SeaBee3MainDisplayForm()
{
    // no need to delete child widgets, Qt does it all for us
}

/*
 *  Sets the strings of the subwidgets using the current
 *  language.
 */
void SeaBee3MainDisplayForm::languageChange()
{
    setCaption( tr( "SeaBee3 Main Display" ) );
    groupBox9_4->setTitle( tr( "Thruster Current" ) );
    textLabel2->setText( tr( "0" ) );
    textLabel2_2->setText( tr( "1" ) );
    textLabel2_2_2->setText( tr( "2" ) );
    textLabel2_2_3->setText( tr( "3" ) );
    textLabel2_2_4->setText( tr( "4" ) );
    textLabel2_2_5->setText( tr( "5" ) );
    textLabel2_2_6->setText( tr( "6" ) );
    textLabel2_2_7->setText( tr( "7" ) );
    textLabel1_3->setText( tr( ".5A" ) );
    textLabel1_2->setText( tr( "1A" ) );
    textLabel1->setText( tr( "3A" ) );
    groupBox1->setTitle( tr( "Sensor Readings" ) );
    groupBox9_3->setTitle( tr( "Internal Pressure" ) );
    field_int_press->setText( tr( "0" ) );
    ExtPressAuto->setText( tr( "Auto\n"
"Scale" ) );
    groupBox6->setTitle( tr( "Heading" ) );
    field_heading->setText( tr( "0" ) );
    IntPressAuto->setText( tr( "Auto\n"
"Scale" ) );
    groupBox10->setTitle( tr( "External Pressure" ) );
    field_ext_press->setText( tr( "0" ) );
    desired_heading_field_2_3->setText( tr( "0" ) );
    desired_heading_field_2_2->setText( tr( "0" ) );
    groupBox10_2->setTitle( tr( "Controls" ) );
    groupBox2->setTitle( tr( "Pose Settings" ) );
    buttonGroup1->setTitle( tr( "Control" ) );
    radio_manual->setText( tr( "Manual" ) );
    radio_auto->setText( tr( "Auto" ) );
    desired_speed_field->setText( tr( "0" ) );
    desired_depth_field->setText( tr( "0" ) );
    textLabel2_3_2_2->setText( tr( "Desired Heading:" ) );
    textLabel2_2_2_3_2_2->setText( tr( "Desired Depth:" ) );
    textLabel2_2_2_3_2_2_2->setText( tr( "Desired Speed:" ) );
    desired_heading_field->setText( tr( "0" ) );
    groupBox4->setTitle( tr( "PID Output Values" ) );
    textLabel2_3_2_2_2->setText( tr( "Heading Output:" ) );
    textLabel2_3_2_2_2_2->setText( tr( "Depth Output:" ) );
    heading_output_field->setText( tr( "0" ) );
    depth_output_field->setText( tr( "0" ) );
    tabWidget3->changeTab( tab, tr( "Main Display" ) );
    tabWidget3->changeTab( TabPage, tr( "Movement Controller" ) );
    groupBox9_2->setTitle( tr( "Depth Constants" ) );
    textLabel2_3_2_2_3_3_2->setText( tr( "I" ) );
    textLabel2_3_2_2_3_2_2->setText( tr( "P" ) );
    field_depth_i->setText( tr( "0" ) );
    field_depth_d->setText( tr( "0" ) );
    field_depth_k->setText( tr( "0" ) );
    textLabel2_3_2_2_3_2_4_2_2->setText( tr( "K" ) );
    textLabel2_3_2_2_3_4_2->setText( tr( "D" ) );
    field_depth_p->setText( tr( "1" ) );
    groupBox11->setTitle( tr( "Data Logger" ) );
    textLabel1_4_2->setText( tr( "Stop" ) );
    textLabel1_4_3->setText( tr( "Erase" ) );
    textLabel1_4_3_2->setText( tr( "Save" ) );
    textLabel1_4->setText( tr( "Record" ) );
    recordButton->setText( trUtf8( "\xe2\x97\x89" ) );
    stopButton->setText( trUtf8( "\xe2\x96\xa3" ) );
    eraseButton->setText( trUtf8( "\xe2\x9c\x82" ) );
    saveButton->setText( trUtf8( "\xe2\x9c\x8d" ) );
    textLabel1_4_4->setText( tr( "Data Points" ) );
    groupBox9->setTitle( tr( "Heading Constants" ) );
    textLabel2_3_2_2_3_2_4_2->setText( tr( "K" ) );
    textLabel2_3_2_2_3_2->setText( tr( "P" ) );
    textLabel2_3_2_2_3_3->setText( tr( "I" ) );
    textLabel2_3_2_2_3_4->setText( tr( "D" ) );
    field_heading_i->setText( tr( "0" ) );
    field_heading_k->setText( tr( "0" ) );
    field_heading_d->setText( tr( "0" ) );
    field_heading_p->setText( tr( "1" ) );
    tabWidget3->changeTab( tab_2, tr( "PID Tuning" ) );
    groupBox15->setTitle( tr( "Export Map" ) );
    textLabel1_5->setText( tr( "Map Name:" ) );
    SaveMapBtn->setText( tr( "Save Map" ) );
    LoadMapBtn->setText( tr( "Load Map" ) );
    groupBox12->setTitle( tr( "Competition Objects" ) );
    PlatformBtn->setText( tr( "Platform" ) );
    PlatformPic->setText( QString::null );
    GatePic->setText( QString::null );
    GateBtn->setText( tr( "Gate" ) );
    PipePic->setText( QString::null );
    PipeBtn->setText( tr( "Pipe" ) );
    BuoyPic->setText( QString::null );
    FlareBtn->setText( tr( "Flare" ) );
    BinPic->setText( QString::null );
    BarbwirePic->setText( QString::null );
    MachineGunNestPic->setText( QString::null );
    OctagonSurfacePic->setText( QString::null );
    BinBtn->setText( tr( "Bin" ) );
    OctagonSurfaceBtn->setText( tr( "Octagon Surface" ) );
    MachineGunNestBtn->setText( tr( "Machine Gun Nest" ) );
    BarbwireBtn->setText( tr( "Barbwire" ) );
    groupBox13->setTitle( tr( "Placed Objects" ) );
    textLabel2_3->setText( tr( "Y-Coord" ) );
    textLabel1_6->setText( tr( "X-Coord" ) );
    textLabel2_4->setText( tr( "Y-Var" ) );
    MapObjectVarX->setText( QString::null );
    textLabel1_7->setText( tr( "X-Var" ) );
    DeleteBtn->setText( tr( "Delete" ) );
    PlaceBtn->setText( tr( "Update" ) );
    tabWidget3->changeTab( TabPage_2, tr( "Mapping" ) );
    BriefcaseFoundBtn->setText( tr( "BriefcaseFound" ) );
    BombingRunBtn->setText( tr( "BombingRunDone" ) );
    ContourFoundBoxesBtn->setText( tr( "ContourFoundBoxes" ) );
    BarbwireDoneBtn->setText( tr( "BarbwireDone" ) );
    ContourFoundBarbwireBtn->setText( tr( "ContourFoundBarbwire" ) );
    FlareDoneBtn->setText( tr( "FlareDone" ) );
    GateFoundBtn->setText( tr( "GateFound" ) );
    InitDoneBtn->setText( tr( "InitDone" ) );
    GateDoneBtn->setText( tr( "GateDone" ) );
    ContourFoundFlareBtn->setText( tr( "ContourFoundFlare" ) );
    tabWidget3->changeTab( TabPage_3, tr( "SeaBee Injector" ) );
    textLabel1_8->setText( tr( "<h2>File:</h2>" ) );
    ColorPickLoadBut->setText( tr( "Load" ) );
    textLabel2_3_2_2_5->setText( tr( "H2:" ) );
    textLabel2_3_2_2_4->setText( tr( "H1:" ) );
    textLabel2_3_2_2_4_3->setText( tr( "Mean" ) );
    textLabel2_3_2_2_7->setText( tr( "V:" ) );
    textLabel2_3_2_2_6->setText( tr( "S:" ) );
    v_mean_val->setText( tr( "0" ) );
    h1__mean_val->setText( tr( "0" ) );
    h2_mean_val->setText( tr( "0" ) );
    s_mean_val->setText( tr( "0" ) );
    textLabel2_3_2_2_4_3_2->setText( tr( "Std Dev" ) );
    h1_std_val->setText( tr( "0" ) );
    h2_std_val->setText( tr( "0" ) );
    v_std_val->setText( tr( "0" ) );
    s_std_val->setText( tr( "0" ) );
    tabWidget3->changeTab( TabPage_4, tr( "Color Picker" ) );
}

