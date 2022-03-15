

[external]
KcallBC
A
	full_text4
2
0%4 = tail call i64 @_Z13get_global_idj(i32 0) #3
XgetelementptrBG
E
	full_text8
6
4%5 = getelementptr inbounds float, float* %0, i64 %4
"i64B

	full_text


i64 %4
HloadB@
>
	full_text1
/
-%6 = load float, float* %5, align 4, !tbaa !8
(float*B

	full_text

	float* %5
1fmulB)
'
	full_text

%7 = fmul float %6, %2
&floatB

	full_text


float %6
EcallB=
;
	full_text.
,
*%8 = tail call float @_Z3logf(float %7) #3
&floatB

	full_text


float %7
HfdivB@
>
	full_text1
/
-%9 = fdiv float 1.000000e+00, %7, !fpmath !12
&floatB

	full_text


float %7
2fmulB*
(
	full_text

%10 = fmul float %9, %9
&floatB

	full_text


float %9
&floatB

	full_text


float %9
ÄcallBx
v
	full_texti
g
e%11 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0BC54DCA0000000, float 0x40400661E0000000)
&floatB

	full_text


float %9
GcallB?
=
	full_text0
.
,%12 = tail call float @_Z3expf(float %11) #3
'floatB

	full_text

	float %11
YgetelementptrBH
F
	full_text9
7
5%13 = getelementptr inbounds float, float* %1, i64 %4
"i64B

	full_text


i64 %4
JstoreBA
?
	full_text2
0
.store float %12, float* %13, align 4, !tbaa !8
'floatB

	full_text

	float %12
)float*B

	full_text


float* %13
ÄcallBx
v
	full_texti
g
e%14 = tail call float @llvm.fmuladd.f32(float %8, float 0x40055C2900000000, float 0x4025A3BA00000000)
&floatB

	full_text


float %8
qcallBi
g
	full_textZ
X
V%15 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0A8BA7740000000, float %14)
&floatB

	full_text


float %9
'floatB

	full_text

	float %14
GcallB?
=
	full_text0
.
,%16 = tail call float @_Z3expf(float %15) #3
'floatB

	full_text

	float %15
-addB&
$
	full_text

%17 = add i64 %4, 8
"i64B

	full_text


i64 %4
ZgetelementptrBI
G
	full_text:
8
6%18 = getelementptr inbounds float, float* %1, i64 %17
#i64B

	full_text
	
i64 %17
JstoreBA
?
	full_text2
0
.store float %16, float* %18, align 4, !tbaa !8
'floatB

	full_text

	float %16
)float*B

	full_text


float* %18
ÄcallBx
v
	full_texti
g
e%19 = tail call float @llvm.fmuladd.f32(float %8, float 0x3FF828F5C0000000, float 0x403330D780000000)
&floatB

	full_text


float %8
qcallBi
g
	full_textZ
X
V%20 = tail call float @llvm.fmuladd.f32(float %9, float 0xC09AF82200000000, float %19)
&floatB

	full_text


float %9
'floatB

	full_text

	float %19
GcallB?
=
	full_text0
.
,%21 = tail call float @_Z3expf(float %20) #3
'floatB

	full_text

	float %20
.addB'
%
	full_text

%22 = add i64 %4, 16
"i64B

	full_text


i64 %4
ZgetelementptrBI
G
	full_text:
8
6%23 = getelementptr inbounds float, float* %1, i64 %22
#i64B

	full_text
	
i64 %22
JstoreBA
?
	full_text2
0
.store float %21, float* %23, align 4, !tbaa !8
'floatB

	full_text

	float %21
)float*B

	full_text


float* %23
ÄcallBx
v
	full_texti
g
e%24 = tail call float @llvm.fmuladd.f32(float %8, float 0x4003333340000000, float 0x4024F73F80000000)
&floatB

	full_text


float %8
qcallBi
g
	full_textZ
X
V%25 = tail call float @llvm.fmuladd.f32(float %9, float 0x4090972600000000, float %24)
&floatB

	full_text


float %9
'floatB

	full_text

	float %24
GcallB?
=
	full_text0
.
,%26 = tail call float @_Z3expf(float %25) #3
'floatB

	full_text

	float %25
.addB'
%
	full_text

%27 = add i64 %4, 24
"i64B

	full_text


i64 %4
ZgetelementptrBI
G
	full_text:
8
6%28 = getelementptr inbounds float, float* %1, i64 %27
#i64B

	full_text
	
i64 %27
JstoreBA
?
	full_text2
0
.store float %26, float* %28, align 4, !tbaa !8
'floatB

	full_text

	float %26
)float*B

	full_text


float* %28
BfmulB:
8
	full_text+
)
'%29 = fmul float %9, 0x43ABC16D60000000
&floatB

	full_text


float %9
.addB'
%
	full_text

%30 = add i64 %4, 32
"i64B

	full_text


i64 %4
ZgetelementptrBI
G
	full_text:
8
6%31 = getelementptr inbounds float, float* %1, i64 %30
#i64B

	full_text
	
i64 %30
JstoreBA
?
	full_text2
0
.store float %29, float* %31, align 4, !tbaa !8
'floatB

	full_text

	float %29
)float*B

	full_text


float* %31
ÄcallBx
v
	full_texti
g
e%32 = tail call float @llvm.fmuladd.f32(float %8, float 0xBFE3333340000000, float 0x404384F060000000)
&floatB

	full_text


float %8
GcallB?
=
	full_text0
.
,%33 = tail call float @_Z3expf(float %32) #3
'floatB

	full_text

	float %32
.addB'
%
	full_text

%34 = add i64 %4, 40
"i64B

	full_text


i64 %4
ZgetelementptrBI
G
	full_text:
8
6%35 = getelementptr inbounds float, float* %1, i64 %34
#i64B

	full_text
	
i64 %34
JstoreBA
?
	full_text2
0
.store float %33, float* %35, align 4, !tbaa !8
'floatB

	full_text

	float %33
)float*B

	full_text


float* %35
{callBs
q
	full_textd
b
`%36 = tail call float @llvm.fmuladd.f32(float %8, float -1.250000e+00, float 0x4046C53B60000000)
&floatB

	full_text


float %8
GcallB?
=
	full_text0
.
,%37 = tail call float @_Z3expf(float %36) #3
'floatB

	full_text

	float %36
.addB'
%
	full_text

%38 = add i64 %4, 48
"i64B

	full_text


i64 %4
ZgetelementptrBI
G
	full_text:
8
6%39 = getelementptr inbounds float, float* %1, i64 %38
#i64B

	full_text
	
i64 %38
JstoreBA
?
	full_text2
0
.store float %37, float* %39, align 4, !tbaa !8
'floatB

	full_text

	float %37
)float*B

	full_text


float* %39
CfmulB;
9
	full_text,
*
(%40 = fmul float %10, 0x443DD0C880000000
'floatB

	full_text

	float %10
.addB'
%
	full_text

%41 = add i64 %4, 56
"i64B

	full_text


i64 %4
ZgetelementptrBI
G
	full_text:
8
6%42 = getelementptr inbounds float, float* %1, i64 %41
#i64B

	full_text
	
i64 %41
JstoreBA
?
	full_text2
0
.store float %40, float* %42, align 4, !tbaa !8
'floatB

	full_text

	float %40
)float*B

	full_text


float* %42
CfmulB;
9
	full_text,
*
(%43 = fmul float %10, 0x4492A27D60000000
'floatB

	full_text

	float %10
.addB'
%
	full_text

%44 = add i64 %4, 64
"i64B

	full_text


i64 %4
ZgetelementptrBI
G
	full_text:
8
6%45 = getelementptr inbounds float, float* %1, i64 %44
#i64B

	full_text
	
i64 %44
JstoreBA
?
	full_text2
0
.store float %43, float* %45, align 4, !tbaa !8
'floatB

	full_text

	float %43
)float*B

	full_text


float* %45
BfmulB:
8
	full_text+
)
'%46 = fmul float %9, 0x439BC16D60000000
&floatB

	full_text


float %9
.addB'
%
	full_text

%47 = add i64 %4, 72
"i64B

	full_text


i64 %4
ZgetelementptrBI
G
	full_text:
8
6%48 = getelementptr inbounds float, float* %1, i64 %47
#i64B

	full_text
	
i64 %47
JstoreBA
?
	full_text2
0
.store float %46, float* %48, align 4, !tbaa !8
'floatB

	full_text

	float %46
)float*B

	full_text


float* %48
BfmulB:
8
	full_text+
)
'%49 = fmul float %9, 0x437AA535E0000000
&floatB

	full_text


float %9
.addB'
%
	full_text

%50 = add i64 %4, 80
"i64B

	full_text


i64 %4
ZgetelementptrBI
G
	full_text:
8
6%51 = getelementptr inbounds float, float* %1, i64 %50
#i64B

	full_text
	
i64 %50
JstoreBA
?
	full_text2
0
.store float %49, float* %51, align 4, !tbaa !8
'floatB

	full_text

	float %49
)float*B

	full_text


float* %51
ÄcallBx
v
	full_texti
g
e%52 = tail call float @llvm.fmuladd.f32(float %8, float 0xBFEB851EC0000000, float 0x40453CF280000000)
&floatB

	full_text


float %8
GcallB?
=
	full_text0
.
,%53 = tail call float @_Z3expf(float %52) #3
'floatB

	full_text

	float %52
.addB'
%
	full_text

%54 = add i64 %4, 88
"i64B

	full_text


i64 %4
ZgetelementptrBI
G
	full_text:
8
6%55 = getelementptr inbounds float, float* %1, i64 %54
#i64B

	full_text
	
i64 %54
JstoreBA
?
	full_text2
0
.store float %53, float* %55, align 4, !tbaa !8
'floatB

	full_text

	float %53
)float*B

	full_text


float* %55
ÄcallBx
v
	full_texti
g
e%56 = tail call float @llvm.fmuladd.f32(float %8, float 0xBFFB851EC0000000, float 0x4047933D80000000)
&floatB

	full_text


float %8
GcallB?
=
	full_text0
.
,%57 = tail call float @_Z3expf(float %56) #3
'floatB

	full_text

	float %56
.addB'
%
	full_text

%58 = add i64 %4, 96
"i64B

	full_text


i64 %4
ZgetelementptrBI
G
	full_text:
8
6%59 = getelementptr inbounds float, float* %1, i64 %58
#i64B

	full_text
	
i64 %58
JstoreBA
?
	full_text2
0
.store float %57, float* %59, align 4, !tbaa !8
'floatB

	full_text

	float %57
)float*B

	full_text


float* %59
ÄcallBx
v
	full_texti
g
e%60 = tail call float @llvm.fmuladd.f32(float %8, float 0xBFE851EB80000000, float 0x4046202420000000)
&floatB

	full_text


float %8
GcallB?
=
	full_text0
.
,%61 = tail call float @_Z3expf(float %60) #3
'floatB

	full_text

	float %60
/addB(
&
	full_text

%62 = add i64 %4, 104
"i64B

	full_text


i64 %4
ZgetelementptrBI
G
	full_text:
8
6%63 = getelementptr inbounds float, float* %1, i64 %62
#i64B

	full_text
	
i64 %62
JstoreBA
?
	full_text2
0
.store float %61, float* %63, align 4, !tbaa !8
'floatB

	full_text

	float %61
)float*B

	full_text


float* %63
ÄcallBx
v
	full_texti
g
e%64 = tail call float @llvm.fmuladd.f32(float %8, float 0xBFF3D70A40000000, float 0x40465A3140000000)
&floatB

	full_text


float %8
GcallB?
=
	full_text0
.
,%65 = tail call float @_Z3expf(float %64) #3
'floatB

	full_text

	float %64
/addB(
&
	full_text

%66 = add i64 %4, 112
"i64B

	full_text


i64 %4
ZgetelementptrBI
G
	full_text:
8
6%67 = getelementptr inbounds float, float* %1, i64 %66
#i64B

	full_text
	
i64 %66
JstoreBA
?
	full_text2
0
.store float %65, float* %67, align 4, !tbaa !8
'floatB

	full_text

	float %65
)float*B

	full_text


float* %67
ÄcallBx
v
	full_texti
g
e%68 = tail call float @llvm.fmuladd.f32(float %8, float 0xBFD7AE1480000000, float 0x403FEF61C0000000)
&floatB

	full_text


float %8
GcallB?
=
	full_text0
.
,%69 = tail call float @_Z3expf(float %68) #3
'floatB

	full_text

	float %68
/addB(
&
	full_text

%70 = add i64 %4, 120
"i64B

	full_text


i64 %4
ZgetelementptrBI
G
	full_text:
8
6%71 = getelementptr inbounds float, float* %1, i64 %70
#i64B

	full_text
	
i64 %70
JstoreBA
?
	full_text2
0
.store float %69, float* %71, align 4, !tbaa !8
'floatB

	full_text

	float %69
)float*B

	full_text


float* %71
ÄcallBx
v
	full_texti
g
e%72 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0751A88C0000000, float 0x403D028160000000)
&floatB

	full_text


float %9
GcallB?
=
	full_text0
.
,%73 = tail call float @_Z3expf(float %72) #3
'floatB

	full_text

	float %72
/addB(
&
	full_text

%74 = add i64 %4, 128
"i64B

	full_text


i64 %4
ZgetelementptrBI
G
	full_text:
8
6%75 = getelementptr inbounds float, float* %1, i64 %74
#i64B

	full_text
	
i64 %74
JstoreBA
?
	full_text2
0
.store float %73, float* %75, align 4, !tbaa !8
'floatB

	full_text

	float %73
)float*B

	full_text


float* %75
ÄcallBx
v
	full_texti
g
e%76 = tail call float @llvm.fmuladd.f32(float %9, float 0xC079CA33E0000000, float 0x403E70BFA0000000)
&floatB

	full_text


float %9
GcallB?
=
	full_text0
.
,%77 = tail call float @_Z3expf(float %76) #3
'floatB

	full_text

	float %76
/addB(
&
	full_text

%78 = add i64 %4, 136
"i64B

	full_text


i64 %4
ZgetelementptrBI
G
	full_text:
8
6%79 = getelementptr inbounds float, float* %1, i64 %78
#i64B

	full_text
	
i64 %78
JstoreBA
?
	full_text2
0
.store float %77, float* %79, align 4, !tbaa !8
'floatB

	full_text

	float %77
)float*B

	full_text


float* %79
ÄcallBx
v
	full_texti
g
e%80 = tail call float @llvm.fmuladd.f32(float %9, float 0xC062DEE140000000, float 0x403FE410C0000000)
&floatB

	full_text


float %9
GcallB?
=
	full_text0
.
,%81 = tail call float @_Z3expf(float %80) #3
'floatB

	full_text

	float %80
/addB(
&
	full_text

%82 = add i64 %4, 144
"i64B

	full_text


i64 %4
ZgetelementptrBI
G
	full_text:
8
6%83 = getelementptr inbounds float, float* %1, i64 %82
#i64B

	full_text
	
i64 %82
JstoreBA
?
	full_text2
0
.store float %81, float* %83, align 4, !tbaa !8
'floatB

	full_text

	float %81
)float*B

	full_text


float* %83
/addB(
&
	full_text

%84 = add i64 %4, 152
"i64B

	full_text


i64 %4
ZgetelementptrBI
G
	full_text:
8
6%85 = getelementptr inbounds float, float* %1, i64 %84
#i64B

	full_text
	
i64 %84
YstoreBP
N
	full_textA
?
=store float 0x42B2309CE0000000, float* %85, align 4, !tbaa !8
)float*B

	full_text


float* %85
ÄcallBx
v
	full_texti
g
e%86 = tail call float @llvm.fmuladd.f32(float %9, float 0x406F737780000000, float 0x403F77E3E0000000)
&floatB

	full_text


float %9
GcallB?
=
	full_text0
.
,%87 = tail call float @_Z3expf(float %86) #3
'floatB

	full_text

	float %86
/addB(
&
	full_text

%88 = add i64 %4, 160
"i64B

	full_text


i64 %4
ZgetelementptrBI
G
	full_text:
8
6%89 = getelementptr inbounds float, float* %1, i64 %88
#i64B

	full_text
	
i64 %88
JstoreBA
?
	full_text2
0
.store float %87, float* %89, align 4, !tbaa !8
'floatB

	full_text

	float %87
)float*B

	full_text


float* %89
ÄcallBx
v
	full_texti
g
e%90 = tail call float @llvm.fmuladd.f32(float %9, float 0x4089A1F200000000, float 0x4039973EC0000000)
&floatB

	full_text


float %9
GcallB?
=
	full_text0
.
,%91 = tail call float @_Z3expf(float %90) #3
'floatB

	full_text

	float %90
/addB(
&
	full_text

%92 = add i64 %4, 168
"i64B

	full_text


i64 %4
ZgetelementptrBI
G
	full_text:
8
6%93 = getelementptr inbounds float, float* %1, i64 %92
#i64B

	full_text
	
i64 %92
JstoreBA
?
	full_text2
0
.store float %91, float* %93, align 4, !tbaa !8
'floatB

	full_text

	float %91
)float*B

	full_text


float* %93
ÄcallBx
v
	full_texti
g
e%94 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0B79699A0000000, float 0x4040D5EC60000000)
&floatB

	full_text


float %9
GcallB?
=
	full_text0
.
,%95 = tail call float @_Z3expf(float %94) #3
'floatB

	full_text

	float %94
/addB(
&
	full_text

%96 = add i64 %4, 176
"i64B

	full_text


i64 %4
ZgetelementptrBI
G
	full_text:
8
6%97 = getelementptr inbounds float, float* %1, i64 %96
#i64B

	full_text
	
i64 %96
JstoreBA
?
	full_text2
0
.store float %95, float* %97, align 4, !tbaa !8
'floatB

	full_text

	float %95
)float*B

	full_text


float* %97
zcallBr
p
	full_textc
a
_%98 = tail call float @llvm.fmuladd.f32(float %8, float 2.000000e+00, float 0x40304F0800000000)
&floatB

	full_text


float %8
qcallBi
g
	full_textZ
X
V%99 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0A4717400000000, float %98)
&floatB

	full_text


float %9
'floatB

	full_text

	float %98
HcallB@
>
	full_text1
/
-%100 = tail call float @_Z3expf(float %99) #3
'floatB

	full_text

	float %99
0addB)
'
	full_text

%101 = add i64 %4, 184
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%102 = getelementptr inbounds float, float* %1, i64 %101
$i64B

	full_text


i64 %101
LstoreBC
A
	full_text4
2
0store float %100, float* %102, align 4, !tbaa !8
(floatB

	full_text


float %100
*float*B

	full_text

float* %102
ÅcallBy
w
	full_textj
h
f%103 = tail call float @llvm.fmuladd.f32(float %9, float 0xC09C4E51E0000000, float 0x403DEF00E0000000)
&floatB

	full_text


float %9
IcallBA
?
	full_text2
0
.%104 = tail call float @_Z3expf(float %103) #3
(floatB

	full_text


float %103
0addB)
'
	full_text

%105 = add i64 %4, 192
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%106 = getelementptr inbounds float, float* %1, i64 %105
$i64B

	full_text


i64 %105
LstoreBC
A
	full_text4
2
0store float %104, float* %106, align 4, !tbaa !8
(floatB

	full_text


float %104
*float*B

	full_text

float* %106
{callBs
q
	full_textd
b
`%107 = tail call float @llvm.fmuladd.f32(float %8, float 2.000000e+00, float 0x40301494C0000000)
&floatB

	full_text


float %8
scallBk
i
	full_text\
Z
X%108 = tail call float @llvm.fmuladd.f32(float %9, float 0xC09F737780000000, float %107)
&floatB

	full_text


float %9
(floatB

	full_text


float %107
IcallBA
?
	full_text2
0
.%109 = tail call float @_Z3expf(float %108) #3
(floatB

	full_text


float %108
0addB)
'
	full_text

%110 = add i64 %4, 200
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%111 = getelementptr inbounds float, float* %1, i64 %110
$i64B

	full_text


i64 %110
LstoreBC
A
	full_text4
2
0store float %109, float* %111, align 4, !tbaa !8
(floatB

	full_text


float %109
*float*B

	full_text

float* %111
ÅcallBy
w
	full_textj
h
f%112 = tail call float @llvm.fmuladd.f32(float %9, float 0xC06420F040000000, float 0x403C30CDA0000000)
&floatB

	full_text


float %9
IcallBA
?
	full_text2
0
.%113 = tail call float @_Z3expf(float %112) #3
(floatB

	full_text


float %112
0addB)
'
	full_text

%114 = add i64 %4, 208
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%115 = getelementptr inbounds float, float* %1, i64 %114
$i64B

	full_text


i64 %114
LstoreBC
A
	full_text4
2
0store float %113, float* %115, align 4, !tbaa !8
(floatB

	full_text


float %113
*float*B

	full_text

float* %115
ÅcallBy
w
	full_textj
h
f%116 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0B2CAC060000000, float 0x4040FF3D00000000)
&floatB

	full_text


float %9
IcallBA
?
	full_text2
0
.%117 = tail call float @_Z3expf(float %116) #3
(floatB

	full_text


float %116
0addB)
'
	full_text

%118 = add i64 %4, 216
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%119 = getelementptr inbounds float, float* %1, i64 %118
$i64B

	full_text


i64 %118
LstoreBC
A
	full_text4
2
0store float %117, float* %119, align 4, !tbaa !8
(floatB

	full_text


float %117
*float*B

	full_text

float* %119
ÅcallBy
w
	full_textj
h
f%120 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0979699A0000000, float 0x40410400E0000000)
&floatB

	full_text


float %9
IcallBA
?
	full_text2
0
.%121 = tail call float @_Z3expf(float %120) #3
(floatB

	full_text


float %120
0addB)
'
	full_text

%122 = add i64 %4, 224
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%123 = getelementptr inbounds float, float* %1, i64 %122
$i64B

	full_text


i64 %122
LstoreBC
A
	full_text4
2
0store float %121, float* %123, align 4, !tbaa !8
(floatB

	full_text


float %121
*float*B

	full_text

float* %123
ÅcallBy
w
	full_textj
h
f%124 = tail call float @llvm.fmuladd.f32(float %8, float 0x3FF3A5E360000000, float 0x4031ADA7E0000000)
&floatB

	full_text


float %8
scallBk
i
	full_text\
Z
X%125 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0419CD240000000, float %124)
&floatB

	full_text


float %9
(floatB

	full_text


float %124
IcallBA
?
	full_text2
0
.%126 = tail call float @_Z3expf(float %125) #3
(floatB

	full_text


float %125
0addB)
'
	full_text

%127 = add i64 %4, 232
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%128 = getelementptr inbounds float, float* %1, i64 %127
$i64B

	full_text


i64 %127
LstoreBC
A
	full_text4
2
0store float %126, float* %128, align 4, !tbaa !8
(floatB

	full_text


float %126
*float*B

	full_text

float* %128
{callBs
q
	full_textd
b
`%129 = tail call float @llvm.fmuladd.f32(float %8, float 1.500000e+00, float 0x403193A340000000)
&floatB

	full_text


float %8
scallBk
i
	full_text\
Z
X%130 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0E38F0180000000, float %129)
&floatB

	full_text


float %9
(floatB

	full_text


float %129
IcallBA
?
	full_text2
0
.%131 = tail call float @_Z3expf(float %130) #3
(floatB

	full_text


float %130
0addB)
'
	full_text

%132 = add i64 %4, 240
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%133 = getelementptr inbounds float, float* %1, i64 %132
$i64B

	full_text


i64 %132
LstoreBC
A
	full_text4
2
0store float %131, float* %133, align 4, !tbaa !8
(floatB

	full_text


float %131
*float*B

	full_text

float* %133
ÅcallBy
w
	full_textj
h
f%134 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0D77D7060000000, float 0x403C8C1CA0000000)
&floatB

	full_text


float %9
IcallBA
?
	full_text2
0
.%135 = tail call float @_Z3expf(float %134) #3
(floatB

	full_text


float %134
0addB)
'
	full_text

%136 = add i64 %4, 248
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%137 = getelementptr inbounds float, float* %1, i64 %136
$i64B

	full_text


i64 %136
LstoreBC
A
	full_text4
2
0store float %135, float* %137, align 4, !tbaa !8
(floatB

	full_text


float %135
*float*B

	full_text

float* %137
ÅcallBy
w
	full_textj
h
f%138 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0C731F4E0000000, float 0x40405221C0000000)
&floatB

	full_text


float %9
IcallBA
?
	full_text2
0
.%139 = tail call float @_Z3expf(float %138) #3
(floatB

	full_text


float %138
0addB)
'
	full_text

%140 = add i64 %4, 256
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%141 = getelementptr inbounds float, float* %1, i64 %140
$i64B

	full_text


i64 %140
LstoreBC
A
	full_text4
2
0store float %139, float* %141, align 4, !tbaa !8
(floatB

	full_text


float %139
*float*B

	full_text

float* %141
0addB)
'
	full_text

%142 = add i64 %4, 264
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%143 = getelementptr inbounds float, float* %1, i64 %142
$i64B

	full_text


i64 %142
ZstoreBQ
O
	full_textB
@
>store float 0x42C9EBAC60000000, float* %143, align 4, !tbaa !8
*float*B

	full_text

float* %143
0addB)
'
	full_text

%144 = add i64 %4, 272
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%145 = getelementptr inbounds float, float* %1, i64 %144
$i64B

	full_text


i64 %144
ZstoreBQ
O
	full_textB
@
>store float 0x42BB48EB60000000, float* %145, align 4, !tbaa !8
*float*B

	full_text

float* %145
ÅcallBy
w
	full_textj
h
f%146 = tail call float @llvm.fmuladd.f32(float %8, float 0x3FFCA3D700000000, float 0x403285B7C0000000)
&floatB

	full_text


float %8
scallBk
i
	full_text\
Z
X%147 = tail call float @llvm.fmuladd.f32(float %9, float 0xC08A42F980000000, float %146)
&floatB

	full_text


float %9
(floatB

	full_text


float %146
IcallBA
?
	full_text2
0
.%148 = tail call float @_Z3expf(float %147) #3
(floatB

	full_text


float %147
0addB)
'
	full_text

%149 = add i64 %4, 280
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%150 = getelementptr inbounds float, float* %1, i64 %149
$i64B

	full_text


i64 %149
LstoreBC
A
	full_text4
2
0store float %148, float* %150, align 4, !tbaa !8
(floatB

	full_text


float %148
*float*B

	full_text

float* %150
ÅcallBy
w
	full_textj
h
f%151 = tail call float @llvm.fmuladd.f32(float %9, float 0x4077BEDB80000000, float 0x403D5F8CA0000000)
&floatB

	full_text


float %9
IcallBA
?
	full_text2
0
.%152 = tail call float @_Z3expf(float %151) #3
(floatB

	full_text


float %151
0addB)
'
	full_text

%153 = add i64 %4, 288
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%154 = getelementptr inbounds float, float* %1, i64 %153
$i64B

	full_text


i64 %153
LstoreBC
A
	full_text4
2
0store float %152, float* %154, align 4, !tbaa !8
(floatB

	full_text


float %152
*float*B

	full_text

float* %154
0addB)
'
	full_text

%155 = add i64 %4, 296
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%156 = getelementptr inbounds float, float* %1, i64 %155
$i64B

	full_text


i64 %155
ZstoreBQ
O
	full_textB
@
>store float 0x42BE036940000000, float* %156, align 4, !tbaa !8
*float*B

	full_text

float* %156
0addB)
'
	full_text

%157 = add i64 %4, 304
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%158 = getelementptr inbounds float, float* %1, i64 %157
$i64B

	full_text


i64 %157
ZstoreBQ
O
	full_textB
@
>store float 0x42C6BCC420000000, float* %158, align 4, !tbaa !8
*float*B

	full_text

float* %158
ÅcallBy
w
	full_textj
h
f%159 = tail call float @llvm.fmuladd.f32(float %9, float 0xC075B38320000000, float 0x403CDAD400000000)
&floatB

	full_text


float %9
IcallBA
?
	full_text2
0
.%160 = tail call float @_Z3expf(float %159) #3
(floatB

	full_text


float %159
0addB)
'
	full_text

%161 = add i64 %4, 312
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%162 = getelementptr inbounds float, float* %1, i64 %161
$i64B

	full_text


i64 %161
LstoreBC
A
	full_text4
2
0store float %160, float* %162, align 4, !tbaa !8
(floatB

	full_text


float %160
*float*B

	full_text

float* %162
ÅcallBy
w
	full_textj
h
f%163 = tail call float @llvm.fmuladd.f32(float %8, float 0x3FDEB851E0000000, float 0x403BB79A60000000)
&floatB

	full_text


float %8
scallBk
i
	full_text\
Z
X%164 = tail call float @llvm.fmuladd.f32(float %9, float 0x40605AC340000000, float %163)
&floatB

	full_text


float %9
(floatB

	full_text


float %163
IcallBA
?
	full_text2
0
.%165 = tail call float @_Z3expf(float %164) #3
(floatB

	full_text


float %164
0addB)
'
	full_text

%166 = add i64 %4, 320
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%167 = getelementptr inbounds float, float* %1, i64 %166
$i64B

	full_text


i64 %166
LstoreBC
A
	full_text4
2
0store float %165, float* %167, align 4, !tbaa !8
(floatB

	full_text


float %165
*float*B

	full_text

float* %167
0addB)
'
	full_text

%168 = add i64 %4, 328
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%169 = getelementptr inbounds float, float* %1, i64 %168
$i64B

	full_text


i64 %168
ZstoreBQ
O
	full_textB
@
>store float 0x42D0B07140000000, float* %169, align 4, !tbaa !8
*float*B

	full_text

float* %169
0addB)
'
	full_text

%170 = add i64 %4, 336
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%171 = getelementptr inbounds float, float* %1, i64 %170
$i64B

	full_text


i64 %170
ZstoreBQ
O
	full_textB
@
>store float 0x42BB48EB60000000, float* %171, align 4, !tbaa !8
*float*B

	full_text

float* %171
0addB)
'
	full_text

%172 = add i64 %4, 344
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%173 = getelementptr inbounds float, float* %1, i64 %172
$i64B

	full_text


i64 %172
ZstoreBQ
O
	full_textB
@
>store float 0x42BB48EB60000000, float* %173, align 4, !tbaa !8
*float*B

	full_text

float* %173
0addB)
'
	full_text

%174 = add i64 %4, 352
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%175 = getelementptr inbounds float, float* %1, i64 %174
$i64B

	full_text


i64 %174
ZstoreBQ
O
	full_textB
@
>store float 0x42C6BCC420000000, float* %175, align 4, !tbaa !8
*float*B

	full_text

float* %175
|callBt
r
	full_texte
c
a%176 = tail call float @llvm.fmuladd.f32(float %8, float -1.000000e+00, float 0x4043E28BA0000000)
&floatB

	full_text


float %8
scallBk
i
	full_text\
Z
X%177 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0C0B55780000000, float %176)
&floatB

	full_text


float %9
(floatB

	full_text


float %176
IcallBA
?
	full_text2
0
.%178 = tail call float @_Z3expf(float %177) #3
(floatB

	full_text


float %177
0addB)
'
	full_text

%179 = add i64 %4, 360
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%180 = getelementptr inbounds float, float* %1, i64 %179
$i64B

	full_text


i64 %179
LstoreBC
A
	full_text4
2
0store float %178, float* %180, align 4, !tbaa !8
(floatB

	full_text


float %178
*float*B

	full_text

float* %180
ÅcallBy
w
	full_textj
h
f%181 = tail call float @llvm.fmuladd.f32(float %9, float 0xC069292C60000000, float 0x403DA8BF60000000)
&floatB

	full_text


float %9
IcallBA
?
	full_text2
0
.%182 = tail call float @_Z3expf(float %181) #3
(floatB

	full_text


float %181
0addB)
'
	full_text

%183 = add i64 %4, 368
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%184 = getelementptr inbounds float, float* %1, i64 %183
$i64B

	full_text


i64 %183
LstoreBC
A
	full_text4
2
0store float %182, float* %184, align 4, !tbaa !8
(floatB

	full_text


float %182
*float*B

	full_text

float* %184
ÅcallBy
w
	full_textj
h
f%185 = tail call float @llvm.fmuladd.f32(float %8, float 0xBFE99999A0000000, float 0x4042E0FAC0000000)
&floatB

	full_text


float %8
IcallBA
?
	full_text2
0
.%186 = tail call float @_Z3expf(float %185) #3
(floatB

	full_text


float %185
0addB)
'
	full_text

%187 = add i64 %4, 376
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%188 = getelementptr inbounds float, float* %1, i64 %187
$i64B

	full_text


i64 %187
LstoreBC
A
	full_text4
2
0store float %186, float* %188, align 4, !tbaa !8
(floatB

	full_text


float %186
*float*B

	full_text

float* %188
{callBs
q
	full_textd
b
`%189 = tail call float @llvm.fmuladd.f32(float %8, float 2.000000e+00, float 0x402A3EA660000000)
&floatB

	full_text


float %8
scallBk
i
	full_text\
Z
X%190 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0AC6C8360000000, float %189)
&floatB

	full_text


float %9
(floatB

	full_text


float %189
IcallBA
?
	full_text2
0
.%191 = tail call float @_Z3expf(float %190) #3
(floatB

	full_text


float %190
0addB)
'
	full_text

%192 = add i64 %4, 384
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%193 = getelementptr inbounds float, float* %1, i64 %192
$i64B

	full_text


i64 %192
LstoreBC
A
	full_text4
2
0store float %191, float* %193, align 4, !tbaa !8
(floatB

	full_text


float %191
*float*B

	full_text

float* %193
0addB)
'
	full_text

%194 = add i64 %4, 392
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%195 = getelementptr inbounds float, float* %1, i64 %194
$i64B

	full_text


i64 %194
ZstoreBQ
O
	full_textB
@
>store float 0x42D2309CE0000000, float* %195, align 4, !tbaa !8
*float*B

	full_text

float* %195
CfmulB;
9
	full_text,
*
(%196 = fmul float %9, 0xC0879699A0000000
&floatB

	full_text


float %9
IcallBA
?
	full_text2
0
.%197 = tail call float @_Z3expf(float %196) #3
(floatB

	full_text


float %196
EfmulB=
;
	full_text.
,
*%198 = fmul float %197, 0x42A3356220000000
(floatB

	full_text


float %197
0addB)
'
	full_text

%199 = add i64 %4, 400
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%200 = getelementptr inbounds float, float* %1, i64 %199
$i64B

	full_text


i64 %199
LstoreBC
A
	full_text4
2
0store float %198, float* %200, align 4, !tbaa !8
(floatB

	full_text


float %198
*float*B

	full_text

float* %200
EfmulB=
;
	full_text.
,
*%201 = fmul float %197, 0x4283356220000000
(floatB

	full_text


float %197
0addB)
'
	full_text

%202 = add i64 %4, 408
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%203 = getelementptr inbounds float, float* %1, i64 %202
$i64B

	full_text


i64 %202
LstoreBC
A
	full_text4
2
0store float %201, float* %203, align 4, !tbaa !8
(floatB

	full_text


float %201
*float*B

	full_text

float* %203
0addB)
'
	full_text

%204 = add i64 %4, 416
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%205 = getelementptr inbounds float, float* %1, i64 %204
$i64B

	full_text


i64 %204
ZstoreBQ
O
	full_textB
@
>store float 0x42B2309CE0000000, float* %205, align 4, !tbaa !8
*float*B

	full_text

float* %205
{callBs
q
	full_textd
b
`%206 = tail call float @llvm.fmuladd.f32(float %8, float 2.000000e+00, float 0x40303D8520000000)
&floatB

	full_text


float %8
scallBk
i
	full_text\
Z
X%207 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0979699A0000000, float %206)
&floatB

	full_text


float %9
(floatB

	full_text


float %206
IcallBA
?
	full_text2
0
.%208 = tail call float @_Z3expf(float %207) #3
(floatB

	full_text


float %207
0addB)
'
	full_text

%209 = add i64 %4, 424
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%210 = getelementptr inbounds float, float* %1, i64 %209
$i64B

	full_text


i64 %209
LstoreBC
A
	full_text4
2
0store float %208, float* %210, align 4, !tbaa !8
(floatB

	full_text


float %208
*float*B

	full_text

float* %210
0addB)
'
	full_text

%211 = add i64 %4, 432
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%212 = getelementptr inbounds float, float* %1, i64 %211
$i64B

	full_text


i64 %211
ZstoreBQ
O
	full_textB
@
>store float 0x42B2309CE0000000, float* %212, align 4, !tbaa !8
*float*B

	full_text

float* %212
{callBs
q
	full_textd
b
`%213 = tail call float @llvm.fmuladd.f32(float %8, float 5.000000e-01, float 0x403B6B98C0000000)
&floatB

	full_text


float %8
scallBk
i
	full_text\
Z
X%214 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0A1BB03A0000000, float %213)
&floatB

	full_text


float %9
(floatB

	full_text


float %213
IcallBA
?
	full_text2
0
.%215 = tail call float @_Z3expf(float %214) #3
(floatB

	full_text


float %214
0addB)
'
	full_text

%216 = add i64 %4, 440
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%217 = getelementptr inbounds float, float* %1, i64 %216
$i64B

	full_text


i64 %216
LstoreBC
A
	full_text4
2
0store float %215, float* %217, align 4, !tbaa !8
(floatB

	full_text


float %215
*float*B

	full_text

float* %217
0addB)
'
	full_text

%218 = add i64 %4, 448
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%219 = getelementptr inbounds float, float* %1, i64 %218
$i64B

	full_text


i64 %218
ZstoreBQ
O
	full_textB
@
>store float 0x42C2309CE0000000, float* %219, align 4, !tbaa !8
*float*B

	full_text

float* %219
0addB)
'
	full_text

%220 = add i64 %4, 456
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%221 = getelementptr inbounds float, float* %1, i64 %220
$i64B

	full_text


i64 %220
ZstoreBQ
O
	full_textB
@
>store float 0x42BD1A94A0000000, float* %221, align 4, !tbaa !8
*float*B

	full_text

float* %221
ÅcallBy
w
	full_textj
h
f%222 = tail call float @llvm.fmuladd.f32(float %9, float 0xC072DEE140000000, float 0x403E56CD60000000)
&floatB

	full_text


float %9
IcallBA
?
	full_text2
0
.%223 = tail call float @_Z3expf(float %222) #3
(floatB

	full_text


float %222
0addB)
'
	full_text

%224 = add i64 %4, 464
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%225 = getelementptr inbounds float, float* %1, i64 %224
$i64B

	full_text


i64 %224
LstoreBC
A
	full_text4
2
0store float %223, float* %225, align 4, !tbaa !8
(floatB

	full_text


float %223
*float*B

	full_text

float* %225
0addB)
'
	full_text

%226 = add i64 %4, 472
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%227 = getelementptr inbounds float, float* %1, i64 %226
$i64B

	full_text


i64 %226
ZstoreBQ
O
	full_textB
@
>store float 0x42BB48EB60000000, float* %227, align 4, !tbaa !8
*float*B

	full_text

float* %227
0addB)
'
	full_text

%228 = add i64 %4, 480
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%229 = getelementptr inbounds float, float* %1, i64 %228
$i64B

	full_text


i64 %228
ZstoreBQ
O
	full_textB
@
>store float 0x42AB48EB60000000, float* %229, align 4, !tbaa !8
*float*B

	full_text

float* %229
0addB)
'
	full_text

%230 = add i64 %4, 488
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%231 = getelementptr inbounds float, float* %1, i64 %230
$i64B

	full_text


i64 %230
ZstoreBQ
O
	full_textB
@
>store float 0x42AB48EB60000000, float* %231, align 4, !tbaa !8
*float*B

	full_text

float* %231
0addB)
'
	full_text

%232 = add i64 %4, 496
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%233 = getelementptr inbounds float, float* %1, i64 %232
$i64B

	full_text


i64 %232
ZstoreBQ
O
	full_textB
@
>store float 0x42BB48EB60000000, float* %233, align 4, !tbaa !8
*float*B

	full_text

float* %233
0addB)
'
	full_text

%234 = add i64 %4, 504
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%235 = getelementptr inbounds float, float* %1, i64 %234
$i64B

	full_text


i64 %234
ZstoreBQ
O
	full_textB
@
>store float 0x42CFD512A0000000, float* %235, align 4, !tbaa !8
*float*B

	full_text

float* %235
0addB)
'
	full_text

%236 = add i64 %4, 512
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%237 = getelementptr inbounds float, float* %1, i64 %236
$i64B

	full_text


i64 %236
ZstoreBQ
O
	full_textB
@
>store float 0x42B9774200000000, float* %237, align 4, !tbaa !8
*float*B

	full_text

float* %237
0addB)
'
	full_text

%238 = add i64 %4, 520
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%239 = getelementptr inbounds float, float* %1, i64 %238
$i64B

	full_text


i64 %238
ZstoreBQ
O
	full_textB
@
>store float 0x42A5D3EF80000000, float* %239, align 4, !tbaa !8
*float*B

	full_text

float* %239
0addB)
'
	full_text

%240 = add i64 %4, 528
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%241 = getelementptr inbounds float, float* %1, i64 %240
$i64B

	full_text


i64 %240
ZstoreBQ
O
	full_textB
@
>store float 0x42BB48EB60000000, float* %241, align 4, !tbaa !8
*float*B

	full_text

float* %241
0addB)
'
	full_text

%242 = add i64 %4, 536
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%243 = getelementptr inbounds float, float* %1, i64 %242
$i64B

	full_text


i64 %242
ZstoreBQ
O
	full_textB
@
>store float 0x42A05EF3A0000000, float* %243, align 4, !tbaa !8
*float*B

	full_text

float* %243
0addB)
'
	full_text

%244 = add i64 %4, 544
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%245 = getelementptr inbounds float, float* %1, i64 %244
$i64B

	full_text


i64 %244
ZstoreBQ
O
	full_textB
@
>store float 0x4299774200000000, float* %245, align 4, !tbaa !8
*float*B

	full_text

float* %245
0addB)
'
	full_text

%246 = add i64 %4, 552
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%247 = getelementptr inbounds float, float* %1, i64 %246
$i64B

	full_text


i64 %246
ZstoreBQ
O
	full_textB
@
>store float 0x42A9774200000000, float* %247, align 4, !tbaa !8
*float*B

	full_text

float* %247
ÅcallBy
w
	full_textj
h
f%248 = tail call float @llvm.fmuladd.f32(float %8, float 0x3FDD0E5600000000, float 0x403B03CC40000000)
&floatB

	full_text


float %8
scallBk
i
	full_text\
Z
X%249 = tail call float @llvm.fmuladd.f32(float %9, float 0xC094717400000000, float %248)
&floatB

	full_text


float %9
(floatB

	full_text


float %248
IcallBA
?
	full_text2
0
.%250 = tail call float @_Z3expf(float %249) #3
(floatB

	full_text


float %249
0addB)
'
	full_text

%251 = add i64 %4, 560
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%252 = getelementptr inbounds float, float* %1, i64 %251
$i64B

	full_text


i64 %251
LstoreBC
A
	full_text4
2
0store float %250, float* %252, align 4, !tbaa !8
(floatB

	full_text


float %250
*float*B

	full_text

float* %252
ÅcallBy
w
	full_textj
h
f%253 = tail call float @llvm.fmuladd.f32(float %8, float 0x3FF0CCCCC0000000, float 0x4037DBD7C0000000)
&floatB

	full_text


float %8
scallBk
i
	full_text\
Z
X%254 = tail call float @llvm.fmuladd.f32(float %9, float 0xC099C02360000000, float %253)
&floatB

	full_text


float %9
(floatB

	full_text


float %253
IcallBA
?
	full_text2
0
.%255 = tail call float @_Z3expf(float %254) #3
(floatB

	full_text


float %254
0addB)
'
	full_text

%256 = add i64 %4, 568
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%257 = getelementptr inbounds float, float* %1, i64 %256
$i64B

	full_text


i64 %256
LstoreBC
A
	full_text4
2
0store float %255, float* %257, align 4, !tbaa !8
(floatB

	full_text


float %255
*float*B

	full_text

float* %257
ÅcallBy
w
	full_textj
h
f%258 = tail call float @llvm.fmuladd.f32(float %9, float 0xC09BD58C40000000, float 0x403F4B69C0000000)
&floatB

	full_text


float %9
IcallBA
?
	full_text2
0
.%259 = tail call float @_Z3expf(float %258) #3
(floatB

	full_text


float %258
0addB)
'
	full_text

%260 = add i64 %4, 576
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%261 = getelementptr inbounds float, float* %1, i64 %260
$i64B

	full_text


i64 %260
LstoreBC
A
	full_text4
2
0store float %259, float* %261, align 4, !tbaa !8
(floatB

	full_text


float %259
*float*B

	full_text

float* %261
ÅcallBy
w
	full_textj
h
f%262 = tail call float @llvm.fmuladd.f32(float %8, float 0x3FF2E147A0000000, float 0x4035F4B100000000)
&floatB

	full_text


float %8
scallBk
i
	full_text\
Z
X%263 = tail call float @llvm.fmuladd.f32(float %9, float 0x406C1E02E0000000, float %262)
&floatB

	full_text


float %9
(floatB

	full_text


float %262
IcallBA
?
	full_text2
0
.%264 = tail call float @_Z3expf(float %263) #3
(floatB

	full_text


float %263
0addB)
'
	full_text

%265 = add i64 %4, 584
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%266 = getelementptr inbounds float, float* %1, i64 %265
$i64B

	full_text


i64 %265
LstoreBC
A
	full_text4
2
0store float %264, float* %266, align 4, !tbaa !8
(floatB

	full_text


float %264
*float*B

	full_text

float* %266
ÅcallBy
w
	full_textj
h
f%267 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0D3A82AA0000000, float 0x40401E3B80000000)
&floatB

	full_text


float %9
IcallBA
?
	full_text2
0
.%268 = tail call float @_Z3expf(float %267) #3
(floatB

	full_text


float %267
0addB)
'
	full_text

%269 = add i64 %4, 592
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%270 = getelementptr inbounds float, float* %1, i64 %269
$i64B

	full_text


i64 %269
LstoreBC
A
	full_text4
2
0store float %268, float* %270, align 4, !tbaa !8
(floatB

	full_text


float %268
*float*B

	full_text

float* %270
CfmulB;
9
	full_text,
*
(%271 = fmul float %9, 0xC0AF737780000000
&floatB

	full_text


float %9
IcallBA
?
	full_text2
0
.%272 = tail call float @_Z3expf(float %271) #3
(floatB

	full_text


float %271
EfmulB=
;
	full_text.
,
*%273 = fmul float %272, 0x426D1A94A0000000
(floatB

	full_text


float %272
0addB)
'
	full_text

%274 = add i64 %4, 600
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%275 = getelementptr inbounds float, float* %1, i64 %274
$i64B

	full_text


i64 %274
LstoreBC
A
	full_text4
2
0store float %273, float* %275, align 4, !tbaa !8
(floatB

	full_text


float %273
*float*B

	full_text

float* %275
EfmulB=
;
	full_text.
,
*%276 = fmul float %272, 0x42C6BCC420000000
(floatB

	full_text


float %272
1addB*
(
	full_text

%277 = add i64 %4, 1008
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%278 = getelementptr inbounds float, float* %1, i64 %277
$i64B

	full_text


i64 %277
LstoreBC
A
	full_text4
2
0store float %276, float* %278, align 4, !tbaa !8
(floatB

	full_text


float %276
*float*B

	full_text

float* %278
EfmulB=
;
	full_text.
,
*%279 = fmul float %272, 0x42A2309CE0000000
(floatB

	full_text


float %272
1addB*
(
	full_text

%280 = add i64 %4, 1024
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%281 = getelementptr inbounds float, float* %1, i64 %280
$i64B

	full_text


i64 %280
LstoreBC
A
	full_text4
2
0store float %279, float* %281, align 4, !tbaa !8
(floatB

	full_text


float %279
*float*B

	full_text

float* %281
ÅcallBy
w
	full_textj
h
f%282 = tail call float @llvm.fmuladd.f32(float %9, float 0x4070328160000000, float 0x4040172080000000)
&floatB

	full_text


float %9
IcallBA
?
	full_text2
0
.%283 = tail call float @_Z3expf(float %282) #3
(floatB

	full_text


float %282
0addB)
'
	full_text

%284 = add i64 %4, 608
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%285 = getelementptr inbounds float, float* %1, i64 %284
$i64B

	full_text


i64 %284
LstoreBC
A
	full_text4
2
0store float %283, float* %285, align 4, !tbaa !8
(floatB

	full_text


float %283
*float*B

	full_text

float* %285
ÅcallBy
w
	full_textj
h
f%286 = tail call float @llvm.fmuladd.f32(float %8, float 0xBFE428F5C0000000, float 0x40428A49E0000000)
&floatB

	full_text


float %8
scallBk
i
	full_text\
Z
X%287 = tail call float @llvm.fmuladd.f32(float %9, float 0xC068176C60000000, float %286)
&floatB

	full_text


float %9
(floatB

	full_text


float %286
IcallBA
?
	full_text2
0
.%288 = tail call float @_Z3expf(float %287) #3
(floatB

	full_text


float %287
0addB)
'
	full_text

%289 = add i64 %4, 616
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%290 = getelementptr inbounds float, float* %1, i64 %289
$i64B

	full_text


i64 %289
LstoreBC
A
	full_text4
2
0store float %288, float* %290, align 4, !tbaa !8
(floatB

	full_text


float %288
*float*B

	full_text

float* %290
0addB)
'
	full_text

%291 = add i64 %4, 624
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%292 = getelementptr inbounds float, float* %1, i64 %291
$i64B

	full_text


i64 %291
ZstoreBQ
O
	full_textB
@
>store float 0x42D32AE7E0000000, float* %292, align 4, !tbaa !8
*float*B

	full_text

float* %292
ÅcallBy
w
	full_textj
h
f%293 = tail call float @llvm.fmuladd.f32(float %8, float 0x3FF99999A0000000, float 0x4031D742C0000000)
&floatB

	full_text


float %8
scallBk
i
	full_text\
Z
X%294 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0A54EDE60000000, float %293)
&floatB

	full_text


float %9
(floatB

	full_text


float %293
IcallBA
?
	full_text2
0
.%295 = tail call float @_Z3expf(float %294) #3
(floatB

	full_text


float %294
0addB)
'
	full_text

%296 = add i64 %4, 632
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%297 = getelementptr inbounds float, float* %1, i64 %296
$i64B

	full_text


i64 %296
LstoreBC
A
	full_text4
2
0store float %295, float* %297, align 4, !tbaa !8
(floatB

	full_text


float %295
*float*B

	full_text

float* %297
0addB)
'
	full_text

%298 = add i64 %4, 640
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%299 = getelementptr inbounds float, float* %1, i64 %298
$i64B

	full_text


i64 %298
ZstoreBQ
O
	full_textB
@
>store float 0x42B6BF1820000000, float* %299, align 4, !tbaa !8
*float*B

	full_text

float* %299
ÅcallBy
w
	full_textj
h
f%300 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0CC4E51E0000000, float 0x403F0F3C00000000)
&floatB

	full_text


float %9
IcallBA
?
	full_text2
0
.%301 = tail call float @_Z3expf(float %300) #3
(floatB

	full_text


float %300
0addB)
'
	full_text

%302 = add i64 %4, 648
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%303 = getelementptr inbounds float, float* %1, i64 %302
$i64B

	full_text


i64 %302
LstoreBC
A
	full_text4
2
0store float %301, float* %303, align 4, !tbaa !8
(floatB

	full_text


float %301
*float*B

	full_text

float* %303
ÅcallBy
w
	full_textj
h
f%304 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0B192C1C0000000, float 0x40384E8980000000)
&floatB

	full_text


float %9
IcallBA
?
	full_text2
0
.%305 = tail call float @_Z3expf(float %304) #3
(floatB

	full_text


float %304
0addB)
'
	full_text

%306 = add i64 %4, 656
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%307 = getelementptr inbounds float, float* %1, i64 %306
$i64B

	full_text


i64 %306
LstoreBC
A
	full_text4
2
0store float %305, float* %307, align 4, !tbaa !8
(floatB

	full_text


float %305
*float*B

	full_text

float* %307
0addB)
'
	full_text

%308 = add i64 %4, 664
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%309 = getelementptr inbounds float, float* %1, i64 %308
$i64B

	full_text


i64 %308
ZstoreBQ
O
	full_textB
@
>store float 0x426D1A94A0000000, float* %309, align 4, !tbaa !8
*float*B

	full_text

float* %309
0addB)
'
	full_text

%310 = add i64 %4, 672
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%311 = getelementptr inbounds float, float* %1, i64 %310
$i64B

	full_text


i64 %310
ZstoreBQ
O
	full_textB
@
>store float 0x42A85FDC80000000, float* %311, align 4, !tbaa !8
*float*B

	full_text

float* %311
ÅcallBy
w
	full_textj
h
f%312 = tail call float @llvm.fmuladd.f32(float %8, float 0x4003C28F60000000, float 0x4024367DC0000000)
&floatB

	full_text


float %8
scallBk
i
	full_text\
Z
X%313 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0A45D5320000000, float %312)
&floatB

	full_text


float %9
(floatB

	full_text


float %312
IcallBA
?
	full_text2
0
.%314 = tail call float @_Z3expf(float %313) #3
(floatB

	full_text


float %313
0addB)
'
	full_text

%315 = add i64 %4, 680
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%316 = getelementptr inbounds float, float* %1, i64 %315
$i64B

	full_text


i64 %315
LstoreBC
A
	full_text4
2
0store float %314, float* %316, align 4, !tbaa !8
(floatB

	full_text


float %314
*float*B

	full_text

float* %316
0addB)
'
	full_text

%317 = add i64 %4, 688
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%318 = getelementptr inbounds float, float* %1, i64 %317
$i64B

	full_text


i64 %317
ZstoreBQ
O
	full_textB
@
>store float 0x42BB48EB60000000, float* %318, align 4, !tbaa !8
*float*B

	full_text

float* %318
0addB)
'
	full_text

%319 = add i64 %4, 696
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%320 = getelementptr inbounds float, float* %1, i64 %319
$i64B

	full_text


i64 %319
ZstoreBQ
O
	full_textB
@
>store float 0x429ED99D80000000, float* %320, align 4, !tbaa !8
*float*B

	full_text

float* %320
0addB)
'
	full_text

%321 = add i64 %4, 704
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%322 = getelementptr inbounds float, float* %1, i64 %321
$i64B

	full_text


i64 %321
ZstoreBQ
O
	full_textB
@
>store float 0x42B05EF3A0000000, float* %322, align 4, !tbaa !8
*float*B

	full_text

float* %322
ÅcallBy
w
	full_textj
h
f%323 = tail call float @llvm.fmuladd.f32(float %8, float 0x40067AE140000000, float 0x4020372720000000)
&floatB

	full_text


float %8
scallBk
i
	full_text\
Z
X%324 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0A709B300000000, float %323)
&floatB

	full_text


float %9
(floatB

	full_text


float %323
IcallBA
?
	full_text2
0
.%325 = tail call float @_Z3expf(float %324) #3
(floatB

	full_text


float %324
0addB)
'
	full_text

%326 = add i64 %4, 712
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%327 = getelementptr inbounds float, float* %1, i64 %326
$i64B

	full_text


i64 %326
LstoreBC
A
	full_text4
2
0store float %325, float* %327, align 4, !tbaa !8
(floatB

	full_text


float %325
*float*B

	full_text

float* %327
0addB)
'
	full_text

%328 = add i64 %4, 720
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%329 = getelementptr inbounds float, float* %1, i64 %328
$i64B

	full_text


i64 %328
ZstoreBQ
O
	full_textB
@
>store float 0x42C2309CE0000000, float* %329, align 4, !tbaa !8
*float*B

	full_text

float* %329
CfmulB;
9
	full_text,
*
(%330 = fmul float %9, 0x4071ED5600000000
&floatB

	full_text


float %9
IcallBA
?
	full_text2
0
.%331 = tail call float @_Z3expf(float %330) #3
(floatB

	full_text


float %330
EfmulB=
;
	full_text.
,
*%332 = fmul float %331, 0x42A5D3EF80000000
(floatB

	full_text


float %331
0addB)
'
	full_text

%333 = add i64 %4, 728
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%334 = getelementptr inbounds float, float* %1, i64 %333
$i64B

	full_text


i64 %333
LstoreBC
A
	full_text4
2
0store float %332, float* %334, align 4, !tbaa !8
(floatB

	full_text


float %332
*float*B

	full_text

float* %334
EfmulB=
;
	full_text.
,
*%335 = fmul float %331, 0x42AD1A94A0000000
(floatB

	full_text


float %331
0addB)
'
	full_text

%336 = add i64 %4, 848
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%337 = getelementptr inbounds float, float* %1, i64 %336
$i64B

	full_text


i64 %336
LstoreBC
A
	full_text4
2
0store float %335, float* %337, align 4, !tbaa !8
(floatB

	full_text


float %335
*float*B

	full_text

float* %337
ÅcallBy
w
	full_textj
h
f%338 = tail call float @llvm.fmuladd.f32(float %8, float 0xBFEF0A3D80000000, float 0x4042CBE020000000)
&floatB

	full_text


float %8
scallBk
i
	full_text\
Z
X%339 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0737FE8C0000000, float %338)
&floatB

	full_text


float %9
(floatB

	full_text


float %338
IcallBA
?
	full_text2
0
.%340 = tail call float @_Z3expf(float %339) #3
(floatB

	full_text


float %339
0addB)
'
	full_text

%341 = add i64 %4, 736
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%342 = getelementptr inbounds float, float* %1, i64 %341
$i64B

	full_text


i64 %341
LstoreBC
A
	full_text4
2
0store float %340, float* %342, align 4, !tbaa !8
(floatB

	full_text


float %340
*float*B

	full_text

float* %342
ÅcallBy
w
	full_textj
h
f%343 = tail call float @llvm.fmuladd.f32(float %8, float 0x3FB99999A0000000, float 0x403D3D0B80000000)
&floatB

	full_text


float %8
scallBk
i
	full_text\
Z
X%344 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0B4D618C0000000, float %343)
&floatB

	full_text


float %9
(floatB

	full_text


float %343
IcallBA
?
	full_text2
0
.%345 = tail call float @_Z3expf(float %344) #3
(floatB

	full_text


float %344
0addB)
'
	full_text

%346 = add i64 %4, 744
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%347 = getelementptr inbounds float, float* %1, i64 %346
$i64B

	full_text


i64 %346
LstoreBC
A
	full_text4
2
0store float %345, float* %347, align 4, !tbaa !8
(floatB

	full_text


float %345
*float*B

	full_text

float* %347
0addB)
'
	full_text

%348 = add i64 %4, 752
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%349 = getelementptr inbounds float, float* %1, i64 %348
$i64B

	full_text


i64 %348
ZstoreBQ
O
	full_textB
@
>store float 0x42C6BCC420000000, float* %349, align 4, !tbaa !8
*float*B

	full_text

float* %349
0addB)
'
	full_text

%350 = add i64 %4, 760
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%351 = getelementptr inbounds float, float* %1, i64 %350
$i64B

	full_text


i64 %350
ZstoreBQ
O
	full_textB
@
>store float 0x42B2309CE0000000, float* %351, align 4, !tbaa !8
*float*B

	full_text

float* %351
0addB)
'
	full_text

%352 = add i64 %4, 768
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%353 = getelementptr inbounds float, float* %1, i64 %352
$i64B

	full_text


i64 %352
ZstoreBQ
O
	full_textB
@
>store float 0x42BD1A94A0000000, float* %353, align 4, !tbaa !8
*float*B

	full_text

float* %353
0addB)
'
	full_text

%354 = add i64 %4, 776
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%355 = getelementptr inbounds float, float* %1, i64 %354
$i64B

	full_text


i64 %354
ZstoreBQ
O
	full_textB
@
>store float 0x42AD1A94A0000000, float* %355, align 4, !tbaa !8
*float*B

	full_text

float* %355
0addB)
'
	full_text

%356 = add i64 %4, 784
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%357 = getelementptr inbounds float, float* %1, i64 %356
$i64B

	full_text


i64 %356
ZstoreBQ
O
	full_textB
@
>store float 0x42A2309CE0000000, float* %357, align 4, !tbaa !8
*float*B

	full_text

float* %357
0addB)
'
	full_text

%358 = add i64 %4, 792
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%359 = getelementptr inbounds float, float* %1, i64 %358
$i64B

	full_text


i64 %358
ZstoreBQ
O
	full_textB
@
>store float 0x4292309CE0000000, float* %359, align 4, !tbaa !8
*float*B

	full_text

float* %359
ÅcallBy
w
	full_textj
h
f%360 = tail call float @llvm.fmuladd.f32(float %8, float 0x401E666660000000, float 0xC03C7ACA80000000)
&floatB

	full_text


float %8
scallBk
i
	full_text\
Z
X%361 = tail call float @llvm.fmuladd.f32(float %9, float 0x409BC16B60000000, float %360)
&floatB

	full_text


float %9
(floatB

	full_text


float %360
IcallBA
?
	full_text2
0
.%362 = tail call float @_Z3expf(float %361) #3
(floatB

	full_text


float %361
0addB)
'
	full_text

%363 = add i64 %4, 800
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%364 = getelementptr inbounds float, float* %1, i64 %363
$i64B

	full_text


i64 %363
LstoreBC
A
	full_text4
2
0store float %362, float* %364, align 4, !tbaa !8
(floatB

	full_text


float %362
*float*B

	full_text

float* %364
ÅcallBy
w
	full_textj
h
f%365 = tail call float @llvm.fmuladd.f32(float %8, float 0x3FF9EB8520000000, float 0x40344EC8C0000000)
&floatB

	full_text


float %8
scallBk
i
	full_text\
Z
X%366 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0B54EDE60000000, float %365)
&floatB

	full_text


float %9
(floatB

	full_text


float %365
IcallBA
?
	full_text2
0
.%367 = tail call float @_Z3expf(float %366) #3
(floatB

	full_text


float %366
0addB)
'
	full_text

%368 = add i64 %4, 808
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%369 = getelementptr inbounds float, float* %1, i64 %368
$i64B

	full_text


i64 %368
LstoreBC
A
	full_text4
2
0store float %367, float* %369, align 4, !tbaa !8
(floatB

	full_text


float %367
*float*B

	full_text

float* %369
{callBs
q
	full_textd
b
`%370 = tail call float @llvm.fmuladd.f32(float %8, float 1.500000e+00, float 0x4034BE39C0000000)
&floatB

	full_text


float %8
scallBk
i
	full_text\
Z
X%371 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0B0E7A9E0000000, float %370)
&floatB

	full_text


float %9
(floatB

	full_text


float %370
IcallBA
?
	full_text2
0
.%372 = tail call float @_Z3expf(float %371) #3
(floatB

	full_text


float %371
0addB)
'
	full_text

%373 = add i64 %4, 816
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%374 = getelementptr inbounds float, float* %1, i64 %373
$i64B

	full_text


i64 %373
LstoreBC
A
	full_text4
2
0store float %372, float* %374, align 4, !tbaa !8
(floatB

	full_text


float %372
*float*B

	full_text

float* %374
ÅcallBy
w
	full_textj
h
f%375 = tail call float @llvm.fmuladd.f32(float %8, float 0x3FF99999A0000000, float 0x40326BB1C0000000)
&floatB

	full_text


float %8
scallBk
i
	full_text\
Z
X%376 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0988824E0000000, float %375)
&floatB

	full_text


float %9
(floatB

	full_text


float %375
IcallBA
?
	full_text2
0
.%377 = tail call float @_Z3expf(float %376) #3
(floatB

	full_text


float %376
0addB)
'
	full_text

%378 = add i64 %4, 824
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%379 = getelementptr inbounds float, float* %1, i64 %378
$i64B

	full_text


i64 %378
LstoreBC
A
	full_text4
2
0store float %377, float* %379, align 4, !tbaa !8
(floatB

	full_text


float %377
*float*B

	full_text

float* %379
0addB)
'
	full_text

%380 = add i64 %4, 832
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%381 = getelementptr inbounds float, float* %1, i64 %380
$i64B

	full_text


i64 %380
ZstoreBQ
O
	full_textB
@
>store float 0x42CB48EB60000000, float* %381, align 4, !tbaa !8
*float*B

	full_text

float* %381
{callBs
q
	full_textd
b
`%382 = tail call float @llvm.fmuladd.f32(float %8, float 2.000000e+00, float 0x402D6E6C80000000)
&floatB

	full_text


float %8
scallBk
i
	full_text\
Z
X%383 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0B0419A20000000, float %382)
&floatB

	full_text


float %9
(floatB

	full_text


float %382
IcallBA
?
	full_text2
0
.%384 = tail call float @_Z3expf(float %383) #3
(floatB

	full_text


float %383
0addB)
'
	full_text

%385 = add i64 %4, 840
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%386 = getelementptr inbounds float, float* %1, i64 %385
$i64B

	full_text


i64 %385
LstoreBC
A
	full_text4
2
0store float %384, float* %386, align 4, !tbaa !8
(floatB

	full_text


float %384
*float*B

	full_text

float* %386
0addB)
'
	full_text

%387 = add i64 %4, 856
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%388 = getelementptr inbounds float, float* %1, i64 %387
$i64B

	full_text


i64 %387
ZstoreBQ
O
	full_textB
@
>store float 0x42D6BCC420000000, float* %388, align 4, !tbaa !8
*float*B

	full_text

float* %388
0addB)
'
	full_text

%389 = add i64 %4, 864
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%390 = getelementptr inbounds float, float* %1, i64 %389
$i64B

	full_text


i64 %389
ZstoreBQ
O
	full_textB
@
>store float 0x42D6BCC420000000, float* %390, align 4, !tbaa !8
*float*B

	full_text

float* %390
ÅcallBy
w
	full_textj
h
f%391 = tail call float @llvm.fmuladd.f32(float %9, float 0xC07ADBF3E0000000, float 0x403C19DCC0000000)
&floatB

	full_text


float %9
IcallBA
?
	full_text2
0
.%392 = tail call float @_Z3expf(float %391) #3
(floatB

	full_text


float %391
0addB)
'
	full_text

%393 = add i64 %4, 872
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%394 = getelementptr inbounds float, float* %1, i64 %393
$i64B

	full_text


i64 %393
LstoreBC
A
	full_text4
2
0store float %392, float* %394, align 4, !tbaa !8
(floatB

	full_text


float %392
*float*B

	full_text

float* %394
0addB)
'
	full_text

%395 = add i64 %4, 880
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%396 = getelementptr inbounds float, float* %1, i64 %395
$i64B

	full_text


i64 %395
ZstoreBQ
O
	full_textB
@
>store float 0x42C6BCC420000000, float* %396, align 4, !tbaa !8
*float*B

	full_text

float* %396
0addB)
'
	full_text

%397 = add i64 %4, 888
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%398 = getelementptr inbounds float, float* %1, i64 %397
$i64B

	full_text


i64 %397
ZstoreBQ
O
	full_textB
@
>store float 0x42BB48EB60000000, float* %398, align 4, !tbaa !8
*float*B

	full_text

float* %398
0addB)
'
	full_text

%399 = add i64 %4, 896
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%400 = getelementptr inbounds float, float* %1, i64 %399
$i64B

	full_text


i64 %399
ZstoreBQ
O
	full_textB
@
>store float 0x42A2309CE0000000, float* %400, align 4, !tbaa !8
*float*B

	full_text

float* %400
ÅcallBy
w
	full_textj
h
f%401 = tail call float @llvm.fmuladd.f32(float %8, float 0xBFE0A3D700000000, float 0x40412866A0000000)
&floatB

	full_text


float %8
scallBk
i
	full_text\
Z
X%402 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0D8F08FC0000000, float %401)
&floatB

	full_text


float %9
(floatB

	full_text


float %401
IcallBA
?
	full_text2
0
.%403 = tail call float @_Z3expf(float %402) #3
(floatB

	full_text


float %402
0addB)
'
	full_text

%404 = add i64 %4, 904
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%405 = getelementptr inbounds float, float* %1, i64 %404
$i64B

	full_text


i64 %404
LstoreBC
A
	full_text4
2
0store float %403, float* %405, align 4, !tbaa !8
(floatB

	full_text


float %403
*float*B

	full_text

float* %405
ÅcallBy
w
	full_textj
h
f%406 = tail call float @llvm.fmuladd.f32(float %8, float 0x3FF9EB8520000000, float 0x4033C57700000000)
&floatB

	full_text


float %8
scallBk
i
	full_text\
Z
X%407 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0D234D200000000, float %406)
&floatB

	full_text


float %9
(floatB

	full_text


float %406
IcallBA
?
	full_text2
0
.%408 = tail call float @_Z3expf(float %407) #3
(floatB

	full_text


float %407
0addB)
'
	full_text

%409 = add i64 %4, 912
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%410 = getelementptr inbounds float, float* %1, i64 %409
$i64B

	full_text


i64 %409
LstoreBC
A
	full_text4
2
0store float %408, float* %410, align 4, !tbaa !8
(floatB

	full_text


float %408
*float*B

	full_text

float* %410
CfmulB;
9
	full_text,
*
(%411 = fmul float %9, 0x408DE0E4C0000000
&floatB

	full_text


float %9
@fsubB8
6
	full_text)
'
%%412 = fsub float -0.000000e+00, %411
(floatB

	full_text


float %411
mcallBe
c
	full_textV
T
R%413 = tail call float @llvm.fmuladd.f32(float %8, float 2.000000e+00, float %412)
&floatB

	full_text


float %8
(floatB

	full_text


float %412
IcallBA
?
	full_text2
0
.%414 = tail call float @_Z3expf(float %413) #3
(floatB

	full_text


float %413
?fmulB7
5
	full_text(
&
$%415 = fmul float %414, 1.632000e+07
(floatB

	full_text


float %414
0addB)
'
	full_text

%416 = add i64 %4, 920
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%417 = getelementptr inbounds float, float* %1, i64 %416
$i64B

	full_text


i64 %416
LstoreBC
A
	full_text4
2
0store float %415, float* %417, align 4, !tbaa !8
(floatB

	full_text


float %415
*float*B

	full_text

float* %417
?fmulB7
5
	full_text(
&
$%418 = fmul float %414, 4.080000e+06
(floatB

	full_text


float %414
0addB)
'
	full_text

%419 = add i64 %4, 928
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%420 = getelementptr inbounds float, float* %1, i64 %419
$i64B

	full_text


i64 %419
LstoreBC
A
	full_text4
2
0store float %418, float* %420, align 4, !tbaa !8
(floatB

	full_text


float %418
*float*B

	full_text

float* %420
{callBs
q
	full_textd
b
`%421 = tail call float @llvm.fmuladd.f32(float %8, float 4.500000e+00, float 0xC020DCAE20000000)
&floatB

	full_text


float %8
scallBk
i
	full_text\
Z
X%422 = tail call float @llvm.fmuladd.f32(float %9, float 0x407F737780000000, float %421)
&floatB

	full_text


float %9
(floatB

	full_text


float %421
IcallBA
?
	full_text2
0
.%423 = tail call float @_Z3expf(float %422) #3
(floatB

	full_text


float %422
0addB)
'
	full_text

%424 = add i64 %4, 936
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%425 = getelementptr inbounds float, float* %1, i64 %424
$i64B

	full_text


i64 %424
LstoreBC
A
	full_text4
2
0store float %423, float* %425, align 4, !tbaa !8
(floatB

	full_text


float %423
*float*B

	full_text

float* %425
{callBs
q
	full_textd
b
`%426 = tail call float @llvm.fmuladd.f32(float %8, float 4.000000e+00, float 0xC01E8ABEE0000000)
&floatB

	full_text


float %8
scallBk
i
	full_text\
Z
X%427 = tail call float @llvm.fmuladd.f32(float %9, float 0x408F737780000000, float %426)
&floatB

	full_text


float %9
(floatB

	full_text


float %426
IcallBA
?
	full_text2
0
.%428 = tail call float @_Z3expf(float %427) #3
(floatB

	full_text


float %427
0addB)
'
	full_text

%429 = add i64 %4, 944
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%430 = getelementptr inbounds float, float* %1, i64 %429
$i64B

	full_text


i64 %429
LstoreBC
A
	full_text4
2
0store float %428, float* %430, align 4, !tbaa !8
(floatB

	full_text


float %428
*float*B

	full_text

float* %430
{callBs
q
	full_textd
b
`%431 = tail call float @llvm.fmuladd.f32(float %8, float 2.000000e+00, float 0x40301E3B80000000)
&floatB

	full_text


float %8
scallBk
i
	full_text\
Z
X%432 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0A79699A0000000, float %431)
&floatB

	full_text


float %9
(floatB

	full_text


float %431
IcallBA
?
	full_text2
0
.%433 = tail call float @_Z3expf(float %432) #3
(floatB

	full_text


float %432
0addB)
'
	full_text

%434 = add i64 %4, 952
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%435 = getelementptr inbounds float, float* %1, i64 %434
$i64B

	full_text


i64 %434
LstoreBC
A
	full_text4
2
0store float %433, float* %435, align 4, !tbaa !8
(floatB

	full_text


float %433
*float*B

	full_text

float* %435
ÅcallBy
w
	full_textj
h
f%436 = tail call float @llvm.fmuladd.f32(float %8, float 0xC027A3D700000000, float 0x405FDB8F80000000)
&floatB

	full_text


float %8
scallBk
i
	full_text\
Z
X%437 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0D18EFBA0000000, float %436)
&floatB

	full_text


float %9
(floatB

	full_text


float %436
IcallBA
?
	full_text2
0
.%438 = tail call float @_Z3expf(float %437) #3
(floatB

	full_text


float %437
0addB)
'
	full_text

%439 = add i64 %4, 960
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%440 = getelementptr inbounds float, float* %1, i64 %439
$i64B

	full_text


i64 %439
LstoreBC
A
	full_text4
2
0store float %438, float* %440, align 4, !tbaa !8
(floatB

	full_text


float %438
*float*B

	full_text

float* %440
0addB)
'
	full_text

%441 = add i64 %4, 968
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%442 = getelementptr inbounds float, float* %1, i64 %441
$i64B

	full_text


i64 %441
ZstoreBQ
O
	full_textB
@
>store float 0x42D6BCC420000000, float* %442, align 4, !tbaa !8
*float*B

	full_text

float* %442
0addB)
'
	full_text

%443 = add i64 %4, 976
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%444 = getelementptr inbounds float, float* %1, i64 %443
$i64B

	full_text


i64 %443
ZstoreBQ
O
	full_textB
@
>store float 0x42D6BCC420000000, float* %444, align 4, !tbaa !8
*float*B

	full_text

float* %444
0addB)
'
	full_text

%445 = add i64 %4, 984
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%446 = getelementptr inbounds float, float* %1, i64 %445
$i64B

	full_text


i64 %445
ZstoreBQ
O
	full_textB
@
>store float 0x42B2309CE0000000, float* %446, align 4, !tbaa !8
*float*B

	full_text

float* %446
0addB)
'
	full_text

%447 = add i64 %4, 992
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%448 = getelementptr inbounds float, float* %1, i64 %447
$i64B

	full_text


i64 %447
ZstoreBQ
O
	full_textB
@
>store float 0x42A2309CE0000000, float* %448, align 4, !tbaa !8
*float*B

	full_text

float* %448
ÅcallBy
w
	full_textj
h
f%449 = tail call float @llvm.fmuladd.f32(float %8, float 0xBFAEB851E0000000, float 0x4040B70E00000000)
&floatB

	full_text


float %8
scallBk
i
	full_text\
Z
X%450 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0B0B55780000000, float %449)
&floatB

	full_text


float %9
(floatB

	full_text


float %449
IcallBA
?
	full_text2
0
.%451 = tail call float @_Z3expf(float %450) #3
(floatB

	full_text


float %450
1addB*
(
	full_text

%452 = add i64 %4, 1000
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%453 = getelementptr inbounds float, float* %1, i64 %452
$i64B

	full_text


i64 %452
LstoreBC
A
	full_text4
2
0store float %451, float* %453, align 4, !tbaa !8
(floatB

	full_text


float %451
*float*B

	full_text

float* %453
ÅcallBy
w
	full_textj
h
f%454 = tail call float @llvm.fmuladd.f32(float %8, float 0x3FF6E147A0000000, float 0x403520F480000000)
&floatB

	full_text


float %8
scallBk
i
	full_text\
Z
X%455 = tail call float @llvm.fmuladd.f32(float %9, float 0xC095269C80000000, float %454)
&floatB

	full_text


float %9
(floatB

	full_text


float %454
IcallBA
?
	full_text2
0
.%456 = tail call float @_Z3expf(float %455) #3
(floatB

	full_text


float %455
1addB*
(
	full_text

%457 = add i64 %4, 1016
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%458 = getelementptr inbounds float, float* %1, i64 %457
$i64B

	full_text


i64 %457
LstoreBC
A
	full_text4
2
0store float %456, float* %458, align 4, !tbaa !8
(floatB

	full_text


float %456
*float*B

	full_text

float* %458
ÅcallBy
w
	full_textj
h
f%459 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0853ABD80000000, float 0x403C30CDA0000000)
&floatB

	full_text


float %9
IcallBA
?
	full_text2
0
.%460 = tail call float @_Z3expf(float %459) #3
(floatB

	full_text


float %459
1addB*
(
	full_text

%461 = add i64 %4, 1032
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%462 = getelementptr inbounds float, float* %1, i64 %461
$i64B

	full_text


i64 %461
LstoreBC
A
	full_text4
2
0store float %460, float* %462, align 4, !tbaa !8
(floatB

	full_text


float %460
*float*B

	full_text

float* %462
CfmulB;
9
	full_text,
*
(%463 = fmul float %9, 0xC08F737780000000
&floatB

	full_text


float %9
IcallBA
?
	full_text2
0
.%464 = tail call float @_Z3expf(float %463) #3
(floatB

	full_text


float %463
EfmulB=
;
	full_text.
,
*%465 = fmul float %464, 0x429B48EB60000000
(floatB

	full_text


float %464
1addB*
(
	full_text

%466 = add i64 %4, 1040
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%467 = getelementptr inbounds float, float* %1, i64 %466
$i64B

	full_text


i64 %466
LstoreBC
A
	full_text4
2
0store float %465, float* %467, align 4, !tbaa !8
(floatB

	full_text


float %465
*float*B

	full_text

float* %467
EfmulB=
;
	full_text.
,
*%468 = fmul float %464, 0x42A2309CE0000000
(floatB

	full_text


float %464
1addB*
(
	full_text

%469 = add i64 %4, 1208
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%470 = getelementptr inbounds float, float* %1, i64 %469
$i64B

	full_text


i64 %469
LstoreBC
A
	full_text4
2
0store float %468, float* %470, align 4, !tbaa !8
(floatB

	full_text


float %468
*float*B

	full_text

float* %470
EfmulB=
;
	full_text.
,
*%471 = fmul float %464, 0x42B2309CE0000000
(floatB

	full_text


float %464
1addB*
(
	full_text

%472 = add i64 %4, 1480
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%473 = getelementptr inbounds float, float* %1, i64 %472
$i64B

	full_text


i64 %472
LstoreBC
A
	full_text4
2
0store float %471, float* %473, align 4, !tbaa !8
(floatB

	full_text


float %471
*float*B

	full_text

float* %473
ÅcallBy
w
	full_textj
h
f%474 = tail call float @llvm.fmuladd.f32(float %8, float 0x3FD147AE20000000, float 0x403D6F9F60000000)
&floatB

	full_text


float %8
scallBk
i
	full_text\
Z
X%475 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0619CD240000000, float %474)
&floatB

	full_text


float %9
(floatB

	full_text


float %474
IcallBA
?
	full_text2
0
.%476 = tail call float @_Z3expf(float %475) #3
(floatB

	full_text


float %475
1addB*
(
	full_text

%477 = add i64 %4, 1048
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%478 = getelementptr inbounds float, float* %1, i64 %477
$i64B

	full_text


i64 %477
LstoreBC
A
	full_text4
2
0store float %476, float* %478, align 4, !tbaa !8
(floatB

	full_text


float %476
*float*B

	full_text

float* %478
1addB*
(
	full_text

%479 = add i64 %4, 1056
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%480 = getelementptr inbounds float, float* %1, i64 %479
$i64B

	full_text


i64 %479
ZstoreBQ
O
	full_textB
@
>store float 0x42BB48EB60000000, float* %480, align 4, !tbaa !8
*float*B

	full_text

float* %480
1addB*
(
	full_text

%481 = add i64 %4, 1064
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%482 = getelementptr inbounds float, float* %1, i64 %481
$i64B

	full_text


i64 %481
ZstoreBQ
O
	full_textB
@
>store float 0x42CB48EB60000000, float* %482, align 4, !tbaa !8
*float*B

	full_text

float* %482
1addB*
(
	full_text

%483 = add i64 %4, 1072
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%484 = getelementptr inbounds float, float* %1, i64 %483
$i64B

	full_text


i64 %483
ZstoreBQ
O
	full_textB
@
>store float 0x42C5D3EF80000000, float* %484, align 4, !tbaa !8
*float*B

	full_text

float* %484
1addB*
(
	full_text

%485 = add i64 %4, 1080
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%486 = getelementptr inbounds float, float* %1, i64 %485
$i64B

	full_text


i64 %485
ZstoreBQ
O
	full_textB
@
>store float 0x42C5D3EF80000000, float* %486, align 4, !tbaa !8
*float*B

	full_text

float* %486
1addB*
(
	full_text

%487 = add i64 %4, 1088
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%488 = getelementptr inbounds float, float* %1, i64 %487
$i64B

	full_text


i64 %487
ZstoreBQ
O
	full_textB
@
>store float 0x42BB6287E0000000, float* %488, align 4, !tbaa !8
*float*B

	full_text

float* %488
ÅcallBy
w
	full_textj
h
f%489 = tail call float @llvm.fmuladd.f32(float %8, float 0x3FF9C28F60000000, float 0x402C376360000000)
&floatB

	full_text


float %8
scallBk
i
	full_text\
Z
X%490 = tail call float @llvm.fmuladd.f32(float %9, float 0x40681DDD60000000, float %489)
&floatB

	full_text


float %9
(floatB

	full_text


float %489
IcallBA
?
	full_text2
0
.%491 = tail call float @_Z3expf(float %490) #3
(floatB

	full_text


float %490
1addB*
(
	full_text

%492 = add i64 %4, 1096
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%493 = getelementptr inbounds float, float* %1, i64 %492
$i64B

	full_text


i64 %492
LstoreBC
A
	full_text4
2
0store float %491, float* %493, align 4, !tbaa !8
(floatB

	full_text


float %491
*float*B

	full_text

float* %493
ÅcallBy
w
	full_textj
h
f%494 = tail call float @llvm.fmuladd.f32(float %8, float 0x3FD28F5C20000000, float 0x403A6D5300000000)
&floatB

	full_text


float %8
scallBk
i
	full_text\
Z
X%495 = tail call float @llvm.fmuladd.f32(float %9, float 0xC016243B80000000, float %494)
&floatB

	full_text


float %9
(floatB

	full_text


float %494
IcallBA
?
	full_text2
0
.%496 = tail call float @_Z3expf(float %495) #3
(floatB

	full_text


float %495
1addB*
(
	full_text

%497 = add i64 %4, 1104
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%498 = getelementptr inbounds float, float* %1, i64 %497
$i64B

	full_text


i64 %497
LstoreBC
A
	full_text4
2
0store float %496, float* %498, align 4, !tbaa !8
(floatB

	full_text


float %496
*float*B

	full_text

float* %498
ÅcallBy
w
	full_textj
h
f%499 = tail call float @llvm.fmuladd.f32(float %8, float 0xBFF63D70A0000000, float 0x40432F0780000000)
&floatB

	full_text


float %8
scallBk
i
	full_text\
Z
X%500 = tail call float @llvm.fmuladd.f32(float %9, float 0xC07FC3FB40000000, float %499)
&floatB

	full_text


float %9
(floatB

	full_text


float %499
IcallBA
?
	full_text2
0
.%501 = tail call float @_Z3expf(float %500) #3
(floatB

	full_text


float %500
1addB*
(
	full_text

%502 = add i64 %4, 1112
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%503 = getelementptr inbounds float, float* %1, i64 %502
$i64B

	full_text


i64 %502
LstoreBC
A
	full_text4
2
0store float %501, float* %503, align 4, !tbaa !8
(floatB

	full_text


float %501
*float*B

	full_text

float* %503
1addB*
(
	full_text

%504 = add i64 %4, 1120
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%505 = getelementptr inbounds float, float* %1, i64 %504
$i64B

	full_text


i64 %504
ZstoreBQ
O
	full_textB
@
>store float 0x42A2309CE0000000, float* %505, align 4, !tbaa !8
*float*B

	full_text

float* %505
ÅcallBy
w
	full_textj
h
f%506 = tail call float @llvm.fmuladd.f32(float %9, float 0x4072BEACA0000000, float 0x4037376AA0000000)
&floatB

	full_text


float %9
IcallBA
?
	full_text2
0
.%507 = tail call float @_Z3expf(float %506) #3
(floatB

	full_text


float %506
1addB*
(
	full_text

%508 = add i64 %4, 1128
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%509 = getelementptr inbounds float, float* %1, i64 %508
$i64B

	full_text


i64 %508
LstoreBC
A
	full_text4
2
0store float %507, float* %509, align 4, !tbaa !8
(floatB

	full_text


float %507
*float*B

	full_text

float* %509
1addB*
(
	full_text

%510 = add i64 %4, 1136
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%511 = getelementptr inbounds float, float* %1, i64 %510
$i64B

	full_text


i64 %510
ZstoreBQ
O
	full_textB
@
>store float 0x42D489E5E0000000, float* %511, align 4, !tbaa !8
*float*B

	full_text

float* %511
1addB*
(
	full_text

%512 = add i64 %4, 1144
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%513 = getelementptr inbounds float, float* %1, i64 %512
$i64B

	full_text


i64 %512
ZstoreBQ
O
	full_textB
@
>store float 0x4256D14160000000, float* %513, align 4, !tbaa !8
*float*B

	full_text

float* %513
1addB*
(
	full_text

%514 = add i64 %4, 1152
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%515 = getelementptr inbounds float, float* %1, i64 %514
$i64B

	full_text


i64 %514
ZstoreBQ
O
	full_textB
@
>store float 0x42B6BCC420000000, float* %515, align 4, !tbaa !8
*float*B

	full_text

float* %515
ÅcallBy
w
	full_textj
h
f%516 = tail call float @llvm.fmuladd.f32(float %8, float 0xC006A3D700000000, float 0x404BD570E0000000)
&floatB

	full_text


float %8
scallBk
i
	full_text\
Z
X%517 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0C24C71A0000000, float %516)
&floatB

	full_text


float %9
(floatB

	full_text


float %516
IcallBA
?
	full_text2
0
.%518 = tail call float @_Z3expf(float %517) #3
(floatB

	full_text


float %517
1addB*
(
	full_text

%519 = add i64 %4, 1160
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%520 = getelementptr inbounds float, float* %1, i64 %519
$i64B

	full_text


i64 %519
LstoreBC
A
	full_text4
2
0store float %518, float* %520, align 4, !tbaa !8
(floatB

	full_text


float %518
*float*B

	full_text

float* %520
ÅcallBy
w
	full_textj
h
f%521 = tail call float @llvm.fmuladd.f32(float %8, float 0xC0224B43A0000000, float 0x40581D7280000000)
&floatB

	full_text


float %8
scallBk
i
	full_text\
Z
X%522 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0D70C3720000000, float %521)
&floatB

	full_text


float %9
(floatB

	full_text


float %521
IcallBA
?
	full_text2
0
.%523 = tail call float @_Z3expf(float %522) #3
(floatB

	full_text


float %522
1addB*
(
	full_text

%524 = add i64 %4, 1168
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%525 = getelementptr inbounds float, float* %1, i64 %524
$i64B

	full_text


i64 %524
LstoreBC
A
	full_text4
2
0store float %523, float* %525, align 4, !tbaa !8
(floatB

	full_text


float %523
*float*B

	full_text

float* %525
1addB*
(
	full_text

%526 = add i64 %4, 1176
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%527 = getelementptr inbounds float, float* %1, i64 %526
$i64B

	full_text


i64 %526
ZstoreBQ
O
	full_textB
@
>store float 0x42D6BCC420000000, float* %527, align 4, !tbaa !8
*float*B

	full_text

float* %527
1addB*
(
	full_text

%528 = add i64 %4, 1184
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%529 = getelementptr inbounds float, float* %1, i64 %528
$i64B

	full_text


i64 %528
ZstoreBQ
O
	full_textB
@
>store float 0x42D476B080000000, float* %529, align 4, !tbaa !8
*float*B

	full_text

float* %529
CfmulB;
9
	full_text,
*
(%530 = fmul float %9, 0xC09F737780000000
&floatB

	full_text


float %9
IcallBA
?
	full_text2
0
.%531 = tail call float @_Z3expf(float %530) #3
(floatB

	full_text


float %530
EfmulB=
;
	full_text.
,
*%532 = fmul float %531, 0x42B2309CE0000000
(floatB

	full_text


float %531
1addB*
(
	full_text

%533 = add i64 %4, 1192
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%534 = getelementptr inbounds float, float* %1, i64 %533
$i64B

	full_text


i64 %533
LstoreBC
A
	full_text4
2
0store float %532, float* %534, align 4, !tbaa !8
(floatB

	full_text


float %532
*float*B

	full_text

float* %534
1addB*
(
	full_text

%535 = add i64 %4, 1200
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%536 = getelementptr inbounds float, float* %1, i64 %535
$i64B

	full_text


i64 %535
LstoreBC
A
	full_text4
2
0store float %532, float* %536, align 4, !tbaa !8
(floatB

	full_text


float %532
*float*B

	full_text

float* %536
1addB*
(
	full_text

%537 = add i64 %4, 1216
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%538 = getelementptr inbounds float, float* %1, i64 %537
$i64B

	full_text


i64 %537
ZstoreBQ
O
	full_textB
@
>store float 0x42404C5340000000, float* %538, align 4, !tbaa !8
*float*B

	full_text

float* %538
1addB*
(
	full_text

%539 = add i64 %4, 1224
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%540 = getelementptr inbounds float, float* %1, i64 %539
$i64B

	full_text


i64 %539
ZstoreBQ
O
	full_textB
@
>store float 0x4210C388C0000000, float* %540, align 4, !tbaa !8
*float*B

	full_text

float* %540
ÅcallBy
w
	full_textj
h
f%541 = tail call float @llvm.fmuladd.f32(float %8, float 0x3FDC28F5C0000000, float 0x403DB5E0E0000000)
&floatB

	full_text


float %8
scallBk
i
	full_text\
Z
X%542 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0E5CFD160000000, float %541)
&floatB

	full_text


float %9
(floatB

	full_text


float %541
IcallBA
?
	full_text2
0
.%543 = tail call float @_Z3expf(float %542) #3
(floatB

	full_text


float %542
1addB*
(
	full_text

%544 = add i64 %4, 1232
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%545 = getelementptr inbounds float, float* %1, i64 %544
$i64B

	full_text


i64 %544
LstoreBC
A
	full_text4
2
0store float %543, float* %545, align 4, !tbaa !8
(floatB

	full_text


float %543
*float*B

	full_text

float* %545
ÅcallBy
w
	full_textj
h
f%546 = tail call float @llvm.fmuladd.f32(float %8, float 0x3FDD0E5600000000, float 0x403BB53E60000000)
&floatB

	full_text


float %8
scallBk
i
	full_text\
Z
X%547 = tail call float @llvm.fmuladd.f32(float %9, float 0xC08C9ED5A0000000, float %546)
&floatB

	full_text


float %9
(floatB

	full_text


float %546
IcallBA
?
	full_text2
0
.%548 = tail call float @_Z3expf(float %547) #3
(floatB

	full_text


float %547
1addB*
(
	full_text

%549 = add i64 %4, 1240
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%550 = getelementptr inbounds float, float* %1, i64 %549
$i64B

	full_text


i64 %549
LstoreBC
A
	full_text4
2
0store float %548, float* %550, align 4, !tbaa !8
(floatB

	full_text


float %548
*float*B

	full_text

float* %550
ÅcallBy
w
	full_textj
h
f%551 = tail call float @llvm.fmuladd.f32(float %8, float 0x3FFEE147A0000000, float 0x4031BDCEC0000000)
&floatB

	full_text


float %8
scallBk
i
	full_text\
Z
X%552 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0B974A7E0000000, float %551)
&floatB

	full_text


float %9
(floatB

	full_text


float %551
IcallBA
?
	full_text2
0
.%553 = tail call float @_Z3expf(float %552) #3
(floatB

	full_text


float %552
1addB*
(
	full_text

%554 = add i64 %4, 1248
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%555 = getelementptr inbounds float, float* %1, i64 %554
$i64B

	full_text


i64 %554
LstoreBC
A
	full_text4
2
0store float %553, float* %555, align 4, !tbaa !8
(floatB

	full_text


float %553
*float*B

	full_text

float* %555
ÅcallBy
w
	full_textj
h
f%556 = tail call float @llvm.fmuladd.f32(float %8, float 0x3FFE8F5C20000000, float 0x403087BB80000000)
&floatB

	full_text


float %8
scallBk
i
	full_text\
Z
X%557 = tail call float @llvm.fmuladd.f32(float %9, float 0xC09D681F20000000, float %556)
&floatB

	full_text


float %9
(floatB

	full_text


float %556
IcallBA
?
	full_text2
0
.%558 = tail call float @_Z3expf(float %557) #3
(floatB

	full_text


float %557
1addB*
(
	full_text

%559 = add i64 %4, 1256
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%560 = getelementptr inbounds float, float* %1, i64 %559
$i64B

	full_text


i64 %559
LstoreBC
A
	full_text4
2
0store float %558, float* %560, align 4, !tbaa !8
(floatB

	full_text


float %558
*float*B

	full_text

float* %560
CfmulB;
9
	full_text,
*
(%561 = fmul float %9, 0x405BAD4A60000000
&floatB

	full_text


float %9
@fsubB8
6
	full_text)
'
%%562 = fsub float -0.000000e+00, %561
(floatB

	full_text


float %561
scallBk
i
	full_text\
Z
X%563 = tail call float @llvm.fmuladd.f32(float %8, float 0x3FFD47AE20000000, float %562)
&floatB

	full_text


float %8
(floatB

	full_text


float %562
IcallBA
?
	full_text2
0
.%564 = tail call float @_Z3expf(float %563) #3
(floatB

	full_text


float %563
?fmulB7
5
	full_text(
&
$%565 = fmul float %564, 1.920000e+07
(floatB

	full_text


float %564
1addB*
(
	full_text

%566 = add i64 %4, 1264
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%567 = getelementptr inbounds float, float* %1, i64 %566
$i64B

	full_text


i64 %566
LstoreBC
A
	full_text4
2
0store float %565, float* %567, align 4, !tbaa !8
(floatB

	full_text


float %565
*float*B

	full_text

float* %567
?fmulB7
5
	full_text(
&
$%568 = fmul float %564, 3.840000e+05
(floatB

	full_text


float %564
1addB*
(
	full_text

%569 = add i64 %4, 1272
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%570 = getelementptr inbounds float, float* %1, i64 %569
$i64B

	full_text


i64 %569
LstoreBC
A
	full_text4
2
0store float %568, float* %570, align 4, !tbaa !8
(floatB

	full_text


float %568
*float*B

	full_text

float* %570
{callBs
q
	full_textd
b
`%571 = tail call float @llvm.fmuladd.f32(float %8, float 2.000000e+00, float 0x402E316120000000)
&floatB

	full_text


float %8
scallBk
i
	full_text\
Z
X%572 = tail call float @llvm.fmuladd.f32(float %9, float 0xC093A82AA0000000, float %571)
&floatB

	full_text


float %9
(floatB

	full_text


float %571
IcallBA
?
	full_text2
0
.%573 = tail call float @_Z3expf(float %572) #3
(floatB

	full_text


float %572
1addB*
(
	full_text

%574 = add i64 %4, 1280
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%575 = getelementptr inbounds float, float* %1, i64 %574
$i64B

	full_text


i64 %574
LstoreBC
A
	full_text4
2
0store float %573, float* %575, align 4, !tbaa !8
(floatB

	full_text


float %573
*float*B

	full_text

float* %575
ÅcallBy
w
	full_textj
h
f%576 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0DDE0E4C0000000, float 0x403F5F99E0000000)
&floatB

	full_text


float %9
IcallBA
?
	full_text2
0
.%577 = tail call float @_Z3expf(float %576) #3
(floatB

	full_text


float %576
1addB*
(
	full_text

%578 = add i64 %4, 1288
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%579 = getelementptr inbounds float, float* %1, i64 %578
$i64B

	full_text


i64 %578
LstoreBC
A
	full_text4
2
0store float %577, float* %579, align 4, !tbaa !8
(floatB

	full_text


float %577
*float*B

	full_text

float* %579
ÅcallBy
w
	full_textj
h
f%580 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0BB850880000000, float 0x403C52FCC0000000)
&floatB

	full_text


float %9
IcallBA
?
	full_text2
0
.%581 = tail call float @_Z3expf(float %580) #3
(floatB

	full_text


float %580
1addB*
(
	full_text

%582 = add i64 %4, 1296
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%583 = getelementptr inbounds float, float* %1, i64 %582
$i64B

	full_text


i64 %582
LstoreBC
A
	full_text4
2
0store float %581, float* %583, align 4, !tbaa !8
(floatB

	full_text


float %581
*float*B

	full_text

float* %583
scallBk
i
	full_text\
Z
X%584 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0AF737780000000, float %431)
&floatB

	full_text


float %9
(floatB

	full_text


float %431
IcallBA
?
	full_text2
0
.%585 = tail call float @_Z3expf(float %584) #3
(floatB

	full_text


float %584
1addB*
(
	full_text

%586 = add i64 %4, 1304
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%587 = getelementptr inbounds float, float* %1, i64 %586
$i64B

	full_text


i64 %586
LstoreBC
A
	full_text4
2
0store float %585, float* %587, align 4, !tbaa !8
(floatB

	full_text


float %585
*float*B

	full_text

float* %587
ÅcallBy
w
	full_textj
h
f%588 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0A79699A0000000, float 0x403EA072E0000000)
&floatB

	full_text


float %9
IcallBA
?
	full_text2
0
.%589 = tail call float @_Z3expf(float %588) #3
(floatB

	full_text


float %588
1addB*
(
	full_text

%590 = add i64 %4, 1312
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%591 = getelementptr inbounds float, float* %1, i64 %590
$i64B

	full_text


i64 %590
LstoreBC
A
	full_text4
2
0store float %589, float* %591, align 4, !tbaa !8
(floatB

	full_text


float %589
*float*B

	full_text

float* %591
1addB*
(
	full_text

%592 = add i64 %4, 1320
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%593 = getelementptr inbounds float, float* %1, i64 %592
$i64B

	full_text


i64 %592
ZstoreBQ
O
	full_textB
@
>store float 0x42C6BCC420000000, float* %593, align 4, !tbaa !8
*float*B

	full_text

float* %593
1addB*
(
	full_text

%594 = add i64 %4, 1328
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%595 = getelementptr inbounds float, float* %1, i64 %594
$i64B

	full_text


i64 %594
ZstoreBQ
O
	full_textB
@
>store float 0x42C6BCC420000000, float* %595, align 4, !tbaa !8
*float*B

	full_text

float* %595
{callBs
q
	full_textd
b
`%596 = tail call float @llvm.fmuladd.f32(float %8, float 2.000000e+00, float 0x4028AA5860000000)
&floatB

	full_text


float %8
scallBk
i
	full_text\
Z
X%597 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0B21597E0000000, float %596)
&floatB

	full_text


float %9
(floatB

	full_text


float %596
IcallBA
?
	full_text2
0
.%598 = tail call float @_Z3expf(float %597) #3
(floatB

	full_text


float %597
1addB*
(
	full_text

%599 = add i64 %4, 1336
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%600 = getelementptr inbounds float, float* %1, i64 %599
$i64B

	full_text


i64 %599
LstoreBC
A
	full_text4
2
0store float %598, float* %600, align 4, !tbaa !8
(floatB

	full_text


float %598
*float*B

	full_text

float* %600
ÅcallBy
w
	full_textj
h
f%601 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0AE458960000000, float 0x403A85B940000000)
&floatB

	full_text


float %9
IcallBA
?
	full_text2
0
.%602 = tail call float @_Z3expf(float %601) #3
(floatB

	full_text


float %601
1addB*
(
	full_text

%603 = add i64 %4, 1344
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%604 = getelementptr inbounds float, float* %1, i64 %603
$i64B

	full_text


i64 %603
LstoreBC
A
	full_text4
2
0store float %602, float* %604, align 4, !tbaa !8
(floatB

	full_text


float %602
*float*B

	full_text

float* %604
ÅcallBy
w
	full_textj
h
f%605 = tail call float @llvm.fmuladd.f32(float %8, float 0xBFEFAE1480000000, float 0x404465B300000000)
&floatB

	full_text


float %8
scallBk
i
	full_text\
Z
X%606 = tail call float @llvm.fmuladd.f32(float %9, float 0xC088D8A8A0000000, float %605)
&floatB

	full_text


float %9
(floatB

	full_text


float %605
IcallBA
?
	full_text2
0
.%607 = tail call float @_Z3expf(float %606) #3
(floatB

	full_text


float %606
1addB*
(
	full_text

%608 = add i64 %4, 1352
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%609 = getelementptr inbounds float, float* %1, i64 %608
$i64B

	full_text


i64 %608
LstoreBC
A
	full_text4
2
0store float %607, float* %609, align 4, !tbaa !8
(floatB

	full_text


float %607
*float*B

	full_text

float* %609
1addB*
(
	full_text

%610 = add i64 %4, 1360
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%611 = getelementptr inbounds float, float* %1, i64 %610
$i64B

	full_text


i64 %610
ZstoreBQ
O
	full_textB
@
>store float 0x427D1A94A0000000, float* %611, align 4, !tbaa !8
*float*B

	full_text

float* %611
1addB*
(
	full_text

%612 = add i64 %4, 1368
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%613 = getelementptr inbounds float, float* %1, i64 %612
$i64B

	full_text


i64 %612
ZstoreBQ
O
	full_textB
@
>store float 0x42AD2D3500000000, float* %613, align 4, !tbaa !8
*float*B

	full_text

float* %613
1addB*
(
	full_text

%614 = add i64 %4, 1376
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%615 = getelementptr inbounds float, float* %1, i64 %614
$i64B

	full_text


i64 %614
ZstoreBQ
O
	full_textB
@
>store float 0x42D23C4120000000, float* %615, align 4, !tbaa !8
*float*B

	full_text

float* %615
1addB*
(
	full_text

%616 = add i64 %4, 1384
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%617 = getelementptr inbounds float, float* %1, i64 %616
$i64B

	full_text


i64 %616
TstoreBK
I
	full_text<
:
8store float 2.000000e+10, float* %617, align 4, !tbaa !8
*float*B

	full_text

float* %617
1addB*
(
	full_text

%618 = add i64 %4, 1392
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%619 = getelementptr inbounds float, float* %1, i64 %618
$i64B

	full_text


i64 %618
ZstoreBQ
O
	full_textB
@
>store float 0x4251765920000000, float* %619, align 4, !tbaa !8
*float*B

	full_text

float* %619
1addB*
(
	full_text

%620 = add i64 %4, 1400
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%621 = getelementptr inbounds float, float* %1, i64 %620
$i64B

	full_text


i64 %620
ZstoreBQ
O
	full_textB
@
>store float 0x4251765920000000, float* %621, align 4, !tbaa !8
*float*B

	full_text

float* %621
1addB*
(
	full_text

%622 = add i64 %4, 1408
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%623 = getelementptr inbounds float, float* %1, i64 %622
$i64B

	full_text


i64 %622
ZstoreBQ
O
	full_textB
@
>store float 0x42B5D3EF80000000, float* %623, align 4, !tbaa !8
*float*B

	full_text

float* %623
ÅcallBy
w
	full_textj
h
f%624 = tail call float @llvm.fmuladd.f32(float %9, float 0xC07EA220E0000000, float 0x4036E2F780000000)
&floatB

	full_text


float %9
IcallBA
?
	full_text2
0
.%625 = tail call float @_Z3expf(float %624) #3
(floatB

	full_text


float %624
1addB*
(
	full_text

%626 = add i64 %4, 1416
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%627 = getelementptr inbounds float, float* %1, i64 %626
$i64B

	full_text


i64 %626
LstoreBC
A
	full_text4
2
0store float %625, float* %627, align 4, !tbaa !8
(floatB

	full_text


float %625
*float*B

	full_text

float* %627
1addB*
(
	full_text

%628 = add i64 %4, 1424
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%629 = getelementptr inbounds float, float* %1, i64 %628
$i64B

	full_text


i64 %628
ZstoreBQ
O
	full_textB
@
>store float 0x42DB48EB60000000, float* %629, align 4, !tbaa !8
*float*B

	full_text

float* %629
ÅcallBy
w
	full_textj
h
f%630 = tail call float @llvm.fmuladd.f32(float %8, float 0x3FFE666660000000, float 0x40328F7920000000)
&floatB

	full_text


float %8
scallBk
i
	full_text\
Z
X%631 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0AD9A7160000000, float %630)
&floatB

	full_text


float %9
(floatB

	full_text


float %630
IcallBA
?
	full_text2
0
.%632 = tail call float @_Z3expf(float %631) #3
(floatB

	full_text


float %631
1addB*
(
	full_text

%633 = add i64 %4, 1432
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%634 = getelementptr inbounds float, float* %1, i64 %633
$i64B

	full_text


i64 %633
LstoreBC
A
	full_text4
2
0store float %632, float* %634, align 4, !tbaa !8
(floatB

	full_text


float %632
*float*B

	full_text

float* %634
ÅcallBy
w
	full_textj
h
f%635 = tail call float @llvm.fmuladd.f32(float %8, float 0x3FFEB851E0000000, float 0x4032502700000000)
&floatB

	full_text


float %8
scallBk
i
	full_text\
Z
X%636 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0A65E9B00000000, float %635)
&floatB

	full_text


float %9
(floatB

	full_text


float %635
IcallBA
?
	full_text2
0
.%637 = tail call float @_Z3expf(float %636) #3
(floatB

	full_text


float %636
1addB*
(
	full_text

%638 = add i64 %4, 1440
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%639 = getelementptr inbounds float, float* %1, i64 %638
$i64B

	full_text


i64 %638
LstoreBC
A
	full_text4
2
0store float %637, float* %639, align 4, !tbaa !8
(floatB

	full_text


float %637
*float*B

	full_text

float* %639
ÅcallBy
w
	full_textj
h
f%640 = tail call float @llvm.fmuladd.f32(float %8, float 0x4000F5C280000000, float 0x402E28C640000000)
&floatB

	full_text


float %8
scallBk
i
	full_text\
Z
X%641 = tail call float @llvm.fmuladd.f32(float %9, float 0xC07B5CC6A0000000, float %640)
&floatB

	full_text


float %9
(floatB

	full_text


float %640
IcallBA
?
	full_text2
0
.%642 = tail call float @_Z3expf(float %641) #3
(floatB

	full_text


float %641
1addB*
(
	full_text

%643 = add i64 %4, 1448
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%644 = getelementptr inbounds float, float* %1, i64 %643
$i64B

	full_text


i64 %643
LstoreBC
A
	full_text4
2
0store float %642, float* %644, align 4, !tbaa !8
(floatB

	full_text


float %642
*float*B

	full_text

float* %644
ÅcallBy
w
	full_textj
h
f%645 = tail call float @llvm.fmuladd.f32(float %9, float 0x40714C4E80000000, float 0x403F51E500000000)
&floatB

	full_text


float %9
IcallBA
?
	full_text2
0
.%646 = tail call float @_Z3expf(float %645) #3
(floatB

	full_text


float %645
1addB*
(
	full_text

%647 = add i64 %4, 1456
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%648 = getelementptr inbounds float, float* %1, i64 %647
$i64B

	full_text


i64 %647
LstoreBC
A
	full_text4
2
0store float %646, float* %648, align 4, !tbaa !8
(floatB

	full_text


float %646
*float*B

	full_text

float* %648
ÅcallBy
w
	full_textj
h
f%649 = tail call float @llvm.fmuladd.f32(float %8, float 0x3FFBD70A40000000, float 0x402F42BB40000000)
&floatB

	full_text


float %8
scallBk
i
	full_text\
Z
X%650 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0B48A9D40000000, float %649)
&floatB

	full_text


float %9
(floatB

	full_text


float %649
IcallBA
?
	full_text2
0
.%651 = tail call float @_Z3expf(float %650) #3
(floatB

	full_text


float %650
1addB*
(
	full_text

%652 = add i64 %4, 1464
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%653 = getelementptr inbounds float, float* %1, i64 %652
$i64B

	full_text


i64 %652
LstoreBC
A
	full_text4
2
0store float %651, float* %653, align 4, !tbaa !8
(floatB

	full_text


float %651
*float*B

	full_text

float* %653
1addB*
(
	full_text

%654 = add i64 %4, 1472
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%655 = getelementptr inbounds float, float* %1, i64 %654
$i64B

	full_text


i64 %654
ZstoreBQ
O
	full_textB
@
>store float 0x42E6BCC420000000, float* %655, align 4, !tbaa !8
*float*B

	full_text

float* %655
1addB*
(
	full_text

%656 = add i64 %4, 1488
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%657 = getelementptr inbounds float, float* %1, i64 %656
$i64B

	full_text


i64 %656
ZstoreBQ
O
	full_textB
@
>store float 0x42835AA2E0000000, float* %657, align 4, !tbaa !8
*float*B

	full_text

float* %657
1addB*
(
	full_text

%658 = add i64 %4, 1496
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%659 = getelementptr inbounds float, float* %1, i64 %658
$i64B

	full_text


i64 %658
ZstoreBQ
O
	full_textB
@
>store float 0x429802BAA0000000, float* %659, align 4, !tbaa !8
*float*B

	full_text

float* %659
1addB*
(
	full_text

%660 = add i64 %4, 1504
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%661 = getelementptr inbounds float, float* %1, i64 %660
$i64B

	full_text


i64 %660
ZstoreBQ
O
	full_textB
@
>store float 0x42CB48EB60000000, float* %661, align 4, !tbaa !8
*float*B

	full_text

float* %661
ÅcallBy
w
	full_textj
h
f%662 = tail call float @llvm.fmuladd.f32(float %9, float 0xC099A35AC0000000, float 0x403E380240000000)
&floatB

	full_text


float %9
IcallBA
?
	full_text2
0
.%663 = tail call float @_Z3expf(float %662) #3
(floatB

	full_text


float %662
1addB*
(
	full_text

%664 = add i64 %4, 1512
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%665 = getelementptr inbounds float, float* %1, i64 %664
$i64B

	full_text


i64 %664
LstoreBC
A
	full_text4
2
0store float %663, float* %665, align 4, !tbaa !8
(floatB

	full_text


float %663
*float*B

	full_text

float* %665
ÅcallBy
w
	full_textj
h
f%666 = tail call float @llvm.fmuladd.f32(float %8, float 0xC0031EB860000000, float 0x4049903D80000000)
&floatB

	full_text


float %8
scallBk
i
	full_text\
Z
X%667 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0B5F9F660000000, float %666)
&floatB

	full_text


float %9
(floatB

	full_text


float %666
IcallBA
?
	full_text2
0
.%668 = tail call float @_Z3expf(float %667) #3
(floatB

	full_text


float %667
1addB*
(
	full_text

%669 = add i64 %4, 1520
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%670 = getelementptr inbounds float, float* %1, i64 %669
$i64B

	full_text


i64 %669
LstoreBC
A
	full_text4
2
0store float %668, float* %670, align 4, !tbaa !8
(floatB

	full_text


float %668
*float*B

	full_text

float* %670
{callBs
q
	full_textd
b
`%671 = tail call float @llvm.fmuladd.f32(float %8, float 2.500000e+00, float 0x4028164CA0000000)
&floatB

	full_text


float %8
scallBk
i
	full_text\
Z
X%672 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0939409C0000000, float %671)
&floatB

	full_text


float %9
(floatB

	full_text


float %671
IcallBA
?
	full_text2
0
.%673 = tail call float @_Z3expf(float %672) #3
(floatB

	full_text


float %672
1addB*
(
	full_text

%674 = add i64 %4, 1528
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%675 = getelementptr inbounds float, float* %1, i64 %674
$i64B

	full_text


i64 %674
LstoreBC
A
	full_text4
2
0store float %673, float* %675, align 4, !tbaa !8
(floatB

	full_text


float %673
*float*B

	full_text

float* %675
ÅcallBy
w
	full_textj
h
f%676 = tail call float @llvm.fmuladd.f32(float %8, float 0x3FFA666660000000, float 0x40329A5E60000000)
&floatB

	full_text


float %8
scallBk
i
	full_text\
Z
X%677 = tail call float @llvm.fmuladd.f32(float %9, float 0xC06491A8C0000000, float %676)
&floatB

	full_text


float %9
(floatB

	full_text


float %676
IcallBA
?
	full_text2
0
.%678 = tail call float @_Z3expf(float %677) #3
(floatB

	full_text


float %677
1addB*
(
	full_text

%679 = add i64 %4, 1536
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%680 = getelementptr inbounds float, float* %1, i64 %679
$i64B

	full_text


i64 %679
LstoreBC
A
	full_text4
2
0store float %678, float* %680, align 4, !tbaa !8
(floatB

	full_text


float %678
*float*B

	full_text

float* %680
ÅcallBy
w
	full_textj
h
f%681 = tail call float @llvm.fmuladd.f32(float %8, float 0x3FFA666660000000, float 0x40315EF0A0000000)
&floatB

	full_text


float %8
scallBk
i
	full_text\
Z
X%682 = tail call float @llvm.fmuladd.f32(float %9, float 0x407E920680000000, float %681)
&floatB

	full_text


float %9
(floatB

	full_text


float %681
IcallBA
?
	full_text2
0
.%683 = tail call float @_Z3expf(float %682) #3
(floatB

	full_text


float %682
1addB*
(
	full_text

%684 = add i64 %4, 1544
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%685 = getelementptr inbounds float, float* %1, i64 %684
$i64B

	full_text


i64 %684
LstoreBC
A
	full_text4
2
0store float %683, float* %685, align 4, !tbaa !8
(floatB

	full_text


float %683
*float*B

	full_text

float* %685
ÅcallBy
w
	full_textj
h
f%686 = tail call float @llvm.fmuladd.f32(float %8, float 0x3FE6666660000000, float 0x4039EA8DA0000000)
&floatB

	full_text


float %8
scallBk
i
	full_text\
Z
X%687 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0A71DD400000000, float %686)
&floatB

	full_text


float %9
(floatB

	full_text


float %686
IcallBA
?
	full_text2
0
.%688 = tail call float @_Z3expf(float %687) #3
(floatB

	full_text


float %687
1addB*
(
	full_text

%689 = add i64 %4, 1552
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%690 = getelementptr inbounds float, float* %1, i64 %689
$i64B

	full_text


i64 %689
LstoreBC
A
	full_text4
2
0store float %688, float* %690, align 4, !tbaa !8
(floatB

	full_text


float %688
*float*B

	full_text

float* %690
{callBs
q
	full_textd
b
`%691 = tail call float @llvm.fmuladd.f32(float %8, float 2.000000e+00, float 0x402DE4D1C0000000)
&floatB

	full_text


float %8
scallBk
i
	full_text\
Z
X%692 = tail call float @llvm.fmuladd.f32(float %9, float 0x4062BEACA0000000, float %691)
&floatB

	full_text


float %9
(floatB

	full_text


float %691
IcallBA
?
	full_text2
0
.%693 = tail call float @_Z3expf(float %692) #3
(floatB

	full_text


float %692
1addB*
(
	full_text

%694 = add i64 %4, 1560
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%695 = getelementptr inbounds float, float* %1, i64 %694
$i64B

	full_text


i64 %694
LstoreBC
A
	full_text4
2
0store float %693, float* %695, align 4, !tbaa !8
(floatB

	full_text


float %693
*float*B

	full_text

float* %695
ÅcallBy
w
	full_textj
h
f%696 = tail call float @llvm.fmuladd.f32(float %8, float 0x4004CCCCC0000000, float 0x402256CB20000000)
&floatB

	full_text


float %8
scallBk
i
	full_text\
Z
X%697 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0BB57BE60000000, float %696)
&floatB

	full_text


float %9
(floatB

	full_text


float %696
IcallBA
?
	full_text2
0
.%698 = tail call float @_Z3expf(float %697) #3
(floatB

	full_text


float %697
1addB*
(
	full_text

%699 = add i64 %4, 1568
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%700 = getelementptr inbounds float, float* %1, i64 %699
$i64B

	full_text


i64 %699
LstoreBC
A
	full_text4
2
0store float %698, float* %700, align 4, !tbaa !8
(floatB

	full_text


float %698
*float*B

	full_text

float* %700
{callBs
q
	full_textd
b
`%701 = tail call float @llvm.fmuladd.f32(float %8, float 3.500000e+00, float 0x3FE93B0AE0000000)
&floatB

	full_text


float %8
scallBk
i
	full_text\
Z
X%702 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0A64F8260000000, float %701)
&floatB

	full_text


float %9
(floatB

	full_text


float %701
IcallBA
?
	full_text2
0
.%703 = tail call float @_Z3expf(float %702) #3
(floatB

	full_text


float %702
1addB*
(
	full_text

%704 = add i64 %4, 1576
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%705 = getelementptr inbounds float, float* %1, i64 %704
$i64B

	full_text


i64 %704
LstoreBC
A
	full_text4
2
0store float %703, float* %705, align 4, !tbaa !8
(floatB

	full_text


float %703
*float*B

	full_text

float* %705
ÅcallBy
w
	full_textj
h
f%706 = tail call float @llvm.fmuladd.f32(float %8, float 0xC0075C2900000000, float 0x404C490200000000)
&floatB

	full_text


float %8
scallBk
i
	full_text\
Z
X%707 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0B894B980000000, float %706)
&floatB

	full_text


float %9
(floatB

	full_text


float %706
IcallBA
?
	full_text2
0
.%708 = tail call float @_Z3expf(float %707) #3
(floatB

	full_text


float %707
1addB*
(
	full_text

%709 = add i64 %4, 1584
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%710 = getelementptr inbounds float, float* %1, i64 %709
$i64B

	full_text


i64 %709
LstoreBC
A
	full_text4
2
0store float %708, float* %710, align 4, !tbaa !8
(floatB

	full_text


float %708
*float*B

	full_text

float* %710
1addB*
(
	full_text

%711 = add i64 %4, 1592
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%712 = getelementptr inbounds float, float* %1, i64 %711
$i64B

	full_text


i64 %711
ZstoreBQ
O
	full_textB
@
>store float 0x427A3185C0000000, float* %712, align 4, !tbaa !8
*float*B

	full_text

float* %712
1addB*
(
	full_text

%713 = add i64 %4, 1600
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%714 = getelementptr inbounds float, float* %1, i64 %713
$i64B

	full_text


i64 %713
ZstoreBQ
O
	full_textB
@
>store float 0x42D5D3EF80000000, float* %714, align 4, !tbaa !8
*float*B

	full_text

float* %714
1addB*
(
	full_text

%715 = add i64 %4, 1608
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%716 = getelementptr inbounds float, float* %1, i64 %715
$i64B

	full_text


i64 %715
ZstoreBQ
O
	full_textB
@
>store float 0x42B5D3EF80000000, float* %716, align 4, !tbaa !8
*float*B

	full_text

float* %716
1addB*
(
	full_text

%717 = add i64 %4, 1616
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%718 = getelementptr inbounds float, float* %1, i64 %717
$i64B

	full_text


i64 %717
ZstoreBQ
O
	full_textB
@
>store float 0x4234F46B00000000, float* %718, align 4, !tbaa !8
*float*B

	full_text

float* %718
1addB*
(
	full_text

%719 = add i64 %4, 1624
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%720 = getelementptr inbounds float, float* %1, i64 %719
$i64B

	full_text


i64 %719
ZstoreBQ
O
	full_textB
@
>store float 0x42B5D3EF80000000, float* %720, align 4, !tbaa !8
*float*B

	full_text

float* %720
1addB*
(
	full_text

%721 = add i64 %4, 1632
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%722 = getelementptr inbounds float, float* %1, i64 %721
$i64B

	full_text


i64 %721
ZstoreBQ
O
	full_textB
@
>store float 0x42A4024620000000, float* %722, align 4, !tbaa !8
*float*B

	full_text

float* %722
ÅcallBy
w
	full_textj
h
f%723 = tail call float @llvm.fmuladd.f32(float %8, float 0xC014E147A0000000, float 0x4052C2CC00000000)
&floatB

	full_text


float %8
scallBk
i
	full_text\
Z
X%724 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0C3688280000000, float %723)
&floatB

	full_text


float %9
(floatB

	full_text


float %723
IcallBA
?
	full_text2
0
.%725 = tail call float @_Z3expf(float %724) #3
(floatB

	full_text


float %724
1addB*
(
	full_text

%726 = add i64 %4, 1640
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%727 = getelementptr inbounds float, float* %1, i64 %726
$i64B

	full_text


i64 %726
LstoreBC
A
	full_text4
2
0store float %725, float* %727, align 4, !tbaa !8
(floatB

	full_text


float %725
*float*B

	full_text

float* %727
"retB

	full_text


ret void
*float*8B

	full_text

	float* %1
*float*8B

	full_text

	float* %0
(float8B

	full_text


float %2
-; undefined function B

	full_text

 
-; undefined function B

	full_text

 
-; undefined function B

	full_text

 
-; undefined function B

	full_text

 
&i648B

	full_text


i64 1400
8float8B+
)
	full_text

float 0xC07FC3FB40000000
&i648B

	full_text


i64 1544
&i648B

	full_text


i64 1576
%i648B

	full_text
	
i64 144
&i648B

	full_text


i64 1272
8float8B+
)
	full_text

float 0xBFD7AE1480000000
8float8B+
)
	full_text

float 0xC0988824E0000000
8float8B+
)
	full_text

float 0x3FFEE147A0000000
8float8B+
)
	full_text

float 0x402256CB20000000
%i648B

	full_text
	
i64 552
8float8B+
)
	full_text

float 0x42D5D3EF80000000
%i648B

	full_text
	
i64 216
8float8B+
)
	full_text

float 0xC014E147A0000000
$i648B

	full_text


i64 88
%i648B

	full_text
	
i64 544
8float8B+
)
	full_text

float 0x40303D8520000000
&i648B

	full_text


i64 1480
&i648B

	full_text


i64 1368
8float8B+
)
	full_text

float 0xC0DDE0E4C0000000
%i648B

	full_text
	
i64 128
8float8B+
)
	full_text

float 0xC0B0E7A9E0000000
8float8B+
)
	full_text

float 0x403520F480000000
&i648B

	full_text


i64 1296
%i648B

	full_text
	
i64 376
%i648B

	full_text
	
i64 840
8float8B+
)
	full_text

float 0x42A2309CE0000000
$i648B

	full_text


i64 72
%i648B

	full_text
	
i64 792
8float8B+
)
	full_text

float 0x403330D780000000
&i648B

	full_text


i64 1144
&i648B

	full_text


i64 1048
8float8B+
)
	full_text

float 0x4003333340000000
%i648B

	full_text
	
i64 952
%i648B

	full_text
	
i64 504
%i648B

	full_text
	
i64 272
%i648B

	full_text
	
i64 312
8float8B+
)
	full_text

float 0x403C30CDA0000000
8float8B+
)
	full_text

float 0x40410400E0000000
&i648B

	full_text


i64 1280
8float8B+
)
	full_text

float 0xC0B974A7E0000000
8float8B+
)
	full_text

float 0x3FF9EB8520000000
&i648B

	full_text


i64 1128
%i648B

	full_text
	
i64 584
8float8B+
)
	full_text

float 0xBFE3333340000000
8float8B+
)
	full_text

float 0xC06420F040000000
8float8B+
)
	full_text

float 0x42A5D3EF80000000
&i648B

	full_text


i64 1072
&i648B

	full_text


i64 1024
&i648B

	full_text


i64 1176
%i648B

	full_text
	
i64 104
8float8B+
)
	full_text

float 0x403DA8BF60000000
%i648B

	full_text
	
i64 600
8float8B+
)
	full_text

float 0x3FDEB851E0000000
%i648B

	full_text
	
i64 112
&i648B

	full_text


i64 1248
8float8B+
)
	full_text

float 0x40412866A0000000
8float8B+
)
	full_text

float 0x403EA072E0000000
%i648B

	full_text
	
i64 320
8float8B+
)
	full_text

float 0xC08A42F980000000
8float8B+
)
	full_text

float 0x403E380240000000
8float8B+
)
	full_text

float 0x4299774200000000
&i648B

	full_text


i64 1608
8float8B+
)
	full_text

float 0xC0979699A0000000
8float8B+
)
	full_text

float 0xC0D18EFBA0000000
%i648B

	full_text
	
i64 488
8float8B+
)
	full_text

float 0x4037376AA0000000
%i648B

	full_text
	
i64 824
8float8B+
)
	full_text

float 0x426D1A94A0000000
%i648B

	full_text
	
i64 720
8float8B+
)
	full_text

float 0xBFE428F5C0000000
&i648B

	full_text


i64 1336
%i648B

	full_text
	
i64 800
&i648B

	full_text


i64 1120
%i648B

	full_text
	
i64 352
8float8B+
)
	full_text

float 0x4210C388C0000000
8float8B+
)
	full_text

float 0x40315EF0A0000000
%i648B

	full_text
	
i64 328
8float8B+
)
	full_text

float 0x427D1A94A0000000
%i648B

	full_text
	
i64 240
8float8B+
)
	full_text

float 0x4251765920000000
8float8B+
)
	full_text

float 0xBFE0A3D700000000
2float8B%
#
	full_text

float 2.000000e+10
8float8B+
)
	full_text

float 0x403D6F9F60000000
8float8B+
)
	full_text

float 0xC0A64F8260000000
8float8B+
)
	full_text

float 0x3FDD0E5600000000
&i648B

	full_text


i64 1096
8float8B+
)
	full_text

float 0x42CFD512A0000000
%i648B

	full_text
	
i64 712
8float8B+
)
	full_text

float 0x427A3185C0000000
&i648B

	full_text


i64 1344
&i648B

	full_text


i64 1264
8float8B+
)
	full_text

float 0x4033C57700000000
8float8B+
)
	full_text

float 0x405BAD4A60000000
8float8B+
)
	full_text

float 0x42B2309CE0000000
%i648B

	full_text
	
i64 416
%i648B

	full_text
	
i64 904
$i648B

	full_text


i64 96
2float8B%
#
	full_text

float 1.632000e+07
8float8B+
)
	full_text

float 0x403BB53E60000000
&i648B

	full_text


i64 1328
2float8B%
#
	full_text

float 4.080000e+06
8float8B+
)
	full_text

float 0xC0B0B55780000000
%i648B

	full_text
	
i64 968
2float8B%
#
	full_text

float 2.500000e+00
8float8B+
)
	full_text

float 0x3FFE8F5C20000000
8float8B+
)
	full_text

float 0xC0D234D200000000
8float8B+
)
	full_text

float 0x4492A27D60000000
%i648B

	full_text
	
i64 736
8float8B+
)
	full_text

float 0x40384E8980000000
8float8B+
)
	full_text

float 0xC0D77D7060000000
%i648B

	full_text
	
i64 288
8float8B+
)
	full_text

float 0x40067AE140000000
8float8B+
)
	full_text

float 0x402C376360000000
$i648B

	full_text


i64 64
8float8B+
)
	full_text

float 0x4024F73F80000000
8float8B+
)
	full_text

float 0x403F5F99E0000000
8float8B+
)
	full_text

float 0xC0B5F9F660000000
8float8B+
)
	full_text

float 0xBFEF0A3D80000000
%i648B

	full_text
	
i64 648
%i648B

	full_text
	
i64 384
8float8B+
)
	full_text

float 0x403193A340000000
%i648B

	full_text
	
i64 432
8float8B+
)
	full_text

float 0x403D3D0B80000000
8float8B+
)
	full_text

float 0x40405221C0000000
2float8B%
#
	full_text

float 1.920000e+07
8float8B+
)
	full_text

float 0x402E28C640000000
8float8B+
)
	full_text

float 0x4077BEDB80000000
8float8B+
)
	full_text

float 0x406F737780000000
8float8B+
)
	full_text

float 0x404384F060000000
8float8B+
)
	full_text

float 0x40432F0780000000
8float8B+
)
	full_text

float 0x402DE4D1C0000000
8float8B+
)
	full_text

float 0x4031D742C0000000
&i648B

	full_text


i64 1624
8float8B+
)
	full_text

float 0xC0CC4E51E0000000
8float8B+
)
	full_text

float 0xC0B48A9D40000000
&i648B

	full_text


i64 1552
8float8B+
)
	full_text

float 0x42B05EF3A0000000
8float8B+
)
	full_text

float 0xC0AD9A7160000000
&i648B

	full_text


i64 1592
8float8B+
)
	full_text

float 0x4039973EC0000000
8float8B+
)
	full_text

float 0x4283356220000000
8float8B+
)
	full_text

float 0x3FD147AE20000000
&i648B

	full_text


i64 1520
%i648B

	full_text
	
i64 344
8float8B+
)
	full_text

float 0xC09BD58C40000000
8float8B+
)
	full_text

float 0xC0D3A82AA0000000
8float8B+
)
	full_text

float 0x409BC16B60000000
%i648B

	full_text
	
i64 864
%i648B

	full_text
	
i64 992
8float8B+
)
	full_text

float 0xC0BC54DCA0000000
%i648B

	full_text
	
i64 920
8float8B+
)
	full_text

float 0x4040B70E00000000
%i648B

	full_text
	
i64 616
8float8B+
)
	full_text

float 0xC0A709B300000000
&i648B

	full_text


i64 1288
8float8B+
)
	full_text

float 0x42AB48EB60000000
8float8B+
)
	full_text

float 0x40301E3B80000000
8float8B+
)
	full_text

float 0xC0D70C3720000000
&i648B

	full_text


i64 1600
%i648B

	full_text
	
i64 392
8float8B+
)
	full_text

float 0x3FF99999A0000000
8float8B+
)
	full_text

float 0xC07B5CC6A0000000
&i648B

	full_text


i64 1432
&i648B

	full_text


i64 1616
8float8B+
)
	full_text

float 0x3FF0CCCCC0000000
8float8B+
)
	full_text

float 0xC0C731F4E0000000
8float8B+
)
	full_text

float 0x40055C2900000000
%i648B

	full_text
	
i64 136
%i648B

	full_text
	
i64 656
3float8B&
$
	full_text

float -1.250000e+00
%i648B

	full_text
	
i64 232
%i648B

	full_text
	
i64 248
8float8B+
)
	full_text

float 0xC0C0B55780000000
8float8B+
)
	full_text

float 0x404C490200000000
&i648B

	full_text


i64 1200
%i648B

	full_text
	
i64 752
8float8B+
)
	full_text

float 0x4070328160000000
8float8B+
)
	full_text

float 0xBFEB851EC0000000
&i648B

	full_text


i64 1240
8float8B+
)
	full_text

float 0x4049903D80000000
%i648B

	full_text
	
i64 448
8float8B+
)
	full_text

float 0x40605AC340000000
8float8B+
)
	full_text

float 0xC088D8A8A0000000
&i648B

	full_text


i64 1320
&i648B

	full_text


i64 1384
8float8B+
)
	full_text

float 0x3FFA666660000000
&i648B

	full_text


i64 1464
8float8B+
)
	full_text

float 0xC06491A8C0000000
8float8B+
)
	full_text

float 0xC062DEE140000000
8float8B+
)
	full_text

float 0x42DB48EB60000000
8float8B+
)
	full_text

float 0x3FF6E147A0000000
8float8B+
)
	full_text

float 0x42D32AE7E0000000
8float8B+
)
	full_text

float 0x402F42BB40000000
8float8B+
)
	full_text

float 0x42D6BCC420000000
8float8B+
)
	full_text

float 0xC095269C80000000
8float8B+
)
	full_text

float 0x40301494C0000000
%i648B

	full_text
	
i64 496
3float8B&
$
	full_text

float -1.000000e+00
2float8B%
#
	full_text

float 4.000000e+00
8float8B+
)
	full_text

float 0x4031BDCEC0000000
8float8B+
)
	full_text

float 0x40581D7280000000
&i648B

	full_text


i64 1392
&i648B

	full_text


i64 1032
8float8B+
)
	full_text

float 0xC068176C60000000
8float8B+
)
	full_text

float 0xBFF3D70A40000000
&i648B

	full_text


i64 1632
8float8B+
)
	full_text

float 0xC0B79699A0000000
8float8B+
)
	full_text

float 0x4028AA5860000000
8float8B+
)
	full_text

float 0xC075B38320000000
&i648B

	full_text


i64 1528
%i648B

	full_text
	
i64 296
$i648B

	full_text


i64 80
8float8B+
)
	full_text

float 0xBFFB851EC0000000
8float8B+
)
	full_text

float 0xC0B2CAC060000000
8float8B+
)
	full_text

float 0xC0C24C71A0000000
&i648B

	full_text


i64 1192
%i648B

	full_text
	
i64 832
8float8B+
)
	full_text

float 0x404465B300000000
8float8B+
)
	full_text

float 0x4234F46B00000000
8float8B+
)
	full_text

float 0x4046202420000000
8float8B+
)
	full_text

float 0x42D476B080000000
8float8B+
)
	full_text

float 0x40400661E0000000
&i648B

	full_text


i64 1256
8float8B+
)
	full_text

float 0x42AD1A94A0000000
8float8B+
)
	full_text

float 0x4071ED5600000000
8float8B+
)
	full_text

float 0x3FB99999A0000000
%i648B

	full_text
	
i64 960
8float8B+
)
	full_text

float 0xBFF63D70A0000000
%i648B

	full_text
	
i64 928
8float8B+
)
	full_text

float 0x3FF828F5C0000000
8float8B+
)
	full_text

float 0xC07ADBF3E0000000
&i648B

	full_text


i64 1488
8float8B+
)
	full_text

float 0xC0BB57BE60000000
%i648B

	full_text
	
i64 592
8float8B+
)
	full_text

float 0x408F737780000000
%i648B

	full_text
	
i64 528
8float8B+
)
	full_text

float 0xC027A3D700000000
&i648B

	full_text


i64 1584
8float8B+
)
	full_text

float 0x42AD2D3500000000
%i648B

	full_text
	
i64 744
8float8B+
)
	full_text

float 0xC099C02360000000
2float8B%
#
	full_text

float 3.500000e+00
8float8B+
)
	full_text

float 0xBFE99999A0000000
%i648B

	full_text
	
i64 664
8float8B+
)
	full_text

float 0x403CDAD400000000
&i648B

	full_text


i64 1232
8float8B+
)
	full_text

float 0xC0075C2900000000
8float8B+
)
	full_text

float 0x42D23C4120000000
8float8B+
)
	full_text

float 0x403C8C1CA0000000
8float8B+
)
	full_text

float 0xC0A79699A0000000
8float8B+
)
	full_text

float 0x4025A3BA00000000
8float8B+
)
	full_text

float 0xC01E8ABEE0000000
8float8B+
)
	full_text

float 0x403DEF00E0000000
8float8B+
)
	full_text

float 0xC03C7ACA80000000
8float8B+
)
	full_text

float 0xC0619CD240000000
%i648B

	full_text
	
i64 224
8float8B+
)
	full_text

float 0x4000F5C280000000
%i648B

	full_text
	
i64 632
8float8B+
)
	full_text

float 0x42CB48EB60000000
8float8B+
)
	full_text

float 0x429ED99D80000000
8float8B+
)
	full_text

float 0x403F4B69C0000000
8float8B+
)
	full_text

float 0x407E920680000000
8float8B+
)
	full_text

float 0x42B6BF1820000000
8float8B+
)
	full_text

float 0x42D489E5E0000000
%i648B

	full_text
	
i64 120
%i648B

	full_text
	
i64 688
8float8B+
)
	full_text

float 0x403E70BFA0000000
3float8B&
$
	full_text

float -0.000000e+00
8float8B+
)
	full_text

float 0x42C9EBAC60000000
%i648B

	full_text
	
i64 512
8float8B+
)
	full_text

float 0x406C1E02E0000000
%i648B

	full_text
	
i64 360
8float8B+
)
	full_text

float 0x403C19DCC0000000
%i648B

	full_text
	
i64 816
%i648B

	full_text
	
i64 856
&i648B

	full_text


i64 1504
8float8B+
)
	full_text

float 0x4040FF3D00000000
8float8B+
)
	full_text

float 0x42B5D3EF80000000
8float8B+
)
	full_text

float 0xC0419CD240000000
%i648B

	full_text
	
i64 912
8float8B+
)
	full_text

float 0xC072DEE140000000
8float8B+
)
	full_text

float 0xC09F737780000000
8float8B+
)
	full_text

float 0x439BC16D60000000
&i648B

	full_text


i64 1056
8float8B+
)
	full_text

float 0x3FF9C28F60000000
8float8B+
)
	full_text

float 0xC0A45D5320000000
8float8B+
)
	full_text

float 0x4062BEACA0000000
%i648B

	full_text
	
i64 696
&i648B

	full_text


i64 1640
8float8B+
)
	full_text

float 0xBFE851EB80000000
%i648B

	full_text
	
i64 480
8float8B+
)
	full_text

float 0x3FF2E147A0000000
8float8B+
)
	full_text

float 0x42A4024620000000
&i648B

	full_text


i64 1560
%i648B

	full_text
	
i64 888
8float8B+
)
	full_text

float 0x403A85B940000000
8float8B+
)
	full_text

float 0x42C6BCC420000000
%i648B

	full_text
	
i64 176
8float8B+
)
	full_text

float 0x437AA535E0000000
8float8B+
)
	full_text

float 0xC0E38F0180000000
8float8B+
)
	full_text

float 0xC069292C60000000
%i648B

	full_text
	
i64 568
8float8B+
)
	full_text

float 0xC0B0419A20000000
&i648B

	full_text


i64 1008
%i648B

	full_text
	
i64 944
8float8B+
)
	full_text

float 0x42A05EF3A0000000
&i648B

	full_text


i64 1000
8float8B+
)
	full_text

float 0xC0B192C1C0000000
&i648B

	full_text


i64 1416
&i648B

	full_text


i64 1536
8float8B+
)
	full_text

float 0xC006A3D700000000
$i648B

	full_text


i64 40
%i648B

	full_text
	
i64 560
8float8B+
)
	full_text

float 0x4040D5EC60000000
8float8B+
)
	full_text

float 0xC0AC6C8360000000
8float8B+
)
	full_text

float 0x4024367DC0000000
%i648B

	full_text
	
i64 704
%i648B

	full_text
	
i64 184
8float8B+
)
	full_text

float 0x42404C5340000000
8float8B+
)
	full_text

float 0xC0AF737780000000
%i648B

	full_text
	
i64 984
8float8B+
)
	full_text

float 0x4047933D80000000
8float8B+
)
	full_text

float 0x443DD0C880000000
8float8B+
)
	full_text

float 0xC016243B80000000
8float8B+
)
	full_text

float 0xC0A8BA7740000000
8float8B+
)
	full_text

float 0x42D2309CE0000000
8float8B+
)
	full_text

float 0xC0B21597E0000000
8float8B+
)
	full_text

float 0xC0BB850880000000
8float8B+
)
	full_text

float 0x40714C4E80000000
&i648B

	full_text


i64 1424
8float8B+
)
	full_text

float 0xBFEFAE1480000000
8float8B+
)
	full_text

float 0x42C5D3EF80000000
8float8B+
)
	full_text

float 0x403D5F8CA0000000
8float8B+
)
	full_text

float 0x40465A3140000000
8float8B+
)
	full_text

float 0x40304F0800000000
8float8B+
)
	full_text

float 0xC0A54EDE60000000
8float8B+
)
	full_text

float 0x40328F7920000000
%i648B

	full_text
	
i64 440
8float8B+
)
	full_text

float 0xC0031EB860000000
&i648B

	full_text


i64 1216
8float8B+
)
	full_text

float 0x42E6BCC420000000
8float8B+
)
	full_text

float 0x4004CCCCC0000000
%i648B

	full_text
	
i64 768
&i648B

	full_text


i64 1016
8float8B+
)
	full_text

float 0x3FE93B0AE0000000
&i648B

	full_text


i64 1040
&i648B

	full_text


i64 1472
8float8B+
)
	full_text

float 0xC0AE458960000000
8float8B+
)
	full_text

float 0xC0939409C0000000
#i648B

	full_text	

i64 8
&i648B

	full_text


i64 1112
8float8B+
)
	full_text

float 0x403DB5E0E0000000
&i648B

	full_text


i64 1376
8float8B+
)
	full_text

float 0x3FFEB851E0000000
%i648B

	full_text
	
i64 680
&i648B

	full_text


i64 1168
%i648B

	full_text
	
i64 152
8float8B+
)
	full_text

float 0xC0E5CFD160000000
8float8B+
)
	full_text

float 0x4028164CA0000000
%i648B

	full_text
	
i64 848
$i648B

	full_text


i64 16
%i648B

	full_text
	
i64 456
8float8B+
)
	full_text

float 0xC0A65E9B00000000
8float8B+
)
	full_text

float 0x3FFCA3D700000000
8float8B+
)
	full_text

float 0x4037DBD7C0000000
8float8B+
)
	full_text

float 0xC094717400000000
%i648B

	full_text
	
i64 472
8float8B+
)
	full_text

float 0x3FF3A5E360000000
8float8B+
)
	full_text

float 0x4040172080000000
8float8B+
)
	full_text

float 0xC0853ABD80000000
%i648B

	full_text
	
i64 304
$i648B

	full_text


i64 56
8float8B+
)
	full_text

float 0x4042E0FAC0000000
&i648B

	full_text


i64 1496
8float8B+
)
	full_text

float 0x403E56CD60000000
8float8B+
)
	full_text

float 0x4039EA8DA0000000
&i648B

	full_text


i64 1304
8float8B+
)
	full_text

float 0x3FE6666660000000
&i648B

	full_text


i64 1512
%i648B

	full_text
	
i64 168
8float8B+
)
	full_text

float 0x4031ADA7E0000000
8float8B+
)
	full_text

float 0xC0879699A0000000
%i648B

	full_text
	
i64 728
8float8B+
)
	full_text

float 0x4046C53B60000000
%i648B

	full_text
	
i64 776
%i648B

	full_text
	
i64 784
8float8B+
)
	full_text

float 0x4292309CE0000000
8float8B+
)
	full_text

float 0xC0751A88C0000000
8float8B+
)
	full_text

float 0x42B6BCC420000000
8float8B+
)
	full_text

float 0x4089A1F200000000
8float8B+
)
	full_text

float 0x403A6D5300000000
8float8B+
)
	full_text

float 0x403C52FCC0000000
&i648B

	full_text


i64 1208
8float8B+
)
	full_text

float 0x4256D14160000000
8float8B+
)
	full_text

float 0x4072BEACA0000000
8float8B+
)
	full_text

float 0x42BB48EB60000000
8float8B+
)
	full_text

float 0x402D6E6C80000000
8float8B+
)
	full_text

float 0xC0B894B980000000
8float8B+
)
	full_text

float 0xC0B54EDE60000000
8float8B+
)
	full_text

float 0x40344EC8C0000000
8float8B+
)
	full_text

float 0x42A85FDC80000000
8float8B+
)
	full_text

float 0x40401E3B80000000
8float8B+
)
	full_text

float 0xC09D681F20000000
8float8B+
)
	full_text

float 0x42BE036940000000
%i648B

	full_text
	
i64 640
8float8B+
)
	full_text

float 0x408DE0E4C0000000
8float8B+
)
	full_text

float 0x42D0B07140000000
&i648B

	full_text


i64 1448
2float8B%
#
	full_text

float 1.500000e+00
%i648B

	full_text
	
i64 408
8float8B+
)
	full_text

float 0x40681DDD60000000
8float8B+
)
	full_text

float 0xC0A71DD400000000
%i648B

	full_text
	
i64 536
8float8B+
)
	full_text

float 0x403D028160000000
8float8B+
)
	full_text

float 0x429B48EB60000000
%i648B

	full_text
	
i64 256
8float8B+
)
	full_text

float 0x42B9774200000000
&i648B

	full_text


i64 1088
8float8B+
)
	full_text

float 0x402A3EA660000000
8float8B+
)
	full_text

float 0xC0737FE8C0000000
8float8B+
)
	full_text

float 0x42BB6287E0000000
8float8B+
)
	full_text

float 0x403B6B98C0000000
8float8B+
)
	full_text

float 0xBFAEB851E0000000
%i648B

	full_text
	
i64 608
8float8B+
)
	full_text

float 0x3FDC28F5C0000000
%i648B

	full_text
	
i64 336
8float8B+
)
	full_text

float 0x407F737780000000
8float8B+
)
	full_text

float 0xC099A35AC0000000
8float8B+
)
	full_text

float 0xC0224B43A0000000
%i648B

	full_text
	
i64 624
%i648B

	full_text
	
i64 160
$i648B

	full_text


i64 48
8float8B+
)
	full_text

float 0x4003C28F60000000
&i648B

	full_text


i64 1352
8float8B+
)
	full_text

float 0x4032502700000000
8float8B+
)
	full_text

float 0x3FFBD70A40000000
8float8B+
)
	full_text

float 0x3FFD47AE20000000
8float8B+
)
	full_text

float 0x403F51E500000000
8float8B+
)
	full_text

float 0x402E316120000000
8float8B+
)
	full_text

float 0x403F0F3C00000000
&i648B

	full_text


i64 1152
8float8B+
)
	full_text

float 0xC09C4E51E0000000
%i648B

	full_text
	
i64 400
8float8B+
)
	full_text

float 0x403F77E3E0000000
&i648B

	full_text


i64 1104
8float8B+
)
	full_text

float 0x404BD570E0000000
8float8B+
)
	full_text

float 0x4052C2CC00000000
&i648B

	full_text


i64 1136
%i648B

	full_text
	
i64 872
&i648B

	full_text


i64 1184
%i648B

	full_text
	
i64 464
8float8B+
)
	full_text

float 0x405FDB8F80000000
%i648B

	full_text
	
i64 264
8float8B+
)
	full_text

float 0x4020372720000000
&i648B

	full_text


i64 1312
%i648B

	full_text
	
i64 200
%i648B

	full_text
	
i64 760
8float8B+
)
	full_text

float 0xC07EA220E0000000
8float8B+
)
	full_text

float 0x403FE410C0000000
&i648B

	full_text


i64 1456
8float8B+
)
	full_text

float 0x403FEF61C0000000
8float8B+
)
	full_text

float 0x42C2309CE0000000
8float8B+
)
	full_text

float 0x3FFE666660000000
2float8B%
#
	full_text

float 2.000000e+00
%i648B

	full_text
	
i64 520
&i648B

	full_text


i64 1568
%i648B

	full_text
	
i64 208
8float8B+
)
	full_text

float 0x43ABC16D60000000
%i648B

	full_text
	
i64 192
8float8B+
)
	full_text

float 0xC0A1BB03A0000000
#i328B

	full_text	

i32 0
8float8B+
)
	full_text

float 0x403BB79A60000000
8float8B+
)
	full_text

float 0x403B03CC40000000
8float8B+
)
	full_text

float 0xC020DCAE20000000
%i648B

	full_text
	
i64 808
&i648B

	full_text


i64 1360
8float8B+
)
	full_text

float 0xC08F737780000000
2float8B%
#
	full_text

float 4.500000e+00
&i648B

	full_text


i64 1224
8float8B+
)
	full_text

float 0x4042CBE020000000
%i648B

	full_text
	
i64 936
$i648B

	full_text


i64 32
2float8B%
#
	full_text

float 3.840000e+05
8float8B+
)
	full_text

float 0x3FD28F5C20000000
8float8B+
)
	full_text

float 0x40428A49E0000000
8float8B+
)
	full_text

float 0x42A9774200000000
8float8B+
)
	full_text

float 0x429802BAA0000000
8float8B+
)
	full_text

float 0x42A3356220000000
&i648B

	full_text


i64 1160
8float8B+
)
	full_text

float 0x40326BB1C0000000
8float8B+
)
	full_text

float 0x4036E2F780000000
8float8B+
)
	full_text

float 0xC079CA33E0000000
8float8B+
)
	full_text

float 0x40453CF280000000
&i648B

	full_text


i64 1408
8float8B+
)
	full_text

float 0x40329A5E60000000
8float8B+
)
	full_text

float 0x42835AA2E0000000
8float8B+
)
	full_text

float 0x4090972600000000
%i648B

	full_text
	
i64 424
8float8B+
)
	full_text

float 0xC0C3688280000000
&i648B

	full_text


i64 1064
8float8B+
)
	full_text

float 0xC093A82AA0000000
&i648B

	full_text


i64 1440
8float8B+
)
	full_text

float 0x4035F4B100000000
8float8B+
)
	full_text

float 0x4034BE39C0000000
8float8B+
)
	full_text

float 0x42BD1A94A0000000
&i648B

	full_text


i64 1080
%i648B

	full_text
	
i64 280
8float8B+
)
	full_text

float 0xC0A4717400000000
8float8B+
)
	full_text

float 0xC09AF82200000000
8float8B+
)
	full_text

float 0x4043E28BA0000000
8float8B+
)
	full_text

float 0x403087BB80000000
8float8B+
)
	full_text

float 0xC0D8F08FC0000000
%i648B

	full_text
	
i64 576
$i648B

	full_text


i64 24
%i648B

	full_text
	
i64 672
2float8B%
#
	full_text

float 1.000000e+00
8float8B+
)
	full_text

float 0x403285B7C0000000
2float8B%
#
	full_text

float 5.000000e-01
%i648B

	full_text
	
i64 976
8float8B+
)
	full_text

float 0xC08C9ED5A0000000
%i648B

	full_text
	
i64 368
8float8B+
)
	full_text

float 0xC0B4D618C0000000
%i648B

	full_text
	
i64 880
%i648B

	full_text
	
i64 896
8float8B+
)
	full_text

float 0x401E666660000000       	  
 

                       !" !! #$ #% ## &' && () (* (( +, ++ -. -- /0 // 12 13 11 45 44 67 68 66 9: 99 ;< ;; => == ?@ ?A ?? BC BB DE DD FG FF HI HJ HH KL KK MN MM OP OO QR QQ ST SU SS VW VV XY XX Z[ ZZ \] \\ ^_ ^` ^^ ab aa cd cc ef ee gh gi gg jk jj lm ll no nn pq pr pp st ss uv uu wx ww yz y{ yy |} || ~ ~~ Ä
Å ÄÄ ÇÉ Ç
Ñ ÇÇ ÖÜ ÖÖ áà áá âä ââ ã
å ãã çé ç
è çç êë êê íì íí îï îî ñ
ó ññ òô ò
ö òò õú õõ ùû ùù ü† üü °
¢ °° £§ £
• ££ ¶ß ¶¶ ®© ®® ™´ ™™ ¨
≠ ¨¨ ÆØ Æ
∞ ÆÆ ±≤ ±± ≥¥ ≥≥ µ∂ µµ ∑
∏ ∑∑ π∫ π
ª ππ ºΩ ºº æø ææ ¿¡ ¿¿ ¬
√ ¬¬ ƒ≈ ƒ
∆ ƒƒ «» «« …  …… ÀÃ ÀÀ Õ
Œ ÕÕ œ– œ
— œœ “” ““ ‘’ ‘‘ ÷◊ ÷÷ ÿ
Ÿ ÿÿ ⁄€ ⁄
‹ ⁄⁄ ›ﬁ ›› ﬂ
‡ ﬂﬂ ·
‚ ·· „‰ „„ ÂÊ ÂÂ ÁË ÁÁ È
Í ÈÈ ÎÏ Î
Ì ÎÎ ÓÔ ÓÓ Ò  ÚÛ ÚÚ Ù
ı ÙÙ ˆ˜ ˆ
¯ ˆˆ ˘˙ ˘˘ ˚¸ ˚˚ ˝˛ ˝˝ ˇ
Ä ˇˇ ÅÇ Å
É ÅÅ ÑÖ ÑÑ Üá Ü
à ÜÜ âä ââ ãå ãã ç
é çç èê è
ë èè íì íí îï îî ñó ññ ò
ô òò öõ ö
ú öö ùû ùù ü† ü
° üü ¢£ ¢¢ §• §§ ¶
ß ¶¶ ®© ®
™ ®® ´¨ ´´ ≠Æ ≠≠ Ø∞ ØØ ±
≤ ±± ≥¥ ≥
µ ≥≥ ∂∑ ∂∂ ∏π ∏∏ ∫ª ∫∫ º
Ω ºº æø æ
¿ ææ ¡¬ ¡¡ √ƒ √√ ≈∆ ≈≈ «
» «« …  …
À …… ÃÕ ÃÃ Œœ Œ
– ŒŒ —“ —— ”‘ ”” ’
÷ ’’ ◊ÿ ◊
Ÿ ◊◊ ⁄€ ⁄⁄ ‹› ‹
ﬁ ‹‹ ﬂ‡ ﬂﬂ ·‚ ·· „
‰ „„ ÂÊ Â
Á ÂÂ ËÈ ËË ÍÎ ÍÍ ÏÌ ÏÏ Ó
Ô ÓÓ Ò 
Ú  ÛÙ ÛÛ ıˆ ıı ˜¯ ˜˜ ˘
˙ ˘˘ ˚¸ ˚
˝ ˚˚ ˛ˇ ˛˛ Ä
Å ÄÄ Ç
É ÇÇ ÑÖ ÑÑ Ü
á ÜÜ à
â àà äã ää åç å
é åå èê èè ëí ëë ì
î ìì ïñ ï
ó ïï òô òò öõ öö úù úú û
ü ûû †° †
¢ †† £§ ££ •
¶ •• ß
® ßß ©™ ©© ´
¨ ´´ ≠
Æ ≠≠ Ø∞ ØØ ±≤ ±± ≥¥ ≥≥ µ
∂ µµ ∑∏ ∑
π ∑∑ ∫ª ∫∫ ºΩ º
æ ºº ø¿ øø ¡¬ ¡¡ √
ƒ √√ ≈∆ ≈
« ≈≈ »… »»  
À    Ã
Õ ÃÃ Œœ ŒŒ –
— –– “
” ““ ‘’ ‘‘ ÷
◊ ÷÷ ÿ
Ÿ ÿÿ ⁄€ ⁄⁄ ‹
› ‹‹ ﬁ
ﬂ ﬁﬁ ‡· ‡‡ ‚„ ‚
‰ ‚‚ ÂÊ ÂÂ ÁË ÁÁ È
Í ÈÈ ÎÏ Î
Ì ÎÎ ÓÔ ÓÓ Ò  ÚÛ ÚÚ Ù
ı ÙÙ ˆ˜ ˆ
¯ ˆˆ ˘˙ ˘˘ ˚¸ ˚˚ ˝˛ ˝˝ ˇ
Ä ˇˇ ÅÇ Å
É ÅÅ ÑÖ ÑÑ Üá Ü
à ÜÜ âä ââ ãå ãã ç
é çç èê è
ë èè íì íí î
ï îî ñ
ó ññ òô òò öõ öö úù úú ûü ûû †
° †† ¢£ ¢
§ ¢¢ •¶ •• ß® ßß ©
™ ©© ´¨ ´
≠ ´´ ÆØ ÆÆ ∞
± ∞∞ ≤
≥ ≤≤ ¥µ ¥¥ ∂∑ ∂
∏ ∂∂ π∫ ππ ªº ªª Ω
æ ΩΩ ø¿ ø
¡ øø ¬√ ¬¬ ƒ
≈ ƒƒ ∆
« ∆∆ »… »»  À  
Ã    ÕŒ ÕÕ œ– œœ —
“ —— ”‘ ”
’ ”” ÷◊ ÷÷ ÿ
Ÿ ÿÿ ⁄
€ ⁄⁄ ‹› ‹‹ ﬁ
ﬂ ﬁﬁ ‡
· ‡‡ ‚„ ‚‚ ‰Â ‰‰ ÊÁ ÊÊ Ë
È ËË ÍÎ Í
Ï ÍÍ ÌÓ ÌÌ Ô
 ÔÔ Ò
Ú ÒÒ ÛÙ ÛÛ ı
ˆ ıı ˜
¯ ˜˜ ˘˙ ˘˘ ˚
¸ ˚˚ ˝
˛ ˝˝ ˇÄ ˇˇ Å
Ç ÅÅ É
Ñ ÉÉ ÖÜ ÖÖ á
à áá â
ä ââ ãå ãã ç
é çç è
ê èè ëí ëë ì
î ìì ï
ñ ïï óò óó ô
ö ôô õ
ú õõ ùû ùù ü
† üü °
¢ °° £§ ££ •
¶ •• ß
® ßß ©™ ©© ´
¨ ´´ ≠
Æ ≠≠ Ø∞ ØØ ±≤ ±
≥ ±± ¥µ ¥¥ ∂∑ ∂∂ ∏
π ∏∏ ∫ª ∫
º ∫∫ Ωæ ΩΩ ø¿ ø
¡ øø ¬√ ¬¬ ƒ≈ ƒƒ ∆
« ∆∆ »… »
  »» ÀÃ ÀÀ ÕŒ ÕÕ œ– œœ —
“ —— ”‘ ”
’ ”” ÷◊ ÷÷ ÿŸ ÿ
⁄ ÿÿ €‹ €€ ›ﬁ ›› ﬂ
‡ ﬂﬂ ·‚ ·
„ ·· ‰Â ‰‰ ÊÁ ÊÊ ËÈ ËË Í
Î ÍÍ ÏÌ Ï
Ó ÏÏ Ô ÔÔ ÒÚ ÒÒ ÛÙ ÛÛ ıˆ ıı ˜
¯ ˜˜ ˘˙ ˘
˚ ˘˘ ¸˝ ¸¸ ˛ˇ ˛˛ Ä
Å ÄÄ ÇÉ Ç
Ñ ÇÇ ÖÜ ÖÖ áà áá â
ä ââ ãå ã
ç ãã éè éé êë êê íì íí î
ï îî ñó ñ
ò ññ ôö ôô õú õ
ù õõ ûü ûû †° †† ¢
£ ¢¢ §• §
¶ §§ ß® ßß ©
™ ©© ´
¨ ´´ ≠Æ ≠≠ Ø∞ Ø
± ØØ ≤≥ ≤≤ ¥µ ¥¥ ∂
∑ ∂∂ ∏π ∏
∫ ∏∏ ªº ªª Ω
æ ΩΩ ø
¿ øø ¡¬ ¡¡ √ƒ √√ ≈∆ ≈≈ «
» «« …  …
À …… ÃÕ ÃÃ Œœ ŒŒ –— –– “
” ““ ‘’ ‘
÷ ‘‘ ◊ÿ ◊◊ Ÿ
⁄ ŸŸ €
‹ €€ ›ﬁ ›› ﬂ
‡ ﬂﬂ ·
‚ ·· „‰ „„ ÂÊ Â
Á ÂÂ ËÈ ËË ÍÎ ÍÍ Ï
Ì ÏÏ ÓÔ Ó
 ÓÓ ÒÚ ÒÒ Û
Ù ÛÛ ı
ˆ ıı ˜¯ ˜˜ ˘
˙ ˘˘ ˚
¸ ˚˚ ˝˛ ˝˝ ˇ
Ä ˇˇ Å
Ç ÅÅ ÉÑ ÉÉ ÖÜ Ö
á ÖÖ àâ àà äã ää å
ç åå éè é
ê éé ëí ëë ì
î ìì ï
ñ ïï óò óó ôö ôô õú õõ ùû ùù ü
† üü °¢ °
£ °° §• §§ ¶ß ¶¶ ®
© ®® ™´ ™
¨ ™™ ≠Æ ≠≠ Ø∞ Ø
± ØØ ≤≥ ≤≤ ¥µ ¥¥ ∂
∑ ∂∂ ∏π ∏
∫ ∏∏ ªº ªª Ωæ Ω
ø ΩΩ ¿¡ ¿¿ ¬√ ¬¬ ƒ
≈ ƒƒ ∆« ∆
» ∆∆ …  …… À
Ã ÀÀ Õ
Œ ÕÕ œ– œœ —
“ —— ”
‘ ”” ’÷ ’’ ◊
ÿ ◊◊ Ÿ
⁄ ŸŸ €‹ €€ ›
ﬁ ›› ﬂ
‡ ﬂﬂ ·‚ ·· „
‰ „„ Â
Ê ÂÂ ÁË ÁÁ È
Í ÈÈ Î
Ï ÎÎ ÌÓ ÌÌ Ô Ô
Ò ÔÔ ÚÛ ÚÚ Ùı ÙÙ ˆ
˜ ˆˆ ¯˘ ¯
˙ ¯¯ ˚¸ ˚˚ ˝˛ ˝
ˇ ˝˝ ÄÅ ÄÄ ÇÉ ÇÇ Ñ
Ö ÑÑ Üá Ü
à ÜÜ âä ââ ãå ã
ç ãã éè éé êë êê í
ì íí îï î
ñ îî óò óó ôö ô
õ ôô úù úú ûü ûû †
° †† ¢£ ¢
§ ¢¢ •¶ •• ß
® ßß ©
™ ©© ´¨ ´´ ≠Æ ≠
Ø ≠≠ ∞± ∞∞ ≤≥ ≤≤ ¥
µ ¥¥ ∂∑ ∂
∏ ∂∂ π∫ ππ ª
º ªª Ω
æ ΩΩ ø¿ øø ¡
¬ ¡¡ √
ƒ √√ ≈∆ ≈≈ «» «« …  …… À
Ã ÀÀ ÕŒ Õ
œ ÕÕ –— –– “
” ““ ‘
’ ‘‘ ÷◊ ÷÷ ÿ
Ÿ ÿÿ ⁄
€ ⁄⁄ ‹› ‹‹ ﬁ
ﬂ ﬁﬁ ‡
· ‡‡ ‚„ ‚‚ ‰Â ‰
Ê ‰‰ ÁË ÁÁ ÈÍ ÈÈ Î
Ï ÎÎ ÌÓ Ì
Ô ÌÌ Ò  ÚÛ Ú
Ù ÚÚ ıˆ ıı ˜¯ ˜˜ ˘
˙ ˘˘ ˚¸ ˚
˝ ˚˚ ˛ˇ ˛˛ Ä	
Å	 Ä	Ä	 Ç	É	 Ç	
Ñ	 Ç	Ç	 Ö	Ü	 Ö	Ö	 á	à	 á	á	 â	ä	 â	â	 ã	
å	 ã	ã	 ç	é	 ç	
è	 ç	ç	 ê	ë	 ê	ê	 í	ì	 í	í	 î	
ï	 î	î	 ñ	ó	 ñ	
ò	 ñ	ñ	 ô	ö	 ô	ô	 õ	ú	 õ	
ù	 õ	õ	 û	ü	 û	û	 †	°	 †	†	 ¢	
£	 ¢	¢	 §	•	 §	
¶	 §	§	 ß	®	 ß	ß	 ©	™	 ©	
´	 ©	©	 ¨	≠	 ¨	¨	 Æ	Ø	 Æ	Æ	 ∞	
±	 ∞	∞	 ≤	≥	 ≤	
¥	 ≤	≤	 µ	∂	 µ	µ	 ∑	∏	 ∑	
π	 ∑	∑	 ∫	ª	 ∫	∫	 º	Ω	 º	º	 æ	
ø	 æ	æ	 ¿	¡	 ¿	
¬	 ¿	¿	 √	ƒ	 √	√	 ≈	∆	 ≈	
«	 ≈	≈	 »	…	 »	»	  	À	  	 	 Ã	
Õ	 Ã	Ã	 Œ	œ	 Œ	
–	 Œ	Œ	 —	“	 —	—	 ”	
‘	 ”	”	 ’	
÷	 ’	’	 ◊	ÿ	 ◊	◊	 Ÿ	
⁄	 Ÿ	Ÿ	 €	
‹	 €	€	 ›	ﬁ	 ›	›	 ﬂ	
‡	 ﬂ	ﬂ	 ·	
‚	 ·	·	 „	‰	 „	„	 Â	
Ê	 Â	Â	 Á	
Ë	 Á	Á	 È	Í	 È	È	 Î	Ï	 Î	
Ì	 Î	Î	 Ó	Ô	 Ó	Ó	 	Ò	 		 Ú	
Û	 Ú	Ú	 Ù	ı	 Ù	
ˆ	 Ù	Ù	 ˜	¯	 ˜	˜	 ˘	˙	 ˘	
˚	 ˘	˘	 ¸	˝	 ¸	¸	 ˛	ˇ	 ˛	˛	 Ä

Å
 Ä
Ä
 Ç
É
 Ç

Ñ
 Ç
Ç
 Ö
Ü
 Ö
Ö
 á
à
 á
á
 â
ä
 â
â
 ã

å
 ã
ã
 ç
é
 ç

è
 ç
ç
 ê
ë
 ê
ê
 í
ì
 í
í
 î
ï
 î
î
 ñ
ó
 ñ
ñ
 ò

ô
 ò
ò
 ö
õ
 ö

ú
 ö
ö
 ù
û
 ù
ù
 ü
†
 ü
ü
 °

¢
 °
°
 £
§
 £

•
 £
£
 ¶
ß
 ¶
¶
 ®
©
 ®
®
 ™

´
 ™
™
 ¨
≠
 ¨

Æ
 ¨
¨
 Ø
∞
 Ø
Ø
 ±
≤
 ±

≥
 ±
±
 ¥
µ
 ¥
¥
 ∂
∑
 ∂
∂
 ∏

π
 ∏
∏
 ∫
ª
 ∫

º
 ∫
∫
 Ω
æ
 Ω
Ω
 ø

¿
 ø
ø
 ¡

¬
 ¡
¡
 √
ƒ
 √
√
 ≈

∆
 ≈
≈
 «

»
 «
«
 …
 
 …
…
 À

Ã
 À
À
 Õ

Œ
 Õ
Õ
 œ
–
 œ
œ
 —

“
 —
—
 ”

‘
 ”
”
 ’
÷
 ’
’
 ◊

ÿ
 ◊
◊
 Ÿ

⁄
 Ÿ
Ÿ
 €
‹
 €
€
 ›
ﬁ
 ›

ﬂ
 ›
›
 ‡
·
 ‡
‡
 ‚
„
 ‚
‚
 ‰

Â
 ‰
‰
 Ê
Á
 Ê

Ë
 Ê
Ê
 È
Í
 È
È
 Î
Ï
 Î

Ì
 Î
Î
 Ó
Ô
 Ó
Ó
 
Ò
 

 Ú

Û
 Ú
Ú
 Ù
ı
 Ù

ˆ
 Ù
Ù
 ˜
¯
 ˜
˜
 ˘
˙
 ˘

˚
 ˘
˘
 ¸
˝
 ¸
¸
 ˛
ˇ
 ˛
˛
 Ä
Å ÄÄ ÇÉ Ç
Ñ ÇÇ ÖÜ ÖÖ á
à áá â
ä ââ ãå ãã çé çç èê èè ë
í ëë ìî ì
ï ìì ñó ññ ò
ô òò ö
õ öö úù úú û
ü ûû †
° †† ¢£ ¢¢ §
• §§ ¶
ß ¶¶ ®© ®® ™´ ™
¨ ™™ ≠Æ ≠≠ Ø∞ ØØ ±
≤ ±± ≥¥ ≥
µ ≥≥ ∂∑ ∂∂ ∏π ∏
∫ ∏∏ ªº ªª Ωæ ΩΩ ø
¿ øø ¡¬ ¡
√ ¡¡ ƒ≈ ƒƒ ∆
« ∆∆ »
… »»  À    Ã
Õ ÃÃ Œ
œ ŒŒ –— –– “” ““ ‘’ ‘‘ ÷◊ ÷÷ ÿ
Ÿ ÿÿ ⁄€ ⁄
‹ ⁄⁄ ›ﬁ ›› ﬂ
‡ ﬂﬂ ·‚ ·
„ ·· ‰Â ‰‰ Ê
Á ÊÊ Ë
È ËË ÍÎ ÍÍ Ï
Ì ÏÏ Ó
Ô ÓÓ Ò  ÚÛ Ú
Ù ÚÚ ıˆ ıı ˜¯ ˜˜ ˘
˙ ˘˘ ˚¸ ˚
˝ ˚˚ ˛ˇ ˛˛ ÄÅ Ä
Ç ÄÄ ÉÑ ÉÉ ÖÜ ÖÖ á
à áá âä â
ã ââ åç åå éè é
ê éé ëí ëë ìî ìì ï
ñ ïï óò ó
ô óó öõ öö úù ú
û úú ü† üü °¢ °° £
§ ££ •¶ •
ß •• ®© ®® ™
´ ™™ ¨≠ ¨
Æ ¨¨ Ø∞ ØØ ±≤ ±± ≥¥ ≥≥ µ
∂ µµ ∑∏ ∑
π ∑∑ ∫ª ∫∫ ºΩ ºº æ
ø ææ ¿¡ ¿
¬ ¿¿ √ƒ √√ ≈∆ ≈
« ≈≈ »… »»  À    Ã
Õ ÃÃ Œœ Œ
– ŒŒ —“ —— ”‘ ”” ’÷ ’’ ◊
ÿ ◊◊ Ÿ⁄ Ÿ
€ ŸŸ ‹› ‹‹ ﬁﬂ ﬁﬁ ‡· ‡‡ ‚
„ ‚‚ ‰Â ‰
Ê ‰‰ ÁË Á
È ÁÁ ÍÎ ÍÍ ÏÌ ÏÏ Ó
Ô ÓÓ Ò 
Ú  ÛÙ ÛÛ ıˆ ıı ˜¯ ˜˜ ˘
˙ ˘˘ ˚¸ ˚
˝ ˚˚ ˛ˇ ˛˛ Ä
Å ÄÄ Ç
É ÇÇ ÑÖ ÑÑ Ü
á ÜÜ à
â àà äã ää åç å
é åå èê èè ëí ëë ì
î ìì ïñ ï
ó ïï òô òò öõ öö úù úú û
ü ûû †° †
¢ †† £§ ££ •¶ •
ß •• ®© ®® ™´ ™™ ¨
≠ ¨¨ ÆØ Æ
∞ ÆÆ ±≤ ±± ≥
¥ ≥≥ µ
∂ µµ ∑∏ ∑∑ π
∫ ππ ª
º ªª Ωæ ΩΩ ø
¿ øø ¡
¬ ¡¡ √ƒ √√ ≈
∆ ≈≈ «
» «« …  …… À
Ã ÀÀ Õ
Œ ÕÕ œ– œœ —
“ —— ”
‘ ”” ’÷ ’’ ◊
ÿ ◊◊ Ÿ
⁄ ŸŸ €‹ €€ ›ﬁ ›› ﬂ‡ ﬂﬂ ·
‚ ·· „‰ „
Â „„ ÊÁ ÊÊ Ë
È ËË Í
Î ÍÍ ÏÌ ÏÏ ÓÔ Ó
 ÓÓ ÒÚ ÒÒ ÛÙ ÛÛ ı
ˆ ıı ˜¯ ˜
˘ ˜˜ ˙˚ ˙˙ ¸˝ ¸
˛ ¸¸ ˇÄ ˇˇ ÅÇ ÅÅ É
Ñ ÉÉ ÖÜ Ö
á ÖÖ àâ àà äã ä
å ää çé çç èê èè ë
í ëë ìî ì
ï ìì ñó ññ òô òò öõ öö ú
ù úú ûü û
† ûû °¢ °° £§ £
• ££ ¶ß ¶¶ ®© ®® ™
´ ™™ ¨≠ ¨
Æ ¨¨ Ø∞ ØØ ±
≤ ±± ≥
¥ ≥≥ µ∂ µµ ∑
∏ ∑∑ π
∫ ππ ªº ªª Ω
æ ΩΩ ø
¿ øø ¡¬ ¡¡ √
ƒ √√ ≈
∆ ≈≈ «» «« …  …… ÀÃ ÀÀ Õ
Œ ÕÕ œ– œ
— œœ “” ““ ‘’ ‘
÷ ‘‘ ◊ÿ ◊◊ Ÿ⁄ ŸŸ €
‹ €€ ›ﬁ ›
ﬂ ›› ‡· ‡‡ ‚„ ‚
‰ ‚‚ ÂÊ ÂÂ ÁË ÁÁ È
Í ÈÈ ÎÏ Î
Ì ÎÎ ÓÔ ÓÓ Ò 
Ú  ÛÙ ÛÛ ıˆ ıı ˜
¯ ˜˜ ˘˙ ˘
˚ ˘˘ ¸˝ ¸¸ ˛ˇ ˛
Ä ˛˛ ÅÇ ÅÅ ÉÑ ÉÉ Ö
Ü ÖÖ áà á
â áá äã ää åç å
é åå èê èè ëí ëë ì
î ìì ïñ ï
ó ïï òô òò öõ ö
ú öö ùû ùù ü† üü °
¢ °° £§ £
• ££ ¶ß ¶¶ ®© ®
™ ®® ´¨ ´´ ≠Æ ≠≠ Ø
∞ ØØ ±≤ ±
≥ ±± ¥µ ¥¥ ∂∑ ∂
∏ ∂∂ π∫ ππ ªº ªª Ω
æ ΩΩ ø¿ ø
¡ øø ¬√ ¬¬ ƒ≈ ƒ
∆ ƒƒ «» «« …  …… À
Ã ÀÀ ÕŒ Õ
œ ÕÕ –— –– “
” ““ ‘
’ ‘‘ ÷◊ ÷÷ ÿ
Ÿ ÿÿ ⁄
€ ⁄⁄ ‹› ‹‹ ﬁ
ﬂ ﬁﬁ ‡
· ‡‡ ‚„ ‚‚ ‰
Â ‰‰ Ê
Á ÊÊ ËÈ ËË Í
Î ÍÍ Ï
Ì ÏÏ ÓÔ ÓÓ 
Ò  Ú
Û ÚÚ Ùı ÙÙ ˆ˜ ˆ
¯ ˆˆ ˘˙ ˘˘ ˚¸ ˚˚ ˝
˛ ˝˝ ˇÄ ˇ
Å ˇˇ ÇÉ É !É /É =É FÉ QÉ \É eÉ nÉ wÉ ÄÉ ãÉ ñÉ °É ¨É ∑É ¬É ÕÉ ÿÉ ﬂÉ ÈÉ ÙÉ ˇÉ çÉ òÉ ¶É ±É ºÉ «É ’É „É ÓÉ ˘É ÄÉ ÜÉ ìÉ ûÉ •É ´É µÉ √É  É –É ÷É ‹É ÈÉ ÙÉ ˇÉ çÉ îÉ †É ©É ∞É ΩÉ ƒÉ —É ÿÉ ﬁÉ ËÉ ÔÉ ıÉ ˚É ÅÉ áÉ çÉ ìÉ ôÉ üÉ •É ´É ∏É ∆É —É ﬂÉ ÍÉ ˜É ÄÉ âÉ îÉ ¢É ©É ∂É ΩÉ «É “É ŸÉ ﬂÉ ÏÉ ÛÉ ˘É ˇÉ åÉ ìÉ üÉ ®É ∂É ƒÉ ÀÉ —É ◊É ›É „É ÈÉ ˆÉ ÑÉ íÉ †É ßÉ ¥É ªÉ ¡É ÀÉ “É ÿÉ ﬁÉ ÎÉ ˘É ã	É î	É ¢	É ∞	É æ	É Ã	É ”	É Ÿ	É ﬂ	É Â	É Ú	É Ä
É ã
É ò
É °
É ™
É ∏
É ø
É ≈
É À
É —
É ◊
É ‰
É Ú
É ÄÉ áÉ ëÉ òÉ ûÉ §É ±É øÉ ∆É ÃÉ ÿÉ ﬂÉ ÊÉ ÏÉ ˘É áÉ ïÉ £É µÉ æÉ ÃÉ ◊É ‚É ÓÉ ˘É ÄÉ ÜÉ ìÉ ûÉ ¨É ≥É πÉ øÉ ≈É ÀÉ —É ◊É ·É ËÉ ıÉ ÉÉ ëÉ úÉ ™É ±É ∑É ΩÉ √É ÕÉ €É ÈÉ ˜É ÖÉ ìÉ °É ØÉ ΩÉ ÀÉ “É ÿÉ ﬁÉ ‰É ÍÉ É ˝Ñ 	Ö     	 
 
 
      
      " $! % '
 )& *( , .- 0+ 2/ 3 5
 74 86 : <; >9 @= A
 C ED GB IF J LK N PO RM TQ U WV Y [Z ]X _\ ` b dc fa he i k ml oj qn r
 t vu xs zw {
 } ~ Å| ÉÄ Ñ ÜÖ à äâ åá éã è ëê ì ïî óí ôñ ö úõ û †ü ¢ù §° • ß¶ © ´™ ≠® Ø¨ ∞ ≤± ¥ ∂µ ∏≥ ∫∑ ª
 Ωº ø ¡¿ √æ ≈¬ ∆
 »«   ÃÀ Œ… –Õ —
 ”“ ’ ◊÷ Ÿ‘ €ÿ ‹ ﬁ› ‡ﬂ ‚
 ‰„ Ê ËÁ ÍÂ ÏÈ Ì
 ÔÓ Ò ÛÚ ı ˜Ù ¯
 ˙˘ ¸ ˛˝ Ä˚ Çˇ É Ö
 áÑ àÜ ä åã éâ êç ë
 ìí ï óñ ôî õò ú û
 †ù °ü £ •§ ß¢ ©¶ ™
 ¨´ Æ ∞Ø ≤≠ ¥± µ
 ∑∂ π ª∫ Ω∏ øº ¿
 ¬¡ ƒ ∆≈ »√  « À Õ
 œÃ –Œ “ ‘” ÷— ÿ’ Ÿ €
 ›⁄ ﬁ‹ ‡ ‚· ‰ﬂ Ê„ Á
 ÈË Î ÌÏ ÔÍ ÒÓ Ú
 ÙÛ ˆ ¯˜ ˙ı ¸˘ ˝ ˇ˛ ÅÄ É ÖÑ áÜ â ã
 çä éå ê íë îè ñì ó
 ôò õ ùú üö °û ¢ §£ ¶• ® ™© ¨´ Æ
 ∞Ø ≤ ¥≥ ∂± ∏µ π ª
 Ω∫ æº ¿ ¬¡ ƒø ∆√ « …» À  Õ œŒ —– ” ’‘ ◊÷ Ÿ €⁄ ›‹ ﬂ ·
 „‡ ‰‚ Ê ËÁ ÍÂ ÏÈ Ì
 ÔÓ Ò ÛÚ ı ˜Ù ¯ ˙˘ ¸ ˛˝ Ä˚ Çˇ É Ö
 áÑ àÜ ä åã éâ êç ë ìí ïî ó
 ôò õö ù üû °ú £† §ö ¶ ®ß ™• ¨© ≠ ØÆ ±∞ ≥ µ
 ∑¥ ∏∂ ∫ ºª æπ ¿Ω ¡ √¬ ≈ƒ « …
 À» Ã  Œ –œ “Õ ‘— ’ ◊÷ Ÿÿ € ›‹ ﬂﬁ ·
 „‚ Â ÁÊ È‰ ÎË Ï ÓÌ Ô Ú ÙÛ ˆı ¯ ˙˘ ¸˚ ˛ Äˇ ÇÅ Ñ ÜÖ àá ä åã éç ê íë îì ñ òó öô ú ûù †ü ¢ §£ ¶• ® ™© ¨´ Æ ∞
 ≤Ø ≥± µ ∑∂ π¥ ª∏ º æ
 ¿Ω ¡ø √ ≈ƒ «¬ …∆  
 ÃÀ Œ –œ “Õ ‘— ’ ◊
 Ÿ÷ ⁄ÿ ‹ ﬁ› ‡€ ‚ﬂ „
 Â‰ Á ÈË ÎÊ ÌÍ Ó
 Ô ÚÒ Ù ˆı ¯Û ˙˜ ˚Ò ˝ ˇ˛ Å¸ ÉÄ ÑÒ Ü àá äÖ åâ ç
 èé ë ìí ïê óî ò ö
 úô ùõ ü °† £û •¢ ¶ ®ß ™© ¨ Æ
 ∞≠ ±Ø ≥ µ¥ ∑≤ π∂ ∫ ºª æΩ ¿
 ¬¡ ƒ ∆≈ »√  « À
 ÕÃ œ —– ”Œ ’“ ÷ ÿ◊ ⁄Ÿ ‹ ﬁ› ‡ﬂ ‚ ‰
 Ê„ ÁÂ È ÎÍ ÌË ÔÏ  ÚÒ ÙÛ ˆ ¯˜ ˙˘ ¸ ˛˝ Äˇ Ç Ñ
 ÜÉ áÖ â ãä çà èå ê íë îì ñ
 òó öô ú ûù †õ ¢ü £ô • ß¶ ©§ ´® ¨ Æ
 ∞≠ ±Ø ≥ µ¥ ∑≤ π∂ ∫ º
 æª øΩ ¡ √¬ ≈¿ «ƒ »  … ÃÀ Œ –œ “— ‘ ÷’ ÿ◊ ⁄ ‹€ ﬁ› ‡ ‚· ‰„ Ê ËÁ ÍÈ Ï Ó
 Ì ÒÔ Û ıÙ ˜Ú ˘ˆ ˙ ¸
 ˛˚ ˇ˝ Å ÉÇ ÖÄ áÑ à ä
 åâ çã è ëê ìé ïí ñ ò
 öó õô ù üû °ú £† § ¶• ®ß ™ ¨
 Æ´ Ø≠ ± ≥≤ µ∞ ∑¥ ∏ ∫π ºª æ ¿ø ¬¡ ƒ
 ∆≈ »  … Ã« ŒÀ œ —– ”“ ’ ◊÷ Ÿÿ € ›‹ ﬂﬁ · „
 Â‚ Ê‰ Ë ÍÈ ÏÁ ÓÎ Ô Ò
 Û ÙÚ ˆ ¯˜ ˙ı ¸˘ ˝
 ˇ˛ Å	 É	Ä	 Ñ	Ç	 Ü	Ö	 à	 ä	â	 å	á	 é	ã	 è	Ö	 ë	 ì	í	 ï	ê	 ó	î	 ò	 ö	
 ú	ô	 ù	õ	 ü	 °	†	 £	û	 •	¢	 ¶	 ®	
 ™	ß	 ´	©	 ≠	 Ø	Æ	 ±	¨	 ≥	∞	 ¥	 ∂	
 ∏	µ	 π	∑	 ª	 Ω	º	 ø	∫	 ¡	æ	 ¬	 ƒ	
 ∆	√	 «	≈	 …	 À	 	 Õ	»	 œ	Ã	 –	 “	—	 ‘	”	 ÷	 ÿ	◊	 ⁄	Ÿ	 ‹	 ﬁ	›	 ‡	ﬂ	 ‚	 ‰	„	 Ê	Â	 Ë	 Í	
 Ï	È	 Ì	Î	 Ô	 Ò		 Û	Ó	 ı	Ú	 ˆ	 ¯	
 ˙	˜	 ˚	˘	 ˝	 ˇ	˛	 Å
¸	 É
Ä
 Ñ

 Ü
Ö
 à
 ä
â
 å
á
 é
ã
 è

 ë
ê
 ì
í
 ï
 ó
ñ
 ô
î
 õ
ò
 ú
í
 û
 †
ü
 ¢
ù
 §
°
 •
í
 ß
 ©
®
 ´
¶
 ≠
™
 Æ
 ∞

 ≤
Ø
 ≥
±
 µ
 ∑
∂
 π
¥
 ª
∏
 º
 æ
Ω
 ¿
ø
 ¬
 ƒ
√
 ∆
≈
 »
  
…
 Ã
À
 Œ
 –
œ
 “
—
 ‘
 ÷
’
 ÿ
◊
 ⁄
 ‹

 ﬁ
€
 ﬂ
›
 ·
 „
‚
 Â
‡
 Á
‰
 Ë
 Í

 Ï
È
 Ì
Î
 Ô
 Ò

 Û
Ó
 ı
Ú
 ˆ
 ¯

 ˙
˜
 ˚
˘
 ˝
 ˇ
˛
 Å¸
 ÉÄ Ñ ÜÖ àá ä
 åã é êè íç îë ï óñ ôò õ ùú üû ° £¢ •§ ß ©
 ´® ¨™ Æ ∞Ø ≤≠ ¥± µ ∑
 π∂ ∫∏ º æΩ ¿ª ¬ø √ ≈ƒ «∆ … À  ÕÃ œ
 —– ”“ ’ ◊÷ Ÿ‘ €ÿ ‹ ﬁ› ‡‘ ‚ﬂ „ Â‰ ÁÊ È ÎÍ ÌÏ Ô Ò
 Û ÙÚ ˆ ¯˜ ˙ı ¸˘ ˝ ˇ
 Å˛ ÇÄ Ñ ÜÖ àÉ äá ã ç
 èå êé í îì ñë òï ô õ
 ùö ûú † ¢° §ü ¶£ ß
 ©® ´ ≠™ Æ¨ ∞Ø ≤ ¥≥ ∂± ∏µ πØ ª Ωº ø∫ ¡æ ¬ ƒ
 ∆√ «≈ … À  Õ» œÃ –
 “— ‘ ÷’ ÿ” ⁄◊ €
 ›‹ ﬂ ·‡ „ﬁ Â‚ Ê
 Ëµ	 ÈÁ Î ÌÏ ÔÍ ÒÓ Ú
 ÙÛ ˆ ¯˜ ˙ı ¸˘ ˝ ˇ˛ ÅÄ É ÖÑ áÜ â ã
 çä éå ê íë îè ñì ó
 ôò õ ùú üö °û ¢ §
 ¶£ ß• © ´™ ≠® Ø¨ ∞ ≤± ¥≥ ∂ ∏∑ ∫π º æΩ ¿ø ¬ ƒ√ ∆≈ »  … ÃÀ Œ –œ “— ‘ ÷’ ÿ◊ ⁄
 ‹€ ﬁ ‡ﬂ ‚› ‰· Â ÁÊ ÈË Î Ì
 ÔÏ Ó Ú ÙÛ ˆÒ ¯ı ˘ ˚
 ˝˙ ˛¸ Ä ÇÅ Ñˇ ÜÉ á â
 ãà åä é êè íç îë ï
 óñ ô õö ùò üú † ¢
 §° •£ ß ©® ´¶ ≠™ Æ ∞Ø ≤± ¥ ∂µ ∏∑ ∫ ºª æΩ ¿ ¬¡ ƒ√ ∆
 »«   ÃÀ Œ… –Õ — ”
 ’“ ÷‘ ÿ ⁄Ÿ ‹◊ ﬁ€ ﬂ ·
 „‡ ‰‚ Ê ËÁ ÍÂ ÏÈ Ì Ô
 ÒÓ Ú Ù ˆı ¯Û ˙˜ ˚ ˝
 ˇ¸ Ä˛ Ç ÑÉ ÜÅ àÖ â ã
 çä éå ê íë îè ñì ó ô
 õò úö û †ü ¢ù §° • ß
 ©¶ ™® ¨ Æ≠ ∞´ ≤Ø ≥ µ
 ∑¥ ∏∂ ∫ ºª æπ ¿Ω ¡ √
 ≈¬ ∆ƒ »  … Ã« ŒÀ œ —– ”“ ’ ◊÷ Ÿÿ € ›‹ ﬂﬁ · „‚ Â‰ Á ÈË ÎÍ Ì ÔÓ Ò Û ı
 ˜Ù ¯ˆ ˙ ¸˚ ˛˘ Ä˝ Å ÜÜ Ç àà ââ áá≠ ââ ≠ ââ È	 ââ È	ô ââ ô⁄ ââ ⁄° ââ °ö ââ ö‹ ââ ‹ı àà ı£ ââ £à àà àÉ àà É« àà « àà ú ââ ú€
 ââ €
ê àà êõ ââ õ≠ ââ ≠Á ââ Áˆ ââ ˆß	 ââ ß	M àà M∂ ââ ∂® ââ ®˜
 ââ ˜
ç àà çØ ââ Øí
 àà í
ø àà ø˚ àà ˚è àà èÉ ââ É€ ââ €˘	 ââ ˘	ò ââ ò∏ ââ ∏˛ ââ ˛Ê àà Ê“ ââ “˚ àà ˚‰ ââ ‰è àà èù àà ùû àà ûÑ ââ Ñé ââ éÜ ââ Üù àà ùÛ ââ Û≈	 ââ ≈	Ò àà ÒÖ ââ Ö„ ââ „¶ ââ ¶ö àà öû	 àà û	ç àà çÂ ââ Â& ââ &« ââ «Ó	 àà Ó	È
 ââ È
Ø ââ Ø˙ ââ ˙ àà ú àà ú+ àà +Œ àà Œ àà ƒ ââ ƒ¥ àà ¥π àà π ÜÜ ¨	 àà ¨	• ââ •˝ ââ ˝ô	 ââ ô	Œ ââ Œ ââ ‡ ââ ‡ ââ å ââ å»	 àà »	Õ àà Õñ ââ ñ˚ ââ ˚ àà Ì ââ ÌÃ ââ Ã÷ ââ ÷â àà â¸ ââ ¸ ââ ¨ ââ ¨6 ââ 6ã ââ ã¶ àà ¶¬ ââ ¬´ ââ ´® àà ®∂ ââ ∂√ àà √∞ àà ∞ª ââ ªí ââ íå ââ åé ââ é≠ àà ≠‚ ââ ‚∑	 ââ ∑	ô ââ ô” àà ”Û ââ ÛÇ	 ââ Ç	Â àà Â« ââ « áá ≠ ââ ≠X àà XÚ ââ ÚV ââ Vâ ââ â‘ ââ ‘´ ââ ´Î	 ââ Î	… àà …ô àà ô± ââ ±õ ââ õö ââ öø ââ øó ââ óπ àà π˘
 ââ ˘
Í àà ÍØ ââ ØÓ ââ Óö àà ö∂ ââ ∂4 ââ 4± àà ±á
 àà á
® àà ®ı àà ı˜	 ââ ˜	… àà …˛ ââ ˛Ø ââ ØÔ ââ Ô¸
 àà ¸
¢ àà ¢( ââ (√ àà √Ä ââ Ä≈ ââ ≈Ú ââ Ú™ ââ ™Ë àà Ë‹ ââ ‹ı àà ıí àà í˘ ââ ˘º ââ ºá àà á‚ ââ ‚Ω ââ Ω¬ àà ¬é àà é≤ àà ≤‡ ââ ‡ ââ â àà â≤ àà ≤¥
 àà ¥
ä ââ ä‰ ââ ‰º ââ ºÍ àà Í¡ ââ ¡è àà èØ
 ââ Ø
ù ââ ùﬁ àà ﬁ ââ ü àà ü» ââ »Ö ââ ÖÎ
 ââ Î
± ââ ±Ó
 àà Ó
ë àà ëÀ ââ ÀÜ ââ Üî àà îõ	 ââ õ	ä ââ äﬂ àà ﬂ˘ ââ ˘∏ àà ∏„ ââ „¸ ââ ¸ÿ ââ ÿÂ àà ÂÓ ââ Óµ	 ââ µ	£ ââ £‚ ââ ‚Ï ââ Ï≥ àà ≥ö àà öÚ àà ÚØ àà ØË ââ Ë√	 ââ √	Â àà Â¡ ââ ¡Ù ââ Ù¶ ââ ¶å ââ å— ââ —≠ àà ≠ê ââ êò àà ò‰ àà ‰¿ àà ¿ã ââ ãÖ	 àà Ö	— àà —Ò àà Ò±
 ââ ±
ª àà ªÖ
 ââ Ö
Ω ââ Ωæ àà æ€ àà €  ââ  Ó ââ Ó√ ââ √Õ àà Õ◊ àà ◊ä ââ äÁ àà Á® ââ ®©	 ââ ©	Ñ ââ Ñä ââ ä9 àà 9“ àà “‡
 àà ‡
Ó ââ Ó“ ââ “K ââ Kà ââ à¥ ââ ¥∫ ââ ∫∫	 àà ∫	‚ ââ ‚≈ ââ ≈ı àà ıå ââ åÄ àà Äü ââ üò ââ ò˘ àà ˘¸	 àà ¸	›
 ââ ›
Û àà Û‘ àà ‘Å àà ÅÃ ââ Ã´ àà ´» àà »› àà ›¥ ââ ¥ˇ àà ˇ∂ ââ ∂ò ââ ò« àà «
ä œ
ã ˘

å É
ç ª
é ÷
è º
ê ±
ë ô
í å
ì ¶
î ©ï ⁄
ñ ∫
ó Ù
ò â
ô £
ö ¥
õ ®

ú ∑
ù —
û ¿
ü ã
† ˜	
° ‡
¢ ˝
£ ≤
§ Ö§ Â§ ‡§ Á	
§ ù
§ â	• u
¶ Á	ß &
® ú
© ∂
	™ 4
´ º	
¨ Ö
≠ Ñ
Æ ≥
Ø ´
Ø Ö

∞ ¡
±  
≤ é
≥ ˚
≥ 
¥ è
µ ›	∂ K
∑ ´∏ ï
∏ õ
π …

∫ á
ª ƒ
º ü
Ω Ó
æ ı
ø ∫
¿ ™
¡ ì
¬ ‚
√ Û
ƒ ¡
≈ å
∆ «« ß
» ‹
… ¡
… ∂
  ≈	
À ˘
Ã ã
Õ û
Œ ÛŒ €
œ ë
– ô
— ë
“ Ù
” Ö
‘ ⁄’ Ó
÷ ¸
◊ »ÿ µ
Ÿ ·⁄ Õ⁄ ”
€ ‚‹ «
› Ø

ﬁ ∂
ﬂ Ø
ﬂ ˛
‡ ‚
· â
‚ ä„ ‘
‰ ú
Â ≥
Ê 
Á ®Ë ·Ë ≤Ë ∆Ë ”Ë ·	
Ë ¶

Ë ‘
È Æ
Í È
Î î
Ï á	
Ì ˛
Ó Ñ
Ô ê	
 Î	
Ò —	
Ú ‡
Û ö
Ù Ú	ı j
ˆ ¥
˜ Ã
¯ Ë
˘ ú
˙ É
˚ €
	¸ l	˝ 4
˛ —
ˇ ‘
Ä ≠
Å ≈
Ç ã
É ⁄
Ñ ¬
Ö ª
Ü Û
á ±
à à
â ò
ä „	ã K
å ˜

ç ò
é ≠
è Ë
ê ¡
ë £
í ëì Å
î Ó
ï –
ñ Ó
ó •
ò Ø

ô Ÿ
ö ‘
õ À
ú ‰
ù Ô
û ø
ü „		† 
° â	
¢ È	
£ †
§ Ö
• ’¶ ˜¶ ˝
ß µ	
® ∏
© ÷
™ í
´ ≠
´ ó
¨ ä
≠ Û
Æ ‚
Ø Ω
∞ Û	± 
≤ À
≥ –	¥ V
µ ”
∂ Ï
∑ ‚
∏ ¬
π ›
∫ …
ª é
º Ö
Ω Ö
æ “
ø ÷
¿ º
¡ •
¬ ˛
√ √
ƒ Ó
ƒ ¸
≈ ®
∆ 
« “» Í
… ˜	  ´
À °Ã ΩÃ √Ã ’	Ã €	Ã »
Õ ˘	
Œ ù
œ ˇ
– ‡
— ß	
“ å
” ∂
‘ …
’ â

÷ õ
◊ ¶
ÿ Ó
Ÿ ˘
⁄ ä
€ Ø
‹ Á
› £	ﬁ ~
ﬂ ê
‡ ∂
· ™
‚ ÷
„ •
‰ £Â Ê
Ê õÁ Œ	Ë 
È °
Í §Í ﬂ
Î ó
Ï ª
Ì  	
Ó ˜

Ô í		 &
Ò ≈
Ú µ
Û ®
Ù Ë
ı ©	
ˆ ó
˜ √	
¯ …˘ ª
˙ ¬
˚ ø
¸ ¥
˝ ˘
˛ ◊
ˇ Ø
Ä ˜
Å ¬Ç ¡
É Ë
Ñ ∑	
Ñ Û	Ö 
Ü ß	
á í
à Ì
â ±

ä ≈
ã à
å ¥ç ©ç «
ç ≈é ˚
è À
ê ˛ë øí ö
ì µ
î Ò
ï «ñ Ä	ñ ™ó Ç
ò ã
ô ÿ
ö Á
õ ≈
ú ê
ù π
û ¡
ü ∂† Ÿ† ‡† Ï
° Œ
¢ ˜
£ ‚
§ ü
§ –	• s
¶ Ω

ß €

® Â
© ö
™ ˜
´ ˚
¨ õ
≠ Û
Æ ÷Ø Ú
∞ ü
± ÷
≤ ò≥ ≠≥ ﬁ
≥ ¸≥ Õ≥ ‘≥ Ç≥ à
¥ ˝	µ |
∂ ‹
∑ Ó
∏ ƒ
π ≠
∫ ˛
ª Æ	º °
Ω 	
æ Ã
ø ﬂ
¿ ı
¡ ®	¬ O
√ ∂
ƒ ˘
≈ Ü
∆ „
« ˝
» ã… Ë
  Ô
  Á
À ›	
Ã ê	Õ a
Œ Î
	œ – ñ
— å
“ ‹
” ñ
‘ Ê
’ £÷ Õ
÷ ”

◊ ò
ÿ ¶
Ÿ Ñ
⁄ Ø
€ Ï
‹ œ
› “
ﬁ ‰ﬂ ≥
‡ ¶
· ’
‚ ˛	
„ ¥
‰ ñ

Â Ø
Ê ò
Á ‚	Ë 
È ˛

Í 
Î Ω
Ï ˙
Ì Í
Ó Ω
Ô ›
 Ú
Ò ‡
Ú ¶	Û -
Ù ‹
ı ¸
ˆ ä
˜ Ω
¯ ±
˘ Ì
˙ Ã
˚ é
¸ Ö

˝ ©	˛ c
ˇ ˘
Ä ª
Å ‚
Ç ä
É Ï
Ñ ä
Ö À
Ü Ú
á Ã
à ò
â ù	ä V
ã €
å ·ç Î
é ºè ¶
ê Ó
ë È

í ‹
ì ü
î †
ï ãñ àñ “ñ ÿñ Òñ Éñ õñ ıñ ⁄ñ ¡

ó ´
ò ƒ
ô ˝
ö ˚õ ·
ú ‰
ù úû ß
ü ª
† ˛° Ã
¢ è
£ ⁄
£ â
§ ß
• ›

¶ å
ß ù
® º
© î

™ ˜´ è
¨ ’

≠ Ñ
Æ ØØ Ÿ

∞ »
± È	
≤ í
≥ 
¥ Œ
µ õ	
∂ «
∑ ∂
∏ ß
π Á	∫ Z
ª „
º ™
Ω ˙
æ °
ø ¨
¿ ñ
¡ √
¬ ¡
√ ¢
ƒ í
≈ û
∆ „
« 

» ®
… Ù
  ñ
À …
Ã  
Õ Ê
Œ √	
œ ˛
– É
— ˜
“ §
” œ
‘ €
’ “
÷ ö
◊ ±ÿ ⁄ÿ ï
Ÿ Ï
⁄ Ñ
⁄ ù
⁄ Ñ
⁄ ¥
⁄ ´
⁄ Ç	
⁄ µ	
⁄ √
⁄ ä
⁄ ò
€ ë
‹ ≠
› Ø	ﬁ B
ﬂ ñ
‡  · 
‚ ∫
„ Ø
‰ ô	
Â Ç
Ê ±
Á ê

Ë ô	
È Í
Í ≠
Î †		Ï D
Ì ∫
Ó È

Ô ô ≠Ò ø
Ú ú
Û Ø
Ù ó
ı €
ˆ «
˜ Ö
¯ ’
˘ Ó˙ π	˚ 6
¸ ª
˝ ˆ
˛ √

ˇ ≈
Ä Å
Å ÷
Ç âÉ ‡É Ÿ
Ñ œ

Ö ë
Ü Ü	á (
à ‡
â ö
ä ‰
ã œ	å ;
ç ›é 

è ä
ê »
ë ◊	
í Ä
ì Ú
î Ω
ï –
ñ ‹
ó Ì"
ratt_kernel"
_Z13get_global_idj"	
_Z3logf"	
_Z3expf"
llvm.fmuladd.f32*ñ
shoc-1.1.5-S3D-ratt_kernel.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282Å

devmap_label
 
 
transfer_bytes_log1p
íÛéA

wgsize
Ä

wgsize_log1p
íÛéA

transfer_bytes
à¢ª