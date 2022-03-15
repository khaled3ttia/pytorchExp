
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
callBw
u
	full_texth
f
d%9 = tail call float @llvm.fmuladd.f32(float %8, float 0xBFECCCCCC0000000, float 0x404523C4C0000000)
&floatB

	full_text


float %8
IfdivBA
?
	full_text2
0
.%10 = fdiv float 1.000000e+00, %7, !fpmath !12
&floatB

	full_text


float %7
qcallBi
g
	full_textZ
X
V%11 = tail call float @llvm.fmuladd.f32(float %10, float 0x408ABBBF20000000, float %9)
'floatB

	full_text

	float %10
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
e%14 = tail call float @llvm.fmuladd.f32(float %8, float 0xC00B5C2900000000, float 0x404FE58580000000)
&floatB

	full_text


float %8
rcallBj
h
	full_text[
Y
W%15 = tail call float @llvm.fmuladd.f32(float %10, float 0xC0E4B9CA60000000, float %14)
'floatB

	full_text

	float %10
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
e%19 = tail call float @llvm.fmuladd.f32(float %8, float 0xC00DEB8520000000, float 0x40505D9020000000)
&floatB

	full_text


float %8
rcallBj
h
	full_text[
Y
W%20 = tail call float @llvm.fmuladd.f32(float %10, float 0xC08E71D1E0000000, float %19)
'floatB

	full_text

	float %10
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
e%24 = tail call float @llvm.fmuladd.f32(float %8, float 0xC0048F5C20000000, float 0x404BC7F460000000)
&floatB

	full_text


float %8
rcallBj
h
	full_text[
Y
W%25 = tail call float @llvm.fmuladd.f32(float %10, float 0xC08668AB80000000, float %24)
'floatB

	full_text

	float %10
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
ÄcallBx
v
	full_texti
g
e%29 = tail call float @llvm.fmuladd.f32(float %8, float 0xC0091EB860000000, float 0x404FAA9E00000000)
&floatB

	full_text


float %8
rcallBj
h
	full_text[
Y
W%30 = tail call float @llvm.fmuladd.f32(float %10, float 0xC08357A6E0000000, float %29)
'floatB

	full_text

	float %10
'floatB

	full_text

	float %29
GcallB?
=
	full_text0
.
,%31 = tail call float @_Z3expf(float %30) #3
'floatB

	full_text

	float %30
.addB'
%
	full_text

%32 = add i64 %4, 32
"i64B

	full_text


i64 %4
ZgetelementptrBI
G
	full_text:
8
6%33 = getelementptr inbounds float, float* %1, i64 %32
#i64B

	full_text
	
i64 %32
JstoreBA
?
	full_text2
0
.store float %31, float* %33, align 4, !tbaa !8
'floatB

	full_text

	float %31
)float*B

	full_text


float* %33
ÄcallBx
v
	full_texti
g
e%34 = tail call float @llvm.fmuladd.f32(float %8, float 0xC01470A3E0000000, float 0x40533E63E0000000)
&floatB

	full_text


float %8
rcallBj
h
	full_text[
Y
W%35 = tail call float @llvm.fmuladd.f32(float %10, float 0xC0ABE4A500000000, float %34)
'floatB

	full_text

	float %10
'floatB

	full_text

	float %34
GcallB?
=
	full_text0
.
,%36 = tail call float @_Z3expf(float %35) #3
'floatB

	full_text

	float %35
.addB'
%
	full_text

%37 = add i64 %4, 40
"i64B

	full_text


i64 %4
ZgetelementptrBI
G
	full_text:
8
6%38 = getelementptr inbounds float, float* %1, i64 %37
#i64B

	full_text
	
i64 %37
JstoreBA
?
	full_text2
0
.store float %36, float* %38, align 4, !tbaa !8
'floatB

	full_text

	float %36
)float*B

	full_text


float* %38
ÄcallBx
v
	full_texti
g
e%39 = tail call float @llvm.fmuladd.f32(float %8, float 0xC013333340000000, float 0x4051776CC0000000)
&floatB

	full_text


float %8
rcallBj
h
	full_text[
Y
W%40 = tail call float @llvm.fmuladd.f32(float %10, float 0xC0A5DBC500000000, float %39)
'floatB

	full_text

	float %10
'floatB

	full_text

	float %39
GcallB?
=
	full_text0
.
,%41 = tail call float @_Z3expf(float %40) #3
'floatB

	full_text

	float %40
.addB'
%
	full_text

%42 = add i64 %4, 48
"i64B

	full_text


i64 %4
ZgetelementptrBI
G
	full_text:
8
6%43 = getelementptr inbounds float, float* %1, i64 %42
#i64B

	full_text
	
i64 %42
JstoreBA
?
	full_text2
0
.store float %41, float* %43, align 4, !tbaa !8
'floatB

	full_text

	float %41
)float*B

	full_text


float* %43
ÄcallBx
v
	full_texti
g
e%44 = tail call float @llvm.fmuladd.f32(float %8, float 0xC0130A3D80000000, float 0x4053391C60000000)
&floatB

	full_text


float %8
rcallBj
h
	full_text[
Y
W%45 = tail call float @llvm.fmuladd.f32(float %10, float 0xC0932F6500000000, float %44)
'floatB

	full_text

	float %10
'floatB

	full_text

	float %44
GcallB?
=
	full_text0
.
,%46 = tail call float @_Z3expf(float %45) #3
'floatB

	full_text

	float %45
.addB'
%
	full_text

%47 = add i64 %4, 56
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
ÄcallBx
v
	full_texti
g
e%49 = tail call float @llvm.fmuladd.f32(float %8, float 0xC0232D0E60000000, float 0x405BD400C0000000)
&floatB

	full_text


float %8
rcallBj
h
	full_text[
Y
W%50 = tail call float @llvm.fmuladd.f32(float %10, float 0xC0A40CCF60000000, float %49)
'floatB

	full_text

	float %10
'floatB

	full_text

	float %49
GcallB?
=
	full_text0
.
,%51 = tail call float @_Z3expf(float %50) #3
'floatB

	full_text

	float %50
.addB'
%
	full_text

%52 = add i64 %4, 64
"i64B

	full_text


i64 %4
ZgetelementptrBI
G
	full_text:
8
6%53 = getelementptr inbounds float, float* %1, i64 %52
#i64B

	full_text
	
i64 %52
JstoreBA
?
	full_text2
0
.store float %51, float* %53, align 4, !tbaa !8
'floatB

	full_text

	float %51
)float*B

	full_text


float* %53
ÄcallBx
v
	full_texti
g
e%54 = tail call float @llvm.fmuladd.f32(float %8, float 0xC023570A40000000, float 0x405CECD0A0000000)
&floatB

	full_text


float %8
rcallBj
h
	full_text[
Y
W%55 = tail call float @llvm.fmuladd.f32(float %10, float 0xC0A87403E0000000, float %54)
'floatB

	full_text

	float %10
'floatB

	full_text

	float %54
GcallB?
=
	full_text0
.
,%56 = tail call float @_Z3expf(float %55) #3
'floatB

	full_text

	float %55
.addB'
%
	full_text

%57 = add i64 %4, 72
"i64B

	full_text


i64 %4
ZgetelementptrBI
G
	full_text:
8
6%58 = getelementptr inbounds float, float* %1, i64 %57
#i64B

	full_text
	
i64 %57
JstoreBA
?
	full_text2
0
.store float %56, float* %58, align 4, !tbaa !8
'floatB

	full_text

	float %56
)float*B

	full_text


float* %58
ÄcallBx
v
	full_texti
g
e%59 = tail call float @llvm.fmuladd.f32(float %8, float 0xBFE47AE140000000, float 0x4041B7A9A0000000)
&floatB

	full_text


float %8
rcallBj
h
	full_text[
Y
W%60 = tail call float @llvm.fmuladd.f32(float %10, float 0xC0D86C77A0000000, float %59)
'floatB

	full_text

	float %10
'floatB

	full_text

	float %59
GcallB?
=
	full_text0
.
,%61 = tail call float @_Z3expf(float %60) #3
'floatB

	full_text

	float %60
.addB'
%
	full_text

%62 = add i64 %4, 80
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
e%64 = tail call float @llvm.fmuladd.f32(float %8, float 0xC00B333340000000, float 0x404F8E4E00000000)
&floatB

	full_text


float %8
rcallBj
h
	full_text[
Y
W%65 = tail call float @llvm.fmuladd.f32(float %10, float 0xC0D197A0C0000000, float %64)
'floatB

	full_text

	float %10
'floatB

	full_text

	float %64
GcallB?
=
	full_text0
.
,%66 = tail call float @_Z3expf(float %65) #3
'floatB

	full_text

	float %65
.addB'
%
	full_text

%67 = add i64 %4, 88
"i64B

	full_text


i64 %4
ZgetelementptrBI
G
	full_text:
8
6%68 = getelementptr inbounds float, float* %1, i64 %67
#i64B

	full_text
	
i64 %67
JstoreBA
?
	full_text2
0
.store float %66, float* %68, align 4, !tbaa !8
'floatB

	full_text

	float %66
)float*B

	full_text


float* %68
ÄcallBx
v
	full_texti
g
e%69 = tail call float @llvm.fmuladd.f32(float %8, float 0xC01E8F5C20000000, float 0x4057EF6C60000000)
&floatB

	full_text


float %8
rcallBj
h
	full_text[
Y
W%70 = tail call float @llvm.fmuladd.f32(float %10, float 0xC0B7644740000000, float %69)
'floatB

	full_text

	float %10
'floatB

	full_text

	float %69
GcallB?
=
	full_text0
.
,%71 = tail call float @_Z3expf(float %70) #3
'floatB

	full_text

	float %70
.addB'
%
	full_text

%72 = add i64 %4, 96
"i64B

	full_text


i64 %4
ZgetelementptrBI
G
	full_text:
8
6%73 = getelementptr inbounds float, float* %1, i64 %72
#i64B

	full_text
	
i64 %72
JstoreBA
?
	full_text2
0
.store float %71, float* %73, align 4, !tbaa !8
'floatB

	full_text

	float %71
)float*B

	full_text


float* %73
ÄcallBx
v
	full_texti
g
e%74 = tail call float @llvm.fmuladd.f32(float %8, float 0xC00EE147A0000000, float 0x40515A7F60000000)
&floatB

	full_text


float %8
rcallBj
h
	full_text[
Y
W%75 = tail call float @llvm.fmuladd.f32(float %10, float 0xC09A1AB7A0000000, float %74)
'floatB

	full_text

	float %10
'floatB

	full_text

	float %74
GcallB?
=
	full_text0
.
,%76 = tail call float @_Z3expf(float %75) #3
'floatB

	full_text

	float %75
/addB(
&
	full_text

%77 = add i64 %4, 104
"i64B

	full_text


i64 %4
ZgetelementptrBI
G
	full_text:
8
6%78 = getelementptr inbounds float, float* %1, i64 %77
#i64B

	full_text
	
i64 %77
JstoreBA
?
	full_text2
0
.store float %76, float* %78, align 4, !tbaa !8
'floatB

	full_text

	float %76
)float*B

	full_text


float* %78
ÄcallBx
v
	full_texti
g
e%79 = tail call float @llvm.fmuladd.f32(float %8, float 0xC027E147A0000000, float 0x4060E00CC0000000)
&floatB

	full_text


float %8
rcallBj
h
	full_text[
Y
W%80 = tail call float @llvm.fmuladd.f32(float %10, float 0xC0B3345380000000, float %79)
'floatB

	full_text

	float %10
'floatB

	full_text

	float %79
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
%82 = add i64 %4, 112
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
ÄcallBx
v
	full_texti
g
e%84 = tail call float @llvm.fmuladd.f32(float %8, float 0xC01D3020C0000000, float 0x4056DCC440000000)
&floatB

	full_text


float %8
rcallBj
h
	full_text[
Y
W%85 = tail call float @llvm.fmuladd.f32(float %10, float 0xC0A27A3CA0000000, float %84)
'floatB

	full_text

	float %10
'floatB

	full_text

	float %84
GcallB?
=
	full_text0
.
,%86 = tail call float @_Z3expf(float %85) #3
'floatB

	full_text

	float %85
/addB(
&
	full_text

%87 = add i64 %4, 120
"i64B

	full_text


i64 %4
ZgetelementptrBI
G
	full_text:
8
6%88 = getelementptr inbounds float, float* %1, i64 %87
#i64B

	full_text
	
i64 %87
JstoreBA
?
	full_text2
0
.store float %86, float* %88, align 4, !tbaa !8
'floatB

	full_text

	float %86
)float*B

	full_text


float* %88
ÄcallBx
v
	full_texti
g
e%89 = tail call float @llvm.fmuladd.f32(float %8, float 0xC0229EB860000000, float 0x405D44CF80000000)
&floatB

	full_text


float %8
rcallBj
h
	full_text[
Y
W%90 = tail call float @llvm.fmuladd.f32(float %10, float 0xC0E88966E0000000, float %89)
'floatB

	full_text

	float %10
'floatB

	full_text

	float %89
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
%92 = add i64 %4, 128
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
e%94 = tail call float @llvm.fmuladd.f32(float %8, float 0xC01E7AE140000000, float 0x4058390460000000)
&floatB

	full_text


float %8
rcallBj
h
	full_text[
Y
W%95 = tail call float @llvm.fmuladd.f32(float %10, float 0xC0AB66D720000000, float %94)
'floatB

	full_text

	float %10
'floatB

	full_text

	float %94
GcallB?
=
	full_text0
.
,%96 = tail call float @_Z3expf(float %95) #3
'floatB

	full_text

	float %95
/addB(
&
	full_text

%97 = add i64 %4, 136
"i64B

	full_text


i64 %4
ZgetelementptrBI
G
	full_text:
8
6%98 = getelementptr inbounds float, float* %1, i64 %97
#i64B

	full_text
	
i64 %97
JstoreBA
?
	full_text2
0
.store float %96, float* %98, align 4, !tbaa !8
'floatB

	full_text

	float %96
)float*B

	full_text


float* %98
ÄcallBx
v
	full_texti
g
e%99 = tail call float @llvm.fmuladd.f32(float %8, float 0xC01C51EB80000000, float 0x4057C60620000000)
&floatB

	full_text


float %8
scallBk
i
	full_text\
Z
X%100 = tail call float @llvm.fmuladd.f32(float %10, float 0xC0AA4801C0000000, float %99)
'floatB

	full_text

	float %10
'floatB

	full_text

	float %99
IcallBA
?
	full_text2
0
.%101 = tail call float @_Z3expf(float %100) #3
(floatB

	full_text


float %100
0addB)
'
	full_text

%102 = add i64 %4, 144
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%103 = getelementptr inbounds float, float* %1, i64 %102
$i64B

	full_text


i64 %102
LstoreBC
A
	full_text4
2
0store float %101, float* %103, align 4, !tbaa !8
(floatB

	full_text


float %101
*float*B

	full_text

float* %103
|callBt
r
	full_texte
c
a%104 = tail call float @llvm.fmuladd.f32(float %8, float -1.200000e+01, float 0x40614E16E0000000)
&floatB

	full_text


float %8
tcallBl
j
	full_text]
[
Y%105 = tail call float @llvm.fmuladd.f32(float %10, float 0xC0A7763160000000, float %104)
'floatB

	full_text

	float %10
(floatB

	full_text


float %104
IcallBA
?
	full_text2
0
.%106 = tail call float @_Z3expf(float %105) #3
(floatB

	full_text


float %105
0addB)
'
	full_text

%107 = add i64 %4, 152
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%108 = getelementptr inbounds float, float* %1, i64 %107
$i64B

	full_text


i64 %107
LstoreBC
A
	full_text4
2
0store float %106, float* %108, align 4, !tbaa !8
(floatB

	full_text


float %106
*float*B

	full_text

float* %108
ÅcallBy
w
	full_textj
h
f%109 = tail call float @llvm.fmuladd.f32(float %8, float 0xC01AA3D700000000, float 0x4056554640000000)
&floatB

	full_text


float %8
tcallBl
j
	full_text]
[
Y%110 = tail call float @llvm.fmuladd.f32(float %10, float 0xC0AB850880000000, float %109)
'floatB

	full_text

	float %10
(floatB

	full_text


float %109
IcallBA
?
	full_text2
0
.%111 = tail call float @_Z3expf(float %110) #3
(floatB

	full_text


float %110
0addB)
'
	full_text

%112 = add i64 %4, 160
"i64B

	full_text


i64 %4
\getelementptrBK
I
	full_text<
:
8%113 = getelementptr inbounds float, float* %1, i64 %112
$i64B

	full_text


i64 %112
LstoreBC
A
	full_text4
2
0store float %111, float* %113, align 4, !tbaa !8
(floatB

	full_text


float %111
*float*B

	full_text

float* %113
"retB

	full_text


ret void
*float*8B

	full_text

	float* %1
(float8B

	full_text


float %2
*float*8B

	full_text

	float* %0
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
8float8B+
)
	full_text

float 0xC0ABE4A500000000
8float8B+
)
	full_text

float 0x405BD400C0000000
8float8B+
)
	full_text

float 0xC00EE147A0000000
8float8B+
)
	full_text

float 0xC01D3020C0000000
8float8B+
)
	full_text

float 0xC0A27A3CA0000000
8float8B+
)
	full_text

float 0xC0E4B9CA60000000
8float8B+
)
	full_text

float 0xBFE47AE140000000
$i648B

	full_text


i64 16
8float8B+
)
	full_text

float 0x405CECD0A0000000
8float8B+
)
	full_text

float 0xC0A87403E0000000
#i648B

	full_text	

i64 8
$i648B

	full_text


i64 48
8float8B+
)
	full_text

float 0x4053391C60000000
8float8B+
)
	full_text

float 0xC0AB66D720000000
8float8B+
)
	full_text

float 0x4057C60620000000
8float8B+
)
	full_text

float 0x4060E00CC0000000
8float8B+
)
	full_text

float 0x4056554640000000
$i648B

	full_text


i64 72
%i648B

	full_text
	
i64 112
%i648B

	full_text
	
i64 104
8float8B+
)
	full_text

float 0xC0E88966E0000000
8float8B+
)
	full_text

float 0xC00DEB8520000000
8float8B+
)
	full_text

float 0x40515A7F60000000
%i648B

	full_text
	
i64 120
8float8B+
)
	full_text

float 0xC0229EB860000000
8float8B+
)
	full_text

float 0xC01E8F5C20000000
8float8B+
)
	full_text

float 0xC0A40CCF60000000
8float8B+
)
	full_text

float 0xC013333340000000
%i648B

	full_text
	
i64 144
8float8B+
)
	full_text

float 0xC0A7763160000000
8float8B+
)
	full_text

float 0xC08357A6E0000000
8float8B+
)
	full_text

float 0x40505D9020000000
8float8B+
)
	full_text

float 0x404F8E4E00000000
8float8B+
)
	full_text

float 0x404FE58580000000
8float8B+
)
	full_text

float 0x4051776CC0000000
$i648B

	full_text


i64 56
$i648B

	full_text


i64 88
8float8B+
)
	full_text

float 0xC0D197A0C0000000
8float8B+
)
	full_text

float 0xC0B3345380000000
8float8B+
)
	full_text

float 0x4058390460000000
3float8B&
$
	full_text

float -1.200000e+01
2float8B%
#
	full_text

float 1.000000e+00
8float8B+
)
	full_text

float 0xC01C51EB80000000
8float8B+
)
	full_text

float 0x404523C4C0000000
8float8B+
)
	full_text

float 0xC027E147A0000000
8float8B+
)
	full_text

float 0xC00B333340000000
8float8B+
)
	full_text

float 0x40614E16E0000000
$i648B

	full_text


i64 64
$i648B

	full_text


i64 80
8float8B+
)
	full_text

float 0x405D44CF80000000
%i648B

	full_text
	
i64 136
$i648B

	full_text


i64 24
8float8B+
)
	full_text

float 0xC01E7AE140000000
8float8B+
)
	full_text

float 0xC01470A3E0000000
8float8B+
)
	full_text

float 0xC0932F6500000000
%i648B

	full_text
	
i64 128
%i648B

	full_text
	
i64 160
8float8B+
)
	full_text

float 0x408ABBBF20000000
8float8B+
)
	full_text

float 0xC0D86C77A0000000
#i328B

	full_text	

i32 0
8float8B+
)
	full_text

float 0xC09A1AB7A0000000
%i648B

	full_text
	
i64 152
8float8B+
)
	full_text

float 0x4041B7A9A0000000
8float8B+
)
	full_text

float 0xC08668AB80000000
$i648B

	full_text


i64 40
8float8B+
)
	full_text

float 0xC0AA4801C0000000
8float8B+
)
	full_text

float 0x4057EF6C60000000
8float8B+
)
	full_text

float 0xC00B5C2900000000
8float8B+
)
	full_text

float 0xC0130A3D80000000
8float8B+
)
	full_text

float 0xBFECCCCCC0000000
8float8B+
)
	full_text

float 0xC0232D0E60000000
8float8B+
)
	full_text

float 0x404BC7F460000000
$i648B

	full_text


i64 96
8float8B+
)
	full_text

float 0xC0091EB860000000
8float8B+
)
	full_text

float 0xC08E71D1E0000000
$i648B

	full_text


i64 32
8float8B+
)
	full_text

float 0x40533E63E0000000
8float8B+
)
	full_text

float 0xC01AA3D700000000
8float8B+
)
	full_text

float 0xC0A5DBC500000000
8float8B+
)
	full_text

float 0xC0AB850880000000
8float8B+
)
	full_text

float 0xC0B7644740000000
8float8B+
)
	full_text

float 0xC0048F5C20000000
8float8B+
)
	full_text

float 0xC023570A40000000
8float8B+
)
	full_text

float 0x4056DCC440000000
8float8B+
)
	full_text

float 0x404FAA9E00000000       	  
 

                       !" !! #$ #% ## &' && () (* (( +, ++ -. -- /0 // 12 13 11 45 44 67 68 66 9: 99 ;< ;; => == ?@ ?A ?? BC BB DE DF DD GH GG IJ II KL KK MN MO MM PQ PP RS RT RR UV UU WX WW YZ YY [\ [] [[ ^_ ^^ `a `b `` cd cc ef ee gh gg ij ik ii lm ll no np nn qr qq st ss uv uu wx wy ww z{ zz |} |~ || Ä  ÅÇ ÅÅ É
Ñ ÉÉ ÖÜ Ö
á ÖÖ àâ àà äã ä
å ää çé çç èê èè ë
í ëë ìî ì
ï ìì ñó ññ òô ò
ö òò õú õõ ùû ùù ü
† üü °¢ °
£ °° §• §§ ¶ß ¶
® ¶¶ ©™ ©© ´¨ ´´ ≠
Æ ≠≠ Ø∞ Ø
± ØØ ≤≥ ≤≤ ¥µ ¥
∂ ¥¥ ∑∏ ∑∑ π∫ ππ ª
º ªª Ωæ Ω
ø ΩΩ ¿¡ ¿¿ ¬√ ¬
ƒ ¬¬ ≈∆ ≈≈ «» «« …
  …… ÀÃ À
Õ ÀÀ Œœ ŒŒ –— –
“ –– ”‘ ”” ’÷ ’’ ◊
ÿ ◊◊ Ÿ⁄ Ÿ
€ ŸŸ ‹› ‹‹ ﬁﬂ ﬁ
‡ ﬁﬁ ·‚ ·· „‰ „„ Â
Ê ÂÂ ÁË Á
È ÁÁ ÍÎ ÍÍ ÏÌ Ï
Ó ÏÏ Ô ÔÔ ÒÚ ÒÒ Û
Ù ÛÛ ıˆ ı
˜ ıı ¯˘ ¯¯ ˙˚ ˙
¸ ˙˙ ˝˛ ˝˝ ˇÄ ˇˇ Å
Ç ÅÅ ÉÑ É
Ö ÉÉ Üá ÜÜ àâ à
ä àà ãå ãã çé çç è
ê èè ëí ë
ì ëë îï îî ñó ñ
ò ññ ôö ôô õú õõ ù
û ùù ü† ü
° üü ¢£ ¢¢ §• §
¶ §§ ß® ßß ©™ ©© ´
¨ ´´ ≠Æ ≠
Ø ≠≠ ∞± ± !± /± =± K± Y± g± u± É± ë± ü± ≠± ª± …± ◊± Â± Û± Å± è± ù± ´	≤ ≥     	   
            " $! % ' )& *( , .- 0+ 2/ 3 5 74 86 : <; >9 @= A C EB FD H JI LG NK O Q SP TR V XW ZU \Y ] _ a^ b` d fe hc jg k m ol pn r ts vq xu y { }z ~| Ä ÇÅ Ñ ÜÉ á â ãà åä é êè íç îë ï ó ôñ öò ú ûù †õ ¢ü £ • ß§ ®¶ ™ ¨´ Æ© ∞≠ ± ≥ µ≤ ∂¥ ∏ ∫π º∑ æª ø ¡ √¿ ƒ¬ ∆ »«  ≈ Ã… Õ œ —Œ “– ‘ ÷’ ÿ” ⁄◊ € › ﬂ‹ ‡ﬁ ‚ ‰„ Ê· ËÂ È Î ÌÍ ÓÏ  ÚÒ ÙÔ ˆÛ ˜ ˘ ˚¯ ¸˙ ˛ Äˇ Ç˝ ÑÅ Ö á âÜ äà å éç êã íè ì ï óî òñ ö úõ ûô †ù ° £ •¢ ¶§ ® ™© ¨ß Æ´ Ø ¥¥ ∞ ∂∂ ∑∑ µµ
 ∑∑ 
B ∑∑ B¥ ∑∑ ¥¢ ∑∑ ¢ﬁ ∑∑ ﬁÏ ∑∑ Ïß ∂∂ ß` ∑∑ `≈ ∂∂ ≈ µµ ç ∂∂ ç˙ ∑∑ ˙+ ∂∂ +6 ∑∑ 69 ∂∂ 9l ∑∑ lP ∑∑ P¬ ∑∑ ¬ñ ∑∑ ñD ∑∑ Dc ∂∂ cã ∂∂ ã ∂∂ Œ ∑∑ Œà ∑∑ àG ∂∂ G| ∑∑ |ä ∑∑ äñ ∑∑ ñò ∑∑ òõ ∂∂ õz ∑∑ z4 ∑∑ 4Í ∑∑ Í§ ∑∑ § ∑∑ î ∑∑ îà ∑∑ à¯ ∑∑ ¯q ∂∂ q© ∂∂ ©∑ ∂∂ ∑– ∑∑ –‹ ∑∑ ‹ô ∂∂ ô¶ ∑∑ ¶Ü ∑∑ Ü¿ ∑∑ ¿n ∑∑ n ∂∂  ∑∑ ˝ ∂∂ ˝≤ ∑∑ ≤” ∂∂ ” ∑∑ § ∑∑ § ¥¥ ( ∑∑ (· ∂∂ ·^ ∑∑ ^U ∂∂ U ∂∂ & ∑∑ &R ∑∑ RÔ ∂∂ Ô	∏ R	π z
∫ ¿
ª ‹
º ﬁ	Ω 
æ ñ	ø -
¿ à
¡ ä	¬ 	√ e	ƒ l
≈ ˙
∆ Ü
« Œ
» ¢
… è
  ’
À «
Ã Ï	Õ &
Œ ¿
œ „
– Í
— ≤	“ |	” ^
‘ ç
’ ñ	÷ D	◊ &
ÿ §	Ÿ 	⁄ ^	€ s
‹ ´
› ¶
ﬁ –
ﬂ ¯
‡ î· 
‚ Ü	„ 

‰ Œ
Â §
Ê î
Á Å
Ë ù
È Í
Í ˇ	Î ;
Ï ¯	Ì P	Ó n
Ô Ò
 ©	Ò 
Ú òÛ 
Ù ¬
ı õ
ˆ ñ	˜ 6	¯ W
˘ à
˙ ≤	˚ 	¸ l	˝ 
	˛ z	ˇ 4
Ä π	Å B	Ç (	É I	Ñ P
Ö ¢	Ü `
á §
à ¥	â 4
ä à
ã ‹	å B"
ratt10_kernel"
_Z13get_global_idj"	
_Z3logf"	
_Z3expf"
llvm.fmuladd.f32*ò
shoc-1.1.5-S3D-ratt10_kernel.clu
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

transfer_bytes
à¢ª

wgsize_log1p
íÛéA

wgsize
Ä