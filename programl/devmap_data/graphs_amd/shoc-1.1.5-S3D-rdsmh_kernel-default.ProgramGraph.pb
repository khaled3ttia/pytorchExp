
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
=faddB5
3
	full_text&
$
"%10 = fadd float %8, -1.000000e+00
&floatB

	full_text


float %8
@fcmpB8
6
	full_text)
'
%%11 = fcmp ogt float %7, 1.000000e+03
&floatB

	full_text


float %7
9brB3
1
	full_text$
"
 br i1 %11, label %12, label %317
!i1B

	full_text


i1 %11
‚call8Bx
v
	full_texti
g
e%13 = tail call float @llvm.fmuladd.f32(float %9, float 0x408DB14580000000, float 0xC009A3E340000000)
(float8B

	full_text


float %9
tcall8Bj
h
	full_text[
Y
W%14 = tail call float @llvm.fmuladd.f32(float %10, float 0x400AB2BF60000000, float %13)
)float8B

	full_text

	float %10
)float8B

	full_text

	float %13
…call8B{
y
	full_textl
j
h%15 = tail call float @llvm.fmuladd.f32(float %7, float 0x3CD2099320000000, float 0xBDB073F440000000) #4
(float8B

	full_text


float %7
vcall8Bl
j
	full_text]
[
Y%16 = tail call float @llvm.fmuladd.f32(float %15, float %7, float 0x3E765866C0000000) #4
)float8B

	full_text

	float %15
(float8B

	full_text


float %7
vcall8Bl
j
	full_text]
[
Y%17 = tail call float @llvm.fmuladd.f32(float %16, float %7, float 0xBEF9E6B000000000) #4
)float8B

	full_text

	float %16
(float8B

	full_text


float %7
5fmul8B+
)
	full_text

%18 = fmul float %7, %17
(float8B

	full_text


float %7
)float8B

	full_text

	float %17
6fadd8B,
*
	full_text

%19 = fadd float %14, %18
)float8B

	full_text

	float %14
)float8B

	full_text

	float %18
Icall8B?
=
	full_text0
.
,%20 = tail call float @_Z3expf(float %19) #3
)float8B

	full_text

	float %19
[getelementptr8BH
F
	full_text9
7
5%21 = getelementptr inbounds float, float* %1, i64 %4
$i648B

	full_text


i64 %4
Lstore8BA
?
	full_text2
0
.store float %20, float* %21, align 4, !tbaa !8
)float8B

	full_text

	float %20
+float*8B

	full_text


float* %21
‚call8Bx
v
	full_texti
g
e%22 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0D8E06A40000000, float 0xBFDC9673E0000000)
(float8B

	full_text


float %9
ncall8Bd
b
	full_textU
S
Q%23 = tail call float @llvm.fmuladd.f32(float %10, float 2.500000e+00, float %22)
)float8B

	full_text

	float %10
)float8B

	full_text

	float %22
…call8B{
y
	full_textl
j
h%24 = tail call float @llvm.fmuladd.f32(float %7, float 0x3B3E1D3B00000000, float 0xBC1D1DB540000000) #4
(float8B

	full_text


float %7
vcall8Bl
j
	full_text]
[
Y%25 = tail call float @llvm.fmuladd.f32(float %24, float %7, float 0x3CE840F100000000) #4
)float8B

	full_text

	float %24
(float8B

	full_text


float %7
vcall8Bl
j
	full_text]
[
Y%26 = tail call float @llvm.fmuladd.f32(float %25, float %7, float 0xBDA961A6E0000000) #4
)float8B

	full_text

	float %25
(float8B

	full_text


float %7
5fmul8B+
)
	full_text

%27 = fmul float %7, %26
(float8B

	full_text


float %7
)float8B

	full_text

	float %26
6fadd8B,
*
	full_text

%28 = fadd float %23, %27
)float8B

	full_text

	float %23
)float8B

	full_text

	float %27
Icall8B?
=
	full_text0
.
,%29 = tail call float @_Z3expf(float %28) #3
)float8B

	full_text

	float %28
/add8B&
$
	full_text

%30 = add i64 %4, 8
$i648B

	full_text


i64 %4
\getelementptr8BI
G
	full_text:
8
6%31 = getelementptr inbounds float, float* %1, i64 %30
%i648B

	full_text
	
i64 %30
Lstore8BA
?
	full_text2
0
.store float %29, float* %31, align 4, !tbaa !8
)float8B

	full_text

	float %29
+float*8B

	full_text


float* %31
‚call8Bx
v
	full_texti
g
e%32 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0DC886500000000, float 0x40132329A0000000)
(float8B

	full_text


float %9
tcall8Bj
h
	full_text[
Y
W%33 = tail call float @llvm.fmuladd.f32(float %10, float 0x40048E2C80000000, float %32)
)float8B

	full_text

	float %10
)float8B

	full_text

	float %32
…call8B{
y
	full_textl
j
h%34 = tail call float @llvm.fmuladd.f32(float %7, float 0x3C91B3C360000000, float 0xBD6D5F5860000000) #4
(float8B

	full_text


float %7
vcall8Bl
j
	full_text]
[
Y%35 = tail call float @llvm.fmuladd.f32(float %34, float %7, float 0x3E3E0722E0000000) #4
)float8B

	full_text

	float %34
(float8B

	full_text


float %7
vcall8Bl
j
	full_text]
[
Y%36 = tail call float @llvm.fmuladd.f32(float %35, float %7, float 0xBF0689A000000000) #4
)float8B

	full_text

	float %35
(float8B

	full_text


float %7
5fmul8B+
)
	full_text

%37 = fmul float %7, %36
(float8B

	full_text


float %7
)float8B

	full_text

	float %36
6fadd8B,
*
	full_text

%38 = fadd float %33, %37
)float8B

	full_text

	float %33
)float8B

	full_text

	float %37
Icall8B?
=
	full_text0
.
,%39 = tail call float @_Z3expf(float %38) #3
)float8B

	full_text

	float %38
0add8B'
%
	full_text

%40 = add i64 %4, 16
$i648B

	full_text


i64 %4
\getelementptr8BI
G
	full_text:
8
6%41 = getelementptr inbounds float, float* %1, i64 %40
%i648B

	full_text
	
i64 %40
Lstore8BA
?
	full_text2
0
.store float %39, float* %41, align 4, !tbaa !8
)float8B

	full_text

	float %39
+float*8B

	full_text


float* %41
‚call8Bx
v
	full_texti
g
e%42 = tail call float @llvm.fmuladd.f32(float %9, float 0x409101D4C0000000, float 0x4015D01BE0000000)
(float8B

	full_text


float %9
tcall8Bj
h
	full_text[
Y
W%43 = tail call float @llvm.fmuladd.f32(float %10, float 0x400A42A340000000, float %42)
)float8B

	full_text

	float %10
)float8B

	full_text

	float %42
…call8B{
y
	full_textl
j
h%44 = tail call float @llvm.fmuladd.f32(float %7, float 0xBCD3852C00000000, float 0x3DB33164A0000000) #4
(float8B

	full_text


float %7
vcall8Bl
j
	full_text]
[
Y%45 = tail call float @llvm.fmuladd.f32(float %44, float %7, float 0xBE80F496E0000000) #4
)float8B

	full_text

	float %44
(float8B

	full_text


float %7
vcall8Bl
j
	full_text]
[
Y%46 = tail call float @llvm.fmuladd.f32(float %45, float %7, float 0x3F484C8520000000) #4
)float8B

	full_text

	float %45
(float8B

	full_text


float %7
5fmul8B+
)
	full_text

%47 = fmul float %7, %46
(float8B

	full_text


float %7
)float8B

	full_text

	float %46
6fadd8B,
*
	full_text

%48 = fadd float %43, %47
)float8B

	full_text

	float %43
)float8B

	full_text

	float %47
Icall8B?
=
	full_text0
.
,%49 = tail call float @_Z3expf(float %48) #3
)float8B

	full_text

	float %48
0add8B'
%
	full_text

%50 = add i64 %4, 24
$i648B

	full_text


i64 %4
\getelementptr8BI
G
	full_text:
8
6%51 = getelementptr inbounds float, float* %1, i64 %50
%i648B

	full_text
	
i64 %50
Lstore8BA
?
	full_text2
0
.store float %49, float* %51, align 4, !tbaa !8
)float8B

	full_text

	float %49
+float*8B

	full_text


float* %51
‚call8Bx
v
	full_texti
g
e%52 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0AE255060000000, float 0x4011E82300000000)
(float8B

	full_text


float %9
tcall8Bj
h
	full_text[
Y
W%53 = tail call float @llvm.fmuladd.f32(float %10, float 0x4008BE3BE0000000, float %52)
)float8B

	full_text

	float %10
)float8B

	full_text

	float %52
…call8B{
y
	full_textl
j
h%54 = tail call float @llvm.fmuladd.f32(float %7, float 0x3CC526B0A0000000, float 0xBDA01DC620000000) #4
(float8B

	full_text


float %7
vcall8Bl
j
	full_text]
[
Y%55 = tail call float @llvm.fmuladd.f32(float %54, float %7, float 0x3E56A39500000000) #4
)float8B

	full_text

	float %54
(float8B

	full_text


float %7
vcall8Bl
j
	full_text]
[
Y%56 = tail call float @llvm.fmuladd.f32(float %55, float %7, float 0x3F31F88FE0000000) #4
)float8B

	full_text

	float %55
(float8B

	full_text


float %7
5fmul8B+
)
	full_text

%57 = fmul float %7, %56
(float8B

	full_text


float %7
)float8B

	full_text

	float %56
6fadd8B,
*
	full_text

%58 = fadd float %53, %57
)float8B

	full_text

	float %53
)float8B

	full_text

	float %57
Icall8B?
=
	full_text0
.
,%59 = tail call float @_Z3expf(float %58) #3
)float8B

	full_text

	float %58
0add8B'
%
	full_text

%60 = add i64 %4, 32
$i648B

	full_text


i64 %4
\getelementptr8BI
G
	full_text:
8
6%61 = getelementptr inbounds float, float* %1, i64 %60
%i648B

	full_text
	
i64 %60
Lstore8BA
?
	full_text2
0
.store float %59, float* %61, align 4, !tbaa !8
)float8B

	full_text

	float %59
+float*8B

	full_text


float* %61
‚call8Bx
v
	full_texti
g
e%62 = tail call float @llvm.fmuladd.f32(float %9, float 0x40DD4D1300000000, float 0x4013DDF900000000)
(float8B

	full_text


float %9
tcall8Bj
h
	full_text[
Y
W%63 = tail call float @llvm.fmuladd.f32(float %10, float 0x4008459DE0000000, float %62)
)float8B

	full_text

	float %10
)float8B

	full_text

	float %62
…call8B{
y
	full_textl
j
h%64 = tail call float @llvm.fmuladd.f32(float %7, float 0x3CCE4CE6E0000000, float 0xBDA1C87B60000000) #4
(float8B

	full_text


float %7
vcall8Bl
j
	full_text]
[
Y%65 = tail call float @llvm.fmuladd.f32(float %64, float %7, float 0xBE5D5CA6E0000000) #4
)float8B

	full_text

	float %64
(float8B

	full_text


float %7
vcall8Bl
j
	full_text]
[
Y%66 = tail call float @llvm.fmuladd.f32(float %65, float %7, float 0x3F51D55400000000) #4
)float8B

	full_text

	float %65
(float8B

	full_text


float %7
5fmul8B+
)
	full_text

%67 = fmul float %7, %66
(float8B

	full_text


float %7
)float8B

	full_text

	float %66
6fadd8B,
*
	full_text

%68 = fadd float %63, %67
)float8B

	full_text

	float %63
)float8B

	full_text

	float %67
Icall8B?
=
	full_text0
.
,%69 = tail call float @_Z3expf(float %68) #3
)float8B

	full_text

	float %68
0add8B'
%
	full_text

%70 = add i64 %4, 40
$i648B

	full_text


i64 %4
\getelementptr8BI
G
	full_text:
8
6%71 = getelementptr inbounds float, float* %1, i64 %70
%i648B

	full_text
	
i64 %70
Lstore8BA
?
	full_text2
0
.store float %69, float* %71, align 4, !tbaa !8
)float8B

	full_text

	float %69
+float*8B

	full_text


float* %71
‚call8Bx
v
	full_texti
g
e%72 = tail call float @llvm.fmuladd.f32(float %9, float 0xC05BF6D460000000, float 0x400E47E3A0000000)
(float8B

	full_text


float %9
tcall8Bj
h
	full_text[
Y
W%73 = tail call float @llvm.fmuladd.f32(float %10, float 0x4010119FC0000000, float %72)
)float8B

	full_text

	float %10
)float8B

	full_text

	float %72
…call8B{
y
	full_textl
j
h%74 = tail call float @llvm.fmuladd.f32(float %7, float 0xBCC3706720000000, float 0x3DA4EF9520000000) #4
(float8B

	full_text


float %7
vcall8Bl
j
	full_text]
[
Y%75 = tail call float @llvm.fmuladd.f32(float %74, float %7, float 0xBE7C597160000000) #4
)float8B

	full_text

	float %74
(float8B

	full_text


float %7
vcall8Bl
j
	full_text]
[
Y%76 = tail call float @llvm.fmuladd.f32(float %75, float %7, float 0x3F52593E40000000) #4
)float8B

	full_text

	float %75
(float8B

	full_text


float %7
5fmul8B+
)
	full_text

%77 = fmul float %7, %76
(float8B

	full_text


float %7
)float8B

	full_text

	float %76
6fadd8B,
*
	full_text

%78 = fadd float %73, %77
)float8B

	full_text

	float %73
)float8B

	full_text

	float %77
Icall8B?
=
	full_text0
.
,%79 = tail call float @_Z3expf(float %78) #3
)float8B

	full_text

	float %78
0add8B'
%
	full_text

%80 = add i64 %4, 48
$i648B

	full_text


i64 %4
\getelementptr8BI
G
	full_text:
8
6%81 = getelementptr inbounds float, float* %1, i64 %80
%i648B

	full_text
	
i64 %80
Lstore8BA
?
	full_text2
0
.store float %79, float* %81, align 4, !tbaa !8
)float8B

	full_text

	float %79
+float*8B

	full_text


float* %81
‚call8Bx
v
	full_texti
g
e%82 = tail call float @llvm.fmuladd.f32(float %9, float 0x40D1717260000000, float 0x40075449E0000000)
(float8B

	full_text


float %9
tcall8Bj
h
	full_text[
Y
W%83 = tail call float @llvm.fmuladd.f32(float %10, float 0x4010A8F680000000, float %82)
)float8B

	full_text

	float %10
)float8B

	full_text

	float %82
…call8B{
y
	full_textl
j
h%84 = tail call float @llvm.fmuladd.f32(float %7, float 0xBCD9EEB6A0000000, float 0x3DC10150C0000000) #4
(float8B

	full_text


float %7
vcall8Bl
j
	full_text]
[
Y%85 = tail call float @llvm.fmuladd.f32(float %84, float %7, float 0xBE95444740000000) #4
)float8B

	full_text

	float %84
(float8B

	full_text


float %7
vcall8Bl
j
	full_text]
[
Y%86 = tail call float @llvm.fmuladd.f32(float %85, float %7, float 0x3F641ABE40000000) #4
)float8B

	full_text

	float %85
(float8B

	full_text


float %7
5fmul8B+
)
	full_text

%87 = fmul float %7, %86
(float8B

	full_text


float %7
)float8B

	full_text

	float %86
6fadd8B,
*
	full_text

%88 = fadd float %83, %87
)float8B

	full_text

	float %83
)float8B

	full_text

	float %87
Icall8B?
=
	full_text0
.
,%89 = tail call float @_Z3expf(float %88) #3
)float8B

	full_text

	float %88
0add8B'
%
	full_text

%90 = add i64 %4, 56
$i648B

	full_text


i64 %4
\getelementptr8BI
G
	full_text:
8
6%91 = getelementptr inbounds float, float* %1, i64 %90
%i648B

	full_text
	
i64 %90
Lstore8BA
?
	full_text2
0
.store float %89, float* %91, align 4, !tbaa !8
)float8B

	full_text

	float %89
+float*8B

	full_text


float* %91
‚call8Bx
v
	full_texti
g
e%92 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0F1564700000000, float 0x4015F09EA0000000)
(float8B

	full_text


float %9
tcall8Bj
h
	full_text[
Y
W%93 = tail call float @llvm.fmuladd.f32(float %10, float 0x4007071880000000, float %92)
)float8B

	full_text

	float %10
)float8B

	full_text

	float %92
…call8B{
y
	full_textl
j
h%94 = tail call float @llvm.fmuladd.f32(float %7, float 0x3CCFB83A80000000, float 0xBDA7F2E4A0000000) #4
(float8B

	full_text


float %7
vcall8Bl
j
	full_text]
[
Y%95 = tail call float @llvm.fmuladd.f32(float %94, float %7, float 0x3E59D97C80000000) #4
)float8B

	full_text

	float %94
(float8B

	full_text


float %7
vcall8Bl
j
	full_text]
[
Y%96 = tail call float @llvm.fmuladd.f32(float %95, float %7, float 0x3F3FD09D40000000) #4
)float8B

	full_text

	float %95
(float8B

	full_text


float %7
5fmul8B+
)
	full_text

%97 = fmul float %7, %96
(float8B

	full_text


float %7
)float8B

	full_text

	float %96
6fadd8B,
*
	full_text

%98 = fadd float %93, %97
)float8B

	full_text

	float %93
)float8B

	full_text

	float %97
Icall8B?
=
	full_text0
.
,%99 = tail call float @_Z3expf(float %98) #3
)float8B

	full_text

	float %98
1add8B(
&
	full_text

%100 = add i64 %4, 64
$i648B

	full_text


i64 %4
^getelementptr8BK
I
	full_text<
:
8%101 = getelementptr inbounds float, float* %1, i64 %100
&i648B

	full_text


i64 %100
Mstore8BB
@
	full_text3
1
/store float %99, float* %101, align 4, !tbaa !8
)float8B

	full_text

	float %99
,float*8B

	full_text

float* %101
ƒcall8By
w
	full_textj
h
f%102 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0E696F360000000, float 0x4018AF4D40000000)
(float8B

	full_text


float %9
vcall8Bl
j
	full_text]
[
Y%103 = tail call float @llvm.fmuladd.f32(float %10, float 0x4006FE28C0000000, float %102)
)float8B

	full_text

	float %10
*float8B

	full_text


float %102
†call8B|
z
	full_textm
k
i%104 = tail call float @llvm.fmuladd.f32(float %7, float 0xBCD0E8B400000000, float 0x3DB7D6D600000000) #4
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%105 = tail call float @llvm.fmuladd.f32(float %104, float %7, float 0xBE8F8480A0000000) #4
*float8B

	full_text


float %104
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%106 = tail call float @llvm.fmuladd.f32(float %105, float %7, float 0x3F5DF40300000000) #4
*float8B

	full_text


float %105
(float8B

	full_text


float %7
7fmul8B-
+
	full_text

%107 = fmul float %7, %106
(float8B

	full_text


float %7
*float8B

	full_text


float %106
9fadd8B/
-
	full_text 

%108 = fadd float %103, %107
*float8B

	full_text


float %103
*float8B

	full_text


float %107
Kcall8BA
?
	full_text2
0
.%109 = tail call float @_Z3expf(float %108) #3
*float8B

	full_text


float %108
1add8B(
&
	full_text

%110 = add i64 %4, 72
$i648B

	full_text


i64 %4
^getelementptr8BK
I
	full_text<
:
8%111 = getelementptr inbounds float, float* %1, i64 %110
&i648B

	full_text


i64 %110
Nstore8BC
A
	full_text4
2
0store float %109, float* %111, align 4, !tbaa !8
*float8B

	full_text


float %109
,float*8B

	full_text

float* %111
~call8Bt
r
	full_texte
c
a%112 = tail call float @llvm.fmuladd.f32(float %9, float -5.092600e+04, float 0x402140C4E0000000)
(float8B

	full_text


float %9
vcall8Bl
j
	full_text]
[
Y%113 = tail call float @llvm.fmuladd.f32(float %10, float 0x4002561840000000, float %112)
)float8B

	full_text

	float %10
*float8B

	full_text


float %112
†call8B|
z
	full_textm
k
i%114 = tail call float @llvm.fmuladd.f32(float %7, float 0xBCDE995380000000, float 0x3DC32540E0000000) #4
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%115 = tail call float @llvm.fmuladd.f32(float %114, float %7, float 0xBE9680C0A0000000) #4
*float8B

	full_text


float %114
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%116 = tail call float @llvm.fmuladd.f32(float %115, float %7, float 0x3F63120D00000000) #4
*float8B

	full_text


float %115
(float8B

	full_text


float %7
7fmul8B-
+
	full_text

%117 = fmul float %7, %116
(float8B

	full_text


float %7
*float8B

	full_text


float %116
9fadd8B/
-
	full_text 

%118 = fadd float %113, %117
*float8B

	full_text


float %113
*float8B

	full_text


float %117
Kcall8BA
?
	full_text2
0
.%119 = tail call float @_Z3expf(float %118) #3
*float8B

	full_text


float %118
1add8B(
&
	full_text

%120 = add i64 %4, 80
$i648B

	full_text


i64 %4
^getelementptr8BK
I
	full_text<
:
8%121 = getelementptr inbounds float, float* %1, i64 %120
&i648B

	full_text


i64 %120
Nstore8BC
A
	full_text4
2
0store float %119, float* %121, align 4, !tbaa !8
*float8B

	full_text


float %119
,float*8B

	full_text

float* %121
ƒcall8By
w
	full_textj
h
f%122 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0D061E560000000, float 0x4020F5CC00000000)
(float8B

	full_text


float %9
vcall8Bl
j
	full_text]
[
Y%123 = tail call float @llvm.fmuladd.f32(float %10, float 0x4002492660000000, float %122)
)float8B

	full_text

	float %10
*float8B

	full_text


float %122
†call8B|
z
	full_textm
k
i%124 = tail call float @llvm.fmuladd.f32(float %7, float 0xBCE509EC60000000, float 0x3DCB4A4360000000) #4
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%125 = tail call float @llvm.fmuladd.f32(float %124, float %7, float 0xBEA0B48FA0000000) #4
*float8B

	full_text


float %124
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%126 = tail call float @llvm.fmuladd.f32(float %125, float %7, float 0x3F6DA79600000000) #4
*float8B

	full_text


float %125
(float8B

	full_text


float %7
7fmul8B-
+
	full_text

%127 = fmul float %7, %126
(float8B

	full_text


float %7
*float8B

	full_text


float %126
9fadd8B/
-
	full_text 

%128 = fadd float %123, %127
*float8B

	full_text


float %123
*float8B

	full_text


float %127
Kcall8BA
?
	full_text2
0
.%129 = tail call float @_Z3expf(float %128) #3
*float8B

	full_text


float %128
1add8B(
&
	full_text

%130 = add i64 %4, 88
$i648B

	full_text


i64 %4
^getelementptr8BK
I
	full_text<
:
8%131 = getelementptr inbounds float, float* %1, i64 %130
&i648B

	full_text


i64 %130
Nstore8BC
A
	full_text4
2
0store float %129, float* %131, align 4, !tbaa !8
*float8B

	full_text


float %129
,float*8B

	full_text

float* %131
ƒcall8By
w
	full_textj
h
f%132 = tail call float @llvm.fmuladd.f32(float %9, float 0x40C27E2C20000000, float 0x40326FF420000000)
(float8B

	full_text


float %9
vcall8Bl
j
	full_text]
[
Y%133 = tail call float @llvm.fmuladd.f32(float %10, float 0x3FB32977C0000000, float %132)
)float8B

	full_text

	float %10
*float8B

	full_text


float %132
†call8B|
z
	full_textm
k
i%134 = tail call float @llvm.fmuladd.f32(float %7, float 0xBCF6ED3FA0000000, float 0x3DDC034F60000000) #4
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%135 = tail call float @llvm.fmuladd.f32(float %134, float %7, float 0xBEB007BD60000000) #4
*float8B

	full_text


float %134
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%136 = tail call float @llvm.fmuladd.f32(float %135, float %7, float 0x3F7B6CB680000000) #4
*float8B

	full_text


float %135
(float8B

	full_text


float %7
7fmul8B-
+
	full_text

%137 = fmul float %7, %136
(float8B

	full_text


float %7
*float8B

	full_text


float %136
9fadd8B/
-
	full_text 

%138 = fadd float %133, %137
*float8B

	full_text


float %133
*float8B

	full_text


float %137
Kcall8BA
?
	full_text2
0
.%139 = tail call float @_Z3expf(float %138) #3
*float8B

	full_text


float %138
1add8B(
&
	full_text

%140 = add i64 %4, 96
$i648B

	full_text


i64 %4
^getelementptr8BK
I
	full_text<
:
8%141 = getelementptr inbounds float, float* %1, i64 %140
&i648B

	full_text


i64 %140
Nstore8BC
A
	full_text4
2
0store float %139, float* %141, align 4, !tbaa !8
*float8B

	full_text


float %139
,float*8B

	full_text

float* %141
ƒcall8By
w
	full_textj
h
f%142 = tail call float @llvm.fmuladd.f32(float %9, float 0x40CBA3EFA0000000, float 0x401F465620000000)
(float8B

	full_text


float %9
vcall8Bl
j
	full_text]
[
Y%143 = tail call float @llvm.fmuladd.f32(float %10, float 0x4005B8B340000000, float %142)
)float8B

	full_text

	float %10
*float8B

	full_text


float %142
†call8B|
z
	full_textm
k
i%144 = tail call float @llvm.fmuladd.f32(float %7, float 0xBCD257CBE0000000, float 0x3DB5142E40000000) #4
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%145 = tail call float @llvm.fmuladd.f32(float %144, float %7, float 0xBE8657E620000000) #4
*float8B

	full_text


float %144
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%146 = tail call float @llvm.fmuladd.f32(float %145, float %7, float 0x3F50E56F00000000) #4
*float8B

	full_text


float %145
(float8B

	full_text


float %7
7fmul8B-
+
	full_text

%147 = fmul float %7, %146
(float8B

	full_text


float %7
*float8B

	full_text


float %146
9fadd8B/
-
	full_text 

%148 = fadd float %143, %147
*float8B

	full_text


float %143
*float8B

	full_text


float %147
Kcall8BA
?
	full_text2
0
.%149 = tail call float @_Z3expf(float %148) #3
*float8B

	full_text


float %148
2add8B)
'
	full_text

%150 = add i64 %4, 104
$i648B

	full_text


i64 %4
^getelementptr8BK
I
	full_text<
:
8%151 = getelementptr inbounds float, float* %1, i64 %150
&i648B

	full_text


i64 %150
Nstore8BC
A
	full_text4
2
0store float %149, float* %151, align 4, !tbaa !8
*float8B

	full_text


float %149
,float*8B

	full_text

float* %151
ƒcall8By
w
	full_textj
h
f%152 = tail call float @llvm.fmuladd.f32(float %9, float 0x40E7CEE540000000, float 0x40022C50A0000000)
(float8B

	full_text


float %9
vcall8Bl
j
	full_text]
[
Y%153 = tail call float @llvm.fmuladd.f32(float %10, float 0x400EDC1420000000, float %152)
)float8B

	full_text

	float %10
*float8B

	full_text


float %152
†call8B|
z
	full_textm
k
i%154 = tail call float @llvm.fmuladd.f32(float %7, float 0xBCE542C280000000, float 0x3DC7FB8EC0000000) #4
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%155 = tail call float @llvm.fmuladd.f32(float %154, float %7, float 0xBE98C5B3E0000000) #4
*float8B

	full_text


float %154
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%156 = tail call float @llvm.fmuladd.f32(float %155, float %7, float 0x3F6214CD80000000) #4
*float8B

	full_text


float %155
(float8B

	full_text


float %7
7fmul8B-
+
	full_text

%157 = fmul float %7, %156
(float8B

	full_text


float %7
*float8B

	full_text


float %156
9fadd8B/
-
	full_text 

%158 = fadd float %153, %157
*float8B

	full_text


float %153
*float8B

	full_text


float %157
Kcall8BA
?
	full_text2
0
.%159 = tail call float @_Z3expf(float %158) #3
*float8B

	full_text


float %158
2add8B)
'
	full_text

%160 = add i64 %4, 112
$i648B

	full_text


i64 %4
^getelementptr8BK
I
	full_text<
:
8%161 = getelementptr inbounds float, float* %1, i64 %160
&i648B

	full_text


i64 %160
Nstore8BC
A
	full_text4
2
0store float %159, float* %161, align 4, !tbaa !8
*float8B

	full_text


float %159
,float*8B

	full_text

float* %161
ƒcall8By
w
	full_textj
h
f%162 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0AF57D620000000, float 0x402398C0A0000000)
(float8B

	full_text


float %9
vcall8Bl
j
	full_text]
[
Y%163 = tail call float @llvm.fmuladd.f32(float %10, float 0x40062D69C0000000, float %162)
)float8B

	full_text

	float %10
*float8B

	full_text


float %162
†call8B|
z
	full_textm
k
i%164 = tail call float @llvm.fmuladd.f32(float %7, float 0xBCE806EFC0000000, float 0x3DCAFDC320000000) #4
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%165 = tail call float @llvm.fmuladd.f32(float %164, float %7, float 0xBE9BC9C5A0000000) #4
*float8B

	full_text


float %164
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%166 = tail call float @llvm.fmuladd.f32(float %165, float %7, float 0x3F644DBE80000000) #4
*float8B

	full_text


float %165
(float8B

	full_text


float %7
7fmul8B-
+
	full_text

%167 = fmul float %7, %166
(float8B

	full_text


float %7
*float8B

	full_text


float %166
9fadd8B/
-
	full_text 

%168 = fadd float %163, %167
*float8B

	full_text


float %163
*float8B

	full_text


float %167
Kcall8BA
?
	full_text2
0
.%169 = tail call float @_Z3expf(float %168) #3
*float8B

	full_text


float %168
2add8B)
'
	full_text

%170 = add i64 %4, 120
$i648B

	full_text


i64 %4
^getelementptr8BK
I
	full_text<
:
8%171 = getelementptr inbounds float, float* %1, i64 %170
&i648B

	full_text


i64 %170
Nstore8BC
A
	full_text4
2
0store float %169, float* %171, align 4, !tbaa !8
*float8B

	full_text


float %169
,float*8B

	full_text

float* %171
ƒcall8By
w
	full_textj
h
f%172 = tail call float @llvm.fmuladd.f32(float %9, float 0x40CB55EA80000000, float 0x402B5009A0000000)
(float8B

	full_text


float %9
vcall8Bl
j
	full_text]
[
Y%173 = tail call float @llvm.fmuladd.f32(float %10, float 0x3FFC2BC960000000, float %172)
)float8B

	full_text

	float %10
*float8B

	full_text


float %172
†call8B|
z
	full_textm
k
i%174 = tail call float @llvm.fmuladd.f32(float %7, float 0xBCF3E714C0000000, float 0x3DD70DA9C0000000) #4
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%175 = tail call float @llvm.fmuladd.f32(float %174, float %7, float 0xBEA8BB9FC0000000) #4
*float8B

	full_text


float %174
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%176 = tail call float @llvm.fmuladd.f32(float %175, float %7, float 0x3F72D77340000000) #4
*float8B

	full_text


float %175
(float8B

	full_text


float %7
7fmul8B-
+
	full_text

%177 = fmul float %7, %176
(float8B

	full_text


float %7
*float8B

	full_text


float %176
9fadd8B/
-
	full_text 

%178 = fadd float %173, %177
*float8B

	full_text


float %173
*float8B

	full_text


float %177
Kcall8BA
?
	full_text2
0
.%179 = tail call float @_Z3expf(float %178) #3
*float8B

	full_text


float %178
2add8B)
'
	full_text

%180 = add i64 %4, 128
$i648B

	full_text


i64 %4
^getelementptr8BK
I
	full_text<
:
8%181 = getelementptr inbounds float, float* %1, i64 %180
&i648B

	full_text


i64 %180
Nstore8BC
A
	full_text4
2
0store float %179, float* %181, align 4, !tbaa !8
*float8B

	full_text


float %179
,float*8B

	full_text

float* %181
ƒcall8By
w
	full_textj
h
f%182 = tail call float @llvm.fmuladd.f32(float %9, float 0xC05FF54800000000, float 0x40076FC500000000)
(float8B

	full_text


float %9
vcall8Bl
j
	full_text]
[
Y%183 = tail call float @llvm.fmuladd.f32(float %10, float 0x400E2A98A0000000, float %182)
)float8B

	full_text

	float %10
*float8B

	full_text


float %182
†call8B|
z
	full_textm
k
i%184 = tail call float @llvm.fmuladd.f32(float %7, float 0xBCD3075C60000000, float 0x3DC21213E0000000) #4
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%185 = tail call float @llvm.fmuladd.f32(float %184, float %7, float 0xBE9DB60E20000000) #4
*float8B

	full_text


float %184
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%186 = tail call float @llvm.fmuladd.f32(float %185, float %7, float 0x3F701EEE80000000) #4
*float8B

	full_text


float %185
(float8B

	full_text


float %7
7fmul8B-
+
	full_text

%187 = fmul float %7, %186
(float8B

	full_text


float %7
*float8B

	full_text


float %186
9fadd8B/
-
	full_text 

%188 = fadd float %183, %187
*float8B

	full_text


float %183
*float8B

	full_text


float %187
Kcall8BA
?
	full_text2
0
.%189 = tail call float @_Z3expf(float %188) #3
*float8B

	full_text


float %188
2add8B)
'
	full_text

%190 = add i64 %4, 136
$i648B

	full_text


i64 %4
^getelementptr8BK
I
	full_text<
:
8%191 = getelementptr inbounds float, float* %1, i64 %190
&i648B

	full_text


i64 %190
Nstore8BC
A
	full_text4
2
0store float %189, float* %191, align 4, !tbaa !8
*float8B

	full_text


float %189
,float*8B

	full_text

float* %191
~call8Bt
r
	full_texte
c
a%192 = tail call float @llvm.fmuladd.f32(float %9, float -2.593600e+04, float 0xBFF3AF3B60000000)
(float8B

	full_text


float %9
vcall8Bl
j
	full_text]
[
Y%193 = tail call float @llvm.fmuladd.f32(float %10, float 0x4010971C80000000, float %192)
)float8B

	full_text

	float %10
*float8B

	full_text


float %192
†call8B|
z
	full_textm
k
i%194 = tail call float @llvm.fmuladd.f32(float %7, float 0xBCE044C220000000, float 0x3DC569DE40000000) #4
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%195 = tail call float @llvm.fmuladd.f32(float %194, float %7, float 0xBE9A8A7DA0000000) #4
*float8B

	full_text


float %194
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%196 = tail call float @llvm.fmuladd.f32(float %195, float %7, float 0x3F686B42C0000000) #4
*float8B

	full_text


float %195
(float8B

	full_text


float %7
7fmul8B-
+
	full_text

%197 = fmul float %7, %196
(float8B

	full_text


float %7
*float8B

	full_text


float %196
9fadd8B/
-
	full_text 

%198 = fadd float %193, %197
*float8B

	full_text


float %193
*float8B

	full_text


float %197
Kcall8BA
?
	full_text2
0
.%199 = tail call float @_Z3expf(float %198) #3
*float8B

	full_text


float %198
2add8B)
'
	full_text

%200 = add i64 %4, 144
$i648B

	full_text


i64 %4
^getelementptr8BK
I
	full_text<
:
8%201 = getelementptr inbounds float, float* %1, i64 %200
&i648B

	full_text


i64 %200
Nstore8BC
A
	full_text4
2
0store float %199, float* %201, align 4, !tbaa !8
*float8B

	full_text


float %199
,float*8B

	full_text

float* %201
ƒcall8By
w
	full_textj
h
f%202 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0E7979600000000, float 0x3FE47CD260000000)
(float8B

	full_text


float %9
vcall8Bl
j
	full_text]
[
Y%203 = tail call float @llvm.fmuladd.f32(float %10, float 0x40111CB500000000, float %202)
)float8B

	full_text

	float %10
*float8B

	full_text


float %202
†call8B|
z
	full_textm
k
i%204 = tail call float @llvm.fmuladd.f32(float %7, float 0xBCCAD12160000000, float 0x3DB7549E80000000) #4
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%205 = tail call float @llvm.fmuladd.f32(float %204, float %7, float 0xBE923B7CA0000000) #4
*float8B

	full_text


float %204
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%206 = tail call float @llvm.fmuladd.f32(float %205, float %7, float 0x3F637B5240000000) #4
*float8B

	full_text


float %205
(float8B

	full_text


float %7
7fmul8B-
+
	full_text

%207 = fmul float %7, %206
(float8B

	full_text


float %7
*float8B

	full_text


float %206
9fadd8B/
-
	full_text 

%208 = fadd float %203, %207
*float8B

	full_text


float %203
*float8B

	full_text


float %207
Kcall8BA
?
	full_text2
0
.%209 = tail call float @_Z3expf(float %208) #3
*float8B

	full_text


float %208
2add8B)
'
	full_text

%210 = add i64 %4, 152
$i648B

	full_text


i64 %4
^getelementptr8BK
I
	full_text<
:
8%211 = getelementptr inbounds float, float* %1, i64 %210
&i648B

	full_text


i64 %210
Nstore8BC
A
	full_text4
2
0store float %209, float* %211, align 4, !tbaa !8
*float8B

	full_text


float %209
,float*8B

	full_text

float* %211
ƒcall8By
w
	full_textj
h
f%212 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0E0E69C00000000, float 0x401F263840000000)
(float8B

	full_text


float %9
vcall8Bl
j
	full_text]
[
Y%213 = tail call float @llvm.fmuladd.f32(float %10, float 0x4008224040000000, float %212)
)float8B

	full_text

	float %10
*float8B

	full_text


float %212
†call8B|
z
	full_textm
k
i%214 = tail call float @llvm.fmuladd.f32(float %7, float 0xBCF36C9740000000, float 0x3DD74F7660000000) #4
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%215 = tail call float @llvm.fmuladd.f32(float %214, float %7, float 0xBEAA2D5400000000) #4
*float8B

	full_text


float %214
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%216 = tail call float @llvm.fmuladd.f32(float %215, float %7, float 0x3F752803E0000000) #4
*float8B

	full_text


float %215
(float8B

	full_text


float %7
7fmul8B-
+
	full_text

%217 = fmul float %7, %216
(float8B

	full_text


float %7
*float8B

	full_text


float %216
9fadd8B/
-
	full_text 

%218 = fadd float %213, %217
*float8B

	full_text


float %213
*float8B

	full_text


float %217
Kcall8BA
?
	full_text2
0
.%219 = tail call float @_Z3expf(float %218) #3
*float8B

	full_text


float %218
2add8B)
'
	full_text

%220 = add i64 %4, 160
$i648B

	full_text


i64 %4
^getelementptr8BK
I
	full_text<
:
8%221 = getelementptr inbounds float, float* %1, i64 %220
&i648B

	full_text


i64 %220
Nstore8BC
A
	full_text4
2
0store float %219, float* %221, align 4, !tbaa !8
*float8B

	full_text


float %219
,float*8B

	full_text

float* %221
ƒcall8By
w
	full_textj
h
f%222 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0B34BE2E0000000, float 0x40249C5960000000)
(float8B

	full_text


float %9
vcall8Bl
j
	full_text]
[
Y%223 = tail call float @llvm.fmuladd.f32(float %10, float 0x400049F4A0000000, float %222)
)float8B

	full_text

	float %10
*float8B

	full_text


float %222
†call8B|
z
	full_textm
k
i%224 = tail call float @llvm.fmuladd.f32(float %7, float 0xBCFC4E7600000000, float 0x3DE0DC9F20000000) #4
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%225 = tail call float @llvm.fmuladd.f32(float %224, float %7, float 0xBEB2C3C340000000) #4
*float8B

	full_text


float %224
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%226 = tail call float @llvm.fmuladd.f32(float %225, float %7, float 0x3F7DFE6A60000000) #4
*float8B

	full_text


float %225
(float8B

	full_text


float %7
7fmul8B-
+
	full_text

%227 = fmul float %7, %226
(float8B

	full_text


float %7
*float8B

	full_text


float %226
9fadd8B/
-
	full_text 

%228 = fadd float %223, %227
*float8B

	full_text


float %223
*float8B

	full_text


float %227
Kcall8BA
?
	full_text2
0
.%229 = tail call float @_Z3expf(float %228) #3
*float8B

	full_text


float %228
2add8B)
'
	full_text

%230 = add i64 %4, 168
$i648B

	full_text


i64 %4
^getelementptr8BK
I
	full_text<
:
8%231 = getelementptr inbounds float, float* %1, i64 %230
&i648B

	full_text


i64 %230
Nstore8BC
A
	full_text4
2
0store float %229, float* %231, align 4, !tbaa !8
*float8B

	full_text


float %229
,float*8B

	full_text

float* %231
ƒcall8By
w
	full_textj
h
f%232 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0C91CC280000000, float 0x402AECC440000000)
(float8B

	full_text


float %9
vcall8Bl
j
	full_text]
[
Y%233 = tail call float @llvm.fmuladd.f32(float %10, float 0x3FFF4645C0000000, float %232)
)float8B

	full_text

	float %10
*float8B

	full_text


float %232
†call8B|
z
	full_textm
k
i%234 = tail call float @llvm.fmuladd.f32(float %7, float 0xBD00D92000000000, float 0x3DE4116FE0000000) #4
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%235 = tail call float @llvm.fmuladd.f32(float %234, float %7, float 0xBEB651C940000000) #4
*float8B

	full_text


float %234
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%236 = tail call float @llvm.fmuladd.f32(float %235, float %7, float 0x3F81D09720000000) #4
*float8B

	full_text


float %235
(float8B

	full_text


float %7
7fmul8B-
+
	full_text

%237 = fmul float %7, %236
(float8B

	full_text


float %7
*float8B

	full_text


float %236
9fadd8B/
-
	full_text 

%238 = fadd float %233, %237
*float8B

	full_text


float %233
*float8B

	full_text


float %237
Kcall8BA
?
	full_text2
0
.%239 = tail call float @_Z3expf(float %238) #3
*float8B

	full_text


float %238
2add8B)
'
	full_text

%240 = add i64 %4, 176
$i648B

	full_text


i64 %4
^getelementptr8BK
I
	full_text<
:
8%241 = getelementptr inbounds float, float* %1, i64 %240
&i648B

	full_text


i64 %240
Nstore8BC
A
	full_text4
2
0store float %239, float* %241, align 4, !tbaa !8
*float8B

	full_text


float %239
,float*8B

	full_text

float* %241
ƒcall8By
w
	full_textj
h
f%242 = tail call float @llvm.fmuladd.f32(float %9, float 0x40C6513260000000, float 0x402E3B3160000000)
(float8B

	full_text


float %9
vcall8Bl
j
	full_text]
[
Y%243 = tail call float @llvm.fmuladd.f32(float %10, float 0x3FF1266D40000000, float %242)
)float8B

	full_text

	float %10
*float8B

	full_text


float %242
†call8B|
z
	full_textm
k
i%244 = tail call float @llvm.fmuladd.f32(float %7, float 0xBD056475E0000000, float 0x3DE95BDE60000000) #4
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%245 = tail call float @llvm.fmuladd.f32(float %244, float %7, float 0xBEBC089BE0000000) #4
*float8B

	full_text


float %244
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%246 = tail call float @llvm.fmuladd.f32(float %245, float %7, float 0x3F8634A9C0000000) #4
*float8B

	full_text


float %245
(float8B

	full_text


float %7
7fmul8B-
+
	full_text

%247 = fmul float %7, %246
(float8B

	full_text


float %7
*float8B

	full_text


float %246
9fadd8B/
-
	full_text 

%248 = fadd float %243, %247
*float8B

	full_text


float %243
*float8B

	full_text


float %247
Kcall8BA
?
	full_text2
0
.%249 = tail call float @_Z3expf(float %248) #3
*float8B

	full_text


float %248
2add8B)
'
	full_text

%250 = add i64 %4, 184
$i648B

	full_text


i64 %4
^getelementptr8BK
I
	full_text<
:
8%251 = getelementptr inbounds float, float* %1, i64 %250
&i648B

	full_text


i64 %250
Nstore8BC
A
	full_text4
2
0store float %249, float* %251, align 4, !tbaa !8
*float8B

	full_text


float %249
,float*8B

	full_text

float* %251
ƒcall8By
w
	full_textj
h
f%252 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0D2DFCDC0000000, float 0xC00F712BE0000000)
(float8B

	full_text


float %9
vcall8Bl
j
	full_text]
[
Y%253 = tail call float @llvm.fmuladd.f32(float %10, float 0x4016834860000000, float %252)
)float8B

	full_text

	float %10
*float8B

	full_text


float %252
†call8B|
z
	full_textm
k
i%254 = tail call float @llvm.fmuladd.f32(float %7, float 0xBCD17B2440000000, float 0x3DBA3A9900000000) #4
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%255 = tail call float @llvm.fmuladd.f32(float %254, float %7, float 0xBE91D28EA0000000) #4
*float8B

	full_text


float %254
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%256 = tail call float @llvm.fmuladd.f32(float %255, float %7, float 0x3F60BBCA20000000) #4
*float8B

	full_text


float %255
(float8B

	full_text


float %7
7fmul8B-
+
	full_text

%257 = fmul float %7, %256
(float8B

	full_text


float %7
*float8B

	full_text


float %256
9fadd8B/
-
	full_text 

%258 = fadd float %253, %257
*float8B

	full_text


float %253
*float8B

	full_text


float %257
Kcall8BA
?
	full_text2
0
.%259 = tail call float @_Z3expf(float %258) #3
*float8B

	full_text


float %258
2add8B)
'
	full_text

%260 = add i64 %4, 192
$i648B

	full_text


i64 %4
^getelementptr8BK
I
	full_text<
:
8%261 = getelementptr inbounds float, float* %1, i64 %260
&i648B

	full_text


i64 %260
Nstore8BC
A
	full_text4
2
0store float %259, float* %261, align 4, !tbaa !8
*float8B

	full_text


float %259
,float*8B

	full_text

float* %261
ƒcall8By
w
	full_textj
h
f%262 = tail call float @llvm.fmuladd.f32(float %9, float 0x40BD7F0DA0000000, float 0x3FE43B5E80000000)
(float8B

	full_text


float %9
vcall8Bl
j
	full_text]
[
Y%263 = tail call float @llvm.fmuladd.f32(float %10, float 0x40120B9180000000, float %262)
)float8B

	full_text

	float %10
*float8B

	full_text


float %262
†call8B|
z
	full_textm
k
i%264 = tail call float @llvm.fmuladd.f32(float %7, float 0xBCF1E5EE20000000, float 0x3DD5268EC0000000) #4
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%265 = tail call float @llvm.fmuladd.f32(float %264, float %7, float 0xBEA75123E0000000) #4
*float8B

	full_text


float %264
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%266 = tail call float @llvm.fmuladd.f32(float %265, float %7, float 0x3F72707A60000000) #4
*float8B

	full_text


float %265
(float8B

	full_text


float %7
7fmul8B-
+
	full_text

%267 = fmul float %7, %266
(float8B

	full_text


float %7
*float8B

	full_text


float %266
9fadd8B/
-
	full_text 

%268 = fadd float %263, %267
*float8B

	full_text


float %263
*float8B

	full_text


float %267
Kcall8BA
?
	full_text2
0
.%269 = tail call float @_Z3expf(float %268) #3
*float8B

	full_text


float %268
2add8B)
'
	full_text

%270 = add i64 %4, 200
$i648B

	full_text


i64 %4
^getelementptr8BK
I
	full_text<
:
8%271 = getelementptr inbounds float, float* %1, i64 %270
&i648B

	full_text


i64 %270
Nstore8BC
A
	full_text4
2
0store float %269, float* %271, align 4, !tbaa !8
*float8B

	full_text


float %269
,float*8B

	full_text

float* %271
ƒcall8By
w
	full_textj
h
f%272 = tail call float @llvm.fmuladd.f32(float %9, float 0xC07EA52600000000, float 0xC01420DBA0000000)
(float8B

	full_text


float %9
vcall8Bl
j
	full_text]
[
Y%273 = tail call float @llvm.fmuladd.f32(float %10, float 0x4017E71600000000, float %272)
)float8B

	full_text

	float %10
*float8B

	full_text


float %272
†call8B|
z
	full_textm
k
i%274 = tail call float @llvm.fmuladd.f32(float %7, float 0xBCD3998DC0000000, float 0x3DC2A5B400000000) #4
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%275 = tail call float @llvm.fmuladd.f32(float %274, float %7, float 0xBE9EAFDA00000000) #4
*float8B

	full_text


float %274
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%276 = tail call float @llvm.fmuladd.f32(float %275, float %7, float 0x3F70A6C580000000) #4
*float8B

	full_text


float %275
(float8B

	full_text


float %7
7fmul8B-
+
	full_text

%277 = fmul float %7, %276
(float8B

	full_text


float %7
*float8B

	full_text


float %276
9fadd8B/
-
	full_text 

%278 = fadd float %273, %277
*float8B

	full_text


float %273
*float8B

	full_text


float %277
Kcall8BA
?
	full_text2
0
.%279 = tail call float @_Z3expf(float %278) #3
*float8B

	full_text


float %278
2add8B)
'
	full_text

%280 = add i64 %4, 208
$i648B

	full_text


i64 %4
^getelementptr8BK
I
	full_text<
:
8%281 = getelementptr inbounds float, float* %1, i64 %280
&i648B

	full_text


i64 %280
Nstore8BC
A
	full_text4
2
0store float %279, float* %281, align 4, !tbaa !8
*float8B

	full_text


float %279
,float*8B

	full_text

float* %281
ƒcall8By
w
	full_textj
h
f%282 = tail call float @llvm.fmuladd.f32(float %9, float 0x40D61047C0000000, float 0xC00BD8A960000000)
(float8B

	full_text


float %9
vcall8Bl
j
	full_text]
[
Y%283 = tail call float @llvm.fmuladd.f32(float %10, float 0x40159DCF40000000, float %282)
)float8B

	full_text

	float %10
*float8B

	full_text


float %282
†call8B|
z
	full_textm
k
i%284 = tail call float @llvm.fmuladd.f32(float %7, float 0xBCE2753BA0000000, float 0x3DCF52CE40000000) #4
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%285 = tail call float @llvm.fmuladd.f32(float %284, float %7, float 0xBEA7A2A060000000) #4
*float8B

	full_text


float %284
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%286 = tail call float @llvm.fmuladd.f32(float %285, float %7, float 0x3F78024260000000) #4
*float8B

	full_text


float %285
(float8B

	full_text


float %7
7fmul8B-
+
	full_text

%287 = fmul float %7, %286
(float8B

	full_text


float %7
*float8B

	full_text


float %286
9fadd8B/
-
	full_text 

%288 = fadd float %283, %287
*float8B

	full_text


float %283
*float8B

	full_text


float %287
Kcall8BA
?
	full_text2
0
.%289 = tail call float @_Z3expf(float %288) #3
*float8B

	full_text


float %288
2add8B)
'
	full_text

%290 = add i64 %4, 216
$i648B

	full_text


i64 %4
^getelementptr8BK
I
	full_text<
:
8%291 = getelementptr inbounds float, float* %1, i64 %290
&i648B

	full_text


i64 %290
Nstore8BC
A
	full_text4
2
0store float %289, float* %291, align 4, !tbaa !8
*float8B

	full_text


float %289
,float*8B

	full_text

float* %291
ƒcall8By
w
	full_textj
h
f%292 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0D1129CC0000000, float 0xC0267C7100000000)
(float8B

	full_text


float %9
vcall8Bl
j
	full_text]
[
Y%293 = tail call float @llvm.fmuladd.f32(float %10, float 0x401A00CE80000000, float %292)
)float8B

	full_text

	float %10
*float8B

	full_text


float %292
†call8B|
z
	full_textm
k
i%294 = tail call float @llvm.fmuladd.f32(float %7, float 0xBCF4591FA0000000, float 0x3DD961D9C0000000) #4
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%295 = tail call float @llvm.fmuladd.f32(float %294, float %7, float 0xBEAFC12CE0000000) #4
*float8B

	full_text


float %294
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%296 = tail call float @llvm.fmuladd.f32(float %295, float %7, float 0x3F7D5648E0000000) #4
*float8B

	full_text


float %295
(float8B

	full_text


float %7
7fmul8B-
+
	full_text

%297 = fmul float %7, %296
(float8B

	full_text


float %7
*float8B

	full_text


float %296
9fadd8B/
-
	full_text 

%298 = fadd float %293, %297
*float8B

	full_text


float %293
*float8B

	full_text


float %297
Kcall8BA
?
	full_text2
0
.%299 = tail call float @_Z3expf(float %298) #3
*float8B

	full_text


float %298
2add8B)
'
	full_text

%300 = add i64 %4, 224
$i648B

	full_text


i64 %4
^getelementptr8BK
I
	full_text<
:
8%301 = getelementptr inbounds float, float* %1, i64 %300
&i648B

	full_text


i64 %300
Nstore8BC
A
	full_text4
2
0store float %299, float* %301, align 4, !tbaa !8
*float8B

	full_text


float %299
,float*8B

	full_text

float* %301
ƒcall8By
w
	full_textj
h
f%302 = tail call float @llvm.fmuladd.f32(float %9, float 0x408CDC9000000000, float 0xC02AA06F60000000)
(float8B

	full_text


float %9
vcall8Bl
j
	full_text]
[
Y%303 = tail call float @llvm.fmuladd.f32(float %10, float 0x401AEDD4C0000000, float %302)
)float8B

	full_text

	float %10
*float8B

	full_text


float %302
†call8B|
z
	full_textm
k
i%304 = tail call float @llvm.fmuladd.f32(float %7, float 0xBCE0F62340000000, float 0x3DD0852CA0000000) #4
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%305 = tail call float @llvm.fmuladd.f32(float %304, float %7, float 0xBEABAE8D20000000) #4
*float8B

	full_text


float %304
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%306 = tail call float @llvm.fmuladd.f32(float %305, float %7, float 0x3F7E884380000000) #4
*float8B

	full_text


float %305
(float8B

	full_text


float %7
7fmul8B-
+
	full_text

%307 = fmul float %7, %306
(float8B

	full_text


float %7
*float8B

	full_text


float %306
9fadd8B/
-
	full_text 

%308 = fadd float %303, %307
*float8B

	full_text


float %303
*float8B

	full_text


float %307
Kcall8BA
?
	full_text2
0
.%309 = tail call float @_Z3expf(float %308) #3
*float8B

	full_text


float %308
2add8B)
'
	full_text

%310 = add i64 %4, 232
$i648B

	full_text


i64 %4
^getelementptr8BK
I
	full_text<
:
8%311 = getelementptr inbounds float, float* %1, i64 %310
&i648B

	full_text


i64 %310
Nstore8BC
A
	full_text4
2
0store float %309, float* %311, align 4, !tbaa !8
*float8B

	full_text


float %309
,float*8B

	full_text

float* %311
ƒcall8By
w
	full_textj
h
f%312 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0BF283940000000, float 0xC02F07D500000000)
(float8B

	full_text


float %9
vcall8Bl
j
	full_text]
[
Y%313 = tail call float @llvm.fmuladd.f32(float %10, float 0x401ED6C820000000, float %312)
)float8B

	full_text

	float %10
*float8B

	full_text


float %312
†call8B|
z
	full_textm
k
i%314 = tail call float @llvm.fmuladd.f32(float %7, float 0xBCE1809100000000, float 0x3DD16223E0000000) #4
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%315 = tail call float @llvm.fmuladd.f32(float %314, float %7, float 0xBEAD7BB920000000) #4
*float8B

	full_text


float %314
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%316 = tail call float @llvm.fmuladd.f32(float %315, float %7, float 0x3F806A8EC0000000) #4
*float8B

	full_text


float %315
(float8B

	full_text


float %7
(br8B 

	full_text

br label %622
ƒcall8By
w
	full_textj
h
f%318 = tail call float @llvm.fmuladd.f32(float %9, float 0x408CAF7B40000000, float 0x3FE5DB3840000000)
(float8B

	full_text


float %9
vcall8Bl
j
	full_text]
[
Y%319 = tail call float @llvm.fmuladd.f32(float %10, float 0x4002C130A0000000, float %318)
)float8B

	full_text

	float %10
*float8B

	full_text


float %318
†call8B|
z
	full_textm
k
i%320 = tail call float @llvm.fmuladd.f32(float %7, float 0xBD59F3D0E0000000, float 0x3E1CDBB200000000) #4
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%321 = tail call float @llvm.fmuladd.f32(float %320, float %7, float 0xBECB3B8080000000) #4
*float8B

	full_text


float %320
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%322 = tail call float @llvm.fmuladd.f32(float %321, float %7, float 0x3F70581760000000) #4
*float8B

	full_text


float %321
(float8B

	full_text


float %7
7fmul8B-
+
	full_text

%323 = fmul float %7, %322
(float8B

	full_text


float %7
*float8B

	full_text


float %322
9fadd8B/
-
	full_text 

%324 = fadd float %319, %323
*float8B

	full_text


float %319
*float8B

	full_text


float %323
Kcall8BA
?
	full_text2
0
.%325 = tail call float @_Z3expf(float %324) #3
*float8B

	full_text


float %324
\getelementptr8BI
G
	full_text:
8
6%326 = getelementptr inbounds float, float* %1, i64 %4
$i648B

	full_text


i64 %4
Nstore8BC
A
	full_text4
2
0store float %325, float* %326, align 4, !tbaa !8
*float8B

	full_text


float %325
,float*8B

	full_text

float* %326
ƒcall8By
w
	full_textj
h
f%327 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0D8E06A40000000, float 0xBFDC9673A0000000)
(float8B

	full_text


float %9
pcall8Bf
d
	full_textW
U
S%328 = tail call float @llvm.fmuladd.f32(float %10, float 2.500000e+00, float %327)
)float8B

	full_text

	float %10
*float8B

	full_text


float %327
†call8B|
z
	full_textm
k
i%329 = tail call float @llvm.fmuladd.f32(float %7, float 0xBB4C09FB40000000, float 0x3C0C4B8820000000) #4
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%330 = tail call float @llvm.fmuladd.f32(float %329, float %7, float 0xBCB7F85EA0000000) #4
*float8B

	full_text


float %329
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%331 = tail call float @llvm.fmuladd.f32(float %330, float %7, float 0x3D58D112C0000000) #4
*float8B

	full_text


float %330
(float8B

	full_text


float %7
7fmul8B-
+
	full_text

%332 = fmul float %7, %331
(float8B

	full_text


float %7
*float8B

	full_text


float %331
9fadd8B/
-
	full_text 

%333 = fadd float %328, %332
*float8B

	full_text


float %328
*float8B

	full_text


float %332
Kcall8BA
?
	full_text2
0
.%334 = tail call float @_Z3expf(float %333) #3
*float8B

	full_text


float %333
0add8B'
%
	full_text

%335 = add i64 %4, 8
$i648B

	full_text


i64 %4
^getelementptr8BK
I
	full_text<
:
8%336 = getelementptr inbounds float, float* %1, i64 %335
&i648B

	full_text


i64 %335
Nstore8BC
A
	full_text4
2
0store float %334, float* %336, align 4, !tbaa !8
*float8B

	full_text


float %334
,float*8B

	full_text

float* %336
ƒcall8By
w
	full_textj
h
f%337 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0DC7090A0000000, float 0x40006A5C20000000)
(float8B

	full_text


float %9
vcall8Bl
j
	full_text]
[
Y%338 = tail call float @llvm.fmuladd.f32(float %10, float 0x4009589C60000000, float %337)
)float8B

	full_text

	float %10
*float8B

	full_text


float %337
†call8B|
z
	full_textm
k
i%339 = tail call float @llvm.fmuladd.f32(float %7, float 0x3D3DBBA8A0000000, float 0xBE018BEB80000000) #4
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%340 = tail call float @llvm.fmuladd.f32(float %339, float %7, float 0x3EB2934A60000000) #4
*float8B

	full_text


float %339
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%341 = tail call float @llvm.fmuladd.f32(float %340, float %7, float 0xBF5ADD3AE0000000) #4
*float8B

	full_text


float %340
(float8B

	full_text


float %7
7fmul8B-
+
	full_text

%342 = fmul float %7, %341
(float8B

	full_text


float %7
*float8B

	full_text


float %341
9fadd8B/
-
	full_text 

%343 = fadd float %338, %342
*float8B

	full_text


float %338
*float8B

	full_text


float %342
Kcall8BA
?
	full_text2
0
.%344 = tail call float @_Z3expf(float %343) #3
*float8B

	full_text


float %343
1add8B(
&
	full_text

%345 = add i64 %4, 16
$i648B

	full_text


i64 %4
^getelementptr8BK
I
	full_text<
:
8%346 = getelementptr inbounds float, float* %1, i64 %345
&i648B

	full_text


i64 %345
Nstore8BC
A
	full_text4
2
0store float %344, float* %346, align 4, !tbaa !8
*float8B

	full_text


float %344
,float*8B

	full_text

float* %346
ƒcall8By
w
	full_textj
h
f%347 = tail call float @llvm.fmuladd.f32(float %9, float 0x40909FC640000000, float 0x400D42EB80000000)
(float8B

	full_text


float %9
vcall8Bl
j
	full_text]
[
Y%348 = tail call float @llvm.fmuladd.f32(float %10, float 0x400E427880000000, float %347)
)float8B

	full_text

	float %10
*float8B

	full_text


float %347
†call8B|
z
	full_textm
k
i%349 = tail call float @llvm.fmuladd.f32(float %7, float 0x3D46D361A0000000, float 0xBE0BB876E0000000) #4
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%350 = tail call float @llvm.fmuladd.f32(float %349, float %7, float 0x3EBB88F920000000) #4
*float8B

	full_text


float %349
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%351 = tail call float @llvm.fmuladd.f32(float %350, float %7, float 0xBF588C9B60000000) #4
*float8B

	full_text


float %350
(float8B

	full_text


float %7
7fmul8B-
+
	full_text

%352 = fmul float %7, %351
(float8B

	full_text


float %7
*float8B

	full_text


float %351
9fadd8B/
-
	full_text 

%353 = fadd float %348, %352
*float8B

	full_text


float %348
*float8B

	full_text


float %352
Kcall8BA
?
	full_text2
0
.%354 = tail call float @_Z3expf(float %353) #3
*float8B

	full_text


float %353
1add8B(
&
	full_text

%355 = add i64 %4, 24
$i648B

	full_text


i64 %4
^getelementptr8BK
I
	full_text<
:
8%356 = getelementptr inbounds float, float* %1, i64 %355
&i648B

	full_text


i64 %355
Nstore8BC
A
	full_text4
2
0store float %354, float* %356, align 4, !tbaa !8
*float8B

	full_text


float %354
,float*8B

	full_text

float* %356
ƒcall8By
w
	full_textj
h
f%357 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0AC3E2940000000, float 0xBFBA9ADBE0000000)
(float8B

	full_text


float %9
vcall8Bl
j
	full_text]
[
Y%358 = tail call float @llvm.fmuladd.f32(float %10, float 0x400FEFA5C0000000, float %357)
)float8B

	full_text

	float %10
*float8B

	full_text


float %357
†call8B|
z
	full_textm
k
i%359 = tail call float @llvm.fmuladd.f32(float %7, float 0x3D3332BDC0000000, float 0xBDF639CD40000000) #4
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%360 = tail call float @llvm.fmuladd.f32(float %359, float %7, float 0x3EA9D34C60000000) #4
*float8B

	full_text


float %359
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%361 = tail call float @llvm.fmuladd.f32(float %360, float %7, float 0xBF53ABED80000000) #4
*float8B

	full_text


float %360
(float8B

	full_text


float %7
7fmul8B-
+
	full_text

%362 = fmul float %7, %361
(float8B

	full_text


float %7
*float8B

	full_text


float %361
9fadd8B/
-
	full_text 

%363 = fadd float %358, %362
*float8B

	full_text


float %358
*float8B

	full_text


float %362
Kcall8BA
?
	full_text2
0
.%364 = tail call float @_Z3expf(float %363) #3
*float8B

	full_text


float %363
1add8B(
&
	full_text

%365 = add i64 %4, 32
$i648B

	full_text


i64 %4
^getelementptr8BK
I
	full_text<
:
8%366 = getelementptr inbounds float, float* %1, i64 %365
&i648B

	full_text


i64 %365
Nstore8BC
A
	full_text4
2
0store float %364, float* %366, align 4, !tbaa !8
*float8B

	full_text


float %364
,float*8B

	full_text

float* %366
ƒcall8By
w
	full_textj
h
f%367 = tail call float @llvm.fmuladd.f32(float %9, float 0x40DD956E80000000, float 0xBFEB2B45A0000000)
(float8B

	full_text


float %9
vcall8Bl
j
	full_text]
[
Y%368 = tail call float @llvm.fmuladd.f32(float %10, float 0x4010CB6860000000, float %367)
)float8B

	full_text

	float %10
*float8B

	full_text


float %367
†call8B|
z
	full_textm
k
i%369 = tail call float @llvm.fmuladd.f32(float %7, float 0x3D38F03960000000, float 0xBDFF6D7340000000) #4
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%370 = tail call float @llvm.fmuladd.f32(float %369, float %7, float 0x3EB23B7C60000000) #4
*float8B

	full_text


float %369
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%371 = tail call float @llvm.fmuladd.f32(float %370, float %7, float 0xBF50AEB640000000) #4
*float8B

	full_text


float %370
(float8B

	full_text


float %7
7fmul8B-
+
	full_text

%372 = fmul float %7, %371
(float8B

	full_text


float %7
*float8B

	full_text


float %371
9fadd8B/
-
	full_text 

%373 = fadd float %368, %372
*float8B

	full_text


float %368
*float8B

	full_text


float %372
Kcall8BA
?
	full_text2
0
.%374 = tail call float @_Z3expf(float %373) #3
*float8B

	full_text


float %373
1add8B(
&
	full_text

%375 = add i64 %4, 40
$i648B

	full_text


i64 %4
^getelementptr8BK
I
	full_text<
:
8%376 = getelementptr inbounds float, float* %1, i64 %375
&i648B

	full_text


i64 %375
Nstore8BC
A
	full_text4
2
0store float %374, float* %376, align 4, !tbaa !8
*float8B

	full_text


float %374
,float*8B

	full_text

float* %376
ƒcall8By
w
	full_textj
h
f%377 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0726CEDC0000000, float 0x400DBBB980000000)
(float8B

	full_text


float %9
vcall8Bl
j
	full_text]
[
Y%378 = tail call float @llvm.fmuladd.f32(float %10, float 0x4011350A80000000, float %377)
)float8B

	full_text

	float %10
*float8B

	full_text


float %377
†call8B|
z
	full_textm
k
i%379 = tail call float @llvm.fmuladd.f32(float %7, float 0x3D6058DBA0000000, float 0xBE2160B200000000) #4
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%380 = tail call float @llvm.fmuladd.f32(float %379, float %7, float 0x3ECD94D8C0000000) #4
*float8B

	full_text


float %379
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%381 = tail call float @llvm.fmuladd.f32(float %380, float %7, float 0xBF6373D060000000) #4
*float8B

	full_text


float %380
(float8B

	full_text


float %7
7fmul8B-
+
	full_text

%382 = fmul float %7, %381
(float8B

	full_text


float %7
*float8B

	full_text


float %381
9fadd8B/
-
	full_text 

%383 = fadd float %378, %382
*float8B

	full_text


float %378
*float8B

	full_text


float %382
Kcall8BA
?
	full_text2
0
.%384 = tail call float @_Z3expf(float %383) #3
*float8B

	full_text


float %383
1add8B(
&
	full_text

%385 = add i64 %4, 48
$i648B

	full_text


i64 %4
^getelementptr8BK
I
	full_text<
:
8%386 = getelementptr inbounds float, float* %1, i64 %385
&i648B

	full_text


i64 %385
Nstore8BC
A
	full_text4
2
0store float %384, float* %386, align 4, !tbaa !8
*float8B

	full_text


float %384
,float*8B

	full_text

float* %386
ƒcall8By
w
	full_textj
h
f%387 = tail call float @llvm.fmuladd.f32(float %9, float 0x40D149A540000000, float 0x400B7AFBE0000000)
(float8B

	full_text


float %9
vcall8Bl
j
	full_text]
[
Y%388 = tail call float @llvm.fmuladd.f32(float %10, float 0x40111ABD40000000, float %387)
)float8B

	full_text

	float %10
*float8B

	full_text


float %387
†call8B|
z
	full_textm
k
i%389 = tail call float @llvm.fmuladd.f32(float %7, float 0x3D5E584C60000000, float 0xBE1EE41580000000) #4
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%390 = tail call float @llvm.fmuladd.f32(float %389, float %7, float 0x3EC7652DA0000000) #4
*float8B

	full_text


float %389
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%391 = tail call float @llvm.fmuladd.f32(float %390, float %7, float 0xBF31C98640000000) #4
*float8B

	full_text


float %390
(float8B

	full_text


float %7
7fmul8B-
+
	full_text

%392 = fmul float %7, %391
(float8B

	full_text


float %7
*float8B

	full_text


float %391
9fadd8B/
-
	full_text 

%393 = fadd float %388, %392
*float8B

	full_text


float %388
*float8B

	full_text


float %392
Kcall8BA
?
	full_text2
0
.%394 = tail call float @_Z3expf(float %393) #3
*float8B

	full_text


float %393
1add8B(
&
	full_text

%395 = add i64 %4, 56
$i648B

	full_text


i64 %4
^getelementptr8BK
I
	full_text<
:
8%396 = getelementptr inbounds float, float* %1, i64 %395
&i648B

	full_text


i64 %395
Nstore8BC
A
	full_text4
2
0store float %394, float* %396, align 4, !tbaa !8
*float8B

	full_text


float %394
,float*8B

	full_text

float* %396
ƒcall8By
w
	full_textj
h
f%397 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0F148D4C0000000, float 0x4000AC0E00000000)
(float8B

	full_text


float %9
vcall8Bl
j
	full_text]
[
Y%398 = tail call float @llvm.fmuladd.f32(float %10, float 0x400BEB2500000000, float %397)
)float8B

	full_text

	float %10
*float8B

	full_text


float %397
†call8B|
z
	full_textm
k
i%399 = tail call float @llvm.fmuladd.f32(float %7, float 0xBD33C9F9C0000000, float 0x3DF21BCB80000000) #4
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%400 = tail call float @llvm.fmuladd.f32(float %399, float %7, float 0xBE92E41B40000000) #4
*float8B

	full_text


float %399
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%401 = tail call float @llvm.fmuladd.f32(float %400, float %7, float 0x3F25390F00000000) #4
*float8B

	full_text


float %400
(float8B

	full_text


float %7
7fmul8B-
+
	full_text

%402 = fmul float %7, %401
(float8B

	full_text


float %7
*float8B

	full_text


float %401
9fadd8B/
-
	full_text 

%403 = fadd float %398, %402
*float8B

	full_text


float %398
*float8B

	full_text


float %402
Kcall8BA
?
	full_text2
0
.%404 = tail call float @_Z3expf(float %403) #3
*float8B

	full_text


float %403
1add8B(
&
	full_text

%405 = add i64 %4, 64
$i648B

	full_text


i64 %4
^getelementptr8BK
I
	full_text<
:
8%406 = getelementptr inbounds float, float* %1, i64 %405
&i648B

	full_text


i64 %405
Nstore8BC
A
	full_text4
2
0store float %404, float* %406, align 4, !tbaa !8
*float8B

	full_text


float %404
,float*8B

	full_text

float* %406
ƒcall8By
w
	full_textj
h
f%407 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0E6768140000000, float 0x3FF9002160000000)
(float8B

	full_text


float %9
vcall8Bl
j
	full_text]
[
Y%408 = tail call float @llvm.fmuladd.f32(float %10, float 0x400E19F740000000, float %407)
)float8B

	full_text

	float %10
*float8B

	full_text


float %407
†call8B|
z
	full_textm
k
i%409 = tail call float @llvm.fmuladd.f32(float %7, float 0x3D37BF8FA0000000, float 0xBDF60D7F00000000) #4
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%410 = tail call float @llvm.fmuladd.f32(float %409, float %7, float 0x3E9F42AA40000000) #4
*float8B

	full_text


float %409
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%411 = tail call float @llvm.fmuladd.f32(float %410, float %7, float 0x3F3FBF7D20000000) #4
*float8B

	full_text


float %410
(float8B

	full_text


float %7
7fmul8B-
+
	full_text

%412 = fmul float %7, %411
(float8B

	full_text


float %7
*float8B

	full_text


float %411
9fadd8B/
-
	full_text 

%413 = fadd float %408, %412
*float8B

	full_text


float %408
*float8B

	full_text


float %412
Kcall8BA
?
	full_text2
0
.%414 = tail call float @_Z3expf(float %413) #3
*float8B

	full_text


float %413
1add8B(
&
	full_text

%415 = add i64 %4, 72
$i648B

	full_text


i64 %4
^getelementptr8BK
I
	full_text<
:
8%416 = getelementptr inbounds float, float* %1, i64 %415
&i648B

	full_text


i64 %415
Nstore8BC
A
	full_text4
2
0store float %414, float* %416, align 4, !tbaa !8
*float8B

	full_text


float %414
,float*8B

	full_text

float* %416
ƒcall8By
w
	full_textj
h
f%417 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0E8A81A20000000, float 0xBFE89C9F60000000)
(float8B

	full_text


float %9
vcall8Bl
j
	full_text]
[
Y%418 = tail call float @llvm.fmuladd.f32(float %10, float 0x4010CB5EE0000000, float %417)
)float8B

	full_text

	float %10
*float8B

	full_text


float %417
†call8B|
z
	full_textm
k
i%419 = tail call float @llvm.fmuladd.f32(float %7, float 0x3D3B58ED20000000, float 0xBE03267920000000) #4
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%420 = tail call float @llvm.fmuladd.f32(float %419, float %7, float 0x3EB7056240000000) #4
*float8B

	full_text


float %419
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%421 = tail call float @llvm.fmuladd.f32(float %420, float %7, float 0xBF53632660000000) #4
*float8B

	full_text


float %420
(float8B

	full_text


float %7
7fmul8B-
+
	full_text

%422 = fmul float %7, %421
(float8B

	full_text


float %7
*float8B

	full_text


float %421
9fadd8B/
-
	full_text 

%423 = fadd float %418, %422
*float8B

	full_text


float %418
*float8B

	full_text


float %422
Kcall8BA
?
	full_text2
0
.%424 = tail call float @_Z3expf(float %423) #3
*float8B

	full_text


float %423
1add8B(
&
	full_text

%425 = add i64 %4, 80
$i648B

	full_text


i64 %4
^getelementptr8BK
I
	full_text<
:
8%426 = getelementptr inbounds float, float* %1, i64 %425
&i648B

	full_text


i64 %425
Nstore8BC
A
	full_text4
2
0store float %424, float* %426, align 4, !tbaa !8
*float8B

	full_text


float %424
,float*8B

	full_text

float* %426
ƒcall8By
w
	full_textj
h
f%427 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0D00F3FE0000000, float 0x3FF9AC4BA0000000)
(float8B

	full_text


float %9
vcall8Bl
j
	full_text]
[
Y%428 = tail call float @llvm.fmuladd.f32(float %10, float 0x400D638360000000, float %427)
)float8B

	full_text

	float %10
*float8B

	full_text


float %427
†call8B|
z
	full_textm
k
i%429 = tail call float @llvm.fmuladd.f32(float %7, float 0x3D41E69B20000000, float 0xBE03AC9FC0000000) #4
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%430 = tail call float @llvm.fmuladd.f32(float %429, float %7, float 0x3EB005D9A0000000) #4
*float8B

	full_text


float %429
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%431 = tail call float @llvm.fmuladd.f32(float %430, float %7, float 0x3F50794580000000) #4
*float8B

	full_text


float %430
(float8B

	full_text


float %7
7fmul8B-
+
	full_text

%432 = fmul float %7, %431
(float8B

	full_text


float %7
*float8B

	full_text


float %431
9fadd8B/
-
	full_text 

%433 = fadd float %428, %432
*float8B

	full_text


float %428
*float8B

	full_text


float %432
Kcall8BA
?
	full_text2
0
.%434 = tail call float @_Z3expf(float %433) #3
*float8B

	full_text


float %433
1add8B(
&
	full_text

%435 = add i64 %4, 88
$i648B

	full_text


i64 %4
^getelementptr8BK
I
	full_text<
:
8%436 = getelementptr inbounds float, float* %1, i64 %435
&i648B

	full_text


i64 %435
Nstore8BC
A
	full_text4
2
0store float %434, float* %436, align 4, !tbaa !8
*float8B

	full_text


float %434
,float*8B

	full_text

float* %436
ƒcall8By
w
	full_textj
h
f%437 = tail call float @llvm.fmuladd.f32(float %9, float 0x40C40352E0000000, float 0xC01290B1E0000000)
(float8B

	full_text


float %9
vcall8Bl
j
	full_text]
[
Y%438 = tail call float @llvm.fmuladd.f32(float %10, float 0x4014997920000000, float %437)
)float8B

	full_text

	float %10
*float8B

	full_text


float %437
†call8B|
z
	full_textm
k
i%439 = tail call float @llvm.fmuladd.f32(float %7, float 0x3D6D533A80000000, float 0xBE31598140000000) #4
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%440 = tail call float @llvm.fmuladd.f32(float %439, float %7, float 0x3EE1308EA0000000) #4
*float8B

	full_text


float %439
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%441 = tail call float @llvm.fmuladd.f32(float %440, float %7, float 0xBF7BFF87C0000000) #4
*float8B

	full_text


float %440
(float8B

	full_text


float %7
7fmul8B-
+
	full_text

%442 = fmul float %7, %441
(float8B

	full_text


float %7
*float8B

	full_text


float %441
9fadd8B/
-
	full_text 

%443 = fadd float %438, %442
*float8B

	full_text


float %438
*float8B

	full_text


float %442
Kcall8BA
?
	full_text2
0
.%444 = tail call float @_Z3expf(float %443) #3
*float8B

	full_text


float %443
1add8B(
&
	full_text

%445 = add i64 %4, 96
$i648B

	full_text


i64 %4
^getelementptr8BK
I
	full_text<
:
8%446 = getelementptr inbounds float, float* %1, i64 %445
&i648B

	full_text


i64 %445
Nstore8BC
A
	full_text4
2
0store float %444, float* %446, align 4, !tbaa !8
*float8B

	full_text


float %444
,float*8B

	full_text

float* %446
ƒcall8By
w
	full_textj
h
f%447 = tail call float @llvm.fmuladd.f32(float %9, float 0x40CC040B00000000, float 0x400C1138E0000000)
(float8B

	full_text


float %9
vcall8Bl
j
	full_text]
[
Y%448 = tail call float @llvm.fmuladd.f32(float %10, float 0x400CA2E280000000, float %447)
)float8B

	full_text

	float %10
*float8B

	full_text


float %447
†call8B|
z
	full_textm
k
i%449 = tail call float @llvm.fmuladd.f32(float %7, float 0xBD297510C0000000, float 0x3DD4C6BD20000000) #4
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%450 = tail call float @llvm.fmuladd.f32(float %449, float %7, float 0x3E86BEE9A0000000) #4
*float8B

	full_text


float %449
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%451 = tail call float @llvm.fmuladd.f32(float %450, float %7, float 0xBF34000480000000) #4
*float8B

	full_text


float %450
(float8B

	full_text


float %7
7fmul8B-
+
	full_text

%452 = fmul float %7, %451
(float8B

	full_text


float %7
*float8B

	full_text


float %451
9fadd8B/
-
	full_text 

%453 = fadd float %448, %452
*float8B

	full_text


float %448
*float8B

	full_text


float %452
Kcall8BA
?
	full_text2
0
.%454 = tail call float @_Z3expf(float %453) #3
*float8B

	full_text


float %453
2add8B)
'
	full_text

%455 = add i64 %4, 104
$i648B

	full_text


i64 %4
^getelementptr8BK
I
	full_text<
:
8%456 = getelementptr inbounds float, float* %1, i64 %455
&i648B

	full_text


i64 %455
Nstore8BC
A
	full_text4
2
0store float %454, float* %456, align 4, !tbaa !8
*float8B

	full_text


float %454
,float*8B

	full_text

float* %456
ƒcall8By
w
	full_textj
h
f%457 = tail call float @llvm.fmuladd.f32(float %9, float 0x40E79E7F00000000, float 0x4023CD56C0000000)
(float8B

	full_text


float %9
vcall8Bl
j
	full_text]
[
Y%458 = tail call float @llvm.fmuladd.f32(float %10, float 0x4002DAAC20000000, float %457)
)float8B

	full_text

	float %10
*float8B

	full_text


float %457
†call8B|
z
	full_textm
k
i%459 = tail call float @llvm.fmuladd.f32(float %7, float 0xBD002DDB80000000, float 0x3DEC2A6C00000000) #4
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%460 = tail call float @llvm.fmuladd.f32(float %459, float %7, float 0xBEB3EB3EA0000000) #4
*float8B

	full_text


float %459
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%461 = tail call float @llvm.fmuladd.f32(float %460, float %7, float 0x3F72668420000000) #4
*float8B

	full_text


float %460
(float8B

	full_text


float %7
7fmul8B-
+
	full_text

%462 = fmul float %7, %461
(float8B

	full_text


float %7
*float8B

	full_text


float %461
9fadd8B/
-
	full_text 

%463 = fadd float %458, %462
*float8B

	full_text


float %458
*float8B

	full_text


float %462
Kcall8BA
?
	full_text2
0
.%464 = tail call float @_Z3expf(float %463) #3
*float8B

	full_text


float %463
2add8B)
'
	full_text

%465 = add i64 %4, 112
$i648B

	full_text


i64 %4
^getelementptr8BK
I
	full_text<
:
8%466 = getelementptr inbounds float, float* %1, i64 %465
&i648B

	full_text


i64 %465
Nstore8BC
A
	full_text4
2
0store float %464, float* %466, align 4, !tbaa !8
*float8B

	full_text


float %464
,float*8B

	full_text

float* %466
ƒcall8By
w
	full_textj
h
f%467 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0ADFF2140000000, float 0x400B27ACC0000000)
(float8B

	full_text


float %9
vcall8Bl
j
	full_text]
[
Y%468 = tail call float @llvm.fmuladd.f32(float %10, float 0x4010E27E80000000, float %467)
)float8B

	full_text

	float %10
*float8B

	full_text


float %467
†call8B|
z
	full_textm
k
i%469 = tail call float @llvm.fmuladd.f32(float %7, float 0x3D4E8615E0000000, float 0xBE130FC860000000) #4
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%470 = tail call float @llvm.fmuladd.f32(float %469, float %7, float 0x3EC34408C0000000) #4
*float8B

	full_text


float %469
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%471 = tail call float @llvm.fmuladd.f32(float %470, float %7, float 0xBF5A930120000000) #4
*float8B

	full_text


float %470
(float8B

	full_text


float %7
7fmul8B-
+
	full_text

%472 = fmul float %7, %471
(float8B

	full_text


float %7
*float8B

	full_text


float %471
9fadd8B/
-
	full_text 

%473 = fadd float %468, %472
*float8B

	full_text


float %468
*float8B

	full_text


float %472
Kcall8BA
?
	full_text2
0
.%474 = tail call float @_Z3expf(float %473) #3
*float8B

	full_text


float %473
2add8B)
'
	full_text

%475 = add i64 %4, 120
$i648B

	full_text


i64 %4
^getelementptr8BK
I
	full_text<
:
8%476 = getelementptr inbounds float, float* %1, i64 %475
&i648B

	full_text


i64 %475
Nstore8BC
A
	full_text4
2
0store float %474, float* %476, align 4, !tbaa !8
*float8B

	full_text


float %474
,float*8B

	full_text

float* %476
ƒcall8By
w
	full_textj
h
f%477 = tail call float @llvm.fmuladd.f32(float %9, float 0x40CBF27A80000000, float 0x3FE34A3E40000000)
(float8B

	full_text


float %9
vcall8Bl
j
	full_text]
[
Y%478 = tail call float @llvm.fmuladd.f32(float %10, float 0x40132CC5C0000000, float %477)
)float8B

	full_text

	float %10
*float8B

	full_text


float %477
†call8B|
z
	full_textm
k
i%479 = tail call float @llvm.fmuladd.f32(float %7, float 0x3D672E8340000000, float 0xBE2B2679E0000000) #4
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%480 = tail call float @llvm.fmuladd.f32(float %479, float %7, float 0x3EDA170840000000) #4
*float8B

	full_text


float %479
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%481 = tail call float @llvm.fmuladd.f32(float %480, float %7, float 0xBF744AD200000000) #4
*float8B

	full_text


float %480
(float8B

	full_text


float %7
7fmul8B-
+
	full_text

%482 = fmul float %7, %481
(float8B

	full_text


float %7
*float8B

	full_text


float %481
9fadd8B/
-
	full_text 

%483 = fadd float %478, %482
*float8B

	full_text


float %478
*float8B

	full_text


float %482
Kcall8BA
?
	full_text2
0
.%484 = tail call float @_Z3expf(float %483) #3
*float8B

	full_text


float %483
2add8B)
'
	full_text

%485 = add i64 %4, 128
$i648B

	full_text


i64 %4
^getelementptr8BK
I
	full_text<
:
8%486 = getelementptr inbounds float, float* %1, i64 %485
&i648B

	full_text


i64 %485
Nstore8BC
A
	full_text4
2
0store float %484, float* %486, align 4, !tbaa !8
*float8B

	full_text


float %484
,float*8B

	full_text

float* %486
ƒcall8By
w
	full_textj
h
f%487 = tail call float @llvm.fmuladd.f32(float %9, float 0xC08E94CF00000000, float 0x402A4DEA20000000)
(float8B

	full_text


float %9
vcall8Bl
j
	full_text]
[
Y%488 = tail call float @llvm.fmuladd.f32(float %10, float 0x4000D98180000000, float %487)
)float8B

	full_text

	float %10
*float8B

	full_text


float %487
†call8B|
z
	full_textm
k
i%489 = tail call float @llvm.fmuladd.f32(float %7, float 0x3D3D362C60000000, float 0xBE051FDD40000000) #4
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%490 = tail call float @llvm.fmuladd.f32(float %489, float %7, float 0x3EADDADAA0000000) #4
*float8B

	full_text


float %489
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%491 = tail call float @llvm.fmuladd.f32(float %490, float %7, float 0x3F6D8F2600000000) #4
*float8B

	full_text


float %490
(float8B

	full_text


float %7
7fmul8B-
+
	full_text

%492 = fmul float %7, %491
(float8B

	full_text


float %7
*float8B

	full_text


float %491
9fadd8B/
-
	full_text 

%493 = fadd float %488, %492
*float8B

	full_text


float %488
*float8B

	full_text


float %492
Kcall8BA
?
	full_text2
0
.%494 = tail call float @_Z3expf(float %493) #3
*float8B

	full_text


float %493
2add8B)
'
	full_text

%495 = add i64 %4, 136
$i648B

	full_text


i64 %4
^getelementptr8BK
I
	full_text<
:
8%496 = getelementptr inbounds float, float* %1, i64 %495
&i648B

	full_text


i64 %495
Nstore8BC
A
	full_text4
2
0store float %494, float* %496, align 4, !tbaa !8
*float8B

	full_text


float %494
,float*8B

	full_text

float* %496
ƒcall8By
w
	full_textj
h
f%497 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0D9CF3EC0000000, float 0x402BE12100000000)
(float8B

	full_text


float %9
vcall8Bl
j
	full_text]
[
Y%498 = tail call float @llvm.fmuladd.f32(float %10, float 0x3FE9E0B720000000, float %497)
)float8B

	full_text

	float %10
*float8B

	full_text


float %497
†call8B|
z
	full_textm
k
i%499 = tail call float @llvm.fmuladd.f32(float %7, float 0xBD5DE8C6E0000000, float 0x3E240DD900000000) #4
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%500 = tail call float @llvm.fmuladd.f32(float %499, float %7, float 0xBED8D40C20000000) #4
*float8B

	full_text


float %499
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%501 = tail call float @llvm.fmuladd.f32(float %500, float %7, float 0x3F87EC1800000000) #4
*float8B

	full_text


float %500
(float8B

	full_text


float %7
7fmul8B-
+
	full_text

%502 = fmul float %7, %501
(float8B

	full_text


float %7
*float8B

	full_text


float %501
9fadd8B/
-
	full_text 

%503 = fadd float %498, %502
*float8B

	full_text


float %498
*float8B

	full_text


float %502
Kcall8BA
?
	full_text2
0
.%504 = tail call float @_Z3expf(float %503) #3
*float8B

	full_text


float %503
2add8B)
'
	full_text

%505 = add i64 %4, 144
$i648B

	full_text


i64 %4
^getelementptr8BK
I
	full_text<
:
8%506 = getelementptr inbounds float, float* %1, i64 %505
&i648B

	full_text


i64 %505
Nstore8BC
A
	full_text4
2
0store float %504, float* %506, align 4, !tbaa !8
*float8B

	full_text


float %504
,float*8B

	full_text

float* %506
ƒcall8By
w
	full_textj
h
f%507 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0E7BDB960000000, float 0x4017AE7B00000000)
(float8B

	full_text


float %9
vcall8Bl
j
	full_text]
[
Y%508 = tail call float @llvm.fmuladd.f32(float %10, float 0x400A409C60000000, float %507)
)float8B

	full_text

	float %10
*float8B

	full_text


float %507
†call8B|
z
	full_textm
k
i%509 = tail call float @llvm.fmuladd.f32(float %7, float 0x3D2BA34D60000000, float 0xBDDBBA1D20000000) #4
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%510 = tail call float @llvm.fmuladd.f32(float %509, float %7, float 0xBE9AAE7FE0000000) #4
*float8B

	full_text


float %509
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%511 = tail call float @llvm.fmuladd.f32(float %510, float %7, float 0x3F6C935E60000000) #4
*float8B

	full_text


float %510
(float8B

	full_text


float %7
7fmul8B-
+
	full_text

%512 = fmul float %7, %511
(float8B

	full_text


float %7
*float8B

	full_text


float %511
9fadd8B/
-
	full_text 

%513 = fadd float %508, %512
*float8B

	full_text


float %508
*float8B

	full_text


float %512
Kcall8BA
?
	full_text2
0
.%514 = tail call float @_Z3expf(float %513) #3
*float8B

	full_text


float %513
2add8B)
'
	full_text

%515 = add i64 %4, 152
$i648B

	full_text


i64 %4
^getelementptr8BK
I
	full_text<
:
8%516 = getelementptr inbounds float, float* %1, i64 %515
&i648B

	full_text


i64 %515
Nstore8BC
A
	full_text4
2
0store float %514, float* %516, align 4, !tbaa !8
*float8B

	full_text


float %514
,float*8B

	full_text

float* %516
ƒcall8By
w
	full_textj
h
f%517 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0E1057B20000000, float 0x4021056580000000)
(float8B

	full_text


float %9
vcall8Bl
j
	full_text]
[
Y%518 = tail call float @llvm.fmuladd.f32(float %10, float 0x4009B321A0000000, float %517)
)float8B

	full_text

	float %10
*float8B

	full_text


float %517
†call8B|
z
	full_textm
k
i%519 = tail call float @llvm.fmuladd.f32(float %7, float 0x3D69E31600000000, float 0xBE299A2640000000) #4
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%520 = tail call float @llvm.fmuladd.f32(float %519, float %7, float 0x3ED21EBBA0000000) #4
*float8B

	full_text


float %519
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%521 = tail call float @llvm.fmuladd.f32(float %520, float %7, float 0x3F48D17F20000000) #4
*float8B

	full_text


float %520
(float8B

	full_text


float %7
7fmul8B-
+
	full_text

%522 = fmul float %7, %521
(float8B

	full_text


float %7
*float8B

	full_text


float %521
9fadd8B/
-
	full_text 

%523 = fadd float %518, %522
*float8B

	full_text


float %518
*float8B

	full_text


float %522
Kcall8BA
?
	full_text2
0
.%524 = tail call float @_Z3expf(float %523) #3
*float8B

	full_text


float %523
2add8B)
'
	full_text

%525 = add i64 %4, 160
$i648B

	full_text


i64 %4
^getelementptr8BK
I
	full_text<
:
8%526 = getelementptr inbounds float, float* %1, i64 %525
&i648B

	full_text


i64 %525
Nstore8BC
A
	full_text4
2
0store float %524, float* %526, align 4, !tbaa !8
*float8B

	full_text


float %524
,float*8B

	full_text

float* %526
ƒcall8By
w
	full_textj
h
f%527 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0B3E1C6A0000000, float 0x401063AAC0000000)
(float8B

	full_text


float %9
vcall8Bl
j
	full_text]
[
Y%528 = tail call float @llvm.fmuladd.f32(float %10, float 0x400FAC71E0000000, float %527)
)float8B

	full_text

	float %10
*float8B

	full_text


float %527
†call8B|
z
	full_textm
k
i%529 = tail call float @llvm.fmuladd.f32(float %7, float 0x3D77BD4180000000, float 0xBE38C0BFC0000000) #4
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%530 = tail call float @llvm.fmuladd.f32(float %529, float %7, float 0x3EE3F52280000000) #4
*float8B

	full_text


float %529
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%531 = tail call float @llvm.fmuladd.f32(float %530, float %7, float 0xBF6F0244A0000000) #4
*float8B

	full_text


float %530
(float8B

	full_text


float %7
7fmul8B-
+
	full_text

%532 = fmul float %7, %531
(float8B

	full_text


float %7
*float8B

	full_text


float %531
9fadd8B/
-
	full_text 

%533 = fadd float %528, %532
*float8B

	full_text


float %528
*float8B

	full_text


float %532
Kcall8BA
?
	full_text2
0
.%534 = tail call float @_Z3expf(float %533) #3
*float8B

	full_text


float %533
2add8B)
'
	full_text

%535 = add i64 %4, 168
$i648B

	full_text


i64 %4
^getelementptr8BK
I
	full_text<
:
8%536 = getelementptr inbounds float, float* %1, i64 %535
&i648B

	full_text


i64 %535
Nstore8BC
A
	full_text4
2
0store float %534, float* %536, align 4, !tbaa !8
*float8B

	full_text


float %534
,float*8B

	full_text

float* %536
ƒcall8By
w
	full_textj
h
f%537 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0C914D040000000, float 0x4012D42EA0000000)
(float8B

	full_text


float %9
vcall8Bl
j
	full_text]
[
Y%538 = tail call float @llvm.fmuladd.f32(float %10, float 0x401139D220000000, float %537)
)float8B

	full_text

	float %10
*float8B

	full_text


float %537
†call8B|
z
	full_textm
k
i%539 = tail call float @llvm.fmuladd.f32(float %7, float 0x3D74469A00000000, float 0xBE35718E40000000) #4
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%540 = tail call float @llvm.fmuladd.f32(float %539, float %7, float 0x3EE1605BC0000000) #4
*float8B

	full_text


float %539
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%541 = tail call float @llvm.fmuladd.f32(float %540, float %7, float 0xBF6125F4E0000000) #4
*float8B

	full_text


float %540
(float8B

	full_text


float %7
7fmul8B-
+
	full_text

%542 = fmul float %7, %541
(float8B

	full_text


float %7
*float8B

	full_text


float %541
9fadd8B/
-
	full_text 

%543 = fadd float %538, %542
*float8B

	full_text


float %538
*float8B

	full_text


float %542
Kcall8BA
?
	full_text2
0
.%544 = tail call float @_Z3expf(float %543) #3
*float8B

	full_text


float %543
2add8B)
'
	full_text

%545 = add i64 %4, 176
$i648B

	full_text


i64 %4
^getelementptr8BK
I
	full_text<
:
8%546 = getelementptr inbounds float, float* %1, i64 %545
&i648B

	full_text


i64 %545
Nstore8BC
A
	full_text4
2
0store float %544, float* %546, align 4, !tbaa !8
*float8B

	full_text


float %544
,float*8B

	full_text

float* %546
ƒcall8By
w
	full_textj
h
f%547 = tail call float @llvm.fmuladd.f32(float %9, float 0x40C6811A40000000, float 0x400555A760000000)
(float8B

	full_text


float %9
vcall8Bl
j
	full_text]
[
Y%548 = tail call float @llvm.fmuladd.f32(float %10, float 0x40112A6B40000000, float %547)
)float8B

	full_text

	float %10
*float8B

	full_text


float %547
†call8B|
z
	full_textm
k
i%549 = tail call float @llvm.fmuladd.f32(float %7, float 0x3D77A24400000000, float 0xBE395B6420000000) #4
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%550 = tail call float @llvm.fmuladd.f32(float %549, float %7, float 0x3EE4F3AEE0000000) #4
*float8B

	full_text


float %549
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%551 = tail call float @llvm.fmuladd.f32(float %550, float %7, float 0xBF6688C920000000) #4
*float8B

	full_text


float %550
(float8B

	full_text


float %7
7fmul8B-
+
	full_text

%552 = fmul float %7, %551
(float8B

	full_text


float %7
*float8B

	full_text


float %551
9fadd8B/
-
	full_text 

%553 = fadd float %548, %552
*float8B

	full_text


float %548
*float8B

	full_text


float %552
Kcall8BA
?
	full_text2
0
.%554 = tail call float @_Z3expf(float %553) #3
*float8B

	full_text


float %553
2add8B)
'
	full_text

%555 = add i64 %4, 184
$i648B

	full_text


i64 %4
^getelementptr8BK
I
	full_text<
:
8%556 = getelementptr inbounds float, float* %1, i64 %555
&i648B

	full_text


i64 %555
Nstore8BC
A
	full_text4
2
0store float %554, float* %556, align 4, !tbaa !8
*float8B

	full_text


float %554
,float*8B

	full_text

float* %556
ƒcall8By
w
	full_textj
h
f%557 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0D396DCC0000000, float 0x4028FB17E0000000)
(float8B

	full_text


float %9
vcall8Bl
j
	full_text]
[
Y%558 = tail call float @llvm.fmuladd.f32(float %10, float 0x4002038680000000, float %557)
)float8B

	full_text

	float %10
*float8B

	full_text


float %557
†call8B|
z
	full_textm
k
i%559 = tail call float @llvm.fmuladd.f32(float %7, float 0xBD51D37B00000000, float 0x3E18BBA200000000) #4
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%560 = tail call float @llvm.fmuladd.f32(float %559, float %7, float 0xBED0967CE0000000) #4
*float8B

	full_text


float %559
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%561 = tail call float @llvm.fmuladd.f32(float %560, float %7, float 0x3F82142860000000) #4
*float8B

	full_text


float %560
(float8B

	full_text


float %7
7fmul8B-
+
	full_text

%562 = fmul float %7, %561
(float8B

	full_text


float %7
*float8B

	full_text


float %561
9fadd8B/
-
	full_text 

%563 = fadd float %558, %562
*float8B

	full_text


float %558
*float8B

	full_text


float %562
Kcall8BA
?
	full_text2
0
.%564 = tail call float @_Z3expf(float %563) #3
*float8B

	full_text


float %563
2add8B)
'
	full_text

%565 = add i64 %4, 192
$i648B

	full_text


i64 %4
^getelementptr8BK
I
	full_text<
:
8%566 = getelementptr inbounds float, float* %1, i64 %565
&i648B

	full_text


i64 %565
Nstore8BC
A
	full_text4
2
0store float %564, float* %566, align 4, !tbaa !8
*float8B

	full_text


float %564
,float*8B

	full_text

float* %566
ƒcall8By
w
	full_textj
h
f%567 = tail call float @llvm.fmuladd.f32(float %9, float 0x40BB82EB00000000, float 0x40286E6960000000)
(float8B

	full_text


float %9
vcall8Bl
j
	full_text]
[
Y%568 = tail call float @llvm.fmuladd.f32(float %10, float 0x4001163160000000, float %567)
)float8B

	full_text

	float %10
*float8B

	full_text


float %567
†call8B|
z
	full_textm
k
i%569 = tail call float @llvm.fmuladd.f32(float %7, float 0xBD3C5A4680000000, float 0x3E0AC134E0000000) #4
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%570 = tail call float @llvm.fmuladd.f32(float %569, float %7, float 0xBEC851D2A0000000) #4
*float8B

	full_text


float %569
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%571 = tail call float @llvm.fmuladd.f32(float %570, float %7, float 0x3F828DC0E0000000) #4
*float8B

	full_text


float %570
(float8B

	full_text


float %7
7fmul8B-
+
	full_text

%572 = fmul float %7, %571
(float8B

	full_text


float %7
*float8B

	full_text


float %571
9fadd8B/
-
	full_text 

%573 = fadd float %568, %572
*float8B

	full_text


float %568
*float8B

	full_text


float %572
Kcall8BA
?
	full_text2
0
.%574 = tail call float @_Z3expf(float %573) #3
*float8B

	full_text


float %573
2add8B)
'
	full_text

%575 = add i64 %4, 200
$i648B

	full_text


i64 %4
^getelementptr8BK
I
	full_text<
:
8%576 = getelementptr inbounds float, float* %1, i64 %575
&i648B

	full_text


i64 %575
Nstore8BC
A
	full_text4
2
0store float %574, float* %576, align 4, !tbaa !8
*float8B

	full_text


float %574
,float*8B

	full_text

float* %576
ƒcall8By
w
	full_textj
h
f%577 = tail call float @llvm.fmuladd.f32(float %9, float 0xC097C5E800000000, float 0x4023249580000000)
(float8B

	full_text


float %9
vcall8Bl
j
	full_text]
[
Y%578 = tail call float @llvm.fmuladd.f32(float %10, float 0x400B45C280000000, float %577)
)float8B

	full_text

	float %10
*float8B

	full_text


float %577
†call8B|
z
	full_textm
k
i%579 = tail call float @llvm.fmuladd.f32(float %7, float 0x3D442D6C00000000, float 0x3E047F4C00000000) #4
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%580 = tail call float @llvm.fmuladd.f32(float %579, float %7, float 0x3E9527EEA0000000) #4
*float8B

	full_text


float %579
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%581 = tail call float @llvm.fmuladd.f32(float %580, float %7, float 0x3F75FE1B00000000) #4
*float8B

	full_text


float %580
(float8B

	full_text


float %7
7fmul8B-
+
	full_text

%582 = fmul float %7, %581
(float8B

	full_text


float %7
*float8B

	full_text


float %581
9fadd8B/
-
	full_text 

%583 = fadd float %578, %582
*float8B

	full_text


float %578
*float8B

	full_text


float %582
Kcall8BA
?
	full_text2
0
.%584 = tail call float @_Z3expf(float %583) #3
*float8B

	full_text


float %583
2add8B)
'
	full_text

%585 = add i64 %4, 208
$i648B

	full_text


i64 %4
^getelementptr8BK
I
	full_text<
:
8%586 = getelementptr inbounds float, float* %1, i64 %585
&i648B

	full_text


i64 %585
Nstore8BC
A
	full_text4
2
0store float %584, float* %586, align 4, !tbaa !8
*float8B

	full_text


float %584
,float*8B

	full_text

float* %586
ƒcall8By
w
	full_textj
h
f%587 = tail call float @llvm.fmuladd.f32(float %9, float 0x40D5113840000000, float 0x4010697D00000000)
(float8B

	full_text


float %9
vcall8Bl
j
	full_text]
[
Y%588 = tail call float @llvm.fmuladd.f32(float %10, float 0x4012EAF760000000, float %587)
)float8B

	full_text

	float %10
*float8B

	full_text


float %587
†call8B|
z
	full_textm
k
i%589 = tail call float @llvm.fmuladd.f32(float %7, float 0x3D734A7280000000, float 0xBE3490B360000000) #4
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%590 = tail call float @llvm.fmuladd.f32(float %589, float %7, float 0x3EE09D5A40000000) #4
*float8B

	full_text


float %589
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%591 = tail call float @llvm.fmuladd.f32(float %590, float %7, float 0xBF5A28CE40000000) #4
*float8B

	full_text


float %590
(float8B

	full_text


float %7
7fmul8B-
+
	full_text

%592 = fmul float %7, %591
(float8B

	full_text


float %7
*float8B

	full_text


float %591
9fadd8B/
-
	full_text 

%593 = fadd float %588, %592
*float8B

	full_text


float %588
*float8B

	full_text


float %592
Kcall8BA
?
	full_text2
0
.%594 = tail call float @_Z3expf(float %593) #3
*float8B

	full_text


float %593
2add8B)
'
	full_text

%595 = add i64 %4, 216
$i648B

	full_text


i64 %4
^getelementptr8BK
I
	full_text<
:
8%596 = getelementptr inbounds float, float* %1, i64 %595
&i648B

	full_text


i64 %595
Nstore8BC
A
	full_text4
2
0store float %594, float* %596, align 4, !tbaa !8
*float8B

	full_text


float %594
,float*8B

	full_text

float* %596
ƒcall8By
w
	full_textj
h
f%597 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0D2CB6840000000, float 0x40312C57C0000000)
(float8B

	full_text


float %9
vcall8Bl
j
	full_text]
[
Y%598 = tail call float @llvm.fmuladd.f32(float %10, float 0x3FF5CF9980000000, float %597)
)float8B

	full_text

	float %10
*float8B

	full_text


float %597
†call8B|
z
	full_textm
k
i%599 = tail call float @llvm.fmuladd.f32(float %7, float 0x3D6BE0A940000000, float 0xBE27E07860000000) #4
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%600 = tail call float @llvm.fmuladd.f32(float %599, float %7, float 0x3EC178DF40000000) #4
*float8B

	full_text


float %599
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%601 = tail call float @llvm.fmuladd.f32(float %600, float %7, float 0x3F844A1300000000) #4
*float8B

	full_text


float %600
(float8B

	full_text


float %7
7fmul8B-
+
	full_text

%602 = fmul float %7, %601
(float8B

	full_text


float %7
*float8B

	full_text


float %601
9fadd8B/
-
	full_text 

%603 = fadd float %598, %602
*float8B

	full_text


float %598
*float8B

	full_text


float %602
Kcall8BA
?
	full_text2
0
.%604 = tail call float @_Z3expf(float %603) #3
*float8B

	full_text


float %603
2add8B)
'
	full_text

%605 = add i64 %4, 224
$i648B

	full_text


i64 %4
^getelementptr8BK
I
	full_text<
:
8%606 = getelementptr inbounds float, float* %1, i64 %605
&i648B

	full_text


i64 %605
Nstore8BC
A
	full_text4
2
0store float %604, float* %606, align 4, !tbaa !8
*float8B

	full_text


float %604
,float*8B

	full_text

float* %606
ƒcall8By
w
	full_textj
h
f%607 = tail call float @llvm.fmuladd.f32(float %9, float 0xC090CB4DE0000000, float 0x4030253500000000)
(float8B

	full_text


float %9
vcall8Bl
j
	full_text]
[
Y%608 = tail call float @llvm.fmuladd.f32(float %10, float 0x3FF7E495E0000000, float %607)
)float8B

	full_text

	float %10
*float8B

	full_text


float %607
†call8B|
z
	full_textm
k
i%609 = tail call float @llvm.fmuladd.f32(float %7, float 0x3D592F7C20000000, float 0xBE17E4A080000000) #4
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%610 = tail call float @llvm.fmuladd.f32(float %609, float %7, float 0x3EA9178B60000000) #4
*float8B

	full_text


float %609
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%611 = tail call float @llvm.fmuladd.f32(float %610, float %7, float 0x3F856D6900000000) #4
*float8B

	full_text


float %610
(float8B

	full_text


float %7
7fmul8B-
+
	full_text

%612 = fmul float %7, %611
(float8B

	full_text


float %7
*float8B

	full_text


float %611
9fadd8B/
-
	full_text 

%613 = fadd float %608, %612
*float8B

	full_text


float %608
*float8B

	full_text


float %612
Kcall8BA
?
	full_text2
0
.%614 = tail call float @_Z3expf(float %613) #3
*float8B

	full_text


float %613
2add8B)
'
	full_text

%615 = add i64 %4, 232
$i648B

	full_text


i64 %4
^getelementptr8BK
I
	full_text<
:
8%616 = getelementptr inbounds float, float* %1, i64 %615
&i648B

	full_text


i64 %615
Nstore8BC
A
	full_text4
2
0store float %614, float* %616, align 4, !tbaa !8
*float8B

	full_text


float %614
,float*8B

	full_text

float* %616
ƒcall8By
w
	full_textj
h
f%617 = tail call float @llvm.fmuladd.f32(float %9, float 0xC0C4242C40000000, float 0x403522D320000000)
(float8B

	full_text


float %9
vcall8Bl
j
	full_text]
[
Y%618 = tail call float @llvm.fmuladd.f32(float %10, float 0x3FF0C92F40000000, float %617)
)float8B

	full_text

	float %10
*float8B

	full_text


float %617
†call8B|
z
	full_textm
k
i%619 = tail call float @llvm.fmuladd.f32(float %7, float 0x3D607CC860000000, float 0xBE1C0DB120000000) #4
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%620 = tail call float @llvm.fmuladd.f32(float %619, float %7, float 0x3E9A54F4A0000000) #4
*float8B

	full_text


float %619
(float8B

	full_text


float %7
xcall8Bn
l
	full_text_
]
[%621 = tail call float @llvm.fmuladd.f32(float %620, float %7, float 0x3F8AA218A0000000) #4
*float8B

	full_text


float %620
(float8B

	full_text


float %7
(br8B 

	full_text

br label %622
Jphi8BA
?
	full_text2
0
.%623 = phi float [ %621, %317 ], [ %316, %12 ]
*float8B

	full_text


float %621
*float8B

	full_text


float %316
Jphi8BA
?
	full_text2
0
.%624 = phi float [ %618, %317 ], [ %313, %12 ]
*float8B

	full_text


float %618
*float8B

	full_text


float %313
7fmul8B-
+
	full_text

%625 = fmul float %7, %623
(float8B

	full_text


float %7
*float8B

	full_text


float %623
9fadd8B/
-
	full_text 

%626 = fadd float %624, %625
*float8B

	full_text


float %624
*float8B

	full_text


float %625
Kcall8BA
?
	full_text2
0
.%627 = tail call float @_Z3expf(float %626) #3
*float8B

	full_text


float %626
2add8B)
'
	full_text

%628 = add i64 %4, 240
$i648B

	full_text


i64 %4
^getelementptr8BK
I
	full_text<
:
8%629 = getelementptr inbounds float, float* %1, i64 %628
&i648B

	full_text


i64 %628
Nstore8BC
A
	full_text4
2
0store float %627, float* %629, align 4, !tbaa !8
*float8B

	full_text


float %627
,float*8B

	full_text

float* %629
$ret8B

	full_text


ret void
*float*8B

	full_text

	float* %0
(float8B

	full_text


float %2
*float*8B

	full_text

	float* %1
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
8float8B+
)
	full_text

float 0x3FF1266D40000000
8float8B+
)
	full_text

float 0x3F78024260000000
8float8B+
)
	full_text

float 0x3ED21EBBA0000000
8float8B+
)
	full_text

float 0x4013DDF900000000
8float8B+
)
	full_text

float 0x3FE47CD260000000
8float8B+
)
	full_text

float 0x40120B9180000000
8float8B+
)
	full_text

float 0x3D3D362C60000000
8float8B+
)
	full_text

float 0x4012D42EA0000000
8float8B+
)
	full_text

float 0xBE051FDD40000000
8float8B+
)
	full_text

float 0xBCE2753BA0000000
8float8B+
)
	full_text

float 0x4008BE3BE0000000
%i648B

	full_text
	
i64 112
8float8B+
)
	full_text

float 0x3E240DD900000000
8float8B+
)
	full_text

float 0xBDA961A6E0000000
8float8B+
)
	full_text

float 0x3D58D112C0000000
8float8B+
)
	full_text

float 0xBD51D37B00000000
8float8B+
)
	full_text

float 0x3E9A54F4A0000000
$i648B

	full_text


i64 72
8float8B+
)
	full_text

float 0x3E59D97C80000000
8float8B+
)
	full_text

float 0x3F63120D00000000
8float8B+
)
	full_text

float 0x3EE09D5A40000000
8float8B+
)
	full_text

float 0x40326FF420000000
8float8B+
)
	full_text

float 0x4010CB5EE0000000
8float8B+
)
	full_text

float 0x3F6D8F2600000000
8float8B+
)
	full_text

float 0x3F3FBF7D20000000
8float8B+
)
	full_text

float 0xBD5DE8C6E0000000
8float8B+
)
	full_text

float 0xBEA8BB9FC0000000
8float8B+
)
	full_text

float 0x3EB005D9A0000000
8float8B+
)
	full_text

float 0x4028FB17E0000000
8float8B+
)
	full_text

float 0x3EC34408C0000000
8float8B+
)
	full_text

float 0x4000AC0E00000000
8float8B+
)
	full_text

float 0xBED8D40C20000000
8float8B+
)
	full_text

float 0xBCD17B2440000000
8float8B+
)
	full_text

float 0x3F484C8520000000
8float8B+
)
	full_text

float 0x3EBB88F920000000
8float8B+
)
	full_text

float 0x40249C5960000000
8float8B+
)
	full_text

float 0xBCE806EFC0000000
8float8B+
)
	full_text

float 0xBFE89C9F60000000
8float8B+
)
	full_text

float 0x4005B8B340000000
8float8B+
)
	full_text

float 0x3DE0DC9F20000000
8float8B+
)
	full_text

float 0x408CAF7B40000000
8float8B+
)
	full_text

float 0xBE5D5CA6E0000000
8float8B+
)
	full_text

float 0x402E3B3160000000
8float8B+
)
	full_text

float 0x3F48D17F20000000
8float8B+
)
	full_text

float 0x3D3332BDC0000000
8float8B+
)
	full_text

float 0x400BEB2500000000
8float8B+
)
	full_text

float 0xBDA1C87B60000000
8float8B+
)
	full_text

float 0x4002561840000000
8float8B+
)
	full_text

float 0x3F701EEE80000000
8float8B+
)
	full_text

float 0x40286E6960000000
8float8B+
)
	full_text

float 0x3F6214CD80000000
$i648B

	full_text


i64 40
$i648B

	full_text


i64 96
8float8B+
)
	full_text

float 0xC0267C7100000000
8float8B+
)
	full_text

float 0x400B27ACC0000000
8float8B+
)
	full_text

float 0x3DC569DE40000000
8float8B+
)
	full_text

float 0x3FFC2BC960000000
8float8B+
)
	full_text

float 0x401F263840000000
$i648B

	full_text


i64 80
8float8B+
)
	full_text

float 0x4023CD56C0000000
8float8B+
)
	full_text

float 0xBF53632660000000
8float8B+
)
	full_text

float 0xC097C5E800000000
8float8B+
)
	full_text

float 0x401139D220000000
8float8B+
)
	full_text

float 0xBCD0E8B400000000
8float8B+
)
	full_text

float 0x3F806A8EC0000000
8float8B+
)
	full_text

float 0xBF34000480000000
8float8B+
)
	full_text

float 0xC0C4242C40000000
$i648B

	full_text


i64 32
8float8B+
)
	full_text

float 0xBE31598140000000
8float8B+
)
	full_text

float 0x40111CB500000000
8float8B+
)
	full_text

float 0xC0C914D040000000
8float8B+
)
	full_text

float 0x3CD2099320000000
8float8B+
)
	full_text

float 0xBE3490B360000000
8float8B+
)
	full_text

float 0x40DD4D1300000000
8float8B+
)
	full_text

float 0xBF53ABED80000000
8float8B+
)
	full_text

float 0xBE9680C0A0000000
8float8B+
)
	full_text

float 0x40159DCF40000000
8float8B+
)
	full_text

float 0x3EA9178B60000000
8float8B+
)
	full_text

float 0xBCF36C9740000000
8float8B+
)
	full_text

float 0x3F752803E0000000
8float8B+
)
	full_text

float 0x400A409C60000000
8float8B+
)
	full_text

float 0x4010971C80000000
8float8B+
)
	full_text

float 0x3FE9E0B720000000
8float8B+
)
	full_text

float 0x3FF9002160000000
%i648B

	full_text
	
i64 168
8float8B+
)
	full_text

float 0x402398C0A0000000
8float8B+
)
	full_text

float 0x401ED6C820000000
8float8B+
)
	full_text

float 0xBF7BFF87C0000000
8float8B+
)
	full_text

float 0x3E9F42AA40000000
8float8B+
)
	full_text

float 0xBE130FC860000000
8float8B+
)
	full_text

float 0x3D442D6C00000000
8float8B+
)
	full_text

float 0xBEC851D2A0000000
8float8B+
)
	full_text

float 0x3DC10150C0000000
8float8B+
)
	full_text

float 0xBDB073F440000000
8float8B+
)
	full_text

float 0xBD33C9F9C0000000
8float8B+
)
	full_text

float 0x3ECD94D8C0000000
8float8B+
)
	full_text

float 0x400555A760000000
8float8B+
)
	full_text

float 0xBCD257CBE0000000
8float8B+
)
	full_text

float 0x402AECC440000000
8float8B+
)
	full_text

float 0x40D61047C0000000
8float8B+
)
	full_text

float 0x3F52593E40000000
8float8B+
)
	full_text

float 0xBE03AC9FC0000000
8float8B+
)
	full_text

float 0x3CCFB83A80000000
8float8B+
)
	full_text

float 0x400B7AFBE0000000
8float8B+
)
	full_text

float 0x40D5113840000000
8float8B+
)
	full_text

float 0x3DD961D9C0000000
8float8B+
)
	full_text

float 0xC02AA06F60000000
8float8B+
)
	full_text

float 0x3DCF52CE40000000
8float8B+
)
	full_text

float 0x3EB23B7C60000000
8float8B+
)
	full_text

float 0x4023249580000000
8float8B+
)
	full_text

float 0xBD59F3D0E0000000
8float8B+
)
	full_text

float 0xBE018BEB80000000
8float8B+
)
	full_text

float 0xC05BF6D460000000
8float8B+
)
	full_text

float 0x4000D98180000000
8float8B+
)
	full_text

float 0xBEAFC12CE0000000
8float8B+
)
	full_text

float 0x400E47E3A0000000
8float8B+
)
	full_text

float 0xBE1EE41580000000
8float8B+
)
	full_text

float 0x40BD7F0DA0000000
8float8B+
)
	full_text

float 0x40048E2C80000000
8float8B+
)
	full_text

float 0xBE17E4A080000000
8float8B+
)
	full_text

float 0x3DBA3A9900000000
8float8B+
)
	full_text

float 0x40BB82EB00000000
8float8B+
)
	full_text

float 0x3D2BA34D60000000
8float8B+
)
	full_text

float 0x4016834860000000
8float8B+
)
	full_text

float 0xBF50AEB640000000
%i648B

	full_text
	
i64 216
8float8B+
)
	full_text

float 0x40DD956E80000000
8float8B+
)
	full_text

float 0xBE9AAE7FE0000000
8float8B+
)
	full_text

float 0x3DD74F7660000000
8float8B+
)
	full_text

float 0xBEABAE8D20000000
8float8B+
)
	full_text

float 0x3CCE4CE6E0000000
8float8B+
)
	full_text

float 0x4011350A80000000
8float8B+
)
	full_text

float 0xBE2160B200000000
8float8B+
)
	full_text

float 0x400D42EB80000000
8float8B+
)
	full_text

float 0x3D3DBBA8A0000000
8float8B+
)
	full_text

float 0x3EE4F3AEE0000000
8float8B+
)
	full_text

float 0xBEB007BD60000000
8float8B+
)
	full_text

float 0x3F7D5648E0000000
%i648B

	full_text
	
i64 176
8float8B+
)
	full_text

float 0x3F828DC0E0000000
8float8B+
)
	full_text

float 0x3DD4C6BD20000000
8float8B+
)
	full_text

float 0x40112A6B40000000
8float8B+
)
	full_text

float 0x40132CC5C0000000
8float8B+
)
	full_text

float 0x3EA9D34C60000000
8float8B+
)
	full_text

float 0x3FFF4645C0000000
8float8B+
)
	full_text

float 0x3EE1308EA0000000
8float8B+
)
	full_text

float 0x400DBBB980000000
8float8B+
)
	full_text

float 0x401F465620000000
8float8B+
)
	full_text

float 0xBCF1E5EE20000000
8float8B+
)
	full_text

float 0x3D607CC860000000
%i648B

	full_text
	
i64 232
8float8B+
)
	full_text

float 0x400FEFA5C0000000
8float8B+
)
	full_text

float 0x3D4E8615E0000000
8float8B+
)
	full_text

float 0x4009589C60000000
8float8B+
)
	full_text

float 0xBD297510C0000000
8float8B+
)
	full_text

float 0x402B5009A0000000
8float8B+
)
	full_text

float 0x3D734A7280000000
8float8B+
)
	full_text

float 0xBE299A2640000000
8float8B+
)
	full_text

float 0x40006A5C20000000
$i648B

	full_text


i64 88
8float8B+
)
	full_text

float 0x3D74469A00000000
8float8B+
)
	full_text

float 0x3D6058DBA0000000
8float8B+
)
	full_text

float 0x3B3E1D3B00000000
8float8B+
)
	full_text

float 0x3EE3F52280000000
8float8B+
)
	full_text

float 0x3E9527EEA0000000
8float8B+
)
	full_text

float 0xBEF9E6B000000000
8float8B+
)
	full_text

float 0xC0DC886500000000
8float8B+
)
	full_text

float 0x3D3B58ED20000000
8float8B+
)
	full_text

float 0xC0DC7090A0000000
8float8B+
)
	full_text

float 0x3D672E8340000000
8float8B+
)
	full_text

float 0x3F6C935E60000000
8float8B+
)
	full_text

float 0xBEA75123E0000000
8float8B+
)
	full_text

float 0x4015D01BE0000000
8float8B+
)
	full_text

float 0x40076FC500000000
8float8B+
)
	full_text

float 0x3FE34A3E40000000
8float8B+
)
	full_text

float 0x40D1717260000000
8float8B+
)
	full_text

float 0x3EB7056240000000
%i648B

	full_text
	
i64 104
8float8B+
)
	full_text

float 0xC009A3E340000000
8float8B+
)
	full_text

float 0x403522D320000000
8float8B+
)
	full_text

float 0x4011E82300000000
8float8B+
)
	full_text

float 0x40C6811A40000000
8float8B+
)
	full_text

float 0x3E0AC134E0000000
8float8B+
)
	full_text

float 0x3F70A6C580000000
8float8B+
)
	full_text

float 0xBE7C597160000000
8float8B+
)
	full_text

float 0xBDA7F2E4A0000000
8float8B+
)
	full_text

float 0x400CA2E280000000
8float8B+
)
	full_text

float 0x3D592F7C20000000
8float8B+
)
	full_text

float 0xBE95444740000000
8float8B+
)
	full_text

float 0x3F856D6900000000
8float8B+
)
	full_text

float 0xBE9BC9C5A0000000
8float8B+
)
	full_text

float 0x400E2A98A0000000
8float8B+
)
	full_text

float 0x3CE840F100000000
%i648B

	full_text
	
i64 160
8float8B+
)
	full_text

float 0xC0BF283940000000
8float8B+
)
	full_text

float 0xBF5A28CE40000000
8float8B+
)
	full_text

float 0x3DB7D6D600000000
8float8B+
)
	full_text

float 0xBEA0B48FA0000000
8float8B+
)
	full_text

float 0x40CBA3EFA0000000
8float8B+
)
	full_text

float 0xBE9A8A7DA0000000
8float8B+
)
	full_text

float 0xBD6D5F5860000000
8float8B+
)
	full_text

float 0xBE92E41B40000000
8float8B+
)
	full_text

float 0x3F641ABE40000000
8float8B+
)
	full_text

float 0x400E19F740000000
8float8B+
)
	full_text

float 0x3FE5DB3840000000
8float8B+
)
	full_text

float 0x3E765866C0000000
8float8B+
)
	full_text

float 0x3F50794580000000
8float8B+
)
	full_text

float 0x4006FE28C0000000
8float8B+
)
	full_text

float 0x400049F4A0000000
3float8B&
$
	full_text

float -1.000000e+00
8float8B+
)
	full_text

float 0x4010CB6860000000
8float8B+
)
	full_text

float 0xBCD9EEB6A0000000
8float8B+
)
	full_text

float 0x40C27E2C20000000
8float8B+
)
	full_text

float 0xC01290B1E0000000
8float8B+
)
	full_text

float 0x3F6DA79600000000
8float8B+
)
	full_text

float 0x3F31F88FE0000000
8float8B+
)
	full_text

float 0x3FF9AC4BA0000000
8float8B+
)
	full_text

float 0xBCE509EC60000000
8float8B+
)
	full_text

float 0xBE91D28EA0000000
8float8B+
)
	full_text

float 0x3DCB4A4360000000
8float8B+
)
	full_text

float 0xC0726CEDC0000000
8float8B+
)
	full_text

float 0xC0B3E1C6A0000000
3float8B&
$
	full_text

float -5.092600e+04
8float8B+
)
	full_text

float 0xC0D1129CC0000000
8float8B+
)
	full_text

float 0xBD00D92000000000
8float8B+
)
	full_text

float 0xBF6373D060000000
8float8B+
)
	full_text

float 0x3DE4116FE0000000
8float8B+
)
	full_text

float 0x3EC178DF40000000
8float8B+
)
	full_text

float 0xBCC3706720000000
8float8B+
)
	full_text

float 0x40E79E7F00000000
8float8B+
)
	full_text

float 0x3C91B3C360000000
8float8B+
)
	full_text

float 0x4010A8F680000000
8float8B+
)
	full_text

float 0x409101D4C0000000
8float8B+
)
	full_text

float 0xBE8F8480A0000000
8float8B+
)
	full_text

float 0xBEB2C3C340000000
8float8B+
)
	full_text

float 0x40075449E0000000
8float8B+
)
	full_text

float 0xBE9DB60E20000000
8float8B+
)
	full_text

float 0x40C6513260000000
8float8B+
)
	full_text

float 0x3D77BD4180000000
8float8B+
)
	full_text

float 0x3DC32540E0000000
8float8B+
)
	full_text

float 0x3DF21BCB80000000
8float8B+
)
	full_text

float 0x40CBF27A80000000
8float8B+
)
	full_text

float 0xC00BD8A960000000
8float8B+
)
	full_text

float 0xBCB7F85EA0000000
8float8B+
)
	full_text

float 0xBE03267920000000
8float8B+
)
	full_text

float 0xBFF3AF3B60000000
8float8B+
)
	full_text

float 0x40CB55EA80000000
8float8B+
)
	full_text

float 0x3DC2A5B400000000
8float8B+
)
	full_text

float 0xBCFC4E7600000000
8float8B+
)
	full_text

float 0xC0D00F3FE0000000
8float8B+
)
	full_text

float 0xBECB3B8080000000
8float8B+
)
	full_text

float 0xC01420DBA0000000
%i648B

	full_text
	
i64 208
8float8B+
)
	full_text

float 0x402140C4E0000000
8float8B+
)
	full_text

float 0xBE0BB876E0000000
8float8B+
)
	full_text

float 0xBFEB2B45A0000000
8float8B+
)
	full_text

float 0xC090CB4DE0000000
8float8B+
)
	full_text

float 0x40111ABD40000000
8float8B+
)
	full_text

float 0xC0E7BDB960000000
8float8B+
)
	full_text

float 0xBD3C5A4680000000
8float8B+
)
	full_text

float 0x3D6D533A80000000
8float8B+
)
	full_text

float 0x400AB2BF60000000
8float8B+
)
	full_text

float 0x3E047F4C00000000
8float8B+
)
	full_text

float 0x4014997920000000
8float8B+
)
	full_text

float 0xBE923B7CA0000000
8float8B+
)
	full_text

float 0x3FF5CF9980000000
8float8B+
)
	full_text

float 0x3DD5268EC0000000
8float8B+
)
	full_text

float 0x3FE43B5E80000000
8float8B+
)
	full_text

float 0xC0D396DCC0000000
8float8B+
)
	full_text

float 0xBEAA2D5400000000
8float8B+
)
	full_text

float 0xBED0967CE0000000
8float8B+
)
	full_text

float 0x40022C50A0000000
8float8B+
)
	full_text

float 0xBF6688C920000000
8float8B+
)
	full_text

float 0x3DA4EF9520000000
8float8B+
)
	full_text

float 0xBE98C5B3E0000000
8float8B+
)
	full_text

float 0x401AEDD4C0000000
8float8B+
)
	full_text

float 0x3EDA170840000000
8float8B+
)
	full_text

float 0x3D77A24400000000
8float8B+
)
	full_text

float 0xC07EA52600000000
8float8B+
)
	full_text

float 0x3FB32977C0000000
$i648B

	full_text


i64 56
8float8B+
)
	full_text

float 0x4008459DE0000000
8float8B+
)
	full_text

float 0xBEAD7BB920000000
8float8B+
)
	full_text

float 0x4030253500000000
%i648B

	full_text
	
i64 240
8float8B+
)
	full_text

float 0xC0E696F360000000
8float8B+
)
	full_text

float 0x3F637B5240000000
8float8B+
)
	full_text

float 0x408DB14580000000
8float8B+
)
	full_text

float 0xBDF60D7F00000000
8float8B+
)
	full_text

float 0xBCD3998DC0000000
8float8B+
)
	full_text

float 0x4002492660000000
8float8B+
)
	full_text

float 0x3F75FE1B00000000
8float8B+
)
	full_text

float 0x3F7E884380000000
8float8B+
)
	full_text

float 0xC05FF54800000000
8float8B+
)
	full_text

float 0x4010119FC0000000
8float8B+
)
	full_text

float 0x3E1CDBB200000000
8float8B+
)
	full_text

float 0xBCF3E714C0000000
8float8B+
)
	full_text

float 0xBE38C0BFC0000000
8float8B+
)
	full_text

float 0x3F7B6CB680000000
8float8B+
)
	full_text

float 0x3F70581760000000
8float8B+
)
	full_text

float 0xBD056475E0000000
8float8B+
)
	full_text

float 0x3F8634A9C0000000
%i648B

	full_text
	
i64 200
8float8B+
)
	full_text

float 0xC0AF57D620000000
8float8B+
)
	full_text

float 0xBCF6ED3FA0000000
8float8B+
)
	full_text

float 0x408CDC9000000000
8float8B+
)
	full_text

float 0xBD002DDB80000000
8float8B+
)
	full_text

float 0xBF5ADD3AE0000000
8float8B+
)
	full_text

float 0x3EADDADAA0000000
8float8B+
)
	full_text

float 0xC0E8A81A20000000
8float8B+
)
	full_text

float 0xC0B34BE2E0000000
8float8B+
)
	full_text

float 0x401A00CE80000000
2float8B%
#
	full_text

float 1.000000e+03
8float8B+
)
	full_text

float 0x40132329A0000000
8float8B+
)
	full_text

float 0xBF31C98640000000
8float8B+
)
	full_text

float 0xBF6125F4E0000000
8float8B+
)
	full_text

float 0x3D6BE0A940000000
8float8B+
)
	full_text

float 0xBE35718E40000000
8float8B+
)
	full_text

float 0x3DB7549E80000000
8float8B+
)
	full_text

float 0xBCE542C280000000
8float8B+
)
	full_text

float 0x3F7DFE6A60000000
8float8B+
)
	full_text

float 0x4002038680000000
8float8B+
)
	full_text

float 0xC0D8E06A40000000
8float8B+
)
	full_text

float 0x3F51D55400000000
8float8B+
)
	full_text

float 0x3F3FD09D40000000
8float8B+
)
	full_text

float 0xBE2B2679E0000000
8float8B+
)
	full_text

float 0xBE1C0DB120000000
8float8B+
)
	full_text

float 0x400FAC71E0000000
8float8B+
)
	full_text

float 0x3F50E56F00000000
8float8B+
)
	full_text

float 0x3FF7E495E0000000
8float8B+
)
	full_text

float 0x3D46D361A0000000
8float8B+
)
	full_text

float 0xBF0689A000000000
8float8B+
)
	full_text

float 0xBCDE995380000000
8float8B+
)
	full_text

float 0x3F81D09720000000
$i648B

	full_text


i64 16
8float8B+
)
	full_text

float 0x3F82142860000000
8float8B+
)
	full_text

float 0xBE395B6420000000
8float8B+
)
	full_text

float 0xC0D9CF3EC0000000
8float8B+
)
	full_text

float 0x40C40352E0000000
8float8B+
)
	full_text

float 0x3DD0852CA0000000
8float8B+
)
	full_text

float 0xC0C91CC280000000
8float8B+
)
	full_text

float 0xC0AC3E2940000000
8float8B+
)
	full_text

float 0x3F72D77340000000
8float8B+
)
	full_text

float 0x3EC7652DA0000000
8float8B+
)
	full_text

float 0x40CC040B00000000
8float8B+
)
	full_text

float 0x4012EAF760000000
8float8B+
)
	full_text

float 0x3F844A1300000000
8float8B+
)
	full_text

float 0x4008224040000000
8float8B+
)
	full_text

float 0x401063AAC0000000
2float8B%
#
	full_text

float 2.500000e+00
8float8B+
)
	full_text

float 0x40909FC640000000
8float8B+
)
	full_text

float 0xBCCAD12160000000
8float8B+
)
	full_text

float 0x3D38F03960000000
8float8B+
)
	full_text

float 0x3F8AA218A0000000
2float8B%
#
	full_text

float 1.000000e+00
8float8B+
)
	full_text

float 0x3DD16223E0000000
8float8B+
)
	full_text

float 0x3D41E69B20000000
8float8B+
)
	full_text

float 0x3DB33164A0000000
8float8B+
)
	full_text

float 0x3DC7FB8EC0000000
%i648B

	full_text
	
i64 152
%i648B

	full_text
	
i64 224
8float8B+
)
	full_text

float 0x400B45C280000000
8float8B+
)
	full_text

float 0xBDDBBA1D20000000
8float8B+
)
	full_text

float 0x40E7CEE540000000
8float8B+
)
	full_text

float 0x3EB2934A60000000
8float8B+
)
	full_text

float 0xBFDC9673E0000000
8float8B+
)
	full_text

float 0xBDA01DC620000000
8float8B+
)
	full_text

float 0xC0D2CB6840000000
8float8B+
)
	full_text

float 0xBFBA9ADBE0000000
8float8B+
)
	full_text

float 0x3DD70DA9C0000000
8float8B+
)
	full_text

float 0x4018AF4D40000000
8float8B+
)
	full_text

float 0x4010E27E80000000
8float8B+
)
	full_text

float 0x4009B321A0000000
8float8B+
)
	full_text

float 0x4010697D00000000
8float8B+
)
	full_text

float 0x400A42A340000000
8float8B+
)
	full_text

float 0x402BE12100000000
8float8B+
)
	full_text

float 0x3F60BBCA20000000
8float8B+
)
	full_text

float 0xBDFF6D7340000000
8float8B+
)
	full_text

float 0x4020F5CC00000000
#i328B

	full_text	

i32 0
8float8B+
)
	full_text

float 0xC02F07D500000000
8float8B+
)
	full_text

float 0xBEB3EB3EA0000000
$i648B

	full_text


i64 64
8float8B+
)
	full_text

float 0x3F5DF40300000000
3float8B&
$
	full_text

float -2.593600e+04
8float8B+
)
	full_text

float 0x3DEC2A6C00000000
8float8B+
)
	full_text

float 0x3E18BBA200000000
8float8B+
)
	full_text

float 0xC0E0E69C00000000
8float8B+
)
	full_text

float 0x3F72668420000000
8float8B+
)
	full_text

float 0x4015F09EA0000000
8float8B+
)
	full_text

float 0xBF588C9B60000000
8float8B+
)
	full_text

float 0xBEBC089BE0000000
8float8B+
)
	full_text

float 0x4002C130A0000000
8float8B+
)
	full_text

float 0x3D37BF8FA0000000
8float8B+
)
	full_text

float 0xC0ADFF2140000000
8float8B+
)
	full_text

float 0xC0AE255060000000
%i648B

	full_text
	
i64 128
%i648B

	full_text
	
i64 192
8float8B+
)
	full_text

float 0x3E3E0722E0000000
8float8B+
)
	full_text

float 0x4017AE7B00000000
8float8B+
)
	full_text

float 0xBCE0F62340000000
8float8B+
)
	full_text

float 0xC0D061E560000000
8float8B+
)
	full_text

float 0xBCF4591FA0000000
8float8B+
)
	full_text

float 0x40D149A540000000
%i648B

	full_text
	
i64 136
8float8B+
)
	full_text

float 0x400C1138E0000000
8float8B+
)
	full_text

float 0xBF744AD200000000
8float8B+
)
	full_text

float 0xC08E94CF00000000
8float8B+
)
	full_text

float 0xBDF639CD40000000
8float8B+
)
	full_text

float 0xBCD3075C60000000
8float8B+
)
	full_text

float 0x3DCAFDC320000000
8float8B+
)
	full_text

float 0xBF6F0244A0000000
8float8B+
)
	full_text

float 0x4001163160000000
%i648B

	full_text
	
i64 184
8float8B+
)
	full_text

float 0xBB4C09FB40000000
8float8B+
)
	full_text

float 0xBF5A930120000000
8float8B+
)
	full_text

float 0x3E86BEE9A0000000
8float8B+
)
	full_text

float 0xC0E7979600000000
$i648B

	full_text


i64 48
8float8B+
)
	full_text

float 0xBEB651C940000000
8float8B+
)
	full_text

float 0xC00F712BE0000000
8float8B+
)
	full_text

float 0x3DDC034F60000000
8float8B+
)
	full_text

float 0xBEA7A2A060000000
8float8B+
)
	full_text

float 0x4017E71600000000
8float8B+
)
	full_text

float 0x4007071880000000
%i648B

	full_text
	
i64 120
8float8B+
)
	full_text

float 0xC0F148D4C0000000
8float8B+
)
	full_text

float 0xBE9EAFDA00000000
%i648B

	full_text
	
i64 144
8float8B+
)
	full_text

float 0x3DC21213E0000000
8float8B+
)
	full_text

float 0xBFDC9673A0000000
8float8B+
)
	full_text

float 0xC0E1057B20000000
8float8B+
)
	full_text

float 0x3F87EC1800000000
8float8B+
)
	full_text

float 0xBCE044C220000000
8float8B+
)
	full_text

float 0xBE27E07860000000
8float8B+
)
	full_text

float 0x3CC526B0A0000000
8float8B+
)
	full_text

float 0xC0D2DFCDC0000000
8float8B+
)
	full_text

float 0x402A4DEA20000000
8float8B+
)
	full_text

float 0x40312C57C0000000
8float8B+
)
	full_text

float 0xBE8657E620000000
8float8B+
)
	full_text

float 0xBCD3852C00000000
8float8B+
)
	full_text

float 0xBC1D1DB540000000
8float8B+
)
	full_text

float 0x3E56A39500000000
#i648B

	full_text	

i64 8
8float8B+
)
	full_text

float 0xBE80F496E0000000
8float8B+
)
	full_text

float 0x400EDC1420000000
8float8B+
)
	full_text

float 0x3D5E584C60000000
8float8B+
)
	full_text

float 0x3F25390F00000000
8float8B+
)
	full_text

float 0x3C0C4B8820000000
8float8B+
)
	full_text

float 0x400D638360000000
8float8B+
)
	full_text

float 0x4002DAAC20000000
8float8B+
)
	full_text

float 0x40062D69C0000000
8float8B+
)
	full_text

float 0x3F644DBE80000000
8float8B+
)
	full_text

float 0x3F72707A60000000
8float8B+
)
	full_text

float 0xC0E6768140000000
$i648B

	full_text


i64 24
8float8B+
)
	full_text

float 0xBCE1809100000000
8float8B+
)
	full_text

float 0x400E427880000000
8float8B+
)
	full_text

float 0x3D69E31600000000
8float8B+
)
	full_text

float 0x3EE1605BC0000000
8float8B+
)
	full_text

float 0x4021056580000000
8float8B+
)
	full_text

float 0xC0F1564700000000
8float8B+
)
	full_text

float 0x3F686B42C0000000
8float8B+
)
	full_text

float 0x3FF0C92F40000000
8float8B+
)
	full_text

float 0x3DB5142E40000000
8float8B+
)
	full_text

float 0x3DE95BDE60000000       	  
 

                     !  "# "$ "" %& %% '( '' )* )+ )) ,- ,, ./ .0 .. 12 11 34 35 33 67 68 66 9: 9; 99 <= <> << ?@ ?? AB AA CD CC EF EG EE HI HH JK JL JJ MN MM OP OQ OO RS RT RR UV UW UU XY XZ XX [\ [[ ]^ ]] _` __ ab ac aa de dd fg fh ff ij ii kl km kk no np nn qr qs qq tu tv tt wx ww yz yy {| {{ }~ } }} € €€ ‚ƒ ‚
„ ‚‚ …† …… ‡ˆ ‡
‰ ‡‡ Š‹ Š
Œ ŠŠ Ž 
  ‘ 
’  “” ““ •– •• —
˜ —— ™š ™
› ™™ œ œœ žŸ ž
  žž ¡¢ ¡¡ £¤ £
¥ ££ ¦§ ¦
¨ ¦¦ ©ª ©
« ©© ¬­ ¬
® ¬¬ ¯° ¯¯ ±² ±± ³
´ ³³ µ¶ µ
· µµ ¸¹ ¸¸ º» º
¼ ºº ½¾ ½½ ¿À ¿
Á ¿¿ ÂÃ Â
Ä ÂÂ ÅÆ Å
Ç ÅÅ ÈÉ È
Ê ÈÈ ËÌ ËË ÍÎ ÍÍ Ï
Ð ÏÏ ÑÒ Ñ
Ó ÑÑ ÔÕ ÔÔ Ö× Ö
Ø ÖÖ ÙÚ ÙÙ ÛÜ Û
Ý ÛÛ Þß Þ
à ÞÞ áâ á
ã áá äå ä
æ ää çè çç éê éé ë
ì ëë íî í
ï íí ðñ ðð òó ò
ô òò õö õõ ÷ø ÷
ù ÷÷ úû ú
ü úú ýþ ý
ÿ ýý € €
‚ €€ ƒ„ ƒƒ …† …… ‡
ˆ ‡‡ ‰Š ‰
‹ ‰‰ Œ ŒŒ Ž Ž
 ŽŽ ‘’ ‘‘ “” “
• ““ –— –
˜ –– ™š ™
› ™™ œ œ
ž œœ Ÿ  ŸŸ ¡¢ ¡¡ £
¤ ££ ¥¦ ¥
§ ¥¥ ¨© ¨¨ ª« ª
¬ ªª ­® ­­ ¯° ¯
± ¯¯ ²³ ²
´ ²² µ¶ µ
· µµ ¸¹ ¸
º ¸¸ »¼ »» ½¾ ½½ ¿
À ¿¿ ÁÂ Á
Ã ÁÁ ÄÅ ÄÄ ÆÇ Æ
È ÆÆ ÉÊ ÉÉ ËÌ Ë
Í ËË ÎÏ Î
Ð ÎÎ ÑÒ Ñ
Ó ÑÑ ÔÕ Ô
Ö ÔÔ ×Ø ×× ÙÚ ÙÙ Û
Ü ÛÛ ÝÞ Ý
ß ÝÝ àá àà âã â
ä ââ åæ åå çè ç
é çç êë ê
ì êê íî í
ï íí ðñ ð
ò ðð óô óó õö õõ ÷
ø ÷÷ ùú ù
û ùù üý üü þÿ þ
€ þþ ‚  ƒ„ ƒ
… ƒƒ †‡ †
ˆ †† ‰Š ‰
‹ ‰‰ Œ Œ
Ž ŒŒ   ‘’ ‘‘ “
” ““ •– •
— •• ˜™ ˜˜ š› š
œ šš ž  Ÿ  Ÿ
¡ ŸŸ ¢£ ¢
¤ ¢¢ ¥¦ ¥
§ ¥¥ ¨© ¨
ª ¨¨ «¬ «« ­® ­­ ¯
° ¯¯ ±² ±
³ ±± ´µ ´´ ¶· ¶
¸ ¶¶ ¹º ¹¹ »¼ »
½ »» ¾¿ ¾
À ¾¾ ÁÂ Á
Ã ÁÁ ÄÅ Ä
Æ ÄÄ ÇÈ ÇÇ ÉÊ ÉÉ Ë
Ì ËË ÍÎ Í
Ï ÍÍ ÐÑ ÐÐ ÒÓ Ò
Ô ÒÒ ÕÖ ÕÕ ×Ø ×
Ù ×× ÚÛ Ú
Ü ÚÚ ÝÞ Ý
ß ÝÝ àá à
â àà ãä ãã åæ åå ç
è çç éê é
ë éé ìí ìì îï î
ð îî ñò ññ óô ó
õ óó ö÷ ö
ø öö ùú ù
û ùù üý ü
þ üü ÿ€ ÿÿ ‚  ƒ
„ ƒƒ …† …
‡ …… ˆ‰ ˆˆ Š‹ Š
Œ ŠŠ Ž   
‘  ’“ ’
” ’’ •– •
— •• ˜™ ˜
š ˜˜ ›œ ›› ž  Ÿ
  ŸŸ ¡¢ ¡
£ ¡¡ ¤¥ ¤¤ ¦§ ¦
¨ ¦¦ ©ª ©© «¬ «
­ «« ®¯ ®
° ®® ±² ±
³ ±± ´µ ´
¶ ´´ ·¸ ·· ¹º ¹¹ »
¼ »» ½¾ ½
¿ ½½ ÀÁ ÀÀ ÂÃ Â
Ä ÂÂ ÅÆ ÅÅ ÇÈ Ç
É ÇÇ ÊË Ê
Ì ÊÊ ÍÎ Í
Ï ÍÍ ÐÑ Ð
Ò ÐÐ ÓÔ ÓÓ ÕÖ ÕÕ ×
Ø ×× ÙÚ Ù
Û ÙÙ ÜÝ ÜÜ Þß Þ
à ÞÞ áâ áá ãä ã
å ãã æç æ
è ææ éê é
ë éé ìí ì
î ìì ïð ïï ñò ññ ó
ô óó õö õ
÷ õõ øù øø úû ú
ü úú ýþ ýý ÿ€ ÿ
 ÿÿ ‚ƒ ‚
„ ‚‚ …† …
‡ …… ˆ‰ ˆ
Š ˆˆ ‹Œ ‹‹ Ž  
  ‘’ ‘
“ ‘‘ ”• ”” –— –
˜ –– ™š ™™ ›œ ›
 ›› žŸ ž
  žž ¡¢ ¡
£ ¡¡ ¤¥ ¤
¦ ¤¤ §¨ §§ ©ª ©© «
¬ «« ­® ­
¯ ­­ °± °° ²³ ²
´ ²² µ¶ µµ ·¸ ·
¹ ·· º» º
¼ ºº ½¾ ½
¿ ½½ ÀÁ À
Â ÀÀ ÃÄ ÃÃ ÅÆ ÅÅ Ç
È ÇÇ ÉÊ É
Ë ÉÉ ÌÍ ÌÌ ÎÏ Î
Ð ÎÎ ÑÒ ÑÑ ÓÔ Ó
Õ ÓÓ Ö× Ö
Ø ÖÖ ÙÚ Ù
Û ÙÙ ÜÝ Ü
Þ ÜÜ ßà ßß áâ áá ã
ä ãã åæ å
ç åå èé èè êë ê
ì êê íî íí ïð ï
ñ ïï òó ò
ô òò õö õ
÷ õõ øù ø
ú øø ûü ûû ýþ ýý ÿ
€ ÿÿ ‚ 
ƒ  „… „„ †‡ †
ˆ †† ‰Š ‰‰ ‹Œ ‹
 ‹‹ Ž Ž
 ŽŽ ‘’ ‘
“ ‘‘ ”• ”
– ”” —˜ —— ™š ™™ ›
œ ›› ž 
Ÿ   ¡    ¢£ ¢
¤ ¢¢ ¥¦ ¥¥ §¨ §
© §§ ª« ª
¬ ªª ­® ­
¯ ­­ °± °
² °° ³´ ³³ µ¶ µµ ·
¸ ·· ¹º ¹
» ¹¹ ¼½ ¼¼ ¾¿ ¾
À ¾¾ ÁÂ ÁÁ ÃÄ Ã
Å ÃÃ ÆÇ Æ
È ÆÆ ÉÊ É
Ë ÉÉ ÌÍ Ì
Î ÌÌ ÏÐ ÏÏ ÑÒ ÑÑ Ó
Ô ÓÓ ÕÖ Õ
× ÕÕ ØÙ ØØ ÚÛ Ú
Ü ÚÚ ÝÞ ÝÝ ßà ß
á ßß âã â
ä ââ åç ææ èé è
ê èè ëì ëë íî í
ï íí ðñ ð
ò ðð óô ó
õ óó ö÷ ö
ø öö ùú ùù û
ü ûû ýþ ý
ÿ ýý € €€ ‚ƒ ‚
„ ‚‚ …† …… ‡ˆ ‡
‰ ‡‡ Š‹ Š
Œ ŠŠ Ž 
  ‘ 
’  “” ““ •– •• —
˜ —— ™š ™
› ™™ œ œœ žŸ ž
  žž ¡¢ ¡¡ £¤ £
¥ ££ ¦§ ¦
¨ ¦¦ ©ª ©
« ©© ¬­ ¬
® ¬¬ ¯° ¯¯ ±² ±± ³
´ ³³ µ¶ µ
· µµ ¸¹ ¸¸ º» º
¼ ºº ½¾ ½½ ¿À ¿
Á ¿¿ ÂÃ Â
Ä ÂÂ ÅÆ Å
Ç ÅÅ ÈÉ È
Ê ÈÈ ËÌ ËË ÍÎ ÍÍ Ï
Ð ÏÏ ÑÒ Ñ
Ó ÑÑ ÔÕ ÔÔ Ö× Ö
Ø ÖÖ ÙÚ ÙÙ ÛÜ Û
Ý ÛÛ Þß Þ
à ÞÞ áâ á
ã áá äå ä
æ ää çè çç éê éé ë
ì ëë íî í
ï íí ðñ ðð òó ò
ô òò õö õõ ÷ø ÷
ù ÷÷ úû ú
ü úú ýþ ý
ÿ ýý € €
‚ €€ ƒ„ ƒƒ …† …… ‡
ˆ ‡‡ ‰Š ‰
‹ ‰‰ Œ ŒŒ Ž Ž
 ŽŽ ‘’ ‘‘ “” “
• ““ –— –
˜ –– ™š ™
› ™™ œ œ
ž œœ Ÿ  ŸŸ ¡¢ ¡¡ £
¤ ££ ¥¦ ¥
§ ¥¥ ¨© ¨¨ ª« ª
¬ ªª ­® ­­ ¯° ¯
± ¯¯ ²³ ²
´ ²² µ¶ µ
· µµ ¸¹ ¸
º ¸¸ »¼ »» ½¾ ½½ ¿
À ¿¿ ÁÂ Á
Ã ÁÁ ÄÅ ÄÄ ÆÇ Æ
È ÆÆ ÉÊ ÉÉ ËÌ Ë
Í ËË ÎÏ Î
Ð ÎÎ ÑÒ Ñ
Ó ÑÑ ÔÕ Ô
Ö ÔÔ ×Ø ×× ÙÚ ÙÙ Û
Ü ÛÛ ÝÞ Ý
ß ÝÝ àá àà âã â
ä ââ åæ åå çè ç
é çç êë ê
ì êê íî í
ï íí ðñ ð
ò ðð óô óó õö õõ ÷
ø ÷÷ ùú ù
û ùù üý üü þÿ þ
€	 þþ 	‚	 		 ƒ	„	 ƒ	
…	 ƒ	ƒ	 †	‡	 †	
ˆ	 †	†	 ‰	Š	 ‰	
‹	 ‰	‰	 Œ		 Œ	
Ž	 Œ	Œ	 		 		 ‘	’	 ‘	‘	 “	
”	 “	“	 •	–	 •	
—	 •	•	 ˜	™	 ˜	˜	 š	›	 š	
œ	 š	š	 	ž	 		 Ÿ	 	 Ÿ	
¡	 Ÿ	Ÿ	 ¢	£	 ¢	
¤	 ¢	¢	 ¥	¦	 ¥	
§	 ¥	¥	 ¨	©	 ¨	
ª	 ¨	¨	 «	¬	 «	«	 ­	®	 ­	­	 ¯	
°	 ¯	¯	 ±	²	 ±	
³	 ±	±	 ´	µ	 ´	´	 ¶	·	 ¶	
¸	 ¶	¶	 ¹	º	 ¹	¹	 »	¼	 »	
½	 »	»	 ¾	¿	 ¾	
À	 ¾	¾	 Á	Â	 Á	
Ã	 Á	Á	 Ä	Å	 Ä	
Æ	 Ä	Ä	 Ç	È	 Ç	Ç	 É	Ê	 É	É	 Ë	
Ì	 Ë	Ë	 Í	Î	 Í	
Ï	 Í	Í	 Ð	Ñ	 Ð	Ð	 Ò	Ó	 Ò	
Ô	 Ò	Ò	 Õ	Ö	 Õ	Õ	 ×	Ø	 ×	
Ù	 ×	×	 Ú	Û	 Ú	
Ü	 Ú	Ú	 Ý	Þ	 Ý	
ß	 Ý	Ý	 à	á	 à	
â	 à	à	 ã	ä	 ã	ã	 å	æ	 å	å	 ç	
è	 ç	ç	 é	ê	 é	
ë	 é	é	 ì	í	 ì	ì	 î	ï	 î	
ð	 î	î	 ñ	ò	 ñ	ñ	 ó	ô	 ó	
õ	 ó	ó	 ö	÷	 ö	
ø	 ö	ö	 ù	ú	 ù	
û	 ù	ù	 ü	ý	 ü	
þ	 ü	ü	 ÿ	€
 ÿ	ÿ	 
‚
 

 ƒ

„
 ƒ
ƒ
 …
†
 …

‡
 …
…
 ˆ
‰
 ˆ
ˆ
 Š
‹
 Š

Œ
 Š
Š
 
Ž
 

 

 

‘
 

 ’
“
 ’

”
 ’
’
 •
–
 •

—
 •
•
 ˜
™
 ˜

š
 ˜
˜
 ›
œ
 ›
›
 
ž
 

 Ÿ

 
 Ÿ
Ÿ
 ¡
¢
 ¡

£
 ¡
¡
 ¤
¥
 ¤
¤
 ¦
§
 ¦

¨
 ¦
¦
 ©
ª
 ©
©
 «
¬
 «

­
 «
«
 ®
¯
 ®

°
 ®
®
 ±
²
 ±

³
 ±
±
 ´
µ
 ´

¶
 ´
´
 ·
¸
 ·
·
 ¹
º
 ¹
¹
 »

¼
 »
»
 ½
¾
 ½

¿
 ½
½
 À
Á
 À
À
 Â
Ã
 Â

Ä
 Â
Â
 Å
Æ
 Å
Å
 Ç
È
 Ç

É
 Ç
Ç
 Ê
Ë
 Ê

Ì
 Ê
Ê
 Í
Î
 Í

Ï
 Í
Í
 Ð
Ñ
 Ð

Ò
 Ð
Ð
 Ó
Ô
 Ó
Ó
 Õ
Ö
 Õ
Õ
 ×

Ø
 ×
×
 Ù
Ú
 Ù

Û
 Ù
Ù
 Ü
Ý
 Ü
Ü
 Þ
ß
 Þ

à
 Þ
Þ
 á
â
 á
á
 ã
ä
 ã

å
 ã
ã
 æ
ç
 æ

è
 æ
æ
 é
ê
 é

ë
 é
é
 ì
í
 ì

î
 ì
ì
 ï
ð
 ï
ï
 ñ
ò
 ñ
ñ
 ó

ô
 ó
ó
 õ
ö
 õ

÷
 õ
õ
 ø
ù
 ø
ø
 ú
û
 ú

ü
 ú
ú
 ý
þ
 ý
ý
 ÿ
€ ÿ

 ÿ
ÿ
 ‚ƒ ‚
„ ‚‚ …† …
‡ …… ˆ‰ ˆ
Š ˆˆ ‹Œ ‹‹ Ž  
  ‘’ ‘
“ ‘‘ ”• ”” –— –
˜ –– ™š ™™ ›œ ›
 ›› žŸ ž
  žž ¡¢ ¡
£ ¡¡ ¤¥ ¤
¦ ¤¤ §¨ §§ ©ª ©© «
¬ «« ­® ­
¯ ­­ °± °° ²³ ²
´ ²² µ¶ µµ ·¸ ·
¹ ·· º» º
¼ ºº ½¾ ½
¿ ½½ ÀÁ À
Â ÀÀ ÃÄ ÃÃ ÅÆ ÅÅ Ç
È ÇÇ ÉÊ É
Ë ÉÉ ÌÍ ÌÌ ÎÏ Î
Ð ÎÎ ÑÒ ÑÑ ÓÔ Ó
Õ ÓÓ Ö× Ö
Ø ÖÖ ÙÚ Ù
Û ÙÙ ÜÝ Ü
Þ ÜÜ ßà ßß áâ áá ã
ä ãã åæ å
ç åå èé èè êë ê
ì êê íî íí ïð ï
ñ ïï òó ò
ô òò õö õ
÷ õõ øù ø
ú øø ûü ûû ýþ ýý ÿ
€ ÿÿ ‚ 
ƒ  „… „„ †‡ †
ˆ †† ‰Š ‰‰ ‹Œ ‹
 ‹‹ Ž Ž
 ŽŽ ‘’ ‘
“ ‘‘ ”• ”
– ”” —˜ —— ™š ™™ ›
œ ›› ž 
Ÿ   ¡    ¢£ ¢
¤ ¢¢ ¥¦ ¥¥ §¨ §
© §§ ª« ª
¬ ªª ­® ­
¯ ­­ °± °
² °° ³´ ³³ µ¶ µµ ·
¸ ·· ¹º ¹
» ¹¹ ¼½ ¼¼ ¾¿ ¾
À ¾¾ ÁÂ ÁÁ ÃÄ Ã
Å ÃÃ ÆÇ Æ
È ÆÆ ÉÊ É
Ë ÉÉ ÌÍ Ì
Î ÌÌ ÏÐ ÏÏ ÑÒ ÑÑ Ó
Ô ÓÓ ÕÖ Õ
× ÕÕ ØÙ ØØ ÚÛ Ú
Ü ÚÚ ÝÞ ÝÝ ßà ß
á ßß âã â
ä ââ åæ å
ç åå èé è
ê èè ëì ëë íî íí ï
ð ïï ñò ñ
ó ññ ôõ ôô ö÷ ö
ø öö ùú ùù ûü û
ý ûû þÿ þ
€ þþ ‚ 
ƒ  „… „
† „„ ‡ˆ ‡‡ ‰Š ‰‰ ‹
Œ ‹‹ Ž 
  ‘  ’“ ’
” ’’ •– •• —˜ —
™ —— š› š
œ šš ž 
Ÿ   ¡  
¢    £¤ ££ ¥¦ ¥¥ §
¨ §§ ©ª ©
« ©© ¬­ ¬¬ ®¯ ®
° ®® ±² ±± ³´ ³
µ ³³ ¶· ¶
¸ ¶¶ ¹» º
¼ ºº ½¾ ½
¿ ½½ ÀÁ À
Â ÀÀ ÃÄ Ã
Å ÃÃ ÆÇ ÆÆ ÈÉ ÈÈ Ê
Ë ÊÊ ÌÍ Ì
Î ÌÌ ÏÐ 	Ñ Ò 'Ò CÒ _Ò {Ò —Ò ³Ò ÏÒ ëÒ ‡Ò £Ò ¿Ò ÛÒ ÷Ò “Ò ¯Ò ËÒ çÒ ƒÒ ŸÒ »Ò ×Ò óÒ Ò «Ò ÇÒ ãÒ ÿÒ ›Ò ·Ò ÓÒ ûÒ —Ò ³Ò ÏÒ ëÒ ‡Ò £Ò ¿Ò ÛÒ ÷Ò “	Ò ¯	Ò Ë	Ò ç	Ò ƒ
Ò Ÿ
Ò »
Ò ×
Ò ó
Ò Ò «Ò ÇÒ ãÒ ÿÒ ›Ò ·Ò ÓÒ ïÒ ‹Ò §Ò Ê    	    
           ! # $" & (% *' +
 - /, 0 21 4 53 7 8 :6 ;. =9 >< @ BA D? FC G
 I KH L NM P QO S T VR WJ YU ZX \ ^] `[ b_ c
 e gd h ji l mk o p rn sf uq vt x zy |w ~{ 
  ƒ€ „ †… ˆ ‰‡ ‹ Œ ŽŠ ‚ ‘ ’ ” –• ˜“ š— ›
  Ÿœ   ¢¡ ¤ ¥£ § ¨ ª¦ «ž ­© ®¬ ° ²± ´¯ ¶³ ·
 ¹ »¸ ¼ ¾½ À Á¿ Ã Ä ÆÂ Çº ÉÅ ÊÈ Ì ÎÍ ÐË ÒÏ Ó
 Õ ×Ô Ø ÚÙ Ü ÝÛ ß à âÞ ãÖ åá æä è êé ìç îë ï
 ñ óð ô öõ ø ù÷ û ü þú ÿò ý ‚€ „ †… ˆƒ Š‡ ‹
  Œ  ’‘ ” •“ — ˜ š– ›Ž ™ žœ   ¢¡ ¤Ÿ ¦£ §
 © «¨ ¬ ®­ ° ±¯ ³ ´ ¶² ·ª ¹µ º¸ ¼ ¾½ À» Â¿ Ã
 Å ÇÄ È ÊÉ Ì ÍË Ï Ð ÒÎ ÓÆ ÕÑ ÖÔ Ø ÚÙ Ü× ÞÛ ß
 á ãà ä æå è éç ë ì îê ïâ ñí òð ô öõ øó ú÷ û
 ý ÿü € ‚ „ …ƒ ‡ ˆ Š† ‹þ ‰ ŽŒ  ’‘ ” –“ —
 ™ ›˜ œ ž   ¡Ÿ £ ¤ ¦¢ §š ©¥ ª¨ ¬ ®­ °« ²¯ ³
 µ ·´ ¸ º¹ ¼ ½» ¿ À Â¾ Ã¶ ÅÁ ÆÄ È ÊÉ ÌÇ ÎË Ï
 Ñ ÓÐ Ô ÖÕ Ø Ù× Û Ü ÞÚ ßÒ áÝ âà ä æå èã êç ë
 í ïì ð òñ ô õó ÷ ø úö ûî ýù þü € ‚ „ÿ †ƒ ‡
 ‰ ‹ˆ Œ Ž  ‘ “ ” –’ —Š ™• š˜ œ ž  › ¢Ÿ £
 ¥ §¤ ¨ ª© ¬ ­« ¯ ° ²® ³¦ µ± ¶´ ¸ º¹ ¼· ¾» ¿
 Á ÃÀ Ä ÆÅ È ÉÇ Ë Ì ÎÊ ÏÂ ÑÍ ÒÐ Ô ÖÕ ØÓ Ú× Û
 Ý ßÜ à âá ä åã ç è êæ ëÞ íé îì ð òñ ôï öó ÷
 ù ûø ü þý € ÿ ƒ „ †‚ ‡ú ‰… Šˆ Œ Ž ‹ ’ “
 • —” ˜ š™ œ › Ÿ   ¢ž £– ¥¡ ¦¤ ¨ ª© ¬§ ®« ¯
 ± ³° ´ ¶µ ¸ ¹· » ¼ ¾º ¿² Á½ ÂÀ Ä ÆÅ ÈÃ ÊÇ Ë
 Í ÏÌ Ð ÒÑ Ô ÕÓ × Ø ÚÖ ÛÎ ÝÙ ÞÜ à âá äß æã ç
 é ëè ì îí ð ñï ó ô öò ÷ê ùõ úø ü þý €û ‚ÿ ƒ
 … ‡„ ˆ Š‰ Œ ‹   ’Ž “† •‘ –” ˜ š™ œ— ž› Ÿ
 ¡ £  ¤ ¦¥ ¨ ©§ « ¬ ®ª ¯¢ ±­ ²° ´ ¶µ ¸³ º· »
 ½ ¿¼ À ÂÁ Ä ÅÃ Ç È ÊÆ Ë¾ ÍÉ ÎÌ Ð ÒÑ ÔÏ ÖÓ ×
 Ù ÛØ Ü ÞÝ à áß ã ä
 ç éæ ê ìë î ïí ñ ò ôð õè ÷ó øö ú üù þû ÿ
  ƒ€ „ †… ˆ ‰‡ ‹ Œ ŽŠ ‚ ‘ ’ ” –• ˜“ š— ›
  Ÿœ   ¢¡ ¤ ¥£ § ¨ ª¦ «ž ­© ®¬ ° ²± ´¯ ¶³ ·
 ¹ »¸ ¼ ¾½ À Á¿ Ã Ä ÆÂ Çº ÉÅ ÊÈ Ì ÎÍ ÐË ÒÏ Ó
 Õ ×Ô Ø ÚÙ Ü ÝÛ ß à âÞ ãÖ åá æä è êé ìç îë ï
 ñ óð ô öõ ø ù÷ û ü þú ÿò ý ‚€ „ †… ˆƒ Š‡ ‹
  Œ  ’‘ ” •“ — ˜ š– ›Ž ™ žœ   ¢¡ ¤Ÿ ¦£ §
 © «¨ ¬ ®­ ° ±¯ ³ ´ ¶² ·ª ¹µ º¸ ¼ ¾½ À» Â¿ Ã
 Å ÇÄ È ÊÉ Ì ÍË Ï Ð ÒÎ ÓÆ ÕÑ ÖÔ Ø ÚÙ Ü× ÞÛ ß
 á ãà ä æå è éç ë ì îê ïâ ñí òð ô öõ øó ú÷ û
 ý ÿü €	 ‚		 „	 …	ƒ	 ‡	 ˆ	 Š	†	 ‹	þ 	‰	 Ž	Œ	 	 ’	‘	 ”		 –	“	 —	
 ™	 ›	˜	 œ	 ž		  	 ¡	Ÿ	 £	 ¤	 ¦	¢	 §	š	 ©	¥	 ª	¨	 ¬	 ®	­	 °	«	 ²	¯	 ³	
 µ	 ·	´	 ¸	 º	¹	 ¼	 ½	»	 ¿	 À	 Â	¾	 Ã	¶	 Å	Á	 Æ	Ä	 È	 Ê	É	 Ì	Ç	 Î	Ë	 Ï	
 Ñ	 Ó	Ð	 Ô	 Ö	Õ	 Ø	 Ù	×	 Û	 Ü	 Þ	Ú	 ß	Ò	 á	Ý	 â	à	 ä	 æ	å	 è	ã	 ê	ç	 ë	
 í	 ï	ì	 ð	 ò	ñ	 ô	 õ	ó	 ÷	 ø	 ú	ö	 û	î	 ý	ù	 þ	ü	 €
 ‚

 „
ÿ	 †
ƒ
 ‡

 ‰
 ‹
ˆ
 Œ
 Ž

 
 ‘

 “
 ”
 –
’
 —
Š
 ™
•
 š
˜
 œ
 ž

  
›
 ¢
Ÿ
 £

 ¥
 §
¤
 ¨
 ª
©
 ¬
 ­
«
 ¯
 °
 ²
®
 ³
¦
 µ
±
 ¶
´
 ¸
 º
¹
 ¼
·
 ¾
»
 ¿

 Á
 Ã
À
 Ä
 Æ
Å
 È
 É
Ç
 Ë
 Ì
 Î
Ê
 Ï
Â
 Ñ
Í
 Ò
Ð
 Ô
 Ö
Õ
 Ø
Ó
 Ú
×
 Û

 Ý
 ß
Ü
 à
 â
á
 ä
 å
ã
 ç
 è
 ê
æ
 ë
Þ
 í
é
 î
ì
 ð
 ò
ñ
 ô
ï
 ö
ó
 ÷

 ù
 û
ø
 ü
 þ
ý
 € ÿ
 ƒ „ †‚ ‡ú
 ‰… Šˆ Œ Ž ‹ ’ “
 • —” ˜ š™ œ › Ÿ   ¢ž £– ¥¡ ¦¤ ¨ ª© ¬§ ®« ¯
 ± ³° ´ ¶µ ¸ ¹· » ¼ ¾º ¿² Á½ ÂÀ Ä ÆÅ ÈÃ ÊÇ Ë
 Í ÏÌ Ð ÒÑ Ô ÕÓ × Ø ÚÖ ÛÎ ÝÙ ÞÜ à âá äß æã ç
 é ëè ì îí ð ñï ó ô öò ÷ê ùõ úø ü þý €û ‚ÿ ƒ
 … ‡„ ˆ Š‰ Œ ‹   ’Ž “† •‘ –” ˜ š™ œ— ž› Ÿ
 ¡ £  ¤ ¦¥ ¨ ©§ « ¬ ®ª ¯¢ ±­ ²° ´ ¶µ ¸³ º· »
 ½ ¿¼ À ÂÁ Ä ÅÃ Ç È ÊÆ Ë¾ ÍÉ ÎÌ Ð ÒÑ ÔÏ ÖÓ ×
 Ù ÛØ Ü ÞÝ à áß ã ä æâ çÚ éå êè ì îí ðë òï ó
 õ ÷ô ø úù ü ýû ÿ € ‚þ ƒö … †„ ˆ Š‰ Œ‡ Ž‹ 
 ‘ “ ” –• ˜ ™— › œ žš Ÿ’ ¡ ¢  ¤ ¦¥ ¨£ ª§ «
 ­ ¯¬ ° ²± ´ µ³ · ¸¶ »â ¼® ¾Ú ¿ Áº Â½ ÄÀ ÅÃ Ç ÉÈ ËÆ ÍÊ Î  æå º¹ º ÕÕ ÓÓ Ï ÖÖ ÔÔÖ ÖÖ Öƒ ÕÕ ƒº ÖÖ º¹	 ÖÖ ¹	› ÕÕ ›Ö ÖÖ ÖÞ ÖÖ ÞÒ	 ÖÖ Ò	À ÖÖ À‰ ÖÖ ‰ ÖÖ H ÖÖ HR ÖÖ Rî ÖÖ î‘ ÖÖ ‘² ÖÖ ²† ÖÖ †§ ÖÖ §€ ÖÖ €Ð	 ÖÖ Ð	É ÖÖ Éê ÖÖ êí ÖÖ í… ÖÖ …° ÖÖ °Ã ÖÖ ÃÝ ÖÖ ÝŽ ÖÖ Žë ÖÖ ëò ÖÖ òª ÖÖ ª— ÕÕ —Å
 ÖÖ Å
† ÖÖ †ƒ ÖÖ ƒÐ ÖÖ ÐÓ ÖÖ Óþ ÖÖ þ« ÖÖ «Ô ÖÖ Ôï ÕÕ ïË ÕÕ Ë§ ÕÕ §“ ÖÖ “ð ÖÖ ð¼ ÖÖ ¼ ÖÖ ¾ ÖÖ ¾ú
 ÖÖ ú
Ô ÖÖ Ôª ÖÖ ªÄ ÖÖ ÄÓ ÕÕ Ó ÖÖ ¥ ÖÖ ¥Œ ÖÖ ŒÝ ÖÖ Ý× ÕÕ ×ç ÖÖ çÛ ÖÖ ÛÏ ÕÕ Ïà ÖÖ àê ÖÖ êš	 ÖÖ š	¶ ÖÖ ¶û ÕÕ û° ÖÖ °‚ ÖÖ ‚ ÖÖ Ë ÖÖ Ë®
 ÖÖ ®
æ
 ÖÖ æ
™ ÖÖ ™â ÖÖ âø ÖÖ ø÷ ÖÖ ÷º ÖÖ º¦ ÖÖ ¦¸ ÖÖ ¸¯ ÖÖ ¯×	 ÖÖ ×	 ÔÔ 
 ÖÖ 
ª ÖÖ ª» ÕÕ »» ÖÖ »Š ÖÖ Š² ÖÖ ²¶	 ÖÖ ¶	§ ÕÕ §ñ ÖÖ ñ” ÖÖ ”© ÖÖ ©Ì ÖÖ Ì• ÖÖ •d ÖÖ d ÖÖ ž ÖÖ žó ÖÖ óÎ ÖÖ Î
 ÖÖ 
1 ÖÖ 1ì ÖÖ ì¤ ÖÖ ¤ã	 ÕÕ ã	›
 ÕÕ ›
™ ÖÖ ™Ø ÖÖ Øô ÖÖ ôö ÖÖ öè ÖÖ èŸ ÕÕ Ÿþ ÖÖ þ¬ ÖÖ ¬Ç ÕÕ ÇŠ ÖÖ Šœ ÖÖ œÒ ÖÖ ÒÂ ÖÖ Âá
 ÖÖ á
ø
 ÖÖ ø
– ÖÖ –ê ÖÖ ê· ÖÖ ·ã ÖÖ ã¢ ÖÖ ¢Â ÖÖ Â ÖÖ Á ÖÖ ÁÖ ÖÖ Öµ ÖÖ µÇ	 ÕÕ Ç	Ÿ ÖÖ Ÿ¥ ÖÖ ¥Ú ÖÖ Úž ÖÖ žì	 ÖÖ ì	– ÖÖ –ü ÖÖ ü? ÕÕ ?“ ÕÕ “þ ÖÖ þ… ÖÖ …¼ ÖÖ ¼Ç
 ÖÖ Ç
Ù ÖÖ ÙØ ÖÖ Øö	 ÖÖ ö	ù ÕÕ ùï ÖÖ ïÿ ÕÕ ÿÜ ÖÖ ÜÑ ÖÖ Ñ¹ ÖÖ ¹– ÖÖ –¡ ÖÖ ¡Æ ÖÖ Æƒ ÕÕ ƒ¨ ÖÖ ¨» ÕÕ »£ ÖÖ £¿ ÖÖ ¿– ÖÖ –Ä ÖÖ Äâ ÖÖ âæ ÖÖ æÞ ÖÖ ÞÀ
 ÖÖ À
‹ ÕÕ ‹ã
 ÖÖ ã
Ÿ	 ÖÖ Ÿ	‹ ÕÕ ‹ß ÖÖ ß£ ÕÕ £ò ÖÖ ò· ÕÕ ·ú ÖÖ ú‹ ÖÖ ‹æ ÖÖ æÂ ÖÖ Âÿ ÖÖ ÿã ÕÕ ãÖ ÖÖ Öç ÕÕ çË ÖÖ Ë› ÖÖ ›ª ÖÖ ª ÖÖ ý
 ÖÖ ý
ß ÕÕ ßí ÖÖ íš ÖÖ š® ÖÖ ®ž ÖÖ ž	 ÖÖ 	³ ÕÕ ³¤
 ÖÖ ¤
ç ÕÕ çÞ ÖÖ Þ’ ÖÖ ’ü ÖÖ üŽ ÖÖ Žó ÕÕ ó«	 ÕÕ «	»	 ÖÖ »	 ÖÖ Ú ÖÖ Ú¡ ÖÖ ¡ð ÖÖ ðß ÕÕ ßÚ	 ÖÖ Ú	ó	 ÖÖ ó	í ÖÖ í½ ÖÖ ½Ê
 ÖÖ Ê
Ÿ ÕÕ Ÿœ ÖÖ œú ÖÖ ú˜	 ÖÖ ˜	¢	 ÖÖ ¢	Ñ ÖÖ Ñ­ ÖÖ ­ò ÖÖ òŽ ÖÖ ŽÆ ÖÖ Æ­ ÖÖ ­õ ÖÖ õ	 ÖÖ 	Ó
 ÕÕ Ó
 ÕÕ Ž ÖÖ Ž‘ ÖÖ ‘J ÖÖ JÌ ÖÖ Ì‡ ÖÖ ‡× ÖÖ ×¦
 ÖÖ ¦
†	 ÖÖ †	Æ ÖÖ Æõ ÖÖ õ ÖÖ ž ÖÖ ž ÓÓ Œ ÖÖ Œÿ
 ÖÖ ÿ
¸ ÖÖ ¸«
 ÖÖ «
  ÖÖ  è ÖÖ è— ÕÕ —û ÖÖ û“ ÕÕ “Ë ÕÕ Ë’ ÖÖ ’´	 ÖÖ ´	Î ÖÖ Î³ ÕÕ ³Ã ÕÕ Ãß ÖÖ ßà ÖÖ à² ÖÖ ²É ÖÖ ÉÕ	 ÖÖ Õ	Õ ÖÖ Õî	 ÖÖ î	ÿ	 ÕÕ ÿ	Æ ÖÖ Æ©
 ÖÖ ©
n ÖÖ n± ÖÖ ±ó ÕÕ ó¨ ÖÖ ¨Á ÖÖ ÁM ÖÖ M² ÖÖ ²Ó ÖÖ ÓÛ ÖÖ Û£ ÖÖ £ð ÖÖ ð„ ÖÖ „Î ÖÖ Î§ ÖÖ §‹ ÖÖ ‹·
 ÕÕ ·
Ã ÖÖ Ã€ ÖÖ €× ÕÕ ×Â
 ÖÖ Â
Ü
 ÖÖ Ü
% ÕÕ %û ÕÕ ûº ÖÖ ºˆ ÖÖ ˆÊ ÖÖ Êú ÖÖ ú¯ ÕÕ ¯Ç ÖÖ Çå ÖÖ åÅ ÖÖ Åi ÖÖ i3 ÖÖ 3ç ÖÖ çñ	 ÖÖ ñ	Š
 ÖÖ Š
  ÖÖ  “ ÖÖ “, ÖÖ ,á ÖÖ áº ÖÖ º„ ÖÖ „‚ ÖÖ ‚› ÖÖ ›[ ÕÕ [† ÖÖ †w ÕÕ wâ ÖÖ â® ÖÖ ®’
 ÖÖ ’
¢ ÖÖ ¢« ÕÕ «” ÖÖ ”µ ÖÖ µê ÖÖ êë ÕÕ ëÆ ÕÕ Æ· ÖÖ ·ý ÖÖ ýƒ	 ÖÖ ƒ	k ÖÖ kŠ ÖÖ Š÷ ÖÖ ÷¦ ÖÖ ¦¯ ÖÖ ¯‡ ÖÖ ‡´ ÖÖ ´â ÖÖ â¯ ÕÕ ¯Î ÖÖ ÎÞ
 ÖÖ Þ
O ÖÖ O³ ÖÖ ³ˆ
 ÖÖ ˆ
‚ ÖÖ ‚‚ ÖÖ ‚Ï ÕÕ Ï‡ ÕÕ ‡ù ÖÖ ùå ÖÖ åÚ ÖÖ Ú¢ ÖÖ ¢¾ ÖÖ ¾¦ ÖÖ ¦Ù ÖÖ Ùš ÖÖ šï
 ÕÕ ï
˜ ÖÖ ˜6 ÖÖ 6¾	 ÖÖ ¾	— ÖÖ —. ÖÖ .	 ÕÕ 	¶ ÖÖ ¶è ÖÖ è ÖÖ ò ÖÖ ò¾ ÖÖ ¾f ÖÖ fö ÖÖ öÃ ÕÕ Ã‰ ÖÖ ‰½ ÖÖ ½ï ÖÖ ï¿ ÖÖ ¿
× –
Ø Ž
Ù ›
Ú œ
Û ¤
Ü Î
Ý Å

Þ Ì
ß Å

à ‰
á ‚
â ­
â 

ã á
	ä 6
å Š
æ ‰
ç ³
è ¡
è õ
é ÷
ê ²
ë ß
ì à
í þ
î Ê

ï ê
ð á

ñ ×
ò Ÿ	
ó „
ô 

õ Ä
ö ã

÷ µ	ø n
ù ¿
ú Ü
û ¹
ü ü
ý þ
þ á
ÿ æ
€ £
 ”
‚ ž
ƒ Ù
„ Æ
… ¡
† ª
‡ ö
ˆ  
‰ ¢
Š ±
Š …
‹ õ
‹ É	
Œ  
 ˆ

Ž 
 Ò
 À
‘ ½
‘ ‘	
’ ì	
“ †	
” ¼
• Î
– ‘
— â
˜ Ú	
™ ¬
š •
š é
› ¹	
œ ¦
 Ì	ž 
Ÿ Ý
  œ
¡ Þ
¢ ¯
£ †
¤ —
¥ Å
¦ Ê
§ ú

¨ Š
© Þ

ª à
« ñ
« Å
¬ ´
­ Ú
® ¾	
¯ ç
° 

± Á
² §
³ Ù	´ 
µ É
¶ “
· è
¸ 
¹ ø
º „
» Â
¼ 	
½ õ
¾ ¨
¿ Ø
À ¥
Á ¼
Â ‰
Ã ÷
Ä ¼
Å ë
Æ ¡
Ç ¸
È Â

É §
Ê ¸
Ë ­
Ì Ì	Í J
Î •
Ï µ
Ð  
Ñ ý

Ò ²
Ó ú
Ô ™
Ô í
Õ ð
Ö ÿ

× Å
Ø Ã
Ù ¡
Ú Ž
Û ‘
Ü ¸
Ý ¡
Þ ï
ß ç
à ª
á 
á á
â ª
ã Õ	
ä ê
å ¦

æ Û
ç ú
è »	
é Œ
ê ü
ë Ñ
ì ±
í Ñ
í ¥
î Ö
ï 

ð ž
ñ Õ	
ò Ð
ó Ý
ô ™
õ œ
ö Ù
ö ­	
÷ Ñ
ø ‘	ù 1
ú ·
û Ã	ü 	ý H
þ 	
ÿ œ
€ ©

 ‚
‚ Ó	ƒ d
„ ì
… ¤

† Ô
‡ ƒ	
ˆ ‘
ˆ å		‰ 
Š ¬
‹ €
Œ è
 ¥
Ž ò
 ¿
 õ
‘ Ò	
’ •
“ Û
” š
• »
– î	— 3
˜ Õ
˜ ©
™ Ø
š â
› ‘
œ Ë
 ü
ž 	Ÿ M
  Ë
¡ Þ
¢ â
£ æ	¤ 
¥ ¢	
¦ Ž
§ Þ	¨ 
© ò
ª Ù
« à
¬ ´	
­ Î
® Š
¯ ˜	
° É
± ·
² É
³ Œ
´ °
µ ¨
¶  
· ý
¸ –
¹ ý
º û
» ½
¼ ì		½ M
¾ Ö	¿ d
À “
Á ã
Â Ô
Ã ó
Ä ”
Å µ
Æ ­
Ç É
È ¤

É „
Ê ‡
Ë 	
Ì ˆ
Í Ð
Î í
Ï á
Ð ˜	
Ñ í
Ò è
Ó ý
Ó Ñ
Ô ¨
Õ ½
Ö ð
× 
Ø ª
Ù ø

Ú ¥
Û ¹		Ü 
Ý Á
Þ ¶	
ß «
à ö
á Ñ
â Ì
ã „
ä Ç
å ‹
æ ˜
ç ò
è ½
é Ÿ
ê ¾
ë «

ì í
í è
î â
ï é
ï ½
ð ž
ñ ß
ò 
ó È
ô Œ
õ ®	ö 
÷ å
ø í
ù Æ
ú Æ
û Æ
ü ì
ý º
þ ë
ÿ Õ
€ µ
 ê
‚ ð
ƒ ™
„ ž
… á
… µ
† ´
‡ å
ˆ ¼
‰ ñ	
Š ¦
‹ Ç

Œ ü
 Ü
Ž ¢	 	 H
‘ ²
’ Ö
“ ù
” Ñ
• ©
– 
— æ
˜ †	™ ,
™ €
š ¦
› ú
œ ©

 ±
ž ²
Ÿ †
  ’
¡ ½	¢ R
£ ­
¤ ‚	¥ ]
¥ ±
¦ Ž
§ í
¨ Ü

© ´	
ª Á
« ø
¬ Ô
­ Ú
® ¯
¯ Ð	
° Ú
± þ
² Â
³ °	´ .
´ ‚
µ ¸
¶ ©
· õ
¸ ¶¹ 

º Ý
» 		¼ i
½ 
¾ ¹
¾ 
¿ µ
¿ ‰
À ¾
Á ý

Â ˜
Ã £	Ä ,
Å …
Æ ô
Ç Ô
È Õ
É Œ
Ê Š

Ë –
Ì Ø	Í f
Î Ü

Ï º
Ð õ
Ñ ÄÒ 
Ó Ø
Ô ó	
Õ …
Õ Ù
Ö –
× ˆ
Ø ñ	
Ù ‰
Ú À
Û ö	
Ü ð
Ý Â
Þ ›
ß è
à å
á ˆ

â €
ã å
ã ¹

ä Å
ä ™	å O
æ ø

ç Á
è Ä
é ¥
ê ¨
ë 
ë Õ

ì Ð	
í ®

î À

ï Ù
ð ñ
ñ ¹
ò º
ó ¢
ô ©
ô ý
õ …
ö ’

÷ ×	
ø ¤
ù Í
ù ¡
ú ÿ
û °
ü å
ý ‹
þ ê
ÿ ò
€ É
€ 

 Ä
‚ ï
ƒ 
ƒ ñ

„ ñ
… €
† ”
‡ æ

ˆ 
‰ ù
Š …
‹ °
Œ À

 ô
Ž ƒ	 i	 1
‘ ‡	’ A
’ •	“ k
” š
• ­
– Î
— …
˜ š	
™ î	
š ¶
› ¾
œ Ö
 à	ž y
ž Í
Ÿ Ý
  º
¡ ™
¢ Ó
£ ”
¤ ð
¥ ’
¦ ®
§ 
¨ ™"
rdsmh_kernel"
_Z13get_global_idj"	
_Z3logf"	
_Z3expf"
llvm.fmuladd.f32*—
shoc-1.1.5-S3D-rdsmh_kernel.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02

transfer_bytes
ˆ¢»

wgsize_log1p
’óŽA

wgsize
€

devmap_label
 
 
transfer_bytes_log1p
’óŽA