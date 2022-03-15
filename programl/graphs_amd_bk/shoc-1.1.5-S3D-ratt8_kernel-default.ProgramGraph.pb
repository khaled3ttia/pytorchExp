

[external]
KcallBC
A
	full_text4
2
0%6 = tail call i64 @_Z13get_global_idj(i32 0) #2
XgetelementptrBG
E
	full_text8
6
4%7 = getelementptr inbounds float, float* %0, i64 %6
"i64B

	full_text


i64 %6
HloadB@
>
	full_text1
/
-%8 = load float, float* %7, align 4, !tbaa !8
(float*B

	full_text

	float* %7
1fmulB)
'
	full_text

%9 = fmul float %8, %4
&floatB

	full_text


float %8
BfmulB:
8
	full_text+
)
'%10 = fmul float %9, 0x4193D2C640000000
&floatB

	full_text


float %9
JfdivBB
@
	full_text3
1
/%11 = fdiv float 1.000000e+00, %10, !fpmath !12
'floatB

	full_text

	float %10
=fmulB5
3
	full_text&
$
"%12 = fmul float %11, 1.013250e+06
'floatB

	full_text

	float %11
.addB'
%
	full_text

%13 = add i64 %6, 16
"i64B

	full_text


i64 %6
ZgetelementptrBI
G
	full_text:
8
6%14 = getelementptr inbounds float, float* %3, i64 %13
#i64B

	full_text
	
i64 %13
JloadBB
@
	full_text3
1
/%15 = load float, float* %14, align 4, !tbaa !8
)float*B

	full_text


float* %14
/addB(
&
	full_text

%16 = add i64 %6, 208
"i64B

	full_text


i64 %6
ZgetelementptrBI
G
	full_text:
8
6%17 = getelementptr inbounds float, float* %3, i64 %16
#i64B

	full_text
	
i64 %16
JloadBB
@
	full_text3
1
/%18 = load float, float* %17, align 4, !tbaa !8
)float*B

	full_text


float* %17
4fmulB,
*
	full_text

%19 = fmul float %15, %18
'floatB

	full_text

	float %15
'floatB

	full_text

	float %18
.addB'
%
	full_text

%20 = add i64 %6, 32
"i64B

	full_text


i64 %6
ZgetelementptrBI
G
	full_text:
8
6%21 = getelementptr inbounds float, float* %3, i64 %20
#i64B

	full_text
	
i64 %20
JloadBB
@
	full_text3
1
/%22 = load float, float* %21, align 4, !tbaa !8
)float*B

	full_text


float* %21
/addB(
&
	full_text

%23 = add i64 %6, 200
"i64B

	full_text


i64 %6
ZgetelementptrBI
G
	full_text:
8
6%24 = getelementptr inbounds float, float* %3, i64 %23
#i64B

	full_text
	
i64 %23
JloadBB
@
	full_text3
1
/%25 = load float, float* %24, align 4, !tbaa !8
)float*B

	full_text


float* %24
4fmulB,
*
	full_text

%26 = fmul float %22, %25
'floatB

	full_text

	float %22
'floatB

	full_text

	float %25
JfdivBB
@
	full_text3
1
/%27 = fdiv float 1.000000e+00, %26, !fpmath !12
'floatB

	full_text

	float %26
4fmulB,
*
	full_text

%28 = fmul float %19, %27
'floatB

	full_text

	float %19
'floatB

	full_text

	float %27
0addB)
'
	full_text

%29 = add i64 %6, 1200
"i64B

	full_text


i64 %6
ZgetelementptrBI
G
	full_text:
8
6%30 = getelementptr inbounds float, float* %1, i64 %29
#i64B

	full_text
	
i64 %29
JloadBB
@
	full_text3
1
/%31 = load float, float* %30, align 4, !tbaa !8
)float*B

	full_text


float* %30
ccallB[
Y
	full_textL
J
H%32 = tail call float @_Z4fminff(float %28, float 0x4415AF1D80000000) #2
'floatB

	full_text

	float %28
4fmulB,
*
	full_text

%33 = fmul float %31, %32
'floatB

	full_text

	float %31
'floatB

	full_text

	float %32
ZgetelementptrBI
G
	full_text:
8
6%34 = getelementptr inbounds float, float* %2, i64 %29
#i64B

	full_text
	
i64 %29
JstoreBA
?
	full_text2
0
.store float %33, float* %34, align 4, !tbaa !8
'floatB

	full_text

	float %33
)float*B

	full_text


float* %34
JloadBB
@
	full_text3
1
/%35 = load float, float* %21, align 4, !tbaa !8
)float*B

	full_text


float* %21
JloadBB
@
	full_text3
1
/%36 = load float, float* %17, align 4, !tbaa !8
)float*B

	full_text


float* %17
4fmulB,
*
	full_text

%37 = fmul float %35, %36
'floatB

	full_text

	float %35
'floatB

	full_text

	float %36
.addB'
%
	full_text

%38 = add i64 %6, 40
"i64B

	full_text


i64 %6
ZgetelementptrBI
G
	full_text:
8
6%39 = getelementptr inbounds float, float* %3, i64 %38
#i64B

	full_text
	
i64 %38
JloadBB
@
	full_text3
1
/%40 = load float, float* %39, align 4, !tbaa !8
)float*B

	full_text


float* %39
JloadBB
@
	full_text3
1
/%41 = load float, float* %24, align 4, !tbaa !8
)float*B

	full_text


float* %24
4fmulB,
*
	full_text

%42 = fmul float %40, %41
'floatB

	full_text

	float %40
'floatB

	full_text

	float %41
JfdivBB
@
	full_text3
1
/%43 = fdiv float 1.000000e+00, %42, !fpmath !12
'floatB

	full_text

	float %42
4fmulB,
*
	full_text

%44 = fmul float %37, %43
'floatB

	full_text

	float %37
'floatB

	full_text

	float %43
0addB)
'
	full_text

%45 = add i64 %6, 1208
"i64B

	full_text


i64 %6
ZgetelementptrBI
G
	full_text:
8
6%46 = getelementptr inbounds float, float* %1, i64 %45
#i64B

	full_text
	
i64 %45
JloadBB
@
	full_text3
1
/%47 = load float, float* %46, align 4, !tbaa !8
)float*B

	full_text


float* %46
ccallB[
Y
	full_textL
J
H%48 = tail call float @_Z4fminff(float %44, float 0x4415AF1D80000000) #2
'floatB

	full_text

	float %44
4fmulB,
*
	full_text

%49 = fmul float %47, %48
'floatB

	full_text

	float %47
'floatB

	full_text

	float %48
ZgetelementptrBI
G
	full_text:
8
6%50 = getelementptr inbounds float, float* %2, i64 %45
#i64B

	full_text
	
i64 %45
JstoreBA
?
	full_text2
0
.store float %49, float* %50, align 4, !tbaa !8
'floatB

	full_text

	float %49
)float*B

	full_text


float* %50
.addB'
%
	full_text

%51 = add i64 %6, 24
"i64B

	full_text


i64 %6
ZgetelementptrBI
G
	full_text:
8
6%52 = getelementptr inbounds float, float* %3, i64 %51
#i64B

	full_text
	
i64 %51
JloadBB
@
	full_text3
1
/%53 = load float, float* %52, align 4, !tbaa !8
)float*B

	full_text


float* %52
JloadBB
@
	full_text3
1
/%54 = load float, float* %17, align 4, !tbaa !8
)float*B

	full_text


float* %17
4fmulB,
*
	full_text

%55 = fmul float %53, %54
'floatB

	full_text

	float %53
'floatB

	full_text

	float %54
.addB'
%
	full_text

%56 = add i64 %6, 48
"i64B

	full_text


i64 %6
ZgetelementptrBI
G
	full_text:
8
6%57 = getelementptr inbounds float, float* %3, i64 %56
#i64B

	full_text
	
i64 %56
JloadBB
@
	full_text3
1
/%58 = load float, float* %57, align 4, !tbaa !8
)float*B

	full_text


float* %57
JloadBB
@
	full_text3
1
/%59 = load float, float* %24, align 4, !tbaa !8
)float*B

	full_text


float* %24
4fmulB,
*
	full_text

%60 = fmul float %58, %59
'floatB

	full_text

	float %58
'floatB

	full_text

	float %59
JfdivBB
@
	full_text3
1
/%61 = fdiv float 1.000000e+00, %60, !fpmath !12
'floatB

	full_text

	float %60
4fmulB,
*
	full_text

%62 = fmul float %55, %61
'floatB

	full_text

	float %55
'floatB

	full_text

	float %61
0addB)
'
	full_text

%63 = add i64 %6, 1216
"i64B

	full_text


i64 %6
ZgetelementptrBI
G
	full_text:
8
6%64 = getelementptr inbounds float, float* %1, i64 %63
#i64B

	full_text
	
i64 %63
JloadBB
@
	full_text3
1
/%65 = load float, float* %64, align 4, !tbaa !8
)float*B

	full_text


float* %64
ccallB[
Y
	full_textL
J
H%66 = tail call float @_Z4fminff(float %62, float 0x4415AF1D80000000) #2
'floatB

	full_text

	float %62
4fmulB,
*
	full_text

%67 = fmul float %65, %66
'floatB

	full_text

	float %65
'floatB

	full_text

	float %66
ZgetelementptrBI
G
	full_text:
8
6%68 = getelementptr inbounds float, float* %2, i64 %63
#i64B

	full_text
	
i64 %63
JstoreBA
?
	full_text2
0
.store float %67, float* %68, align 4, !tbaa !8
'floatB

	full_text

	float %67
)float*B

	full_text


float* %68
JloadBB
@
	full_text3
1
/%69 = load float, float* %52, align 4, !tbaa !8
)float*B

	full_text


float* %52
JloadBB
@
	full_text3
1
/%70 = load float, float* %17, align 4, !tbaa !8
)float*B

	full_text


float* %17
4fmulB,
*
	full_text

%71 = fmul float %69, %70
'floatB

	full_text

	float %69
'floatB

	full_text

	float %70
JloadBB
@
	full_text3
1
/%72 = load float, float* %21, align 4, !tbaa !8
)float*B

	full_text


float* %21
/addB(
&
	full_text

%73 = add i64 %6, 104
"i64B

	full_text


i64 %6
ZgetelementptrBI
G
	full_text:
8
6%74 = getelementptr inbounds float, float* %3, i64 %73
#i64B

	full_text
	
i64 %73
JloadBB
@
	full_text3
1
/%75 = load float, float* %74, align 4, !tbaa !8
)float*B

	full_text


float* %74
4fmulB,
*
	full_text

%76 = fmul float %72, %75
'floatB

	full_text

	float %72
'floatB

	full_text

	float %75
/addB(
&
	full_text

%77 = add i64 %6, 128
"i64B

	full_text


i64 %6
ZgetelementptrBI
G
	full_text:
8
6%78 = getelementptr inbounds float, float* %3, i64 %77
#i64B

	full_text
	
i64 %77
JloadBB
@
	full_text3
1
/%79 = load float, float* %78, align 4, !tbaa !8
)float*B

	full_text


float* %78
4fmulB,
*
	full_text

%80 = fmul float %76, %79
'floatB

	full_text

	float %76
'floatB

	full_text

	float %79
4fmulB,
*
	full_text

%81 = fmul float %12, %80
'floatB

	full_text

	float %12
'floatB

	full_text

	float %80
JfdivBB
@
	full_text3
1
/%82 = fdiv float 1.000000e+00, %81, !fpmath !12
'floatB

	full_text

	float %81
4fmulB,
*
	full_text

%83 = fmul float %71, %82
'floatB

	full_text

	float %71
'floatB

	full_text

	float %82
0addB)
'
	full_text

%84 = add i64 %6, 1224
"i64B

	full_text


i64 %6
ZgetelementptrBI
G
	full_text:
8
6%85 = getelementptr inbounds float, float* %1, i64 %84
#i64B

	full_text
	
i64 %84
JloadBB
@
	full_text3
1
/%86 = load float, float* %85, align 4, !tbaa !8
)float*B

	full_text


float* %85
ccallB[
Y
	full_textL
J
H%87 = tail call float @_Z4fminff(float %83, float 0x4415AF1D80000000) #2
'floatB

	full_text

	float %83
4fmulB,
*
	full_text

%88 = fmul float %86, %87
'floatB

	full_text

	float %86
'floatB

	full_text

	float %87
ZgetelementptrBI
G
	full_text:
8
6%89 = getelementptr inbounds float, float* %2, i64 %84
#i64B

	full_text
	
i64 %84
JstoreBA
?
	full_text2
0
.store float %88, float* %89, align 4, !tbaa !8
'floatB

	full_text

	float %88
)float*B

	full_text


float* %89
/addB(
&
	full_text

%90 = add i64 %6, 168
"i64B

	full_text


i64 %6
ZgetelementptrBI
G
	full_text:
8
6%91 = getelementptr inbounds float, float* %3, i64 %90
#i64B

	full_text
	
i64 %90
JloadBB
@
	full_text3
1
/%92 = load float, float* %91, align 4, !tbaa !8
)float*B

	full_text


float* %91
YgetelementptrBH
F
	full_text9
7
5%93 = getelementptr inbounds float, float* %3, i64 %6
"i64B

	full_text


i64 %6
JloadBB
@
	full_text3
1
/%94 = load float, float* %93, align 4, !tbaa !8
)float*B

	full_text


float* %93
/addB(
&
	full_text

%95 = add i64 %6, 152
"i64B

	full_text


i64 %6
ZgetelementptrBI
G
	full_text:
8
6%96 = getelementptr inbounds float, float* %3, i64 %95
#i64B

	full_text
	
i64 %95
JloadBB
@
	full_text3
1
/%97 = load float, float* %96, align 4, !tbaa !8
)float*B

	full_text


float* %96
4fmulB,
*
	full_text

%98 = fmul float %94, %97
'floatB

	full_text

	float %94
'floatB

	full_text

	float %97
4fmulB,
*
	full_text

%99 = fmul float %12, %98
'floatB

	full_text

	float %12
'floatB

	full_text

	float %98
KfdivBC
A
	full_text4
2
0%100 = fdiv float 1.000000e+00, %99, !fpmath !12
'floatB

	full_text

	float %99
6fmulB.
,
	full_text

%101 = fmul float %92, %100
'floatB

	full_text

	float %92
(floatB

	full_text


float %100
1addB*
(
	full_text

%102 = add i64 %6, 1232
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%103 = getelementptr inbounds float, float* %1, i64 %102
$i64B

	full_text


i64 %102
LloadBD
B
	full_text5
3
1%104 = load float, float* %103, align 4, !tbaa !8
*float*B

	full_text

float* %103
ecallB]
[
	full_textN
L
J%105 = tail call float @_Z4fminff(float %101, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %101
7fmulB/
-
	full_text 

%106 = fmul float %104, %105
(floatB

	full_text


float %104
(floatB

	full_text


float %105
\getelementptrBK
I
	full_text<
:
8%107 = getelementptr inbounds float, float* %2, i64 %102
$i64B

	full_text


i64 %102
LstoreBC
A
	full_text4
2
0store float %106, float* %107, align 4, !tbaa !8
(floatB

	full_text


float %106
*float*B

	full_text

float* %107
.addB'
%
	full_text

%108 = add i64 %6, 8
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%109 = getelementptr inbounds float, float* %3, i64 %108
$i64B

	full_text


i64 %108
LloadBD
B
	full_text5
3
1%110 = load float, float* %109, align 4, !tbaa !8
*float*B

	full_text

float* %109
KloadBC
A
	full_text4
2
0%111 = load float, float* %91, align 4, !tbaa !8
)float*B

	full_text


float* %91
7fmulB/
-
	full_text 

%112 = fmul float %110, %111
(floatB

	full_text


float %110
(floatB

	full_text


float %111
6fmulB.
,
	full_text

%113 = fmul float %12, %112
'floatB

	full_text

	float %12
(floatB

	full_text


float %112
0addB)
'
	full_text

%114 = add i64 %6, 176
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%115 = getelementptr inbounds float, float* %3, i64 %114
$i64B

	full_text


i64 %114
LloadBD
B
	full_text5
3
1%116 = load float, float* %115, align 4, !tbaa !8
*float*B

	full_text

float* %115
LfdivBD
B
	full_text5
3
1%117 = fdiv float 1.000000e+00, %116, !fpmath !12
(floatB

	full_text


float %116
7fmulB/
-
	full_text 

%118 = fmul float %113, %117
(floatB

	full_text


float %113
(floatB

	full_text


float %117
1addB*
(
	full_text

%119 = add i64 %6, 1240
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%120 = getelementptr inbounds float, float* %1, i64 %119
$i64B

	full_text


i64 %119
LloadBD
B
	full_text5
3
1%121 = load float, float* %120, align 4, !tbaa !8
*float*B

	full_text

float* %120
ecallB]
[
	full_textN
L
J%122 = tail call float @_Z4fminff(float %118, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %118
7fmulB/
-
	full_text 

%123 = fmul float %121, %122
(floatB

	full_text


float %121
(floatB

	full_text


float %122
\getelementptrBK
I
	full_text<
:
8%124 = getelementptr inbounds float, float* %2, i64 %119
$i64B

	full_text


i64 %119
LstoreBC
A
	full_text4
2
0store float %123, float* %124, align 4, !tbaa !8
(floatB

	full_text


float %123
*float*B

	full_text

float* %124
LloadBD
B
	full_text5
3
1%125 = load float, float* %109, align 4, !tbaa !8
*float*B

	full_text

float* %109
KloadBC
A
	full_text4
2
0%126 = load float, float* %91, align 4, !tbaa !8
)float*B

	full_text


float* %91
7fmulB/
-
	full_text 

%127 = fmul float %125, %126
(floatB

	full_text


float %125
(floatB

	full_text


float %126
KloadBC
A
	full_text4
2
0%128 = load float, float* %93, align 4, !tbaa !8
)float*B

	full_text


float* %93
0addB)
'
	full_text

%129 = add i64 %6, 160
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%130 = getelementptr inbounds float, float* %3, i64 %129
$i64B

	full_text


i64 %129
LloadBD
B
	full_text5
3
1%131 = load float, float* %130, align 4, !tbaa !8
*float*B

	full_text

float* %130
7fmulB/
-
	full_text 

%132 = fmul float %128, %131
(floatB

	full_text


float %128
(floatB

	full_text


float %131
LfdivBD
B
	full_text5
3
1%133 = fdiv float 1.000000e+00, %132, !fpmath !12
(floatB

	full_text


float %132
7fmulB/
-
	full_text 

%134 = fmul float %127, %133
(floatB

	full_text


float %127
(floatB

	full_text


float %133
1addB*
(
	full_text

%135 = add i64 %6, 1248
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%136 = getelementptr inbounds float, float* %1, i64 %135
$i64B

	full_text


i64 %135
LloadBD
B
	full_text5
3
1%137 = load float, float* %136, align 4, !tbaa !8
*float*B

	full_text

float* %136
ecallB]
[
	full_textN
L
J%138 = tail call float @_Z4fminff(float %134, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %134
7fmulB/
-
	full_text 

%139 = fmul float %137, %138
(floatB

	full_text


float %137
(floatB

	full_text


float %138
\getelementptrBK
I
	full_text<
:
8%140 = getelementptr inbounds float, float* %2, i64 %135
$i64B

	full_text


i64 %135
LstoreBC
A
	full_text4
2
0store float %139, float* %140, align 4, !tbaa !8
(floatB

	full_text


float %139
*float*B

	full_text

float* %140
KloadBC
A
	full_text4
2
0%141 = load float, float* %14, align 4, !tbaa !8
)float*B

	full_text


float* %14
KloadBC
A
	full_text4
2
0%142 = load float, float* %91, align 4, !tbaa !8
)float*B

	full_text


float* %91
7fmulB/
-
	full_text 

%143 = fmul float %141, %142
(floatB

	full_text


float %141
(floatB

	full_text


float %142
KloadBC
A
	full_text4
2
0%144 = load float, float* %21, align 4, !tbaa !8
)float*B

	full_text


float* %21
LloadBD
B
	full_text5
3
1%145 = load float, float* %130, align 4, !tbaa !8
*float*B

	full_text

float* %130
7fmulB/
-
	full_text 

%146 = fmul float %144, %145
(floatB

	full_text


float %144
(floatB

	full_text


float %145
LfdivBD
B
	full_text5
3
1%147 = fdiv float 1.000000e+00, %146, !fpmath !12
(floatB

	full_text


float %146
7fmulB/
-
	full_text 

%148 = fmul float %143, %147
(floatB

	full_text


float %143
(floatB

	full_text


float %147
1addB*
(
	full_text

%149 = add i64 %6, 1256
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%150 = getelementptr inbounds float, float* %1, i64 %149
$i64B

	full_text


i64 %149
LloadBD
B
	full_text5
3
1%151 = load float, float* %150, align 4, !tbaa !8
*float*B

	full_text

float* %150
ecallB]
[
	full_textN
L
J%152 = tail call float @_Z4fminff(float %148, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %148
7fmulB/
-
	full_text 

%153 = fmul float %151, %152
(floatB

	full_text


float %151
(floatB

	full_text


float %152
\getelementptrBK
I
	full_text<
:
8%154 = getelementptr inbounds float, float* %2, i64 %149
$i64B

	full_text


i64 %149
LstoreBC
A
	full_text4
2
0store float %153, float* %154, align 4, !tbaa !8
(floatB

	full_text


float %153
*float*B

	full_text

float* %154
KloadBC
A
	full_text4
2
0%155 = load float, float* %14, align 4, !tbaa !8
)float*B

	full_text


float* %14
KloadBC
A
	full_text4
2
0%156 = load float, float* %91, align 4, !tbaa !8
)float*B

	full_text


float* %91
7fmulB/
-
	full_text 

%157 = fmul float %155, %156
(floatB

	full_text


float %155
(floatB

	full_text


float %156
/addB(
&
	full_text

%158 = add i64 %6, 88
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%159 = getelementptr inbounds float, float* %3, i64 %158
$i64B

	full_text


i64 %158
LloadBD
B
	full_text5
3
1%160 = load float, float* %159, align 4, !tbaa !8
*float*B

	full_text

float* %159
0addB)
'
	full_text

%161 = add i64 %6, 120
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%162 = getelementptr inbounds float, float* %3, i64 %161
$i64B

	full_text


i64 %161
LloadBD
B
	full_text5
3
1%163 = load float, float* %162, align 4, !tbaa !8
*float*B

	full_text

float* %162
7fmulB/
-
	full_text 

%164 = fmul float %160, %163
(floatB

	full_text


float %160
(floatB

	full_text


float %163
LfdivBD
B
	full_text5
3
1%165 = fdiv float 1.000000e+00, %164, !fpmath !12
(floatB

	full_text


float %164
7fmulB/
-
	full_text 

%166 = fmul float %157, %165
(floatB

	full_text


float %157
(floatB

	full_text


float %165
1addB*
(
	full_text

%167 = add i64 %6, 1264
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%168 = getelementptr inbounds float, float* %1, i64 %167
$i64B

	full_text


i64 %167
LloadBD
B
	full_text5
3
1%169 = load float, float* %168, align 4, !tbaa !8
*float*B

	full_text

float* %168
ecallB]
[
	full_textN
L
J%170 = tail call float @_Z4fminff(float %166, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %166
7fmulB/
-
	full_text 

%171 = fmul float %169, %170
(floatB

	full_text


float %169
(floatB

	full_text


float %170
\getelementptrBK
I
	full_text<
:
8%172 = getelementptr inbounds float, float* %2, i64 %167
$i64B

	full_text


i64 %167
LstoreBC
A
	full_text4
2
0store float %171, float* %172, align 4, !tbaa !8
(floatB

	full_text


float %171
*float*B

	full_text

float* %172
KloadBC
A
	full_text4
2
0%173 = load float, float* %14, align 4, !tbaa !8
)float*B

	full_text


float* %14
KloadBC
A
	full_text4
2
0%174 = load float, float* %91, align 4, !tbaa !8
)float*B

	full_text


float* %91
7fmulB/
-
	full_text 

%175 = fmul float %173, %174
(floatB

	full_text


float %173
(floatB

	full_text


float %174
/addB(
&
	full_text

%176 = add i64 %6, 72
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%177 = getelementptr inbounds float, float* %3, i64 %176
$i64B

	full_text


i64 %176
LloadBD
B
	full_text5
3
1%178 = load float, float* %177, align 4, !tbaa !8
*float*B

	full_text

float* %177
KloadBC
A
	full_text4
2
0%179 = load float, float* %78, align 4, !tbaa !8
)float*B

	full_text


float* %78
7fmulB/
-
	full_text 

%180 = fmul float %178, %179
(floatB

	full_text


float %178
(floatB

	full_text


float %179
LfdivBD
B
	full_text5
3
1%181 = fdiv float 1.000000e+00, %180, !fpmath !12
(floatB

	full_text


float %180
7fmulB/
-
	full_text 

%182 = fmul float %175, %181
(floatB

	full_text


float %175
(floatB

	full_text


float %181
1addB*
(
	full_text

%183 = add i64 %6, 1272
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%184 = getelementptr inbounds float, float* %1, i64 %183
$i64B

	full_text


i64 %183
LloadBD
B
	full_text5
3
1%185 = load float, float* %184, align 4, !tbaa !8
*float*B

	full_text

float* %184
ecallB]
[
	full_textN
L
J%186 = tail call float @_Z4fminff(float %182, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %182
7fmulB/
-
	full_text 

%187 = fmul float %185, %186
(floatB

	full_text


float %185
(floatB

	full_text


float %186
\getelementptrBK
I
	full_text<
:
8%188 = getelementptr inbounds float, float* %2, i64 %183
$i64B

	full_text


i64 %183
LstoreBC
A
	full_text4
2
0store float %187, float* %188, align 4, !tbaa !8
(floatB

	full_text


float %187
*float*B

	full_text

float* %188
KloadBC
A
	full_text4
2
0%189 = load float, float* %21, align 4, !tbaa !8
)float*B

	full_text


float* %21
KloadBC
A
	full_text4
2
0%190 = load float, float* %91, align 4, !tbaa !8
)float*B

	full_text


float* %91
7fmulB/
-
	full_text 

%191 = fmul float %189, %190
(floatB

	full_text


float %189
(floatB

	full_text


float %190
KloadBC
A
	full_text4
2
0%192 = load float, float* %39, align 4, !tbaa !8
)float*B

	full_text


float* %39
LloadBD
B
	full_text5
3
1%193 = load float, float* %130, align 4, !tbaa !8
*float*B

	full_text

float* %130
7fmulB/
-
	full_text 

%194 = fmul float %192, %193
(floatB

	full_text


float %192
(floatB

	full_text


float %193
LfdivBD
B
	full_text5
3
1%195 = fdiv float 1.000000e+00, %194, !fpmath !12
(floatB

	full_text


float %194
7fmulB/
-
	full_text 

%196 = fmul float %191, %195
(floatB

	full_text


float %191
(floatB

	full_text


float %195
1addB*
(
	full_text

%197 = add i64 %6, 1280
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%198 = getelementptr inbounds float, float* %1, i64 %197
$i64B

	full_text


i64 %197
LloadBD
B
	full_text5
3
1%199 = load float, float* %198, align 4, !tbaa !8
*float*B

	full_text

float* %198
ecallB]
[
	full_textN
L
J%200 = tail call float @_Z4fminff(float %196, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %196
7fmulB/
-
	full_text 

%201 = fmul float %199, %200
(floatB

	full_text


float %199
(floatB

	full_text


float %200
\getelementptrBK
I
	full_text<
:
8%202 = getelementptr inbounds float, float* %2, i64 %197
$i64B

	full_text


i64 %197
LstoreBC
A
	full_text4
2
0store float %201, float* %202, align 4, !tbaa !8
(floatB

	full_text


float %201
*float*B

	full_text

float* %202
KloadBC
A
	full_text4
2
0%203 = load float, float* %52, align 4, !tbaa !8
)float*B

	full_text


float* %52
KloadBC
A
	full_text4
2
0%204 = load float, float* %91, align 4, !tbaa !8
)float*B

	full_text


float* %91
7fmulB/
-
	full_text 

%205 = fmul float %203, %204
(floatB

	full_text


float %203
(floatB

	full_text


float %204
KloadBC
A
	full_text4
2
0%206 = load float, float* %57, align 4, !tbaa !8
)float*B

	full_text


float* %57
LloadBD
B
	full_text5
3
1%207 = load float, float* %130, align 4, !tbaa !8
*float*B

	full_text

float* %130
7fmulB/
-
	full_text 

%208 = fmul float %206, %207
(floatB

	full_text


float %206
(floatB

	full_text


float %207
LfdivBD
B
	full_text5
3
1%209 = fdiv float 1.000000e+00, %208, !fpmath !12
(floatB

	full_text


float %208
7fmulB/
-
	full_text 

%210 = fmul float %205, %209
(floatB

	full_text


float %205
(floatB

	full_text


float %209
1addB*
(
	full_text

%211 = add i64 %6, 1288
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%212 = getelementptr inbounds float, float* %1, i64 %211
$i64B

	full_text


i64 %211
LloadBD
B
	full_text5
3
1%213 = load float, float* %212, align 4, !tbaa !8
*float*B

	full_text

float* %212
ecallB]
[
	full_textN
L
J%214 = tail call float @_Z4fminff(float %210, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %210
7fmulB/
-
	full_text 

%215 = fmul float %213, %214
(floatB

	full_text


float %213
(floatB

	full_text


float %214
\getelementptrBK
I
	full_text<
:
8%216 = getelementptr inbounds float, float* %2, i64 %211
$i64B

	full_text


i64 %211
LstoreBC
A
	full_text4
2
0store float %215, float* %216, align 4, !tbaa !8
(floatB

	full_text


float %215
*float*B

	full_text

float* %216
KloadBC
A
	full_text4
2
0%217 = load float, float* %57, align 4, !tbaa !8
)float*B

	full_text


float* %57
KloadBC
A
	full_text4
2
0%218 = load float, float* %91, align 4, !tbaa !8
)float*B

	full_text


float* %91
7fmulB/
-
	full_text 

%219 = fmul float %217, %218
(floatB

	full_text


float %217
(floatB

	full_text


float %218
KloadBC
A
	full_text4
2
0%220 = load float, float* %21, align 4, !tbaa !8
)float*B

	full_text


float* %21
0addB)
'
	full_text

%221 = add i64 %6, 216
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%222 = getelementptr inbounds float, float* %3, i64 %221
$i64B

	full_text


i64 %221
LloadBD
B
	full_text5
3
1%223 = load float, float* %222, align 4, !tbaa !8
*float*B

	full_text

float* %222
7fmulB/
-
	full_text 

%224 = fmul float %220, %223
(floatB

	full_text


float %220
(floatB

	full_text


float %223
LfdivBD
B
	full_text5
3
1%225 = fdiv float 1.000000e+00, %224, !fpmath !12
(floatB

	full_text


float %224
7fmulB/
-
	full_text 

%226 = fmul float %219, %225
(floatB

	full_text


float %219
(floatB

	full_text


float %225
1addB*
(
	full_text

%227 = add i64 %6, 1296
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%228 = getelementptr inbounds float, float* %1, i64 %227
$i64B

	full_text


i64 %227
LloadBD
B
	full_text5
3
1%229 = load float, float* %228, align 4, !tbaa !8
*float*B

	full_text

float* %228
ecallB]
[
	full_textN
L
J%230 = tail call float @_Z4fminff(float %226, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %226
7fmulB/
-
	full_text 

%231 = fmul float %229, %230
(floatB

	full_text


float %229
(floatB

	full_text


float %230
\getelementptrBK
I
	full_text<
:
8%232 = getelementptr inbounds float, float* %2, i64 %227
$i64B

	full_text


i64 %227
LstoreBC
A
	full_text4
2
0store float %231, float* %232, align 4, !tbaa !8
(floatB

	full_text


float %231
*float*B

	full_text

float* %232
LloadBD
B
	full_text5
3
1%233 = load float, float* %162, align 4, !tbaa !8
*float*B

	full_text

float* %162
KloadBC
A
	full_text4
2
0%234 = load float, float* %91, align 4, !tbaa !8
)float*B

	full_text


float* %91
7fmulB/
-
	full_text 

%235 = fmul float %233, %234
(floatB

	full_text


float %233
(floatB

	full_text


float %234
KloadBC
A
	full_text4
2
0%236 = load float, float* %74, align 4, !tbaa !8
)float*B

	full_text


float* %74
LloadBD
B
	full_text5
3
1%237 = load float, float* %115, align 4, !tbaa !8
*float*B

	full_text

float* %115
7fmulB/
-
	full_text 

%238 = fmul float %236, %237
(floatB

	full_text


float %236
(floatB

	full_text


float %237
LfdivBD
B
	full_text5
3
1%239 = fdiv float 1.000000e+00, %238, !fpmath !12
(floatB

	full_text


float %238
7fmulB/
-
	full_text 

%240 = fmul float %235, %239
(floatB

	full_text


float %235
(floatB

	full_text


float %239
1addB*
(
	full_text

%241 = add i64 %6, 1304
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%242 = getelementptr inbounds float, float* %1, i64 %241
$i64B

	full_text


i64 %241
LloadBD
B
	full_text5
3
1%243 = load float, float* %242, align 4, !tbaa !8
*float*B

	full_text

float* %242
ecallB]
[
	full_textN
L
J%244 = tail call float @_Z4fminff(float %240, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %240
7fmulB/
-
	full_text 

%245 = fmul float %243, %244
(floatB

	full_text


float %243
(floatB

	full_text


float %244
\getelementptrBK
I
	full_text<
:
8%246 = getelementptr inbounds float, float* %2, i64 %241
$i64B

	full_text


i64 %241
LstoreBC
A
	full_text4
2
0store float %245, float* %246, align 4, !tbaa !8
(floatB

	full_text


float %245
*float*B

	full_text

float* %246
LloadBD
B
	full_text5
3
1%247 = load float, float* %177, align 4, !tbaa !8
*float*B

	full_text

float* %177
KloadBC
A
	full_text4
2
0%248 = load float, float* %91, align 4, !tbaa !8
)float*B

	full_text


float* %91
7fmulB/
-
	full_text 

%249 = fmul float %247, %248
(floatB

	full_text


float %247
(floatB

	full_text


float %248
LloadBD
B
	full_text5
3
1%250 = load float, float* %109, align 4, !tbaa !8
*float*B

	full_text

float* %109
0addB)
'
	full_text

%251 = add i64 %6, 224
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%252 = getelementptr inbounds float, float* %3, i64 %251
$i64B

	full_text


i64 %251
LloadBD
B
	full_text5
3
1%253 = load float, float* %252, align 4, !tbaa !8
*float*B

	full_text

float* %252
7fmulB/
-
	full_text 

%254 = fmul float %250, %253
(floatB

	full_text


float %250
(floatB

	full_text


float %253
LfdivBD
B
	full_text5
3
1%255 = fdiv float 1.000000e+00, %254, !fpmath !12
(floatB

	full_text


float %254
7fmulB/
-
	full_text 

%256 = fmul float %249, %255
(floatB

	full_text


float %249
(floatB

	full_text


float %255
1addB*
(
	full_text

%257 = add i64 %6, 1312
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%258 = getelementptr inbounds float, float* %1, i64 %257
$i64B

	full_text


i64 %257
LloadBD
B
	full_text5
3
1%259 = load float, float* %258, align 4, !tbaa !8
*float*B

	full_text

float* %258
ecallB]
[
	full_textN
L
J%260 = tail call float @_Z4fminff(float %256, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %256
7fmulB/
-
	full_text 

%261 = fmul float %259, %260
(floatB

	full_text


float %259
(floatB

	full_text


float %260
\getelementptrBK
I
	full_text<
:
8%262 = getelementptr inbounds float, float* %2, i64 %257
$i64B

	full_text


i64 %257
LstoreBC
A
	full_text4
2
0store float %261, float* %262, align 4, !tbaa !8
(floatB

	full_text


float %261
*float*B

	full_text

float* %262
/addB(
&
	full_text

%263 = add i64 %6, 80
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%264 = getelementptr inbounds float, float* %3, i64 %263
$i64B

	full_text


i64 %263
LloadBD
B
	full_text5
3
1%265 = load float, float* %264, align 4, !tbaa !8
*float*B

	full_text

float* %264
KloadBC
A
	full_text4
2
0%266 = load float, float* %91, align 4, !tbaa !8
)float*B

	full_text


float* %91
7fmulB/
-
	full_text 

%267 = fmul float %265, %266
(floatB

	full_text


float %265
(floatB

	full_text


float %266
/addB(
&
	full_text

%268 = add i64 %6, 96
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%269 = getelementptr inbounds float, float* %3, i64 %268
$i64B

	full_text


i64 %268
LloadBD
B
	full_text5
3
1%270 = load float, float* %269, align 4, !tbaa !8
*float*B

	full_text

float* %269
KloadBC
A
	full_text4
2
0%271 = load float, float* %96, align 4, !tbaa !8
)float*B

	full_text


float* %96
7fmulB/
-
	full_text 

%272 = fmul float %270, %271
(floatB

	full_text


float %270
(floatB

	full_text


float %271
LfdivBD
B
	full_text5
3
1%273 = fdiv float 1.000000e+00, %272, !fpmath !12
(floatB

	full_text


float %272
7fmulB/
-
	full_text 

%274 = fmul float %267, %273
(floatB

	full_text


float %267
(floatB

	full_text


float %273
1addB*
(
	full_text

%275 = add i64 %6, 1320
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%276 = getelementptr inbounds float, float* %1, i64 %275
$i64B

	full_text


i64 %275
LloadBD
B
	full_text5
3
1%277 = load float, float* %276, align 4, !tbaa !8
*float*B

	full_text

float* %276
ecallB]
[
	full_textN
L
J%278 = tail call float @_Z4fminff(float %274, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %274
7fmulB/
-
	full_text 

%279 = fmul float %277, %278
(floatB

	full_text


float %277
(floatB

	full_text


float %278
\getelementptrBK
I
	full_text<
:
8%280 = getelementptr inbounds float, float* %2, i64 %275
$i64B

	full_text


i64 %275
LstoreBC
A
	full_text4
2
0store float %279, float* %280, align 4, !tbaa !8
(floatB

	full_text


float %279
*float*B

	full_text

float* %280
LloadBD
B
	full_text5
3
1%281 = load float, float* %264, align 4, !tbaa !8
*float*B

	full_text

float* %264
KloadBC
A
	full_text4
2
0%282 = load float, float* %91, align 4, !tbaa !8
)float*B

	full_text


float* %91
7fmulB/
-
	full_text 

%283 = fmul float %281, %282
(floatB

	full_text


float %281
(floatB

	full_text


float %282
LloadBD
B
	full_text5
3
1%284 = load float, float* %109, align 4, !tbaa !8
*float*B

	full_text

float* %109
LloadBD
B
	full_text5
3
1%285 = load float, float* %252, align 4, !tbaa !8
*float*B

	full_text

float* %252
7fmulB/
-
	full_text 

%286 = fmul float %284, %285
(floatB

	full_text


float %284
(floatB

	full_text


float %285
LfdivBD
B
	full_text5
3
1%287 = fdiv float 1.000000e+00, %286, !fpmath !12
(floatB

	full_text


float %286
7fmulB/
-
	full_text 

%288 = fmul float %283, %287
(floatB

	full_text


float %283
(floatB

	full_text


float %287
1addB*
(
	full_text

%289 = add i64 %6, 1328
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%290 = getelementptr inbounds float, float* %1, i64 %289
$i64B

	full_text


i64 %289
LloadBD
B
	full_text5
3
1%291 = load float, float* %290, align 4, !tbaa !8
*float*B

	full_text

float* %290
ecallB]
[
	full_textN
L
J%292 = tail call float @_Z4fminff(float %288, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %288
7fmulB/
-
	full_text 

%293 = fmul float %291, %292
(floatB

	full_text


float %291
(floatB

	full_text


float %292
\getelementptrBK
I
	full_text<
:
8%294 = getelementptr inbounds float, float* %2, i64 %289
$i64B

	full_text


i64 %289
LstoreBC
A
	full_text4
2
0store float %293, float* %294, align 4, !tbaa !8
(floatB

	full_text


float %293
*float*B

	full_text

float* %294
LloadBD
B
	full_text5
3
1%295 = load float, float* %159, align 4, !tbaa !8
*float*B

	full_text

float* %159
KloadBC
A
	full_text4
2
0%296 = load float, float* %91, align 4, !tbaa !8
)float*B

	full_text


float* %91
7fmulB/
-
	full_text 

%297 = fmul float %295, %296
(floatB

	full_text


float %295
(floatB

	full_text


float %296
LloadBD
B
	full_text5
3
1%298 = load float, float* %269, align 4, !tbaa !8
*float*B

	full_text

float* %269
LloadBD
B
	full_text5
3
1%299 = load float, float* %130, align 4, !tbaa !8
*float*B

	full_text

float* %130
7fmulB/
-
	full_text 

%300 = fmul float %298, %299
(floatB

	full_text


float %298
(floatB

	full_text


float %299
LfdivBD
B
	full_text5
3
1%301 = fdiv float 1.000000e+00, %300, !fpmath !12
(floatB

	full_text


float %300
7fmulB/
-
	full_text 

%302 = fmul float %297, %301
(floatB

	full_text


float %297
(floatB

	full_text


float %301
1addB*
(
	full_text

%303 = add i64 %6, 1336
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%304 = getelementptr inbounds float, float* %1, i64 %303
$i64B

	full_text


i64 %303
LloadBD
B
	full_text5
3
1%305 = load float, float* %304, align 4, !tbaa !8
*float*B

	full_text

float* %304
ecallB]
[
	full_textN
L
J%306 = tail call float @_Z4fminff(float %302, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %302
7fmulB/
-
	full_text 

%307 = fmul float %305, %306
(floatB

	full_text


float %305
(floatB

	full_text


float %306
\getelementptrBK
I
	full_text<
:
8%308 = getelementptr inbounds float, float* %2, i64 %303
$i64B

	full_text


i64 %303
LstoreBC
A
	full_text4
2
0store float %307, float* %308, align 4, !tbaa !8
(floatB

	full_text


float %307
*float*B

	full_text

float* %308
LloadBD
B
	full_text5
3
1%309 = load float, float* %159, align 4, !tbaa !8
*float*B

	full_text

float* %159
KloadBC
A
	full_text4
2
0%310 = load float, float* %91, align 4, !tbaa !8
)float*B

	full_text


float* %91
7fmulB/
-
	full_text 

%311 = fmul float %309, %310
(floatB

	full_text


float %309
(floatB

	full_text


float %310
6fmulB.
,
	full_text

%312 = fmul float %12, %311
'floatB

	full_text

	float %12
(floatB

	full_text


float %311
0addB)
'
	full_text

%313 = add i64 %6, 240
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%314 = getelementptr inbounds float, float* %3, i64 %313
$i64B

	full_text


i64 %313
LloadBD
B
	full_text5
3
1%315 = load float, float* %314, align 4, !tbaa !8
*float*B

	full_text

float* %314
LfdivBD
B
	full_text5
3
1%316 = fdiv float 1.000000e+00, %315, !fpmath !12
(floatB

	full_text


float %315
7fmulB/
-
	full_text 

%317 = fmul float %312, %316
(floatB

	full_text


float %312
(floatB

	full_text


float %316
1addB*
(
	full_text

%318 = add i64 %6, 1344
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%319 = getelementptr inbounds float, float* %1, i64 %318
$i64B

	full_text


i64 %318
LloadBD
B
	full_text5
3
1%320 = load float, float* %319, align 4, !tbaa !8
*float*B

	full_text

float* %319
ecallB]
[
	full_textN
L
J%321 = tail call float @_Z4fminff(float %317, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %317
7fmulB/
-
	full_text 

%322 = fmul float %320, %321
(floatB

	full_text


float %320
(floatB

	full_text


float %321
\getelementptrBK
I
	full_text<
:
8%323 = getelementptr inbounds float, float* %2, i64 %318
$i64B

	full_text


i64 %318
LstoreBC
A
	full_text4
2
0store float %322, float* %323, align 4, !tbaa !8
(floatB

	full_text


float %322
*float*B

	full_text

float* %323
LloadBD
B
	full_text5
3
1%324 = load float, float* %109, align 4, !tbaa !8
*float*B

	full_text

float* %109
LloadBD
B
	full_text5
3
1%325 = load float, float* %115, align 4, !tbaa !8
*float*B

	full_text

float* %115
7fmulB/
-
	full_text 

%326 = fmul float %324, %325
(floatB

	full_text


float %324
(floatB

	full_text


float %325
6fmulB.
,
	full_text

%327 = fmul float %12, %326
'floatB

	full_text

	float %12
(floatB

	full_text


float %326
0addB)
'
	full_text

%328 = add i64 %6, 184
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%329 = getelementptr inbounds float, float* %3, i64 %328
$i64B

	full_text


i64 %328
LloadBD
B
	full_text5
3
1%330 = load float, float* %329, align 4, !tbaa !8
*float*B

	full_text

float* %329
LfdivBD
B
	full_text5
3
1%331 = fdiv float 1.000000e+00, %330, !fpmath !12
(floatB

	full_text


float %330
7fmulB/
-
	full_text 

%332 = fmul float %327, %331
(floatB

	full_text


float %327
(floatB

	full_text


float %331
1addB*
(
	full_text

%333 = add i64 %6, 1352
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%334 = getelementptr inbounds float, float* %1, i64 %333
$i64B

	full_text


i64 %333
LloadBD
B
	full_text5
3
1%335 = load float, float* %334, align 4, !tbaa !8
*float*B

	full_text

float* %334
ecallB]
[
	full_textN
L
J%336 = tail call float @_Z4fminff(float %332, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %332
7fmulB/
-
	full_text 

%337 = fmul float %335, %336
(floatB

	full_text


float %335
(floatB

	full_text


float %336
\getelementptrBK
I
	full_text<
:
8%338 = getelementptr inbounds float, float* %2, i64 %333
$i64B

	full_text


i64 %333
LstoreBC
A
	full_text4
2
0store float %337, float* %338, align 4, !tbaa !8
(floatB

	full_text


float %337
*float*B

	full_text

float* %338
LloadBD
B
	full_text5
3
1%339 = load float, float* %109, align 4, !tbaa !8
*float*B

	full_text

float* %109
LloadBD
B
	full_text5
3
1%340 = load float, float* %115, align 4, !tbaa !8
*float*B

	full_text

float* %115
7fmulB/
-
	full_text 

%341 = fmul float %339, %340
(floatB

	full_text


float %339
(floatB

	full_text


float %340
KloadBC
A
	full_text4
2
0%342 = load float, float* %93, align 4, !tbaa !8
)float*B

	full_text


float* %93
KloadBC
A
	full_text4
2
0%343 = load float, float* %91, align 4, !tbaa !8
)float*B

	full_text


float* %91
7fmulB/
-
	full_text 

%344 = fmul float %342, %343
(floatB

	full_text


float %342
(floatB

	full_text


float %343
LfdivBD
B
	full_text5
3
1%345 = fdiv float 1.000000e+00, %344, !fpmath !12
(floatB

	full_text


float %344
7fmulB/
-
	full_text 

%346 = fmul float %341, %345
(floatB

	full_text


float %341
(floatB

	full_text


float %345
1addB*
(
	full_text

%347 = add i64 %6, 1360
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%348 = getelementptr inbounds float, float* %1, i64 %347
$i64B

	full_text


i64 %347
LloadBD
B
	full_text5
3
1%349 = load float, float* %348, align 4, !tbaa !8
*float*B

	full_text

float* %348
ecallB]
[
	full_textN
L
J%350 = tail call float @_Z4fminff(float %346, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %346
7fmulB/
-
	full_text 

%351 = fmul float %349, %350
(floatB

	full_text


float %349
(floatB

	full_text


float %350
\getelementptrBK
I
	full_text<
:
8%352 = getelementptr inbounds float, float* %2, i64 %347
$i64B

	full_text


i64 %347
LstoreBC
A
	full_text4
2
0store float %351, float* %352, align 4, !tbaa !8
(floatB

	full_text


float %351
*float*B

	full_text

float* %352
KloadBC
A
	full_text4
2
0%353 = load float, float* %14, align 4, !tbaa !8
)float*B

	full_text


float* %14
LloadBD
B
	full_text5
3
1%354 = load float, float* %115, align 4, !tbaa !8
*float*B

	full_text

float* %115
7fmulB/
-
	full_text 

%355 = fmul float %353, %354
(floatB

	full_text


float %353
(floatB

	full_text


float %354
LloadBD
B
	full_text5
3
1%356 = load float, float* %159, align 4, !tbaa !8
*float*B

	full_text

float* %159
KloadBC
A
	full_text4
2
0%357 = load float, float* %78, align 4, !tbaa !8
)float*B

	full_text


float* %78
7fmulB/
-
	full_text 

%358 = fmul float %356, %357
(floatB

	full_text


float %356
(floatB

	full_text


float %357
LfdivBD
B
	full_text5
3
1%359 = fdiv float 1.000000e+00, %358, !fpmath !12
(floatB

	full_text


float %358
7fmulB/
-
	full_text 

%360 = fmul float %355, %359
(floatB

	full_text


float %355
(floatB

	full_text


float %359
1addB*
(
	full_text

%361 = add i64 %6, 1368
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%362 = getelementptr inbounds float, float* %1, i64 %361
$i64B

	full_text


i64 %361
LloadBD
B
	full_text5
3
1%363 = load float, float* %362, align 4, !tbaa !8
*float*B

	full_text

float* %362
ecallB]
[
	full_textN
L
J%364 = tail call float @_Z4fminff(float %360, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %360
7fmulB/
-
	full_text 

%365 = fmul float %363, %364
(floatB

	full_text


float %363
(floatB

	full_text


float %364
\getelementptrBK
I
	full_text<
:
8%366 = getelementptr inbounds float, float* %2, i64 %361
$i64B

	full_text


i64 %361
LstoreBC
A
	full_text4
2
0store float %365, float* %366, align 4, !tbaa !8
(floatB

	full_text


float %365
*float*B

	full_text

float* %366
KloadBC
A
	full_text4
2
0%367 = load float, float* %14, align 4, !tbaa !8
)float*B

	full_text


float* %14
LloadBD
B
	full_text5
3
1%368 = load float, float* %115, align 4, !tbaa !8
*float*B

	full_text

float* %115
7fmulB/
-
	full_text 

%369 = fmul float %367, %368
(floatB

	full_text


float %367
(floatB

	full_text


float %368
LloadBD
B
	full_text5
3
1%370 = load float, float* %109, align 4, !tbaa !8
*float*B

	full_text

float* %109
LloadBD
B
	full_text5
3
1%371 = load float, float* %222, align 4, !tbaa !8
*float*B

	full_text

float* %222
7fmulB/
-
	full_text 

%372 = fmul float %370, %371
(floatB

	full_text


float %370
(floatB

	full_text


float %371
LfdivBD
B
	full_text5
3
1%373 = fdiv float 1.000000e+00, %372, !fpmath !12
(floatB

	full_text


float %372
7fmulB/
-
	full_text 

%374 = fmul float %369, %373
(floatB

	full_text


float %369
(floatB

	full_text


float %373
1addB*
(
	full_text

%375 = add i64 %6, 1376
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%376 = getelementptr inbounds float, float* %1, i64 %375
$i64B

	full_text


i64 %375
LloadBD
B
	full_text5
3
1%377 = load float, float* %376, align 4, !tbaa !8
*float*B

	full_text

float* %376
ecallB]
[
	full_textN
L
J%378 = tail call float @_Z4fminff(float %374, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %374
7fmulB/
-
	full_text 

%379 = fmul float %377, %378
(floatB

	full_text


float %377
(floatB

	full_text


float %378
\getelementptrBK
I
	full_text<
:
8%380 = getelementptr inbounds float, float* %2, i64 %375
$i64B

	full_text


i64 %375
LstoreBC
A
	full_text4
2
0store float %379, float* %380, align 4, !tbaa !8
(floatB

	full_text


float %379
*float*B

	full_text

float* %380
KloadBC
A
	full_text4
2
0%381 = load float, float* %52, align 4, !tbaa !8
)float*B

	full_text


float* %52
LloadBD
B
	full_text5
3
1%382 = load float, float* %115, align 4, !tbaa !8
*float*B

	full_text

float* %115
7fmulB/
-
	full_text 

%383 = fmul float %381, %382
(floatB

	full_text


float %381
(floatB

	full_text


float %382
KloadBC
A
	full_text4
2
0%384 = load float, float* %57, align 4, !tbaa !8
)float*B

	full_text


float* %57
KloadBC
A
	full_text4
2
0%385 = load float, float* %91, align 4, !tbaa !8
)float*B

	full_text


float* %91
7fmulB/
-
	full_text 

%386 = fmul float %384, %385
(floatB

	full_text


float %384
(floatB

	full_text


float %385
LfdivBD
B
	full_text5
3
1%387 = fdiv float 1.000000e+00, %386, !fpmath !12
(floatB

	full_text


float %386
7fmulB/
-
	full_text 

%388 = fmul float %383, %387
(floatB

	full_text


float %383
(floatB

	full_text


float %387
1addB*
(
	full_text

%389 = add i64 %6, 1384
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%390 = getelementptr inbounds float, float* %1, i64 %389
$i64B

	full_text


i64 %389
LloadBD
B
	full_text5
3
1%391 = load float, float* %390, align 4, !tbaa !8
*float*B

	full_text

float* %390
ecallB]
[
	full_textN
L
J%392 = tail call float @_Z4fminff(float %388, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %388
7fmulB/
-
	full_text 

%393 = fmul float %391, %392
(floatB

	full_text


float %391
(floatB

	full_text


float %392
\getelementptrBK
I
	full_text<
:
8%394 = getelementptr inbounds float, float* %2, i64 %389
$i64B

	full_text


i64 %389
LstoreBC
A
	full_text4
2
0store float %393, float* %394, align 4, !tbaa !8
(floatB

	full_text


float %393
*float*B

	full_text

float* %394
KloadBC
A
	full_text4
2
0%395 = load float, float* %57, align 4, !tbaa !8
)float*B

	full_text


float* %57
LloadBD
B
	full_text5
3
1%396 = load float, float* %115, align 4, !tbaa !8
*float*B

	full_text

float* %115
7fmulB/
-
	full_text 

%397 = fmul float %395, %396
(floatB

	full_text


float %395
(floatB

	full_text


float %396
KloadBC
A
	full_text4
2
0%398 = load float, float* %52, align 4, !tbaa !8
)float*B

	full_text


float* %52
LloadBD
B
	full_text5
3
1%399 = load float, float* %329, align 4, !tbaa !8
*float*B

	full_text

float* %329
7fmulB/
-
	full_text 

%400 = fmul float %398, %399
(floatB

	full_text


float %398
(floatB

	full_text


float %399
LfdivBD
B
	full_text5
3
1%401 = fdiv float 1.000000e+00, %400, !fpmath !12
(floatB

	full_text


float %400
7fmulB/
-
	full_text 

%402 = fmul float %397, %401
(floatB

	full_text


float %397
(floatB

	full_text


float %401
1addB*
(
	full_text

%403 = add i64 %6, 1392
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%404 = getelementptr inbounds float, float* %1, i64 %403
$i64B

	full_text


i64 %403
LloadBD
B
	full_text5
3
1%405 = load float, float* %404, align 4, !tbaa !8
*float*B

	full_text

float* %404
ecallB]
[
	full_textN
L
J%406 = tail call float @_Z4fminff(float %402, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %402
7fmulB/
-
	full_text 

%407 = fmul float %405, %406
(floatB

	full_text


float %405
(floatB

	full_text


float %406
\getelementptrBK
I
	full_text<
:
8%408 = getelementptr inbounds float, float* %2, i64 %403
$i64B

	full_text


i64 %403
LstoreBC
A
	full_text4
2
0store float %407, float* %408, align 4, !tbaa !8
(floatB

	full_text


float %407
*float*B

	full_text

float* %408
"retB

	full_text


ret void
*float*8B

	full_text

	float* %3
*float*8B

	full_text

	float* %2
*float*8B

	full_text

	float* %0
(float8B

	full_text


float %4
*float*8B
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
$i648B

	full_text


i64 72
&i648B

	full_text


i64 1216
%i648B

	full_text
	
i64 176
%i648B

	full_text
	
i64 104
$i648B

	full_text


i64 96
&i648B

	full_text


i64 1240
%i648B

	full_text
	
i64 240
%i648B

	full_text
	
i64 152
2float8B%
#
	full_text

float 1.013250e+06
$i648B

	full_text


i64 24
&i648B

	full_text


i64 1248
&i648B

	full_text


i64 1392
%i648B

	full_text
	
i64 168
$i648B

	full_text


i64 80
&i648B

	full_text


i64 1208
%i648B

	full_text
	
i64 200
&i648B

	full_text


i64 1232
&i648B

	full_text


i64 1288
&i648B

	full_text


i64 1328
&i648B

	full_text


i64 1336
#i328B

	full_text	

i32 0
%i648B

	full_text
	
i64 160
&i648B

	full_text


i64 1360
&i648B

	full_text


i64 1320
&i648B

	full_text


i64 1376
%i648B

	full_text
	
i64 208
$i648B

	full_text


i64 40
&i648B

	full_text


i64 1264
%i648B

	full_text
	
i64 120
&i648B

	full_text


i64 1272
$i648B

	full_text


i64 16
8float8B+
)
	full_text

float 0x4415AF1D80000000
$i648B

	full_text


i64 32
&i648B

	full_text


i64 1224
&i648B

	full_text


i64 1280
&i648B

	full_text


i64 1200
&i648B

	full_text


i64 1296
2float8B%
#
	full_text

float 1.000000e+00
%i648B

	full_text
	
i64 184
&i648B

	full_text


i64 1256
&i648B

	full_text


i64 1344
&i648B

	full_text


i64 1352
&i648B

	full_text


i64 1384
$i648B

	full_text


i64 48
%i648B

	full_text
	
i64 128
&i648B

	full_text


i64 1304
&i648B

	full_text


i64 1312
8float8B+
)
	full_text

float 0x4193D2C640000000
#i648B

	full_text	

i64 8
%i648B

	full_text
	
i64 216
&i648B

	full_text


i64 1368
$i648B

	full_text


i64 88
%i648B

	full_text
	
i64 224       	  
 

                       !" !! #$ ## %& %% '( '' )* )+ )) ,- ,, ./ .0 .. 12 11 34 33 56 55 78 77 9: 9; 99 <= << >? >@ >> AB AA CD CC EF EG EE HI HH JK JJ LM LL NO NN PQ PR PP ST SS UV UW UU XY XX Z[ ZZ \] \\ ^_ ^^ `a `b `` cd cc ef eg ee hi hh jk jj lm ll no nn pq pr pp st ss uv uu wx ww yz yy {| {} {{ ~ ~~ ÄÅ Ä
Ç ÄÄ ÉÑ ÉÉ Ö
Ü ÖÖ áà áá âä ââ ãå ã
ç ãã é
è éé êë ê
í êê ìî ìì ïñ ïï óò ó
ô óó öõ öö úù úú û
ü ûû †° †† ¢£ ¢
§ ¢¢ •¶ •• ß
® ßß ©™ ©© ´¨ ´
≠ ´´ ÆØ Æ
∞ ÆÆ ±
≤ ±± ≥¥ ≥
µ ≥≥ ∂∑ ∂∂ ∏
π ∏∏ ∫ª ∫∫ ºΩ ºº æø æ
¿ ææ ¡
¬ ¡¡ √ƒ √
≈ √√ ∆« ∆∆ »
… »»  À    Ã
Õ ÃÃ Œœ ŒŒ –— –– “
” ““ ‘’ ‘‘ ÷◊ ÷
ÿ ÷÷ Ÿ⁄ Ÿ
€ ŸŸ ‹
› ‹‹ ﬁﬂ ﬁ
‡ ﬁﬁ ·‚ ·· „
‰ „„ ÂÊ ÂÂ ÁË ÁÁ ÈÍ È
Î ÈÈ Ï
Ì ÏÏ ÓÔ Ó
 ÓÓ ÒÚ ÒÒ Û
Ù ÛÛ ıˆ ıı ˜¯ ˜˜ ˘˙ ˘
˚ ˘˘ ¸˝ ¸
˛ ¸¸ ˇÄ ˇˇ Å
Ç ÅÅ ÉÑ ÉÉ Ö
Ü ÖÖ áà á
â áá äã ää å
ç åå éè éé êë êê íì í
î íí ï
ñ ïï óò ó
ô óó öõ öö úù úú ûü û
† ûû °¢ °° £§ ££ •
¶ •• ß® ßß ©™ ©
´ ©© ¨
≠ ¨¨ ÆØ Æ
∞ ÆÆ ±≤ ±± ≥
¥ ≥≥ µ∂ µµ ∑∏ ∑∑ π∫ π
ª ππ º
Ω ºº æø æ
¿ ææ ¡¬ ¡¡ √ƒ √√ ≈∆ ≈
« ≈≈ »… »»  À    ÃÕ Ã
Œ ÃÃ œ
– œœ —“ —
” —— ‘’ ‘‘ ÷
◊ ÷÷ ÿŸ ÿÿ ⁄€ ⁄⁄ ‹› ‹
ﬁ ‹‹ ﬂ
‡ ﬂﬂ ·‚ ·
„ ·· ‰Â ‰‰ ÊÁ ÊÊ ËÈ Ë
Í ËË ÎÏ ÎÎ Ì
Ó ÌÌ Ô ÔÔ ÒÚ ÒÒ Û
Ù ÛÛ ıˆ ıı ˜¯ ˜
˘ ˜˜ ˙
˚ ˙˙ ¸˝ ¸
˛ ¸¸ ˇÄ ˇˇ Å
Ç ÅÅ ÉÑ ÉÉ ÖÜ ÖÖ áà á
â áá ä
ã ää åç å
é åå èê èè ëí ëë ìî ì
ï ìì ñó ññ ò
ô òò öõ öö úù úú ûü û
† ûû °
¢ °° £§ £
• ££ ¶ß ¶¶ ®
© ®® ™´ ™™ ¨≠ ¨¨ ÆØ Æ
∞ ÆÆ ±
≤ ±± ≥¥ ≥
µ ≥≥ ∂∑ ∂∂ ∏π ∏∏ ∫ª ∫
º ∫∫ Ωæ ΩΩ ø¿ øø ¡¬ ¡
√ ¡¡ ƒ
≈ ƒƒ ∆« ∆
» ∆∆ …  …… À
Ã ÀÀ ÕŒ ÕÕ œ– œœ —“ —
” —— ‘
’ ‘‘ ÷◊ ÷
ÿ ÷÷ Ÿ⁄ ŸŸ €‹ €€ ›ﬁ ›
ﬂ ›› ‡· ‡‡ ‚„ ‚‚ ‰Â ‰
Ê ‰‰ Á
Ë ÁÁ ÈÍ È
Î ÈÈ ÏÌ ÏÏ Ó
Ô ÓÓ Ò  ÚÛ ÚÚ Ùı Ù
ˆ ÙÙ ˜
¯ ˜˜ ˘˙ ˘
˚ ˘˘ ¸˝ ¸¸ ˛ˇ ˛˛ ÄÅ Ä
Ç ÄÄ ÉÑ ÉÉ ÖÜ ÖÖ á
à áá âä ââ ãå ã
ç ãã é
è éé êë ê
í êê ìî ìì ï
ñ ïï óò óó ôö ôô õú õ
ù õõ û
ü ûû †° †
¢ †† £§ ££ •¶ •• ß® ß
© ßß ™´ ™™ ¨≠ ¨¨ ÆØ Æ
∞ ÆÆ ±
≤ ±± ≥¥ ≥
µ ≥≥ ∂∑ ∂∂ ∏
π ∏∏ ∫ª ∫∫ ºΩ ºº æø æ
¿ ææ ¡
¬ ¡¡ √ƒ √
≈ √√ ∆« ∆∆ »… »»  À  
Ã    ÕŒ ÕÕ œ– œœ —
“ —— ”‘ ”” ’÷ ’
◊ ’’ ÿ
Ÿ ÿÿ ⁄€ ⁄
‹ ⁄⁄ ›ﬁ ›› ﬂ
‡ ﬂﬂ ·‚ ·· „‰ „„ ÂÊ Â
Á ÂÂ Ë
È ËË ÍÎ Í
Ï ÍÍ ÌÓ ÌÌ Ô
 ÔÔ ÒÚ ÒÒ ÛÙ ÛÛ ıˆ ı
˜ ıı ¯˘ ¯¯ ˙
˚ ˙˙ ¸˝ ¸¸ ˛ˇ ˛˛ ÄÅ Ä
Ç ÄÄ É
Ñ ÉÉ ÖÜ Ö
á ÖÖ àâ àà ä
ã ää åç åå éè éé êë ê
í êê ì
î ìì ïñ ï
ó ïï òô òò öõ öö úù ú
û úú ü† üü °¢ °° £§ £
• ££ ¶
ß ¶¶ ®© ®
™ ®® ´¨ ´´ ≠
Æ ≠≠ Ø∞ ØØ ±≤ ±± ≥¥ ≥
µ ≥≥ ∂
∑ ∂∂ ∏π ∏
∫ ∏∏ ªº ªª Ωæ ΩΩ ø¿ ø
¡ øø ¬√ ¬¬ ƒ≈ ƒƒ ∆« ∆
» ∆∆ …
  …… ÀÃ À
Õ ÀÀ Œœ ŒŒ –
— –– “” ““ ‘’ ‘‘ ÷◊ ÷
ÿ ÷÷ Ÿ
⁄ ŸŸ €‹ €
› €€ ﬁﬂ ﬁﬁ ‡· ‡‡ ‚„ ‚
‰ ‚‚ ÂÊ Â
Á ÂÂ ËÈ ËË Í
Î ÍÍ ÏÌ ÏÏ Ó
Ô ÓÓ Ò 
Ú  ÛÙ ÛÛ ı
ˆ ıı ˜¯ ˜˜ ˘˙ ˘˘ ˚¸ ˚
˝ ˚˚ ˛
ˇ ˛˛ ÄÅ Ä
Ç ÄÄ ÉÑ ÉÉ ÖÜ ÖÖ áà á
â áá äã ä
å ää çé çç è
ê èè ëí ëë ì
î ìì ïñ ï
ó ïï òô òò ö
õ öö úù úú ûü ûû †° †
¢ †† £
§ ££ •¶ •
ß •• ®© ®® ™´ ™™ ¨≠ ¨
Æ ¨¨ Ø∞ ØØ ±≤ ±± ≥¥ ≥
µ ≥≥ ∂
∑ ∂∂ ∏π ∏
∫ ∏∏ ªº ªª Ω
æ ΩΩ ø¿ øø ¡¬ ¡¡ √ƒ √
≈ √√ ∆
« ∆∆ »… »
  »» ÀÃ ÀÀ ÕŒ ÕÕ œ– œ
— œœ “” ““ ‘’ ‘‘ ÷◊ ÷
ÿ ÷÷ Ÿ
⁄ ŸŸ €‹ €
› €€ ﬁﬂ ﬁﬁ ‡
· ‡‡ ‚„ ‚‚ ‰Â ‰‰ ÊÁ Ê
Ë ÊÊ È
Í ÈÈ ÎÏ Î
Ì ÎÎ ÓÔ ÓÓ Ò  ÚÛ Ú
Ù ÚÚ ıˆ ıı ˜¯ ˜˜ ˘˙ ˘
˚ ˘˘ ¸
˝ ¸¸ ˛ˇ ˛
Ä ˛˛ ÅÇ ÅÅ É
Ñ ÉÉ ÖÜ ÖÖ áà áá âä â
ã ââ å
ç åå éè é
ê éé ëí ëë ìî ìì ïñ ï
ó ïï òô òò öõ öö úù ú
û úú ü
† üü °¢ °
£ °° §• §§ ¶
ß ¶¶ ®© ®® ™´ ™™ ¨≠ ¨
Æ ¨¨ Ø
∞ ØØ ±≤ ±
≥ ±± ¥µ ¥¥ ∂∑ ∂∂ ∏π ∏
∫ ∏∏ ªº ªª Ωæ ΩΩ ø¿ ø
¡ øø ¬
√ ¬¬ ƒ≈ ƒ
∆ ƒƒ «» «« …
  …… ÀÃ ÀÀ ÕŒ ÕÕ œ– œ
— œœ “
” ““ ‘’ ‘
÷ ‘‘ ◊ÿ ÿ ÿ ÿ %ÿ Jÿ jÿ uÿ ûÿ ßÿ »ÿ Ãÿ “ÿ Ûÿ Åÿ •ÿ Ìÿ Ûÿ òÿ áÿ —ÿ Ôÿ ˙ÿ Íÿ èŸ <Ÿ cŸ éŸ ¡Ÿ ÏŸ ïŸ ºŸ ﬂŸ äŸ ±Ÿ ‘Ÿ ˜Ÿ ûŸ ¡Ÿ ËŸ ìŸ ∂Ÿ ŸŸ ˛Ÿ £Ÿ ∆Ÿ ÈŸ åŸ ØŸ “⁄ 	€ ‹ 3‹ Z‹ Ö‹ ∏‹ „‹ å‹ ≥‹ ÷‹ Å‹ ®‹ À‹ Ó‹ ï‹ ∏‹ ﬂ‹ ä‹ ≠‹ –‹ ı‹ ö‹ Ω‹ ‡‹ É‹ ¶‹ …    	 
             " $# &% (! *' +) - /, 0 21 43 6. 85 :7 ;1 =9 ?< @ B DA FC G IH KJ M% OL QN RP TE VS W YX [Z ]U _\ a^ bX d` fc g ih kj m ol qn r ts vu x% zw |y }{ p Å~ Ç ÑÉ ÜÖ àÄ äá åâ çÉ èã ëé íj î ñì òï ô õ ùú üû °ö £† § ¶• ®ß ™¢ ¨© ≠ Ø´ ∞Æ ≤ó ¥± µ ∑∂ π∏ ª≥ Ω∫ øº ¿∂ ¬æ ƒ¡ ≈ «∆ …» À ÕÃ œ —– ”“ ’Œ ◊‘ ÿ ⁄÷ €Ÿ ›  ﬂ‹ ‡ ‚· ‰„ Êﬁ ËÂ ÍÁ Î· ÌÈ ÔÏ  ÚÒ ÙÛ ˆ» ¯ı ˙˜ ˚ ˝˘ ˛ Äˇ ÇÅ ÑÉ Ü¸ àÖ â ãä çå èá ëé ìê îä ñí òï ôÛ õ» ùö üú †Ã ¢ §£ ¶• ®° ™ß ´© ≠û Ø¨ ∞ ≤± ¥≥ ∂Æ ∏µ ∫∑ ª± Ωπ øº ¿ ¬» ƒ¡ ∆√ « …• À» Õ  ŒÃ –≈ “œ ” ’‘ ◊÷ Ÿ— €ÿ ›⁄ ﬁ‘ ‡‹ ‚ﬂ „ Â» Á‰ ÈÊ Í ÏÎ ÓÌ  ÚÒ ÙÛ ˆÔ ¯ı ˘˜ ˚Ë ˝˙ ˛ Äˇ ÇÅ Ñ¸ ÜÉ àÖ âˇ ãá çä é ê» íè îë ï óñ ôò õß ùö üú †û ¢ì §° • ß¶ ©® ´£ ≠™ Ø¨ ∞¶ ≤Æ ¥± µ ∑» π∂ ª∏ ºJ æ• ¿Ω ¬ø √¡ ≈∫ «ƒ »  … ÃÀ Œ∆ –Õ “œ ”… ’— ◊‘ ÿj ⁄» ‹Ÿ ﬁ€ ﬂu ·• „‡ Â‚ Ê‰ Ë› ÍÁ Î ÌÏ ÔÓ ÒÈ Û ıÚ ˆÏ ¯Ù ˙˜ ˚u ˝» ˇ¸ Å˛ Ç Ñ ÜÖ àá äÉ åâ çã èÄ ëé í îì ñï òê öó úô ùì üõ °û ¢Û §» ¶£ ®• ©û ´Å ≠™ Ø¨ ∞Æ ≤ß ¥± µ ∑∂ π∏ ª≥ Ω∫ øº ¿∂ ¬æ ƒ¡ ≈ò «» …∆ À» ÃÛ Œ –œ “— ‘Õ ÷” ◊’ Ÿ  €ÿ ‹ ﬁ› ‡ﬂ ‚⁄ ‰· Ê„ Á› ÈÂ ÎË Ï ÓÌ Ô Ú» ÙÒ ˆÛ ˜ ˘¯ ˚˙ ˝“ ˇ¸ Å˛ ÇÄ Ñı ÜÉ á âà ãä çÖ èå ëé íà îê ñì óÔ ô» õò ùö ûÛ †— ¢ü §° •£ ßú ©¶ ™ ¨´ Æ≠ ∞® ≤Ø ¥± µ´ ∑≥ π∂ ∫Ì º» æª ¿Ω ¡˙ √• ≈¬ «ƒ »∆  ø Ã… Õ œŒ —– ”À ’“ ◊‘ ÿŒ ⁄÷ ‹Ÿ ›Ì ﬂ» ·ﬁ „‡ ‰ Ê‚ Á ÈË ÎÍ ÌÏ ÔÂ ÒÓ Ú ÙÛ ˆı ¯ ˙˜ ¸˘ ˝Û ˇ˚ Å˛ ÇÛ ÑÅ ÜÉ àÖ â ãá å éç êè íë îä ñì ó ôò õö ùï üú °û ¢ò §† ¶£ ßÛ ©Å ´® ≠™ ÆÃ ∞» ≤Ø ¥± µ≥ ∑¨ π∂ ∫ ºª æΩ ¿∏ ¬ø ƒ¡ ≈ª «√ …∆   ÃÅ ŒÀ –Õ —Ì ”ß ’“ ◊‘ ÿ÷ ⁄œ ‹Ÿ › ﬂﬁ ·‡ „€ Â‚ Á‰ Ëﬁ ÍÊ ÏÈ Ì ÔÅ ÒÓ Û ÙÛ ˆá ¯ı ˙˜ ˚˘ ˝Ú ˇ¸ Ä ÇÅ ÑÉ Ü˛ àÖ äá ãÅ çâ èå êj íÅ îë ñì óu ô» õò ùö ûú †ï ¢ü £ •§ ß¶ ©° ´® ≠™ Æ§ ∞¨ ≤Ø ≥u µÅ ∑¥ π∂ ∫j ºè æª ¿Ω ¡ø √∏ ≈¬ ∆ »«  … Ãƒ ŒÀ –Õ —« ”œ ’“ ÷ ◊ ›› ﬁﬁº ﬁﬁ º ›› ô ﬁﬁ ôâ ﬁﬁ â˘ ﬁﬁ ˘û ﬁﬁ û¡ ﬁﬁ ¡™ ﬁﬁ ™^ ﬁﬁ ^º ﬁﬁ º⁄ ﬁﬁ ⁄Õ ﬁﬁ Õ‘ ﬁﬁ ‘‰ ﬁﬁ ‰á ﬁﬁ áê ﬁﬁ êÚ ﬁﬁ Úœ ﬁﬁ œÁ ﬁﬁ Á¨ ﬁﬁ ¨± ﬁﬁ ±é ﬁﬁ éÖ ﬁﬁ Ö7 ﬁﬁ 7∑ ﬁﬁ ∑„ ﬁﬁ „
ﬂ ñ
‡ É
· ˇ
‚ ú
„ ¯
‰ ä
Â Ë
Ê –	Á 	Ë h
È ±
Í «
Î ∆
Ï Ì	Ì X	Ó #
Ô ·
 Ï
Ò ´
Ú ŒÛ 
Ù £
ı ª
ˆ à
˜ Å	¯ 	˘ H
˙ ˇ
˚ Ò
¸ ¶	˝ 	˛ 7	˛ ^
˛ â
˛ º
˛ Á
˛ ê
˛ ∑
˛ ⁄
˛ Ö
˛ ¨
˛ œ
˛ Ú
˛ ô
˛ º
˛ „
˛ é
˛ ±
˛ ‘
˛ ˘
˛ û
˛ ¡
˛ ‰
˛ á
˛ ™
˛ Õ	ˇ 
Ä ∂
Å …	Ç 1
É ìÑ 
Ñ ,Ñ SÑ ~Ñ ±Ñ ‹Ñ ÖÑ ¨Ñ œÑ ˙Ñ °Ñ ƒÑ ÁÑ éÑ ±Ñ ÿÑ ÉÑ ¶Ñ …Ñ ÓÑ ìÑ ∂Ñ ŸÑ ¸Ñ üÑ ¬
Ö ç
Ü ‘
á Û
à ò
â §	ä s
ã •
å ∂
ç ›	é 
è Ò
ê Ö
ë ﬁ
í Î
ì œ"
ratt8_kernel"
_Z13get_global_idj"
	_Z4fminff*ó
shoc-1.1.5-S3D-ratt8_kernel.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02Å

wgsize
Ä

wgsize_log1p
íÛéA

devmap_label
 
 
transfer_bytes_log1p
íÛéA

transfer_bytes
à¢ª