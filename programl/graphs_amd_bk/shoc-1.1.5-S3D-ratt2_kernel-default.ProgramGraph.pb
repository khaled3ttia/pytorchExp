
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
-addB&
$
	full_text

%13 = add i64 %6, 8
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
.addB'
%
	full_text

%16 = add i64 %6, 24
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
%20 = add i64 %6, 16
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
.addB'
%
	full_text

%23 = add i64 %6, 32
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
YgetelementptrBH
F
	full_text9
7
5%29 = getelementptr inbounds float, float* %1, i64 %6
"i64B

	full_text


i64 %6
JloadBB
@
	full_text3
1
/%30 = load float, float* %29, align 4, !tbaa !8
)float*B

	full_text


float* %29
ccallB[
Y
	full_textL
J
H%31 = tail call float @_Z4fminff(float %28, float 0x4415AF1D80000000) #2
'floatB

	full_text

	float %28
4fmulB,
*
	full_text

%32 = fmul float %30, %31
'floatB

	full_text

	float %30
'floatB

	full_text

	float %31
YgetelementptrBH
F
	full_text9
7
5%33 = getelementptr inbounds float, float* %2, i64 %6
"i64B

	full_text


i64 %6
JstoreBA
?
	full_text2
0
.store float %32, float* %33, align 4, !tbaa !8
'floatB

	full_text

	float %32
)float*B

	full_text


float* %33
YgetelementptrBH
F
	full_text9
7
5%34 = getelementptr inbounds float, float* %3, i64 %6
"i64B

	full_text


i64 %6
JloadBB
@
	full_text3
1
/%35 = load float, float* %34, align 4, !tbaa !8
)float*B

	full_text


float* %34
JloadBB
@
	full_text3
1
/%36 = load float, float* %21, align 4, !tbaa !8
)float*B

	full_text


float* %21
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
JloadBB
@
	full_text3
1
/%38 = load float, float* %14, align 4, !tbaa !8
)float*B

	full_text


float* %14
JloadBB
@
	full_text3
1
/%39 = load float, float* %24, align 4, !tbaa !8
)float*B

	full_text


float* %24
4fmulB,
*
	full_text

%40 = fmul float %38, %39
'floatB

	full_text

	float %38
'floatB

	full_text

	float %39
JfdivBB
@
	full_text3
1
/%41 = fdiv float 1.000000e+00, %40, !fpmath !12
'floatB

	full_text

	float %40
4fmulB,
*
	full_text

%42 = fmul float %37, %41
'floatB

	full_text

	float %37
'floatB

	full_text

	float %41
ZgetelementptrBI
G
	full_text:
8
6%43 = getelementptr inbounds float, float* %1, i64 %13
#i64B

	full_text
	
i64 %13
JloadBB
@
	full_text3
1
/%44 = load float, float* %43, align 4, !tbaa !8
)float*B

	full_text


float* %43
ccallB[
Y
	full_textL
J
H%45 = tail call float @_Z4fminff(float %42, float 0x4415AF1D80000000) #2
'floatB

	full_text

	float %42
4fmulB,
*
	full_text

%46 = fmul float %44, %45
'floatB

	full_text

	float %44
'floatB

	full_text

	float %45
ZgetelementptrBI
G
	full_text:
8
6%47 = getelementptr inbounds float, float* %2, i64 %13
#i64B

	full_text
	
i64 %13
JstoreBA
?
	full_text2
0
.store float %46, float* %47, align 4, !tbaa !8
'floatB

	full_text

	float %46
)float*B

	full_text


float* %47
JloadBB
@
	full_text3
1
/%48 = load float, float* %34, align 4, !tbaa !8
)float*B

	full_text


float* %34
JloadBB
@
	full_text3
1
/%49 = load float, float* %24, align 4, !tbaa !8
)float*B

	full_text


float* %24
4fmulB,
*
	full_text

%50 = fmul float %48, %49
'floatB

	full_text

	float %48
'floatB

	full_text

	float %49
JloadBB
@
	full_text3
1
/%51 = load float, float* %14, align 4, !tbaa !8
)float*B

	full_text


float* %14
.addB'
%
	full_text

%52 = add i64 %6, 40
"i64B

	full_text


i64 %6
ZgetelementptrBI
G
	full_text:
8
6%53 = getelementptr inbounds float, float* %3, i64 %52
#i64B

	full_text
	
i64 %52
JloadBB
@
	full_text3
1
/%54 = load float, float* %53, align 4, !tbaa !8
)float*B

	full_text


float* %53
4fmulB,
*
	full_text

%55 = fmul float %51, %54
'floatB

	full_text

	float %51
'floatB

	full_text

	float %54
JfdivBB
@
	full_text3
1
/%56 = fdiv float 1.000000e+00, %55, !fpmath !12
'floatB

	full_text

	float %55
4fmulB,
*
	full_text

%57 = fmul float %50, %56
'floatB

	full_text

	float %50
'floatB

	full_text

	float %56
ZgetelementptrBI
G
	full_text:
8
6%58 = getelementptr inbounds float, float* %1, i64 %20
#i64B

	full_text
	
i64 %20
JloadBB
@
	full_text3
1
/%59 = load float, float* %58, align 4, !tbaa !8
)float*B

	full_text


float* %58
ccallB[
Y
	full_textL
J
H%60 = tail call float @_Z4fminff(float %57, float 0x4415AF1D80000000) #2
'floatB

	full_text

	float %57
4fmulB,
*
	full_text

%61 = fmul float %59, %60
'floatB

	full_text

	float %59
'floatB

	full_text

	float %60
ZgetelementptrBI
G
	full_text:
8
6%62 = getelementptr inbounds float, float* %2, i64 %20
#i64B

	full_text
	
i64 %20
JstoreBA
?
	full_text2
0
.store float %61, float* %62, align 4, !tbaa !8
'floatB

	full_text

	float %61
)float*B

	full_text


float* %62
JloadBB
@
	full_text3
1
/%63 = load float, float* %24, align 4, !tbaa !8
)float*B

	full_text


float* %24
4fmulB,
*
	full_text

%64 = fmul float %63, %63
'floatB

	full_text

	float %63
'floatB

	full_text

	float %63
JloadBB
@
	full_text3
1
/%65 = load float, float* %21, align 4, !tbaa !8
)float*B

	full_text


float* %21
JloadBB
@
	full_text3
1
/%66 = load float, float* %53, align 4, !tbaa !8
)float*B

	full_text


float* %53
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
JfdivBB
@
	full_text3
1
/%68 = fdiv float 1.000000e+00, %67, !fpmath !12
'floatB

	full_text

	float %67
4fmulB,
*
	full_text

%69 = fmul float %64, %68
'floatB

	full_text

	float %64
'floatB

	full_text

	float %68
ZgetelementptrBI
G
	full_text:
8
6%70 = getelementptr inbounds float, float* %1, i64 %16
#i64B

	full_text
	
i64 %16
JloadBB
@
	full_text3
1
/%71 = load float, float* %70, align 4, !tbaa !8
)float*B

	full_text


float* %70
ccallB[
Y
	full_textL
J
H%72 = tail call float @_Z4fminff(float %69, float 0x4415AF1D80000000) #2
'floatB

	full_text

	float %69
4fmulB,
*
	full_text

%73 = fmul float %71, %72
'floatB

	full_text

	float %71
'floatB

	full_text

	float %72
ZgetelementptrBI
G
	full_text:
8
6%74 = getelementptr inbounds float, float* %2, i64 %16
#i64B

	full_text
	
i64 %16
JstoreBA
?
	full_text2
0
.store float %73, float* %74, align 4, !tbaa !8
'floatB

	full_text

	float %73
)float*B

	full_text


float* %74
JloadBB
@
	full_text3
1
/%75 = load float, float* %14, align 4, !tbaa !8
)float*B

	full_text


float* %14
4fmulB,
*
	full_text

%76 = fmul float %75, %75
'floatB

	full_text

	float %75
'floatB

	full_text

	float %75
4fmulB,
*
	full_text

%77 = fmul float %12, %76
'floatB

	full_text

	float %12
'floatB

	full_text

	float %76
JloadBB
@
	full_text3
1
/%78 = load float, float* %34, align 4, !tbaa !8
)float*B

	full_text


float* %34
JfdivBB
@
	full_text3
1
/%79 = fdiv float 1.000000e+00, %78, !fpmath !12
'floatB

	full_text

	float %78
4fmulB,
*
	full_text

%80 = fmul float %77, %79
'floatB

	full_text

	float %77
'floatB

	full_text

	float %79
ZgetelementptrBI
G
	full_text:
8
6%81 = getelementptr inbounds float, float* %1, i64 %23
#i64B

	full_text
	
i64 %23
JloadBB
@
	full_text3
1
/%82 = load float, float* %81, align 4, !tbaa !8
)float*B

	full_text


float* %81
ccallB[
Y
	full_textL
J
H%83 = tail call float @_Z4fminff(float %80, float 0x4415AF1D80000000) #2
'floatB

	full_text

	float %80
4fmulB,
*
	full_text

%84 = fmul float %82, %83
'floatB

	full_text

	float %82
'floatB

	full_text

	float %83
ZgetelementptrBI
G
	full_text:
8
6%85 = getelementptr inbounds float, float* %2, i64 %23
#i64B

	full_text
	
i64 %23
JstoreBA
?
	full_text2
0
.store float %84, float* %85, align 4, !tbaa !8
'floatB

	full_text

	float %84
)float*B

	full_text


float* %85
JloadBB
@
	full_text3
1
/%86 = load float, float* %14, align 4, !tbaa !8
)float*B

	full_text


float* %14
4fmulB,
*
	full_text

%87 = fmul float %86, %86
'floatB

	full_text

	float %86
'floatB

	full_text

	float %86
4fmulB,
*
	full_text

%88 = fmul float %12, %87
'floatB

	full_text

	float %12
'floatB

	full_text

	float %87
JloadBB
@
	full_text3
1
/%89 = load float, float* %34, align 4, !tbaa !8
)float*B

	full_text


float* %34
JfdivBB
@
	full_text3
1
/%90 = fdiv float 1.000000e+00, %89, !fpmath !12
'floatB

	full_text

	float %89
4fmulB,
*
	full_text

%91 = fmul float %88, %90
'floatB

	full_text

	float %88
'floatB

	full_text

	float %90
ZgetelementptrBI
G
	full_text:
8
6%92 = getelementptr inbounds float, float* %1, i64 %52
#i64B

	full_text
	
i64 %52
JloadBB
@
	full_text3
1
/%93 = load float, float* %92, align 4, !tbaa !8
)float*B

	full_text


float* %92
ccallB[
Y
	full_textL
J
H%94 = tail call float @_Z4fminff(float %91, float 0x4415AF1D80000000) #2
'floatB

	full_text

	float %91
4fmulB,
*
	full_text

%95 = fmul float %93, %94
'floatB

	full_text

	float %93
'floatB

	full_text

	float %94
ZgetelementptrBI
G
	full_text:
8
6%96 = getelementptr inbounds float, float* %2, i64 %52
#i64B

	full_text
	
i64 %52
JstoreBA
?
	full_text2
0
.store float %95, float* %96, align 4, !tbaa !8
'floatB

	full_text

	float %95
)float*B

	full_text


float* %96
JloadBB
@
	full_text3
1
/%97 = load float, float* %14, align 4, !tbaa !8
)float*B

	full_text


float* %14
4fmulB,
*
	full_text

%98 = fmul float %97, %97
'floatB

	full_text

	float %97
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
KloadBC
A
	full_text4
2
0%100 = load float, float* %34, align 4, !tbaa !8
)float*B

	full_text


float* %34
LfdivBD
B
	full_text5
3
1%101 = fdiv float 1.000000e+00, %100, !fpmath !12
(floatB

	full_text


float %100
6fmulB.
,
	full_text

%102 = fmul float %99, %101
'floatB

	full_text

	float %99
(floatB

	full_text


float %101
/addB(
&
	full_text

%103 = add i64 %6, 48
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%104 = getelementptr inbounds float, float* %1, i64 %103
$i64B

	full_text


i64 %103
LloadBD
B
	full_text5
3
1%105 = load float, float* %104, align 4, !tbaa !8
*float*B

	full_text

float* %104
ecallB]
[
	full_textN
L
J%106 = tail call float @_Z4fminff(float %102, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %102
7fmulB/
-
	full_text 

%107 = fmul float %105, %106
(floatB

	full_text


float %105
(floatB

	full_text


float %106
\getelementptrBK
I
	full_text<
:
8%108 = getelementptr inbounds float, float* %2, i64 %103
$i64B

	full_text


i64 %103
LstoreBC
A
	full_text4
2
0store float %107, float* %108, align 4, !tbaa !8
(floatB

	full_text


float %107
*float*B

	full_text

float* %108
KloadBC
A
	full_text4
2
0%109 = load float, float* %14, align 4, !tbaa !8
)float*B

	full_text


float* %14
7fmulB/
-
	full_text 

%110 = fmul float %109, %109
(floatB

	full_text


float %109
(floatB

	full_text


float %109
6fmulB.
,
	full_text

%111 = fmul float %12, %110
'floatB

	full_text

	float %12
(floatB

	full_text


float %110
KloadBC
A
	full_text4
2
0%112 = load float, float* %34, align 4, !tbaa !8
)float*B

	full_text


float* %34
LfdivBD
B
	full_text5
3
1%113 = fdiv float 1.000000e+00, %112, !fpmath !12
(floatB

	full_text


float %112
7fmulB/
-
	full_text 

%114 = fmul float %111, %113
(floatB

	full_text


float %111
(floatB

	full_text


float %113
/addB(
&
	full_text

%115 = add i64 %6, 56
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%116 = getelementptr inbounds float, float* %1, i64 %115
$i64B

	full_text


i64 %115
LloadBD
B
	full_text5
3
1%117 = load float, float* %116, align 4, !tbaa !8
*float*B

	full_text

float* %116
ecallB]
[
	full_textN
L
J%118 = tail call float @_Z4fminff(float %114, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %114
7fmulB/
-
	full_text 

%119 = fmul float %117, %118
(floatB

	full_text


float %117
(floatB

	full_text


float %118
\getelementptrBK
I
	full_text<
:
8%120 = getelementptr inbounds float, float* %2, i64 %115
$i64B

	full_text


i64 %115
LstoreBC
A
	full_text4
2
0store float %119, float* %120, align 4, !tbaa !8
(floatB

	full_text


float %119
*float*B

	full_text

float* %120
KloadBC
A
	full_text4
2
0%121 = load float, float* %14, align 4, !tbaa !8
)float*B

	full_text


float* %14
KloadBC
A
	full_text4
2
0%122 = load float, float* %24, align 4, !tbaa !8
)float*B

	full_text


float* %24
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
6fmulB.
,
	full_text

%124 = fmul float %12, %123
'floatB

	full_text

	float %12
(floatB

	full_text


float %123
KloadBC
A
	full_text4
2
0%125 = load float, float* %53, align 4, !tbaa !8
)float*B

	full_text


float* %53
LfdivBD
B
	full_text5
3
1%126 = fdiv float 1.000000e+00, %125, !fpmath !12
(floatB

	full_text


float %125
7fmulB/
-
	full_text 

%127 = fmul float %124, %126
(floatB

	full_text


float %124
(floatB

	full_text


float %126
/addB(
&
	full_text

%128 = add i64 %6, 64
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%129 = getelementptr inbounds float, float* %1, i64 %128
$i64B

	full_text


i64 %128
LloadBD
B
	full_text5
3
1%130 = load float, float* %129, align 4, !tbaa !8
*float*B

	full_text

float* %129
ecallB]
[
	full_textN
L
J%131 = tail call float @_Z4fminff(float %127, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %127
7fmulB/
-
	full_text 

%132 = fmul float %130, %131
(floatB

	full_text


float %130
(floatB

	full_text


float %131
\getelementptrBK
I
	full_text<
:
8%133 = getelementptr inbounds float, float* %2, i64 %128
$i64B

	full_text


i64 %128
LstoreBC
A
	full_text4
2
0store float %132, float* %133, align 4, !tbaa !8
(floatB

	full_text


float %132
*float*B

	full_text

float* %133
KloadBC
A
	full_text4
2
0%134 = load float, float* %14, align 4, !tbaa !8
)float*B

	full_text


float* %14
KloadBC
A
	full_text4
2
0%135 = load float, float* %21, align 4, !tbaa !8
)float*B

	full_text


float* %21
7fmulB/
-
	full_text 

%136 = fmul float %134, %135
(floatB

	full_text


float %134
(floatB

	full_text


float %135
6fmulB.
,
	full_text

%137 = fmul float %12, %136
'floatB

	full_text

	float %12
(floatB

	full_text


float %136
KloadBC
A
	full_text4
2
0%138 = load float, float* %24, align 4, !tbaa !8
)float*B

	full_text


float* %24
LfdivBD
B
	full_text5
3
1%139 = fdiv float 1.000000e+00, %138, !fpmath !12
(floatB

	full_text


float %138
7fmulB/
-
	full_text 

%140 = fmul float %137, %139
(floatB

	full_text


float %137
(floatB

	full_text


float %139
/addB(
&
	full_text

%141 = add i64 %6, 72
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%142 = getelementptr inbounds float, float* %1, i64 %141
$i64B

	full_text


i64 %141
LloadBD
B
	full_text5
3
1%143 = load float, float* %142, align 4, !tbaa !8
*float*B

	full_text

float* %142
ecallB]
[
	full_textN
L
J%144 = tail call float @_Z4fminff(float %140, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %140
7fmulB/
-
	full_text 

%145 = fmul float %143, %144
(floatB

	full_text


float %143
(floatB

	full_text


float %144
\getelementptrBK
I
	full_text<
:
8%146 = getelementptr inbounds float, float* %2, i64 %141
$i64B

	full_text


i64 %141
LstoreBC
A
	full_text4
2
0store float %145, float* %146, align 4, !tbaa !8
(floatB

	full_text


float %145
*float*B

	full_text

float* %146
KloadBC
A
	full_text4
2
0%147 = load float, float* %21, align 4, !tbaa !8
)float*B

	full_text


float* %21
7fmulB/
-
	full_text 

%148 = fmul float %147, %147
(floatB

	full_text


float %147
(floatB

	full_text


float %147
6fmulB.
,
	full_text

%149 = fmul float %12, %148
'floatB

	full_text

	float %12
(floatB

	full_text


float %148
KloadBC
A
	full_text4
2
0%150 = load float, float* %17, align 4, !tbaa !8
)float*B

	full_text


float* %17
LfdivBD
B
	full_text5
3
1%151 = fdiv float 1.000000e+00, %150, !fpmath !12
(floatB

	full_text


float %150
7fmulB/
-
	full_text 

%152 = fmul float %149, %151
(floatB

	full_text


float %149
(floatB

	full_text


float %151
/addB(
&
	full_text

%153 = add i64 %6, 80
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%154 = getelementptr inbounds float, float* %1, i64 %153
$i64B

	full_text


i64 %153
LloadBD
B
	full_text5
3
1%155 = load float, float* %154, align 4, !tbaa !8
*float*B

	full_text

float* %154
ecallB]
[
	full_textN
L
J%156 = tail call float @_Z4fminff(float %152, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %152
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
\getelementptrBK
I
	full_text<
:
8%158 = getelementptr inbounds float, float* %2, i64 %153
$i64B

	full_text


i64 %153
LstoreBC
A
	full_text4
2
0store float %157, float* %158, align 4, !tbaa !8
(floatB

	full_text


float %157
*float*B

	full_text

float* %158
KloadBC
A
	full_text4
2
0%159 = load float, float* %14, align 4, !tbaa !8
)float*B

	full_text


float* %14
KloadBC
A
	full_text4
2
0%160 = load float, float* %17, align 4, !tbaa !8
)float*B

	full_text


float* %17
7fmulB/
-
	full_text 

%161 = fmul float %159, %160
(floatB

	full_text


float %159
(floatB

	full_text


float %160
6fmulB.
,
	full_text

%162 = fmul float %12, %161
'floatB

	full_text

	float %12
(floatB

	full_text


float %161
\getelementptrBK
I
	full_text<
:
8%163 = getelementptr inbounds float, float* %3, i64 %103
$i64B

	full_text


i64 %103
LloadBD
B
	full_text5
3
1%164 = load float, float* %163, align 4, !tbaa !8
*float*B

	full_text

float* %163
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

%166 = fmul float %162, %165
(floatB

	full_text


float %162
(floatB

	full_text


float %165
/addB(
&
	full_text

%167 = add i64 %6, 88
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
0%174 = load float, float* %17, align 4, !tbaa !8
)float*B

	full_text


float* %17
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
6fmulB.
,
	full_text

%176 = fmul float %12, %175
'floatB

	full_text

	float %12
(floatB

	full_text


float %175
LloadBD
B
	full_text5
3
1%177 = load float, float* %163, align 4, !tbaa !8
*float*B

	full_text

float* %163
LfdivBD
B
	full_text5
3
1%178 = fdiv float 1.000000e+00, %177, !fpmath !12
(floatB

	full_text


float %177
7fmulB/
-
	full_text 

%179 = fmul float %176, %178
(floatB

	full_text


float %176
(floatB

	full_text


float %178
/addB(
&
	full_text

%180 = add i64 %6, 96
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%181 = getelementptr inbounds float, float* %1, i64 %180
$i64B

	full_text


i64 %180
LloadBD
B
	full_text5
3
1%182 = load float, float* %181, align 4, !tbaa !8
*float*B

	full_text

float* %181
ecallB]
[
	full_textN
L
J%183 = tail call float @_Z4fminff(float %179, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %179
7fmulB/
-
	full_text 

%184 = fmul float %182, %183
(floatB

	full_text


float %182
(floatB

	full_text


float %183
\getelementptrBK
I
	full_text<
:
8%185 = getelementptr inbounds float, float* %2, i64 %180
$i64B

	full_text


i64 %180
LstoreBC
A
	full_text4
2
0store float %184, float* %185, align 4, !tbaa !8
(floatB

	full_text


float %184
*float*B

	full_text

float* %185
KloadBC
A
	full_text4
2
0%186 = load float, float* %14, align 4, !tbaa !8
)float*B

	full_text


float* %14
KloadBC
A
	full_text4
2
0%187 = load float, float* %17, align 4, !tbaa !8
)float*B

	full_text


float* %17
7fmulB/
-
	full_text 

%188 = fmul float %186, %187
(floatB

	full_text


float %186
(floatB

	full_text


float %187
6fmulB.
,
	full_text

%189 = fmul float %12, %188
'floatB

	full_text

	float %12
(floatB

	full_text


float %188
LloadBD
B
	full_text5
3
1%190 = load float, float* %163, align 4, !tbaa !8
*float*B

	full_text

float* %163
LfdivBD
B
	full_text5
3
1%191 = fdiv float 1.000000e+00, %190, !fpmath !12
(floatB

	full_text


float %190
7fmulB/
-
	full_text 

%192 = fmul float %189, %191
(floatB

	full_text


float %189
(floatB

	full_text


float %191
0addB)
'
	full_text

%193 = add i64 %6, 104
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%194 = getelementptr inbounds float, float* %1, i64 %193
$i64B

	full_text


i64 %193
LloadBD
B
	full_text5
3
1%195 = load float, float* %194, align 4, !tbaa !8
*float*B

	full_text

float* %194
ecallB]
[
	full_textN
L
J%196 = tail call float @_Z4fminff(float %192, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %192
7fmulB/
-
	full_text 

%197 = fmul float %195, %196
(floatB

	full_text


float %195
(floatB

	full_text


float %196
\getelementptrBK
I
	full_text<
:
8%198 = getelementptr inbounds float, float* %2, i64 %193
$i64B

	full_text


i64 %193
LstoreBC
A
	full_text4
2
0store float %197, float* %198, align 4, !tbaa !8
(floatB

	full_text


float %197
*float*B

	full_text

float* %198
KloadBC
A
	full_text4
2
0%199 = load float, float* %14, align 4, !tbaa !8
)float*B

	full_text


float* %14
KloadBC
A
	full_text4
2
0%200 = load float, float* %17, align 4, !tbaa !8
)float*B

	full_text


float* %17
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
6fmulB.
,
	full_text

%202 = fmul float %12, %201
'floatB

	full_text

	float %12
(floatB

	full_text


float %201
LloadBD
B
	full_text5
3
1%203 = load float, float* %163, align 4, !tbaa !8
*float*B

	full_text

float* %163
LfdivBD
B
	full_text5
3
1%204 = fdiv float 1.000000e+00, %203, !fpmath !12
(floatB

	full_text


float %203
7fmulB/
-
	full_text 

%205 = fmul float %202, %204
(floatB

	full_text


float %202
(floatB

	full_text


float %204
0addB)
'
	full_text

%206 = add i64 %6, 112
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%207 = getelementptr inbounds float, float* %1, i64 %206
$i64B

	full_text


i64 %206
LloadBD
B
	full_text5
3
1%208 = load float, float* %207, align 4, !tbaa !8
*float*B

	full_text

float* %207
ecallB]
[
	full_textN
L
J%209 = tail call float @_Z4fminff(float %205, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %205
7fmulB/
-
	full_text 

%210 = fmul float %208, %209
(floatB

	full_text


float %208
(floatB

	full_text


float %209
\getelementptrBK
I
	full_text<
:
8%211 = getelementptr inbounds float, float* %2, i64 %206
$i64B

	full_text


i64 %206
LstoreBC
A
	full_text4
2
0store float %210, float* %211, align 4, !tbaa !8
(floatB

	full_text


float %210
*float*B

	full_text

float* %211
KloadBC
A
	full_text4
2
0%212 = load float, float* %24, align 4, !tbaa !8
)float*B

	full_text


float* %24
7fmulB/
-
	full_text 

%213 = fmul float %212, %212
(floatB

	full_text


float %212
(floatB

	full_text


float %212
6fmulB.
,
	full_text

%214 = fmul float %12, %213
'floatB

	full_text

	float %12
(floatB

	full_text


float %213
\getelementptrBK
I
	full_text<
:
8%215 = getelementptr inbounds float, float* %3, i64 %115
$i64B

	full_text


i64 %115
LloadBD
B
	full_text5
3
1%216 = load float, float* %215, align 4, !tbaa !8
*float*B

	full_text

float* %215
LfdivBD
B
	full_text5
3
1%217 = fdiv float 1.000000e+00, %216, !fpmath !12
(floatB

	full_text


float %216
7fmulB/
-
	full_text 

%218 = fmul float %214, %217
(floatB

	full_text


float %214
(floatB

	full_text


float %217
0addB)
'
	full_text

%219 = add i64 %6, 120
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%220 = getelementptr inbounds float, float* %1, i64 %219
$i64B

	full_text


i64 %219
LloadBD
B
	full_text5
3
1%221 = load float, float* %220, align 4, !tbaa !8
*float*B

	full_text

float* %220
ecallB]
[
	full_textN
L
J%222 = tail call float @_Z4fminff(float %218, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %218
7fmulB/
-
	full_text 

%223 = fmul float %221, %222
(floatB

	full_text


float %221
(floatB

	full_text


float %222
\getelementptrBK
I
	full_text<
:
8%224 = getelementptr inbounds float, float* %2, i64 %219
$i64B

	full_text


i64 %219
LstoreBC
A
	full_text4
2
0store float %223, float* %224, align 4, !tbaa !8
(floatB

	full_text


float %223
*float*B

	full_text

float* %224
KloadBC
A
	full_text4
2
0%225 = load float, float* %14, align 4, !tbaa !8
)float*B

	full_text


float* %14
LloadBD
B
	full_text5
3
1%226 = load float, float* %163, align 4, !tbaa !8
*float*B

	full_text

float* %163
7fmulB/
-
	full_text 

%227 = fmul float %225, %226
(floatB

	full_text


float %225
(floatB

	full_text


float %226
KloadBC
A
	full_text4
2
0%228 = load float, float* %21, align 4, !tbaa !8
)float*B

	full_text


float* %21
KloadBC
A
	full_text4
2
0%229 = load float, float* %53, align 4, !tbaa !8
)float*B

	full_text


float* %53
7fmulB/
-
	full_text 

%230 = fmul float %228, %229
(floatB

	full_text


float %228
(floatB

	full_text


float %229
LfdivBD
B
	full_text5
3
1%231 = fdiv float 1.000000e+00, %230, !fpmath !12
(floatB

	full_text


float %230
7fmulB/
-
	full_text 

%232 = fmul float %227, %231
(floatB

	full_text


float %227
(floatB

	full_text


float %231
0addB)
'
	full_text

%233 = add i64 %6, 128
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%234 = getelementptr inbounds float, float* %1, i64 %233
$i64B

	full_text


i64 %233
LloadBD
B
	full_text5
3
1%235 = load float, float* %234, align 4, !tbaa !8
*float*B

	full_text

float* %234
ecallB]
[
	full_textN
L
J%236 = tail call float @_Z4fminff(float %232, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %232
7fmulB/
-
	full_text 

%237 = fmul float %235, %236
(floatB

	full_text


float %235
(floatB

	full_text


float %236
\getelementptrBK
I
	full_text<
:
8%238 = getelementptr inbounds float, float* %2, i64 %233
$i64B

	full_text


i64 %233
LstoreBC
A
	full_text4
2
0store float %237, float* %238, align 4, !tbaa !8
(floatB

	full_text


float %237
*float*B

	full_text

float* %238
KloadBC
A
	full_text4
2
0%239 = load float, float* %14, align 4, !tbaa !8
)float*B

	full_text


float* %14
LloadBD
B
	full_text5
3
1%240 = load float, float* %163, align 4, !tbaa !8
*float*B

	full_text

float* %163
7fmulB/
-
	full_text 

%241 = fmul float %239, %240
(floatB

	full_text


float %239
(floatB

	full_text


float %240
KloadBC
A
	full_text4
2
0%242 = load float, float* %34, align 4, !tbaa !8
)float*B

	full_text


float* %34
KloadBC
A
	full_text4
2
0%243 = load float, float* %17, align 4, !tbaa !8
)float*B

	full_text


float* %17
7fmulB/
-
	full_text 

%244 = fmul float %242, %243
(floatB

	full_text


float %242
(floatB

	full_text


float %243
LfdivBD
B
	full_text5
3
1%245 = fdiv float 1.000000e+00, %244, !fpmath !12
(floatB

	full_text


float %244
7fmulB/
-
	full_text 

%246 = fmul float %241, %245
(floatB

	full_text


float %241
(floatB

	full_text


float %245
0addB)
'
	full_text

%247 = add i64 %6, 136
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%248 = getelementptr inbounds float, float* %1, i64 %247
$i64B

	full_text


i64 %247
LloadBD
B
	full_text5
3
1%249 = load float, float* %248, align 4, !tbaa !8
*float*B

	full_text

float* %248
ecallB]
[
	full_textN
L
J%250 = tail call float @_Z4fminff(float %246, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %246
7fmulB/
-
	full_text 

%251 = fmul float %249, %250
(floatB

	full_text


float %249
(floatB

	full_text


float %250
\getelementptrBK
I
	full_text<
:
8%252 = getelementptr inbounds float, float* %2, i64 %247
$i64B

	full_text


i64 %247
LstoreBC
A
	full_text4
2
0store float %251, float* %252, align 4, !tbaa !8
(floatB

	full_text


float %251
*float*B

	full_text

float* %252
KloadBC
A
	full_text4
2
0%253 = load float, float* %14, align 4, !tbaa !8
)float*B

	full_text


float* %14
LloadBD
B
	full_text5
3
1%254 = load float, float* %163, align 4, !tbaa !8
*float*B

	full_text

float* %163
7fmulB/
-
	full_text 

%255 = fmul float %253, %254
(floatB

	full_text


float %253
(floatB

	full_text


float %254
KloadBC
A
	full_text4
2
0%256 = load float, float* %24, align 4, !tbaa !8
)float*B

	full_text


float* %24
7fmulB/
-
	full_text 

%257 = fmul float %256, %256
(floatB

	full_text


float %256
(floatB

	full_text


float %256
LfdivBD
B
	full_text5
3
1%258 = fdiv float 1.000000e+00, %257, !fpmath !12
(floatB

	full_text


float %257
7fmulB/
-
	full_text 

%259 = fmul float %255, %258
(floatB

	full_text


float %255
(floatB

	full_text


float %258
0addB)
'
	full_text

%260 = add i64 %6, 144
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%261 = getelementptr inbounds float, float* %1, i64 %260
$i64B

	full_text


i64 %260
LloadBD
B
	full_text5
3
1%262 = load float, float* %261, align 4, !tbaa !8
*float*B

	full_text

float* %261
ecallB]
[
	full_textN
L
J%263 = tail call float @_Z4fminff(float %259, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %259
7fmulB/
-
	full_text 

%264 = fmul float %262, %263
(floatB

	full_text


float %262
(floatB

	full_text


float %263
\getelementptrBK
I
	full_text<
:
8%265 = getelementptr inbounds float, float* %2, i64 %260
$i64B

	full_text


i64 %260
LstoreBC
A
	full_text4
2
0store float %264, float* %265, align 4, !tbaa !8
(floatB

	full_text


float %264
*float*B

	full_text

float* %265
KloadBC
A
	full_text4
2
0%266 = load float, float* %21, align 4, !tbaa !8
)float*B

	full_text


float* %21
LloadBD
B
	full_text5
3
1%267 = load float, float* %163, align 4, !tbaa !8
*float*B

	full_text

float* %163
7fmulB/
-
	full_text 

%268 = fmul float %266, %267
(floatB

	full_text


float %266
(floatB

	full_text


float %267
KloadBC
A
	full_text4
2
0%269 = load float, float* %17, align 4, !tbaa !8
)float*B

	full_text


float* %17
KloadBC
A
	full_text4
2
0%270 = load float, float* %24, align 4, !tbaa !8
)float*B

	full_text


float* %24
7fmulB/
-
	full_text 

%271 = fmul float %269, %270
(floatB

	full_text


float %269
(floatB

	full_text


float %270
LfdivBD
B
	full_text5
3
1%272 = fdiv float 1.000000e+00, %271, !fpmath !12
(floatB

	full_text


float %271
7fmulB/
-
	full_text 

%273 = fmul float %268, %272
(floatB

	full_text


float %268
(floatB

	full_text


float %272
0addB)
'
	full_text

%274 = add i64 %6, 152
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%275 = getelementptr inbounds float, float* %1, i64 %274
$i64B

	full_text


i64 %274
LloadBD
B
	full_text5
3
1%276 = load float, float* %275, align 4, !tbaa !8
*float*B

	full_text

float* %275
ecallB]
[
	full_textN
L
J%277 = tail call float @_Z4fminff(float %273, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %273
7fmulB/
-
	full_text 

%278 = fmul float %276, %277
(floatB

	full_text


float %276
(floatB

	full_text


float %277
\getelementptrBK
I
	full_text<
:
8%279 = getelementptr inbounds float, float* %2, i64 %274
$i64B

	full_text


i64 %274
LstoreBC
A
	full_text4
2
0store float %278, float* %279, align 4, !tbaa !8
(floatB

	full_text


float %278
*float*B

	full_text

float* %279
KloadBC
A
	full_text4
2
0%280 = load float, float* %24, align 4, !tbaa !8
)float*B

	full_text


float* %24
LloadBD
B
	full_text5
3
1%281 = load float, float* %163, align 4, !tbaa !8
*float*B

	full_text

float* %163
7fmulB/
-
	full_text 

%282 = fmul float %280, %281
(floatB

	full_text


float %280
(floatB

	full_text


float %281
KloadBC
A
	full_text4
2
0%283 = load float, float* %17, align 4, !tbaa !8
)float*B

	full_text


float* %17
KloadBC
A
	full_text4
2
0%284 = load float, float* %53, align 4, !tbaa !8
)float*B

	full_text


float* %53
7fmulB/
-
	full_text 

%285 = fmul float %283, %284
(floatB

	full_text


float %283
(floatB

	full_text


float %284
LfdivBD
B
	full_text5
3
1%286 = fdiv float 1.000000e+00, %285, !fpmath !12
(floatB

	full_text


float %285
7fmulB/
-
	full_text 

%287 = fmul float %282, %286
(floatB

	full_text


float %282
(floatB

	full_text


float %286
0addB)
'
	full_text

%288 = add i64 %6, 160
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%289 = getelementptr inbounds float, float* %1, i64 %288
$i64B

	full_text


i64 %288
LloadBD
B
	full_text5
3
1%290 = load float, float* %289, align 4, !tbaa !8
*float*B

	full_text

float* %289
ecallB]
[
	full_textN
L
J%291 = tail call float @_Z4fminff(float %287, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %287
7fmulB/
-
	full_text 

%292 = fmul float %290, %291
(floatB

	full_text


float %290
(floatB

	full_text


float %291
\getelementptrBK
I
	full_text<
:
8%293 = getelementptr inbounds float, float* %2, i64 %288
$i64B

	full_text


i64 %288
LstoreBC
A
	full_text4
2
0store float %292, float* %293, align 4, !tbaa !8
(floatB

	full_text


float %292
*float*B

	full_text

float* %293
LloadBD
B
	full_text5
3
1%294 = load float, float* %163, align 4, !tbaa !8
*float*B

	full_text

float* %163
7fmulB/
-
	full_text 

%295 = fmul float %294, %294
(floatB

	full_text


float %294
(floatB

	full_text


float %294
KloadBC
A
	full_text4
2
0%296 = load float, float* %17, align 4, !tbaa !8
)float*B

	full_text


float* %17
LloadBD
B
	full_text5
3
1%297 = load float, float* %215, align 4, !tbaa !8
*float*B

	full_text

float* %215
7fmulB/
-
	full_text 

%298 = fmul float %296, %297
(floatB

	full_text


float %296
(floatB

	full_text


float %297
LfdivBD
B
	full_text5
3
1%299 = fdiv float 1.000000e+00, %298, !fpmath !12
(floatB

	full_text


float %298
7fmulB/
-
	full_text 

%300 = fmul float %295, %299
(floatB

	full_text


float %295
(floatB

	full_text


float %299
0addB)
'
	full_text

%301 = add i64 %6, 168
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%302 = getelementptr inbounds float, float* %1, i64 %301
$i64B

	full_text


i64 %301
LloadBD
B
	full_text5
3
1%303 = load float, float* %302, align 4, !tbaa !8
*float*B

	full_text

float* %302
ecallB]
[
	full_textN
L
J%304 = tail call float @_Z4fminff(float %300, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %300
7fmulB/
-
	full_text 

%305 = fmul float %303, %304
(floatB

	full_text


float %303
(floatB

	full_text


float %304
\getelementptrBK
I
	full_text<
:
8%306 = getelementptr inbounds float, float* %2, i64 %301
$i64B

	full_text


i64 %301
LstoreBC
A
	full_text4
2
0store float %305, float* %306, align 4, !tbaa !8
(floatB

	full_text


float %305
*float*B

	full_text

float* %306
LloadBD
B
	full_text5
3
1%307 = load float, float* %163, align 4, !tbaa !8
*float*B

	full_text

float* %163
7fmulB/
-
	full_text 

%308 = fmul float %307, %307
(floatB

	full_text


float %307
(floatB

	full_text


float %307
KloadBC
A
	full_text4
2
0%309 = load float, float* %17, align 4, !tbaa !8
)float*B

	full_text


float* %17
LloadBD
B
	full_text5
3
1%310 = load float, float* %215, align 4, !tbaa !8
*float*B

	full_text

float* %215
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
LfdivBD
B
	full_text5
3
1%312 = fdiv float 1.000000e+00, %311, !fpmath !12
(floatB

	full_text


float %311
7fmulB/
-
	full_text 

%313 = fmul float %308, %312
(floatB

	full_text


float %308
(floatB

	full_text


float %312
0addB)
'
	full_text

%314 = add i64 %6, 176
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%315 = getelementptr inbounds float, float* %1, i64 %314
$i64B

	full_text


i64 %314
LloadBD
B
	full_text5
3
1%316 = load float, float* %315, align 4, !tbaa !8
*float*B

	full_text

float* %315
ecallB]
[
	full_textN
L
J%317 = tail call float @_Z4fminff(float %313, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %313
7fmulB/
-
	full_text 

%318 = fmul float %316, %317
(floatB

	full_text


float %316
(floatB

	full_text


float %317
\getelementptrBK
I
	full_text<
:
8%319 = getelementptr inbounds float, float* %2, i64 %314
$i64B

	full_text


i64 %314
LstoreBC
A
	full_text4
2
0store float %318, float* %319, align 4, !tbaa !8
(floatB

	full_text


float %318
*float*B

	full_text

float* %319
KloadBC
A
	full_text4
2
0%320 = load float, float* %14, align 4, !tbaa !8
)float*B

	full_text


float* %14
LloadBD
B
	full_text5
3
1%321 = load float, float* %215, align 4, !tbaa !8
*float*B

	full_text

float* %215
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
KloadBC
A
	full_text4
2
0%323 = load float, float* %34, align 4, !tbaa !8
)float*B

	full_text


float* %34
LloadBD
B
	full_text5
3
1%324 = load float, float* %163, align 4, !tbaa !8
*float*B

	full_text

float* %163
7fmulB/
-
	full_text 

%325 = fmul float %323, %324
(floatB

	full_text


float %323
(floatB

	full_text


float %324
LfdivBD
B
	full_text5
3
1%326 = fdiv float 1.000000e+00, %325, !fpmath !12
(floatB

	full_text


float %325
7fmulB/
-
	full_text 

%327 = fmul float %322, %326
(floatB

	full_text


float %322
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
8%329 = getelementptr inbounds float, float* %1, i64 %328
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
ecallB]
[
	full_textN
L
J%331 = tail call float @_Z4fminff(float %327, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %327
7fmulB/
-
	full_text 

%332 = fmul float %330, %331
(floatB

	full_text


float %330
(floatB

	full_text


float %331
\getelementptrBK
I
	full_text<
:
8%333 = getelementptr inbounds float, float* %2, i64 %328
$i64B

	full_text


i64 %328
LstoreBC
A
	full_text4
2
0store float %332, float* %333, align 4, !tbaa !8
(floatB

	full_text


float %332
*float*B

	full_text

float* %333
KloadBC
A
	full_text4
2
0%334 = load float, float* %14, align 4, !tbaa !8
)float*B

	full_text


float* %14
LloadBD
B
	full_text5
3
1%335 = load float, float* %215, align 4, !tbaa !8
*float*B

	full_text

float* %215
7fmulB/
-
	full_text 

%336 = fmul float %334, %335
(floatB

	full_text


float %334
(floatB

	full_text


float %335
KloadBC
A
	full_text4
2
0%337 = load float, float* %24, align 4, !tbaa !8
)float*B

	full_text


float* %24
KloadBC
A
	full_text4
2
0%338 = load float, float* %53, align 4, !tbaa !8
)float*B

	full_text


float* %53
7fmulB/
-
	full_text 

%339 = fmul float %337, %338
(floatB

	full_text


float %337
(floatB

	full_text


float %338
LfdivBD
B
	full_text5
3
1%340 = fdiv float 1.000000e+00, %339, !fpmath !12
(floatB

	full_text


float %339
7fmulB/
-
	full_text 

%341 = fmul float %336, %340
(floatB

	full_text


float %336
(floatB

	full_text


float %340
0addB)
'
	full_text

%342 = add i64 %6, 192
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%343 = getelementptr inbounds float, float* %1, i64 %342
$i64B

	full_text


i64 %342
LloadBD
B
	full_text5
3
1%344 = load float, float* %343, align 4, !tbaa !8
*float*B

	full_text

float* %343
ecallB]
[
	full_textN
L
J%345 = tail call float @_Z4fminff(float %341, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %341
7fmulB/
-
	full_text 

%346 = fmul float %344, %345
(floatB

	full_text


float %344
(floatB

	full_text


float %345
\getelementptrBK
I
	full_text<
:
8%347 = getelementptr inbounds float, float* %2, i64 %342
$i64B

	full_text


i64 %342
LstoreBC
A
	full_text4
2
0store float %346, float* %347, align 4, !tbaa !8
(floatB

	full_text


float %346
*float*B

	full_text

float* %347
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


float %4
*float*8B

	full_text

	float* %2
*float*8B

	full_text

	float* %3
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
$i648B

	full_text


i64 24
%i648B

	full_text
	
i64 128
%i648B

	full_text
	
i64 184
$i648B

	full_text


i64 64
%i648B

	full_text
	
i64 136
#i328B

	full_text	

i32 0
8float8B+
)
	full_text

float 0x4193D2C640000000
%i648B

	full_text
	
i64 120
%i648B

	full_text
	
i64 176
%i648B

	full_text
	
i64 144
$i648B

	full_text


i64 88
%i648B

	full_text
	
i64 192
2float8B%
#
	full_text

float 1.000000e+00
$i648B

	full_text


i64 72
$i648B

	full_text


i64 16
%i648B

	full_text
	
i64 168
%i648B

	full_text
	
i64 104
%i648B

	full_text
	
i64 112
8float8B+
)
	full_text

float 0x4415AF1D80000000
$i648B

	full_text


i64 48
$i648B

	full_text


i64 32
$i648B

	full_text


i64 80
%i648B

	full_text
	
i64 160
2float8B%
#
	full_text

float 1.013250e+06
$i648B

	full_text


i64 40
#i648B

	full_text	

i64 8
%i648B

	full_text
	
i64 152
$i648B

	full_text


i64 96
$i648B

	full_text


i64 56       	  
 

                       !" !! #$ ## %& %% '( '' )* )+ )) ,- ,, ./ .0 .. 12 11 34 33 56 55 78 79 77 :; :: <= <> << ?@ ?? AB AA CD CC EF EG EE HI HH JK JJ LM LN LL OP OO QR QS QQ TU TT VW VV XY XX Z[ Z\ ZZ ]^ ]] _` _a __ bc bb de dd fg fh ff ij ii kl kk mn mm op oo qr qs qq tu tt vw vx vv yz yy {| {{ }~ }} Ä 	Å  Ç
É ÇÇ ÑÖ Ñ
Ü ÑÑ áà áá âä â
ã ââ åç åå éè éé êë ê
í êê ì
î ìì ïñ ï
ó ïï ò
ô òò öõ öö úù úú ûü û
† ûû °
¢ °° £§ £
• ££ ¶ß ¶¶ ®© ®
™ ®® ´¨ ´
≠ ´´ ÆØ ÆÆ ∞
± ∞∞ ≤≥ ≤
¥ ≤≤ µ
∂ µµ ∑∏ ∑∑ π∫ ππ ªº ª
Ω ªª æ
ø ææ ¿¡ ¿
¬ ¿¿ √ƒ √√ ≈∆ ≈
« ≈≈ »… »
  »» ÀÃ ÀÀ Õ
Œ ÕÕ œ– œ
— œœ “
” ““ ‘’ ‘‘ ÷◊ ÷÷ ÿŸ ÿ
⁄ ÿÿ €
‹ €€ ›ﬁ ›
ﬂ ›› ‡· ‡‡ ‚„ ‚
‰ ‚‚ ÂÊ Â
Á ÂÂ ËÈ ËË Í
Î ÍÍ ÏÌ Ï
Ó ÏÏ Ô ÔÔ Ò
Ú ÒÒ ÛÙ ÛÛ ıˆ ıı ˜¯ ˜
˘ ˜˜ ˙
˚ ˙˙ ¸˝ ¸
˛ ¸¸ ˇÄ ˇˇ ÅÇ Å
É ÅÅ ÑÖ Ñ
Ü ÑÑ áà áá â
ä ââ ãå ã
ç ãã éè éé ê
ë êê íì íí îï îî ñó ñ
ò ññ ô
ö ôô õú õ
ù õõ ûü ûû †° †† ¢£ ¢
§ ¢¢ •¶ •
ß •• ®© ®® ™
´ ™™ ¨≠ ¨
Æ ¨¨ Ø∞ ØØ ±
≤ ±± ≥¥ ≥≥ µ∂ µµ ∑∏ ∑
π ∑∑ ∫
ª ∫∫ ºΩ º
æ ºº ø¿ øø ¡¬ ¡¡ √ƒ √
≈ √√ ∆« ∆
» ∆∆ …  …… À
Ã ÀÀ ÕŒ Õ
œ ÕÕ –— –– “
” ““ ‘’ ‘‘ ÷◊ ÷÷ ÿŸ ÿ
⁄ ÿÿ €
‹ €€ ›ﬁ ›
ﬂ ›› ‡· ‡‡ ‚„ ‚
‰ ‚‚ ÂÊ Â
Á ÂÂ ËÈ ËË Í
Î ÍÍ ÏÌ Ï
Ó ÏÏ Ô ÔÔ Ò
Ú ÒÒ ÛÙ ÛÛ ıˆ ıı ˜¯ ˜
˘ ˜˜ ˙
˚ ˙˙ ¸˝ ¸
˛ ¸¸ ˇÄ ˇˇ ÅÇ ÅÅ ÉÑ É
Ö ÉÉ Üá Ü
à ÜÜ â
ä ââ ãå ãã ç
é çç èê è
ë èè íì íí î
ï îî ñó ññ òô òò öõ ö
ú öö ù
û ùù ü† ü
° üü ¢£ ¢¢ §• §§ ¶ß ¶
® ¶¶ ©™ ©
´ ©© ¨≠ ¨¨ Æ
Ø ÆÆ ∞± ∞
≤ ∞∞ ≥¥ ≥≥ µ
∂ µµ ∑∏ ∑∑ π∫ ππ ªº ª
Ω ªª æ
ø ææ ¿¡ ¿
¬ ¿¿ √ƒ √√ ≈∆ ≈≈ «» «
… ««  À  
Ã    ÕŒ ÕÕ œ
– œœ —“ —
” —— ‘’ ‘‘ ÷
◊ ÷÷ ÿŸ ÿÿ ⁄€ ⁄⁄ ‹› ‹
ﬁ ‹‹ ﬂ
‡ ﬂﬂ ·‚ ·
„ ·· ‰Â ‰‰ ÊÁ ÊÊ ËÈ Ë
Í ËË ÎÏ Î
Ì ÎÎ ÓÔ ÓÓ 
Ò  ÚÛ Ú
Ù ÚÚ ıˆ ıı ˜
¯ ˜˜ ˘˙ ˘˘ ˚¸ ˚˚ ˝˛ ˝
ˇ ˝˝ Ä
Å ÄÄ ÇÉ Ç
Ñ ÇÇ ÖÜ ÖÖ áà á
â áá äã ä
å ää ç
é çç èê èè ë
í ëë ìî ì
ï ìì ñó ññ ò
ô òò öõ öö úù úú ûü û
† ûû °
¢ °° £§ £
• ££ ¶ß ¶¶ ®© ®® ™´ ™
¨ ™™ ≠Æ ≠≠ Ø∞ ØØ ±≤ ±
≥ ±± ¥
µ ¥¥ ∂∑ ∂
∏ ∂∂ π∫ ππ ª
º ªª Ωæ ΩΩ ø¿ øø ¡¬ ¡
√ ¡¡ ƒ
≈ ƒƒ ∆« ∆
» ∆∆ …  …… ÀÃ ÀÀ ÕŒ Õ
œ ÕÕ –— –– “” ““ ‘’ ‘
÷ ‘‘ ◊
ÿ ◊◊ Ÿ⁄ Ÿ
€ ŸŸ ‹› ‹‹ ﬁ
ﬂ ﬁﬁ ‡· ‡‡ ‚„ ‚‚ ‰Â ‰
Ê ‰‰ Á
Ë ÁÁ ÈÍ È
Î ÈÈ ÏÌ ÏÏ ÓÔ ÓÓ Ò 
Ú  ÛÙ ÛÛ ıˆ ı
˜ ıı ¯
˘ ¯¯ ˙˚ ˙
¸ ˙˙ ˝˛ ˝˝ ˇ
Ä ˇˇ ÅÇ ÅÅ ÉÑ ÉÉ ÖÜ Ö
á ÖÖ à
â àà äã ä
å ää çé çç èê èè ëí ë
ì ëë îï îî ñó ññ òô ò
ö òò õ
ú õõ ùû ù
ü ùù †° †† ¢
£ ¢¢ §• §§ ¶ß ¶¶ ®© ®
™ ®® ´
¨ ´´ ≠Æ ≠
Ø ≠≠ ∞± ∞∞ ≤≥ ≤≤ ¥µ ¥
∂ ¥¥ ∑∏ ∑∑ π∫ ππ ªº ª
Ω ªª æ
ø ææ ¿¡ ¿
¬ ¿¿ √ƒ √√ ≈
∆ ≈≈ «» «« …  …… ÀÃ À
Õ ÀÀ Œ
œ ŒŒ –— –
“ –– ”‘ ”” ’÷ ’
◊ ’’ ÿŸ ÿÿ ⁄€ ⁄⁄ ‹› ‹
ﬁ ‹‹ ﬂ
‡ ﬂﬂ ·‚ ·
„ ·· ‰Â ‰‰ Ê
Á ÊÊ ËÈ ËË ÍÎ ÍÍ ÏÌ Ï
Ó ÏÏ Ô
 ÔÔ ÒÚ Ò
Û ÒÒ Ùı ÙÙ ˆ˜ ˆ
¯ ˆˆ ˘˙ ˘˘ ˚¸ ˚˚ ˝˛ ˝
ˇ ˝˝ Ä
Å ÄÄ ÇÉ Ç
Ñ ÇÇ ÖÜ ÖÖ á
à áá âä ââ ãå ãã çé ç
è çç ê
ë êê íì í
î íí ïñ ïï óò óó ôö ô
õ ôô úù úú ûü ûû †° †
¢ †† £
§ ££ •¶ •
ß •• ®© ®® ™
´ ™™ ¨≠ ¨¨ ÆØ ÆÆ ∞± ∞
≤ ∞∞ ≥
¥ ≥≥ µ∂ µ
∑ µµ ∏π ∏∏ ∫ª ∫∫ ºΩ º
æ ºº ø¿ øø ¡¬ ¡¡ √ƒ √
≈ √√ ∆
« ∆∆ »… »
  »» ÀÃ ÀÀ Õ
Œ ÕÕ œ– œœ —“ —— ”‘ ”
’ ”” ÷
◊ ÷÷ ÿŸ ÿ
⁄ ÿÿ €‹ 1‹ T‹ y‹ ò‹ µ‹ “‹ Ò‹ ê‹ ±‹ “‹ Ò‹ î‹ µ‹ ÷‹ ˜‹ ò‹ ª‹ ﬁ‹ ˇ‹ ¢‹ ≈‹ Ê‹ á‹ ™‹ Õ	› ﬁ :ﬁ ]ﬁ Çﬁ °ﬁ æﬁ €ﬁ ˙ﬁ ôﬁ ∫ﬁ €ﬁ ˙ﬁ ùﬁ æﬁ ﬂﬁ Äﬁ °ﬁ ƒﬁ Áﬁ àﬁ ´ﬁ Œﬁ Ôﬁ êﬁ ≥ﬁ ÷ﬂ ﬂ ﬂ ﬂ %ﬂ ?ﬂ mﬂ âﬂ ç‡     	 
             " $# &% (! *' +) - /, 0 21 4. 63 85 9 ;7 =: > @? B DA FC G I% KH MJ NL PE RO S UT WQ YV [X \ ^Z `] a? c% eb gd h j lk nm pi ro sq uf wt x zy |v ~{ Ä} Å É ÖÇ Ü% àá äá ã çm èå ëé íê îâ ñì ó ôò õï ùö üú † ¢û §° • ß¶ ©¶ ™ ¨® ≠? ØÆ ±´ ≥∞ ¥# ∂µ ∏≤ ∫∑ ºπ Ω# øª ¡æ ¬ ƒ√ ∆√ « …≈  ? ÃÀ Œ» –Õ —k ”“ ’œ ◊‘ Ÿ÷ ⁄k ‹ÿ ﬁ€ ﬂ ·‡ „‡ ‰ Ê‚ Á? ÈË ÎÂ ÌÍ Ó Ô ÚÒ ÙÏ ˆÛ ¯ı ˘Ô ˚˜ ˝˙ ˛ Äˇ Çˇ É ÖÅ Ü? àá äÑ åâ ç èé ëê ìã ïí óî òé öñ úô ù ü% °û £† § ¶¢ ßm ©® ´• ≠™ Æ ∞Ø ≤± ¥¨ ∂≥ ∏µ πØ ª∑ Ω∫ æ ¿ ¬ø ƒ¡ ≈ «√ »%  … Ã∆ ŒÀ œ —– ”“ ’Õ ◊‘ Ÿ÷ ⁄– ‹ÿ ﬁ€ ﬂ ·‡ „‡ ‰ Ê‚ Á ÈË ÎÂ ÌÍ Ó Ô ÚÒ ÙÏ ˆÛ ¯ı ˘Ô ˚˜ ˝˙ ˛ Ä Çˇ ÑÅ Ö áÉ àÔ äâ åã éÜ êç ë ìí ïî óè ôñ õò úí ûö †ù ° £ •¢ ß§ ® ™¶ ´â ≠¨ Ø© ±Æ ≤ ¥≥ ∂µ ∏∞ ∫∑ ºπ Ω≥ øª ¡æ ¬ ƒ ∆√ »≈ … À« Ãâ ŒÕ –  “œ ” ’‘ ◊÷ Ÿ— €ÿ ›⁄ ﬁ‘ ‡‹ ‚ﬂ „ Â Á‰ ÈÊ Í ÏË Ìâ ÔÓ ÒÎ Û Ù ˆı ¯˜ ˙Ú ¸˘ ˛˚ ˇı Å˝ ÉÄ Ñ% ÜÖ àÖ â ãá åé éç êè íä îë ï óñ ôò õì ùö üú †ñ ¢û §° • ßâ ©¶ ´® ¨ Æm ∞≠ ≤Ø ≥± µ™ ∑¥ ∏ ∫π ºª æ∂ ¿Ω ¬ø √π ≈¡ «ƒ »  â Ã… ŒÀ œ? — ”– ’“ ÷‘ ÿÕ ⁄◊ € ›‹ ﬂﬁ ·Ÿ „‡ Â‚ Ê‹ Ë‰ ÍÁ Î Ìâ ÔÏ ÒÓ Ú% ÙÛ ˆÛ ˜ı ˘ ˚¯ ¸ ˛˝ Äˇ Ç˙ ÑÅ ÜÉ á˝ âÖ ãà å éâ êç íè ì ï% óî ôñ öò úë ûõ ü °† £¢ •ù ß§ ©¶ ™† ¨® Æ´ Ø% ±â ≥∞ µ≤ ∂ ∏m ∫∑ ºπ Ωª ø¥ ¡æ ¬ ƒ√ ∆≈ »¿  « Ã… Õ√ œÀ —Œ “â ‘” ÷” ◊ Ÿç €ÿ ›⁄ ﬁ‹ ‡’ ‚ﬂ „ Â‰ ÁÊ È· ÎË ÌÍ Ó‰ Ï ÚÔ Ûâ ıÙ ˜Ù ¯ ˙ç ¸˘ ˛˚ ˇ˝ Åˆ ÉÄ Ñ ÜÖ àá äÇ åâ éã èÖ ëç ìê î ñç òï öó õ? ùâ üú °û ¢† §ô ¶£ ß ©® ´™ ≠• Ø¨ ±Æ ≤® ¥∞ ∂≥ ∑ πç ª∏ Ω∫ æ% ¿m ¬ø ƒ¡ ≈√ «º …∆   ÃÀ ŒÕ –» “œ ‘— ’À ◊” Ÿ÷ ⁄ € ·· ‚‚X ‚‚ X˚ ‚‚ ˚ø ‚‚ ø… ‚‚ …÷ ‚‚ ÷î ‚‚ îı ‚‚ ıú ‚‚ úµ ‚‚ µú ‚‚ ú ·· É ‚‚ É⁄ ‚‚ ⁄— ‚‚ —5 ‚‚ 5ã ‚‚ ãπ ‚‚ π‚ ‚‚ ‚¶ ‚‚ ¶ı ‚‚ ıÍ ‚‚ Í} ‚‚ }π ‚‚ πÆ ‚‚ Æ÷ ‚‚ ÷ò ‚‚ ò	„ 
‰ π
Â ®
Ê Ø
Á ‹Ë 	È 
Í ñ
Î Ö
Ï ˝
Ì í
Ó ÀÔ 
Ô ,Ô OÔ tÔ ìÔ ∞Ô ÕÔ ÍÔ âÔ ™Ô ÀÔ ÍÔ çÔ ÆÔ œÔ Ô ëÔ ¥Ô ◊Ô ¯Ô õÔ æÔ ﬂÔ ÄÔ £Ô ∆
 –	Ò 
Ú ‰
Û ‘
Ù ı	ı 5	ı X	ı }
ı ú
ı π
ı ÷
ı ı
ı î
ı µ
ı ÷
ı ı
ı ò
ı π
ı ⁄
ı ˚
ı ú
ı ø
ı ‚
ı É
ı ¶
ı …
ı Í
ı ã
ı Æ
ı —
ˆ Ô	˜ #
¯ Ô
˘ √	˙ 	˚ k	¸ 
˝ †
˛ ≥
ˇ é"
ratt2_kernel"
_Z13get_global_idj"
	_Z4fminff*ó
shoc-1.1.5-S3D-ratt2_kernel.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282Å

wgsize_log1p
íÛéA
 
transfer_bytes_log1p
íÛéA

transfer_bytes
à¢ª

wgsize
Ä

devmap_label
 