
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
%13 = add i64 %6, 48
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
%16 = add i64 %6, 128
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
%20 = add i64 %6, 56
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
%23 = add i64 %6, 120
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
/addB(
&
	full_text

%29 = add i64 %6, 600
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
.addB'
%
	full_text

%35 = add i64 %6, 64
"i64B

	full_text


i64 %6
ZgetelementptrBI
G
	full_text:
8
6%36 = getelementptr inbounds float, float* %3, i64 %35
#i64B

	full_text
	
i64 %35
JloadBB
@
	full_text3
1
/%37 = load float, float* %36, align 4, !tbaa !8
)float*B

	full_text


float* %36
JloadBB
@
	full_text3
1
/%38 = load float, float* %17, align 4, !tbaa !8
)float*B

	full_text


float* %17
4fmulB,
*
	full_text

%39 = fmul float %37, %38
'floatB

	full_text

	float %37
'floatB

	full_text

	float %38
-addB&
$
	full_text

%40 = add i64 %6, 8
"i64B

	full_text


i64 %6
ZgetelementptrBI
G
	full_text:
8
6%41 = getelementptr inbounds float, float* %3, i64 %40
#i64B

	full_text
	
i64 %40
JloadBB
@
	full_text3
1
/%42 = load float, float* %41, align 4, !tbaa !8
)float*B

	full_text


float* %41
/addB(
&
	full_text

%43 = add i64 %6, 200
"i64B

	full_text


i64 %6
ZgetelementptrBI
G
	full_text:
8
6%44 = getelementptr inbounds float, float* %3, i64 %43
#i64B

	full_text
	
i64 %43
JloadBB
@
	full_text3
1
/%45 = load float, float* %44, align 4, !tbaa !8
)float*B

	full_text


float* %44
4fmulB,
*
	full_text

%46 = fmul float %42, %45
'floatB

	full_text

	float %42
'floatB

	full_text

	float %45
JfdivBB
@
	full_text3
1
/%47 = fdiv float 1.000000e+00, %46, !fpmath !12
'floatB

	full_text

	float %46
4fmulB,
*
	full_text

%48 = fmul float %39, %47
'floatB

	full_text

	float %39
'floatB

	full_text

	float %47
/addB(
&
	full_text

%49 = add i64 %6, 608
"i64B

	full_text


i64 %6
ZgetelementptrBI
G
	full_text:
8
6%50 = getelementptr inbounds float, float* %1, i64 %49
#i64B

	full_text
	
i64 %49
JloadBB
@
	full_text3
1
/%51 = load float, float* %50, align 4, !tbaa !8
)float*B

	full_text


float* %50
ccallB[
Y
	full_textL
J
H%52 = tail call float @_Z4fminff(float %48, float 0x4415AF1D80000000) #2
'floatB

	full_text

	float %48
4fmulB,
*
	full_text

%53 = fmul float %51, %52
'floatB

	full_text

	float %51
'floatB

	full_text

	float %52
ZgetelementptrBI
G
	full_text:
8
6%54 = getelementptr inbounds float, float* %2, i64 %49
#i64B

	full_text
	
i64 %49
JstoreBA
?
	full_text2
0
.store float %53, float* %54, align 4, !tbaa !8
'floatB

	full_text

	float %53
)float*B

	full_text


float* %54
JloadBB
@
	full_text3
1
/%55 = load float, float* %41, align 4, !tbaa !8
)float*B

	full_text


float* %41
.addB'
%
	full_text

%56 = add i64 %6, 88
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
4fmulB,
*
	full_text

%59 = fmul float %55, %58
'floatB

	full_text

	float %55
'floatB

	full_text

	float %58
4fmulB,
*
	full_text

%60 = fmul float %12, %59
'floatB

	full_text

	float %12
'floatB

	full_text

	float %59
.addB'
%
	full_text

%61 = add i64 %6, 96
"i64B

	full_text


i64 %6
ZgetelementptrBI
G
	full_text:
8
6%62 = getelementptr inbounds float, float* %3, i64 %61
#i64B

	full_text
	
i64 %61
JloadBB
@
	full_text3
1
/%63 = load float, float* %62, align 4, !tbaa !8
)float*B

	full_text


float* %62
JfdivBB
@
	full_text3
1
/%64 = fdiv float 1.000000e+00, %63, !fpmath !12
'floatB

	full_text

	float %63
4fmulB,
*
	full_text

%65 = fmul float %60, %64
'floatB

	full_text

	float %60
'floatB

	full_text

	float %64
/addB(
&
	full_text

%66 = add i64 %6, 616
"i64B

	full_text


i64 %6
ZgetelementptrBI
G
	full_text:
8
6%67 = getelementptr inbounds float, float* %1, i64 %66
#i64B

	full_text
	
i64 %66
JloadBB
@
	full_text3
1
/%68 = load float, float* %67, align 4, !tbaa !8
)float*B

	full_text


float* %67
ccallB[
Y
	full_textL
J
H%69 = tail call float @_Z4fminff(float %65, float 0x4415AF1D80000000) #2
'floatB

	full_text

	float %65
4fmulB,
*
	full_text

%70 = fmul float %68, %69
'floatB

	full_text

	float %68
'floatB

	full_text

	float %69
ZgetelementptrBI
G
	full_text:
8
6%71 = getelementptr inbounds float, float* %2, i64 %66
#i64B

	full_text
	
i64 %66
JstoreBA
?
	full_text2
0
.store float %70, float* %71, align 4, !tbaa !8
'floatB

	full_text

	float %70
)float*B

	full_text


float* %71
.addB'
%
	full_text

%72 = add i64 %6, 16
"i64B

	full_text


i64 %6
ZgetelementptrBI
G
	full_text:
8
6%73 = getelementptr inbounds float, float* %3, i64 %72
#i64B

	full_text
	
i64 %72
JloadBB
@
	full_text3
1
/%74 = load float, float* %73, align 4, !tbaa !8
)float*B

	full_text


float* %73
JloadBB
@
	full_text3
1
/%75 = load float, float* %57, align 4, !tbaa !8
)float*B

	full_text


float* %57
4fmulB,
*
	full_text

%76 = fmul float %74, %75
'floatB

	full_text

	float %74
'floatB

	full_text

	float %75
JloadBB
@
	full_text3
1
/%77 = load float, float* %41, align 4, !tbaa !8
)float*B

	full_text


float* %41
JloadBB
@
	full_text3
1
/%78 = load float, float* %17, align 4, !tbaa !8
)float*B

	full_text


float* %17
4fmulB,
*
	full_text

%79 = fmul float %77, %78
'floatB

	full_text

	float %77
'floatB

	full_text

	float %78
JfdivBB
@
	full_text3
1
/%80 = fdiv float 1.000000e+00, %79, !fpmath !12
'floatB

	full_text

	float %79
4fmulB,
*
	full_text

%81 = fmul float %76, %80
'floatB

	full_text

	float %76
'floatB

	full_text

	float %80
/addB(
&
	full_text

%82 = add i64 %6, 624
"i64B

	full_text


i64 %6
ZgetelementptrBI
G
	full_text:
8
6%83 = getelementptr inbounds float, float* %1, i64 %82
#i64B

	full_text
	
i64 %82
JloadBB
@
	full_text3
1
/%84 = load float, float* %83, align 4, !tbaa !8
)float*B

	full_text


float* %83
ccallB[
Y
	full_textL
J
H%85 = tail call float @_Z4fminff(float %81, float 0x4415AF1D80000000) #2
'floatB

	full_text

	float %81
4fmulB,
*
	full_text

%86 = fmul float %84, %85
'floatB

	full_text

	float %84
'floatB

	full_text

	float %85
ZgetelementptrBI
G
	full_text:
8
6%87 = getelementptr inbounds float, float* %2, i64 %82
#i64B

	full_text
	
i64 %82
JstoreBA
?
	full_text2
0
.store float %86, float* %87, align 4, !tbaa !8
'floatB

	full_text

	float %86
)float*B

	full_text


float* %87
.addB'
%
	full_text

%88 = add i64 %6, 32
"i64B

	full_text


i64 %6
ZgetelementptrBI
G
	full_text:
8
6%89 = getelementptr inbounds float, float* %3, i64 %88
#i64B

	full_text
	
i64 %88
JloadBB
@
	full_text3
1
/%90 = load float, float* %89, align 4, !tbaa !8
)float*B

	full_text


float* %89
JloadBB
@
	full_text3
1
/%91 = load float, float* %57, align 4, !tbaa !8
)float*B

	full_text


float* %57
4fmulB,
*
	full_text

%92 = fmul float %90, %91
'floatB

	full_text

	float %90
'floatB

	full_text

	float %91
.addB'
%
	full_text

%93 = add i64 %6, 40
"i64B

	full_text


i64 %6
ZgetelementptrBI
G
	full_text:
8
6%94 = getelementptr inbounds float, float* %3, i64 %93
#i64B

	full_text
	
i64 %93
JloadBB
@
	full_text3
1
/%95 = load float, float* %94, align 4, !tbaa !8
)float*B

	full_text


float* %94
.addB'
%
	full_text

%96 = add i64 %6, 72
"i64B

	full_text


i64 %6
ZgetelementptrBI
G
	full_text:
8
6%97 = getelementptr inbounds float, float* %3, i64 %96
#i64B

	full_text
	
i64 %96
JloadBB
@
	full_text3
1
/%98 = load float, float* %97, align 4, !tbaa !8
)float*B

	full_text


float* %97
4fmulB,
*
	full_text

%99 = fmul float %95, %98
'floatB

	full_text

	float %95
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
0addB)
'
	full_text

%102 = add i64 %6, 632
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
KloadBC
A
	full_text4
2
0%108 = load float, float* %89, align 4, !tbaa !8
)float*B

	full_text


float* %89
KloadBC
A
	full_text4
2
0%109 = load float, float* %57, align 4, !tbaa !8
)float*B

	full_text


float* %57
7fmulB/
-
	full_text 

%110 = fmul float %108, %109
(floatB

	full_text


float %108
(floatB

	full_text


float %109
KloadBC
A
	full_text4
2
0%111 = load float, float* %94, align 4, !tbaa !8
)float*B

	full_text


float* %94
/addB(
&
	full_text

%112 = add i64 %6, 80
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%113 = getelementptr inbounds float, float* %3, i64 %112
$i64B

	full_text


i64 %112
LloadBD
B
	full_text5
3
1%114 = load float, float* %113, align 4, !tbaa !8
*float*B

	full_text

float* %113
7fmulB/
-
	full_text 

%115 = fmul float %111, %114
(floatB

	full_text


float %111
(floatB

	full_text


float %114
LfdivBD
B
	full_text5
3
1%116 = fdiv float 1.000000e+00, %115, !fpmath !12
(floatB

	full_text


float %115
7fmulB/
-
	full_text 

%117 = fmul float %110, %116
(floatB

	full_text


float %110
(floatB

	full_text


float %116
0addB)
'
	full_text

%118 = add i64 %6, 640
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%119 = getelementptr inbounds float, float* %1, i64 %118
$i64B

	full_text


i64 %118
LloadBD
B
	full_text5
3
1%120 = load float, float* %119, align 4, !tbaa !8
*float*B

	full_text

float* %119
ecallB]
[
	full_textN
L
J%121 = tail call float @_Z4fminff(float %117, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %117
7fmulB/
-
	full_text 

%122 = fmul float %120, %121
(floatB

	full_text


float %120
(floatB

	full_text


float %121
\getelementptrBK
I
	full_text<
:
8%123 = getelementptr inbounds float, float* %2, i64 %118
$i64B

	full_text


i64 %118
LstoreBC
A
	full_text4
2
0store float %122, float* %123, align 4, !tbaa !8
(floatB

	full_text


float %122
*float*B

	full_text

float* %123
/addB(
&
	full_text

%124 = add i64 %6, 24
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%125 = getelementptr inbounds float, float* %3, i64 %124
$i64B

	full_text


i64 %124
LloadBD
B
	full_text5
3
1%126 = load float, float* %125, align 4, !tbaa !8
*float*B

	full_text

float* %125
KloadBC
A
	full_text4
2
0%127 = load float, float* %57, align 4, !tbaa !8
)float*B

	full_text


float* %57
7fmulB/
-
	full_text 

%128 = fmul float %126, %127
(floatB

	full_text


float %126
(floatB

	full_text


float %127
KloadBC
A
	full_text4
2
0%129 = load float, float* %73, align 4, !tbaa !8
)float*B

	full_text


float* %73
0addB)
'
	full_text

%130 = add i64 %6, 136
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%131 = getelementptr inbounds float, float* %3, i64 %130
$i64B

	full_text


i64 %130
LloadBD
B
	full_text5
3
1%132 = load float, float* %131, align 4, !tbaa !8
*float*B

	full_text

float* %131
7fmulB/
-
	full_text 

%133 = fmul float %129, %132
(floatB

	full_text


float %129
(floatB

	full_text


float %132
LfdivBD
B
	full_text5
3
1%134 = fdiv float 1.000000e+00, %133, !fpmath !12
(floatB

	full_text


float %133
7fmulB/
-
	full_text 

%135 = fmul float %128, %134
(floatB

	full_text


float %128
(floatB

	full_text


float %134
0addB)
'
	full_text

%136 = add i64 %6, 648
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%137 = getelementptr inbounds float, float* %1, i64 %136
$i64B

	full_text


i64 %136
LloadBD
B
	full_text5
3
1%138 = load float, float* %137, align 4, !tbaa !8
*float*B

	full_text

float* %137
ecallB]
[
	full_textN
L
J%139 = tail call float @_Z4fminff(float %135, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %135
7fmulB/
-
	full_text 

%140 = fmul float %138, %139
(floatB

	full_text


float %138
(floatB

	full_text


float %139
\getelementptrBK
I
	full_text<
:
8%141 = getelementptr inbounds float, float* %2, i64 %136
$i64B

	full_text


i64 %136
LstoreBC
A
	full_text4
2
0store float %140, float* %141, align 4, !tbaa !8
(floatB

	full_text


float %140
*float*B

	full_text

float* %141
LloadBD
B
	full_text5
3
1%142 = load float, float* %125, align 4, !tbaa !8
*float*B

	full_text

float* %125
KloadBC
A
	full_text4
2
0%143 = load float, float* %57, align 4, !tbaa !8
)float*B

	full_text


float* %57
7fmulB/
-
	full_text 

%144 = fmul float %142, %143
(floatB

	full_text


float %142
(floatB

	full_text


float %143
KloadBC
A
	full_text4
2
0%145 = load float, float* %89, align 4, !tbaa !8
)float*B

	full_text


float* %89
KloadBC
A
	full_text4
2
0%146 = load float, float* %17, align 4, !tbaa !8
)float*B

	full_text


float* %17
7fmulB/
-
	full_text 

%147 = fmul float %145, %146
(floatB

	full_text


float %145
(floatB

	full_text


float %146
LfdivBD
B
	full_text5
3
1%148 = fdiv float 1.000000e+00, %147, !fpmath !12
(floatB

	full_text


float %147
7fmulB/
-
	full_text 

%149 = fmul float %144, %148
(floatB

	full_text


float %144
(floatB

	full_text


float %148
0addB)
'
	full_text

%150 = add i64 %6, 656
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%151 = getelementptr inbounds float, float* %1, i64 %150
$i64B

	full_text


i64 %150
LloadBD
B
	full_text5
3
1%152 = load float, float* %151, align 4, !tbaa !8
*float*B

	full_text

float* %151
ecallB]
[
	full_textN
L
J%153 = tail call float @_Z4fminff(float %149, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %149
7fmulB/
-
	full_text 

%154 = fmul float %152, %153
(floatB

	full_text


float %152
(floatB

	full_text


float %153
\getelementptrBK
I
	full_text<
:
8%155 = getelementptr inbounds float, float* %2, i64 %150
$i64B

	full_text


i64 %150
LstoreBC
A
	full_text4
2
0store float %154, float* %155, align 4, !tbaa !8
(floatB

	full_text


float %154
*float*B

	full_text

float* %155
KloadBC
A
	full_text4
2
0%156 = load float, float* %14, align 4, !tbaa !8
)float*B

	full_text


float* %14
KloadBC
A
	full_text4
2
0%157 = load float, float* %57, align 4, !tbaa !8
)float*B

	full_text


float* %57
7fmulB/
-
	full_text 

%158 = fmul float %156, %157
(floatB

	full_text


float %156
(floatB

	full_text


float %157
LloadBD
B
	full_text5
3
1%159 = load float, float* %125, align 4, !tbaa !8
*float*B

	full_text

float* %125
KloadBC
A
	full_text4
2
0%160 = load float, float* %62, align 4, !tbaa !8
)float*B

	full_text


float* %62
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
LfdivBD
B
	full_text5
3
1%162 = fdiv float 1.000000e+00, %161, !fpmath !12
(floatB

	full_text


float %161
7fmulB/
-
	full_text 

%163 = fmul float %158, %162
(floatB

	full_text


float %158
(floatB

	full_text


float %162
0addB)
'
	full_text

%164 = add i64 %6, 664
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%165 = getelementptr inbounds float, float* %1, i64 %164
$i64B

	full_text


i64 %164
LloadBD
B
	full_text5
3
1%166 = load float, float* %165, align 4, !tbaa !8
*float*B

	full_text

float* %165
ecallB]
[
	full_textN
L
J%167 = tail call float @_Z4fminff(float %163, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %163
7fmulB/
-
	full_text 

%168 = fmul float %166, %167
(floatB

	full_text


float %166
(floatB

	full_text


float %167
\getelementptrBK
I
	full_text<
:
8%169 = getelementptr inbounds float, float* %2, i64 %164
$i64B

	full_text


i64 %164
LstoreBC
A
	full_text4
2
0store float %168, float* %169, align 4, !tbaa !8
(floatB

	full_text


float %168
*float*B

	full_text

float* %169
KloadBC
A
	full_text4
2
0%170 = load float, float* %14, align 4, !tbaa !8
)float*B

	full_text


float* %14
KloadBC
A
	full_text4
2
0%171 = load float, float* %57, align 4, !tbaa !8
)float*B

	full_text


float* %57
7fmulB/
-
	full_text 

%172 = fmul float %170, %171
(floatB

	full_text


float %170
(floatB

	full_text


float %171
KloadBC
A
	full_text4
2
0%173 = load float, float* %89, align 4, !tbaa !8
)float*B

	full_text


float* %89
LloadBD
B
	full_text5
3
1%174 = load float, float* %131, align 4, !tbaa !8
*float*B

	full_text

float* %131
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
LfdivBD
B
	full_text5
3
1%176 = fdiv float 1.000000e+00, %175, !fpmath !12
(floatB

	full_text


float %175
7fmulB/
-
	full_text 

%177 = fmul float %172, %176
(floatB

	full_text


float %172
(floatB

	full_text


float %176
0addB)
'
	full_text

%178 = add i64 %6, 672
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%179 = getelementptr inbounds float, float* %1, i64 %178
$i64B

	full_text


i64 %178
LloadBD
B
	full_text5
3
1%180 = load float, float* %179, align 4, !tbaa !8
*float*B

	full_text

float* %179
ecallB]
[
	full_textN
L
J%181 = tail call float @_Z4fminff(float %177, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %177
7fmulB/
-
	full_text 

%182 = fmul float %180, %181
(floatB

	full_text


float %180
(floatB

	full_text


float %181
\getelementptrBK
I
	full_text<
:
8%183 = getelementptr inbounds float, float* %2, i64 %178
$i64B

	full_text


i64 %178
LstoreBC
A
	full_text4
2
0store float %182, float* %183, align 4, !tbaa !8
(floatB

	full_text


float %182
*float*B

	full_text

float* %183
KloadBC
A
	full_text4
2
0%184 = load float, float* %21, align 4, !tbaa !8
)float*B

	full_text


float* %21
KloadBC
A
	full_text4
2
0%185 = load float, float* %57, align 4, !tbaa !8
)float*B

	full_text


float* %57
7fmulB/
-
	full_text 

%186 = fmul float %184, %185
(floatB

	full_text


float %184
(floatB

	full_text


float %185
KloadBC
A
	full_text4
2
0%187 = load float, float* %14, align 4, !tbaa !8
)float*B

	full_text


float* %14
KloadBC
A
	full_text4
2
0%188 = load float, float* %62, align 4, !tbaa !8
)float*B

	full_text


float* %62
7fmulB/
-
	full_text 

%189 = fmul float %187, %188
(floatB

	full_text


float %187
(floatB

	full_text


float %188
LfdivBD
B
	full_text5
3
1%190 = fdiv float 1.000000e+00, %189, !fpmath !12
(floatB

	full_text


float %189
7fmulB/
-
	full_text 

%191 = fmul float %186, %190
(floatB

	full_text


float %186
(floatB

	full_text


float %190
0addB)
'
	full_text

%192 = add i64 %6, 680
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%193 = getelementptr inbounds float, float* %1, i64 %192
$i64B

	full_text


i64 %192
LloadBD
B
	full_text5
3
1%194 = load float, float* %193, align 4, !tbaa !8
*float*B

	full_text

float* %193
ecallB]
[
	full_textN
L
J%195 = tail call float @_Z4fminff(float %191, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %191
7fmulB/
-
	full_text 

%196 = fmul float %194, %195
(floatB

	full_text


float %194
(floatB

	full_text


float %195
\getelementptrBK
I
	full_text<
:
8%197 = getelementptr inbounds float, float* %2, i64 %192
$i64B

	full_text


i64 %192
LstoreBC
A
	full_text4
2
0store float %196, float* %197, align 4, !tbaa !8
(floatB

	full_text


float %196
*float*B

	full_text

float* %197
KloadBC
A
	full_text4
2
0%198 = load float, float* %36, align 4, !tbaa !8
)float*B

	full_text


float* %36
KloadBC
A
	full_text4
2
0%199 = load float, float* %57, align 4, !tbaa !8
)float*B

	full_text


float* %57
7fmulB/
-
	full_text 

%200 = fmul float %198, %199
(floatB

	full_text


float %198
(floatB

	full_text


float %199
KloadBC
A
	full_text4
2
0%201 = load float, float* %41, align 4, !tbaa !8
)float*B

	full_text


float* %41
0addB)
'
	full_text

%202 = add i64 %6, 160
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%203 = getelementptr inbounds float, float* %3, i64 %202
$i64B

	full_text


i64 %202
LloadBD
B
	full_text5
3
1%204 = load float, float* %203, align 4, !tbaa !8
*float*B

	full_text

float* %203
7fmulB/
-
	full_text 

%205 = fmul float %201, %204
(floatB

	full_text


float %201
(floatB

	full_text


float %204
LfdivBD
B
	full_text5
3
1%206 = fdiv float 1.000000e+00, %205, !fpmath !12
(floatB

	full_text


float %205
7fmulB/
-
	full_text 

%207 = fmul float %200, %206
(floatB

	full_text


float %200
(floatB

	full_text


float %206
0addB)
'
	full_text

%208 = add i64 %6, 688
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%209 = getelementptr inbounds float, float* %1, i64 %208
$i64B

	full_text


i64 %208
LloadBD
B
	full_text5
3
1%210 = load float, float* %209, align 4, !tbaa !8
*float*B

	full_text

float* %209
ecallB]
[
	full_textN
L
J%211 = tail call float @_Z4fminff(float %207, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %207
7fmulB/
-
	full_text 

%212 = fmul float %210, %211
(floatB

	full_text


float %210
(floatB

	full_text


float %211
\getelementptrBK
I
	full_text<
:
8%213 = getelementptr inbounds float, float* %2, i64 %208
$i64B

	full_text


i64 %208
LstoreBC
A
	full_text4
2
0store float %212, float* %213, align 4, !tbaa !8
(floatB

	full_text


float %212
*float*B

	full_text

float* %213
KloadBC
A
	full_text4
2
0%214 = load float, float* %57, align 4, !tbaa !8
)float*B

	full_text


float* %57
KloadBC
A
	full_text4
2
0%215 = load float, float* %24, align 4, !tbaa !8
)float*B

	full_text


float* %24
7fmulB/
-
	full_text 

%216 = fmul float %214, %215
(floatB

	full_text


float %214
(floatB

	full_text


float %215
KloadBC
A
	full_text4
2
0%217 = load float, float* %62, align 4, !tbaa !8
)float*B

	full_text


float* %62
0addB)
'
	full_text

%218 = add i64 %6, 104
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%219 = getelementptr inbounds float, float* %3, i64 %218
$i64B

	full_text


i64 %218
LloadBD
B
	full_text5
3
1%220 = load float, float* %219, align 4, !tbaa !8
*float*B

	full_text

float* %219
7fmulB/
-
	full_text 

%221 = fmul float %217, %220
(floatB

	full_text


float %217
(floatB

	full_text


float %220
LfdivBD
B
	full_text5
3
1%222 = fdiv float 1.000000e+00, %221, !fpmath !12
(floatB

	full_text


float %221
7fmulB/
-
	full_text 

%223 = fmul float %216, %222
(floatB

	full_text


float %216
(floatB

	full_text


float %222
0addB)
'
	full_text

%224 = add i64 %6, 696
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%225 = getelementptr inbounds float, float* %1, i64 %224
$i64B

	full_text


i64 %224
LloadBD
B
	full_text5
3
1%226 = load float, float* %225, align 4, !tbaa !8
*float*B

	full_text

float* %225
ecallB]
[
	full_textN
L
J%227 = tail call float @_Z4fminff(float %223, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %223
7fmulB/
-
	full_text 

%228 = fmul float %226, %227
(floatB

	full_text


float %226
(floatB

	full_text


float %227
\getelementptrBK
I
	full_text<
:
8%229 = getelementptr inbounds float, float* %2, i64 %224
$i64B

	full_text


i64 %224
LstoreBC
A
	full_text4
2
0store float %228, float* %229, align 4, !tbaa !8
(floatB

	full_text


float %228
*float*B

	full_text

float* %229
KloadBC
A
	full_text4
2
0%230 = load float, float* %57, align 4, !tbaa !8
)float*B

	full_text


float* %57
KloadBC
A
	full_text4
2
0%231 = load float, float* %24, align 4, !tbaa !8
)float*B

	full_text


float* %24
7fmulB/
-
	full_text 

%232 = fmul float %230, %231
(floatB

	full_text


float %230
(floatB

	full_text


float %231
6fmulB.
,
	full_text

%233 = fmul float %12, %232
'floatB

	full_text

	float %12
(floatB

	full_text


float %232
0addB)
'
	full_text

%234 = add i64 %6, 216
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%235 = getelementptr inbounds float, float* %3, i64 %234
$i64B

	full_text


i64 %234
LloadBD
B
	full_text5
3
1%236 = load float, float* %235, align 4, !tbaa !8
*float*B

	full_text

float* %235
LfdivBD
B
	full_text5
3
1%237 = fdiv float 1.000000e+00, %236, !fpmath !12
(floatB

	full_text


float %236
7fmulB/
-
	full_text 

%238 = fmul float %233, %237
(floatB

	full_text


float %233
(floatB

	full_text


float %237
0addB)
'
	full_text

%239 = add i64 %6, 704
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%240 = getelementptr inbounds float, float* %1, i64 %239
$i64B

	full_text


i64 %239
LloadBD
B
	full_text5
3
1%241 = load float, float* %240, align 4, !tbaa !8
*float*B

	full_text

float* %240
ecallB]
[
	full_textN
L
J%242 = tail call float @_Z4fminff(float %238, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %238
7fmulB/
-
	full_text 

%243 = fmul float %241, %242
(floatB

	full_text


float %241
(floatB

	full_text


float %242
\getelementptrBK
I
	full_text<
:
8%244 = getelementptr inbounds float, float* %2, i64 %239
$i64B

	full_text


i64 %239
LstoreBC
A
	full_text4
2
0store float %243, float* %244, align 4, !tbaa !8
(floatB

	full_text


float %243
*float*B

	full_text

float* %244
KloadBC
A
	full_text4
2
0%245 = load float, float* %57, align 4, !tbaa !8
)float*B

	full_text


float* %57
KloadBC
A
	full_text4
2
0%246 = load float, float* %17, align 4, !tbaa !8
)float*B

	full_text


float* %17
7fmulB/
-
	full_text 

%247 = fmul float %245, %246
(floatB

	full_text


float %245
(floatB

	full_text


float %246
KloadBC
A
	full_text4
2
0%248 = load float, float* %62, align 4, !tbaa !8
)float*B

	full_text


float* %62
KloadBC
A
	full_text4
2
0%249 = load float, float* %24, align 4, !tbaa !8
)float*B

	full_text


float* %24
7fmulB/
-
	full_text 

%250 = fmul float %248, %249
(floatB

	full_text


float %248
(floatB

	full_text


float %249
LfdivBD
B
	full_text5
3
1%251 = fdiv float 1.000000e+00, %250, !fpmath !12
(floatB

	full_text


float %250
7fmulB/
-
	full_text 

%252 = fmul float %247, %251
(floatB

	full_text


float %247
(floatB

	full_text


float %251
0addB)
'
	full_text

%253 = add i64 %6, 712
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%254 = getelementptr inbounds float, float* %1, i64 %253
$i64B

	full_text


i64 %253
LloadBD
B
	full_text5
3
1%255 = load float, float* %254, align 4, !tbaa !8
*float*B

	full_text

float* %254
ecallB]
[
	full_textN
L
J%256 = tail call float @_Z4fminff(float %252, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %252
7fmulB/
-
	full_text 

%257 = fmul float %255, %256
(floatB

	full_text


float %255
(floatB

	full_text


float %256
\getelementptrBK
I
	full_text<
:
8%258 = getelementptr inbounds float, float* %2, i64 %253
$i64B

	full_text


i64 %253
LstoreBC
A
	full_text4
2
0store float %257, float* %258, align 4, !tbaa !8
(floatB

	full_text


float %257
*float*B

	full_text

float* %258
KloadBC
A
	full_text4
2
0%259 = load float, float* %97, align 4, !tbaa !8
)float*B

	full_text


float* %97
KloadBC
A
	full_text4
2
0%260 = load float, float* %57, align 4, !tbaa !8
)float*B

	full_text


float* %57
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
KloadBC
A
	full_text4
2
0%262 = load float, float* %41, align 4, !tbaa !8
)float*B

	full_text


float* %41
0addB)
'
	full_text

%263 = add i64 %6, 168
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
7fmulB/
-
	full_text 

%266 = fmul float %262, %265
(floatB

	full_text


float %262
(floatB

	full_text


float %265
LfdivBD
B
	full_text5
3
1%267 = fdiv float 1.000000e+00, %266, !fpmath !12
(floatB

	full_text


float %266
7fmulB/
-
	full_text 

%268 = fmul float %261, %267
(floatB

	full_text


float %261
(floatB

	full_text


float %267
0addB)
'
	full_text

%269 = add i64 %6, 720
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%270 = getelementptr inbounds float, float* %1, i64 %269
$i64B

	full_text


i64 %269
LloadBD
B
	full_text5
3
1%271 = load float, float* %270, align 4, !tbaa !8
*float*B

	full_text

float* %270
ecallB]
[
	full_textN
L
J%272 = tail call float @_Z4fminff(float %268, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %268
7fmulB/
-
	full_text 

%273 = fmul float %271, %272
(floatB

	full_text


float %271
(floatB

	full_text


float %272
\getelementptrBK
I
	full_text<
:
8%274 = getelementptr inbounds float, float* %2, i64 %269
$i64B

	full_text


i64 %269
LstoreBC
A
	full_text4
2
0store float %273, float* %274, align 4, !tbaa !8
(floatB

	full_text


float %273
*float*B

	full_text

float* %274
LloadBD
B
	full_text5
3
1%275 = load float, float* %113, align 4, !tbaa !8
*float*B

	full_text

float* %113
KloadBC
A
	full_text4
2
0%276 = load float, float* %57, align 4, !tbaa !8
)float*B

	full_text


float* %57
7fmulB/
-
	full_text 

%277 = fmul float %275, %276
(floatB

	full_text


float %275
(floatB

	full_text


float %276
KloadBC
A
	full_text4
2
0%278 = load float, float* %41, align 4, !tbaa !8
)float*B

	full_text


float* %41
LloadBD
B
	full_text5
3
1%279 = load float, float* %264, align 4, !tbaa !8
*float*B

	full_text

float* %264
7fmulB/
-
	full_text 

%280 = fmul float %278, %279
(floatB

	full_text


float %278
(floatB

	full_text


float %279
LfdivBD
B
	full_text5
3
1%281 = fdiv float 1.000000e+00, %280, !fpmath !12
(floatB

	full_text


float %280
7fmulB/
-
	full_text 

%282 = fmul float %277, %281
(floatB

	full_text


float %277
(floatB

	full_text


float %281
0addB)
'
	full_text

%283 = add i64 %6, 728
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%284 = getelementptr inbounds float, float* %1, i64 %283
$i64B

	full_text


i64 %283
LloadBD
B
	full_text5
3
1%285 = load float, float* %284, align 4, !tbaa !8
*float*B

	full_text

float* %284
ecallB]
[
	full_textN
L
J%286 = tail call float @_Z4fminff(float %282, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %282
7fmulB/
-
	full_text 

%287 = fmul float %285, %286
(floatB

	full_text


float %285
(floatB

	full_text


float %286
\getelementptrBK
I
	full_text<
:
8%288 = getelementptr inbounds float, float* %2, i64 %283
$i64B

	full_text


i64 %283
LstoreBC
A
	full_text4
2
0store float %287, float* %288, align 4, !tbaa !8
(floatB

	full_text


float %287
*float*B

	full_text

float* %288
KloadBC
A
	full_text4
2
0%289 = load float, float* %57, align 4, !tbaa !8
)float*B

	full_text


float* %57
7fmulB/
-
	full_text 

%290 = fmul float %289, %289
(floatB

	full_text


float %289
(floatB

	full_text


float %289
6fmulB.
,
	full_text

%291 = fmul float %12, %290
'floatB

	full_text

	float %12
(floatB

	full_text


float %290
0addB)
'
	full_text

%292 = add i64 %6, 184
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%293 = getelementptr inbounds float, float* %3, i64 %292
$i64B

	full_text


i64 %292
LloadBD
B
	full_text5
3
1%294 = load float, float* %293, align 4, !tbaa !8
*float*B

	full_text

float* %293
LfdivBD
B
	full_text5
3
1%295 = fdiv float 1.000000e+00, %294, !fpmath !12
(floatB

	full_text


float %294
7fmulB/
-
	full_text 

%296 = fmul float %291, %295
(floatB

	full_text


float %291
(floatB

	full_text


float %295
0addB)
'
	full_text

%297 = add i64 %6, 736
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%298 = getelementptr inbounds float, float* %1, i64 %297
$i64B

	full_text


i64 %297
LloadBD
B
	full_text5
3
1%299 = load float, float* %298, align 4, !tbaa !8
*float*B

	full_text

float* %298
ecallB]
[
	full_textN
L
J%300 = tail call float @_Z4fminff(float %296, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %296
7fmulB/
-
	full_text 

%301 = fmul float %299, %300
(floatB

	full_text


float %299
(floatB

	full_text


float %300
\getelementptrBK
I
	full_text<
:
8%302 = getelementptr inbounds float, float* %2, i64 %297
$i64B

	full_text


i64 %297
LstoreBC
A
	full_text4
2
0store float %301, float* %302, align 4, !tbaa !8
(floatB

	full_text


float %301
*float*B

	full_text

float* %302
KloadBC
A
	full_text4
2
0%303 = load float, float* %57, align 4, !tbaa !8
)float*B

	full_text


float* %57
7fmulB/
-
	full_text 

%304 = fmul float %303, %303
(floatB

	full_text


float %303
(floatB

	full_text


float %303
KloadBC
A
	full_text4
2
0%305 = load float, float* %41, align 4, !tbaa !8
)float*B

	full_text


float* %41
0addB)
'
	full_text

%306 = add i64 %6, 176
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%307 = getelementptr inbounds float, float* %3, i64 %306
$i64B

	full_text


i64 %306
LloadBD
B
	full_text5
3
1%308 = load float, float* %307, align 4, !tbaa !8
*float*B

	full_text

float* %307
7fmulB/
-
	full_text 

%309 = fmul float %305, %308
(floatB

	full_text


float %305
(floatB

	full_text


float %308
LfdivBD
B
	full_text5
3
1%310 = fdiv float 1.000000e+00, %309, !fpmath !12
(floatB

	full_text


float %309
7fmulB/
-
	full_text 

%311 = fmul float %304, %310
(floatB

	full_text


float %304
(floatB

	full_text


float %310
0addB)
'
	full_text

%312 = add i64 %6, 744
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%313 = getelementptr inbounds float, float* %1, i64 %312
$i64B

	full_text


i64 %312
LloadBD
B
	full_text5
3
1%314 = load float, float* %313, align 4, !tbaa !8
*float*B

	full_text

float* %313
ecallB]
[
	full_textN
L
J%315 = tail call float @_Z4fminff(float %311, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %311
7fmulB/
-
	full_text 

%316 = fmul float %314, %315
(floatB

	full_text


float %314
(floatB

	full_text


float %315
\getelementptrBK
I
	full_text<
:
8%317 = getelementptr inbounds float, float* %2, i64 %312
$i64B

	full_text


i64 %312
LstoreBC
A
	full_text4
2
0store float %316, float* %317, align 4, !tbaa !8
(floatB

	full_text


float %316
*float*B

	full_text

float* %317
KloadBC
A
	full_text4
2
0%318 = load float, float* %57, align 4, !tbaa !8
)float*B

	full_text


float* %57
0addB)
'
	full_text

%319 = add i64 %6, 192
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%320 = getelementptr inbounds float, float* %3, i64 %319
$i64B

	full_text


i64 %319
LloadBD
B
	full_text5
3
1%321 = load float, float* %320, align 4, !tbaa !8
*float*B

	full_text

float* %320
7fmulB/
-
	full_text 

%322 = fmul float %318, %321
(floatB

	full_text


float %318
(floatB

	full_text


float %321
LloadBD
B
	full_text5
3
1%323 = load float, float* %219, align 4, !tbaa !8
*float*B

	full_text

float* %219
LloadBD
B
	full_text5
3
1%324 = load float, float* %264, align 4, !tbaa !8
*float*B

	full_text

float* %264
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
%328 = add i64 %6, 752
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
0%334 = load float, float* %41, align 4, !tbaa !8
)float*B

	full_text


float* %41
LloadBD
B
	full_text5
3
1%335 = load float, float* %131, align 4, !tbaa !8
*float*B

	full_text

float* %131
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
ZgetelementptrBI
G
	full_text:
8
6%337 = getelementptr inbounds float, float* %3, i64 %6
"i64B

	full_text


i64 %6
LloadBD
B
	full_text5
3
1%338 = load float, float* %337, align 4, !tbaa !8
*float*B

	full_text

float* %337
KloadBC
A
	full_text4
2
0%339 = load float, float* %17, align 4, !tbaa !8
)float*B

	full_text


float* %17
7fmulB/
-
	full_text 

%340 = fmul float %338, %339
(floatB

	full_text


float %338
(floatB

	full_text


float %339
LfdivBD
B
	full_text5
3
1%341 = fdiv float 1.000000e+00, %340, !fpmath !12
(floatB

	full_text


float %340
7fmulB/
-
	full_text 

%342 = fmul float %336, %341
(floatB

	full_text


float %336
(floatB

	full_text


float %341
0addB)
'
	full_text

%343 = add i64 %6, 760
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%344 = getelementptr inbounds float, float* %1, i64 %343
$i64B

	full_text


i64 %343
LloadBD
B
	full_text5
3
1%345 = load float, float* %344, align 4, !tbaa !8
*float*B

	full_text

float* %344
ecallB]
[
	full_textN
L
J%346 = tail call float @_Z4fminff(float %342, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %342
7fmulB/
-
	full_text 

%347 = fmul float %345, %346
(floatB

	full_text


float %345
(floatB

	full_text


float %346
\getelementptrBK
I
	full_text<
:
8%348 = getelementptr inbounds float, float* %2, i64 %343
$i64B

	full_text


i64 %343
LstoreBC
A
	full_text4
2
0store float %347, float* %348, align 4, !tbaa !8
(floatB

	full_text


float %347
*float*B

	full_text

float* %348
KloadBC
A
	full_text4
2
0%349 = load float, float* %41, align 4, !tbaa !8
)float*B

	full_text


float* %41
LloadBD
B
	full_text5
3
1%350 = load float, float* %131, align 4, !tbaa !8
*float*B

	full_text

float* %131
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
KloadBC
A
	full_text4
2
0%352 = load float, float* %89, align 4, !tbaa !8
)float*B

	full_text


float* %89
KloadBC
A
	full_text4
2
0%353 = load float, float* %57, align 4, !tbaa !8
)float*B

	full_text


float* %57
7fmulB/
-
	full_text 

%354 = fmul float %352, %353
(floatB

	full_text


float %352
(floatB

	full_text


float %353
LfdivBD
B
	full_text5
3
1%355 = fdiv float 1.000000e+00, %354, !fpmath !12
(floatB

	full_text


float %354
7fmulB/
-
	full_text 

%356 = fmul float %351, %355
(floatB

	full_text


float %351
(floatB

	full_text


float %355
0addB)
'
	full_text

%357 = add i64 %6, 768
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%358 = getelementptr inbounds float, float* %1, i64 %357
$i64B

	full_text


i64 %357
LloadBD
B
	full_text5
3
1%359 = load float, float* %358, align 4, !tbaa !8
*float*B

	full_text

float* %358
ecallB]
[
	full_textN
L
J%360 = tail call float @_Z4fminff(float %356, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %356
7fmulB/
-
	full_text 

%361 = fmul float %359, %360
(floatB

	full_text


float %359
(floatB

	full_text


float %360
\getelementptrBK
I
	full_text<
:
8%362 = getelementptr inbounds float, float* %2, i64 %357
$i64B

	full_text


i64 %357
LstoreBC
A
	full_text4
2
0store float %361, float* %362, align 4, !tbaa !8
(floatB

	full_text


float %361
*float*B

	full_text

float* %362
KloadBC
A
	full_text4
2
0%363 = load float, float* %41, align 4, !tbaa !8
)float*B

	full_text


float* %41
LloadBD
B
	full_text5
3
1%364 = load float, float* %131, align 4, !tbaa !8
*float*B

	full_text

float* %131
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
KloadBC
A
	full_text4
2
0%366 = load float, float* %94, align 4, !tbaa !8
)float*B

	full_text


float* %94
LloadBD
B
	full_text5
3
1%367 = load float, float* %113, align 4, !tbaa !8
*float*B

	full_text

float* %113
7fmulB/
-
	full_text 

%368 = fmul float %366, %367
(floatB

	full_text


float %366
(floatB

	full_text


float %367
LfdivBD
B
	full_text5
3
1%369 = fdiv float 1.000000e+00, %368, !fpmath !12
(floatB

	full_text


float %368
7fmulB/
-
	full_text 

%370 = fmul float %365, %369
(floatB

	full_text


float %365
(floatB

	full_text


float %369
0addB)
'
	full_text

%371 = add i64 %6, 776
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%372 = getelementptr inbounds float, float* %1, i64 %371
$i64B

	full_text


i64 %371
LloadBD
B
	full_text5
3
1%373 = load float, float* %372, align 4, !tbaa !8
*float*B

	full_text

float* %372
ecallB]
[
	full_textN
L
J%374 = tail call float @_Z4fminff(float %370, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %370
7fmulB/
-
	full_text 

%375 = fmul float %373, %374
(floatB

	full_text


float %373
(floatB

	full_text


float %374
\getelementptrBK
I
	full_text<
:
8%376 = getelementptr inbounds float, float* %2, i64 %371
$i64B

	full_text


i64 %371
LstoreBC
A
	full_text4
2
0store float %375, float* %376, align 4, !tbaa !8
(floatB

	full_text


float %375
*float*B

	full_text

float* %376
KloadBC
A
	full_text4
2
0%377 = load float, float* %73, align 4, !tbaa !8
)float*B

	full_text


float* %73
LloadBD
B
	full_text5
3
1%378 = load float, float* %131, align 4, !tbaa !8
*float*B

	full_text

float* %131
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
KloadBC
A
	full_text4
2
0%380 = load float, float* %89, align 4, !tbaa !8
)float*B

	full_text


float* %89
KloadBC
A
	full_text4
2
0%381 = load float, float* %17, align 4, !tbaa !8
)float*B

	full_text


float* %17
7fmulB/
-
	full_text 

%382 = fmul float %380, %381
(floatB

	full_text


float %380
(floatB

	full_text


float %381
LfdivBD
B
	full_text5
3
1%383 = fdiv float 1.000000e+00, %382, !fpmath !12
(floatB

	full_text


float %382
7fmulB/
-
	full_text 

%384 = fmul float %379, %383
(floatB

	full_text


float %379
(floatB

	full_text


float %383
0addB)
'
	full_text

%385 = add i64 %6, 784
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%386 = getelementptr inbounds float, float* %1, i64 %385
$i64B

	full_text


i64 %385
LloadBD
B
	full_text5
3
1%387 = load float, float* %386, align 4, !tbaa !8
*float*B

	full_text

float* %386
ecallB]
[
	full_textN
L
J%388 = tail call float @_Z4fminff(float %384, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %384
7fmulB/
-
	full_text 

%389 = fmul float %387, %388
(floatB

	full_text


float %387
(floatB

	full_text


float %388
\getelementptrBK
I
	full_text<
:
8%390 = getelementptr inbounds float, float* %2, i64 %385
$i64B

	full_text


i64 %385
LstoreBC
A
	full_text4
2
0store float %389, float* %390, align 4, !tbaa !8
(floatB

	full_text


float %389
*float*B

	full_text

float* %390
KloadBC
A
	full_text4
2
0%391 = load float, float* %89, align 4, !tbaa !8
)float*B

	full_text


float* %89
LloadBD
B
	full_text5
3
1%392 = load float, float* %131, align 4, !tbaa !8
*float*B

	full_text

float* %131
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
KloadBC
A
	full_text4
2
0%394 = load float, float* %94, align 4, !tbaa !8
)float*B

	full_text


float* %94
KloadBC
A
	full_text4
2
0%395 = load float, float* %17, align 4, !tbaa !8
)float*B

	full_text


float* %17
7fmulB/
-
	full_text 

%396 = fmul float %394, %395
(floatB

	full_text


float %394
(floatB

	full_text


float %395
LfdivBD
B
	full_text5
3
1%397 = fdiv float 1.000000e+00, %396, !fpmath !12
(floatB

	full_text


float %396
7fmulB/
-
	full_text 

%398 = fmul float %393, %397
(floatB

	full_text


float %393
(floatB

	full_text


float %397
0addB)
'
	full_text

%399 = add i64 %6, 792
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%400 = getelementptr inbounds float, float* %1, i64 %399
$i64B

	full_text


i64 %399
LloadBD
B
	full_text5
3
1%401 = load float, float* %400, align 4, !tbaa !8
*float*B

	full_text

float* %400
ecallB]
[
	full_textN
L
J%402 = tail call float @_Z4fminff(float %398, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %398
7fmulB/
-
	full_text 

%403 = fmul float %401, %402
(floatB

	full_text


float %401
(floatB

	full_text


float %402
\getelementptrBK
I
	full_text<
:
8%404 = getelementptr inbounds float, float* %2, i64 %399
$i64B

	full_text


i64 %399
LstoreBC
A
	full_text4
2
0store float %403, float* %404, align 4, !tbaa !8
(floatB

	full_text


float %403
*float*B

	full_text

float* %404
"retB

	full_text


ret void
*float*8B

	full_text

	float* %0
*float*8B

	full_text

	float* %1
*float*8B

	full_text

	float* %2
*float*8B

	full_text

	float* %3
(float8B

	full_text


float %4
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
%i648B

	full_text
	
i64 640
%i648B

	full_text
	
i64 696
%i648B

	full_text
	
i64 720
%i648B

	full_text
	
i64 672
%i648B

	full_text
	
i64 128
$i648B

	full_text


i64 96
$i648B

	full_text


i64 32
$i648B

	full_text


i64 72
8float8B+
)
	full_text

float 0x4415AF1D80000000
%i648B

	full_text
	
i64 704
%i648B

	full_text
	
i64 736
%i648B

	full_text
	
i64 104
%i648B

	full_text
	
i64 624
%i648B

	full_text
	
i64 656
%i648B

	full_text
	
i64 136
%i648B

	full_text
	
i64 744
%i648B

	full_text
	
i64 760
%i648B

	full_text
	
i64 120
%i648B

	full_text
	
i64 600
%i648B

	full_text
	
i64 160
%i648B

	full_text
	
i64 216
%i648B

	full_text
	
i64 712
%i648B

	full_text
	
i64 784
$i648B

	full_text


i64 88
$i648B

	full_text


i64 80
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
	
i64 664
$i648B

	full_text


i64 24
%i648B

	full_text
	
i64 168
%i648B

	full_text
	
i64 688
$i648B

	full_text


i64 64
%i648B

	full_text
	
i64 728
$i648B

	full_text


i64 48
%i648B

	full_text
	
i64 192
%i648B

	full_text
	
i64 200
$i648B

	full_text


i64 40
%i648B

	full_text
	
i64 632
%i648B

	full_text
	
i64 792
%i648B

	full_text
	
i64 184
#i328B

	full_text	

i32 0
%i648B

	full_text
	
i64 608
$i648B

	full_text


i64 16
%i648B

	full_text
	
i64 776
%i648B

	full_text
	
i64 680
2float8B%
#
	full_text

float 1.013250e+06
%i648B

	full_text
	
i64 176
2float8B%
#
	full_text

float 1.000000e+00
%i648B

	full_text
	
i64 648
%i648B

	full_text
	
i64 752
$i648B

	full_text


i64 56
%i648B

	full_text
	
i64 768
%i648B

	full_text
	
i64 616       	  
 

                       !" !! #$ ## %& %% '( '' )* )+ )) ,- ,, ./ .0 .. 12 11 34 33 56 55 78 77 9: 9; 99 <= << >? >@ >> AB AA CD CC EF EE GH GG IJ IK II LM LL NO NN PQ PP RS RR TU TT VW VV XY XZ XX [\ [[ ]^ ]_ ]] `a `` bc bb de dd fg ff hi hj hh kl kk mn mo mm pq pp rs rr tu tt vw vv xy xz xx {| {} {{ ~ ~~ Ä
Å ÄÄ ÇÉ ÇÇ Ñ
Ö ÑÑ Üá Ü
à ÜÜ âä ââ ã
å ãã çé çç èê èè ëí ë
ì ëë î
ï îî ñó ñ
ò ññ ôö ôô õ
ú õõ ùû ùù ü† üü °¢ °
£ °° §• §§ ¶ß ¶¶ ®© ®
™ ®® ´
¨ ´´ ≠Æ ≠
Ø ≠≠ ∞± ∞∞ ≤
≥ ≤≤ ¥µ ¥¥ ∂∑ ∂∂ ∏π ∏
∫ ∏∏ ª
º ªª Ωæ Ω
ø ΩΩ ¿¡ ¿¿ ¬
√ ¬¬ ƒ≈ ƒƒ ∆« ∆∆ »… »
  »» ÀÃ ÀÀ Õ
Œ ÕÕ œ– œœ —“ —— ”
‘ ”” ’÷ ’’ ◊ÿ ◊
Ÿ ◊◊ ⁄
€ ⁄⁄ ‹› ‹
ﬁ ‹‹ ﬂ‡ ﬂﬂ ·
‚ ·· „‰ „„ ÂÊ ÂÂ ÁË Á
È ÁÁ Í
Î ÍÍ ÏÌ Ï
Ó ÏÏ Ô ÔÔ ÒÚ ÒÒ ÛÙ Û
ı ÛÛ ˆ˜ ˆˆ ¯˘ ¯¯ ˙
˚ ˙˙ ¸˝ ¸¸ ˛ˇ ˛
Ä ˛˛ Å
Ç ÅÅ ÉÑ É
Ö ÉÉ Üá ÜÜ à
â àà äã ää åç åå éè é
ê éé ë
í ëë ìî ì
ï ìì ñó ññ ò
ô òò öõ öö úù úú ûü û
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
Í ËË ÎÏ ÎÎ ÌÓ ÌÌ Ô Ô
Ò ÔÔ Ú
Û ÚÚ Ùı Ù
ˆ ÙÙ ˜¯ ˜˜ ˘
˙ ˘˘ ˚¸ ˚˚ ˝˛ ˝˝ ˇÄ ˇ
Å ˇˇ Ç
É ÇÇ ÑÖ Ñ
Ü ÑÑ áà áá âä ââ ãå ã
ç ãã éè éé êë êê íì í
î íí ï
ñ ïï óò ó
ô óó öõ öö ú
ù úú ûü ûû †° †† ¢£ ¢
§ ¢¢ •
¶ •• ß® ß
© ßß ™´ ™™ ¨≠ ¨¨ ÆØ Æ
∞ ÆÆ ±≤ ±± ≥¥ ≥≥ µ∂ µ
∑ µµ ∏
π ∏∏ ∫ª ∫
º ∫∫ Ωæ ΩΩ ø
¿ øø ¡¬ ¡¡ √ƒ √√ ≈∆ ≈
« ≈≈ »
… »»  À  
Ã    ÕŒ ÕÕ œ– œœ —“ —
” —— ‘’ ‘‘ ÷◊ ÷÷ ÿ
Ÿ ÿÿ ⁄€ ⁄⁄ ‹› ‹
ﬁ ‹‹ ﬂ
‡ ﬂﬂ ·‚ ·
„ ·· ‰Â ‰‰ Ê
Á ÊÊ ËÈ ËË ÍÎ ÍÍ ÏÌ Ï
Ó ÏÏ Ô
 ÔÔ ÒÚ Ò
Û ÒÒ Ùı ÙÙ ˆ˜ ˆˆ ¯˘ ¯
˙ ¯¯ ˚¸ ˚˚ ˝˛ ˝˝ ˇ
Ä ˇˇ ÅÇ ÅÅ ÉÑ É
Ö ÉÉ Ü
á ÜÜ àâ à
ä àà ãå ãã ç
é çç èê èè ëí ëë ìî ì
ï ìì ñ
ó ññ òô ò
ö òò õú õõ ùû ùù ü† ü
° üü ¢£ ¢
§ ¢¢ •¶ •• ß
® ßß ©™ ©© ´
¨ ´´ ≠Æ ≠
Ø ≠≠ ∞± ∞∞ ≤
≥ ≤≤ ¥µ ¥¥ ∂∑ ∂∂ ∏π ∏
∫ ∏∏ ª
º ªª Ωæ Ω
ø ΩΩ ¿¡ ¿¿ ¬√ ¬¬ ƒ≈ ƒ
∆ ƒƒ «» «« …  …… ÀÃ À
Õ ÀÀ Œ
œ ŒŒ –— –
“ –– ”‘ ”” ’
÷ ’’ ◊ÿ ◊◊ Ÿ⁄ ŸŸ €‹ €
› €€ ﬁ
ﬂ ﬁﬁ ‡· ‡
‚ ‡‡ „‰ „„ ÂÊ ÂÂ ÁË Á
È ÁÁ ÍÎ ÍÍ ÏÌ ÏÏ Ó
Ô ÓÓ Ò  ÚÛ Ú
Ù ÚÚ ı
ˆ ıı ˜¯ ˜
˘ ˜˜ ˙˚ ˙˙ ¸
˝ ¸¸ ˛ˇ ˛˛ ÄÅ ÄÄ ÇÉ Ç
Ñ ÇÇ Ö
Ü ÖÖ áà á
â áá äã ää åç åå éè é
ê éé ëí ëë ìî ìì ïñ ï
ó ïï ò
ô òò öõ ö
ú öö ùû ùù ü
† üü °¢ °° £§ ££ •¶ •
ß •• ®
© ®® ™´ ™
¨ ™™ ≠Æ ≠≠ Ø∞ Ø
± ØØ ≤≥ ≤
¥ ≤≤ µ∂ µµ ∑
∏ ∑∑ π∫ ππ ª
º ªª Ωæ Ω
ø ΩΩ ¿¡ ¿¿ ¬
√ ¬¬ ƒ≈ ƒƒ ∆« ∆∆ »… »
  »» À
Ã ÀÀ ÕŒ Õ
œ ÕÕ –— –– “” “
‘ ““ ’÷ ’’ ◊ÿ ◊◊ Ÿ
⁄ ŸŸ €‹ €€ ›ﬁ ›
ﬂ ›› ‡
· ‡‡ ‚„ ‚
‰ ‚‚ ÂÊ ÂÂ Á
Ë ÁÁ ÈÍ ÈÈ ÎÏ ÎÎ ÌÓ Ì
Ô ÌÌ 
Ò  ÚÛ Ú
Ù ÚÚ ıˆ ıı ˜¯ ˜˜ ˘
˙ ˘˘ ˚¸ ˚˚ ˝˛ ˝
ˇ ˝˝ ÄÅ ÄÄ ÇÉ ÇÇ ÑÖ Ñ
Ü ÑÑ á
à áá âä â
ã ââ åç åå é
è éé êë êê íì íí îï î
ñ îî ó
ò óó ôö ô
õ ôô úù úú ûü ûû †° †
¢ †† £
§ ££ •¶ •• ß® ßß ©™ ©
´ ©© ¨
≠ ¨¨ ÆØ Æ
∞ ÆÆ ±≤ ±± ≥
¥ ≥≥ µ∂ µµ ∑∏ ∑∑ π∫ π
ª ππ º
Ω ºº æø æ
¿ ææ ¡¬ ¡¡ √ƒ √√ ≈∆ ≈
« ≈≈ »… »»  À    ÃÕ Ã
Œ ÃÃ œ
– œœ —“ —
” —— ‘’ ‘‘ ÷
◊ ÷÷ ÿŸ ÿÿ ⁄€ ⁄⁄ ‹› ‹
ﬁ ‹‹ ﬂ
‡ ﬂﬂ ·‚ ·
„ ·· ‰Â ‰‰ ÊÁ ÊÊ ËÈ Ë
Í ËË ÎÏ ÎÎ ÌÓ ÌÌ Ô Ô
Ò ÔÔ Ú
Û ÚÚ Ùı Ù
ˆ ÙÙ ˜¯ ˜˜ ˘
˙ ˘˘ ˚¸ ˚˚ ˝˛ ˝˝ ˇÄ ˇ
Å ˇˇ Ç
É ÇÇ ÑÖ Ñ
Ü ÑÑ áà áá âä ââ ãå ã
ç ãã éè éé êë êê íì í
î íí ï
ñ ïï óò ó
ô óó öõ öö ú
ù úú ûü ûû †° †† ¢£ ¢
§ ¢¢ •
¶ •• ß® ß
© ßß ™´ ™™ ¨≠ ¨¨ ÆØ Æ
∞ ÆÆ ±≤ ±± ≥¥ ≥≥ µ∂ µ
∑ µµ ∏
π ∏∏ ∫ª ∫
º ∫∫ Ωæ ΩΩ ø
¿ øø ¡¬ ¡¡ √ƒ √√ ≈∆ ≈
« ≈≈ »
… »»  À  
Ã    ÕŒ œ 3œ bœ ãœ ≤œ ·œ àœ ≥œ ÷œ ˘œ úœ øœ Êœ çœ ≤œ ’œ ¸œ üœ ¬œ Áœ éœ ≥œ ÷œ ˘œ úœ ø– <– k– î– ª– Í– ë– º– ﬂ– Ç– •– »– Ô– ñ– ª– ﬁ– Ö– ®– À– – ó– º– ﬂ– Ç– •– »— — — — %— C— N— T— t— Ä— õ— ¬— Õ— ”— ˙— ò— •— ÿ— ˇ— ß— Ó— ∑— Ÿ— ˘— £	“     	 
             " $# &% (! *' +) - /, 0 21 43 6. 85 :7 ;1 =9 ?< @ BA DC F HE JG K ML ON Q SR UT WP YV ZX \I ^[ _ a` cb e] gd if j` lh nk oN q sr ut wp yv z |x } ~ ÅÄ ÉÇ Ö{ áÑ à äâ åã éÜ êç íè ìâ ïë óî ò öô úõ ût †ù ¢ü £N • ß§ ©¶ ™® ¨° Æ´ Ø ±∞ ≥≤ µ≠ ∑¥ π∂ ∫∞ º∏ æª ø ¡¿ √¬ ≈t «ƒ …∆   ÃÀ ŒÕ – “— ‘” ÷œ ÿ’ Ÿ◊ €» ›⁄ ﬁ ‡ﬂ ‚· ‰‹ Ê„ ËÂ Èﬂ ÎÁ ÌÍ Ó¬ t ÚÔ ÙÒ ıÕ ˜ ˘¯ ˚˙ ˝ˆ ˇ¸ Ä˛ ÇÛ ÑÅ Ö áÜ âà ãÉ çä èå êÜ íé îë ï óñ ôò õt ùö üú †õ ¢ §£ ¶• ®° ™ß ´© ≠û Ø¨ ∞ ≤± ¥≥ ∂Æ ∏µ ∫∑ ª± Ωπ øº ¿ò ¬t ƒ¡ ∆√ «¬ … À» Õ  ŒÃ –≈ “œ ” ’‘ ◊÷ Ÿ— €ÿ ›⁄ ﬁ‘ ‡‹ ‚ﬂ „ Ât Á‰ ÈÊ Íò ÏÄ ÓÎ Ì ÒÔ ÛË ıÚ ˆ ¯˜ ˙˘ ¸Ù ˛˚ Ä˝ Å˜ Éˇ ÖÇ Ü àt äá åâ ç¬ è• ëé ìê îí ñã òï ô õö ùú üó °û £† §ö ¶¢ ®• © ´t ≠™ Ø¨ ∞ ≤Ä ¥± ∂≥ ∑µ πÆ ª∏ º æΩ ¿ø ¬∫ ƒ¡ ∆√ «Ω …≈ À» ÃC Œt –Õ “œ ”N ’ ◊÷ Ÿÿ €‘ ›⁄ ﬁ‹ ‡— ‚ﬂ „ Â‰ ÁÊ È· ÎË ÌÍ Ó‰ Ï ÚÔ Ût ı% ˜Ù ˘ˆ ˙Ä ¸ ˛˝ Äˇ Ç˚ ÑÅ ÖÉ á¯ âÜ ä åã éç êà íè îë ïã óì ôñ öt ú% ûõ †ù ° £ü § ¶• ®ß ™© ¨¢ Æ´ Ø ±∞ ≥≤ µ≠ ∑¥ π∂ ∫∞ º∏ æª øt ¡ √¿ ≈¬ ∆Ä »%  « Ã… ÕÀ œƒ —Œ “ ‘” ÷’ ÿ– ⁄◊ ‹Ÿ ›” ﬂ€ ·ﬁ ‚” ‰t Ê„ ËÂ ÈN Î ÌÏ ÔÓ ÒÍ Û ÙÚ ˆÁ ¯ı ˘ ˚˙ ˝¸ ˇ˜ Å˛ ÉÄ Ñ˙ ÜÇ àÖ â˙ ãt çä èå êN íÓ îë ñì óï ôé õò ú ûù †ü ¢ö §° ¶£ ßù ©• ´® ¨t Æ≠ ∞≠ ± ≥Ø ¥ ∂µ ∏∑ ∫π º≤ æª ø ¡¿ √¬ ≈Ω «ƒ …∆  ¿ Ã» ŒÀ œt —– ”– ‘N ÷ ÿ◊ ⁄Ÿ ‹’ ﬁ€ ﬂ› ·“ „‡ ‰ ÊÂ ËÁ Í‚ ÏÈ ÓÎ ÔÂ ÒÌ Û Ùt ˆ ¯˜ ˙˘ ¸ı ˛˚ ˇˇ ÅÓ ÉÄ ÖÇ ÜÑ à˝ äá ã çå èé ëâ ìê ïí ñå òî öó õN ù• üú °û ¢ §£ ¶ ®• ™ß ´© ≠† Ø¨ ∞ ≤± ¥≥ ∂Æ ∏µ ∫∑ ª± Ωπ øº ¿N ¬• ƒ¡ ∆√ «¬ …t À» Õ  ŒÃ –≈ “œ ” ’‘ ◊÷ Ÿ— €ÿ ›⁄ ﬁ‘ ‡‹ ‚ﬂ „N Â• Á‰ ÈÊ ÍÕ Ï˙ ÓÎ Ì ÒÔ ÛË ıÚ ˆ ¯˜ ˙˘ ¸Ù ˛˚ Ä˝ Å˜ Éˇ ÖÇ Üõ à• äá åâ ç¬ è ëé ìê îí ñã òï ô õö ùú üó °û £† §ö ¶¢ ®• ©¬ ´• ≠™ Ø¨ ∞Õ ≤ ¥± ∂≥ ∑µ πÆ ª∏ º æΩ ¿ø ¬∫ ƒ¡ ∆√ «Ω …≈ À» Ã Õ ”” ‘‘å ‘‘ å˝ ‘‘ ˝∂ ‘‘ ∂Î ‘‘ Î⁄ ‘‘ ⁄⁄ ‘‘ ⁄√ ‘‘ √† ‘‘ †Ÿ ‘‘ ŸÄ ‘‘ ÄÍ ‘‘ Í∑ ‘‘ ∑Â ‘‘ Â∑ ‘‘ ∑f ‘‘ f7 ‘‘ 7£ ‘‘ £è ‘‘ è∆ ‘‘ ∆† ‘‘ †√ ‘‘ √∂ ‘‘ ∂ ”” ë ‘‘ ëí ‘‘ í˝ ‘‘ ˝
’ Ü
÷ ã
◊ ˙
ÿ ö	Ÿ 	⁄ ~
€ ¿
‹ —	› 7	› f
› è
› ∂
› Â
› å
› ∑
› ⁄
› ˝
› †
› √
› Í
› ë
› ∂
› Ÿ
› Ä
› £
› ∆
› Î
› í
› ∑
› ⁄
› ˝
› †
› √
ﬁ ∞
ﬂ ¿
‡ ˝
· ∞
‚ ‘
„ £
‰ Â
Â ±	Ê #	Á 1
Ë ÷
È •
Í ”
Î ö	Ï r
Ì ¯	Ó 	Ô L
 ˜
Ò ñ
Ú Ï
Û ‰	Ù A
ı ù	ˆ 
˜ ˜	¯ R
˘ À
˙ ﬂ
˚ Ω
¸ µ˝ 	˛ `
ˇ ô
Ä ˜
Å Ω	Ç 
É ◊Ñ 
Ñ ,Ñ [Ñ ÑÑ ´Ñ ⁄Ñ ÅÑ ¨Ñ œÑ ÚÑ ïÑ ∏Ñ ﬂÑ ÜÑ ´Ñ ŒÑ ıÑ òÑ ªÑ ‡Ñ áÑ ¨Ñ œÑ ÚÑ ïÑ ∏
Ö ±
Ü å	á 
à ‘
â â"
ratt5_kernel"
_Z13get_global_idj"
	_Z4fminff*ó
shoc-1.1.5-S3D-ratt5_kernel.clu
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

transfer_bytes
à¢ª
 
transfer_bytes_log1p
íÛéA

wgsize
Ä

devmap_label
 