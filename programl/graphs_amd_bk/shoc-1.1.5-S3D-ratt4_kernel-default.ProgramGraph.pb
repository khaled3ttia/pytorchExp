
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
%13 = add i64 %6, 24
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
%16 = add i64 %6, 72
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
%29 = add i64 %6, 400
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
/%35 = load float, float* %14, align 4, !tbaa !8
)float*B

	full_text


float* %14
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
-addB&
$
	full_text

%38 = add i64 %6, 8
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
4fmulB,
*
	full_text

%41 = fmul float %40, %40
'floatB

	full_text

	float %40
'floatB

	full_text

	float %40
/addB(
&
	full_text

%42 = add i64 %6, 112
"i64B

	full_text


i64 %6
ZgetelementptrBI
G
	full_text:
8
6%43 = getelementptr inbounds float, float* %3, i64 %42
#i64B

	full_text
	
i64 %42
JloadBB
@
	full_text3
1
/%44 = load float, float* %43, align 4, !tbaa !8
)float*B

	full_text


float* %43
4fmulB,
*
	full_text

%45 = fmul float %41, %44
'floatB

	full_text

	float %41
'floatB

	full_text

	float %44
4fmulB,
*
	full_text

%46 = fmul float %12, %45
'floatB

	full_text

	float %12
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
%48 = fmul float %37, %47
'floatB

	full_text

	float %37
'floatB

	full_text

	float %47
/addB(
&
	full_text

%49 = add i64 %6, 408
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
/%55 = load float, float* %21, align 4, !tbaa !8
)float*B

	full_text


float* %21
JloadBB
@
	full_text3
1
/%56 = load float, float* %17, align 4, !tbaa !8
)float*B

	full_text


float* %17
4fmulB,
*
	full_text

%57 = fmul float %55, %56
'floatB

	full_text

	float %55
'floatB

	full_text

	float %56
JloadBB
@
	full_text3
1
/%58 = load float, float* %39, align 4, !tbaa !8
)float*B

	full_text


float* %39
/addB(
&
	full_text

%59 = add i64 %6, 128
"i64B

	full_text


i64 %6
ZgetelementptrBI
G
	full_text:
8
6%60 = getelementptr inbounds float, float* %3, i64 %59
#i64B

	full_text
	
i64 %59
JloadBB
@
	full_text3
1
/%61 = load float, float* %60, align 4, !tbaa !8
)float*B

	full_text


float* %60
4fmulB,
*
	full_text

%62 = fmul float %58, %61
'floatB

	full_text

	float %58
'floatB

	full_text

	float %61
JfdivBB
@
	full_text3
1
/%63 = fdiv float 1.000000e+00, %62, !fpmath !12
'floatB

	full_text

	float %62
4fmulB,
*
	full_text

%64 = fmul float %57, %63
'floatB

	full_text

	float %57
'floatB

	full_text

	float %63
/addB(
&
	full_text

%65 = add i64 %6, 416
"i64B

	full_text


i64 %6
ZgetelementptrBI
G
	full_text:
8
6%66 = getelementptr inbounds float, float* %1, i64 %65
#i64B

	full_text
	
i64 %65
JloadBB
@
	full_text3
1
/%67 = load float, float* %66, align 4, !tbaa !8
)float*B

	full_text


float* %66
ccallB[
Y
	full_textL
J
H%68 = tail call float @_Z4fminff(float %64, float 0x4415AF1D80000000) #2
'floatB

	full_text

	float %64
4fmulB,
*
	full_text

%69 = fmul float %67, %68
'floatB

	full_text

	float %67
'floatB

	full_text

	float %68
ZgetelementptrBI
G
	full_text:
8
6%70 = getelementptr inbounds float, float* %2, i64 %65
#i64B

	full_text
	
i64 %65
JstoreBA
?
	full_text2
0
.store float %69, float* %70, align 4, !tbaa !8
'floatB

	full_text

	float %69
)float*B

	full_text


float* %70
JloadBB
@
	full_text3
1
/%71 = load float, float* %21, align 4, !tbaa !8
)float*B

	full_text


float* %21
JloadBB
@
	full_text3
1
/%72 = load float, float* %17, align 4, !tbaa !8
)float*B

	full_text


float* %17
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
.addB'
%
	full_text

%74 = add i64 %6, 40
"i64B

	full_text


i64 %6
ZgetelementptrBI
G
	full_text:
8
6%75 = getelementptr inbounds float, float* %3, i64 %74
#i64B

	full_text
	
i64 %74
JloadBB
@
	full_text3
1
/%76 = load float, float* %75, align 4, !tbaa !8
)float*B

	full_text


float* %75
.addB'
%
	full_text

%77 = add i64 %6, 64
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
JfdivBB
@
	full_text3
1
/%81 = fdiv float 1.000000e+00, %80, !fpmath !12
'floatB

	full_text

	float %80
4fmulB,
*
	full_text

%82 = fmul float %73, %81
'floatB

	full_text

	float %73
'floatB

	full_text

	float %81
/addB(
&
	full_text

%83 = add i64 %6, 424
"i64B

	full_text


i64 %6
ZgetelementptrBI
G
	full_text:
8
6%84 = getelementptr inbounds float, float* %1, i64 %83
#i64B

	full_text
	
i64 %83
JloadBB
@
	full_text3
1
/%85 = load float, float* %84, align 4, !tbaa !8
)float*B

	full_text


float* %84
ccallB[
Y
	full_textL
J
H%86 = tail call float @_Z4fminff(float %82, float 0x4415AF1D80000000) #2
'floatB

	full_text

	float %82
4fmulB,
*
	full_text

%87 = fmul float %85, %86
'floatB

	full_text

	float %85
'floatB

	full_text

	float %86
ZgetelementptrBI
G
	full_text:
8
6%88 = getelementptr inbounds float, float* %2, i64 %83
#i64B

	full_text
	
i64 %83
JstoreBA
?
	full_text2
0
.store float %87, float* %88, align 4, !tbaa !8
'floatB

	full_text

	float %87
)float*B

	full_text


float* %88
.addB'
%
	full_text

%89 = add i64 %6, 48
"i64B

	full_text


i64 %6
ZgetelementptrBI
G
	full_text:
8
6%90 = getelementptr inbounds float, float* %3, i64 %89
#i64B

	full_text
	
i64 %89
JloadBB
@
	full_text3
1
/%91 = load float, float* %90, align 4, !tbaa !8
)float*B

	full_text


float* %90
JloadBB
@
	full_text3
1
/%92 = load float, float* %17, align 4, !tbaa !8
)float*B

	full_text


float* %17
4fmulB,
*
	full_text

%93 = fmul float %91, %92
'floatB

	full_text

	float %91
'floatB

	full_text

	float %92
JloadBB
@
	full_text3
1
/%94 = load float, float* %21, align 4, !tbaa !8
)float*B

	full_text


float* %21
JloadBB
@
	full_text3
1
/%95 = load float, float* %60, align 4, !tbaa !8
)float*B

	full_text


float* %60
4fmulB,
*
	full_text

%96 = fmul float %94, %95
'floatB

	full_text

	float %94
'floatB

	full_text

	float %95
JfdivBB
@
	full_text3
1
/%97 = fdiv float 1.000000e+00, %96, !fpmath !12
'floatB

	full_text

	float %96
4fmulB,
*
	full_text

%98 = fmul float %93, %97
'floatB

	full_text

	float %93
'floatB

	full_text

	float %97
/addB(
&
	full_text

%99 = add i64 %6, 432
"i64B

	full_text


i64 %6
[getelementptrBJ
H
	full_text;
9
7%100 = getelementptr inbounds float, float* %1, i64 %99
#i64B

	full_text
	
i64 %99
LloadBD
B
	full_text5
3
1%101 = load float, float* %100, align 4, !tbaa !8
*float*B

	full_text

float* %100
dcallB\
Z
	full_textM
K
I%102 = tail call float @_Z4fminff(float %98, float 0x4415AF1D80000000) #2
'floatB

	full_text

	float %98
7fmulB/
-
	full_text 

%103 = fmul float %101, %102
(floatB

	full_text


float %101
(floatB

	full_text


float %102
[getelementptrBJ
H
	full_text;
9
7%104 = getelementptr inbounds float, float* %2, i64 %99
#i64B

	full_text
	
i64 %99
LstoreBC
A
	full_text4
2
0store float %103, float* %104, align 4, !tbaa !8
(floatB

	full_text


float %103
*float*B

	full_text

float* %104
KloadBC
A
	full_text4
2
0%105 = load float, float* %17, align 4, !tbaa !8
)float*B

	full_text


float* %17
0addB)
'
	full_text

%106 = add i64 %6, 104
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%107 = getelementptr inbounds float, float* %3, i64 %106
$i64B

	full_text


i64 %106
LloadBD
B
	full_text5
3
1%108 = load float, float* %107, align 4, !tbaa !8
*float*B

	full_text

float* %107
7fmulB/
-
	full_text 

%109 = fmul float %105, %108
(floatB

	full_text


float %105
(floatB

	full_text


float %108
6fmulB.
,
	full_text

%110 = fmul float %12, %109
'floatB

	full_text

	float %12
(floatB

	full_text


float %109
0addB)
'
	full_text

%111 = add i64 %6, 200
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%112 = getelementptr inbounds float, float* %3, i64 %111
$i64B

	full_text


i64 %111
LloadBD
B
	full_text5
3
1%113 = load float, float* %112, align 4, !tbaa !8
*float*B

	full_text

float* %112
LfdivBD
B
	full_text5
3
1%114 = fdiv float 1.000000e+00, %113, !fpmath !12
(floatB

	full_text


float %113
7fmulB/
-
	full_text 

%115 = fmul float %110, %114
(floatB

	full_text


float %110
(floatB

	full_text


float %114
0addB)
'
	full_text

%116 = add i64 %6, 440
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%117 = getelementptr inbounds float, float* %1, i64 %116
$i64B

	full_text


i64 %116
LloadBD
B
	full_text5
3
1%118 = load float, float* %117, align 4, !tbaa !8
*float*B

	full_text

float* %117
ecallB]
[
	full_textN
L
J%119 = tail call float @_Z4fminff(float %115, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %115
7fmulB/
-
	full_text 

%120 = fmul float %118, %119
(floatB

	full_text


float %118
(floatB

	full_text


float %119
\getelementptrBK
I
	full_text<
:
8%121 = getelementptr inbounds float, float* %2, i64 %116
$i64B

	full_text


i64 %116
LstoreBC
A
	full_text4
2
0store float %120, float* %121, align 4, !tbaa !8
(floatB

	full_text


float %120
*float*B

	full_text

float* %121
KloadBC
A
	full_text4
2
0%122 = load float, float* %78, align 4, !tbaa !8
)float*B

	full_text


float* %78
KloadBC
A
	full_text4
2
0%123 = load float, float* %17, align 4, !tbaa !8
)float*B

	full_text


float* %17
7fmulB/
-
	full_text 

%124 = fmul float %122, %123
(floatB

	full_text


float %122
(floatB

	full_text


float %123
KloadBC
A
	full_text4
2
0%125 = load float, float* %39, align 4, !tbaa !8
)float*B

	full_text


float* %39
0addB)
'
	full_text

%126 = add i64 %6, 144
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%127 = getelementptr inbounds float, float* %3, i64 %126
$i64B

	full_text


i64 %126
LloadBD
B
	full_text5
3
1%128 = load float, float* %127, align 4, !tbaa !8
*float*B

	full_text

float* %127
7fmulB/
-
	full_text 

%129 = fmul float %125, %128
(floatB

	full_text


float %125
(floatB

	full_text


float %128
LfdivBD
B
	full_text5
3
1%130 = fdiv float 1.000000e+00, %129, !fpmath !12
(floatB

	full_text


float %129
7fmulB/
-
	full_text 

%131 = fmul float %124, %130
(floatB

	full_text


float %124
(floatB

	full_text


float %130
0addB)
'
	full_text

%132 = add i64 %6, 448
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%133 = getelementptr inbounds float, float* %1, i64 %132
$i64B

	full_text


i64 %132
LloadBD
B
	full_text5
3
1%134 = load float, float* %133, align 4, !tbaa !8
*float*B

	full_text

float* %133
ecallB]
[
	full_textN
L
J%135 = tail call float @_Z4fminff(float %131, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %131
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
\getelementptrBK
I
	full_text<
:
8%137 = getelementptr inbounds float, float* %2, i64 %132
$i64B

	full_text


i64 %132
LstoreBC
A
	full_text4
2
0store float %136, float* %137, align 4, !tbaa !8
(floatB

	full_text


float %136
*float*B

	full_text

float* %137
KloadBC
A
	full_text4
2
0%138 = load float, float* %17, align 4, !tbaa !8
)float*B

	full_text


float* %17
7fmulB/
-
	full_text 

%139 = fmul float %138, %138
(floatB

	full_text


float %138
(floatB

	full_text


float %138
ZgetelementptrBI
G
	full_text:
8
6%140 = getelementptr inbounds float, float* %3, i64 %6
"i64B

	full_text


i64 %6
LloadBD
B
	full_text5
3
1%141 = load float, float* %140, align 4, !tbaa !8
*float*B

	full_text

float* %140
LloadBD
B
	full_text5
3
1%142 = load float, float* %127, align 4, !tbaa !8
*float*B

	full_text

float* %127
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
LfdivBD
B
	full_text5
3
1%144 = fdiv float 1.000000e+00, %143, !fpmath !12
(floatB

	full_text


float %143
7fmulB/
-
	full_text 

%145 = fmul float %139, %144
(floatB

	full_text


float %139
(floatB

	full_text


float %144
0addB)
'
	full_text

%146 = add i64 %6, 456
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%147 = getelementptr inbounds float, float* %1, i64 %146
$i64B

	full_text


i64 %146
LloadBD
B
	full_text5
3
1%148 = load float, float* %147, align 4, !tbaa !8
*float*B

	full_text

float* %147
ecallB]
[
	full_textN
L
J%149 = tail call float @_Z4fminff(float %145, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %145
7fmulB/
-
	full_text 

%150 = fmul float %148, %149
(floatB

	full_text


float %148
(floatB

	full_text


float %149
\getelementptrBK
I
	full_text<
:
8%151 = getelementptr inbounds float, float* %2, i64 %146
$i64B

	full_text


i64 %146
LstoreBC
A
	full_text4
2
0store float %150, float* %151, align 4, !tbaa !8
(floatB

	full_text


float %150
*float*B

	full_text

float* %151
/addB(
&
	full_text

%152 = add i64 %6, 80
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%153 = getelementptr inbounds float, float* %3, i64 %152
$i64B

	full_text


i64 %152
LloadBD
B
	full_text5
3
1%154 = load float, float* %153, align 4, !tbaa !8
*float*B

	full_text

float* %153
KloadBC
A
	full_text4
2
0%155 = load float, float* %17, align 4, !tbaa !8
)float*B

	full_text


float* %17
LfdivBD
B
	full_text5
3
1%156 = fdiv float 1.000000e+00, %155, !fpmath !12
(floatB

	full_text


float %155
7fmulB/
-
	full_text 

%157 = fmul float %154, %156
(floatB

	full_text


float %154
(floatB

	full_text


float %156
0addB)
'
	full_text

%158 = add i64 %6, 464
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%159 = getelementptr inbounds float, float* %1, i64 %158
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
ecallB]
[
	full_textN
L
J%161 = tail call float @_Z4fminff(float %157, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %157
7fmulB/
-
	full_text 

%162 = fmul float %160, %161
(floatB

	full_text


float %160
(floatB

	full_text


float %161
\getelementptrBK
I
	full_text<
:
8%163 = getelementptr inbounds float, float* %2, i64 %158
$i64B

	full_text


i64 %158
LstoreBC
A
	full_text4
2
0store float %162, float* %163, align 4, !tbaa !8
(floatB

	full_text


float %162
*float*B

	full_text

float* %163
KloadBC
A
	full_text4
2
0%164 = load float, float* %39, align 4, !tbaa !8
)float*B

	full_text


float* %39
LloadBD
B
	full_text5
3
1%165 = load float, float* %153, align 4, !tbaa !8
*float*B

	full_text

float* %153
7fmulB/
-
	full_text 

%166 = fmul float %164, %165
(floatB

	full_text


float %164
(floatB

	full_text


float %165
LloadBD
B
	full_text5
3
1%167 = load float, float* %140, align 4, !tbaa !8
*float*B

	full_text

float* %140
KloadBC
A
	full_text4
2
0%168 = load float, float* %78, align 4, !tbaa !8
)float*B

	full_text


float* %78
7fmulB/
-
	full_text 

%169 = fmul float %167, %168
(floatB

	full_text


float %167
(floatB

	full_text


float %168
LfdivBD
B
	full_text5
3
1%170 = fdiv float 1.000000e+00, %169, !fpmath !12
(floatB

	full_text


float %169
7fmulB/
-
	full_text 

%171 = fmul float %166, %170
(floatB

	full_text


float %166
(floatB

	full_text


float %170
0addB)
'
	full_text

%172 = add i64 %6, 472
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%173 = getelementptr inbounds float, float* %1, i64 %172
$i64B

	full_text


i64 %172
LloadBD
B
	full_text5
3
1%174 = load float, float* %173, align 4, !tbaa !8
*float*B

	full_text

float* %173
ecallB]
[
	full_textN
L
J%175 = tail call float @_Z4fminff(float %171, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %171
7fmulB/
-
	full_text 

%176 = fmul float %174, %175
(floatB

	full_text


float %174
(floatB

	full_text


float %175
\getelementptrBK
I
	full_text<
:
8%177 = getelementptr inbounds float, float* %2, i64 %172
$i64B

	full_text


i64 %172
LstoreBC
A
	full_text4
2
0store float %176, float* %177, align 4, !tbaa !8
(floatB

	full_text


float %176
*float*B

	full_text

float* %177
/addB(
&
	full_text

%178 = add i64 %6, 16
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%179 = getelementptr inbounds float, float* %3, i64 %178
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
LloadBD
B
	full_text5
3
1%181 = load float, float* %153, align 4, !tbaa !8
*float*B

	full_text

float* %153
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
LloadBD
B
	full_text5
3
1%183 = load float, float* %140, align 4, !tbaa !8
*float*B

	full_text

float* %140
LloadBD
B
	full_text5
3
1%184 = load float, float* %107, align 4, !tbaa !8
*float*B

	full_text

float* %107
7fmulB/
-
	full_text 

%185 = fmul float %183, %184
(floatB

	full_text


float %183
(floatB

	full_text


float %184
LfdivBD
B
	full_text5
3
1%186 = fdiv float 1.000000e+00, %185, !fpmath !12
(floatB

	full_text


float %185
7fmulB/
-
	full_text 

%187 = fmul float %182, %186
(floatB

	full_text


float %182
(floatB

	full_text


float %186
0addB)
'
	full_text

%188 = add i64 %6, 480
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%189 = getelementptr inbounds float, float* %1, i64 %188
$i64B

	full_text


i64 %188
LloadBD
B
	full_text5
3
1%190 = load float, float* %189, align 4, !tbaa !8
*float*B

	full_text

float* %189
ecallB]
[
	full_textN
L
J%191 = tail call float @_Z4fminff(float %187, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %187
7fmulB/
-
	full_text 

%192 = fmul float %190, %191
(floatB

	full_text


float %190
(floatB

	full_text


float %191
\getelementptrBK
I
	full_text<
:
8%193 = getelementptr inbounds float, float* %2, i64 %188
$i64B

	full_text


i64 %188
LstoreBC
A
	full_text4
2
0store float %192, float* %193, align 4, !tbaa !8
(floatB

	full_text


float %192
*float*B

	full_text

float* %193
LloadBD
B
	full_text5
3
1%194 = load float, float* %179, align 4, !tbaa !8
*float*B

	full_text

float* %179
LloadBD
B
	full_text5
3
1%195 = load float, float* %153, align 4, !tbaa !8
*float*B

	full_text

float* %153
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
KloadBC
A
	full_text4
2
0%197 = load float, float* %39, align 4, !tbaa !8
)float*B

	full_text


float* %39
KloadBC
A
	full_text4
2
0%198 = load float, float* %24, align 4, !tbaa !8
)float*B

	full_text


float* %24
7fmulB/
-
	full_text 

%199 = fmul float %197, %198
(floatB

	full_text


float %197
(floatB

	full_text


float %198
LfdivBD
B
	full_text5
3
1%200 = fdiv float 1.000000e+00, %199, !fpmath !12
(floatB

	full_text


float %199
7fmulB/
-
	full_text 

%201 = fmul float %196, %200
(floatB

	full_text


float %196
(floatB

	full_text


float %200
0addB)
'
	full_text

%202 = add i64 %6, 488
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%203 = getelementptr inbounds float, float* %1, i64 %202
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
ecallB]
[
	full_textN
L
J%205 = tail call float @_Z4fminff(float %201, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %201
7fmulB/
-
	full_text 

%206 = fmul float %204, %205
(floatB

	full_text


float %204
(floatB

	full_text


float %205
\getelementptrBK
I
	full_text<
:
8%207 = getelementptr inbounds float, float* %2, i64 %202
$i64B

	full_text


i64 %202
LstoreBC
A
	full_text4
2
0store float %206, float* %207, align 4, !tbaa !8
(floatB

	full_text


float %206
*float*B

	full_text

float* %207
KloadBC
A
	full_text4
2
0%208 = load float, float* %21, align 4, !tbaa !8
)float*B

	full_text


float* %21
LloadBD
B
	full_text5
3
1%209 = load float, float* %153, align 4, !tbaa !8
*float*B

	full_text

float* %153
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
KloadBC
A
	full_text4
2
0%211 = load float, float* %39, align 4, !tbaa !8
)float*B

	full_text


float* %39
KloadBC
A
	full_text4
2
0%212 = load float, float* %60, align 4, !tbaa !8
)float*B

	full_text


float* %60
7fmulB/
-
	full_text 

%213 = fmul float %211, %212
(floatB

	full_text


float %211
(floatB

	full_text


float %212
LfdivBD
B
	full_text5
3
1%214 = fdiv float 1.000000e+00, %213, !fpmath !12
(floatB

	full_text


float %213
7fmulB/
-
	full_text 

%215 = fmul float %210, %214
(floatB

	full_text


float %210
(floatB

	full_text


float %214
0addB)
'
	full_text

%216 = add i64 %6, 496
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%217 = getelementptr inbounds float, float* %1, i64 %216
$i64B

	full_text


i64 %216
LloadBD
B
	full_text5
3
1%218 = load float, float* %217, align 4, !tbaa !8
*float*B

	full_text

float* %217
ecallB]
[
	full_textN
L
J%219 = tail call float @_Z4fminff(float %215, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %215
7fmulB/
-
	full_text 

%220 = fmul float %218, %219
(floatB

	full_text


float %218
(floatB

	full_text


float %219
\getelementptrBK
I
	full_text<
:
8%221 = getelementptr inbounds float, float* %2, i64 %216
$i64B

	full_text


i64 %216
LstoreBC
A
	full_text4
2
0store float %220, float* %221, align 4, !tbaa !8
(floatB

	full_text


float %220
*float*B

	full_text

float* %221
LloadBD
B
	full_text5
3
1%222 = load float, float* %140, align 4, !tbaa !8
*float*B

	full_text

float* %140
LloadBD
B
	full_text5
3
1%223 = load float, float* %153, align 4, !tbaa !8
*float*B

	full_text

float* %153
7fmulB/
-
	full_text 

%224 = fmul float %222, %223
(floatB

	full_text


float %222
(floatB

	full_text


float %223
KloadBC
A
	full_text4
2
0%225 = load float, float* %39, align 4, !tbaa !8
)float*B

	full_text


float* %39
/addB(
&
	full_text

%226 = add i64 %6, 88
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%227 = getelementptr inbounds float, float* %3, i64 %226
$i64B

	full_text


i64 %226
LloadBD
B
	full_text5
3
1%228 = load float, float* %227, align 4, !tbaa !8
*float*B

	full_text

float* %227
7fmulB/
-
	full_text 

%229 = fmul float %225, %228
(floatB

	full_text


float %225
(floatB

	full_text


float %228
LfdivBD
B
	full_text5
3
1%230 = fdiv float 1.000000e+00, %229, !fpmath !12
(floatB

	full_text


float %229
7fmulB/
-
	full_text 

%231 = fmul float %224, %230
(floatB

	full_text


float %224
(floatB

	full_text


float %230
0addB)
'
	full_text

%232 = add i64 %6, 504
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%233 = getelementptr inbounds float, float* %1, i64 %232
$i64B

	full_text


i64 %232
LloadBD
B
	full_text5
3
1%234 = load float, float* %233, align 4, !tbaa !8
*float*B

	full_text

float* %233
ecallB]
[
	full_textN
L
J%235 = tail call float @_Z4fminff(float %231, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %231
7fmulB/
-
	full_text 

%236 = fmul float %234, %235
(floatB

	full_text


float %234
(floatB

	full_text


float %235
\getelementptrBK
I
	full_text<
:
8%237 = getelementptr inbounds float, float* %2, i64 %232
$i64B

	full_text


i64 %232
LstoreBC
A
	full_text4
2
0store float %236, float* %237, align 4, !tbaa !8
(floatB

	full_text


float %236
*float*B

	full_text

float* %237
KloadBC
A
	full_text4
2
0%238 = load float, float* %14, align 4, !tbaa !8
)float*B

	full_text


float* %14
LloadBD
B
	full_text5
3
1%239 = load float, float* %153, align 4, !tbaa !8
*float*B

	full_text

float* %153
7fmulB/
-
	full_text 

%240 = fmul float %238, %239
(floatB

	full_text


float %238
(floatB

	full_text


float %239
KloadBC
A
	full_text4
2
0%241 = load float, float* %39, align 4, !tbaa !8
)float*B

	full_text


float* %39
KloadBC
A
	full_text4
2
0%242 = load float, float* %21, align 4, !tbaa !8
)float*B

	full_text


float* %21
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
LloadBD
B
	full_text5
3
1%244 = load float, float* %107, align 4, !tbaa !8
*float*B

	full_text

float* %107
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
6fmulB.
,
	full_text

%246 = fmul float %12, %245
'floatB

	full_text

	float %12
(floatB

	full_text


float %245
LfdivBD
B
	full_text5
3
1%247 = fdiv float 1.000000e+00, %246, !fpmath !12
(floatB

	full_text


float %246
7fmulB/
-
	full_text 

%248 = fmul float %240, %247
(floatB

	full_text


float %240
(floatB

	full_text


float %247
0addB)
'
	full_text

%249 = add i64 %6, 512
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%250 = getelementptr inbounds float, float* %1, i64 %249
$i64B

	full_text


i64 %249
LloadBD
B
	full_text5
3
1%251 = load float, float* %250, align 4, !tbaa !8
*float*B

	full_text

float* %250
ecallB]
[
	full_textN
L
J%252 = tail call float @_Z4fminff(float %248, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %248
7fmulB/
-
	full_text 

%253 = fmul float %251, %252
(floatB

	full_text


float %251
(floatB

	full_text


float %252
\getelementptrBK
I
	full_text<
:
8%254 = getelementptr inbounds float, float* %2, i64 %249
$i64B

	full_text


i64 %249
LstoreBC
A
	full_text4
2
0store float %253, float* %254, align 4, !tbaa !8
(floatB

	full_text


float %253
*float*B

	full_text

float* %254
KloadBC
A
	full_text4
2
0%255 = load float, float* %14, align 4, !tbaa !8
)float*B

	full_text


float* %14
LloadBD
B
	full_text5
3
1%256 = load float, float* %153, align 4, !tbaa !8
*float*B

	full_text

float* %153
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
KloadBC
A
	full_text4
2
0%258 = load float, float* %75, align 4, !tbaa !8
)float*B

	full_text


float* %75
LloadBD
B
	full_text5
3
1%259 = load float, float* %107, align 4, !tbaa !8
*float*B

	full_text

float* %107
7fmulB/
-
	full_text 

%260 = fmul float %258, %259
(floatB

	full_text


float %258
(floatB

	full_text


float %259
LfdivBD
B
	full_text5
3
1%261 = fdiv float 1.000000e+00, %260, !fpmath !12
(floatB

	full_text


float %260
7fmulB/
-
	full_text 

%262 = fmul float %257, %261
(floatB

	full_text


float %257
(floatB

	full_text


float %261
0addB)
'
	full_text

%263 = add i64 %6, 520
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%264 = getelementptr inbounds float, float* %1, i64 %263
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
ecallB]
[
	full_textN
L
J%266 = tail call float @_Z4fminff(float %262, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %262
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
\getelementptrBK
I
	full_text<
:
8%268 = getelementptr inbounds float, float* %2, i64 %263
$i64B

	full_text


i64 %263
LstoreBC
A
	full_text4
2
0store float %267, float* %268, align 4, !tbaa !8
(floatB

	full_text


float %267
*float*B

	full_text

float* %268
LloadBD
B
	full_text5
3
1%269 = load float, float* %153, align 4, !tbaa !8
*float*B

	full_text

float* %153
KloadBC
A
	full_text4
2
0%270 = load float, float* %17, align 4, !tbaa !8
)float*B

	full_text


float* %17
LfdivBD
B
	full_text5
3
1%271 = fdiv float 1.000000e+00, %270, !fpmath !12
(floatB

	full_text


float %270
7fmulB/
-
	full_text 

%272 = fmul float %269, %271
(floatB

	full_text


float %269
(floatB

	full_text


float %271
0addB)
'
	full_text

%273 = add i64 %6, 528
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%274 = getelementptr inbounds float, float* %1, i64 %273
$i64B

	full_text


i64 %273
LloadBD
B
	full_text5
3
1%275 = load float, float* %274, align 4, !tbaa !8
*float*B

	full_text

float* %274
ecallB]
[
	full_textN
L
J%276 = tail call float @_Z4fminff(float %272, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %272
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
\getelementptrBK
I
	full_text<
:
8%278 = getelementptr inbounds float, float* %2, i64 %273
$i64B

	full_text


i64 %273
LstoreBC
A
	full_text4
2
0store float %277, float* %278, align 4, !tbaa !8
(floatB

	full_text


float %277
*float*B

	full_text

float* %278
LloadBD
B
	full_text5
3
1%279 = load float, float* %153, align 4, !tbaa !8
*float*B

	full_text

float* %153
KloadBC
A
	full_text4
2
0%280 = load float, float* %17, align 4, !tbaa !8
)float*B

	full_text


float* %17
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

%282 = fmul float %279, %281
(floatB

	full_text


float %279
(floatB

	full_text


float %281
0addB)
'
	full_text

%283 = add i64 %6, 536
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
LloadBD
B
	full_text5
3
1%289 = load float, float* %153, align 4, !tbaa !8
*float*B

	full_text

float* %153
KloadBC
A
	full_text4
2
0%290 = load float, float* %17, align 4, !tbaa !8
)float*B

	full_text


float* %17
LfdivBD
B
	full_text5
3
1%291 = fdiv float 1.000000e+00, %290, !fpmath !12
(floatB

	full_text


float %290
7fmulB/
-
	full_text 

%292 = fmul float %289, %291
(floatB

	full_text


float %289
(floatB

	full_text


float %291
0addB)
'
	full_text

%293 = add i64 %6, 544
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%294 = getelementptr inbounds float, float* %1, i64 %293
$i64B

	full_text


i64 %293
LloadBD
B
	full_text5
3
1%295 = load float, float* %294, align 4, !tbaa !8
*float*B

	full_text

float* %294
ecallB]
[
	full_textN
L
J%296 = tail call float @_Z4fminff(float %292, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %292
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
\getelementptrBK
I
	full_text<
:
8%298 = getelementptr inbounds float, float* %2, i64 %293
$i64B

	full_text


i64 %293
LstoreBC
A
	full_text4
2
0store float %297, float* %298, align 4, !tbaa !8
(floatB

	full_text


float %297
*float*B

	full_text

float* %298
LloadBD
B
	full_text5
3
1%299 = load float, float* %153, align 4, !tbaa !8
*float*B

	full_text

float* %153
KloadBC
A
	full_text4
2
0%300 = load float, float* %43, align 4, !tbaa !8
)float*B

	full_text


float* %43
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
LloadBD
B
	full_text5
3
1%302 = load float, float* %107, align 4, !tbaa !8
*float*B

	full_text

float* %107
KloadBC
A
	full_text4
2
0%303 = load float, float* %60, align 4, !tbaa !8
)float*B

	full_text


float* %60
7fmulB/
-
	full_text 

%304 = fmul float %302, %303
(floatB

	full_text


float %302
(floatB

	full_text


float %303
LfdivBD
B
	full_text5
3
1%305 = fdiv float 1.000000e+00, %304, !fpmath !12
(floatB

	full_text


float %304
7fmulB/
-
	full_text 

%306 = fmul float %301, %305
(floatB

	full_text


float %301
(floatB

	full_text


float %305
0addB)
'
	full_text

%307 = add i64 %6, 552
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%308 = getelementptr inbounds float, float* %1, i64 %307
$i64B

	full_text


i64 %307
LloadBD
B
	full_text5
3
1%309 = load float, float* %308, align 4, !tbaa !8
*float*B

	full_text

float* %308
ecallB]
[
	full_textN
L
J%310 = tail call float @_Z4fminff(float %306, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %306
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
\getelementptrBK
I
	full_text<
:
8%312 = getelementptr inbounds float, float* %2, i64 %307
$i64B

	full_text


i64 %307
LstoreBC
A
	full_text4
2
0store float %311, float* %312, align 4, !tbaa !8
(floatB

	full_text


float %311
*float*B

	full_text

float* %312
KloadBC
A
	full_text4
2
0%313 = load float, float* %39, align 4, !tbaa !8
)float*B

	full_text


float* %39
KloadBC
A
	full_text4
2
0%314 = load float, float* %60, align 4, !tbaa !8
)float*B

	full_text


float* %60
7fmulB/
-
	full_text 

%315 = fmul float %313, %314
(floatB

	full_text


float %313
(floatB

	full_text


float %314
6fmulB.
,
	full_text

%316 = fmul float %12, %315
'floatB

	full_text

	float %12
(floatB

	full_text


float %315
0addB)
'
	full_text

%317 = add i64 %6, 136
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%318 = getelementptr inbounds float, float* %3, i64 %317
$i64B

	full_text


i64 %317
LloadBD
B
	full_text5
3
1%319 = load float, float* %318, align 4, !tbaa !8
*float*B

	full_text

float* %318
LfdivBD
B
	full_text5
3
1%320 = fdiv float 1.000000e+00, %319, !fpmath !12
(floatB

	full_text


float %319
7fmulB/
-
	full_text 

%321 = fmul float %316, %320
(floatB

	full_text


float %316
(floatB

	full_text


float %320
0addB)
'
	full_text

%322 = add i64 %6, 560
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%323 = getelementptr inbounds float, float* %1, i64 %322
$i64B

	full_text


i64 %322
LloadBD
B
	full_text5
3
1%324 = load float, float* %323, align 4, !tbaa !8
*float*B

	full_text

float* %323
ecallB]
[
	full_textN
L
J%325 = tail call float @_Z4fminff(float %321, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %321
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
\getelementptrBK
I
	full_text<
:
8%327 = getelementptr inbounds float, float* %2, i64 %322
$i64B

	full_text


i64 %322
LstoreBC
A
	full_text4
2
0store float %326, float* %327, align 4, !tbaa !8
(floatB

	full_text


float %326
*float*B

	full_text

float* %327
KloadBC
A
	full_text4
2
0%328 = load float, float* %39, align 4, !tbaa !8
)float*B

	full_text


float* %39
KloadBC
A
	full_text4
2
0%329 = load float, float* %60, align 4, !tbaa !8
)float*B

	full_text


float* %60
7fmulB/
-
	full_text 

%330 = fmul float %328, %329
(floatB

	full_text


float %328
(floatB

	full_text


float %329
LloadBD
B
	full_text5
3
1%331 = load float, float* %140, align 4, !tbaa !8
*float*B

	full_text

float* %140
KloadBC
A
	full_text4
2
0%332 = load float, float* %24, align 4, !tbaa !8
)float*B

	full_text


float* %24
7fmulB/
-
	full_text 

%333 = fmul float %331, %332
(floatB

	full_text


float %331
(floatB

	full_text


float %332
LfdivBD
B
	full_text5
3
1%334 = fdiv float 1.000000e+00, %333, !fpmath !12
(floatB

	full_text


float %333
7fmulB/
-
	full_text 

%335 = fmul float %330, %334
(floatB

	full_text


float %330
(floatB

	full_text


float %334
0addB)
'
	full_text

%336 = add i64 %6, 568
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%337 = getelementptr inbounds float, float* %1, i64 %336
$i64B

	full_text


i64 %336
LloadBD
B
	full_text5
3
1%338 = load float, float* %337, align 4, !tbaa !8
*float*B

	full_text

float* %337
ecallB]
[
	full_textN
L
J%339 = tail call float @_Z4fminff(float %335, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %335
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
\getelementptrBK
I
	full_text<
:
8%341 = getelementptr inbounds float, float* %2, i64 %336
$i64B

	full_text


i64 %336
LstoreBC
A
	full_text4
2
0store float %340, float* %341, align 4, !tbaa !8
(floatB

	full_text


float %340
*float*B

	full_text

float* %341
LloadBD
B
	full_text5
3
1%342 = load float, float* %179, align 4, !tbaa !8
*float*B

	full_text

float* %179
KloadBC
A
	full_text4
2
0%343 = load float, float* %60, align 4, !tbaa !8
)float*B

	full_text


float* %60
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
KloadBC
A
	full_text4
2
0%345 = load float, float* %21, align 4, !tbaa !8
)float*B

	full_text


float* %21
KloadBC
A
	full_text4
2
0%346 = load float, float* %24, align 4, !tbaa !8
)float*B

	full_text


float* %24
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
LfdivBD
B
	full_text5
3
1%348 = fdiv float 1.000000e+00, %347, !fpmath !12
(floatB

	full_text


float %347
7fmulB/
-
	full_text 

%349 = fmul float %344, %348
(floatB

	full_text


float %344
(floatB

	full_text


float %348
0addB)
'
	full_text

%350 = add i64 %6, 576
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%351 = getelementptr inbounds float, float* %1, i64 %350
$i64B

	full_text


i64 %350
LloadBD
B
	full_text5
3
1%352 = load float, float* %351, align 4, !tbaa !8
*float*B

	full_text

float* %351
ecallB]
[
	full_textN
L
J%353 = tail call float @_Z4fminff(float %349, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %349
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
\getelementptrBK
I
	full_text<
:
8%355 = getelementptr inbounds float, float* %2, i64 %350
$i64B

	full_text


i64 %350
LstoreBC
A
	full_text4
2
0store float %354, float* %355, align 4, !tbaa !8
(floatB

	full_text


float %354
*float*B

	full_text

float* %355
KloadBC
A
	full_text4
2
0%356 = load float, float* %21, align 4, !tbaa !8
)float*B

	full_text


float* %21
KloadBC
A
	full_text4
2
0%357 = load float, float* %60, align 4, !tbaa !8
)float*B

	full_text


float* %60
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
KloadBC
A
	full_text4
2
0%359 = load float, float* %75, align 4, !tbaa !8
)float*B

	full_text


float* %75
KloadBC
A
	full_text4
2
0%360 = load float, float* %24, align 4, !tbaa !8
)float*B

	full_text


float* %24
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
LfdivBD
B
	full_text5
3
1%362 = fdiv float 1.000000e+00, %361, !fpmath !12
(floatB

	full_text


float %361
7fmulB/
-
	full_text 

%363 = fmul float %358, %362
(floatB

	full_text


float %358
(floatB

	full_text


float %362
0addB)
'
	full_text

%364 = add i64 %6, 584
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%365 = getelementptr inbounds float, float* %1, i64 %364
$i64B

	full_text


i64 %364
LloadBD
B
	full_text5
3
1%366 = load float, float* %365, align 4, !tbaa !8
*float*B

	full_text

float* %365
ecallB]
[
	full_textN
L
J%367 = tail call float @_Z4fminff(float %363, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %363
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
\getelementptrBK
I
	full_text<
:
8%369 = getelementptr inbounds float, float* %2, i64 %364
$i64B

	full_text


i64 %364
LstoreBC
A
	full_text4
2
0store float %368, float* %369, align 4, !tbaa !8
(floatB

	full_text


float %368
*float*B

	full_text

float* %369
KloadBC
A
	full_text4
2
0%370 = load float, float* %14, align 4, !tbaa !8
)float*B

	full_text


float* %14
KloadBC
A
	full_text4
2
0%371 = load float, float* %60, align 4, !tbaa !8
)float*B

	full_text


float* %60
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
KloadBC
A
	full_text4
2
0%373 = load float, float* %90, align 4, !tbaa !8
)float*B

	full_text


float* %90
KloadBC
A
	full_text4
2
0%374 = load float, float* %24, align 4, !tbaa !8
)float*B

	full_text


float* %24
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
LfdivBD
B
	full_text5
3
1%376 = fdiv float 1.000000e+00, %375, !fpmath !12
(floatB

	full_text


float %375
7fmulB/
-
	full_text 

%377 = fmul float %372, %376
(floatB

	full_text


float %372
(floatB

	full_text


float %376
0addB)
'
	full_text

%378 = add i64 %6, 592
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%379 = getelementptr inbounds float, float* %1, i64 %378
$i64B

	full_text


i64 %378
LloadBD
B
	full_text5
3
1%380 = load float, float* %379, align 4, !tbaa !8
*float*B

	full_text

float* %379
ecallB]
[
	full_textN
L
J%381 = tail call float @_Z4fminff(float %377, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %377
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
\getelementptrBK
I
	full_text<
:
8%383 = getelementptr inbounds float, float* %2, i64 %378
$i64B

	full_text


i64 %378
LstoreBC
A
	full_text4
2
0store float %382, float* %383, align 4, !tbaa !8
(floatB

	full_text


float %382
*float*B

	full_text

float* %383
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

	float* %3
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
8float8B+
)
	full_text

float 0x4415AF1D80000000
$i648B

	full_text


i64 64
%i648B

	full_text
	
i64 480
2float8B%
#
	full_text

float 1.013250e+06
%i648B

	full_text
	
i64 448
%i648B

	full_text
	
i64 400
$i648B

	full_text


i64 88
%i648B

	full_text
	
i64 144
%i648B

	full_text
	
i64 456
%i648B

	full_text
	
i64 512
$i648B

	full_text


i64 40
%i648B

	full_text
	
i64 536
#i328B

	full_text	

i32 0
%i648B

	full_text
	
i64 568
%i648B

	full_text
	
i64 528
$i648B

	full_text


i64 80
%i648B

	full_text
	
i64 464
%i648B

	full_text
	
i64 488
%i648B

	full_text
	
i64 544
%i648B

	full_text
	
i64 136
$i648B

	full_text


i64 72
%i648B

	full_text
	
i64 584
%i648B

	full_text
	
i64 440
%i648B

	full_text
	
i64 592
$i648B

	full_text


i64 32
%i648B

	full_text
	
i64 200
%i648B

	full_text
	
i64 432
%i648B

	full_text
	
i64 520
%i648B

	full_text
	
i64 576
%i648B

	full_text
	
i64 408
$i648B

	full_text


i64 48
#i648B

	full_text	

i64 8
%i648B

	full_text
	
i64 128
%i648B

	full_text
	
i64 552
%i648B

	full_text
	
i64 504
%i648B

	full_text
	
i64 104
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
	
i64 560
%i648B

	full_text
	
i64 472
$i648B

	full_text


i64 16
2float8B%
#
	full_text

float 1.000000e+00
%i648B

	full_text
	
i64 424
%i648B

	full_text
	
i64 496
$i648B

	full_text


i64 24
%i648B

	full_text
	
i64 112
%i648B

	full_text
	
i64 416       	  
 

                       !" !! #$ ## %& %% '( '' )* )+ )) ,- ,, ./ .0 .. 12 11 34 33 56 55 78 77 9: 9; 99 <= << >? >@ >> AB AA CD CC EF EG EE HI HH JK JJ LM LL NO NP NN QR QQ ST SS UV UU WX WY WW Z[ Z\ ZZ ]^ ]] _` _a __ bc bb de dd fg ff hi hh jk jl jj mn mm op oq oo rs rr tu tt vw vx vv yz yy {| {{ }~ }} Ä  ÅÇ Å
É ÅÅ Ñ
Ö ÑÑ Üá Ü
à ÜÜ âä ââ ã
å ãã çé çç èê èè ëí ë
ì ëë î
ï îî ñó ñ
ò ññ ôö ôô õú õõ ùû ù
ü ùù †° †† ¢
£ ¢¢ §• §§ ¶ß ¶¶ ®
© ®® ™´ ™™ ¨≠ ¨
Æ ¨¨ Ø
∞ ØØ ±≤ ±
≥ ±± ¥µ ¥¥ ∂
∑ ∂∂ ∏π ∏∏ ∫ª ∫∫ ºΩ º
æ ºº ø
¿ øø ¡¬ ¡
√ ¡¡ ƒ≈ ƒƒ ∆
« ∆∆ »… »»  À    ÃÕ Ã
Œ ÃÃ œ– œœ —“ —— ”‘ ”
’ ”” ÷
◊ ÷÷ ÿŸ ÿ
⁄ ÿÿ €‹ €€ ›
ﬁ ›› ﬂ‡ ﬂﬂ ·‚ ·· „‰ „
Â „„ Ê
Á ÊÊ ËÈ Ë
Í ËË ÎÏ ÎÎ ÌÓ ÌÌ Ô
 ÔÔ ÒÚ ÒÒ ÛÙ Û
ı ÛÛ ˆ˜ ˆ
¯ ˆˆ ˘˙ ˘˘ ˚
¸ ˚˚ ˝˛ ˝˝ ˇ
Ä ˇˇ ÅÇ Å
É ÅÅ ÑÖ ÑÑ Ü
á ÜÜ àâ àà äã ää åç å
é åå è
ê èè ëí ë
ì ëë îï îî ñó ññ òô ò
ö òò õú õõ ùû ùù ü
† üü °¢ °° £§ £
• ££ ¶
ß ¶¶ ®© ®
™ ®® ´¨ ´´ ≠
Æ ≠≠ Ø∞ ØØ ±≤ ±± ≥¥ ≥
µ ≥≥ ∂
∑ ∂∂ ∏π ∏
∫ ∏∏ ªº ªª Ωæ Ω
ø ΩΩ ¿
¡ ¿¿ ¬√ ¬¬ ƒ≈ ƒƒ ∆« ∆
» ∆∆ …
  …… ÀÃ À
Õ ÀÀ Œœ ŒŒ –
— –– “” ““ ‘’ ‘‘ ÷◊ ÷
ÿ ÷÷ Ÿ
⁄ ŸŸ €‹ €
› €€ ﬁﬂ ﬁﬁ ‡
· ‡‡ ‚„ ‚‚ ‰Â ‰‰ Ê
Á ÊÊ ËÈ Ë
Í ËË ÎÏ ÎÎ Ì
Ó ÌÌ Ô ÔÔ ÒÚ ÒÒ ÛÙ Û
ı ÛÛ ˆ
˜ ˆˆ ¯˘ ¯
˙ ¯¯ ˚¸ ˚˚ ˝˛ ˝˝ ˇÄ ˇ
Å ˇˇ ÇÉ ÇÇ ÑÖ ÑÑ Üá Ü
à ÜÜ â
ä ââ ãå ã
ç ãã éè éé ê
ë êê íì íí îï îî ñó ñ
ò ññ ô
ö ôô õú õ
ù õõ ûü ûû †
° †† ¢£ ¢¢ §• §§ ¶ß ¶
® ¶¶ ©™ ©© ´¨ ´´ ≠Æ ≠
Ø ≠≠ ∞
± ∞∞ ≤≥ ≤
¥ ≤≤ µ∂ µµ ∑
∏ ∑∑ π∫ ππ ªº ªª Ωæ Ω
ø ΩΩ ¿
¡ ¿¿ ¬√ ¬
ƒ ¬¬ ≈∆ ≈≈ «» «« …  …
À …… ÃÕ ÃÃ Œœ ŒŒ –— –
“ –– ”
‘ ”” ’÷ ’
◊ ’’ ÿŸ ÿÿ ⁄
€ ⁄⁄ ‹› ‹‹ ﬁﬂ ﬁﬁ ‡· ‡
‚ ‡‡ „
‰ „„ ÂÊ Â
Á ÂÂ ËÈ ËË ÍÎ ÍÍ ÏÌ Ï
Ó ÏÏ Ô ÔÔ ÒÚ ÒÒ ÛÙ Û
ı ÛÛ ˆ
˜ ˆˆ ¯˘ ¯
˙ ¯¯ ˚¸ ˚˚ ˝
˛ ˝˝ ˇÄ ˇˇ ÅÇ ÅÅ ÉÑ É
Ö ÉÉ Ü
á ÜÜ àâ à
ä àà ãå ãã çé çç èê è
ë èè íì íí îï îî ñ
ó ññ òô òò öõ ö
ú öö ù
û ùù ü† ü
° üü ¢£ ¢¢ §
• §§ ¶ß ¶¶ ®© ®® ™´ ™
¨ ™™ ≠
Æ ≠≠ Ø∞ Ø
± ØØ ≤≥ ≤≤ ¥µ ¥¥ ∂∑ ∂
∏ ∂∂ π∫ ππ ªº ªª Ωæ Ω
ø ΩΩ ¿¡ ¿¿ ¬√ ¬
ƒ ¬¬ ≈∆ ≈
« ≈≈ »
… »»  À  
Ã    ÕŒ ÕÕ œ
– œœ —“ —— ”‘ ”” ’÷ ’
◊ ’’ ÿ
Ÿ ÿÿ ⁄€ ⁄
‹ ⁄⁄ ›ﬁ ›› ﬂ‡ ﬂﬂ ·‚ ·
„ ·· ‰Â ‰‰ ÊÁ ÊÊ ËÈ Ë
Í ËË Î
Ï ÎÎ ÌÓ Ì
Ô ÌÌ Ò  Ú
Û ÚÚ Ùı ÙÙ ˆ˜ ˆˆ ¯˘ ¯
˙ ¯¯ ˚
¸ ˚˚ ˝˛ ˝
ˇ ˝˝ ÄÅ ÄÄ ÇÉ ÇÇ Ñ
Ö ÑÑ Üá Ü
à ÜÜ âä ââ ã
å ãã çé çç èê èè ëí ë
ì ëë î
ï îî ñó ñ
ò ññ ôö ôô õú õõ ù
û ùù ü† ü
° üü ¢£ ¢¢ §
• §§ ¶ß ¶¶ ®© ®® ™´ ™
¨ ™™ ≠
Æ ≠≠ Ø∞ Ø
± ØØ ≤≥ ≤≤ ¥µ ¥¥ ∂
∑ ∂∂ ∏π ∏
∫ ∏∏ ªº ªª Ω
æ ΩΩ ø¿ øø ¡¬ ¡¡ √ƒ √
≈ √√ ∆
« ∆∆ »… »
  »» ÀÃ ÀÀ ÕŒ ÕÕ œ– œ
— œœ “” ““ ‘’ ‘‘ ÷◊ ÷
ÿ ÷÷ Ÿ
⁄ ŸŸ €‹ €
› €€ ﬁﬂ ﬁﬁ ‡
· ‡‡ ‚„ ‚‚ ‰Â ‰‰ ÊÁ Ê
Ë ÊÊ È
Í ÈÈ ÎÏ Î
Ì ÎÎ ÓÔ ÓÓ Ò  ÚÛ Ú
Ù ÚÚ ıˆ ı
˜ ıı ¯˘ ¯¯ ˙
˚ ˙˙ ¸˝ ¸¸ ˛
ˇ ˛˛ ÄÅ Ä
Ç ÄÄ ÉÑ ÉÉ Ö
Ü ÖÖ áà áá âä ââ ãå ã
ç ãã é
è éé êë ê
í êê ìî ìì ïñ ïï óò ó
ô óó öõ öö úù úú ûü û
† ûû °
¢ °° £§ £
• ££ ¶ß ¶¶ ®
© ®® ™´ ™™ ¨≠ ¨¨ ÆØ Æ
∞ ÆÆ ±
≤ ±± ≥¥ ≥
µ ≥≥ ∂∑ ∂∂ ∏π ∏∏ ∫ª ∫
º ∫∫ Ωæ ΩΩ ø¿ øø ¡¬ ¡
√ ¡¡ ƒ
≈ ƒƒ ∆« ∆
» ∆∆ …  …… À
Ã ÀÀ ÕŒ ÕÕ œ– œœ —“ —
” —— ‘
’ ‘‘ ÷◊ ÷
ÿ ÷÷ Ÿ⁄ ŸŸ €‹ €€ ›ﬁ ›
ﬂ ›› ‡· ‡‡ ‚„ ‚‚ ‰Â ‰
Ê ‰‰ Á
Ë ÁÁ ÈÍ È
Î ÈÈ ÏÌ ÏÏ Ó
Ô ÓÓ Ò  ÚÛ ÚÚ Ùı Ù
ˆ ÙÙ ˜
¯ ˜˜ ˘˙ ˘
˚ ˘˘ ¸˝ ¸¸ ˛ˇ ˛˛ ÄÅ Ä
Ç ÄÄ ÉÑ ÉÉ ÖÜ ÖÖ áà á
â áá ä
ã ää åç å
é åå èê èè ë
í ëë ìî ìì ïñ ïï óò ó
ô óó ö
õ öö úù ú
û úú ü† 3† d† ã† ∂† ›† Ü† ≠† –† Ì† ê† ∑† ⁄† ˝† §† œ† Ú† ã† §† Ω† ‡† Ö† ®† À† Ó† ë° <° m° î° ø° Ê° è° ∂° Ÿ° ˆ° ô° ¿° „° Ü° ≠° ÿ° ˚° î° ≠° ∆° È° é° ±° ‘° ˜° ö¢ 	£ § § § § %§ J§ S§ }§ ¢§ ®§ ∆§ Ô§ ˚§ ü§ ¿§ ‡§ †§ ñ§ ˙    	 
             " $# &% (! *' +) - /, 0 21 43 6. 85 :7 ;1 =9 ?< @ B DA FC G IH KJ ML OL P RQ TS VN XU Y [W \Z ^E `] a cb ed g_ if kh lb nj pm q s ur wt xJ z |{ ~} Äy Ç ÉÅ Öv áÑ à äâ åã éÜ êç íè ìâ ïë óî ò ö úô ûõ ü °† £¢ • ß¶ ©® ´§ ≠™ Æ¨ ∞ù ≤Ø ≥ µ¥ ∑∂ π± ª∏ Ω∫ æ¥ ¿º ¬ø √ ≈ƒ «∆ … À» Õ  Œ –} “œ ‘— ’” ◊Ã Ÿ÷ ⁄ ‹€ ﬁ› ‡ÿ ‚ﬂ ‰· Â€ Á„ ÈÊ Í Ï ÓÌ Ô ÚÎ ÙÒ ı ˜Û ¯ ˙˘ ¸˚ ˛˝ Äˆ Çˇ É ÖÑ áÜ âÅ ãà çä éÑ êå íè ì® ï óî ôñ öJ ú ûù †ü ¢õ §° •£ ßò ©¶ ™ ¨´ Æ≠ ∞® ≤Ø ¥± µ´ ∑≥ π∂ ∫ ºª æª ø ¡¿ √ü ≈¬ «ƒ »∆  Ω Ã… Õ œŒ —– ”À ’“ ◊‘ ÿŒ ⁄÷ ‹Ÿ › ﬂﬁ ·‡ „ Â‰ Á‚ ÈÊ Í ÏÎ ÓÌ Ë ÚÔ ÙÒ ıÎ ˜Û ˘ˆ ˙J ¸‡ ˛˚ Ä˝ Å¿ É® ÖÇ áÑ àÜ äˇ åâ ç èé ëê ìã ïí óî òé öñ úô ù üû °† £‡ •¢ ß§ ®¿ ™Ô ¨© Æ´ Ø≠ ±¶ ≥∞ ¥ ∂µ ∏∑ ∫≤ ºπ æª øµ ¡Ω √¿ ƒ† ∆‡ »≈  « ÀJ Õ% œÃ —Œ “– ‘… ÷” ◊ Ÿÿ €⁄ ›’ ﬂ‹ ·ﬁ ‚ÿ ‰‡ Ê„ Á È‡ ÎË ÌÍ ÓJ } ÚÔ ÙÒ ıÛ ˜Ï ˘ˆ ˙ ¸˚ ˛˝ Ä¯ Çˇ ÑÅ Ö˚ áÉ âÜ ä¿ å‡ éã êç ëJ ì ïî óñ ôí õò úö ûè †ù ° £¢ •§ ßü ©¶ ´® ¨¢ Æ™ ∞≠ ± ≥‡ µ≤ ∑¥ ∏J ∫ ºπ æª øÔ ¡Ω √¿ ƒ ∆¬ «≈ …∂ À» Ã ŒÕ –œ “  ‘— ÷” ◊Õ Ÿ’ €ÿ ‹ ﬁ‡ ‡› ‚ﬂ „¢ ÂÔ Á‰ ÈÊ ÍË Ï· ÓÎ Ô Ò ÛÚ ıÌ ˜Ù ˘ˆ ˙ ¸¯ ˛˚ ˇ‡ Å ÉÇ ÖÄ áÑ à äâ åã éÜ êç íè ìâ ïë óî ò‡ ö úõ ûô †ù ° £¢ •§ ßü ©¶ ´® ¨¢ Æ™ ∞≠ ±‡ ≥ µ¥ ∑≤ π∂ ∫ ºª æΩ ¿∏ ¬ø ƒ¡ ≈ª «√ …∆  ‡ ÃS ŒÀ –Õ —Ô ”} ’“ ◊‘ ÿ÷ ⁄œ ‹Ÿ › ﬂﬁ ·‡ „€ Â‚ Á‰ Ëﬁ ÍÊ ÏÈ ÌJ Ô} ÒÓ Û Ù ˆÚ ˜ ˘¯ ˚˙ ˝¸ ˇı Å˛ Ç ÑÉ ÜÖ àÄ äá åâ çÉ èã ëé íJ î} ñì òï ô¿ õ% ùö üú †û ¢ó §° • ß¶ ©® ´£ ≠™ Ø¨ ∞¶ ≤Æ ¥± µ† ∑} π∂ ª∏ º æ% ¿Ω ¬ø √¡ ≈∫ «ƒ »  … ÃÀ Œ∆ –Õ “œ ”… ’— ◊‘ ÿ ⁄} ‹Ÿ ﬁ€ ﬂ¢ ·% „‡ Â‚ Ê‰ Ë› ÍÁ Î ÌÏ ÔÓ ÒÈ Û ıÚ ˆÏ ¯Ù ˙˜ ˚ ˝} ˇ¸ Å˛ Ç∆ Ñ% ÜÉ àÖ âá ãÄ çä é êè íë îå ñì òï ôè õó ùö û •• ¶¶ üª ¶¶ ªÅ ¶¶ Å¨ ¶¶ ¨œ ¶¶ œÚ ¶¶ Ú •• Ò ¶¶ Òè ¶¶ èâ ¶¶ â” ¶¶ ”∫ ¶¶ ∫· ¶¶ ·ä ¶¶ äï ¶¶ ï® ¶¶ ®‰ ¶¶ ‰‘ ¶¶ ‘¡ ¶¶ ¡ﬁ ¶¶ ﬁ® ¶¶ ®ˆ ¶¶ ˆè ¶¶ è7 ¶¶ 7î ¶¶ îh ¶¶ h± ¶¶ ±	ß 7	ß h
ß è
ß ∫
ß ·
ß ä
ß ±
ß ‘
ß Ò
ß î
ß ª
ß ﬁ
ß Å
ß ®
ß ”
ß ˆ
ß è
ß ®
ß ¡
ß ‰
ß â
ß ¨
ß œ
ß Ú
ß ï
® ¶
© µ	™ 
´ ´	¨ 1
≠ î
Æ ù
Ø Œ
∞ Õ
± †
≤ ¢≥ 
¥ ¶
µ â
∂ ﬁ
∑ Î
∏ ÿ
π ª
∫ ¯	ª 
º Ï
Ω Ñ
æ è	ø 
¿ ˘
¡ €
¬ 
√ …	ƒ b
≈ ƒ	∆ H	« {
» ﬁ
… ¢
  Ì	À 	Ã #
Õ É
Œ é
œ û– 
– ,– ]– Ñ– Ø– ÷– ˇ– ¶– …– Ê– â– ∞– ”– ˆ– ù– »– Î– Ñ– ù– ∂– Ÿ– ˛– °– ƒ– Á– ä
— ¥
“ ˚	” 	‘ Q
’ â"
ratt4_kernel"
_Z13get_global_idj"
	_Z4fminff*ó
shoc-1.1.5-S3D-ratt4_kernel.clu
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

wgsize_log1p
íÛéA

wgsize
Ä

transfer_bytes
à¢ª
 
transfer_bytes_log1p
íÛéA