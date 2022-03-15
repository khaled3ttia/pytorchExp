
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
.addB'
%
	full_text

%16 = add i64 %6, 56
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
.addB'
%
	full_text

%23 = add i64 %6, 48
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
%29 = add i64 %6, 200
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
/addB(
&
	full_text

%45 = add i64 %6, 208
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
JloadBB
@
	full_text3
1
/%51 = load float, float* %21, align 4, !tbaa !8
)float*B

	full_text


float* %21
JloadBB
@
	full_text3
1
/%52 = load float, float* %17, align 4, !tbaa !8
)float*B

	full_text


float* %17
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
JloadBB
@
	full_text3
1
/%54 = load float, float* %39, align 4, !tbaa !8
)float*B

	full_text


float* %39
JloadBB
@
	full_text3
1
/%55 = load float, float* %24, align 4, !tbaa !8
)float*B

	full_text


float* %24
4fmulB,
*
	full_text

%56 = fmul float %54, %55
'floatB

	full_text

	float %54
'floatB

	full_text

	float %55
JfdivBB
@
	full_text3
1
/%57 = fdiv float 1.000000e+00, %56, !fpmath !12
'floatB

	full_text

	float %56
4fmulB,
*
	full_text

%58 = fmul float %53, %57
'floatB

	full_text

	float %53
'floatB

	full_text

	float %57
/addB(
&
	full_text

%59 = add i64 %6, 216
"i64B

	full_text


i64 %6
ZgetelementptrBI
G
	full_text:
8
6%60 = getelementptr inbounds float, float* %1, i64 %59
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
ccallB[
Y
	full_textL
J
H%62 = tail call float @_Z4fminff(float %58, float 0x4415AF1D80000000) #2
'floatB

	full_text

	float %58
4fmulB,
*
	full_text

%63 = fmul float %61, %62
'floatB

	full_text

	float %61
'floatB

	full_text

	float %62
ZgetelementptrBI
G
	full_text:
8
6%64 = getelementptr inbounds float, float* %2, i64 %59
#i64B

	full_text
	
i64 %59
JstoreBA
?
	full_text2
0
.store float %63, float* %64, align 4, !tbaa !8
'floatB

	full_text

	float %63
)float*B

	full_text


float* %64
JloadBB
@
	full_text3
1
/%65 = load float, float* %14, align 4, !tbaa !8
)float*B

	full_text


float* %14
/addB(
&
	full_text

%66 = add i64 %6, 104
"i64B

	full_text


i64 %6
ZgetelementptrBI
G
	full_text:
8
6%67 = getelementptr inbounds float, float* %3, i64 %66
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
4fmulB,
*
	full_text

%69 = fmul float %65, %68
'floatB

	full_text

	float %65
'floatB

	full_text

	float %68
4fmulB,
*
	full_text

%70 = fmul float %12, %69
'floatB

	full_text

	float %12
'floatB

	full_text

	float %69
/addB(
&
	full_text

%71 = add i64 %6, 112
"i64B

	full_text


i64 %6
ZgetelementptrBI
G
	full_text:
8
6%72 = getelementptr inbounds float, float* %3, i64 %71
#i64B

	full_text
	
i64 %71
JloadBB
@
	full_text3
1
/%73 = load float, float* %72, align 4, !tbaa !8
)float*B

	full_text


float* %72
JfdivBB
@
	full_text3
1
/%74 = fdiv float 1.000000e+00, %73, !fpmath !12
'floatB

	full_text

	float %73
4fmulB,
*
	full_text

%75 = fmul float %70, %74
'floatB

	full_text

	float %70
'floatB

	full_text

	float %74
/addB(
&
	full_text

%76 = add i64 %6, 224
"i64B

	full_text


i64 %6
ZgetelementptrBI
G
	full_text:
8
6%77 = getelementptr inbounds float, float* %1, i64 %76
#i64B

	full_text
	
i64 %76
JloadBB
@
	full_text3
1
/%78 = load float, float* %77, align 4, !tbaa !8
)float*B

	full_text


float* %77
ccallB[
Y
	full_textL
J
H%79 = tail call float @_Z4fminff(float %75, float 0x4415AF1D80000000) #2
'floatB

	full_text

	float %75
4fmulB,
*
	full_text

%80 = fmul float %78, %79
'floatB

	full_text

	float %78
'floatB

	full_text

	float %79
ZgetelementptrBI
G
	full_text:
8
6%81 = getelementptr inbounds float, float* %2, i64 %76
#i64B

	full_text
	
i64 %76
JstoreBA
?
	full_text2
0
.store float %80, float* %81, align 4, !tbaa !8
'floatB

	full_text

	float %80
)float*B

	full_text


float* %81
JloadBB
@
	full_text3
1
/%82 = load float, float* %21, align 4, !tbaa !8
)float*B

	full_text


float* %21
JloadBB
@
	full_text3
1
/%83 = load float, float* %67, align 4, !tbaa !8
)float*B

	full_text


float* %67
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
-addB&
$
	full_text

%85 = add i64 %6, 8
"i64B

	full_text


i64 %6
ZgetelementptrBI
G
	full_text:
8
6%86 = getelementptr inbounds float, float* %3, i64 %85
#i64B

	full_text
	
i64 %85
JloadBB
@
	full_text3
1
/%87 = load float, float* %86, align 4, !tbaa !8
)float*B

	full_text


float* %86
JloadBB
@
	full_text3
1
/%88 = load float, float* %72, align 4, !tbaa !8
)float*B

	full_text


float* %72
4fmulB,
*
	full_text

%89 = fmul float %87, %88
'floatB

	full_text

	float %87
'floatB

	full_text

	float %88
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
%91 = fmul float %84, %90
'floatB

	full_text

	float %84
'floatB

	full_text

	float %90
/addB(
&
	full_text

%92 = add i64 %6, 232
"i64B

	full_text


i64 %6
ZgetelementptrBI
G
	full_text:
8
6%93 = getelementptr inbounds float, float* %1, i64 %92
#i64B

	full_text
	
i64 %92
JloadBB
@
	full_text3
1
/%94 = load float, float* %93, align 4, !tbaa !8
)float*B

	full_text


float* %93
ccallB[
Y
	full_textL
J
H%95 = tail call float @_Z4fminff(float %91, float 0x4415AF1D80000000) #2
'floatB

	full_text

	float %91
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
ZgetelementptrBI
G
	full_text:
8
6%97 = getelementptr inbounds float, float* %2, i64 %92
#i64B

	full_text
	
i64 %92
JstoreBA
?
	full_text2
0
.store float %96, float* %97, align 4, !tbaa !8
'floatB

	full_text

	float %96
)float*B

	full_text


float* %97
YgetelementptrBH
F
	full_text9
7
5%98 = getelementptr inbounds float, float* %3, i64 %6
"i64B

	full_text


i64 %6
JloadBB
@
	full_text3
1
/%99 = load float, float* %98, align 4, !tbaa !8
)float*B

	full_text


float* %98
KloadBC
A
	full_text4
2
0%100 = load float, float* %67, align 4, !tbaa !8
)float*B

	full_text


float* %67
6fmulB.
,
	full_text

%101 = fmul float %99, %100
'floatB

	full_text

	float %99
(floatB

	full_text


float %100
6fmulB.
,
	full_text

%102 = fmul float %12, %101
'floatB

	full_text

	float %12
(floatB

	full_text


float %101
0addB)
'
	full_text

%103 = add i64 %6, 128
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%104 = getelementptr inbounds float, float* %3, i64 %103
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
LfdivBD
B
	full_text5
3
1%106 = fdiv float 1.000000e+00, %105, !fpmath !12
(floatB

	full_text


float %105
7fmulB/
-
	full_text 

%107 = fmul float %102, %106
(floatB

	full_text


float %102
(floatB

	full_text


float %106
0addB)
'
	full_text

%108 = add i64 %6, 240
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%109 = getelementptr inbounds float, float* %1, i64 %108
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
ecallB]
[
	full_textN
L
J%111 = tail call float @_Z4fminff(float %107, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %107
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
\getelementptrBK
I
	full_text<
:
8%113 = getelementptr inbounds float, float* %2, i64 %108
$i64B

	full_text


i64 %108
LstoreBC
A
	full_text4
2
0store float %112, float* %113, align 4, !tbaa !8
(floatB

	full_text


float %112
*float*B

	full_text

float* %113
/addB(
&
	full_text

%114 = add i64 %6, 24
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
KloadBC
A
	full_text4
2
0%117 = load float, float* %67, align 4, !tbaa !8
)float*B

	full_text


float* %67
7fmulB/
-
	full_text 

%118 = fmul float %116, %117
(floatB

	full_text


float %116
(floatB

	full_text


float %117
KloadBC
A
	full_text4
2
0%119 = load float, float* %14, align 4, !tbaa !8
)float*B

	full_text


float* %14
KloadBC
A
	full_text4
2
0%120 = load float, float* %72, align 4, !tbaa !8
)float*B

	full_text


float* %72
7fmulB/
-
	full_text 

%121 = fmul float %119, %120
(floatB

	full_text


float %119
(floatB

	full_text


float %120
LfdivBD
B
	full_text5
3
1%122 = fdiv float 1.000000e+00, %121, !fpmath !12
(floatB

	full_text


float %121
7fmulB/
-
	full_text 

%123 = fmul float %118, %122
(floatB

	full_text


float %118
(floatB

	full_text


float %122
0addB)
'
	full_text

%124 = add i64 %6, 248
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%125 = getelementptr inbounds float, float* %1, i64 %124
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
ecallB]
[
	full_textN
L
J%127 = tail call float @_Z4fminff(float %123, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %123
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
\getelementptrBK
I
	full_text<
:
8%129 = getelementptr inbounds float, float* %2, i64 %124
$i64B

	full_text


i64 %124
LstoreBC
A
	full_text4
2
0store float %128, float* %129, align 4, !tbaa !8
(floatB

	full_text


float %128
*float*B

	full_text

float* %129
KloadBC
A
	full_text4
2
0%130 = load float, float* %24, align 4, !tbaa !8
)float*B

	full_text


float* %24
KloadBC
A
	full_text4
2
0%131 = load float, float* %67, align 4, !tbaa !8
)float*B

	full_text


float* %67
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
KloadBC
A
	full_text4
2
0%133 = load float, float* %21, align 4, !tbaa !8
)float*B

	full_text


float* %21
KloadBC
A
	full_text4
2
0%134 = load float, float* %72, align 4, !tbaa !8
)float*B

	full_text


float* %72
7fmulB/
-
	full_text 

%135 = fmul float %133, %134
(floatB

	full_text


float %133
(floatB

	full_text


float %134
LfdivBD
B
	full_text5
3
1%136 = fdiv float 1.000000e+00, %135, !fpmath !12
(floatB

	full_text


float %135
7fmulB/
-
	full_text 

%137 = fmul float %132, %136
(floatB

	full_text


float %132
(floatB

	full_text


float %136
0addB)
'
	full_text

%138 = add i64 %6, 256
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%139 = getelementptr inbounds float, float* %1, i64 %138
$i64B

	full_text


i64 %138
LloadBD
B
	full_text5
3
1%140 = load float, float* %139, align 4, !tbaa !8
*float*B

	full_text

float* %139
ecallB]
[
	full_textN
L
J%141 = tail call float @_Z4fminff(float %137, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %137
7fmulB/
-
	full_text 

%142 = fmul float %140, %141
(floatB

	full_text


float %140
(floatB

	full_text


float %141
\getelementptrBK
I
	full_text<
:
8%143 = getelementptr inbounds float, float* %2, i64 %138
$i64B

	full_text


i64 %138
LstoreBC
A
	full_text4
2
0store float %142, float* %143, align 4, !tbaa !8
(floatB

	full_text


float %142
*float*B

	full_text

float* %143
KloadBC
A
	full_text4
2
0%144 = load float, float* %14, align 4, !tbaa !8
)float*B

	full_text


float* %14
/addB(
&
	full_text

%145 = add i64 %6, 64
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%146 = getelementptr inbounds float, float* %3, i64 %145
$i64B

	full_text


i64 %145
LloadBD
B
	full_text5
3
1%147 = load float, float* %146, align 4, !tbaa !8
*float*B

	full_text

float* %146
7fmulB/
-
	full_text 

%148 = fmul float %144, %147
(floatB

	full_text


float %144
(floatB

	full_text


float %147
KloadBC
A
	full_text4
2
0%149 = load float, float* %86, align 4, !tbaa !8
)float*B

	full_text


float* %86
KloadBC
A
	full_text4
2
0%150 = load float, float* %67, align 4, !tbaa !8
)float*B

	full_text


float* %67
7fmulB/
-
	full_text 

%151 = fmul float %149, %150
(floatB

	full_text


float %149
(floatB

	full_text


float %150
LfdivBD
B
	full_text5
3
1%152 = fdiv float 1.000000e+00, %151, !fpmath !12
(floatB

	full_text


float %151
7fmulB/
-
	full_text 

%153 = fmul float %148, %152
(floatB

	full_text


float %148
(floatB

	full_text


float %152
0addB)
'
	full_text

%154 = add i64 %6, 264
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%155 = getelementptr inbounds float, float* %1, i64 %154
$i64B

	full_text


i64 %154
LloadBD
B
	full_text5
3
1%156 = load float, float* %155, align 4, !tbaa !8
*float*B

	full_text

float* %155
ecallB]
[
	full_textN
L
J%157 = tail call float @_Z4fminff(float %153, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %153
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
\getelementptrBK
I
	full_text<
:
8%159 = getelementptr inbounds float, float* %2, i64 %154
$i64B

	full_text


i64 %154
LstoreBC
A
	full_text4
2
0store float %158, float* %159, align 4, !tbaa !8
(floatB

	full_text


float %158
*float*B

	full_text

float* %159
KloadBC
A
	full_text4
2
0%160 = load float, float* %21, align 4, !tbaa !8
)float*B

	full_text


float* %21
LloadBD
B
	full_text5
3
1%161 = load float, float* %146, align 4, !tbaa !8
*float*B

	full_text

float* %146
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
KloadBC
A
	full_text4
2
0%163 = load float, float* %86, align 4, !tbaa !8
)float*B

	full_text


float* %86
0addB)
'
	full_text

%164 = add i64 %6, 120
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%165 = getelementptr inbounds float, float* %3, i64 %164
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
7fmulB/
-
	full_text 

%167 = fmul float %163, %166
(floatB

	full_text


float %163
(floatB

	full_text


float %166
LfdivBD
B
	full_text5
3
1%168 = fdiv float 1.000000e+00, %167, !fpmath !12
(floatB

	full_text


float %167
7fmulB/
-
	full_text 

%169 = fmul float %162, %168
(floatB

	full_text


float %162
(floatB

	full_text


float %168
0addB)
'
	full_text

%170 = add i64 %6, 272
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%171 = getelementptr inbounds float, float* %1, i64 %170
$i64B

	full_text


i64 %170
LloadBD
B
	full_text5
3
1%172 = load float, float* %171, align 4, !tbaa !8
*float*B

	full_text

float* %171
ecallB]
[
	full_textN
L
J%173 = tail call float @_Z4fminff(float %169, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %169
7fmulB/
-
	full_text 

%174 = fmul float %172, %173
(floatB

	full_text


float %172
(floatB

	full_text


float %173
\getelementptrBK
I
	full_text<
:
8%175 = getelementptr inbounds float, float* %2, i64 %170
$i64B

	full_text


i64 %170
LstoreBC
A
	full_text4
2
0store float %174, float* %175, align 4, !tbaa !8
(floatB

	full_text


float %174
*float*B

	full_text

float* %175
KloadBC
A
	full_text4
2
0%176 = load float, float* %98, align 4, !tbaa !8
)float*B

	full_text


float* %98
LloadBD
B
	full_text5
3
1%177 = load float, float* %146, align 4, !tbaa !8
*float*B

	full_text

float* %146
7fmulB/
-
	full_text 

%178 = fmul float %176, %177
(floatB

	full_text


float %176
(floatB

	full_text


float %177
KloadBC
A
	full_text4
2
0%179 = load float, float* %86, align 4, !tbaa !8
)float*B

	full_text


float* %86
/addB(
&
	full_text

%180 = add i64 %6, 72
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%181 = getelementptr inbounds float, float* %3, i64 %180
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
7fmulB/
-
	full_text 

%183 = fmul float %179, %182
(floatB

	full_text


float %179
(floatB

	full_text


float %182
LfdivBD
B
	full_text5
3
1%184 = fdiv float 1.000000e+00, %183, !fpmath !12
(floatB

	full_text


float %183
7fmulB/
-
	full_text 

%185 = fmul float %178, %184
(floatB

	full_text


float %178
(floatB

	full_text


float %184
0addB)
'
	full_text

%186 = add i64 %6, 280
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%187 = getelementptr inbounds float, float* %1, i64 %186
$i64B

	full_text


i64 %186
LloadBD
B
	full_text5
3
1%188 = load float, float* %187, align 4, !tbaa !8
*float*B

	full_text

float* %187
ecallB]
[
	full_textN
L
J%189 = tail call float @_Z4fminff(float %185, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %185
7fmulB/
-
	full_text 

%190 = fmul float %188, %189
(floatB

	full_text


float %188
(floatB

	full_text


float %189
\getelementptrBK
I
	full_text<
:
8%191 = getelementptr inbounds float, float* %2, i64 %186
$i64B

	full_text


i64 %186
LstoreBC
A
	full_text4
2
0store float %190, float* %191, align 4, !tbaa !8
(floatB

	full_text


float %190
*float*B

	full_text

float* %191
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
1%193 = load float, float* %146, align 4, !tbaa !8
*float*B

	full_text

float* %146
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
KloadBC
A
	full_text4
2
0%195 = load float, float* %86, align 4, !tbaa !8
)float*B

	full_text


float* %86
LloadBD
B
	full_text5
3
1%196 = load float, float* %104, align 4, !tbaa !8
*float*B

	full_text

float* %104
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
LfdivBD
B
	full_text5
3
1%198 = fdiv float 1.000000e+00, %197, !fpmath !12
(floatB

	full_text


float %197
7fmulB/
-
	full_text 

%199 = fmul float %194, %198
(floatB

	full_text


float %194
(floatB

	full_text


float %198
0addB)
'
	full_text

%200 = add i64 %6, 288
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%201 = getelementptr inbounds float, float* %1, i64 %200
$i64B

	full_text


i64 %200
LloadBD
B
	full_text5
3
1%202 = load float, float* %201, align 4, !tbaa !8
*float*B

	full_text

float* %201
ecallB]
[
	full_textN
L
J%203 = tail call float @_Z4fminff(float %199, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %199
7fmulB/
-
	full_text 

%204 = fmul float %202, %203
(floatB

	full_text


float %202
(floatB

	full_text


float %203
\getelementptrBK
I
	full_text<
:
8%205 = getelementptr inbounds float, float* %2, i64 %200
$i64B

	full_text


i64 %200
LstoreBC
A
	full_text4
2
0store float %204, float* %205, align 4, !tbaa !8
(floatB

	full_text


float %204
*float*B

	full_text

float* %205
LloadBD
B
	full_text5
3
1%206 = load float, float* %115, align 4, !tbaa !8
*float*B

	full_text

float* %115
LloadBD
B
	full_text5
3
1%207 = load float, float* %146, align 4, !tbaa !8
*float*B

	full_text

float* %146
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
KloadBC
A
	full_text4
2
0%209 = load float, float* %14, align 4, !tbaa !8
)float*B

	full_text


float* %14
LloadBD
B
	full_text5
3
1%210 = load float, float* %165, align 4, !tbaa !8
*float*B

	full_text

float* %165
7fmulB/
-
	full_text 

%211 = fmul float %209, %210
(floatB

	full_text


float %209
(floatB

	full_text


float %210
LfdivBD
B
	full_text5
3
1%212 = fdiv float 1.000000e+00, %211, !fpmath !12
(floatB

	full_text


float %211
7fmulB/
-
	full_text 

%213 = fmul float %208, %212
(floatB

	full_text


float %208
(floatB

	full_text


float %212
0addB)
'
	full_text

%214 = add i64 %6, 296
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%215 = getelementptr inbounds float, float* %1, i64 %214
$i64B

	full_text


i64 %214
LloadBD
B
	full_text5
3
1%216 = load float, float* %215, align 4, !tbaa !8
*float*B

	full_text

float* %215
ecallB]
[
	full_textN
L
J%217 = tail call float @_Z4fminff(float %213, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %213
7fmulB/
-
	full_text 

%218 = fmul float %216, %217
(floatB

	full_text


float %216
(floatB

	full_text


float %217
\getelementptrBK
I
	full_text<
:
8%219 = getelementptr inbounds float, float* %2, i64 %214
$i64B

	full_text


i64 %214
LstoreBC
A
	full_text4
2
0store float %218, float* %219, align 4, !tbaa !8
(floatB

	full_text


float %218
*float*B

	full_text

float* %219
LloadBD
B
	full_text5
3
1%220 = load float, float* %146, align 4, !tbaa !8
*float*B

	full_text

float* %146
KloadBC
A
	full_text4
2
0%221 = load float, float* %67, align 4, !tbaa !8
)float*B

	full_text


float* %67
7fmulB/
-
	full_text 

%222 = fmul float %220, %221
(floatB

	full_text


float %220
(floatB

	full_text


float %221
6fmulB.
,
	full_text

%223 = fmul float %12, %222
'floatB

	full_text

	float %12
(floatB

	full_text


float %222
0addB)
'
	full_text

%224 = add i64 %6, 192
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%225 = getelementptr inbounds float, float* %3, i64 %224
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
LfdivBD
B
	full_text5
3
1%227 = fdiv float 1.000000e+00, %226, !fpmath !12
(floatB

	full_text


float %226
7fmulB/
-
	full_text 

%228 = fmul float %223, %227
(floatB

	full_text


float %223
(floatB

	full_text


float %227
0addB)
'
	full_text

%229 = add i64 %6, 304
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%230 = getelementptr inbounds float, float* %1, i64 %229
$i64B

	full_text


i64 %229
LloadBD
B
	full_text5
3
1%231 = load float, float* %230, align 4, !tbaa !8
*float*B

	full_text

float* %230
ecallB]
[
	full_textN
L
J%232 = tail call float @_Z4fminff(float %228, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %228
7fmulB/
-
	full_text 

%233 = fmul float %231, %232
(floatB

	full_text


float %231
(floatB

	full_text


float %232
\getelementptrBK
I
	full_text<
:
8%234 = getelementptr inbounds float, float* %2, i64 %229
$i64B

	full_text


i64 %229
LstoreBC
A
	full_text4
2
0store float %233, float* %234, align 4, !tbaa !8
(floatB

	full_text


float %233
*float*B

	full_text

float* %234
LloadBD
B
	full_text5
3
1%235 = load float, float* %146, align 4, !tbaa !8
*float*B

	full_text

float* %146
KloadBC
A
	full_text4
2
0%236 = load float, float* %72, align 4, !tbaa !8
)float*B

	full_text


float* %72
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
KloadBC
A
	full_text4
2
0%238 = load float, float* %67, align 4, !tbaa !8
)float*B

	full_text


float* %67
LloadBD
B
	full_text5
3
1%239 = load float, float* %165, align 4, !tbaa !8
*float*B

	full_text

float* %165
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
LfdivBD
B
	full_text5
3
1%241 = fdiv float 1.000000e+00, %240, !fpmath !12
(floatB

	full_text


float %240
7fmulB/
-
	full_text 

%242 = fmul float %237, %241
(floatB

	full_text


float %237
(floatB

	full_text


float %241
0addB)
'
	full_text

%243 = add i64 %6, 312
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%244 = getelementptr inbounds float, float* %1, i64 %243
$i64B

	full_text


i64 %243
LloadBD
B
	full_text5
3
1%245 = load float, float* %244, align 4, !tbaa !8
*float*B

	full_text

float* %244
ecallB]
[
	full_textN
L
J%246 = tail call float @_Z4fminff(float %242, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %242
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
\getelementptrBK
I
	full_text<
:
8%248 = getelementptr inbounds float, float* %2, i64 %243
$i64B

	full_text


i64 %243
LstoreBC
A
	full_text4
2
0store float %247, float* %248, align 4, !tbaa !8
(floatB

	full_text


float %247
*float*B

	full_text

float* %248
KloadBC
A
	full_text4
2
0%249 = load float, float* %86, align 4, !tbaa !8
)float*B

	full_text


float* %86
LloadBD
B
	full_text5
3
1%250 = load float, float* %165, align 4, !tbaa !8
*float*B

	full_text

float* %165
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
6fmulB.
,
	full_text

%252 = fmul float %12, %251
'floatB

	full_text

	float %12
(floatB

	full_text


float %251
LloadBD
B
	full_text5
3
1%253 = load float, float* %104, align 4, !tbaa !8
*float*B

	full_text

float* %104
LfdivBD
B
	full_text5
3
1%254 = fdiv float 1.000000e+00, %253, !fpmath !12
(floatB

	full_text


float %253
7fmulB/
-
	full_text 

%255 = fmul float %252, %254
(floatB

	full_text


float %252
(floatB

	full_text


float %254
0addB)
'
	full_text

%256 = add i64 %6, 320
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%257 = getelementptr inbounds float, float* %1, i64 %256
$i64B

	full_text


i64 %256
LloadBD
B
	full_text5
3
1%258 = load float, float* %257, align 4, !tbaa !8
*float*B

	full_text

float* %257
ecallB]
[
	full_textN
L
J%259 = tail call float @_Z4fminff(float %255, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %255
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
\getelementptrBK
I
	full_text<
:
8%261 = getelementptr inbounds float, float* %2, i64 %256
$i64B

	full_text


i64 %256
LstoreBC
A
	full_text4
2
0store float %260, float* %261, align 4, !tbaa !8
(floatB

	full_text


float %260
*float*B

	full_text

float* %261
KloadBC
A
	full_text4
2
0%262 = load float, float* %86, align 4, !tbaa !8
)float*B

	full_text


float* %86
LloadBD
B
	full_text5
3
1%263 = load float, float* %165, align 4, !tbaa !8
*float*B

	full_text

float* %165
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
KloadBC
A
	full_text4
2
0%265 = load float, float* %98, align 4, !tbaa !8
)float*B

	full_text


float* %98
KloadBC
A
	full_text4
2
0%266 = load float, float* %67, align 4, !tbaa !8
)float*B

	full_text


float* %67
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
LfdivBD
B
	full_text5
3
1%268 = fdiv float 1.000000e+00, %267, !fpmath !12
(floatB

	full_text


float %267
7fmulB/
-
	full_text 

%269 = fmul float %264, %268
(floatB

	full_text


float %264
(floatB

	full_text


float %268
0addB)
'
	full_text

%270 = add i64 %6, 328
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%271 = getelementptr inbounds float, float* %1, i64 %270
$i64B

	full_text


i64 %270
LloadBD
B
	full_text5
3
1%272 = load float, float* %271, align 4, !tbaa !8
*float*B

	full_text

float* %271
ecallB]
[
	full_textN
L
J%273 = tail call float @_Z4fminff(float %269, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %269
7fmulB/
-
	full_text 

%274 = fmul float %272, %273
(floatB

	full_text


float %272
(floatB

	full_text


float %273
\getelementptrBK
I
	full_text<
:
8%275 = getelementptr inbounds float, float* %2, i64 %270
$i64B

	full_text


i64 %270
LstoreBC
A
	full_text4
2
0store float %274, float* %275, align 4, !tbaa !8
(floatB

	full_text


float %274
*float*B

	full_text

float* %275
KloadBC
A
	full_text4
2
0%276 = load float, float* %14, align 4, !tbaa !8
)float*B

	full_text


float* %14
LloadBD
B
	full_text5
3
1%277 = load float, float* %165, align 4, !tbaa !8
*float*B

	full_text

float* %165
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
KloadBC
A
	full_text4
2
0%279 = load float, float* %21, align 4, !tbaa !8
)float*B

	full_text


float* %21
KloadBC
A
	full_text4
2
0%280 = load float, float* %67, align 4, !tbaa !8
)float*B

	full_text


float* %67
7fmulB/
-
	full_text 

%281 = fmul float %279, %280
(floatB

	full_text


float %279
(floatB

	full_text


float %280
LfdivBD
B
	full_text5
3
1%282 = fdiv float 1.000000e+00, %281, !fpmath !12
(floatB

	full_text


float %281
7fmulB/
-
	full_text 

%283 = fmul float %278, %282
(floatB

	full_text


float %278
(floatB

	full_text


float %282
0addB)
'
	full_text

%284 = add i64 %6, 336
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%285 = getelementptr inbounds float, float* %1, i64 %284
$i64B

	full_text


i64 %284
LloadBD
B
	full_text5
3
1%286 = load float, float* %285, align 4, !tbaa !8
*float*B

	full_text

float* %285
ecallB]
[
	full_textN
L
J%287 = tail call float @_Z4fminff(float %283, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %283
7fmulB/
-
	full_text 

%288 = fmul float %286, %287
(floatB

	full_text


float %286
(floatB

	full_text


float %287
\getelementptrBK
I
	full_text<
:
8%289 = getelementptr inbounds float, float* %2, i64 %284
$i64B

	full_text


i64 %284
LstoreBC
A
	full_text4
2
0store float %288, float* %289, align 4, !tbaa !8
(floatB

	full_text


float %288
*float*B

	full_text

float* %289
KloadBC
A
	full_text4
2
0%290 = load float, float* %14, align 4, !tbaa !8
)float*B

	full_text


float* %14
LloadBD
B
	full_text5
3
1%291 = load float, float* %165, align 4, !tbaa !8
*float*B

	full_text

float* %165
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
KloadBC
A
	full_text4
2
0%293 = load float, float* %86, align 4, !tbaa !8
)float*B

	full_text


float* %86
KloadBC
A
	full_text4
2
0%294 = load float, float* %72, align 4, !tbaa !8
)float*B

	full_text


float* %72
7fmulB/
-
	full_text 

%295 = fmul float %293, %294
(floatB

	full_text


float %293
(floatB

	full_text


float %294
LfdivBD
B
	full_text5
3
1%296 = fdiv float 1.000000e+00, %295, !fpmath !12
(floatB

	full_text


float %295
7fmulB/
-
	full_text 

%297 = fmul float %292, %296
(floatB

	full_text


float %292
(floatB

	full_text


float %296
0addB)
'
	full_text

%298 = add i64 %6, 344
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%299 = getelementptr inbounds float, float* %1, i64 %298
$i64B

	full_text


i64 %298
LloadBD
B
	full_text5
3
1%300 = load float, float* %299, align 4, !tbaa !8
*float*B

	full_text

float* %299
ecallB]
[
	full_textN
L
J%301 = tail call float @_Z4fminff(float %297, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %297
7fmulB/
-
	full_text 

%302 = fmul float %300, %301
(floatB

	full_text


float %300
(floatB

	full_text


float %301
\getelementptrBK
I
	full_text<
:
8%303 = getelementptr inbounds float, float* %2, i64 %298
$i64B

	full_text


i64 %298
LstoreBC
A
	full_text4
2
0store float %302, float* %303, align 4, !tbaa !8
(floatB

	full_text


float %302
*float*B

	full_text

float* %303
KloadBC
A
	full_text4
2
0%304 = load float, float* %21, align 4, !tbaa !8
)float*B

	full_text


float* %21
LloadBD
B
	full_text5
3
1%305 = load float, float* %165, align 4, !tbaa !8
*float*B

	full_text

float* %165
7fmulB/
-
	full_text 

%306 = fmul float %304, %305
(floatB

	full_text


float %304
(floatB

	full_text


float %305
KloadBC
A
	full_text4
2
0%307 = load float, float* %39, align 4, !tbaa !8
)float*B

	full_text


float* %39
KloadBC
A
	full_text4
2
0%308 = load float, float* %67, align 4, !tbaa !8
)float*B

	full_text


float* %67
7fmulB/
-
	full_text 

%309 = fmul float %307, %308
(floatB

	full_text


float %307
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

%311 = fmul float %306, %310
(floatB

	full_text


float %306
(floatB

	full_text


float %310
0addB)
'
	full_text

%312 = add i64 %6, 352
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
LloadBD
B
	full_text5
3
1%318 = load float, float* %165, align 4, !tbaa !8
*float*B

	full_text

float* %165
6fmulB.
,
	full_text

%319 = fmul float %12, %318
'floatB

	full_text

	float %12
(floatB

	full_text


float %318
KloadBC
A
	full_text4
2
0%320 = load float, float* %86, align 4, !tbaa !8
)float*B

	full_text


float* %86
KloadBC
A
	full_text4
2
0%321 = load float, float* %67, align 4, !tbaa !8
)float*B

	full_text


float* %67
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
LfdivBD
B
	full_text5
3
1%323 = fdiv float 1.000000e+00, %322, !fpmath !12
(floatB

	full_text


float %322
7fmulB/
-
	full_text 

%324 = fmul float %319, %323
(floatB

	full_text


float %319
(floatB

	full_text


float %323
0addB)
'
	full_text

%325 = add i64 %6, 360
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%326 = getelementptr inbounds float, float* %1, i64 %325
$i64B

	full_text


i64 %325
LloadBD
B
	full_text5
3
1%327 = load float, float* %326, align 4, !tbaa !8
*float*B

	full_text

float* %326
ecallB]
[
	full_textN
L
J%328 = tail call float @_Z4fminff(float %324, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %324
7fmulB/
-
	full_text 

%329 = fmul float %327, %328
(floatB

	full_text


float %327
(floatB

	full_text


float %328
\getelementptrBK
I
	full_text<
:
8%330 = getelementptr inbounds float, float* %2, i64 %325
$i64B

	full_text


i64 %325
LstoreBC
A
	full_text4
2
0store float %329, float* %330, align 4, !tbaa !8
(floatB

	full_text


float %329
*float*B

	full_text

float* %330
LloadBD
B
	full_text5
3
1%331 = load float, float* %115, align 4, !tbaa !8
*float*B

	full_text

float* %115
LloadBD
B
	full_text5
3
1%332 = load float, float* %165, align 4, !tbaa !8
*float*B

	full_text

float* %165
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
KloadBC
A
	full_text4
2
0%334 = load float, float* %24, align 4, !tbaa !8
)float*B

	full_text


float* %24
KloadBC
A
	full_text4
2
0%335 = load float, float* %67, align 4, !tbaa !8
)float*B

	full_text


float* %67
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
LfdivBD
B
	full_text5
3
1%337 = fdiv float 1.000000e+00, %336, !fpmath !12
(floatB

	full_text


float %336
7fmulB/
-
	full_text 

%338 = fmul float %333, %337
(floatB

	full_text


float %333
(floatB

	full_text


float %337
0addB)
'
	full_text

%339 = add i64 %6, 368
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%340 = getelementptr inbounds float, float* %1, i64 %339
$i64B

	full_text


i64 %339
LloadBD
B
	full_text5
3
1%341 = load float, float* %340, align 4, !tbaa !8
*float*B

	full_text

float* %340
ecallB]
[
	full_textN
L
J%342 = tail call float @_Z4fminff(float %338, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %338
7fmulB/
-
	full_text 

%343 = fmul float %341, %342
(floatB

	full_text


float %341
(floatB

	full_text


float %342
\getelementptrBK
I
	full_text<
:
8%344 = getelementptr inbounds float, float* %2, i64 %339
$i64B

	full_text


i64 %339
LstoreBC
A
	full_text4
2
0store float %343, float* %344, align 4, !tbaa !8
(floatB

	full_text


float %343
*float*B

	full_text

float* %344
KloadBC
A
	full_text4
2
0%345 = load float, float* %86, align 4, !tbaa !8
)float*B

	full_text


float* %86
LloadBD
B
	full_text5
3
1%346 = load float, float* %181, align 4, !tbaa !8
*float*B

	full_text

float* %181
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
6fmulB.
,
	full_text

%348 = fmul float %12, %347
'floatB

	full_text

	float %12
(floatB

	full_text


float %347
/addB(
&
	full_text

%349 = add i64 %6, 88
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%350 = getelementptr inbounds float, float* %3, i64 %349
$i64B

	full_text


i64 %349
LloadBD
B
	full_text5
3
1%351 = load float, float* %350, align 4, !tbaa !8
*float*B

	full_text

float* %350
LfdivBD
B
	full_text5
3
1%352 = fdiv float 1.000000e+00, %351, !fpmath !12
(floatB

	full_text


float %351
7fmulB/
-
	full_text 

%353 = fmul float %348, %352
(floatB

	full_text


float %348
(floatB

	full_text


float %352
0addB)
'
	full_text

%354 = add i64 %6, 376
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%355 = getelementptr inbounds float, float* %1, i64 %354
$i64B

	full_text


i64 %354
LloadBD
B
	full_text5
3
1%356 = load float, float* %355, align 4, !tbaa !8
*float*B

	full_text

float* %355
ecallB]
[
	full_textN
L
J%357 = tail call float @_Z4fminff(float %353, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %353
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
\getelementptrBK
I
	full_text<
:
8%359 = getelementptr inbounds float, float* %2, i64 %354
$i64B

	full_text


i64 %354
LstoreBC
A
	full_text4
2
0store float %358, float* %359, align 4, !tbaa !8
(floatB

	full_text


float %358
*float*B

	full_text

float* %359
KloadBC
A
	full_text4
2
0%360 = load float, float* %98, align 4, !tbaa !8
)float*B

	full_text


float* %98
LloadBD
B
	full_text5
3
1%361 = load float, float* %181, align 4, !tbaa !8
*float*B

	full_text

float* %181
7fmulB/
-
	full_text 

%362 = fmul float %360, %361
(floatB

	full_text


float %360
(floatB

	full_text


float %361
KloadBC
A
	full_text4
2
0%363 = load float, float* %86, align 4, !tbaa !8
)float*B

	full_text


float* %86
LloadBD
B
	full_text5
3
1%364 = load float, float* %350, align 4, !tbaa !8
*float*B

	full_text

float* %350
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
LfdivBD
B
	full_text5
3
1%366 = fdiv float 1.000000e+00, %365, !fpmath !12
(floatB

	full_text


float %365
7fmulB/
-
	full_text 

%367 = fmul float %362, %366
(floatB

	full_text


float %362
(floatB

	full_text


float %366
0addB)
'
	full_text

%368 = add i64 %6, 384
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%369 = getelementptr inbounds float, float* %1, i64 %368
$i64B

	full_text


i64 %368
LloadBD
B
	full_text5
3
1%370 = load float, float* %369, align 4, !tbaa !8
*float*B

	full_text

float* %369
ecallB]
[
	full_textN
L
J%371 = tail call float @_Z4fminff(float %367, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %367
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
\getelementptrBK
I
	full_text<
:
8%373 = getelementptr inbounds float, float* %2, i64 %368
$i64B

	full_text


i64 %368
LstoreBC
A
	full_text4
2
0store float %372, float* %373, align 4, !tbaa !8
(floatB

	full_text


float %372
*float*B

	full_text

float* %373
KloadBC
A
	full_text4
2
0%374 = load float, float* %14, align 4, !tbaa !8
)float*B

	full_text


float* %14
LloadBD
B
	full_text5
3
1%375 = load float, float* %181, align 4, !tbaa !8
*float*B

	full_text

float* %181
7fmulB/
-
	full_text 

%376 = fmul float %374, %375
(floatB

	full_text


float %374
(floatB

	full_text


float %375
KloadBC
A
	full_text4
2
0%377 = load float, float* %86, align 4, !tbaa !8
)float*B

	full_text


float* %86
LloadBD
B
	full_text5
3
1%378 = load float, float* %165, align 4, !tbaa !8
*float*B

	full_text

float* %165
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
LfdivBD
B
	full_text5
3
1%380 = fdiv float 1.000000e+00, %379, !fpmath !12
(floatB

	full_text


float %379
7fmulB/
-
	full_text 

%381 = fmul float %376, %380
(floatB

	full_text


float %376
(floatB

	full_text


float %380
0addB)
'
	full_text

%382 = add i64 %6, 392
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%383 = getelementptr inbounds float, float* %1, i64 %382
$i64B

	full_text


i64 %382
LloadBD
B
	full_text5
3
1%384 = load float, float* %383, align 4, !tbaa !8
*float*B

	full_text

float* %383
ecallB]
[
	full_textN
L
J%385 = tail call float @_Z4fminff(float %381, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %381
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
\getelementptrBK
I
	full_text<
:
8%387 = getelementptr inbounds float, float* %2, i64 %382
$i64B

	full_text


i64 %382
LstoreBC
A
	full_text4
2
0store float %386, float* %387, align 4, !tbaa !8
(floatB

	full_text


float %386
*float*B

	full_text

float* %387
"retB

	full_text


ret void
*float*8B

	full_text

	float* %3
(float8B

	full_text


float %4
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
	
i64 216
2float8B%
#
	full_text

float 1.013250e+06
$i648B

	full_text


i64 56
%i648B

	full_text
	
i64 104
$i648B

	full_text


i64 64
#i648B

	full_text	

i64 8
%i648B

	full_text
	
i64 200
%i648B

	full_text
	
i64 264
$i648B

	full_text


i64 72
$i648B

	full_text


i64 40
%i648B

	full_text
	
i64 240
%i648B

	full_text
	
i64 112
%i648B

	full_text
	
i64 128
%i648B

	full_text
	
i64 256
#i328B

	full_text	

i32 0
$i648B

	full_text


i64 48
%i648B

	full_text
	
i64 328
%i648B

	full_text
	
i64 344
8float8B+
)
	full_text

float 0x4193D2C640000000
%i648B

	full_text
	
i64 192
%i648B

	full_text
	
i64 296
%i648B

	full_text
	
i64 288
%i648B

	full_text
	
i64 376
%i648B

	full_text
	
i64 384
%i648B

	full_text
	
i64 336
$i648B

	full_text


i64 32
%i648B

	full_text
	
i64 248
%i648B

	full_text
	
i64 320
%i648B

	full_text
	
i64 232
$i648B

	full_text


i64 88
2float8B%
#
	full_text

float 1.000000e+00
%i648B

	full_text
	
i64 392
%i648B

	full_text
	
i64 224
$i648B

	full_text


i64 16
%i648B

	full_text
	
i64 360
%i648B

	full_text
	
i64 352
%i648B

	full_text
	
i64 368
%i648B

	full_text
	
i64 312
%i648B

	full_text
	
i64 280
%i648B

	full_text
	
i64 304
8float8B+
)
	full_text

float 0x4415AF1D80000000
%i648B

	full_text
	
i64 272
$i648B

	full_text


i64 24
%i648B

	full_text
	
i64 120
%i648B

	full_text
	
i64 208       	  
 

                       !" !! #$ ## %& %% '( '' )* )+ )) ,- ,, ./ .0 .. 12 11 34 33 56 55 78 77 9: 9; 99 <= << >? >@ >> AB AA CD CC EF EG EE HI HH JK JJ LM LL NO NN PQ PR PP ST SS UV UW UU XY XX Z[ ZZ \] \\ ^_ ^^ `a `b `` cd cc ef eg ee hi hh jk jj lm ln ll op oo qr qq st su ss vw vv xy xz xx {| {{ }~ }} Ä  ÅÇ ÅÅ ÉÑ É
Ö ÉÉ Ü
á ÜÜ àâ à
ä àà ãå ãã çé çç è
ê èè ëí ëë ìî ì
ï ìì ñó ñ
ò ññ ôö ôô õ
ú õõ ùû ùù ü
† üü °¢ °
£ °° §• §§ ¶
ß ¶¶ ®© ®® ™´ ™™ ¨≠ ¨
Æ ¨¨ Ø
∞ ØØ ±≤ ±
≥ ±± ¥µ ¥¥ ∂∑ ∂∂ ∏π ∏
∫ ∏∏ ªº ªª Ω
æ ΩΩ ø¿ øø ¡¬ ¡¡ √ƒ √
≈ √√ ∆
« ∆∆ »… »
  »» ÀÃ ÀÀ Õ
Œ ÕÕ œ– œœ —“ —— ”‘ ”
’ ”” ÷
◊ ÷÷ ÿŸ ÿ
⁄ ÿÿ €
‹ €€ ›ﬁ ›› ﬂ‡ ﬂﬂ ·‚ ·
„ ·· ‰Â ‰
Ê ‰‰ ÁË ÁÁ È
Í ÈÈ ÎÏ ÎÎ Ì
Ó ÌÌ Ô Ô
Ò ÔÔ ÚÛ ÚÚ Ù
ı ÙÙ ˆ˜ ˆˆ ¯˘ ¯¯ ˙˚ ˙
¸ ˙˙ ˝
˛ ˝˝ ˇÄ ˇ
Å ˇˇ ÇÉ ÇÇ Ñ
Ö ÑÑ Üá ÜÜ àâ àà äã ä
å ää çé çç èê èè ëí ë
ì ëë î
ï îî ñó ñ
ò ññ ôö ôô õ
ú õõ ùû ùù ü† üü °¢ °
£ °° §
• §§ ¶ß ¶
® ¶¶ ©™ ©© ´¨ ´´ ≠Æ ≠
Ø ≠≠ ∞± ∞∞ ≤≥ ≤≤ ¥µ ¥
∂ ¥¥ ∑
∏ ∑∑ π∫ π
ª ππ ºΩ ºº æ
ø ææ ¿¡ ¿¿ ¬√ ¬¬ ƒ≈ ƒ
∆ ƒƒ «
» «« …  …
À …… ÃÕ ÃÃ Œœ ŒŒ –
— –– “” ““ ‘’ ‘
÷ ‘‘ ◊ÿ ◊◊ Ÿ⁄ ŸŸ €‹ €
› €€ ﬁ
ﬂ ﬁﬁ ‡· ‡
‚ ‡‡ „‰ „„ Â
Ê ÂÂ ÁË ÁÁ ÈÍ ÈÈ ÎÏ Î
Ì ÎÎ Ó
Ô ÓÓ Ò 
Ú  ÛÙ ÛÛ ıˆ ıı ˜¯ ˜
˘ ˜˜ ˙˚ ˙˙ ¸˝ ¸¸ ˛
ˇ ˛˛ ÄÅ ÄÄ ÇÉ Ç
Ñ ÇÇ Ö
Ü ÖÖ áà á
â áá äã ää å
ç åå éè éé êë êê íì í
î íí ï
ñ ïï óò ó
ô óó öõ öö úù úú ûü û
† ûû °¢ °° £§ ££ •
¶ •• ß® ßß ©™ ©
´ ©© ¨
≠ ¨¨ ÆØ Æ
∞ ÆÆ ±≤ ±± ≥
¥ ≥≥ µ∂ µµ ∑∏ ∑∑ π∫ π
ª ππ º
Ω ºº æø æ
¿ ææ ¡¬ ¡¡ √ƒ √√ ≈∆ ≈
« ≈≈ »… »»  À    ÃÕ Ã
Œ ÃÃ œ
– œœ —“ —
” —— ‘’ ‘‘ ÷
◊ ÷÷ ÿŸ ÿÿ ⁄€ ⁄⁄ ‹› ‹
ﬁ ‹‹ ﬂ
‡ ﬂﬂ ·‚ ·
„ ·· ‰Â ‰‰ ÊÁ ÊÊ ËÈ Ë
Í ËË ÎÏ ÎÎ ÌÓ ÌÌ Ô Ô
Ò ÔÔ Ú
Û ÚÚ Ùı Ù
ˆ ÙÙ ˜¯ ˜˜ ˘
˙ ˘˘ ˚¸ ˚˚ ˝˛ ˝˝ ˇÄ ˇ
Å ˇˇ Ç
É ÇÇ ÑÖ Ñ
Ü ÑÑ áà áá âä ââ ãå ã
ç ãã éè é
ê éé ëí ëë ì
î ìì ïñ ïï ó
ò óó ôö ô
õ ôô úù úú û
ü ûû †° †† ¢£ ¢¢ §• §
¶ §§ ß
® ßß ©™ ©
´ ©© ¨≠ ¨¨ ÆØ ÆÆ ∞± ∞
≤ ∞∞ ≥¥ ≥≥ µ∂ µµ ∑∏ ∑
π ∑∑ ∫
ª ∫∫ ºΩ º
æ ºº ø¿ øø ¡
¬ ¡¡ √ƒ √√ ≈∆ ≈≈ «» «
… ««  
À    ÃÕ Ã
Œ ÃÃ œ– œœ —“ —— ”‘ ”
’ ”” ÷◊ ÷
ÿ ÷÷ Ÿ⁄ ŸŸ €
‹ €€ ›ﬁ ›
ﬂ ›› ‡· ‡‡ ‚
„ ‚‚ ‰Â ‰‰ ÊÁ ÊÊ ËÈ Ë
Í ËË Î
Ï ÎÎ ÌÓ Ì
Ô ÌÌ Ò  ÚÛ ÚÚ Ùı Ù
ˆ ÙÙ ˜¯ ˜˜ ˘˙ ˘˘ ˚¸ ˚
˝ ˚˚ ˛
ˇ ˛˛ ÄÅ Ä
Ç ÄÄ ÉÑ ÉÉ Ö
Ü ÖÖ áà áá âä ââ ãå ã
ç ãã é
è éé êë ê
í êê ìî ìì ïñ ïï óò ó
ô óó öõ öö úù úú ûü û
† ûû °
¢ °° £§ £
• ££ ¶ß ¶¶ ®
© ®® ™´ ™™ ¨≠ ¨¨ ÆØ Æ
∞ ÆÆ ±
≤ ±± ≥¥ ≥
µ ≥≥ ∂∑ ∂∂ ∏π ∏∏ ∫ª ∫
º ∫∫ Ωæ ΩΩ ø¿ øø ¡¬ ¡
√ ¡¡ ƒ
≈ ƒƒ ∆« ∆
» ∆∆ …  …… À
Ã ÀÀ ÕŒ ÕÕ œ– œœ —“ —
” —— ‘
’ ‘‘ ÷◊ ÷
ÿ ÷÷ Ÿ⁄ ŸŸ €‹ €€ ›ﬁ ›
ﬂ ›› ‡· ‡‡ ‚„ ‚‚ ‰Â ‰
Ê ‰‰ Á
Ë ÁÁ ÈÍ È
Î ÈÈ ÏÌ ÏÏ Ó
Ô ÓÓ Ò  ÚÛ ÚÚ Ùı Ù
ˆ ÙÙ ˜
¯ ˜˜ ˘˙ ˘
˚ ˘˘ ¸˝ ¸¸ ˛ˇ ˛
Ä ˛˛ ÅÇ ÅÅ ÉÑ ÉÉ ÖÜ Ö
á ÖÖ à
â àà äã ä
å ää çé çç è
ê èè ëí ëë ìî ìì ïñ ï
ó ïï ò
ô òò öõ ö
ú öö ùû ùù ü† üü °¢ °
£ °° §• §§ ¶ß ¶¶ ®© ®
™ ®® ´
¨ ´´ ≠Æ ≠
Ø ≠≠ ∞± ∞∞ ≤
≥ ≤≤ ¥µ ¥¥ ∂∑ ∂∂ ∏π ∏
∫ ∏∏ ª
º ªª Ωæ Ω
ø ΩΩ ¿¡ ¿¿ ¬√ ¬¬ ƒ≈ ƒ
∆ ƒƒ «» «
… ««  À    Ã
Õ ÃÃ Œœ ŒŒ –
— –– “” “
‘ ““ ’÷ ’’ ◊
ÿ ◊◊ Ÿ⁄ ŸŸ €‹ €€ ›ﬁ ›
ﬂ ›› ‡
· ‡‡ ‚„ ‚
‰ ‚‚ ÂÊ ÂÂ ÁË ÁÁ ÈÍ È
Î ÈÈ ÏÌ ÏÏ ÓÔ ÓÓ Ò 
Ú  Û
Ù ÛÛ ıˆ ı
˜ ıı ¯˘ ¯¯ ˙
˚ ˙˙ ¸˝ ¸¸ ˛ˇ ˛˛ ÄÅ Ä
Ç ÄÄ É
Ñ ÉÉ ÖÜ Ö
á ÖÖ àâ àà äã ää åç å
é åå èê èè ëí ëë ìî ì
ï ìì ñ
ó ññ òô ò
ö òò õú õõ ù
û ùù ü† üü °¢ °° £§ £
• ££ ¶
ß ¶¶ ®© ®
™ ®® ´¨ ¨ ¨ ¨ %¨ J¨ è¨ õ¨ Ω¨ €¨ È¨ Ñ¨ –¨ ˛¨ •¨ ì¨ Ã	≠ Æ 3Æ ZÆ }Æ ¶Æ ÕÆ ÙÆ õÆ æÆ ÂÆ åÆ ≥Æ ÷Æ ˘Æ ûÆ ¡Æ ‚Æ ÖÆ ®Æ ÀÆ ÓÆ èÆ ≤Æ ◊Æ ˙Æ ùØ <Ø cØ ÜØ ØØ ÷Ø ˝Ø §Ø «Ø ÓØ ïØ ºØ ﬂØ ÇØ ßØ  Ø ÎØ éØ ±Ø ‘Ø ˜Ø òØ ªØ ‡Ø ÉØ ¶∞     	 
             " $# &% (! *' +) - /, 0 21 43 6. 85 :7 ;1 =9 ?< @ B DA FC G IH KJ M% OL QN RP TE VS W YX [Z ]U _\ a^ bX d` fc g i kh mj nJ p% ro tq us wl yv z |{ ~} Äx Ç ÑÅ Ö{ áÉ âÜ ä å éç êè íã îë ï óì ò öô úõ ûù †ñ ¢ü £ •§ ß¶ ©° ´® ≠™ Æ§ ∞¨ ≤Ø ≥ µè ∑¥ π∂ ∫ ºª æΩ ¿õ ¬ø ƒ¡ ≈√ «∏ …∆   ÃÀ ŒÕ –» “œ ‘— ’À ◊” Ÿ÷ ⁄ ‹€ ﬁè ‡› ‚ﬂ „ Â· Ê ËÁ ÍÈ ÏÎ Ó‰ Ì Ò ÛÚ ıÙ ˜Ô ˘ˆ ˚¯ ¸Ú ˛˙ Ä˝ Å ÉÇ ÖÑ áè âÜ ãà å éõ êç íè ìë ïä óî ò öô úõ ûñ †ù ¢ü £ô •° ß§ ®% ™è ¨© Æ´ Ø ±õ ≥∞ µ≤ ∂¥ ∏≠ ∫∑ ª Ωº øæ ¡π √¿ ≈¬ ∆º »ƒ  « À Õ œŒ —– ”Ã ’“ ÷Ω ÿè ⁄◊ ‹Ÿ ›€ ﬂ‘ ·ﬁ ‚ ‰„ ÊÂ Ë‡ ÍÁ ÏÈ Ì„ ÔÎ ÒÓ Ú Ù– ˆÛ ¯ı ˘Ω ˚ ˝¸ ˇ˛ Å˙ ÉÄ ÑÇ Ü˜ àÖ â ãä çå èá ëé ìê îä ñí òï ô€ õ– ùö üú †Ω ¢ §£ ¶• ®° ™ß ´© ≠û Ø¨ ∞ ≤± ¥≥ ∂Æ ∏µ ∫∑ ª± Ωπ øº ¿J ¬– ƒ¡ ∆√ «Ω …È À» Õ  ŒÃ –≈ “œ ” ’‘ ◊÷ Ÿ— €ÿ ›⁄ ﬁ‘ ‡‹ ‚ﬂ „Ñ Â– Á‰ ÈÊ Í Ï˛ ÓÎ Ì ÒÔ ÛË ıÚ ˆ ¯˜ ˙˘ ¸Ù ˛˚ Ä˝ Å˜ Éˇ ÖÇ Ü– àè äá åâ ç èã ê íë îì ñï òé öó õ ùú üû °ô £† •¢ ¶ú ®§ ™ß ´– ≠õ Ø¨ ±Æ ≤è ¥˛ ∂≥ ∏µ π∑ ª∞ Ω∫ æ ¿ø ¬¡ ƒº ∆√ »≈ …ø À« Õ  ŒΩ –˛ “œ ‘— ’ ◊” ÿÈ ⁄Ÿ ‹÷ ﬁ€ ﬂ ·‡ „‚ Â› Á‰ ÈÊ Í‡ ÏË ÓÎ ÔΩ Ò˛ Û ıÚ ˆ€ ¯è ˙˜ ¸˘ ˝˚ ˇÙ Å˛ Ç ÑÉ ÜÖ àÄ äá åâ çÉ èã ëé í î˛ ñì òï ô õè ùö üú †û ¢ó §° • ß¶ ©® ´£ ≠™ Ø¨ ∞¶ ≤Æ ¥± µ ∑˛ π∂ ª∏ ºΩ æõ ¿Ω ¬ø √¡ ≈∫ «ƒ »  … ÃÀ Œ∆ –Õ “œ ”… ’— ◊‘ ÿ ⁄˛ ‹Ÿ ﬁ€ ﬂJ ·è „‡ Â‚ Ê‰ Ë› ÍÁ Î ÌÏ ÔÓ ÒÈ Û ıÚ ˆÏ ¯Ù ˙˜ ˚˛ ˝ ˇ¸ ÄΩ Çè ÑÅ ÜÉ áÖ â˛ ãà å éç êè íä îë ñì óç ôï õò úÑ û˛ †ù ¢ü £% •è ß§ ©¶ ™® ¨° Æ´ Ø ±∞ ≥≤ µ≠ ∑¥ π∂ ∫∞ º∏ æª øΩ ¡• √¿ ≈¬ ∆ »ƒ … À  ÕÃ œŒ —« ”– ‘ ÷’ ÿ◊ ⁄“ ‹Ÿ ﬁ€ ﬂ’ ·› „‡ ‰€ Ê• ËÂ ÍÁ ÎΩ ÌÃ ÔÏ ÒÓ Ú ÙÈ ˆÛ ˜ ˘¯ ˚˙ ˝ı ˇ¸ Å˛ Ç¯ ÑÄ ÜÉ á â• ãà çä éΩ ê˛ íè îë ïì óå ôñ ö úõ ûù †ò ¢ü §° •õ ß£ ©¶ ™ ±± ≤≤ ´7 ≤≤ 7¯ ≤≤ ¯Å ≤≤ Å^ ≤≤ ^Ê ≤≤ Êœ ≤≤ œ ±± ⁄ ≤≤ ⁄€ ≤≤ €ü ≤≤ ü∂ ≤≤ ∂— ≤≤ —˛ ≤≤ ˛∑ ≤≤ ∑ì ≤≤ ì¬ ≤≤ ¬È ≤≤ È≈ ≤≤ ≈â ≤≤ â¢ ≤≤ ¢˝ ≤≤ ˝Ú ≤≤ Ú™ ≤≤ ™¨ ≤≤ ¨° ≤≤ °ê ≤≤ ê	≥ {	¥ 	µ 
∂ ç
∑ Œ
∏ ª	π 1
∫ „
ª £	º H
Ω Ú
æ ô
ø Á
¿ º¡ 	¬ #
√ É
ƒ …	≈ 
∆ ë
« ˜
» ‘
… ’
  ¯
À ¶	Ã 
Õ ô
Œ ‡
œ À
–  — 
— ,— S— v— ü— ∆— Ì— î— ∑— ﬁ— Ö— ¨— œ— Ú— ó— ∫— €— ˛— °— ƒ— Á— à— ´— –— Û— ñ
“ õ
” §	‘ 
’ ç
÷ Ï
◊ ∞
ÿ ø
Ÿ ±
⁄ ú	€ 7	€ ^
€ Å
€ ™
€ —
€ ¯
€ ü
€ ¬
€ È
€ ê
€ ∑
€ ⁄
€ ˝
€ ¢
€ ≈
€ Ê
€ â
€ ¨
€ œ
€ Ú
€ ì
€ ∂
€ €
€ ˛
€ °
‹ ä
› Ç
ﬁ ¸	ﬂ X"
ratt3_kernel"
_Z13get_global_idj"
	_Z4fminff*ó
shoc-1.1.5-S3D-ratt3_kernel.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282Å

transfer_bytes
à¢ª

wgsize_log1p
íÛéA

devmap_label
 
 
transfer_bytes_log1p
íÛéA

wgsize
Ä