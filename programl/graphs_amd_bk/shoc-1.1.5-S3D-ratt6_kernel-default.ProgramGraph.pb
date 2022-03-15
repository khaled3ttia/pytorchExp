
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
/addB(
&
	full_text

%16 = add i64 %6, 136
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
%20 = add i64 %6, 48
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
%23 = add i64 %6, 128
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
%29 = add i64 %6, 800
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
-addB&
$
	full_text

%35 = add i64 %6, 8
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
.addB'
%
	full_text

%38 = add i64 %6, 96
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
%41 = fmul float %37, %40
'floatB

	full_text

	float %37
'floatB

	full_text

	float %40
YgetelementptrBH
F
	full_text9
7
5%42 = getelementptr inbounds float, float* %3, i64 %6
"i64B

	full_text


i64 %6
JloadBB
@
	full_text3
1
/%43 = load float, float* %42, align 4, !tbaa !8
)float*B

	full_text


float* %42
.addB'
%
	full_text

%44 = add i64 %6, 88
"i64B

	full_text


i64 %6
ZgetelementptrBI
G
	full_text:
8
6%45 = getelementptr inbounds float, float* %3, i64 %44
#i64B

	full_text
	
i64 %44
JloadBB
@
	full_text3
1
/%46 = load float, float* %45, align 4, !tbaa !8
)float*B

	full_text


float* %45
4fmulB,
*
	full_text

%47 = fmul float %43, %46
'floatB

	full_text

	float %43
'floatB

	full_text

	float %46
JfdivBB
@
	full_text3
1
/%48 = fdiv float 1.000000e+00, %47, !fpmath !12
'floatB

	full_text

	float %47
4fmulB,
*
	full_text

%49 = fmul float %41, %48
'floatB

	full_text

	float %41
'floatB

	full_text

	float %48
/addB(
&
	full_text

%50 = add i64 %6, 808
"i64B

	full_text


i64 %6
ZgetelementptrBI
G
	full_text:
8
6%51 = getelementptr inbounds float, float* %1, i64 %50
#i64B

	full_text
	
i64 %50
JloadBB
@
	full_text3
1
/%52 = load float, float* %51, align 4, !tbaa !8
)float*B

	full_text


float* %51
ccallB[
Y
	full_textL
J
H%53 = tail call float @_Z4fminff(float %49, float 0x4415AF1D80000000) #2
'floatB

	full_text

	float %49
4fmulB,
*
	full_text

%54 = fmul float %52, %53
'floatB

	full_text

	float %52
'floatB

	full_text

	float %53
ZgetelementptrBI
G
	full_text:
8
6%55 = getelementptr inbounds float, float* %2, i64 %50
#i64B

	full_text
	
i64 %50
JstoreBA
?
	full_text2
0
.store float %54, float* %55, align 4, !tbaa !8
'floatB

	full_text

	float %54
)float*B

	full_text


float* %55
.addB'
%
	full_text

%56 = add i64 %6, 16
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
/%59 = load float, float* %39, align 4, !tbaa !8
)float*B

	full_text


float* %39
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
.addB'
%
	full_text

%61 = add i64 %6, 32
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
JloadBB
@
	full_text3
1
/%64 = load float, float* %45, align 4, !tbaa !8
)float*B

	full_text


float* %45
4fmulB,
*
	full_text

%65 = fmul float %63, %64
'floatB

	full_text

	float %63
'floatB

	full_text

	float %64
JfdivBB
@
	full_text3
1
/%66 = fdiv float 1.000000e+00, %65, !fpmath !12
'floatB

	full_text

	float %65
4fmulB,
*
	full_text

%67 = fmul float %60, %66
'floatB

	full_text

	float %60
'floatB

	full_text

	float %66
/addB(
&
	full_text

%68 = add i64 %6, 816
"i64B

	full_text


i64 %6
ZgetelementptrBI
G
	full_text:
8
6%69 = getelementptr inbounds float, float* %1, i64 %68
#i64B

	full_text
	
i64 %68
JloadBB
@
	full_text3
1
/%70 = load float, float* %69, align 4, !tbaa !8
)float*B

	full_text


float* %69
ccallB[
Y
	full_textL
J
H%71 = tail call float @_Z4fminff(float %67, float 0x4415AF1D80000000) #2
'floatB

	full_text

	float %67
4fmulB,
*
	full_text

%72 = fmul float %70, %71
'floatB

	full_text

	float %70
'floatB

	full_text

	float %71
ZgetelementptrBI
G
	full_text:
8
6%73 = getelementptr inbounds float, float* %2, i64 %68
#i64B

	full_text
	
i64 %68
JstoreBA
?
	full_text2
0
.store float %72, float* %73, align 4, !tbaa !8
'floatB

	full_text

	float %72
)float*B

	full_text


float* %73
JloadBB
@
	full_text3
1
/%74 = load float, float* %62, align 4, !tbaa !8
)float*B

	full_text


float* %62
JloadBB
@
	full_text3
1
/%75 = load float, float* %39, align 4, !tbaa !8
)float*B

	full_text


float* %39
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
.addB'
%
	full_text

%77 = add i64 %6, 40
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
JloadBB
@
	full_text3
1
/%80 = load float, float* %45, align 4, !tbaa !8
)float*B

	full_text


float* %45
4fmulB,
*
	full_text

%81 = fmul float %79, %80
'floatB

	full_text

	float %79
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
%83 = fmul float %76, %82
'floatB

	full_text

	float %76
'floatB

	full_text

	float %82
/addB(
&
	full_text

%84 = add i64 %6, 824
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
.addB'
%
	full_text

%90 = add i64 %6, 64
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
JloadBB
@
	full_text3
1
/%93 = load float, float* %39, align 4, !tbaa !8
)float*B

	full_text


float* %39
4fmulB,
*
	full_text

%94 = fmul float %92, %93
'floatB

	full_text

	float %92
'floatB

	full_text

	float %93
JloadBB
@
	full_text3
1
/%95 = load float, float* %36, align 4, !tbaa !8
)float*B

	full_text


float* %36
/addB(
&
	full_text

%96 = add i64 %6, 168
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

%101 = fmul float %94, %100
'floatB

	full_text

	float %94
(floatB

	full_text


float %100
0addB)
'
	full_text

%102 = add i64 %6, 832
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
/addB(
&
	full_text

%108 = add i64 %6, 72
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
0%111 = load float, float* %39, align 4, !tbaa !8
)float*B

	full_text


float* %39
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
KloadBC
A
	full_text4
2
0%113 = load float, float* %45, align 4, !tbaa !8
)float*B

	full_text


float* %45
7fmulB/
-
	full_text 

%114 = fmul float %113, %113
(floatB

	full_text


float %113
(floatB

	full_text


float %113
LfdivBD
B
	full_text5
3
1%115 = fdiv float 1.000000e+00, %114, !fpmath !12
(floatB

	full_text


float %114
7fmulB/
-
	full_text 

%116 = fmul float %112, %115
(floatB

	full_text


float %112
(floatB

	full_text


float %115
0addB)
'
	full_text

%117 = add i64 %6, 840
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%118 = getelementptr inbounds float, float* %1, i64 %117
$i64B

	full_text


i64 %117
LloadBD
B
	full_text5
3
1%119 = load float, float* %118, align 4, !tbaa !8
*float*B

	full_text

float* %118
ecallB]
[
	full_textN
L
J%120 = tail call float @_Z4fminff(float %116, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %116
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
\getelementptrBK
I
	full_text<
:
8%122 = getelementptr inbounds float, float* %2, i64 %117
$i64B

	full_text


i64 %117
LstoreBC
A
	full_text4
2
0store float %121, float* %122, align 4, !tbaa !8
(floatB

	full_text


float %121
*float*B

	full_text

float* %122
/addB(
&
	full_text

%123 = add i64 %6, 80
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%124 = getelementptr inbounds float, float* %3, i64 %123
$i64B

	full_text


i64 %123
LloadBD
B
	full_text5
3
1%125 = load float, float* %124, align 4, !tbaa !8
*float*B

	full_text

float* %124
KloadBC
A
	full_text4
2
0%126 = load float, float* %39, align 4, !tbaa !8
)float*B

	full_text


float* %39
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
0%128 = load float, float* %45, align 4, !tbaa !8
)float*B

	full_text


float* %45
7fmulB/
-
	full_text 

%129 = fmul float %128, %128
(floatB

	full_text


float %128
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

%131 = fmul float %127, %130
(floatB

	full_text


float %127
(floatB

	full_text


float %130
0addB)
'
	full_text

%132 = add i64 %6, 848
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
0%138 = load float, float* %36, align 4, !tbaa !8
)float*B

	full_text


float* %36
0addB)
'
	full_text

%139 = add i64 %6, 192
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%140 = getelementptr inbounds float, float* %3, i64 %139
$i64B

	full_text


i64 %139
LloadBD
B
	full_text5
3
1%141 = load float, float* %140, align 4, !tbaa !8
*float*B

	full_text

float* %140
7fmulB/
-
	full_text 

%142 = fmul float %138, %141
(floatB

	full_text


float %138
(floatB

	full_text


float %141
LloadBD
B
	full_text5
3
1%143 = load float, float* %124, align 4, !tbaa !8
*float*B

	full_text

float* %124
0addB)
'
	full_text

%144 = add i64 %6, 104
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%145 = getelementptr inbounds float, float* %3, i64 %144
$i64B

	full_text


i64 %144
LloadBD
B
	full_text5
3
1%146 = load float, float* %145, align 4, !tbaa !8
*float*B

	full_text

float* %145
7fmulB/
-
	full_text 

%147 = fmul float %143, %146
(floatB

	full_text


float %143
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

%149 = fmul float %142, %148
(floatB

	full_text


float %142
(floatB

	full_text


float %148
0addB)
'
	full_text

%150 = add i64 %6, 856
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
0%156 = load float, float* %57, align 4, !tbaa !8
)float*B

	full_text


float* %57
LloadBD
B
	full_text5
3
1%157 = load float, float* %140, align 4, !tbaa !8
*float*B

	full_text

float* %140
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
KloadBC
A
	full_text4
2
0%159 = load float, float* %36, align 4, !tbaa !8
)float*B

	full_text


float* %36
LloadBD
B
	full_text5
3
1%160 = load float, float* %145, align 4, !tbaa !8
*float*B

	full_text

float* %145
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
6fmulB.
,
	full_text

%163 = fmul float %12, %162
'floatB

	full_text

	float %12
(floatB

	full_text


float %162
LfdivBD
B
	full_text5
3
1%164 = fdiv float 1.000000e+00, %163, !fpmath !12
(floatB

	full_text


float %163
7fmulB/
-
	full_text 

%165 = fmul float %158, %164
(floatB

	full_text


float %158
(floatB

	full_text


float %164
0addB)
'
	full_text

%166 = add i64 %6, 864
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%167 = getelementptr inbounds float, float* %1, i64 %166
$i64B

	full_text


i64 %166
LloadBD
B
	full_text5
3
1%168 = load float, float* %167, align 4, !tbaa !8
*float*B

	full_text

float* %167
ecallB]
[
	full_textN
L
J%169 = tail call float @_Z4fminff(float %165, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %165
7fmulB/
-
	full_text 

%170 = fmul float %168, %169
(floatB

	full_text


float %168
(floatB

	full_text


float %169
\getelementptrBK
I
	full_text<
:
8%171 = getelementptr inbounds float, float* %2, i64 %166
$i64B

	full_text


i64 %166
LstoreBC
A
	full_text4
2
0store float %170, float* %171, align 4, !tbaa !8
(floatB

	full_text


float %170
*float*B

	full_text

float* %171
KloadBC
A
	full_text4
2
0%172 = load float, float* %14, align 4, !tbaa !8
)float*B

	full_text


float* %14
LloadBD
B
	full_text5
3
1%173 = load float, float* %140, align 4, !tbaa !8
*float*B

	full_text

float* %140
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
KloadBC
A
	full_text4
2
0%175 = load float, float* %62, align 4, !tbaa !8
)float*B

	full_text


float* %62
LloadBD
B
	full_text5
3
1%176 = load float, float* %145, align 4, !tbaa !8
*float*B

	full_text

float* %145
7fmulB/
-
	full_text 

%177 = fmul float %175, %176
(floatB

	full_text


float %175
(floatB

	full_text


float %176
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
6fmulB.
,
	full_text

%179 = fmul float %12, %178
'floatB

	full_text

	float %12
(floatB

	full_text


float %178
LfdivBD
B
	full_text5
3
1%180 = fdiv float 1.000000e+00, %179, !fpmath !12
(floatB

	full_text


float %179
7fmulB/
-
	full_text 

%181 = fmul float %174, %180
(floatB

	full_text


float %174
(floatB

	full_text


float %180
0addB)
'
	full_text

%182 = add i64 %6, 872
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%183 = getelementptr inbounds float, float* %1, i64 %182
$i64B

	full_text


i64 %182
LloadBD
B
	full_text5
3
1%184 = load float, float* %183, align 4, !tbaa !8
*float*B

	full_text

float* %183
ecallB]
[
	full_textN
L
J%185 = tail call float @_Z4fminff(float %181, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %181
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
\getelementptrBK
I
	full_text<
:
8%187 = getelementptr inbounds float, float* %2, i64 %182
$i64B

	full_text


i64 %182
LstoreBC
A
	full_text4
2
0store float %186, float* %187, align 4, !tbaa !8
(floatB

	full_text


float %186
*float*B

	full_text

float* %187
KloadBC
A
	full_text4
2
0%188 = load float, float* %91, align 4, !tbaa !8
)float*B

	full_text


float* %91
LloadBD
B
	full_text5
3
1%189 = load float, float* %140, align 4, !tbaa !8
*float*B

	full_text

float* %140
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
LloadBD
B
	full_text5
3
1%191 = load float, float* %145, align 4, !tbaa !8
*float*B

	full_text

float* %145
0addB)
'
	full_text

%192 = add i64 %6, 144
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%193 = getelementptr inbounds float, float* %3, i64 %192
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
7fmulB/
-
	full_text 

%195 = fmul float %191, %194
(floatB

	full_text


float %191
(floatB

	full_text


float %194
LfdivBD
B
	full_text5
3
1%196 = fdiv float 1.000000e+00, %195, !fpmath !12
(floatB

	full_text


float %195
7fmulB/
-
	full_text 

%197 = fmul float %190, %196
(floatB

	full_text


float %190
(floatB

	full_text


float %196
0addB)
'
	full_text

%198 = add i64 %6, 880
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%199 = getelementptr inbounds float, float* %1, i64 %198
$i64B

	full_text


i64 %198
LloadBD
B
	full_text5
3
1%200 = load float, float* %199, align 4, !tbaa !8
*float*B

	full_text

float* %199
ecallB]
[
	full_textN
L
J%201 = tail call float @_Z4fminff(float %197, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %197
7fmulB/
-
	full_text 

%202 = fmul float %200, %201
(floatB

	full_text


float %200
(floatB

	full_text


float %201
\getelementptrBK
I
	full_text<
:
8%203 = getelementptr inbounds float, float* %2, i64 %198
$i64B

	full_text


i64 %198
LstoreBC
A
	full_text4
2
0store float %202, float* %203, align 4, !tbaa !8
(floatB

	full_text


float %202
*float*B

	full_text

float* %203
LloadBD
B
	full_text5
3
1%204 = load float, float* %109, align 4, !tbaa !8
*float*B

	full_text

float* %109
LloadBD
B
	full_text5
3
1%205 = load float, float* %140, align 4, !tbaa !8
*float*B

	full_text

float* %140
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
LloadBD
B
	full_text5
3
1%207 = load float, float* %145, align 4, !tbaa !8
*float*B

	full_text

float* %145
0addB)
'
	full_text

%208 = add i64 %6, 160
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%209 = getelementptr inbounds float, float* %3, i64 %208
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
7fmulB/
-
	full_text 

%211 = fmul float %207, %210
(floatB

	full_text


float %207
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

%213 = fmul float %206, %212
(floatB

	full_text


float %206
(floatB

	full_text


float %212
0addB)
'
	full_text

%214 = add i64 %6, 888
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
1%220 = load float, float* %140, align 4, !tbaa !8
*float*B

	full_text

float* %140
7fmulB/
-
	full_text 

%221 = fmul float %220, %220
(floatB

	full_text


float %220
(floatB

	full_text


float %220
LloadBD
B
	full_text5
3
1%222 = load float, float* %145, align 4, !tbaa !8
*float*B

	full_text

float* %145
7fmulB/
-
	full_text 

%223 = fmul float %222, %222
(floatB

	full_text


float %222
(floatB

	full_text


float %222
LloadBD
B
	full_text5
3
1%224 = load float, float* %193, align 4, !tbaa !8
*float*B

	full_text

float* %193
7fmulB/
-
	full_text 

%225 = fmul float %223, %224
(floatB

	full_text


float %223
(floatB

	full_text


float %224
6fmulB.
,
	full_text

%226 = fmul float %12, %225
'floatB

	full_text

	float %12
(floatB

	full_text


float %225
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

%228 = fmul float %221, %227
(floatB

	full_text


float %221
(floatB

	full_text


float %227
0addB)
'
	full_text

%229 = add i64 %6, 896
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
1%235 = load float, float* %193, align 4, !tbaa !8
*float*B

	full_text

float* %193
0addB)
'
	full_text

%236 = add i64 %6, 152
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%237 = getelementptr inbounds float, float* %3, i64 %236
$i64B

	full_text


i64 %236
LloadBD
B
	full_text5
3
1%238 = load float, float* %237, align 4, !tbaa !8
*float*B

	full_text

float* %237
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
0addB)
'
	full_text

%241 = add i64 %6, 904
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
1%247 = load float, float* %209, align 4, !tbaa !8
*float*B

	full_text

float* %209
KloadBC
A
	full_text4
2
0%248 = load float, float* %36, align 4, !tbaa !8
)float*B

	full_text


float* %36
LloadBD
B
	full_text5
3
1%249 = load float, float* %193, align 4, !tbaa !8
*float*B

	full_text

float* %193
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
6fmulB.
,
	full_text

%251 = fmul float %12, %250
'floatB

	full_text

	float %12
(floatB

	full_text


float %250
LfdivBD
B
	full_text5
3
1%252 = fdiv float 1.000000e+00, %251, !fpmath !12
(floatB

	full_text


float %251
7fmulB/
-
	full_text 

%253 = fmul float %247, %252
(floatB

	full_text


float %247
(floatB

	full_text


float %252
0addB)
'
	full_text

%254 = add i64 %6, 912
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%255 = getelementptr inbounds float, float* %1, i64 %254
$i64B

	full_text


i64 %254
LloadBD
B
	full_text5
3
1%256 = load float, float* %255, align 4, !tbaa !8
*float*B

	full_text

float* %255
ecallB]
[
	full_textN
L
J%257 = tail call float @_Z4fminff(float %253, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %253
7fmulB/
-
	full_text 

%258 = fmul float %256, %257
(floatB

	full_text


float %256
(floatB

	full_text


float %257
\getelementptrBK
I
	full_text<
:
8%259 = getelementptr inbounds float, float* %2, i64 %254
$i64B

	full_text


i64 %254
LstoreBC
A
	full_text4
2
0store float %258, float* %259, align 4, !tbaa !8
(floatB

	full_text


float %258
*float*B

	full_text

float* %259
KloadBC
A
	full_text4
2
0%260 = load float, float* %57, align 4, !tbaa !8
)float*B

	full_text


float* %57
LloadBD
B
	full_text5
3
1%261 = load float, float* %193, align 4, !tbaa !8
*float*B

	full_text

float* %193
7fmulB/
-
	full_text 

%262 = fmul float %260, %261
(floatB

	full_text


float %260
(floatB

	full_text


float %261
KloadBC
A
	full_text4
2
0%263 = load float, float* %36, align 4, !tbaa !8
)float*B

	full_text


float* %36
LloadBD
B
	full_text5
3
1%264 = load float, float* %140, align 4, !tbaa !8
*float*B

	full_text

float* %140
7fmulB/
-
	full_text 

%265 = fmul float %263, %264
(floatB

	full_text


float %263
(floatB

	full_text


float %264
LfdivBD
B
	full_text5
3
1%266 = fdiv float 1.000000e+00, %265, !fpmath !12
(floatB

	full_text


float %265
7fmulB/
-
	full_text 

%267 = fmul float %262, %266
(floatB

	full_text


float %262
(floatB

	full_text


float %266
0addB)
'
	full_text

%268 = add i64 %6, 920
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%269 = getelementptr inbounds float, float* %1, i64 %268
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
ecallB]
[
	full_textN
L
J%271 = tail call float @_Z4fminff(float %267, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %267
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
\getelementptrBK
I
	full_text<
:
8%273 = getelementptr inbounds float, float* %2, i64 %268
$i64B

	full_text


i64 %268
LstoreBC
A
	full_text4
2
0store float %272, float* %273, align 4, !tbaa !8
(floatB

	full_text


float %272
*float*B

	full_text

float* %273
KloadBC
A
	full_text4
2
0%274 = load float, float* %57, align 4, !tbaa !8
)float*B

	full_text


float* %57
LloadBD
B
	full_text5
3
1%275 = load float, float* %193, align 4, !tbaa !8
*float*B

	full_text

float* %193
7fmulB/
-
	full_text 

%276 = fmul float %274, %275
(floatB

	full_text


float %274
(floatB

	full_text


float %275
LloadBD
B
	full_text5
3
1%277 = load float, float* %109, align 4, !tbaa !8
*float*B

	full_text

float* %109
LloadBD
B
	full_text5
3
1%278 = load float, float* %145, align 4, !tbaa !8
*float*B

	full_text

float* %145
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
LfdivBD
B
	full_text5
3
1%280 = fdiv float 1.000000e+00, %279, !fpmath !12
(floatB

	full_text


float %279
7fmulB/
-
	full_text 

%281 = fmul float %276, %280
(floatB

	full_text


float %276
(floatB

	full_text


float %280
0addB)
'
	full_text

%282 = add i64 %6, 928
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%283 = getelementptr inbounds float, float* %1, i64 %282
$i64B

	full_text


i64 %282
LloadBD
B
	full_text5
3
1%284 = load float, float* %283, align 4, !tbaa !8
*float*B

	full_text

float* %283
ecallB]
[
	full_textN
L
J%285 = tail call float @_Z4fminff(float %281, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %281
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
\getelementptrBK
I
	full_text<
:
8%287 = getelementptr inbounds float, float* %2, i64 %282
$i64B

	full_text


i64 %282
LstoreBC
A
	full_text4
2
0store float %286, float* %287, align 4, !tbaa !8
(floatB

	full_text


float %286
*float*B

	full_text

float* %287
KloadBC
A
	full_text4
2
0%288 = load float, float* %62, align 4, !tbaa !8
)float*B

	full_text


float* %62
LloadBD
B
	full_text5
3
1%289 = load float, float* %193, align 4, !tbaa !8
*float*B

	full_text

float* %193
7fmulB/
-
	full_text 

%290 = fmul float %288, %289
(floatB

	full_text


float %288
(floatB

	full_text


float %289
KloadBC
A
	full_text4
2
0%291 = load float, float* %36, align 4, !tbaa !8
)float*B

	full_text


float* %36
0addB)
'
	full_text

%292 = add i64 %6, 200
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
7fmulB/
-
	full_text 

%295 = fmul float %291, %294
(floatB

	full_text


float %291
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

%297 = fmul float %290, %296
(floatB

	full_text


float %290
(floatB

	full_text


float %296
0addB)
'
	full_text

%298 = add i64 %6, 936
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
0%304 = load float, float* %62, align 4, !tbaa !8
)float*B

	full_text


float* %62
LloadBD
B
	full_text5
3
1%305 = load float, float* %193, align 4, !tbaa !8
*float*B

	full_text

float* %193
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
0%307 = load float, float* %45, align 4, !tbaa !8
)float*B

	full_text


float* %45
LloadBD
B
	full_text5
3
1%308 = load float, float* %145, align 4, !tbaa !8
*float*B

	full_text

float* %145
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
%312 = add i64 %6, 944
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
0addB)
'
	full_text

%318 = add i64 %6, 120
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%319 = getelementptr inbounds float, float* %3, i64 %318
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
LloadBD
B
	full_text5
3
1%321 = load float, float* %193, align 4, !tbaa !8
*float*B

	full_text

float* %193
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
LloadBD
B
	full_text5
3
1%323 = load float, float* %145, align 4, !tbaa !8
*float*B

	full_text

float* %145
LloadBD
B
	full_text5
3
1%324 = load float, float* %209, align 4, !tbaa !8
*float*B

	full_text

float* %209
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
%328 = add i64 %6, 952
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
0%334 = load float, float* %45, align 4, !tbaa !8
)float*B

	full_text


float* %45
LloadBD
B
	full_text5
3
1%335 = load float, float* %193, align 4, !tbaa !8
*float*B

	full_text

float* %193
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
6fmulB.
,
	full_text

%337 = fmul float %12, %336
'floatB

	full_text

	float %12
(floatB

	full_text


float %336
0addB)
'
	full_text

%338 = add i64 %6, 224
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%339 = getelementptr inbounds float, float* %3, i64 %338
$i64B

	full_text


i64 %338
LloadBD
B
	full_text5
3
1%340 = load float, float* %339, align 4, !tbaa !8
*float*B

	full_text

float* %339
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

%342 = fmul float %337, %341
(floatB

	full_text


float %337
(floatB

	full_text


float %341
0addB)
'
	full_text

%343 = add i64 %6, 960
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
LloadBD
B
	full_text5
3
1%349 = load float, float* %193, align 4, !tbaa !8
*float*B

	full_text

float* %193
LloadBD
B
	full_text5
3
1%350 = load float, float* %237, align 4, !tbaa !8
*float*B

	full_text

float* %237
LfdivBD
B
	full_text5
3
1%351 = fdiv float 1.000000e+00, %350, !fpmath !12
(floatB

	full_text


float %350
7fmulB/
-
	full_text 

%352 = fmul float %349, %351
(floatB

	full_text


float %349
(floatB

	full_text


float %351
0addB)
'
	full_text

%353 = add i64 %6, 968
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%354 = getelementptr inbounds float, float* %1, i64 %353
$i64B

	full_text


i64 %353
LloadBD
B
	full_text5
3
1%355 = load float, float* %354, align 4, !tbaa !8
*float*B

	full_text

float* %354
ecallB]
[
	full_textN
L
J%356 = tail call float @_Z4fminff(float %352, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %352
7fmulB/
-
	full_text 

%357 = fmul float %355, %356
(floatB

	full_text


float %355
(floatB

	full_text


float %356
\getelementptrBK
I
	full_text<
:
8%358 = getelementptr inbounds float, float* %2, i64 %353
$i64B

	full_text


i64 %353
LstoreBC
A
	full_text4
2
0store float %357, float* %358, align 4, !tbaa !8
(floatB

	full_text


float %357
*float*B

	full_text

float* %358
KloadBC
A
	full_text4
2
0%359 = load float, float* %57, align 4, !tbaa !8
)float*B

	full_text


float* %57
LloadBD
B
	full_text5
3
1%360 = load float, float* %237, align 4, !tbaa !8
*float*B

	full_text

float* %237
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
LloadBD
B
	full_text5
3
1%362 = load float, float* %109, align 4, !tbaa !8
*float*B

	full_text

float* %109
LloadBD
B
	full_text5
3
1%363 = load float, float* %145, align 4, !tbaa !8
*float*B

	full_text

float* %145
7fmulB/
-
	full_text 

%364 = fmul float %362, %363
(floatB

	full_text


float %362
(floatB

	full_text


float %363
LfdivBD
B
	full_text5
3
1%365 = fdiv float 1.000000e+00, %364, !fpmath !12
(floatB

	full_text


float %364
7fmulB/
-
	full_text 

%366 = fmul float %361, %365
(floatB

	full_text


float %361
(floatB

	full_text


float %365
0addB)
'
	full_text

%367 = add i64 %6, 976
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%368 = getelementptr inbounds float, float* %1, i64 %367
$i64B

	full_text


i64 %367
LloadBD
B
	full_text5
3
1%369 = load float, float* %368, align 4, !tbaa !8
*float*B

	full_text

float* %368
ecallB]
[
	full_textN
L
J%370 = tail call float @_Z4fminff(float %366, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %366
7fmulB/
-
	full_text 

%371 = fmul float %369, %370
(floatB

	full_text


float %369
(floatB

	full_text


float %370
\getelementptrBK
I
	full_text<
:
8%372 = getelementptr inbounds float, float* %2, i64 %367
$i64B

	full_text


i64 %367
LstoreBC
A
	full_text4
2
0store float %371, float* %372, align 4, !tbaa !8
(floatB

	full_text


float %371
*float*B

	full_text

float* %372
KloadBC
A
	full_text4
2
0%373 = load float, float* %62, align 4, !tbaa !8
)float*B

	full_text


float* %62
LloadBD
B
	full_text5
3
1%374 = load float, float* %237, align 4, !tbaa !8
*float*B

	full_text

float* %237
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
KloadBC
A
	full_text4
2
0%376 = load float, float* %36, align 4, !tbaa !8
)float*B

	full_text


float* %36
LloadBD
B
	full_text5
3
1%377 = load float, float* %293, align 4, !tbaa !8
*float*B

	full_text

float* %293
7fmulB/
-
	full_text 

%378 = fmul float %376, %377
(floatB

	full_text


float %376
(floatB

	full_text


float %377
LfdivBD
B
	full_text5
3
1%379 = fdiv float 1.000000e+00, %378, !fpmath !12
(floatB

	full_text


float %378
7fmulB/
-
	full_text 

%380 = fmul float %375, %379
(floatB

	full_text


float %375
(floatB

	full_text


float %379
0addB)
'
	full_text

%381 = add i64 %6, 984
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%382 = getelementptr inbounds float, float* %1, i64 %381
$i64B

	full_text


i64 %381
LloadBD
B
	full_text5
3
1%383 = load float, float* %382, align 4, !tbaa !8
*float*B

	full_text

float* %382
ecallB]
[
	full_textN
L
J%384 = tail call float @_Z4fminff(float %380, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %380
7fmulB/
-
	full_text 

%385 = fmul float %383, %384
(floatB

	full_text


float %383
(floatB

	full_text


float %384
\getelementptrBK
I
	full_text<
:
8%386 = getelementptr inbounds float, float* %2, i64 %381
$i64B

	full_text


i64 %381
LstoreBC
A
	full_text4
2
0store float %385, float* %386, align 4, !tbaa !8
(floatB

	full_text


float %385
*float*B

	full_text

float* %386
KloadBC
A
	full_text4
2
0%387 = load float, float* %14, align 4, !tbaa !8
)float*B

	full_text


float* %14
LloadBD
B
	full_text5
3
1%388 = load float, float* %237, align 4, !tbaa !8
*float*B

	full_text

float* %237
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
LloadBD
B
	full_text5
3
1%390 = load float, float* %109, align 4, !tbaa !8
*float*B

	full_text

float* %109
0addB)
'
	full_text

%391 = add i64 %6, 112
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%392 = getelementptr inbounds float, float* %3, i64 %391
$i64B

	full_text


i64 %391
LloadBD
B
	full_text5
3
1%393 = load float, float* %392, align 4, !tbaa !8
*float*B

	full_text

float* %392
7fmulB/
-
	full_text 

%394 = fmul float %390, %393
(floatB

	full_text


float %390
(floatB

	full_text


float %393
LfdivBD
B
	full_text5
3
1%395 = fdiv float 1.000000e+00, %394, !fpmath !12
(floatB

	full_text


float %394
7fmulB/
-
	full_text 

%396 = fmul float %389, %395
(floatB

	full_text


float %389
(floatB

	full_text


float %395
0addB)
'
	full_text

%397 = add i64 %6, 992
"i64B

	full_text


i64 %6
\getelementptrBK
I
	full_text<
:
8%398 = getelementptr inbounds float, float* %1, i64 %397
$i64B

	full_text


i64 %397
LloadBD
B
	full_text5
3
1%399 = load float, float* %398, align 4, !tbaa !8
*float*B

	full_text

float* %398
ecallB]
[
	full_textN
L
J%400 = tail call float @_Z4fminff(float %396, float 0x4415AF1D80000000) #2
(floatB

	full_text


float %396
7fmulB/
-
	full_text 

%401 = fmul float %399, %400
(floatB

	full_text


float %399
(floatB

	full_text


float %400
\getelementptrBK
I
	full_text<
:
8%402 = getelementptr inbounds float, float* %2, i64 %397
$i64B

	full_text


i64 %397
LstoreBC
A
	full_text4
2
0store float %401, float* %402, align 4, !tbaa !8
(floatB

	full_text


float %401
*float*B

	full_text

float* %402
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
	
i64 856
%i648B

	full_text
	
i64 832
%i648B

	full_text
	
i64 848
%i648B

	full_text
	
i64 104
#i648B

	full_text	

i64 8
$i648B

	full_text


i64 88
%i648B

	full_text
	
i64 168
%i648B

	full_text
	
i64 928
%i648B

	full_text
	
i64 872
%i648B

	full_text
	
i64 888
$i648B

	full_text


i64 48
$i648B

	full_text


i64 16
%i648B

	full_text
	
i64 192
%i648B

	full_text
	
i64 912
%i648B

	full_text
	
i64 936
%i648B

	full_text
	
i64 896
%i648B

	full_text
	
i64 952
%i648B

	full_text
	
i64 960
%i648B

	full_text
	
i64 992
%i648B

	full_text
	
i64 904
%i648B

	full_text
	
i64 112
%i648B

	full_text
	
i64 808
$i648B

	full_text


i64 80
$i648B

	full_text


i64 40
%i648B

	full_text
	
i64 128
$i648B

	full_text


i64 72
%i648B

	full_text
	
i64 864
8float8B+
)
	full_text

float 0x4415AF1D80000000
$i648B

	full_text


i64 96
%i648B

	full_text
	
i64 968
#i328B

	full_text	

i32 0
%i648B

	full_text
	
i64 880
%i648B

	full_text
	
i64 144
%i648B

	full_text
	
i64 944
2float8B%
#
	full_text

float 1.013250e+06
%i648B

	full_text
	
i64 136
%i648B

	full_text
	
i64 816
%i648B

	full_text
	
i64 160
%i648B

	full_text
	
i64 152
%i648B

	full_text
	
i64 224
%i648B

	full_text
	
i64 800
%i648B

	full_text
	
i64 984
$i648B

	full_text


i64 64
2float8B%
#
	full_text

float 1.000000e+00
%i648B

	full_text
	
i64 824
%i648B

	full_text
	
i64 840
$i648B

	full_text


i64 32
%i648B

	full_text
	
i64 120
%i648B

	full_text
	
i64 976
%i648B

	full_text
	
i64 920
%i648B

	full_text
	
i64 200
$i648B

	full_text


i64 24
8float8B+
)
	full_text

float 0x4193D2C640000000       	  
 

                       !" !! #$ ## %& %% '( '' )* )+ )) ,- ,, ./ .0 .. 12 11 34 33 56 55 78 77 9: 9; 99 <= << >? >@ >> AB AA CD CC EF EE GH GG IJ II KL KK MN MO MM PQ PP RS RR TU TT VW VV XY XX Z[ Z\ ZZ ]^ ]] _` _a __ bc bb de dd fg ff hi hh jk jl jj mn mm op oq oo rs rr tu tt vw vv xy xx z{ z| zz }~ }} 	Ä  ÅÇ ÅÅ ÉÑ ÉÉ ÖÜ Ö
á ÖÖ à
â àà äã ä
å ää çé çç è
ê èè ëí ëë ìî ìì ïñ ï
ó ïï ò
ô òò öõ ö
ú öö ùû ùù ü† üü °¢ °
£ °° §• §§ ¶
ß ¶¶ ®© ®® ™´ ™™ ¨≠ ¨
Æ ¨¨ Ø
∞ ØØ ±≤ ±
≥ ±± ¥µ ¥¥ ∂
∑ ∂∂ ∏π ∏∏ ∫ª ∫∫ ºΩ º
æ ºº ø
¿ øø ¡¬ ¡
√ ¡¡ ƒ≈ ƒƒ ∆
« ∆∆ »… »»  À    ÃÕ Ã
Œ ÃÃ œ– œœ —“ —— ”
‘ ”” ’÷ ’’ ◊ÿ ◊
Ÿ ◊◊ ⁄
€ ⁄⁄ ‹› ‹
ﬁ ‹‹ ﬂ‡ ﬂﬂ ·
‚ ·· „‰ „„ ÂÊ ÂÂ ÁË Á
È ÁÁ Í
Î ÍÍ ÏÌ Ï
Ó ÏÏ Ô ÔÔ Ò
Ú ÒÒ ÛÙ ÛÛ ıˆ ıı ˜¯ ˜
˘ ˜˜ ˙˚ ˙˙ ¸˝ ¸
˛ ¸¸ ˇ
Ä ˇˇ ÅÇ Å
É ÅÅ ÑÖ ÑÑ Ü
á ÜÜ àâ àà äã ää åç å
é åå è
ê èè ëí ë
ì ëë îï îî ñ
ó ññ òô òò öõ öö úù ú
û úú ü† üü °¢ °
£ °° §
• §§ ¶ß ¶
® ¶¶ ©™ ©© ´
¨ ´´ ≠Æ ≠≠ Ø∞ ØØ ±≤ ±
≥ ±± ¥
µ ¥¥ ∂∑ ∂
∏ ∂∂ π∫ ππ ªº ªª Ω
æ ΩΩ ø¿ øø ¡¬ ¡
√ ¡¡ ƒ≈ ƒƒ ∆« ∆∆ »
… »»  À    ÃÕ Ã
Œ ÃÃ œ
– œœ —“ —
” —— ‘’ ‘‘ ÷
◊ ÷÷ ÿŸ ÿÿ ⁄€ ⁄⁄ ‹› ‹
ﬁ ‹‹ ﬂ
‡ ﬂﬂ ·‚ ·
„ ·· ‰Â ‰‰ ÊÁ ÊÊ ËÈ Ë
Í ËË ÎÏ ÎÎ ÌÓ ÌÌ Ô Ô
Ò ÔÔ ÚÛ Ú
Ù ÚÚ ıˆ ı
˜ ıı ¯
˘ ¯¯ ˙˚ ˙
¸ ˙˙ ˝˛ ˝˝ ˇ
Ä ˇˇ ÅÇ ÅÅ ÉÑ ÉÉ ÖÜ Ö
á ÖÖ à
â àà äã ä
å ää çé çç èê èè ëí ë
ì ëë îï îî ñó ññ òô ò
ö òò õú õ
ù õõ ûü û
† ûû °
¢ °° £§ £
• ££ ¶ß ¶¶ ®
© ®® ™´ ™™ ¨≠ ¨¨ ÆØ Æ
∞ ÆÆ ±
≤ ±± ≥¥ ≥
µ ≥≥ ∂∑ ∂∂ ∏π ∏∏ ∫ª ∫
º ∫∫ Ωæ ΩΩ ø¿ øø ¡
¬ ¡¡ √ƒ √√ ≈∆ ≈
« ≈≈ »
… »»  À  
Ã    ÕŒ ÕÕ œ
– œœ —“ —— ”‘ ”” ’÷ ’
◊ ’’ ÿ
Ÿ ÿÿ ⁄€ ⁄
‹ ⁄⁄ ›ﬁ ›› ﬂ‡ ﬂﬂ ·‚ ·
„ ·· ‰Â ‰‰ ÊÁ ÊÊ Ë
È ËË ÍÎ ÍÍ ÏÌ Ï
Ó ÏÏ Ô
 ÔÔ ÒÚ Ò
Û ÒÒ Ùı ÙÙ ˆ
˜ ˆˆ ¯˘ ¯¯ ˙˚ ˙˙ ¸˝ ¸
˛ ¸¸ ˇ
Ä ˇˇ ÅÇ Å
É ÅÅ ÑÖ ÑÑ Üá Ü
à ÜÜ âä ââ ãå ã
ç ãã éè éé êë ê
í êê ìî ì
ï ìì ñ
ó ññ òô ò
ö òò õú õõ ù
û ùù ü† üü °¢ °° £§ £
• ££ ¶
ß ¶¶ ®© ®
™ ®® ´¨ ´´ ≠Æ ≠≠ Ø
∞ ØØ ±≤ ±± ≥
¥ ≥≥ µ∂ µ
∑ µµ ∏π ∏∏ ∫
ª ∫∫ ºΩ ºº æø ææ ¿¡ ¿
¬ ¿¿ √
ƒ √√ ≈∆ ≈
« ≈≈ »… »»  À    ÃÕ ÃÃ Œœ Œ
– ŒŒ —“ —
” —— ‘
’ ‘‘ ÷◊ ÷
ÿ ÷÷ Ÿ⁄ ŸŸ €
‹ €€ ›ﬁ ›› ﬂ‡ ﬂﬂ ·‚ ·
„ ·· ‰
Â ‰‰ ÊÁ Ê
Ë ÊÊ ÈÍ ÈÈ ÎÏ ÎÎ ÌÓ Ì
Ô ÌÌ Ò  ÚÛ ÚÚ Ùı Ù
ˆ ÙÙ ˜
¯ ˜˜ ˘˙ ˘
˚ ˘˘ ¸˝ ¸¸ ˛
ˇ ˛˛ ÄÅ ÄÄ ÇÉ ÇÇ ÑÖ Ñ
Ü ÑÑ á
à áá âä â
ã ââ åç åå éè éé êë ê
í êê ìî ìì ïñ ïï óò ó
ô óó ö
õ öö úù ú
û úú ü† üü °
¢ °° £§ ££ •¶ •• ß® ß
© ßß ™
´ ™™ ¨≠ ¨
Æ ¨¨ Ø∞ ØØ ±≤ ±± ≥¥ ≥
µ ≥≥ ∂∑ ∂∂ ∏π ∏∏ ∫
ª ∫∫ ºΩ ºº æø æ
¿ ææ ¡
¬ ¡¡ √ƒ √
≈ √√ ∆« ∆∆ »
… »»  À    ÃÕ ÃÃ Œœ Œ
– ŒŒ —
“ —— ”‘ ”
’ ”” ÷◊ ÷÷ ÿŸ ÿÿ ⁄€ ⁄
‹ ⁄⁄ ›ﬁ ›› ﬂ‡ ﬂﬂ ·‚ ·
„ ·· ‰
Â ‰‰ ÊÁ Ê
Ë ÊÊ ÈÍ ÈÈ Î
Ï ÎÎ ÌÓ ÌÌ Ô ÔÔ ÒÚ Ò
Û ÒÒ Ù
ı ÙÙ ˆ˜ ˆ
¯ ˆˆ ˘˙ ˘˘ ˚
¸ ˚˚ ˝˛ ˝˝ ˇÄ ˇˇ ÅÇ Å
É ÅÅ ÑÖ ÑÑ Üá ÜÜ àâ à
ä àà ã
å ãã çé ç
è çç êë êê í
ì íí îï îî ñó ññ òô ò
ö òò õ
ú õõ ùû ù
ü ùù †° †† ¢£ ¢¢ §• §
¶ §§ ß® ß
© ßß ™´ ™™ ¨
≠ ¨¨ ÆØ ÆÆ ∞
± ∞∞ ≤≥ ≤
¥ ≤≤ µ∂ µµ ∑
∏ ∑∑ π∫ ππ ªº ªª Ωæ Ω
ø ΩΩ ¿
¡ ¿¿ ¬√ ¬
ƒ ¬¬ ≈∆ ≈≈ «» «« …
  …… ÀÃ À
Õ ÀÀ Œœ ŒŒ –
— –– “” ““ ‘’ ‘‘ ÷◊ ÷
ÿ ÷÷ Ÿ
⁄ ŸŸ €‹ €
› €€ ﬁﬂ ﬁﬁ ‡· ‡‡ ‚„ ‚
‰ ‚‚ ÂÊ ÂÂ ÁË ÁÁ ÈÍ È
Î ÈÈ Ï
Ì ÏÏ ÓÔ Ó
 ÓÓ ÒÚ ÒÒ Û
Ù ÛÛ ıˆ ıı ˜¯ ˜˜ ˘˙ ˘
˚ ˘˘ ¸
˝ ¸¸ ˛ˇ ˛
Ä ˛˛ ÅÇ ÅÅ ÉÑ ÉÉ ÖÜ Ö
á ÖÖ àâ àà äã ää åç å
é åå è
ê èè ëí ë
ì ëë îï îî ñ
ó ññ òô òò öõ öö úù ú
û úú ü
† üü °¢ °
£ °° §• §§ ¶ß ¶¶ ®© ®
™ ®® ´¨ ´´ ≠Æ ≠≠ Ø
∞ ØØ ±≤ ±± ≥¥ ≥
µ ≥≥ ∂
∑ ∂∂ ∏π ∏
∫ ∏∏ ªº ªª Ω
æ ΩΩ ø¿ øø ¡¬ ¡¡ √ƒ √
≈ √√ ∆
« ∆∆ »… »
  »» ÀÃ Ã Ã Ã %Ã CÃ IÃ PÃ VÃ tÃ Ã ¶Ã ∆Ã ”Ã ÒÃ ñÃ ΩÃ »Ã ¡Ã ËÃ ØÃ ∫Ã ˚Ã ¨Ã ØÕ 3Õ dÕ èÕ ∂Õ ·Õ ÜÕ ´Õ ÷Õ ˇÕ ®Õ œÕ ˆÕ ùÕ ∫Õ €Õ ˛Õ °Õ »Õ ÎÕ íÕ ∑Õ –Õ ÛÕ ñÕ ΩŒ <Œ mŒ òŒ øŒ ÍŒ èŒ ¥Œ ﬂŒ àŒ ±Œ ÿŒ ˇŒ ¶Œ √Œ ‰Œ áŒ ™Œ —Œ ÙŒ õŒ ¿Œ ŸŒ ¸Œ üŒ ∆œ 	–     	 
             " $# &% (! *' +) - /, 0 21 43 6. 85 :7 ;1 =9 ?< @ BA DC F HG JI LE NK O QP S UT WV YR [X \Z ^M `] a cb ed g_ if kh lb nj pm q sr ut wI yv {x | ~} Ä ÇV ÑÅ ÜÉ áÖ âz ãà å éç êè íä îë ñì óç ôï õò ú ûI †ù ¢ü £ •§ ß¶ ©V ´® ≠™ Æ¨ ∞° ≤Ø ≥ µ¥ ∑∂ π± ª∏ Ω∫ æ¥ ¿º ¬ø √ ≈ƒ «∆ …I À» Õ  ŒC – “— ‘” ÷œ ÿ’ Ÿ◊ €Ã ›⁄ ﬁ ‡ﬂ ‚· ‰‹ Ê„ ËÂ Èﬂ ÎÁ ÌÍ Ó Ô ÚÒ ÙI ˆÛ ¯ı ˘V ˚˙ ˝˙ ˛¸ Ä˜ Çˇ É ÖÑ áÜ âÅ ãà çä éÑ êå íè ì ïî óñ ôI õò ùö ûV †ü ¢ü £° •ú ß§ ® ™© ¨´ Æ¶ ∞≠ ≤Ø ≥© µ± ∑¥ ∏C ∫ ºª æΩ ¿π ¬ø √ñ ≈ «∆ …» Àƒ Õ  ŒÃ –¡ “œ ” ’‘ ◊÷ Ÿ— €ÿ ›⁄ ﬁ‘ ‡‹ ‚ﬂ „t ÂΩ Á‰ ÈÊ ÍC Ï» ÓÎ Ì ÒÌ ÛÔ Ù ˆÚ ˜ı ˘Ë ˚¯ ¸ ˛˝ Äˇ Ç˙ ÑÅ ÜÉ á˝ âÖ ãà å éΩ êç íè ì ï» óî ôñ öñ úò ù üõ †û ¢ë §° • ß¶ ©® ´£ ≠™ Ø¨ ∞¶ ≤Æ ¥± µ∆ ∑Ω π∂ ª∏ º» æ ¿ø ¬¡ ƒΩ ∆√ «≈ …∫ À» Ã ŒÕ –œ “  ‘— ÷” ◊Õ Ÿ’ €ÿ ‹Ò ﬁΩ ‡› ‚ﬂ „» Â ÁÊ ÈË Î‰ ÌÍ ÓÏ · ÚÔ Û ıÙ ˜ˆ ˘Ò ˚¯ ˝˙ ˛Ù Ä¸ Çˇ ÉΩ ÖÑ áÑ à» äâ åâ ç¡ èã ëé í îê ïì óÜ ôñ ö úõ ûù †ò ¢ü §° •õ ß£ ©¶ ™¡ ¨ Æ≠ ∞Ø ≤± ¥´ ∂≥ ∑ π∏ ª∫ Ωµ øº ¡æ ¬∏ ƒ¿ ∆√ «Ë …C À¡ Õ  œÃ – “Œ ”— ’» ◊‘ ÿ ⁄Ÿ ‹€ ﬁ÷ ‡› ‚ﬂ „Ÿ Â· Á‰ Ët Í¡ ÏÈ ÓÎ ÔC ÒΩ Û ıÚ ˆÙ ¯Ì ˙˜ ˚ ˝¸ ˇ˛ Å˘ ÉÄ ÖÇ Ü¸ àÑ äá ãt ç¡ èå ëé íÒ î» ñì òï ôó õê ùö û †ü ¢° §ú ¶£ ®• ©ü ´ß ≠™ Æ ∞¡ ≤Ø ¥± µC ∑ π∏ ª∫ Ω∂ øº ¿æ ¬≥ ƒ¡ ≈ «∆ …» À√ Õ  œÃ –∆ “Œ ‘— ’ ◊¡ Ÿ÷ €ÿ ‹V ﬁ» ‡› ‚ﬂ „· Â⁄ Á‰ Ë ÍÈ ÏÎ ÓÊ Ì ÚÔ ÛÈ ıÒ ˜Ù ¯ ˙˘ ¸˚ ˛¡ Ä˝ Çˇ É» ÖË áÑ âÜ äà åÅ éã è ëê ìí ïç óî ôñ öê úò ûõ üV °¡ £† •¢ ¶ ®§ © ´™ ≠¨ ØÆ ±ß ≥∞ ¥ ∂µ ∏∑ ∫≤ ºπ æª øµ ¡Ω √¿ ƒ¡ ∆Ø »«  ≈ Ã… Õ œŒ —– ”À ’“ ◊‘ ÿŒ ⁄÷ ‹Ÿ ›t ﬂØ ·ﬁ „‡ ‰Ò Ê» ËÂ ÍÁ ÎÈ Ì‚ ÔÏ  ÚÒ ÙÛ ˆÓ ¯ı ˙˜ ˚Ò ˝˘ ˇ¸ Ä ÇØ ÑÅ ÜÉ áC â∫ ãà çä éå êÖ íè ì ïî óñ ôë õò ùö ûî †ú ¢ü £ •Ø ß§ ©¶ ™Ò ¨ Æ≠ ∞Ø ≤´ ¥± µ≥ ∑® π∂ ∫ ºª æΩ ¿∏ ¬ø ƒ¡ ≈ª «√ …∆   ““ À ——ä ““ ä° ““ °• ““ •h ““ hñ ““ ñ¨ ““ ¨” ““ ”æ ““ æﬂ ““ ﬂÔ ““ Ô˜ ““ ˜¡ ““ ¡Ç ““ Ç˙ ““ ˙É ““ É∫ ““ ∫ª ““ ª7 ““ 7ì ““ ìØ ““ ØÂ ““ ÂÃ ““ Ã‘ ““ ‘ö ““ ö⁄ ““ ⁄ —— 
” ‘
‘ ﬂ
’ ©
÷ ∆	◊ A	ÿ T
Ÿ —
⁄ ü
€ ¶
‹ Ù	› 	ﬁ r
ﬂ ª
‡ Ÿ
· ∆
‚ õ
„ ê
‰ µ
Â ª
Ê ∏
Á ≠	Ë b
È î
Í §	Î #
Ï Ô
Ì ˝	Ó 7	Ó h
Ó ì
Ó ∫
Ó Â
Ó ä
Ó Ø
Ó ⁄
Ó É
Ó ¨
Ó ”
Ó ˙
Ó °
Ó æ
Ó ﬂ
Ó Ç
Ó •
Ó Ã
Ó Ô
Ó ñ
Ó ª
Ó ‘
Ó ˜
Ó ö
Ó ¡	Ô G
 ŒÒ 
Ú Õ
Û ø
Ù È	ı 	ˆ 
˜ ç
¯ Ê
˘ ≠
˙ ™	˚ 1
¸ î
˝ ƒ˛ 
˛ ,˛ ]˛ à˛ Ø˛ ⁄˛ ˇ˛ §˛ œ˛ ¯˛ °˛ »˛ Ô˛ ñ˛ ≥˛ ‘˛ ˜˛ ö˛ ¡˛ ‰˛ ã˛ ∞˛ …˛ Ï˛ è˛ ∂
ˇ ¥
Ä Ñ	Å }
Ç ˘
É Ò
Ñ ¸
Ö ∏	Ü 	á "
ratt6_kernel"
_Z13get_global_idj"
	_Z4fminff*ó
shoc-1.1.5-S3D-ratt6_kernel.clu
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

wgsize
Ä

devmap_label
 
 
transfer_bytes_log1p
íÛéA