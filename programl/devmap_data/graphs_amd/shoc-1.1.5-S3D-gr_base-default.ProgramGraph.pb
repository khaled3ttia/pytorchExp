

[external]
KcallBC
A
	full_text4
2
0%7 = tail call i64 @_Z13get_global_idj(i32 0) #2
XgetelementptrBG
E
	full_text8
6
4%8 = getelementptr inbounds float, float* %1, i64 %7
"i64B

	full_text


i64 %7
HloadB@
>
	full_text1
/
-%9 = load float, float* %8, align 4, !tbaa !8
(float*B

	full_text

	float* %8
2fmulB*
(
	full_text

%10 = fmul float %9, %4
&floatB

	full_text


float %9
YgetelementptrBH
F
	full_text9
7
5%11 = getelementptr inbounds float, float* %0, i64 %7
"i64B

	full_text


i64 %7
JloadBB
@
	full_text3
1
/%12 = load float, float* %11, align 4, !tbaa !8
)float*B

	full_text


float* %11
3fmulB+
)
	full_text

%13 = fmul float %12, %5
'floatB

	full_text

	float %12
YgetelementptrBH
F
	full_text9
7
5%14 = getelementptr inbounds float, float* %2, i64 %7
"i64B

	full_text


i64 %7
JloadBB
@
	full_text3
1
/%15 = load float, float* %14, align 4, !tbaa !8
)float*B

	full_text


float* %14
CfmulB;
9
	full_text,
*
(%16 = fmul float %15, 0x3FDFBF39E0000000
'floatB

	full_text

	float %15
YgetelementptrBH
F
	full_text9
7
5%17 = getelementptr inbounds float, float* %3, i64 %7
"i64B

	full_text


i64 %7
JstoreBA
?
	full_text2
0
.store float %16, float* %17, align 4, !tbaa !8
'floatB

	full_text

	float %16
)float*B

	full_text


float* %17
=faddB5
3
	full_text&
$
"%18 = fadd float %16, 0.000000e+00
'floatB

	full_text

	float %16
-addB&
$
	full_text

%19 = add i64 %7, 8
"i64B

	full_text


i64 %7
ZgetelementptrBI
G
	full_text:
8
6%20 = getelementptr inbounds float, float* %2, i64 %19
#i64B

	full_text
	
i64 %19
JloadBB
@
	full_text3
1
/%21 = load float, float* %20, align 4, !tbaa !8
)float*B

	full_text


float* %20
CfmulB;
9
	full_text,
*
(%22 = fmul float %21, 0x3FEFBF39E0000000
'floatB

	full_text

	float %21
ZgetelementptrBI
G
	full_text:
8
6%23 = getelementptr inbounds float, float* %3, i64 %19
#i64B

	full_text
	
i64 %19
JstoreBA
?
	full_text2
0
.store float %22, float* %23, align 4, !tbaa !8
'floatB

	full_text

	float %22
)float*B

	full_text


float* %23
4faddB,
*
	full_text

%24 = fadd float %18, %22
'floatB

	full_text

	float %18
'floatB

	full_text

	float %22
.addB'
%
	full_text

%25 = add i64 %7, 16
"i64B

	full_text


i64 %7
ZgetelementptrBI
G
	full_text:
8
6%26 = getelementptr inbounds float, float* %2, i64 %25
#i64B

	full_text
	
i64 %25
JloadBB
@
	full_text3
1
/%27 = load float, float* %26, align 4, !tbaa !8
)float*B

	full_text


float* %26
CfmulB;
9
	full_text,
*
(%28 = fmul float %27, 0x3FB0002760000000
'floatB

	full_text

	float %27
ZgetelementptrBI
G
	full_text:
8
6%29 = getelementptr inbounds float, float* %3, i64 %25
#i64B

	full_text
	
i64 %25
JstoreBA
?
	full_text2
0
.store float %28, float* %29, align 4, !tbaa !8
'floatB

	full_text

	float %28
)float*B

	full_text


float* %29
4faddB,
*
	full_text

%30 = fadd float %24, %28
'floatB

	full_text

	float %24
'floatB

	full_text

	float %28
.addB'
%
	full_text

%31 = add i64 %7, 24
"i64B

	full_text


i64 %7
ZgetelementptrBI
G
	full_text:
8
6%32 = getelementptr inbounds float, float* %2, i64 %31
#i64B

	full_text
	
i64 %31
JloadBB
@
	full_text3
1
/%33 = load float, float* %32, align 4, !tbaa !8
)float*B

	full_text


float* %32
CfmulB;
9
	full_text,
*
(%34 = fmul float %33, 0x3FA0002740000000
'floatB

	full_text

	float %33
ZgetelementptrBI
G
	full_text:
8
6%35 = getelementptr inbounds float, float* %3, i64 %31
#i64B

	full_text
	
i64 %31
JstoreBA
?
	full_text2
0
.store float %34, float* %35, align 4, !tbaa !8
'floatB

	full_text

	float %34
)float*B

	full_text


float* %35
4faddB,
*
	full_text

%36 = fadd float %30, %34
'floatB

	full_text

	float %30
'floatB

	full_text

	float %34
.addB'
%
	full_text

%37 = add i64 %7, 32
"i64B

	full_text


i64 %7
ZgetelementptrBI
G
	full_text:
8
6%38 = getelementptr inbounds float, float* %2, i64 %37
#i64B

	full_text
	
i64 %37
JloadBB
@
	full_text3
1
/%39 = load float, float* %38, align 4, !tbaa !8
)float*B

	full_text


float* %38
CfmulB;
9
	full_text,
*
(%40 = fmul float %39, 0x3FAE1AC6C0000000
'floatB

	full_text

	float %39
ZgetelementptrBI
G
	full_text:
8
6%41 = getelementptr inbounds float, float* %3, i64 %37
#i64B

	full_text
	
i64 %37
JstoreBA
?
	full_text2
0
.store float %40, float* %41, align 4, !tbaa !8
'floatB

	full_text

	float %40
)float*B

	full_text


float* %41
4faddB,
*
	full_text

%42 = fadd float %36, %40
'floatB

	full_text

	float %36
'floatB

	full_text

	float %40
.addB'
%
	full_text

%43 = add i64 %7, 40
"i64B

	full_text


i64 %7
ZgetelementptrBI
G
	full_text:
8
6%44 = getelementptr inbounds float, float* %2, i64 %43
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
CfmulB;
9
	full_text,
*
(%46 = fmul float %45, 0x3FAC6B93C0000000
'floatB

	full_text

	float %45
ZgetelementptrBI
G
	full_text:
8
6%47 = getelementptr inbounds float, float* %3, i64 %43
#i64B

	full_text
	
i64 %43
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
4faddB,
*
	full_text

%48 = fadd float %42, %46
'floatB

	full_text

	float %42
'floatB

	full_text

	float %46
.addB'
%
	full_text

%49 = add i64 %7, 48
"i64B

	full_text


i64 %7
ZgetelementptrBI
G
	full_text:
8
6%50 = getelementptr inbounds float, float* %2, i64 %49
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
CfmulB;
9
	full_text,
*
(%52 = fmul float %51, 0x3F9F0620C0000000
'floatB

	full_text

	float %51
ZgetelementptrBI
G
	full_text:
8
6%53 = getelementptr inbounds float, float* %3, i64 %49
#i64B

	full_text
	
i64 %49
JstoreBA
?
	full_text2
0
.store float %52, float* %53, align 4, !tbaa !8
'floatB

	full_text

	float %52
)float*B

	full_text


float* %53
4faddB,
*
	full_text

%54 = fadd float %48, %52
'floatB

	full_text

	float %48
'floatB

	full_text

	float %52
.addB'
%
	full_text

%55 = add i64 %7, 56
"i64B

	full_text


i64 %7
ZgetelementptrBI
G
	full_text:
8
6%56 = getelementptr inbounds float, float* %2, i64 %55
#i64B

	full_text
	
i64 %55
JloadBB
@
	full_text3
1
/%57 = load float, float* %56, align 4, !tbaa !8
)float*B

	full_text


float* %56
CfmulB;
9
	full_text,
*
(%58 = fmul float %57, 0x3F9E1AC6C0000000
'floatB

	full_text

	float %57
ZgetelementptrBI
G
	full_text:
8
6%59 = getelementptr inbounds float, float* %3, i64 %55
#i64B

	full_text
	
i64 %55
JstoreBA
?
	full_text2
0
.store float %58, float* %59, align 4, !tbaa !8
'floatB

	full_text

	float %58
)float*B

	full_text


float* %59
4faddB,
*
	full_text

%60 = fadd float %54, %58
'floatB

	full_text

	float %54
'floatB

	full_text

	float %58
.addB'
%
	full_text

%61 = add i64 %7, 64
"i64B

	full_text


i64 %7
ZgetelementptrBI
G
	full_text:
8
6%62 = getelementptr inbounds float, float* %2, i64 %61
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
CfmulB;
9
	full_text,
*
(%64 = fmul float %63, 0x3FB106E0E0000000
'floatB

	full_text

	float %63
ZgetelementptrBI
G
	full_text:
8
6%65 = getelementptr inbounds float, float* %3, i64 %61
#i64B

	full_text
	
i64 %61
JstoreBA
?
	full_text2
0
.store float %64, float* %65, align 4, !tbaa !8
'floatB

	full_text

	float %64
)float*B

	full_text


float* %65
4faddB,
*
	full_text

%66 = fadd float %60, %64
'floatB

	full_text

	float %60
'floatB

	full_text

	float %64
.addB'
%
	full_text

%67 = add i64 %7, 72
"i64B

	full_text


i64 %7
ZgetelementptrBI
G
	full_text:
8
6%68 = getelementptr inbounds float, float* %2, i64 %67
#i64B

	full_text
	
i64 %67
JloadBB
@
	full_text3
1
/%69 = load float, float* %68, align 4, !tbaa !8
)float*B

	full_text


float* %68
CfmulB;
9
	full_text,
*
(%70 = fmul float %69, 0x3FAFEA0720000000
'floatB

	full_text

	float %69
ZgetelementptrBI
G
	full_text:
8
6%71 = getelementptr inbounds float, float* %3, i64 %67
#i64B

	full_text
	
i64 %67
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
4faddB,
*
	full_text

%72 = fadd float %66, %70
'floatB

	full_text

	float %66
'floatB

	full_text

	float %70
.addB'
%
	full_text

%73 = add i64 %7, 80
"i64B

	full_text


i64 %7
ZgetelementptrBI
G
	full_text:
8
6%74 = getelementptr inbounds float, float* %2, i64 %73
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
CfmulB;
9
	full_text,
*
(%76 = fmul float %75, 0x3FA2476140000000
'floatB

	full_text

	float %75
ZgetelementptrBI
G
	full_text:
8
6%77 = getelementptr inbounds float, float* %3, i64 %73
#i64B

	full_text
	
i64 %73
JstoreBA
?
	full_text2
0
.store float %76, float* %77, align 4, !tbaa !8
'floatB

	full_text

	float %76
)float*B

	full_text


float* %77
4faddB,
*
	full_text

%78 = fadd float %72, %76
'floatB

	full_text

	float %72
'floatB

	full_text

	float %76
.addB'
%
	full_text

%79 = add i64 %7, 88
"i64B

	full_text


i64 %7
ZgetelementptrBI
G
	full_text:
8
6%80 = getelementptr inbounds float, float* %2, i64 %79
#i64B

	full_text
	
i64 %79
JloadBB
@
	full_text3
1
/%81 = load float, float* %80, align 4, !tbaa !8
)float*B

	full_text


float* %80
CfmulB;
9
	full_text,
*
(%82 = fmul float %81, 0x3F974478A0000000
'floatB

	full_text

	float %81
ZgetelementptrBI
G
	full_text:
8
6%83 = getelementptr inbounds float, float* %3, i64 %79
#i64B

	full_text
	
i64 %79
JstoreBA
?
	full_text2
0
.store float %82, float* %83, align 4, !tbaa !8
'floatB

	full_text

	float %82
)float*B

	full_text


float* %83
4faddB,
*
	full_text

%84 = fadd float %78, %82
'floatB

	full_text

	float %78
'floatB

	full_text

	float %82
.addB'
%
	full_text

%85 = add i64 %7, 96
"i64B

	full_text


i64 %7
ZgetelementptrBI
G
	full_text:
8
6%86 = getelementptr inbounds float, float* %2, i64 %85
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
CfmulB;
9
	full_text,
*
(%88 = fmul float %87, 0x3FA10D3640000000
'floatB

	full_text

	float %87
ZgetelementptrBI
G
	full_text:
8
6%89 = getelementptr inbounds float, float* %3, i64 %85
#i64B

	full_text
	
i64 %85
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
4faddB,
*
	full_text

%90 = fadd float %84, %88
'floatB

	full_text

	float %84
'floatB

	full_text

	float %88
/addB(
&
	full_text

%91 = add i64 %7, 104
"i64B

	full_text


i64 %7
ZgetelementptrBI
G
	full_text:
8
6%92 = getelementptr inbounds float, float* %2, i64 %91
#i64B

	full_text
	
i64 %91
JloadBB
@
	full_text3
1
/%93 = load float, float* %92, align 4, !tbaa !8
)float*B

	full_text


float* %92
CfmulB;
9
	full_text,
*
(%94 = fmul float %93, 0x3FA3A9D3C0000000
'floatB

	full_text

	float %93
ZgetelementptrBI
G
	full_text:
8
6%95 = getelementptr inbounds float, float* %3, i64 %91
#i64B

	full_text
	
i64 %91
JstoreBA
?
	full_text2
0
.store float %94, float* %95, align 4, !tbaa !8
'floatB

	full_text

	float %94
)float*B

	full_text


float* %95
4faddB,
*
	full_text

%96 = fadd float %90, %94
'floatB

	full_text

	float %90
'floatB

	full_text

	float %94
/addB(
&
	full_text

%97 = add i64 %7, 112
"i64B

	full_text


i64 %7
ZgetelementptrBI
G
	full_text:
8
6%98 = getelementptr inbounds float, float* %2, i64 %97
#i64B

	full_text
	
i64 %97
JloadBB
@
	full_text3
1
/%99 = load float, float* %98, align 4, !tbaa !8
)float*B

	full_text


float* %98
DfmulB<
:
	full_text-
+
)%100 = fmul float %99, 0x3FA2401A20000000
'floatB

	full_text

	float %99
[getelementptrBJ
H
	full_text;
9
7%101 = getelementptr inbounds float, float* %3, i64 %97
#i64B

	full_text
	
i64 %97
LstoreBC
A
	full_text4
2
0store float %100, float* %101, align 4, !tbaa !8
(floatB

	full_text


float %100
*float*B

	full_text

float* %101
6faddB.
,
	full_text

%102 = fadd float %96, %100
'floatB

	full_text

	float %96
(floatB

	full_text


float %100
0addB)
'
	full_text

%103 = add i64 %7, 120
"i64B

	full_text


i64 %7
\getelementptrBK
I
	full_text<
:
8%104 = getelementptr inbounds float, float* %2, i64 %103
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
EfmulB=
;
	full_text.
,
*%106 = fmul float %105, 0x3FA106E0E0000000
(floatB

	full_text


float %105
\getelementptrBK
I
	full_text<
:
8%107 = getelementptr inbounds float, float* %3, i64 %103
$i64B

	full_text


i64 %103
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
7faddB/
-
	full_text 

%108 = fadd float %102, %106
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
%109 = add i64 %7, 128
"i64B

	full_text


i64 %7
\getelementptrBK
I
	full_text<
:
8%110 = getelementptr inbounds float, float* %2, i64 %109
$i64B

	full_text


i64 %109
LloadBD
B
	full_text5
3
1%111 = load float, float* %110, align 4, !tbaa !8
*float*B

	full_text

float* %110
EfmulB=
;
	full_text.
,
*%112 = fmul float %111, 0x3F98F521E0000000
(floatB

	full_text


float %111
\getelementptrBK
I
	full_text<
:
8%113 = getelementptr inbounds float, float* %3, i64 %109
$i64B

	full_text


i64 %109
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
7faddB/
-
	full_text 

%114 = fadd float %108, %112
(floatB

	full_text


float %108
(floatB

	full_text


float %112
0addB)
'
	full_text

%115 = add i64 %7, 136
"i64B

	full_text


i64 %7
\getelementptrBK
I
	full_text<
:
8%116 = getelementptr inbounds float, float* %2, i64 %115
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
EfmulB=
;
	full_text.
,
*%118 = fmul float %117, 0x3F985BEF60000000
(floatB

	full_text


float %117
\getelementptrBK
I
	full_text<
:
8%119 = getelementptr inbounds float, float* %3, i64 %115
$i64B

	full_text


i64 %115
LstoreBC
A
	full_text4
2
0store float %118, float* %119, align 4, !tbaa !8
(floatB

	full_text


float %118
*float*B

	full_text

float* %119
7faddB/
-
	full_text 

%120 = fadd float %114, %118
(floatB

	full_text


float %114
(floatB

	full_text


float %118
0addB)
'
	full_text

%121 = add i64 %7, 144
"i64B

	full_text


i64 %7
\getelementptrBK
I
	full_text<
:
8%122 = getelementptr inbounds float, float* %2, i64 %121
$i64B

	full_text


i64 %121
LloadBD
B
	full_text5
3
1%123 = load float, float* %122, align 4, !tbaa !8
*float*B

	full_text

float* %122
EfmulB=
;
	full_text.
,
*%124 = fmul float %123, 0x3F973E9260000000
(floatB

	full_text


float %123
\getelementptrBK
I
	full_text<
:
8%125 = getelementptr inbounds float, float* %3, i64 %121
$i64B

	full_text


i64 %121
LstoreBC
A
	full_text4
2
0store float %124, float* %125, align 4, !tbaa !8
(floatB

	full_text


float %124
*float*B

	full_text

float* %125
7faddB/
-
	full_text 

%126 = fadd float %120, %124
(floatB

	full_text


float %120
(floatB

	full_text


float %124
0addB)
'
	full_text

%127 = add i64 %7, 152
"i64B

	full_text


i64 %7
\getelementptrBK
I
	full_text<
:
8%128 = getelementptr inbounds float, float* %2, i64 %127
$i64B

	full_text


i64 %127
LloadBD
B
	full_text5
3
1%129 = load float, float* %128, align 4, !tbaa !8
*float*B

	full_text

float* %128
EfmulB=
;
	full_text.
,
*%130 = fmul float %129, 0x3F98EE5880000000
(floatB

	full_text


float %129
\getelementptrBK
I
	full_text<
:
8%131 = getelementptr inbounds float, float* %3, i64 %127
$i64B

	full_text


i64 %127
LstoreBC
A
	full_text4
2
0store float %130, float* %131, align 4, !tbaa !8
(floatB

	full_text


float %130
*float*B

	full_text

float* %131
7faddB/
-
	full_text 

%132 = fadd float %126, %130
(floatB

	full_text


float %126
(floatB

	full_text


float %130
0addB)
'
	full_text

%133 = add i64 %7, 160
"i64B

	full_text


i64 %7
\getelementptrBK
I
	full_text<
:
8%134 = getelementptr inbounds float, float* %2, i64 %133
$i64B

	full_text


i64 %133
LloadBD
B
	full_text5
3
1%135 = load float, float* %134, align 4, !tbaa !8
*float*B

	full_text

float* %134
EfmulB=
;
	full_text.
,
*%136 = fmul float %135, 0x3F98557840000000
(floatB

	full_text


float %135
\getelementptrBK
I
	full_text<
:
8%137 = getelementptr inbounds float, float* %3, i64 %133
$i64B

	full_text


i64 %133
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
7faddB/
-
	full_text 

%138 = fadd float %132, %136
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
%139 = add i64 %7, 168
"i64B

	full_text


i64 %7
\getelementptrBK
I
	full_text<
:
8%140 = getelementptr inbounds float, float* %2, i64 %139
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
EfmulB=
;
	full_text.
,
*%142 = fmul float %141, 0x3FA246E760000000
(floatB

	full_text


float %141
\getelementptrBK
I
	full_text<
:
8%143 = getelementptr inbounds float, float* %3, i64 %139
$i64B

	full_text


i64 %139
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
7faddB/
-
	full_text 

%144 = fadd float %138, %142
(floatB

	full_text


float %138
(floatB

	full_text


float %142
6fmulB.
,
	full_text

%145 = fmul float %10, %144
'floatB

	full_text

	float %10
(floatB

	full_text


float %144
EfmulB=
;
	full_text.
,
*%146 = fmul float %145, 0x4193D2C640000000
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
6fmulB.
,
	full_text

%148 = fmul float %13, %147
'floatB

	full_text

	float %13
(floatB

	full_text


float %147
&brB 

	full_text

br label %150
$ret8B

	full_text


ret void
Dphi8B;
9
	full_text,
*
(%151 = phi i64 [ 1, %6 ], [ %159, %150 ]
&i648B

	full_text


i64 %159
2shl8B)
'
	full_text

%152 = shl i64 %151, 3
&i648B

	full_text


i64 %151
7add8B.
,
	full_text

%153 = add nsw i64 %152, -8
&i648B

	full_text


i64 %152
3add8B*
(
	full_text

%154 = add i64 %7, %153
$i648B

	full_text


i64 %7
&i648B

	full_text


i64 %153
^getelementptr8BK
I
	full_text<
:
8%155 = getelementptr inbounds float, float* %3, i64 %154
&i648B

	full_text


i64 %154
Nload8BD
B
	full_text5
3
1%156 = load float, float* %155, align 4, !tbaa !8
,float*8B

	full_text

float* %155
gcall8B]
[
	full_textN
L
J%157 = tail call float @_Z4fmaxff(float %156, float 0x3810000000000000) #2
*float8B

	full_text


float %156
9fmul8B/
-
	full_text 

%158 = fmul float %148, %157
*float8B

	full_text


float %148
*float8B

	full_text


float %157
Nstore8BC
A
	full_text4
2
0store float %158, float* %155, align 4, !tbaa !8
*float8B

	full_text


float %158
,float*8B

	full_text

float* %155
:add8B1
/
	full_text"
 
%159 = add nuw nsw i64 %151, 1
&i648B

	full_text


i64 %151
8icmp8B.
,
	full_text

%160 = icmp eq i64 %159, 23
&i648B

	full_text


i64 %159
=br8B5
3
	full_text&
$
"br i1 %160, label %149, label %150
$i18B

	full_text
	
i1 %160
*float*8B

	full_text

	float* %2
*float*8B

	full_text

	float* %3
*float*8B

	full_text

	float* %1
*float*8B

	full_text

	float* %0
(float8B

	full_text


float %4
(float8B

	full_text


float %5
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
%i648B

	full_text
	
i64 128
8float8B+
)
	full_text

float 0x3F98557840000000
#i648B

	full_text	

i64 1
#i328B

	full_text	

i32 0
2float8B%
#
	full_text

float 0.000000e+00
$i648B

	full_text


i64 72
%i648B

	full_text
	
i64 168
8float8B+
)
	full_text

float 0x3810000000000000
8float8B+
)
	full_text

float 0x3FDFBF39E0000000
$i648B

	full_text


i64 32
8float8B+
)
	full_text

float 0x3FA0002740000000
8float8B+
)
	full_text

float 0x3F9E1AC6C0000000
8float8B+
)
	full_text

float 0x3FB106E0E0000000
$i648B

	full_text


i64 24
%i648B

	full_text
	
i64 104
8float8B+
)
	full_text

float 0x3FAFEA0720000000
8float8B+
)
	full_text

float 0x3FA2401A20000000
8float8B+
)
	full_text

float 0x3F98F521E0000000
%i648B

	full_text
	
i64 152
8float8B+
)
	full_text

float 0x3FAE1AC6C0000000
8float8B+
)
	full_text

float 0x3FA3A9D3C0000000
8float8B+
)
	full_text

float 0x4193D2C640000000
8float8B+
)
	full_text

float 0x3FA2476140000000
#i648B

	full_text	

i64 8
8float8B+
)
	full_text

float 0x3FAC6B93C0000000
8float8B+
)
	full_text

float 0x3FEFBF39E0000000
$i648B

	full_text


i64 96
%i648B

	full_text
	
i64 112
8float8B+
)
	full_text

float 0x3F973E9260000000
8float8B+
)
	full_text

float 0x3F98EE5880000000
8float8B+
)
	full_text

float 0x3FB0002760000000
8float8B+
)
	full_text

float 0x3F974478A0000000
$i648B

	full_text


i64 88
%i648B

	full_text
	
i64 136
$i648B

	full_text


i64 80
%i648B

	full_text
	
i64 160
8float8B+
)
	full_text

float 0x3FA246E760000000
$i648B

	full_text


i64 56
#i648B

	full_text	

i64 3
$i648B

	full_text


i64 -8
$i648B

	full_text


i64 23
$i648B

	full_text


i64 48
%i648B

	full_text
	
i64 144
8float8B+
)
	full_text

float 0x3F9F0620C0000000
$i648B

	full_text


i64 40
$i648B

	full_text


i64 64
8float8B+
)
	full_text

float 0x3FA10D3640000000
%i648B

	full_text
	
i64 120
2float8B%
#
	full_text

float 1.000000e+00
$i648B

	full_text


i64 16
8float8B+
)
	full_text

float 0x3FA106E0E0000000
8float8B+
)
	full_text

float 0x3F985BEF60000000       	  
 

                       !" !! #$ ## %& %' %% () (* (( +, ++ -. -- /0 // 12 11 34 33 56 57 55 89 8: 88 ;< ;; => == ?@ ?? AB AA CD CC EF EG EE HI HJ HH KL KK MN MM OP OO QR QQ ST SS UV UW UU XY XZ XX [\ [[ ]^ ]] _` __ ab aa cd cc ef eg ee hi hj hh kl kk mn mm op oo qr qq st ss uv uw uu xy xz xx {| {{ }~ }} Ä  ÅÇ ÅÅ É
Ñ ÉÉ ÖÜ Ö
á ÖÖ àâ à
ä àà ãå ãã ç
é çç èê èè ëí ëë ì
î ìì ïñ ï
ó ïï òô ò
ö òò õú õõ ù
û ùù ü† üü °¢ °° £
§ ££ •¶ •
ß •• ®© ®
™ ®® ´¨ ´´ ≠
Æ ≠≠ Ø∞ ØØ ±≤ ±± ≥
¥ ≥≥ µ∂ µ
∑ µµ ∏π ∏
∫ ∏∏ ªº ªª Ω
æ ΩΩ ø¿ øø ¡¬ ¡¡ √
ƒ √√ ≈∆ ≈
« ≈≈ »… »
  »» ÀÃ ÀÀ Õ
Œ ÕÕ œ– œœ —“ —— ”
‘ ”” ’÷ ’
◊ ’’ ÿŸ ÿ
⁄ ÿÿ €‹ €€ ›
ﬁ ›› ﬂ‡ ﬂﬂ ·‚ ·· „
‰ „„ ÂÊ Â
Á ÂÂ ËÈ Ë
Í ËË ÎÏ ÎÎ Ì
Ó ÌÌ Ô ÔÔ ÒÚ ÒÒ Û
Ù ÛÛ ıˆ ı
˜ ıı ¯˘ ¯
˙ ¯¯ ˚¸ ˚˚ ˝
˛ ˝˝ ˇÄ ˇˇ ÅÇ ÅÅ É
Ñ ÉÉ ÖÜ Ö
á ÖÖ àâ à
ä àà ãå ãã ç
é çç èê èè ëí ëë ì
î ìì ïñ ï
ó ïï òô ò
ö òò õú õõ ù
û ùù ü† üü °¢ °° £
§ ££ •¶ •
ß •• ®© ®
™ ®® ´¨ ´´ ≠
Æ ≠≠ Ø∞ ØØ ±≤ ±± ≥
¥ ≥≥ µ∂ µ
∑ µµ ∏π ∏
∫ ∏∏ ªº ªª Ω
æ ΩΩ ø¿ øø ¡¬ ¡¡ √
ƒ √√ ≈∆ ≈
« ≈≈ »… »
  »» ÀÃ ÀÀ Õ
Œ ÕÕ œ– œœ —“ —— ”
‘ ”” ’÷ ’
◊ ’’ ÿŸ ÿ
⁄ ÿÿ €‹ €€ ›
ﬁ ›› ﬂ‡ ﬂﬂ ·‚ ·· „
‰ „„ ÂÊ Â
Á ÂÂ ËÈ Ë
Í ËË ÎÏ Î
Ì ÎÎ ÓÔ ÓÓ 
Ò  ÚÛ Ú
Ù ÚÚ ı
¯ ˜˜ ˘˙ ˘˘ ˚¸ ˚˚ ˝˛ ˝
ˇ ˝˝ Ä
Å ÄÄ ÇÉ ÇÇ ÑÖ ÑÑ Üá Ü
à ÜÜ âä â
ã ââ åç åå éè éé êë êí í í -í =í Mí ]í mí }í çí ùí ≠í Ωí Õí ›í Ìí ˝í çí ùí ≠í Ωí Õí ›ì ì #ì 3ì Cì Sì cì sì Éì ìì £ì ≥ì √ì ”ì „ì Ûì Éì ìì £ì ≥ì √ì ”ì „ì Äî ï 	ñ 	ó     	 
             " $! &# ' )! * ,+ .- 0/ 2+ 41 63 7( 91 : <; >= @? B; DA FC G8 IA J LK NM PO RK TQ VS WH YQ Z \[ ^] `_ b[ da fc gX ia j lk nm po rk tq vs wh yq z |{ ~} Ä Ç{ ÑÅ ÜÉ áx âÅ ä åã éç êè íã îë ñì óà ôë ö úõ ûù †ü ¢õ §° ¶£ ßò ©° ™ ¨´ Æ≠ ∞Ø ≤´ ¥± ∂≥ ∑® π± ∫ ºª æΩ ¿ø ¬ª ƒ¡ ∆√ «∏ …¡   ÃÀ ŒÕ –œ “À ‘— ÷” ◊» Ÿ— ⁄ ‹€ ﬁ› ‡ﬂ ‚€ ‰· Ê„ Áÿ È· Í ÏÎ ÓÌ Ô ÚÎ ÙÒ ˆÛ ˜Ë ˘Ò ˙ ¸˚ ˛˝ Äˇ Ç˚ ÑÅ ÜÉ á¯ âÅ ä åã éç êè íã îë ñì óà ôë ö úõ ûù †ü ¢õ §° ¶£ ßò ©° ™ ¨´ Æ≠ ∞Ø ≤´ ¥± ∂≥ ∑® π± ∫ ºª æΩ ¿ø ¬ª ƒ¡ ∆√ «∏ …¡   ÃÀ ŒÕ –œ “À ‘— ÷” ◊» Ÿ— ⁄ ‹€ ﬁ› ‡ﬂ ‚€ ‰· Ê„ Áÿ È· Í ÏË ÌÎ ÔÓ Ò Û Ùå ¯˜ ˙˘ ¸ ˛˚ ˇ˝ ÅÄ ÉÇ ÖÚ áÑ àÜ äÄ ã˜ çå èé ëı ˜ê ˆê ˜ ôô ˆ òò òò Ñ ôô Ñ
ö ã
õ —ú ˜
ú åù 	û 
ü õ
† €
° Ñ	¢ 	£ K	§ A
• Å
¶ ë	ß ;
® €
© °
™ Ò
´ ë
¨ ª	≠ Q
Æ ·
Ø Ó
∞ ±	± 	≤ a	≥ !
¥ À
µ Î
∂ ±
∑ ¡	∏ 1
π ¡
∫ ª
ª õ
º ´
Ω À
æ ·	ø {
¿ ˘
¡ ˚
¬ é	√ k
ƒ ´	≈ q	∆ [
« ã
» —
… ˚  	À +
Ã Å
Õ °"	
gr_base"
_Z13get_global_idj"
	_Z4fmaxff*í
shoc-1.1.5-S3D-gr_base.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282Å

wgsize
Ä

wgsize_log1p
íÛéA

transfer_bytes
à¢ª
 
transfer_bytes_log1p
íÛéA

devmap_label
 