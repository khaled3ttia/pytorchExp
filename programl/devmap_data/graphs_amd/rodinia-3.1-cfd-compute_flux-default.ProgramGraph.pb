

[external]
AallocaB7
5
	full_text(
&
$%11 = alloca %struct.FLOAT3, align 8
AallocaB7
5
	full_text(
&
$%12 = alloca %struct.FLOAT3, align 4
AallocaB7
5
	full_text(
&
$%13 = alloca %struct.FLOAT3, align 4
AallocaB7
5
	full_text(
&
$%14 = alloca %struct.FLOAT3, align 4
AallocaB7
5
	full_text(
&
$%15 = alloca %struct.FLOAT3, align 4
AallocaB7
5
	full_text(
&
$%16 = alloca %struct.FLOAT3, align 8
AallocaB7
5
	full_text(
&
$%17 = alloca %struct.FLOAT3, align 4
AallocaB7
5
	full_text(
&
$%18 = alloca %struct.FLOAT3, align 4
AallocaB7
5
	full_text(
&
$%19 = alloca %struct.FLOAT3, align 4
AallocaB7
5
	full_text(
&
$%20 = alloca %struct.FLOAT3, align 4
LcallBD
B
	full_text5
3
1%21 = tail call i64 @_Z13get_global_idj(i32 0) #5
6truncB-
+
	full_text

%22 = trunc i64 %21 to i32
#i64B

	full_text
	
i64 %21
5icmpB-
+
	full_text

%23 = icmp slt i32 %22, %9
#i32B

	full_text
	
i32 %22
9brB3
1
	full_text$
"
 br i1 %23, label %24, label %320
!i1B

	full_text


i1 %23
1shl8B(
&
	full_text

%25 = shl i64 %21, 32
%i648B

	full_text
	
i64 %21
9ashr8B/
-
	full_text 

%26 = ashr exact i64 %25, 32
%i648B

	full_text
	
i64 %25
\getelementptr8BI
G
	full_text:
8
6%27 = getelementptr inbounds float, float* %2, i64 %26
%i648B

	full_text
	
i64 %26
Lload8BB
@
	full_text3
1
/%28 = load float, float* %27, align 4, !tbaa !8
+float*8B

	full_text


float* %27
5add8B,
*
	full_text

%29 = add nsw i32 %22, %9
%i328B

	full_text
	
i32 %22
6sext8B,
*
	full_text

%30 = sext i32 %29 to i64
%i328B

	full_text
	
i32 %29
\getelementptr8BI
G
	full_text:
8
6%31 = getelementptr inbounds float, float* %2, i64 %30
%i648B

	full_text
	
i64 %30
Lload8BB
@
	full_text3
1
/%32 = load float, float* %31, align 4, !tbaa !8
+float*8B

	full_text


float* %31
]insertelement8BJ
H
	full_text;
9
7%33 = insertelement <2 x float> undef, float %32, i32 0
)float8B

	full_text

	float %32
3shl8B*
(
	full_text

%34 = shl nsw i32 %9, 1
6add8B-
+
	full_text

%35 = add nsw i32 %34, %22
%i328B

	full_text
	
i32 %34
%i328B

	full_text
	
i32 %22
6sext8B,
*
	full_text

%36 = sext i32 %35 to i64
%i328B

	full_text
	
i32 %35
\getelementptr8BI
G
	full_text:
8
6%37 = getelementptr inbounds float, float* %2, i64 %36
%i648B

	full_text
	
i64 %36
Lload8BB
@
	full_text3
1
/%38 = load float, float* %37, align 4, !tbaa !8
+float*8B

	full_text


float* %37
[insertelement8BH
F
	full_text9
7
5%39 = insertelement <2 x float> %33, float %38, i32 1
5<2 x float>8B"
 
	full_text

<2 x float> %33
)float8B

	full_text

	float %38
3mul8B*
(
	full_text

%40 = mul nsw i32 %9, 3
6add8B-
+
	full_text

%41 = add nsw i32 %40, %22
%i328B

	full_text
	
i32 %40
%i328B

	full_text
	
i32 %22
6sext8B,
*
	full_text

%42 = sext i32 %41 to i64
%i328B

	full_text
	
i32 %41
\getelementptr8BI
G
	full_text:
8
6%43 = getelementptr inbounds float, float* %2, i64 %42
%i648B

	full_text
	
i64 %42
Lload8BB
@
	full_text3
1
/%44 = load float, float* %43, align 4, !tbaa !8
+float*8B

	full_text


float* %43
3shl8B*
(
	full_text

%45 = shl nsw i32 %9, 2
6add8B-
+
	full_text

%46 = add nsw i32 %45, %22
%i328B

	full_text
	
i32 %45
%i328B

	full_text
	
i32 %22
6sext8B,
*
	full_text

%47 = sext i32 %46 to i64
%i328B

	full_text
	
i32 %46
\getelementptr8BI
G
	full_text:
8
6%48 = getelementptr inbounds float, float* %2, i64 %47
%i648B

	full_text
	
i64 %47
Lload8BB
@
	full_text3
1
/%49 = load float, float* %48, align 4, !tbaa !8
+float*8B

	full_text


float* %48
Hbitcast8B;
9
	full_text,
*
(%50 = bitcast %struct.FLOAT3* %11 to i8*
-struct*8B

	full_text

struct* %11
\call8BR
P
	full_textC
A
?call void @llvm.lifetime.start.p0i8(i64 12, i8* nonnull %50) #6
%i8*8B

	full_text
	
i8* %50
call8Bu
s
	full_textf
d
bcall void @compute_velocity(float %28, <2 x float> %39, float %44, %struct.FLOAT3* nonnull %11) #6
)float8B

	full_text

	float %28
5<2 x float>8B"
 
	full_text

<2 x float> %39
)float8B

	full_text

	float %44
-struct*8B

	full_text

struct* %11
Qbitcast8BD
B
	full_text5
3
1%51 = bitcast %struct.FLOAT3* %11 to <2 x float>*
-struct*8B

	full_text

struct* %11
Nload8BD
B
	full_text5
3
1%52 = load <2 x float>, <2 x float>* %51, align 8
7<2 x float>*8B#
!
	full_text

<2 x float>* %51
tgetelementptr8Ba
_
	full_textR
P
N%53 = getelementptr inbounds %struct.FLOAT3, %struct.FLOAT3* %11, i64 0, i32 2
-struct*8B

	full_text

struct* %11
Bload8B8
6
	full_text)
'
%%54 = load float, float* %53, align 8
+float*8B

	full_text


float* %53
_call8BU
S
	full_textF
D
B%55 = call float @compute_speed_sqd(<2 x float> %52, float %54) #6
5<2 x float>8B"
 
	full_text

<2 x float> %52
)float8B

	full_text

	float %54
Ecall8B;
9
	full_text,
*
(%56 = call float @_Z4sqrtf(float %55) #5
)float8B

	full_text

	float %55
ccall8BY
W
	full_textJ
H
F%57 = call float @compute_pressure(float %28, float %49, float %55) #6
)float8B

	full_text

	float %28
)float8B

	full_text

	float %49
)float8B

	full_text

	float %55
^call8BT
R
	full_textE
C
A%58 = call float @compute_speed_of_sound(float %28, float %57) #6
)float8B

	full_text

	float %28
)float8B

	full_text

	float %57
Hbitcast8B;
9
	full_text,
*
(%59 = bitcast %struct.FLOAT3* %12 to i8*
-struct*8B

	full_text

struct* %12
\call8BR
P
	full_textC
A
?call void @llvm.lifetime.start.p0i8(i64 12, i8* nonnull %59) #6
%i8*8B

	full_text
	
i8* %59
Hbitcast8B;
9
	full_text,
*
(%60 = bitcast %struct.FLOAT3* %13 to i8*
-struct*8B

	full_text

struct* %13
\call8BR
P
	full_textC
A
?call void @llvm.lifetime.start.p0i8(i64 12, i8* nonnull %60) #6
%i8*8B

	full_text
	
i8* %60
Hbitcast8B;
9
	full_text,
*
(%61 = bitcast %struct.FLOAT3* %14 to i8*
-struct*8B

	full_text

struct* %14
\call8BR
P
	full_textC
A
?call void @llvm.lifetime.start.p0i8(i64 12, i8* nonnull %61) #6
%i8*8B

	full_text
	
i8* %61
Hbitcast8B;
9
	full_text,
*
(%62 = bitcast %struct.FLOAT3* %15 to i8*
-struct*8B

	full_text

struct* %15
\call8BR
P
	full_textC
A
?call void @llvm.lifetime.start.p0i8(i64 12, i8* nonnull %62) #6
%i8*8B

	full_text
	
i8* %62
Nload8BD
B
	full_text5
3
1%63 = load <2 x float>, <2 x float>* %51, align 8
7<2 x float>*8B#
!
	full_text

<2 x float>* %51
Bload8B8
6
	full_text)
'
%%64 = load float, float* %53, align 8
+float*8B

	full_text


float* %53
ñcall8Bã
à
	full_text˙
˜
Ùcall void @compute_flux_contribution(float %28, <2 x float> %39, float %44, float %49, float %57, <2 x float> %63, float %64, %struct.FLOAT3* nonnull %12, %struct.FLOAT3* nonnull %13, %struct.FLOAT3* nonnull %14, %struct.FLOAT3* nonnull %15) #6
)float8B

	full_text

	float %28
5<2 x float>8B"
 
	full_text

<2 x float> %39
)float8B

	full_text

	float %44
)float8B

	full_text

	float %49
)float8B

	full_text

	float %57
5<2 x float>8B"
 
	full_text

<2 x float> %63
)float8B

	full_text

	float %64
-struct*8B

	full_text

struct* %12
-struct*8B

	full_text

struct* %13
-struct*8B

	full_text

struct* %14
-struct*8B

	full_text

struct* %15
Hbitcast8B;
9
	full_text,
*
(%65 = bitcast %struct.FLOAT3* %16 to i8*
-struct*8B

	full_text

struct* %16
\call8BR
P
	full_textC
A
?call void @llvm.lifetime.start.p0i8(i64 12, i8* nonnull %65) #6
%i8*8B

	full_text
	
i8* %65
Hbitcast8B;
9
	full_text,
*
(%66 = bitcast %struct.FLOAT3* %17 to i8*
-struct*8B

	full_text

struct* %17
\call8BR
P
	full_textC
A
?call void @llvm.lifetime.start.p0i8(i64 12, i8* nonnull %66) #6
%i8*8B

	full_text
	
i8* %66
Hbitcast8B;
9
	full_text,
*
(%67 = bitcast %struct.FLOAT3* %18 to i8*
-struct*8B

	full_text

struct* %18
\call8BR
P
	full_textC
A
?call void @llvm.lifetime.start.p0i8(i64 12, i8* nonnull %67) #6
%i8*8B

	full_text
	
i8* %67
Hbitcast8B;
9
	full_text,
*
(%68 = bitcast %struct.FLOAT3* %19 to i8*
-struct*8B

	full_text

struct* %19
\call8BR
P
	full_textC
A
?call void @llvm.lifetime.start.p0i8(i64 12, i8* nonnull %68) #6
%i8*8B

	full_text
	
i8* %68
Hbitcast8B;
9
	full_text,
*
(%69 = bitcast %struct.FLOAT3* %20 to i8*
-struct*8B

	full_text

struct* %20
\call8BR
P
	full_textC
A
?call void @llvm.lifetime.start.p0i8(i64 12, i8* nonnull %69) #6
%i8*8B

	full_text
	
i8* %69
Qbitcast8BD
B
	full_text5
3
1%70 = bitcast %struct.FLOAT3* %16 to <2 x float>*
-struct*8B

	full_text

struct* %16
tgetelementptr8Ba
_
	full_textR
P
N%71 = getelementptr inbounds %struct.FLOAT3, %struct.FLOAT3* %16, i64 0, i32 2
-struct*8B

	full_text

struct* %16
tgetelementptr8Ba
_
	full_textR
P
N%72 = getelementptr inbounds %struct.FLOAT3, %struct.FLOAT3* %20, i64 0, i32 0
-struct*8B

	full_text

struct* %20
tgetelementptr8Ba
_
	full_textR
P
N%73 = getelementptr inbounds %struct.FLOAT3, %struct.FLOAT3* %15, i64 0, i32 0
-struct*8B

	full_text

struct* %15
tgetelementptr8Ba
_
	full_textR
P
N%74 = getelementptr inbounds %struct.FLOAT3, %struct.FLOAT3* %17, i64 0, i32 0
-struct*8B

	full_text

struct* %17
tgetelementptr8Ba
_
	full_textR
P
N%75 = getelementptr inbounds %struct.FLOAT3, %struct.FLOAT3* %12, i64 0, i32 0
-struct*8B

	full_text

struct* %12
tgetelementptr8Ba
_
	full_textR
P
N%76 = getelementptr inbounds %struct.FLOAT3, %struct.FLOAT3* %18, i64 0, i32 0
-struct*8B

	full_text

struct* %18
tgetelementptr8Ba
_
	full_textR
P
N%77 = getelementptr inbounds %struct.FLOAT3, %struct.FLOAT3* %13, i64 0, i32 0
-struct*8B

	full_text

struct* %13
tgetelementptr8Ba
_
	full_textR
P
N%78 = getelementptr inbounds %struct.FLOAT3, %struct.FLOAT3* %19, i64 0, i32 0
-struct*8B

	full_text

struct* %19
tgetelementptr8Ba
_
	full_textR
P
N%79 = getelementptr inbounds %struct.FLOAT3, %struct.FLOAT3* %14, i64 0, i32 0
-struct*8B

	full_text

struct* %14
tgetelementptr8Ba
_
	full_textR
P
N%80 = getelementptr inbounds %struct.FLOAT3, %struct.FLOAT3* %20, i64 0, i32 1
-struct*8B

	full_text

struct* %20
tgetelementptr8Ba
_
	full_textR
P
N%81 = getelementptr inbounds %struct.FLOAT3, %struct.FLOAT3* %15, i64 0, i32 1
-struct*8B

	full_text

struct* %15
tgetelementptr8Ba
_
	full_textR
P
N%82 = getelementptr inbounds %struct.FLOAT3, %struct.FLOAT3* %17, i64 0, i32 1
-struct*8B

	full_text

struct* %17
tgetelementptr8Ba
_
	full_textR
P
N%83 = getelementptr inbounds %struct.FLOAT3, %struct.FLOAT3* %12, i64 0, i32 1
-struct*8B

	full_text

struct* %12
tgetelementptr8Ba
_
	full_textR
P
N%84 = getelementptr inbounds %struct.FLOAT3, %struct.FLOAT3* %18, i64 0, i32 1
-struct*8B

	full_text

struct* %18
tgetelementptr8Ba
_
	full_textR
P
N%85 = getelementptr inbounds %struct.FLOAT3, %struct.FLOAT3* %13, i64 0, i32 1
-struct*8B

	full_text

struct* %13
tgetelementptr8Ba
_
	full_textR
P
N%86 = getelementptr inbounds %struct.FLOAT3, %struct.FLOAT3* %19, i64 0, i32 1
-struct*8B

	full_text

struct* %19
tgetelementptr8Ba
_
	full_textR
P
N%87 = getelementptr inbounds %struct.FLOAT3, %struct.FLOAT3* %14, i64 0, i32 1
-struct*8B

	full_text

struct* %14
tgetelementptr8Ba
_
	full_textR
P
N%88 = getelementptr inbounds %struct.FLOAT3, %struct.FLOAT3* %20, i64 0, i32 2
-struct*8B

	full_text

struct* %20
tgetelementptr8Ba
_
	full_textR
P
N%89 = getelementptr inbounds %struct.FLOAT3, %struct.FLOAT3* %15, i64 0, i32 2
-struct*8B

	full_text

struct* %15
tgetelementptr8Ba
_
	full_textR
P
N%90 = getelementptr inbounds %struct.FLOAT3, %struct.FLOAT3* %17, i64 0, i32 2
-struct*8B

	full_text

struct* %17
tgetelementptr8Ba
_
	full_textR
P
N%91 = getelementptr inbounds %struct.FLOAT3, %struct.FLOAT3* %12, i64 0, i32 2
-struct*8B

	full_text

struct* %12
tgetelementptr8Ba
_
	full_textR
P
N%92 = getelementptr inbounds %struct.FLOAT3, %struct.FLOAT3* %18, i64 0, i32 2
-struct*8B

	full_text

struct* %18
tgetelementptr8Ba
_
	full_textR
P
N%93 = getelementptr inbounds %struct.FLOAT3, %struct.FLOAT3* %13, i64 0, i32 2
-struct*8B

	full_text

struct* %13
tgetelementptr8Ba
_
	full_textR
P
N%94 = getelementptr inbounds %struct.FLOAT3, %struct.FLOAT3* %19, i64 0, i32 2
-struct*8B

	full_text

struct* %19
tgetelementptr8Ba
_
	full_textR
P
N%95 = getelementptr inbounds %struct.FLOAT3, %struct.FLOAT3* %14, i64 0, i32 2
-struct*8B

	full_text

struct* %14
Zgetelementptr8BG
E
	full_text8
6
4%96 = getelementptr inbounds float, float* %3, i64 1
sgetelementptr8B`
^
	full_textQ
O
M%97 = getelementptr inbounds %struct.FLOAT3, %struct.FLOAT3* %5, i64 0, i32 0
sgetelementptr8B`
^
	full_textQ
O
M%98 = getelementptr inbounds %struct.FLOAT3, %struct.FLOAT3* %6, i64 0, i32 0
sgetelementptr8B`
^
	full_textQ
O
M%99 = getelementptr inbounds %struct.FLOAT3, %struct.FLOAT3* %7, i64 0, i32 0
tgetelementptr8Ba
_
	full_textR
P
N%100 = getelementptr inbounds %struct.FLOAT3, %struct.FLOAT3* %8, i64 0, i32 0
[getelementptr8BH
F
	full_text9
7
5%101 = getelementptr inbounds float, float* %3, i64 2
tgetelementptr8Ba
_
	full_textR
P
N%102 = getelementptr inbounds %struct.FLOAT3, %struct.FLOAT3* %5, i64 0, i32 1
tgetelementptr8Ba
_
	full_textR
P
N%103 = getelementptr inbounds %struct.FLOAT3, %struct.FLOAT3* %6, i64 0, i32 1
tgetelementptr8Ba
_
	full_textR
P
N%104 = getelementptr inbounds %struct.FLOAT3, %struct.FLOAT3* %7, i64 0, i32 1
tgetelementptr8Ba
_
	full_textR
P
N%105 = getelementptr inbounds %struct.FLOAT3, %struct.FLOAT3* %8, i64 0, i32 1
[getelementptr8BH
F
	full_text9
7
5%106 = getelementptr inbounds float, float* %3, i64 3
tgetelementptr8Ba
_
	full_textR
P
N%107 = getelementptr inbounds %struct.FLOAT3, %struct.FLOAT3* %5, i64 0, i32 2
tgetelementptr8Ba
_
	full_textR
P
N%108 = getelementptr inbounds %struct.FLOAT3, %struct.FLOAT3* %6, i64 0, i32 2
tgetelementptr8Ba
_
	full_textR
P
N%109 = getelementptr inbounds %struct.FLOAT3, %struct.FLOAT3* %7, i64 0, i32 2
tgetelementptr8Ba
_
	full_textR
P
N%110 = getelementptr inbounds %struct.FLOAT3, %struct.FLOAT3* %8, i64 0, i32 2
6sext8B,
*
	full_text

%111 = sext i32 %9 to i64
2shl8B)
'
	full_text

%112 = shl i64 %21, 32
%i648B

	full_text
	
i64 %21
;ashr8B1
/
	full_text"
 
%113 = ashr exact i64 %112, 32
&i648B

	full_text


i64 %112
Zgetelementptr8BG
E
	full_text8
6
4%114 = getelementptr inbounds i32, i32* %0, i64 %113
&i648B

	full_text


i64 %113
Kload8BA
?
	full_text2
0
.%115 = load i32, i32* %114, align 4, !tbaa !12
(i32*8B

	full_text

	i32* %114
^getelementptr8BK
I
	full_text<
:
8%116 = getelementptr inbounds float, float* %1, i64 %113
&i648B

	full_text


i64 %113
Nload8BD
B
	full_text5
3
1%117 = load float, float* %116, align 4, !tbaa !8
,float*8B

	full_text

float* %116
6shl8B-
+
	full_text

%118 = shl nsw i64 %111, 2
&i648B

	full_text


i64 %111
9add8B0
.
	full_text!

%119 = add nsw i64 %118, %113
&i648B

	full_text


i64 %118
&i648B

	full_text


i64 %113
^getelementptr8BK
I
	full_text<
:
8%120 = getelementptr inbounds float, float* %1, i64 %119
&i648B

	full_text


i64 %119
Nload8BD
B
	full_text5
3
1%121 = load float, float* %120, align 4, !tbaa !8
,float*8B

	full_text

float* %120
6shl8B-
+
	full_text

%122 = shl nsw i64 %111, 3
&i648B

	full_text


i64 %111
9add8B0
.
	full_text!

%123 = add nsw i64 %122, %113
&i648B

	full_text


i64 %122
&i648B

	full_text


i64 %113
^getelementptr8BK
I
	full_text<
:
8%124 = getelementptr inbounds float, float* %1, i64 %123
&i648B

	full_text


i64 %123
Nload8BD
B
	full_text5
3
1%125 = load float, float* %124, align 4, !tbaa !8
,float*8B

	full_text

float* %124
9icmp8B/
-
	full_text 

%126 = icmp sgt i32 %115, -1
&i328B

	full_text


i32 %115
=br8B5
3
	full_text&
$
"br i1 %126, label %127, label %234
$i18B

	full_text
	
i1 %126
9fmul8B/
-
	full_text 

%128 = fmul float %121, %121
*float8B

	full_text


float %121
*float8B

	full_text


float %121
dcall8BZ
X
	full_textK
I
G%129 = call float @llvm.fmuladd.f32(float %117, float %117, float %128)
*float8B

	full_text


float %117
*float8B

	full_text


float %117
*float8B

	full_text


float %128
dcall8BZ
X
	full_textK
I
G%130 = call float @llvm.fmuladd.f32(float %125, float %125, float %129)
*float8B

	full_text


float %125
*float8B

	full_text


float %125
*float8B

	full_text


float %129
Gcall8B=
;
	full_text.
,
*%131 = call float @_Z4sqrtf(float %130) #5
*float8B

	full_text


float %130
8sext8B.
,
	full_text

%132 = sext i32 %115 to i64
&i328B

	full_text


i32 %115
^getelementptr8BK
I
	full_text<
:
8%133 = getelementptr inbounds float, float* %2, i64 %132
&i648B

	full_text


i64 %132
Nload8BD
B
	full_text5
3
1%134 = load float, float* %133, align 4, !tbaa !8
,float*8B

	full_text

float* %133
7add8B.
,
	full_text

%135 = add nsw i32 %115, %9
&i328B

	full_text


i32 %115
8sext8B.
,
	full_text

%136 = sext i32 %135 to i64
&i328B

	full_text


i32 %135
^getelementptr8BK
I
	full_text<
:
8%137 = getelementptr inbounds float, float* %2, i64 %136
&i648B

	full_text


i64 %136
Nload8BD
B
	full_text5
3
1%138 = load float, float* %137, align 4, !tbaa !8
,float*8B

	full_text

float* %137
_insertelement8BL
J
	full_text=
;
9%139 = insertelement <2 x float> undef, float %138, i32 0
*float8B

	full_text


float %138
8add8B/
-
	full_text 

%140 = add nsw i32 %115, %34
&i328B

	full_text


i32 %115
%i328B

	full_text
	
i32 %34
8sext8B.
,
	full_text

%141 = sext i32 %140 to i64
&i328B

	full_text


i32 %140
^getelementptr8BK
I
	full_text<
:
8%142 = getelementptr inbounds float, float* %2, i64 %141
&i648B

	full_text


i64 %141
Nload8BD
B
	full_text5
3
1%143 = load float, float* %142, align 4, !tbaa !8
,float*8B

	full_text

float* %142
^insertelement8BK
I
	full_text<
:
8%144 = insertelement <2 x float> %139, float %143, i32 1
6<2 x float>8B#
!
	full_text

<2 x float> %139
*float8B

	full_text


float %143
8add8B/
-
	full_text 

%145 = add nsw i32 %115, %40
&i328B

	full_text


i32 %115
%i328B

	full_text
	
i32 %40
8sext8B.
,
	full_text

%146 = sext i32 %145 to i64
&i328B

	full_text


i32 %145
^getelementptr8BK
I
	full_text<
:
8%147 = getelementptr inbounds float, float* %2, i64 %146
&i648B

	full_text


i64 %146
Nload8BD
B
	full_text5
3
1%148 = load float, float* %147, align 4, !tbaa !8
,float*8B

	full_text

float* %147
8add8B/
-
	full_text 

%149 = add nsw i32 %115, %45
&i328B

	full_text


i32 %115
%i328B

	full_text
	
i32 %45
8sext8B.
,
	full_text

%150 = sext i32 %149 to i64
&i328B

	full_text


i32 %149
^getelementptr8BK
I
	full_text<
:
8%151 = getelementptr inbounds float, float* %2, i64 %150
&i648B

	full_text


i64 %150
Nload8BD
B
	full_text5
3
1%152 = load float, float* %151, align 4, !tbaa !8
,float*8B

	full_text

float* %151
Çcall8Bx
v
	full_texti
g
ecall void @compute_velocity(float %134, <2 x float> %144, float %148, %struct.FLOAT3* nonnull %16) #6
*float8B

	full_text


float %134
6<2 x float>8B#
!
	full_text

<2 x float> %144
*float8B

	full_text


float %148
-struct*8B

	full_text

struct* %16
Oload8BE
C
	full_text6
4
2%153 = load <2 x float>, <2 x float>* %70, align 8
7<2 x float>*8B#
!
	full_text

<2 x float>* %70
Cload8B9
7
	full_text*
(
&%154 = load float, float* %71, align 8
+float*8B

	full_text


float* %71
bcall8BX
V
	full_textI
G
E%155 = call float @compute_speed_sqd(<2 x float> %153, float %154) #6
6<2 x float>8B#
!
	full_text

<2 x float> %153
*float8B

	full_text


float %154
gcall8B]
[
	full_textN
L
J%156 = call float @compute_pressure(float %134, float %152, float %155) #6
*float8B

	full_text


float %134
*float8B

	full_text


float %152
*float8B

	full_text


float %155
acall8BW
U
	full_textH
F
D%157 = call float @compute_speed_of_sound(float %134, float %156) #6
*float8B

	full_text


float %134
*float8B

	full_text


float %156
Oload8BE
C
	full_text6
4
2%158 = load <2 x float>, <2 x float>* %70, align 8
7<2 x float>*8B#
!
	full_text

<2 x float>* %70
Cload8B9
7
	full_text*
(
&%159 = load float, float* %71, align 8
+float*8B

	full_text


float* %71
ùcall8Bí
è
	full_textÅ
˛
˚call void @compute_flux_contribution(float %134, <2 x float> %144, float %148, float %152, float %156, <2 x float> %158, float %159, %struct.FLOAT3* nonnull %17, %struct.FLOAT3* nonnull %18, %struct.FLOAT3* nonnull %19, %struct.FLOAT3* nonnull %20) #6
*float8B

	full_text


float %134
6<2 x float>8B#
!
	full_text

<2 x float> %144
*float8B

	full_text


float %148
*float8B

	full_text


float %152
*float8B

	full_text


float %156
6<2 x float>8B#
!
	full_text

<2 x float> %158
*float8B

	full_text


float %159
-struct*8B

	full_text

struct* %17
-struct*8B

	full_text

struct* %18
-struct*8B

	full_text

struct* %19
-struct*8B

	full_text

struct* %20
Gfmul8B=
;
	full_text.
,
*%160 = fmul float %131, 0xBFC99999A0000000
*float8B

	full_text


float %131
Afmul8B7
5
	full_text(
&
$%161 = fmul float %160, 5.000000e-01
*float8B

	full_text


float %160
Gcall8B=
;
	full_text.
,
*%162 = call float @_Z4sqrtf(float %155) #5
*float8B

	full_text


float %155
8fadd8B.
,
	full_text

%163 = fadd float %56, %162
)float8B

	full_text

	float %56
*float8B

	full_text


float %162
8fadd8B.
,
	full_text

%164 = fadd float %58, %163
)float8B

	full_text

	float %58
*float8B

	full_text


float %163
9fadd8B/
-
	full_text 

%165 = fadd float %157, %164
*float8B

	full_text


float %157
*float8B

	full_text


float %164
9fmul8B/
-
	full_text 

%166 = fmul float %161, %165
*float8B

	full_text


float %161
*float8B

	full_text


float %165
8fsub8B.
,
	full_text

%167 = fsub float %28, %134
)float8B

	full_text

	float %28
*float8B

	full_text


float %134
lcall8Bb
`
	full_textS
Q
O%168 = call float @llvm.fmuladd.f32(float %166, float %167, float 0.000000e+00)
*float8B

	full_text


float %166
*float8B

	full_text


float %167
8fsub8B.
,
	full_text

%169 = fsub float %49, %152
)float8B

	full_text

	float %49
*float8B

	full_text


float %152
lcall8Bb
`
	full_textS
Q
O%170 = call float @llvm.fmuladd.f32(float %166, float %169, float 0.000000e+00)
*float8B

	full_text


float %166
*float8B

	full_text


float %169
8fsub8B.
,
	full_text

%171 = fsub float %32, %138
)float8B

	full_text

	float %32
*float8B

	full_text


float %138
lcall8Bb
`
	full_textS
Q
O%172 = call float @llvm.fmuladd.f32(float %166, float %171, float 0.000000e+00)
*float8B

	full_text


float %166
*float8B

	full_text


float %171
8fsub8B.
,
	full_text

%173 = fsub float %38, %143
)float8B

	full_text

	float %38
*float8B

	full_text


float %143
lcall8Bb
`
	full_textS
Q
O%174 = call float @llvm.fmuladd.f32(float %166, float %173, float 0.000000e+00)
*float8B

	full_text


float %166
*float8B

	full_text


float %173
8fsub8B.
,
	full_text

%175 = fsub float %44, %148
)float8B

	full_text

	float %44
*float8B

	full_text


float %148
lcall8Bb
`
	full_textS
Q
O%176 = call float @llvm.fmuladd.f32(float %166, float %175, float 0.000000e+00)
*float8B

	full_text


float %166
*float8B

	full_text


float %175
Afmul8B7
5
	full_text(
&
$%177 = fmul float %117, 5.000000e-01
*float8B

	full_text


float %117
8fadd8B.
,
	full_text

%178 = fadd float %32, %138
)float8B

	full_text

	float %32
*float8B

	full_text


float %138
dcall8BZ
X
	full_textK
I
G%179 = call float @llvm.fmuladd.f32(float %177, float %178, float %168)
*float8B

	full_text


float %177
*float8B

	full_text


float %178
*float8B

	full_text


float %168
Nload8BD
B
	full_text5
3
1%180 = load float, float* %72, align 4, !tbaa !14
+float*8B

	full_text


float* %72
Nload8BD
B
	full_text5
3
1%181 = load float, float* %73, align 4, !tbaa !14
+float*8B

	full_text


float* %73
9fadd8B/
-
	full_text 

%182 = fadd float %180, %181
*float8B

	full_text


float %180
*float8B

	full_text


float %181
dcall8BZ
X
	full_textK
I
G%183 = call float @llvm.fmuladd.f32(float %177, float %182, float %170)
*float8B

	full_text


float %177
*float8B

	full_text


float %182
*float8B

	full_text


float %170
Nload8BD
B
	full_text5
3
1%184 = load float, float* %74, align 4, !tbaa !14
+float*8B

	full_text


float* %74
Nload8BD
B
	full_text5
3
1%185 = load float, float* %75, align 4, !tbaa !14
+float*8B

	full_text


float* %75
9fadd8B/
-
	full_text 

%186 = fadd float %184, %185
*float8B

	full_text


float %184
*float8B

	full_text


float %185
dcall8BZ
X
	full_textK
I
G%187 = call float @llvm.fmuladd.f32(float %177, float %186, float %172)
*float8B

	full_text


float %177
*float8B

	full_text


float %186
*float8B

	full_text


float %172
Nload8BD
B
	full_text5
3
1%188 = load float, float* %76, align 4, !tbaa !14
+float*8B

	full_text


float* %76
Nload8BD
B
	full_text5
3
1%189 = load float, float* %77, align 4, !tbaa !14
+float*8B

	full_text


float* %77
9fadd8B/
-
	full_text 

%190 = fadd float %188, %189
*float8B

	full_text


float %188
*float8B

	full_text


float %189
dcall8BZ
X
	full_textK
I
G%191 = call float @llvm.fmuladd.f32(float %177, float %190, float %174)
*float8B

	full_text


float %177
*float8B

	full_text


float %190
*float8B

	full_text


float %174
Nload8BD
B
	full_text5
3
1%192 = load float, float* %78, align 4, !tbaa !14
+float*8B

	full_text


float* %78
Nload8BD
B
	full_text5
3
1%193 = load float, float* %79, align 4, !tbaa !14
+float*8B

	full_text


float* %79
9fadd8B/
-
	full_text 

%194 = fadd float %192, %193
*float8B

	full_text


float %192
*float8B

	full_text


float %193
dcall8BZ
X
	full_textK
I
G%195 = call float @llvm.fmuladd.f32(float %177, float %194, float %176)
*float8B

	full_text


float %177
*float8B

	full_text


float %194
*float8B

	full_text


float %176
Afmul8B7
5
	full_text(
&
$%196 = fmul float %121, 5.000000e-01
*float8B

	full_text


float %121
8fadd8B.
,
	full_text

%197 = fadd float %38, %143
)float8B

	full_text

	float %38
*float8B

	full_text


float %143
dcall8BZ
X
	full_textK
I
G%198 = call float @llvm.fmuladd.f32(float %196, float %197, float %179)
*float8B

	full_text


float %196
*float8B

	full_text


float %197
*float8B

	full_text


float %179
Nload8BD
B
	full_text5
3
1%199 = load float, float* %80, align 4, !tbaa !16
+float*8B

	full_text


float* %80
Nload8BD
B
	full_text5
3
1%200 = load float, float* %81, align 4, !tbaa !16
+float*8B

	full_text


float* %81
9fadd8B/
-
	full_text 

%201 = fadd float %199, %200
*float8B

	full_text


float %199
*float8B

	full_text


float %200
dcall8BZ
X
	full_textK
I
G%202 = call float @llvm.fmuladd.f32(float %196, float %201, float %183)
*float8B

	full_text


float %196
*float8B

	full_text


float %201
*float8B

	full_text


float %183
Nload8BD
B
	full_text5
3
1%203 = load float, float* %82, align 4, !tbaa !16
+float*8B

	full_text


float* %82
Nload8BD
B
	full_text5
3
1%204 = load float, float* %83, align 4, !tbaa !16
+float*8B

	full_text


float* %83
9fadd8B/
-
	full_text 

%205 = fadd float %203, %204
*float8B

	full_text


float %203
*float8B

	full_text


float %204
dcall8BZ
X
	full_textK
I
G%206 = call float @llvm.fmuladd.f32(float %196, float %205, float %187)
*float8B

	full_text


float %196
*float8B

	full_text


float %205
*float8B

	full_text


float %187
Nload8BD
B
	full_text5
3
1%207 = load float, float* %84, align 4, !tbaa !16
+float*8B

	full_text


float* %84
Nload8BD
B
	full_text5
3
1%208 = load float, float* %85, align 4, !tbaa !16
+float*8B

	full_text


float* %85
9fadd8B/
-
	full_text 

%209 = fadd float %207, %208
*float8B

	full_text


float %207
*float8B

	full_text


float %208
dcall8BZ
X
	full_textK
I
G%210 = call float @llvm.fmuladd.f32(float %196, float %209, float %191)
*float8B

	full_text


float %196
*float8B

	full_text


float %209
*float8B

	full_text


float %191
Nload8BD
B
	full_text5
3
1%211 = load float, float* %86, align 4, !tbaa !16
+float*8B

	full_text


float* %86
Nload8BD
B
	full_text5
3
1%212 = load float, float* %87, align 4, !tbaa !16
+float*8B

	full_text


float* %87
9fadd8B/
-
	full_text 

%213 = fadd float %211, %212
*float8B

	full_text


float %211
*float8B

	full_text


float %212
dcall8BZ
X
	full_textK
I
G%214 = call float @llvm.fmuladd.f32(float %196, float %213, float %195)
*float8B

	full_text


float %196
*float8B

	full_text


float %213
*float8B

	full_text


float %195
Afmul8B7
5
	full_text(
&
$%215 = fmul float %125, 5.000000e-01
*float8B

	full_text


float %125
8fadd8B.
,
	full_text

%216 = fadd float %44, %148
)float8B

	full_text

	float %44
*float8B

	full_text


float %148
dcall8BZ
X
	full_textK
I
G%217 = call float @llvm.fmuladd.f32(float %215, float %216, float %198)
*float8B

	full_text


float %215
*float8B

	full_text


float %216
*float8B

	full_text


float %198
Nload8BD
B
	full_text5
3
1%218 = load float, float* %88, align 4, !tbaa !17
+float*8B

	full_text


float* %88
Nload8BD
B
	full_text5
3
1%219 = load float, float* %89, align 4, !tbaa !17
+float*8B

	full_text


float* %89
9fadd8B/
-
	full_text 

%220 = fadd float %218, %219
*float8B

	full_text


float %218
*float8B

	full_text


float %219
dcall8BZ
X
	full_textK
I
G%221 = call float @llvm.fmuladd.f32(float %215, float %220, float %202)
*float8B

	full_text


float %215
*float8B

	full_text


float %220
*float8B

	full_text


float %202
Nload8BD
B
	full_text5
3
1%222 = load float, float* %90, align 4, !tbaa !17
+float*8B

	full_text


float* %90
Nload8BD
B
	full_text5
3
1%223 = load float, float* %91, align 4, !tbaa !17
+float*8B

	full_text


float* %91
9fadd8B/
-
	full_text 

%224 = fadd float %222, %223
*float8B

	full_text


float %222
*float8B

	full_text


float %223
dcall8BZ
X
	full_textK
I
G%225 = call float @llvm.fmuladd.f32(float %215, float %224, float %206)
*float8B

	full_text


float %215
*float8B

	full_text


float %224
*float8B

	full_text


float %206
Nload8BD
B
	full_text5
3
1%226 = load float, float* %92, align 4, !tbaa !17
+float*8B

	full_text


float* %92
Nload8BD
B
	full_text5
3
1%227 = load float, float* %93, align 4, !tbaa !17
+float*8B

	full_text


float* %93
9fadd8B/
-
	full_text 

%228 = fadd float %226, %227
*float8B

	full_text


float %226
*float8B

	full_text


float %227
dcall8BZ
X
	full_textK
I
G%229 = call float @llvm.fmuladd.f32(float %215, float %228, float %210)
*float8B

	full_text


float %215
*float8B

	full_text


float %228
*float8B

	full_text


float %210
Nload8BD
B
	full_text5
3
1%230 = load float, float* %94, align 4, !tbaa !17
+float*8B

	full_text


float* %94
Nload8BD
B
	full_text5
3
1%231 = load float, float* %95, align 4, !tbaa !17
+float*8B

	full_text


float* %95
9fadd8B/
-
	full_text 

%232 = fadd float %230, %231
*float8B

	full_text


float %230
*float8B

	full_text


float %231
dcall8BZ
X
	full_textK
I
G%233 = call float @llvm.fmuladd.f32(float %215, float %232, float %214)
*float8B

	full_text


float %215
*float8B

	full_text


float %232
*float8B

	full_text


float %214
(br8B 

	full_text

br label %300
nswitch8Bb
`
	full_textS
Q
Oswitch i32 %115, label %300 [
    i32 -1, label %235
    i32 -2, label %239
  ]
&i328B

	full_text


i32 %115
kcall8Ba
_
	full_textR
P
N%236 = call float @llvm.fmuladd.f32(float %117, float %57, float 0.000000e+00)
*float8B

	full_text


float %117
)float8B

	full_text

	float %57
kcall8Ba
_
	full_textR
P
N%237 = call float @llvm.fmuladd.f32(float %121, float %57, float 0.000000e+00)
*float8B

	full_text


float %121
)float8B

	full_text

	float %57
kcall8Ba
_
	full_textR
P
N%238 = call float @llvm.fmuladd.f32(float %125, float %57, float 0.000000e+00)
*float8B

	full_text


float %125
)float8B

	full_text

	float %57
(br8B 

	full_text

br label %300
Afmul8B7
5
	full_text(
&
$%240 = fmul float %117, 5.000000e-01
*float8B

	full_text


float %117
Mload8BC
A
	full_text4
2
0%241 = load float, float* %96, align 4, !tbaa !8
+float*8B

	full_text


float* %96
8fadd8B.
,
	full_text

%242 = fadd float %32, %241
)float8B

	full_text

	float %32
*float8B

	full_text


float %241
lcall8Bb
`
	full_textS
Q
O%243 = call float @llvm.fmuladd.f32(float %240, float %242, float 0.000000e+00)
*float8B

	full_text


float %240
*float8B

	full_text


float %242
Nload8BD
B
	full_text5
3
1%244 = load float, float* %97, align 4, !tbaa !14
+float*8B

	full_text


float* %97
Nload8BD
B
	full_text5
3
1%245 = load float, float* %73, align 4, !tbaa !14
+float*8B

	full_text


float* %73
9fadd8B/
-
	full_text 

%246 = fadd float %244, %245
*float8B

	full_text


float %244
*float8B

	full_text


float %245
lcall8Bb
`
	full_textS
Q
O%247 = call float @llvm.fmuladd.f32(float %240, float %246, float 0.000000e+00)
*float8B

	full_text


float %240
*float8B

	full_text


float %246
Nload8BD
B
	full_text5
3
1%248 = load float, float* %98, align 4, !tbaa !14
+float*8B

	full_text


float* %98
Nload8BD
B
	full_text5
3
1%249 = load float, float* %75, align 4, !tbaa !14
+float*8B

	full_text


float* %75
9fadd8B/
-
	full_text 

%250 = fadd float %248, %249
*float8B

	full_text


float %248
*float8B

	full_text


float %249
lcall8Bb
`
	full_textS
Q
O%251 = call float @llvm.fmuladd.f32(float %240, float %250, float 0.000000e+00)
*float8B

	full_text


float %240
*float8B

	full_text


float %250
Nload8BD
B
	full_text5
3
1%252 = load float, float* %99, align 4, !tbaa !14
+float*8B

	full_text


float* %99
Nload8BD
B
	full_text5
3
1%253 = load float, float* %77, align 4, !tbaa !14
+float*8B

	full_text


float* %77
9fadd8B/
-
	full_text 

%254 = fadd float %252, %253
*float8B

	full_text


float %252
*float8B

	full_text


float %253
lcall8Bb
`
	full_textS
Q
O%255 = call float @llvm.fmuladd.f32(float %240, float %254, float 0.000000e+00)
*float8B

	full_text


float %240
*float8B

	full_text


float %254
Oload8BE
C
	full_text6
4
2%256 = load float, float* %100, align 4, !tbaa !14
,float*8B

	full_text

float* %100
Nload8BD
B
	full_text5
3
1%257 = load float, float* %79, align 4, !tbaa !14
+float*8B

	full_text


float* %79
9fadd8B/
-
	full_text 

%258 = fadd float %256, %257
*float8B

	full_text


float %256
*float8B

	full_text


float %257
lcall8Bb
`
	full_textS
Q
O%259 = call float @llvm.fmuladd.f32(float %240, float %258, float 0.000000e+00)
*float8B

	full_text


float %240
*float8B

	full_text


float %258
Afmul8B7
5
	full_text(
&
$%260 = fmul float %121, 5.000000e-01
*float8B

	full_text


float %121
Nload8BD
B
	full_text5
3
1%261 = load float, float* %101, align 4, !tbaa !8
,float*8B

	full_text

float* %101
8fadd8B.
,
	full_text

%262 = fadd float %38, %261
)float8B

	full_text

	float %38
*float8B

	full_text


float %261
dcall8BZ
X
	full_textK
I
G%263 = call float @llvm.fmuladd.f32(float %260, float %262, float %243)
*float8B

	full_text


float %260
*float8B

	full_text


float %262
*float8B

	full_text


float %243
Oload8BE
C
	full_text6
4
2%264 = load float, float* %102, align 4, !tbaa !16
,float*8B

	full_text

float* %102
Nload8BD
B
	full_text5
3
1%265 = load float, float* %81, align 4, !tbaa !16
+float*8B

	full_text


float* %81
9fadd8B/
-
	full_text 

%266 = fadd float %264, %265
*float8B

	full_text


float %264
*float8B

	full_text


float %265
dcall8BZ
X
	full_textK
I
G%267 = call float @llvm.fmuladd.f32(float %260, float %266, float %247)
*float8B

	full_text


float %260
*float8B

	full_text


float %266
*float8B

	full_text


float %247
Oload8BE
C
	full_text6
4
2%268 = load float, float* %103, align 4, !tbaa !16
,float*8B

	full_text

float* %103
Nload8BD
B
	full_text5
3
1%269 = load float, float* %83, align 4, !tbaa !16
+float*8B

	full_text


float* %83
9fadd8B/
-
	full_text 

%270 = fadd float %268, %269
*float8B

	full_text


float %268
*float8B

	full_text


float %269
dcall8BZ
X
	full_textK
I
G%271 = call float @llvm.fmuladd.f32(float %260, float %270, float %251)
*float8B

	full_text


float %260
*float8B

	full_text


float %270
*float8B

	full_text


float %251
Oload8BE
C
	full_text6
4
2%272 = load float, float* %104, align 4, !tbaa !16
,float*8B

	full_text

float* %104
Nload8BD
B
	full_text5
3
1%273 = load float, float* %85, align 4, !tbaa !16
+float*8B

	full_text


float* %85
9fadd8B/
-
	full_text 

%274 = fadd float %272, %273
*float8B

	full_text


float %272
*float8B

	full_text


float %273
dcall8BZ
X
	full_textK
I
G%275 = call float @llvm.fmuladd.f32(float %260, float %274, float %255)
*float8B

	full_text


float %260
*float8B

	full_text


float %274
*float8B

	full_text


float %255
Oload8BE
C
	full_text6
4
2%276 = load float, float* %105, align 4, !tbaa !16
,float*8B

	full_text

float* %105
Nload8BD
B
	full_text5
3
1%277 = load float, float* %87, align 4, !tbaa !16
+float*8B

	full_text


float* %87
9fadd8B/
-
	full_text 

%278 = fadd float %276, %277
*float8B

	full_text


float %276
*float8B

	full_text


float %277
dcall8BZ
X
	full_textK
I
G%279 = call float @llvm.fmuladd.f32(float %260, float %278, float %259)
*float8B

	full_text


float %260
*float8B

	full_text


float %278
*float8B

	full_text


float %259
Afmul8B7
5
	full_text(
&
$%280 = fmul float %125, 5.000000e-01
*float8B

	full_text


float %125
Nload8BD
B
	full_text5
3
1%281 = load float, float* %106, align 4, !tbaa !8
,float*8B

	full_text

float* %106
8fadd8B.
,
	full_text

%282 = fadd float %44, %281
)float8B

	full_text

	float %44
*float8B

	full_text


float %281
dcall8BZ
X
	full_textK
I
G%283 = call float @llvm.fmuladd.f32(float %280, float %282, float %263)
*float8B

	full_text


float %280
*float8B

	full_text


float %282
*float8B

	full_text


float %263
Oload8BE
C
	full_text6
4
2%284 = load float, float* %107, align 4, !tbaa !17
,float*8B

	full_text

float* %107
Nload8BD
B
	full_text5
3
1%285 = load float, float* %89, align 4, !tbaa !17
+float*8B

	full_text


float* %89
9fadd8B/
-
	full_text 

%286 = fadd float %284, %285
*float8B

	full_text


float %284
*float8B

	full_text


float %285
dcall8BZ
X
	full_textK
I
G%287 = call float @llvm.fmuladd.f32(float %280, float %286, float %267)
*float8B

	full_text


float %280
*float8B

	full_text


float %286
*float8B

	full_text


float %267
Oload8BE
C
	full_text6
4
2%288 = load float, float* %108, align 4, !tbaa !17
,float*8B

	full_text

float* %108
Nload8BD
B
	full_text5
3
1%289 = load float, float* %91, align 4, !tbaa !17
+float*8B

	full_text


float* %91
9fadd8B/
-
	full_text 

%290 = fadd float %288, %289
*float8B

	full_text


float %288
*float8B

	full_text


float %289
dcall8BZ
X
	full_textK
I
G%291 = call float @llvm.fmuladd.f32(float %280, float %290, float %271)
*float8B

	full_text


float %280
*float8B

	full_text


float %290
*float8B

	full_text


float %271
Oload8BE
C
	full_text6
4
2%292 = load float, float* %109, align 4, !tbaa !17
,float*8B

	full_text

float* %109
Nload8BD
B
	full_text5
3
1%293 = load float, float* %93, align 4, !tbaa !17
+float*8B

	full_text


float* %93
9fadd8B/
-
	full_text 

%294 = fadd float %292, %293
*float8B

	full_text


float %292
*float8B

	full_text


float %293
dcall8BZ
X
	full_textK
I
G%295 = call float @llvm.fmuladd.f32(float %280, float %294, float %275)
*float8B

	full_text


float %280
*float8B

	full_text


float %294
*float8B

	full_text


float %275
Oload8BE
C
	full_text6
4
2%296 = load float, float* %110, align 4, !tbaa !17
,float*8B

	full_text

float* %110
Nload8BD
B
	full_text5
3
1%297 = load float, float* %95, align 4, !tbaa !17
+float*8B

	full_text


float* %95
9fadd8B/
-
	full_text 

%298 = fadd float %296, %297
*float8B

	full_text


float %296
*float8B

	full_text


float %297
dcall8BZ
X
	full_textK
I
G%299 = call float @llvm.fmuladd.f32(float %280, float %298, float %279)
*float8B

	full_text


float %280
*float8B

	full_text


float %298
*float8B

	full_text


float %279
(br8B 

	full_text

br label %300
{phi8Br
p
	full_textc
a
_%301 = phi float [ %221, %127 ], [ 0.000000e+00, %234 ], [ 0.000000e+00, %235 ], [ %287, %239 ]
*float8B

	full_text


float %221
*float8B

	full_text


float %287
sphi8Bj
h
	full_text[
Y
W%302 = phi float [ %225, %127 ], [ 0.000000e+00, %234 ], [ %236, %235 ], [ %291, %239 ]
*float8B

	full_text


float %225
*float8B

	full_text


float %236
*float8B

	full_text


float %291
sphi8Bj
h
	full_text[
Y
W%303 = phi float [ %229, %127 ], [ 0.000000e+00, %234 ], [ %237, %235 ], [ %295, %239 ]
*float8B

	full_text


float %229
*float8B

	full_text


float %237
*float8B

	full_text


float %295
sphi8Bj
h
	full_text[
Y
W%304 = phi float [ %233, %127 ], [ 0.000000e+00, %234 ], [ %238, %235 ], [ %299, %239 ]
*float8B

	full_text


float %233
*float8B

	full_text


float %238
*float8B

	full_text


float %299
{phi8Br
p
	full_textc
a
_%305 = phi float [ %217, %127 ], [ 0.000000e+00, %234 ], [ 0.000000e+00, %235 ], [ %283, %239 ]
*float8B

	full_text


float %217
*float8B

	full_text


float %283
9add8B0
.
	full_text!

%306 = add nsw i64 %113, %111
&i648B

	full_text


i64 %113
&i648B

	full_text


i64 %111
Zgetelementptr8BG
E
	full_text8
6
4%307 = getelementptr inbounds i32, i32* %0, i64 %306
&i648B

	full_text


i64 %306
Kload8BA
?
	full_text2
0
.%308 = load i32, i32* %307, align 4, !tbaa !12
(i32*8B

	full_text

	i32* %307
^getelementptr8BK
I
	full_text<
:
8%309 = getelementptr inbounds float, float* %1, i64 %306
&i648B

	full_text


i64 %306
Nload8BD
B
	full_text5
3
1%310 = load float, float* %309, align 4, !tbaa !8
,float*8B

	full_text

float* %309
6mul8B-
+
	full_text

%311 = mul nsw i64 %111, 5
&i648B

	full_text


i64 %111
9add8B0
.
	full_text!

%312 = add nsw i64 %311, %113
&i648B

	full_text


i64 %311
&i648B

	full_text


i64 %113
^getelementptr8BK
I
	full_text<
:
8%313 = getelementptr inbounds float, float* %1, i64 %312
&i648B

	full_text


i64 %312
Nload8BD
B
	full_text5
3
1%314 = load float, float* %313, align 4, !tbaa !8
,float*8B

	full_text

float* %313
6mul8B-
+
	full_text

%315 = mul nsw i64 %111, 9
&i648B

	full_text


i64 %111
9add8B0
.
	full_text!

%316 = add nsw i64 %315, %113
&i648B

	full_text


i64 %315
&i648B

	full_text


i64 %113
^getelementptr8BK
I
	full_text<
:
8%317 = getelementptr inbounds float, float* %1, i64 %316
&i648B

	full_text


i64 %316
Nload8BD
B
	full_text5
3
1%318 = load float, float* %317, align 4, !tbaa !8
,float*8B

	full_text

float* %317
9icmp8B/
-
	full_text 

%319 = icmp sgt i32 %308, -1
&i328B

	full_text


i32 %308
=br8B5
3
	full_text&
$
"br i1 %319, label %387, label %321
$i18B

	full_text
	
i1 %319
$ret8B

	full_text


ret void
nswitch8Bb
`
	full_textS
Q
Oswitch i32 %308, label %494 [
    i32 -1, label %383
    i32 -2, label %322
  ]
&i328B

	full_text


i32 %308
Afmul8	B7
5
	full_text(
&
$%323 = fmul float %310, 5.000000e-01
*float8	B

	full_text


float %310
Mload8	BC
A
	full_text4
2
0%324 = load float, float* %96, align 4, !tbaa !8
+float*8	B

	full_text


float* %96
8fadd8	B.
,
	full_text

%325 = fadd float %32, %324
)float8	B

	full_text

	float %32
*float8	B

	full_text


float %324
dcall8	BZ
X
	full_textK
I
G%326 = call float @llvm.fmuladd.f32(float %323, float %325, float %305)
*float8	B

	full_text


float %323
*float8	B

	full_text


float %325
*float8	B

	full_text


float %305
Nload8	BD
B
	full_text5
3
1%327 = load float, float* %97, align 4, !tbaa !14
+float*8	B

	full_text


float* %97
Nload8	BD
B
	full_text5
3
1%328 = load float, float* %73, align 4, !tbaa !14
+float*8	B

	full_text


float* %73
9fadd8	B/
-
	full_text 

%329 = fadd float %327, %328
*float8	B

	full_text


float %327
*float8	B

	full_text


float %328
dcall8	BZ
X
	full_textK
I
G%330 = call float @llvm.fmuladd.f32(float %323, float %329, float %301)
*float8	B

	full_text


float %323
*float8	B

	full_text


float %329
*float8	B

	full_text


float %301
Nload8	BD
B
	full_text5
3
1%331 = load float, float* %98, align 4, !tbaa !14
+float*8	B

	full_text


float* %98
Nload8	BD
B
	full_text5
3
1%332 = load float, float* %75, align 4, !tbaa !14
+float*8	B

	full_text


float* %75
9fadd8	B/
-
	full_text 

%333 = fadd float %331, %332
*float8	B

	full_text


float %331
*float8	B

	full_text


float %332
dcall8	BZ
X
	full_textK
I
G%334 = call float @llvm.fmuladd.f32(float %323, float %333, float %302)
*float8	B

	full_text


float %323
*float8	B

	full_text


float %333
*float8	B

	full_text


float %302
Nload8	BD
B
	full_text5
3
1%335 = load float, float* %99, align 4, !tbaa !14
+float*8	B

	full_text


float* %99
Nload8	BD
B
	full_text5
3
1%336 = load float, float* %77, align 4, !tbaa !14
+float*8	B

	full_text


float* %77
9fadd8	B/
-
	full_text 

%337 = fadd float %335, %336
*float8	B

	full_text


float %335
*float8	B

	full_text


float %336
dcall8	BZ
X
	full_textK
I
G%338 = call float @llvm.fmuladd.f32(float %323, float %337, float %303)
*float8	B

	full_text


float %323
*float8	B

	full_text


float %337
*float8	B

	full_text


float %303
Oload8	BE
C
	full_text6
4
2%339 = load float, float* %100, align 4, !tbaa !14
,float*8	B

	full_text

float* %100
Nload8	BD
B
	full_text5
3
1%340 = load float, float* %79, align 4, !tbaa !14
+float*8	B

	full_text


float* %79
9fadd8	B/
-
	full_text 

%341 = fadd float %339, %340
*float8	B

	full_text


float %339
*float8	B

	full_text


float %340
dcall8	BZ
X
	full_textK
I
G%342 = call float @llvm.fmuladd.f32(float %323, float %341, float %304)
*float8	B

	full_text


float %323
*float8	B

	full_text


float %341
*float8	B

	full_text


float %304
Afmul8	B7
5
	full_text(
&
$%343 = fmul float %314, 5.000000e-01
*float8	B

	full_text


float %314
Nload8	BD
B
	full_text5
3
1%344 = load float, float* %101, align 4, !tbaa !8
,float*8	B

	full_text

float* %101
8fadd8	B.
,
	full_text

%345 = fadd float %38, %344
)float8	B

	full_text

	float %38
*float8	B

	full_text


float %344
dcall8	BZ
X
	full_textK
I
G%346 = call float @llvm.fmuladd.f32(float %343, float %345, float %326)
*float8	B

	full_text


float %343
*float8	B

	full_text


float %345
*float8	B

	full_text


float %326
Oload8	BE
C
	full_text6
4
2%347 = load float, float* %102, align 4, !tbaa !16
,float*8	B

	full_text

float* %102
Nload8	BD
B
	full_text5
3
1%348 = load float, float* %81, align 4, !tbaa !16
+float*8	B

	full_text


float* %81
9fadd8	B/
-
	full_text 

%349 = fadd float %347, %348
*float8	B

	full_text


float %347
*float8	B

	full_text


float %348
dcall8	BZ
X
	full_textK
I
G%350 = call float @llvm.fmuladd.f32(float %343, float %349, float %330)
*float8	B

	full_text


float %343
*float8	B

	full_text


float %349
*float8	B

	full_text


float %330
Oload8	BE
C
	full_text6
4
2%351 = load float, float* %103, align 4, !tbaa !16
,float*8	B

	full_text

float* %103
Nload8	BD
B
	full_text5
3
1%352 = load float, float* %83, align 4, !tbaa !16
+float*8	B

	full_text


float* %83
9fadd8	B/
-
	full_text 

%353 = fadd float %351, %352
*float8	B

	full_text


float %351
*float8	B

	full_text


float %352
dcall8	BZ
X
	full_textK
I
G%354 = call float @llvm.fmuladd.f32(float %343, float %353, float %334)
*float8	B

	full_text


float %343
*float8	B

	full_text


float %353
*float8	B

	full_text


float %334
Oload8	BE
C
	full_text6
4
2%355 = load float, float* %104, align 4, !tbaa !16
,float*8	B

	full_text

float* %104
Nload8	BD
B
	full_text5
3
1%356 = load float, float* %85, align 4, !tbaa !16
+float*8	B

	full_text


float* %85
9fadd8	B/
-
	full_text 

%357 = fadd float %355, %356
*float8	B

	full_text


float %355
*float8	B

	full_text


float %356
dcall8	BZ
X
	full_textK
I
G%358 = call float @llvm.fmuladd.f32(float %343, float %357, float %338)
*float8	B

	full_text


float %343
*float8	B

	full_text


float %357
*float8	B

	full_text


float %338
Oload8	BE
C
	full_text6
4
2%359 = load float, float* %105, align 4, !tbaa !16
,float*8	B

	full_text

float* %105
Nload8	BD
B
	full_text5
3
1%360 = load float, float* %87, align 4, !tbaa !16
+float*8	B

	full_text


float* %87
9fadd8	B/
-
	full_text 

%361 = fadd float %359, %360
*float8	B

	full_text


float %359
*float8	B

	full_text


float %360
dcall8	BZ
X
	full_textK
I
G%362 = call float @llvm.fmuladd.f32(float %343, float %361, float %342)
*float8	B

	full_text


float %343
*float8	B

	full_text


float %361
*float8	B

	full_text


float %342
Afmul8	B7
5
	full_text(
&
$%363 = fmul float %318, 5.000000e-01
*float8	B

	full_text


float %318
Nload8	BD
B
	full_text5
3
1%364 = load float, float* %106, align 4, !tbaa !8
,float*8	B

	full_text

float* %106
8fadd8	B.
,
	full_text

%365 = fadd float %44, %364
)float8	B

	full_text

	float %44
*float8	B

	full_text


float %364
dcall8	BZ
X
	full_textK
I
G%366 = call float @llvm.fmuladd.f32(float %363, float %365, float %346)
*float8	B

	full_text


float %363
*float8	B

	full_text


float %365
*float8	B

	full_text


float %346
Oload8	BE
C
	full_text6
4
2%367 = load float, float* %107, align 4, !tbaa !17
,float*8	B

	full_text

float* %107
Nload8	BD
B
	full_text5
3
1%368 = load float, float* %89, align 4, !tbaa !17
+float*8	B

	full_text


float* %89
9fadd8	B/
-
	full_text 

%369 = fadd float %367, %368
*float8	B

	full_text


float %367
*float8	B

	full_text


float %368
dcall8	BZ
X
	full_textK
I
G%370 = call float @llvm.fmuladd.f32(float %363, float %369, float %350)
*float8	B

	full_text


float %363
*float8	B

	full_text


float %369
*float8	B

	full_text


float %350
Oload8	BE
C
	full_text6
4
2%371 = load float, float* %108, align 4, !tbaa !17
,float*8	B

	full_text

float* %108
Nload8	BD
B
	full_text5
3
1%372 = load float, float* %91, align 4, !tbaa !17
+float*8	B

	full_text


float* %91
9fadd8	B/
-
	full_text 

%373 = fadd float %371, %372
*float8	B

	full_text


float %371
*float8	B

	full_text


float %372
dcall8	BZ
X
	full_textK
I
G%374 = call float @llvm.fmuladd.f32(float %363, float %373, float %354)
*float8	B

	full_text


float %363
*float8	B

	full_text


float %373
*float8	B

	full_text


float %354
Oload8	BE
C
	full_text6
4
2%375 = load float, float* %109, align 4, !tbaa !17
,float*8	B

	full_text

float* %109
Nload8	BD
B
	full_text5
3
1%376 = load float, float* %93, align 4, !tbaa !17
+float*8	B

	full_text


float* %93
9fadd8	B/
-
	full_text 

%377 = fadd float %375, %376
*float8	B

	full_text


float %375
*float8	B

	full_text


float %376
dcall8	BZ
X
	full_textK
I
G%378 = call float @llvm.fmuladd.f32(float %363, float %377, float %358)
*float8	B

	full_text


float %363
*float8	B

	full_text


float %377
*float8	B

	full_text


float %358
Oload8	BE
C
	full_text6
4
2%379 = load float, float* %110, align 4, !tbaa !17
,float*8	B

	full_text

float* %110
Nload8	BD
B
	full_text5
3
1%380 = load float, float* %95, align 4, !tbaa !17
+float*8	B

	full_text


float* %95
9fadd8	B/
-
	full_text 

%381 = fadd float %379, %380
*float8	B

	full_text


float %379
*float8	B

	full_text


float %380
dcall8	BZ
X
	full_textK
I
G%382 = call float @llvm.fmuladd.f32(float %363, float %381, float %362)
*float8	B

	full_text


float %363
*float8	B

	full_text


float %381
*float8	B

	full_text


float %362
(br8	B 

	full_text

br label %494
ccall8
BY
W
	full_textJ
H
F%384 = call float @llvm.fmuladd.f32(float %310, float %57, float %302)
*float8
B

	full_text


float %310
)float8
B

	full_text

	float %57
*float8
B

	full_text


float %302
ccall8
BY
W
	full_textJ
H
F%385 = call float @llvm.fmuladd.f32(float %314, float %57, float %303)
*float8
B

	full_text


float %314
)float8
B

	full_text

	float %57
*float8
B

	full_text


float %303
ccall8
BY
W
	full_textJ
H
F%386 = call float @llvm.fmuladd.f32(float %318, float %57, float %304)
*float8
B

	full_text


float %318
)float8
B

	full_text

	float %57
*float8
B

	full_text


float %304
(br8
B 

	full_text

br label %494
9fmul8B/
-
	full_text 

%388 = fmul float %314, %314
*float8B

	full_text


float %314
*float8B

	full_text


float %314
dcall8BZ
X
	full_textK
I
G%389 = call float @llvm.fmuladd.f32(float %310, float %310, float %388)
*float8B

	full_text


float %310
*float8B

	full_text


float %310
*float8B

	full_text


float %388
dcall8BZ
X
	full_textK
I
G%390 = call float @llvm.fmuladd.f32(float %318, float %318, float %389)
*float8B

	full_text


float %318
*float8B

	full_text


float %318
*float8B

	full_text


float %389
Gcall8B=
;
	full_text.
,
*%391 = call float @_Z4sqrtf(float %390) #5
*float8B

	full_text


float %390
8sext8B.
,
	full_text

%392 = sext i32 %308 to i64
&i328B

	full_text


i32 %308
^getelementptr8BK
I
	full_text<
:
8%393 = getelementptr inbounds float, float* %2, i64 %392
&i648B

	full_text


i64 %392
Nload8BD
B
	full_text5
3
1%394 = load float, float* %393, align 4, !tbaa !8
,float*8B

	full_text

float* %393
7add8B.
,
	full_text

%395 = add nsw i32 %308, %9
&i328B

	full_text


i32 %308
8sext8B.
,
	full_text

%396 = sext i32 %395 to i64
&i328B

	full_text


i32 %395
^getelementptr8BK
I
	full_text<
:
8%397 = getelementptr inbounds float, float* %2, i64 %396
&i648B

	full_text


i64 %396
Nload8BD
B
	full_text5
3
1%398 = load float, float* %397, align 4, !tbaa !8
,float*8B

	full_text

float* %397
_insertelement8BL
J
	full_text=
;
9%399 = insertelement <2 x float> undef, float %398, i32 0
*float8B

	full_text


float %398
8add8B/
-
	full_text 

%400 = add nsw i32 %308, %34
&i328B

	full_text


i32 %308
%i328B

	full_text
	
i32 %34
8sext8B.
,
	full_text

%401 = sext i32 %400 to i64
&i328B

	full_text


i32 %400
^getelementptr8BK
I
	full_text<
:
8%402 = getelementptr inbounds float, float* %2, i64 %401
&i648B

	full_text


i64 %401
Nload8BD
B
	full_text5
3
1%403 = load float, float* %402, align 4, !tbaa !8
,float*8B

	full_text

float* %402
^insertelement8BK
I
	full_text<
:
8%404 = insertelement <2 x float> %399, float %403, i32 1
6<2 x float>8B#
!
	full_text

<2 x float> %399
*float8B

	full_text


float %403
8add8B/
-
	full_text 

%405 = add nsw i32 %308, %40
&i328B

	full_text


i32 %308
%i328B

	full_text
	
i32 %40
8sext8B.
,
	full_text

%406 = sext i32 %405 to i64
&i328B

	full_text


i32 %405
^getelementptr8BK
I
	full_text<
:
8%407 = getelementptr inbounds float, float* %2, i64 %406
&i648B

	full_text


i64 %406
Nload8BD
B
	full_text5
3
1%408 = load float, float* %407, align 4, !tbaa !8
,float*8B

	full_text

float* %407
8add8B/
-
	full_text 

%409 = add nsw i32 %308, %45
&i328B

	full_text


i32 %308
%i328B

	full_text
	
i32 %45
8sext8B.
,
	full_text

%410 = sext i32 %409 to i64
&i328B

	full_text


i32 %409
^getelementptr8BK
I
	full_text<
:
8%411 = getelementptr inbounds float, float* %2, i64 %410
&i648B

	full_text


i64 %410
Nload8BD
B
	full_text5
3
1%412 = load float, float* %411, align 4, !tbaa !8
,float*8B

	full_text

float* %411
Çcall8Bx
v
	full_texti
g
ecall void @compute_velocity(float %394, <2 x float> %404, float %408, %struct.FLOAT3* nonnull %16) #6
*float8B

	full_text


float %394
6<2 x float>8B#
!
	full_text

<2 x float> %404
*float8B

	full_text


float %408
-struct*8B

	full_text

struct* %16
Oload8BE
C
	full_text6
4
2%413 = load <2 x float>, <2 x float>* %70, align 8
7<2 x float>*8B#
!
	full_text

<2 x float>* %70
Cload8B9
7
	full_text*
(
&%414 = load float, float* %71, align 8
+float*8B

	full_text


float* %71
bcall8BX
V
	full_textI
G
E%415 = call float @compute_speed_sqd(<2 x float> %413, float %414) #6
6<2 x float>8B#
!
	full_text

<2 x float> %413
*float8B

	full_text


float %414
gcall8B]
[
	full_textN
L
J%416 = call float @compute_pressure(float %394, float %412, float %415) #6
*float8B

	full_text


float %394
*float8B

	full_text


float %412
*float8B

	full_text


float %415
acall8BW
U
	full_textH
F
D%417 = call float @compute_speed_of_sound(float %394, float %416) #6
*float8B

	full_text


float %394
*float8B

	full_text


float %416
Oload8BE
C
	full_text6
4
2%418 = load <2 x float>, <2 x float>* %70, align 8
7<2 x float>*8B#
!
	full_text

<2 x float>* %70
Cload8B9
7
	full_text*
(
&%419 = load float, float* %71, align 8
+float*8B

	full_text


float* %71
ùcall8Bí
è
	full_textÅ
˛
˚call void @compute_flux_contribution(float %394, <2 x float> %404, float %408, float %412, float %416, <2 x float> %418, float %419, %struct.FLOAT3* nonnull %17, %struct.FLOAT3* nonnull %18, %struct.FLOAT3* nonnull %19, %struct.FLOAT3* nonnull %20) #6
*float8B

	full_text


float %394
6<2 x float>8B#
!
	full_text

<2 x float> %404
*float8B

	full_text


float %408
*float8B

	full_text


float %412
*float8B

	full_text


float %416
6<2 x float>8B#
!
	full_text

<2 x float> %418
*float8B

	full_text


float %419
-struct*8B

	full_text

struct* %17
-struct*8B

	full_text

struct* %18
-struct*8B

	full_text

struct* %19
-struct*8B

	full_text

struct* %20
Gfmul8B=
;
	full_text.
,
*%420 = fmul float %391, 0xBFC99999A0000000
*float8B

	full_text


float %391
Afmul8B7
5
	full_text(
&
$%421 = fmul float %420, 5.000000e-01
*float8B

	full_text


float %420
Gcall8B=
;
	full_text.
,
*%422 = call float @_Z4sqrtf(float %415) #5
*float8B

	full_text


float %415
8fadd8B.
,
	full_text

%423 = fadd float %56, %422
)float8B

	full_text

	float %56
*float8B

	full_text


float %422
8fadd8B.
,
	full_text

%424 = fadd float %58, %423
)float8B

	full_text

	float %58
*float8B

	full_text


float %423
9fadd8B/
-
	full_text 

%425 = fadd float %417, %424
*float8B

	full_text


float %417
*float8B

	full_text


float %424
9fmul8B/
-
	full_text 

%426 = fmul float %421, %425
*float8B

	full_text


float %421
*float8B

	full_text


float %425
8fsub8B.
,
	full_text

%427 = fsub float %28, %394
)float8B

	full_text

	float %28
*float8B

	full_text


float %394
dcall8BZ
X
	full_textK
I
G%428 = call float @llvm.fmuladd.f32(float %426, float %427, float %305)
*float8B

	full_text


float %426
*float8B

	full_text


float %427
*float8B

	full_text


float %305
8fsub8B.
,
	full_text

%429 = fsub float %49, %412
)float8B

	full_text

	float %49
*float8B

	full_text


float %412
dcall8BZ
X
	full_textK
I
G%430 = call float @llvm.fmuladd.f32(float %426, float %429, float %301)
*float8B

	full_text


float %426
*float8B

	full_text


float %429
*float8B

	full_text


float %301
8fsub8B.
,
	full_text

%431 = fsub float %32, %398
)float8B

	full_text

	float %32
*float8B

	full_text


float %398
dcall8BZ
X
	full_textK
I
G%432 = call float @llvm.fmuladd.f32(float %426, float %431, float %302)
*float8B

	full_text


float %426
*float8B

	full_text


float %431
*float8B

	full_text


float %302
8fsub8B.
,
	full_text

%433 = fsub float %38, %403
)float8B

	full_text

	float %38
*float8B

	full_text


float %403
dcall8BZ
X
	full_textK
I
G%434 = call float @llvm.fmuladd.f32(float %426, float %433, float %303)
*float8B

	full_text


float %426
*float8B

	full_text


float %433
*float8B

	full_text


float %303
8fsub8B.
,
	full_text

%435 = fsub float %44, %408
)float8B

	full_text

	float %44
*float8B

	full_text


float %408
dcall8BZ
X
	full_textK
I
G%436 = call float @llvm.fmuladd.f32(float %426, float %435, float %304)
*float8B

	full_text


float %426
*float8B

	full_text


float %435
*float8B

	full_text


float %304
Afmul8B7
5
	full_text(
&
$%437 = fmul float %310, 5.000000e-01
*float8B

	full_text


float %310
8fadd8B.
,
	full_text

%438 = fadd float %32, %398
)float8B

	full_text

	float %32
*float8B

	full_text


float %398
dcall8BZ
X
	full_textK
I
G%439 = call float @llvm.fmuladd.f32(float %437, float %438, float %428)
*float8B

	full_text


float %437
*float8B

	full_text


float %438
*float8B

	full_text


float %428
Nload8BD
B
	full_text5
3
1%440 = load float, float* %72, align 4, !tbaa !14
+float*8B

	full_text


float* %72
Nload8BD
B
	full_text5
3
1%441 = load float, float* %73, align 4, !tbaa !14
+float*8B

	full_text


float* %73
9fadd8B/
-
	full_text 

%442 = fadd float %440, %441
*float8B

	full_text


float %440
*float8B

	full_text


float %441
dcall8BZ
X
	full_textK
I
G%443 = call float @llvm.fmuladd.f32(float %437, float %442, float %430)
*float8B

	full_text


float %437
*float8B

	full_text


float %442
*float8B

	full_text


float %430
Nload8BD
B
	full_text5
3
1%444 = load float, float* %74, align 4, !tbaa !14
+float*8B

	full_text


float* %74
Nload8BD
B
	full_text5
3
1%445 = load float, float* %75, align 4, !tbaa !14
+float*8B

	full_text


float* %75
9fadd8B/
-
	full_text 

%446 = fadd float %444, %445
*float8B

	full_text


float %444
*float8B

	full_text


float %445
dcall8BZ
X
	full_textK
I
G%447 = call float @llvm.fmuladd.f32(float %437, float %446, float %432)
*float8B

	full_text


float %437
*float8B

	full_text


float %446
*float8B

	full_text


float %432
Nload8BD
B
	full_text5
3
1%448 = load float, float* %76, align 4, !tbaa !14
+float*8B

	full_text


float* %76
Nload8BD
B
	full_text5
3
1%449 = load float, float* %77, align 4, !tbaa !14
+float*8B

	full_text


float* %77
9fadd8B/
-
	full_text 

%450 = fadd float %448, %449
*float8B

	full_text


float %448
*float8B

	full_text


float %449
dcall8BZ
X
	full_textK
I
G%451 = call float @llvm.fmuladd.f32(float %437, float %450, float %434)
*float8B

	full_text


float %437
*float8B

	full_text


float %450
*float8B

	full_text


float %434
Nload8BD
B
	full_text5
3
1%452 = load float, float* %78, align 4, !tbaa !14
+float*8B

	full_text


float* %78
Nload8BD
B
	full_text5
3
1%453 = load float, float* %79, align 4, !tbaa !14
+float*8B

	full_text


float* %79
9fadd8B/
-
	full_text 

%454 = fadd float %452, %453
*float8B

	full_text


float %452
*float8B

	full_text


float %453
dcall8BZ
X
	full_textK
I
G%455 = call float @llvm.fmuladd.f32(float %437, float %454, float %436)
*float8B

	full_text


float %437
*float8B

	full_text


float %454
*float8B

	full_text


float %436
Afmul8B7
5
	full_text(
&
$%456 = fmul float %314, 5.000000e-01
*float8B

	full_text


float %314
8fadd8B.
,
	full_text

%457 = fadd float %38, %403
)float8B

	full_text

	float %38
*float8B

	full_text


float %403
dcall8BZ
X
	full_textK
I
G%458 = call float @llvm.fmuladd.f32(float %456, float %457, float %439)
*float8B

	full_text


float %456
*float8B

	full_text


float %457
*float8B

	full_text


float %439
Nload8BD
B
	full_text5
3
1%459 = load float, float* %80, align 4, !tbaa !16
+float*8B

	full_text


float* %80
Nload8BD
B
	full_text5
3
1%460 = load float, float* %81, align 4, !tbaa !16
+float*8B

	full_text


float* %81
9fadd8B/
-
	full_text 

%461 = fadd float %459, %460
*float8B

	full_text


float %459
*float8B

	full_text


float %460
dcall8BZ
X
	full_textK
I
G%462 = call float @llvm.fmuladd.f32(float %456, float %461, float %443)
*float8B

	full_text


float %456
*float8B

	full_text


float %461
*float8B

	full_text


float %443
Nload8BD
B
	full_text5
3
1%463 = load float, float* %82, align 4, !tbaa !16
+float*8B

	full_text


float* %82
Nload8BD
B
	full_text5
3
1%464 = load float, float* %83, align 4, !tbaa !16
+float*8B

	full_text


float* %83
9fadd8B/
-
	full_text 

%465 = fadd float %463, %464
*float8B

	full_text


float %463
*float8B

	full_text


float %464
dcall8BZ
X
	full_textK
I
G%466 = call float @llvm.fmuladd.f32(float %456, float %465, float %447)
*float8B

	full_text


float %456
*float8B

	full_text


float %465
*float8B

	full_text


float %447
Nload8BD
B
	full_text5
3
1%467 = load float, float* %84, align 4, !tbaa !16
+float*8B

	full_text


float* %84
Nload8BD
B
	full_text5
3
1%468 = load float, float* %85, align 4, !tbaa !16
+float*8B

	full_text


float* %85
9fadd8B/
-
	full_text 

%469 = fadd float %467, %468
*float8B

	full_text


float %467
*float8B

	full_text


float %468
dcall8BZ
X
	full_textK
I
G%470 = call float @llvm.fmuladd.f32(float %456, float %469, float %451)
*float8B

	full_text


float %456
*float8B

	full_text


float %469
*float8B

	full_text


float %451
Nload8BD
B
	full_text5
3
1%471 = load float, float* %86, align 4, !tbaa !16
+float*8B

	full_text


float* %86
Nload8BD
B
	full_text5
3
1%472 = load float, float* %87, align 4, !tbaa !16
+float*8B

	full_text


float* %87
9fadd8B/
-
	full_text 

%473 = fadd float %471, %472
*float8B

	full_text


float %471
*float8B

	full_text


float %472
dcall8BZ
X
	full_textK
I
G%474 = call float @llvm.fmuladd.f32(float %456, float %473, float %455)
*float8B

	full_text


float %456
*float8B

	full_text


float %473
*float8B

	full_text


float %455
Afmul8B7
5
	full_text(
&
$%475 = fmul float %318, 5.000000e-01
*float8B

	full_text


float %318
8fadd8B.
,
	full_text

%476 = fadd float %44, %408
)float8B

	full_text

	float %44
*float8B

	full_text


float %408
dcall8BZ
X
	full_textK
I
G%477 = call float @llvm.fmuladd.f32(float %475, float %476, float %458)
*float8B

	full_text


float %475
*float8B

	full_text


float %476
*float8B

	full_text


float %458
Nload8BD
B
	full_text5
3
1%478 = load float, float* %88, align 4, !tbaa !17
+float*8B

	full_text


float* %88
Nload8BD
B
	full_text5
3
1%479 = load float, float* %89, align 4, !tbaa !17
+float*8B

	full_text


float* %89
9fadd8B/
-
	full_text 

%480 = fadd float %478, %479
*float8B

	full_text


float %478
*float8B

	full_text


float %479
dcall8BZ
X
	full_textK
I
G%481 = call float @llvm.fmuladd.f32(float %475, float %480, float %462)
*float8B

	full_text


float %475
*float8B

	full_text


float %480
*float8B

	full_text


float %462
Nload8BD
B
	full_text5
3
1%482 = load float, float* %90, align 4, !tbaa !17
+float*8B

	full_text


float* %90
Nload8BD
B
	full_text5
3
1%483 = load float, float* %91, align 4, !tbaa !17
+float*8B

	full_text


float* %91
9fadd8B/
-
	full_text 

%484 = fadd float %482, %483
*float8B

	full_text


float %482
*float8B

	full_text


float %483
dcall8BZ
X
	full_textK
I
G%485 = call float @llvm.fmuladd.f32(float %475, float %484, float %466)
*float8B

	full_text


float %475
*float8B

	full_text


float %484
*float8B

	full_text


float %466
Nload8BD
B
	full_text5
3
1%486 = load float, float* %92, align 4, !tbaa !17
+float*8B

	full_text


float* %92
Nload8BD
B
	full_text5
3
1%487 = load float, float* %93, align 4, !tbaa !17
+float*8B

	full_text


float* %93
9fadd8B/
-
	full_text 

%488 = fadd float %486, %487
*float8B

	full_text


float %486
*float8B

	full_text


float %487
dcall8BZ
X
	full_textK
I
G%489 = call float @llvm.fmuladd.f32(float %475, float %488, float %470)
*float8B

	full_text


float %475
*float8B

	full_text


float %488
*float8B

	full_text


float %470
Nload8BD
B
	full_text5
3
1%490 = load float, float* %94, align 4, !tbaa !17
+float*8B

	full_text


float* %94
Nload8BD
B
	full_text5
3
1%491 = load float, float* %95, align 4, !tbaa !17
+float*8B

	full_text


float* %95
9fadd8B/
-
	full_text 

%492 = fadd float %490, %491
*float8B

	full_text


float %490
*float8B

	full_text


float %491
dcall8BZ
X
	full_textK
I
G%493 = call float @llvm.fmuladd.f32(float %475, float %492, float %474)
*float8B

	full_text


float %475
*float8B

	full_text


float %492
*float8B

	full_text


float %474
(br8B 

	full_text

br label %494
kphi8Bb
`
	full_textS
Q
O%495 = phi float [ %481, %387 ], [ %301, %321 ], [ %301, %383 ], [ %370, %322 ]
*float8B

	full_text


float %481
*float8B

	full_text


float %301
*float8B

	full_text


float %301
*float8B

	full_text


float %370
kphi8Bb
`
	full_textS
Q
O%496 = phi float [ %485, %387 ], [ %302, %321 ], [ %384, %383 ], [ %374, %322 ]
*float8B

	full_text


float %485
*float8B

	full_text


float %302
*float8B

	full_text


float %384
*float8B

	full_text


float %374
kphi8Bb
`
	full_textS
Q
O%497 = phi float [ %489, %387 ], [ %303, %321 ], [ %385, %383 ], [ %378, %322 ]
*float8B

	full_text


float %489
*float8B

	full_text


float %303
*float8B

	full_text


float %385
*float8B

	full_text


float %378
kphi8Bb
`
	full_textS
Q
O%498 = phi float [ %493, %387 ], [ %304, %321 ], [ %386, %383 ], [ %382, %322 ]
*float8B

	full_text


float %493
*float8B

	full_text


float %304
*float8B

	full_text


float %386
*float8B

	full_text


float %382
kphi8Bb
`
	full_textS
Q
O%499 = phi float [ %477, %387 ], [ %305, %321 ], [ %305, %383 ], [ %366, %322 ]
*float8B

	full_text


float %477
*float8B

	full_text


float %305
*float8B

	full_text


float %305
*float8B

	full_text


float %366
6shl8B-
+
	full_text

%500 = shl nsw i64 %111, 1
&i648B

	full_text


i64 %111
9add8B0
.
	full_text!

%501 = add nsw i64 %500, %113
&i648B

	full_text


i64 %500
&i648B

	full_text


i64 %113
Zgetelementptr8BG
E
	full_text8
6
4%502 = getelementptr inbounds i32, i32* %0, i64 %501
&i648B

	full_text


i64 %501
Kload8BA
?
	full_text2
0
.%503 = load i32, i32* %502, align 4, !tbaa !12
(i32*8B

	full_text

	i32* %502
^getelementptr8BK
I
	full_text<
:
8%504 = getelementptr inbounds float, float* %1, i64 %501
&i648B

	full_text


i64 %501
Nload8BD
B
	full_text5
3
1%505 = load float, float* %504, align 4, !tbaa !8
,float*8B

	full_text

float* %504
6mul8B-
+
	full_text

%506 = mul nsw i64 %111, 6
&i648B

	full_text


i64 %111
9add8B0
.
	full_text!

%507 = add nsw i64 %506, %113
&i648B

	full_text


i64 %506
&i648B

	full_text


i64 %113
^getelementptr8BK
I
	full_text<
:
8%508 = getelementptr inbounds float, float* %1, i64 %507
&i648B

	full_text


i64 %507
Nload8BD
B
	full_text5
3
1%509 = load float, float* %508, align 4, !tbaa !8
,float*8B

	full_text

float* %508
7mul8B.
,
	full_text

%510 = mul nsw i64 %111, 10
&i648B

	full_text


i64 %111
9add8B0
.
	full_text!

%511 = add nsw i64 %510, %113
&i648B

	full_text


i64 %510
&i648B

	full_text


i64 %113
^getelementptr8BK
I
	full_text<
:
8%512 = getelementptr inbounds float, float* %1, i64 %511
&i648B

	full_text


i64 %511
Nload8BD
B
	full_text5
3
1%513 = load float, float* %512, align 4, !tbaa !8
,float*8B

	full_text

float* %512
9icmp8B/
-
	full_text 

%514 = icmp sgt i32 %503, -1
&i328B

	full_text


i32 %503
=br8B5
3
	full_text&
$
"br i1 %514, label %581, label %515
$i18B

	full_text
	
i1 %514
nswitch8Bb
`
	full_textS
Q
Oswitch i32 %503, label %688 [
    i32 -1, label %577
    i32 -2, label %516
  ]
&i328B

	full_text


i32 %503
Afmul8B7
5
	full_text(
&
$%517 = fmul float %505, 5.000000e-01
*float8B

	full_text


float %505
Mload8BC
A
	full_text4
2
0%518 = load float, float* %96, align 4, !tbaa !8
+float*8B

	full_text


float* %96
8fadd8B.
,
	full_text

%519 = fadd float %32, %518
)float8B

	full_text

	float %32
*float8B

	full_text


float %518
dcall8BZ
X
	full_textK
I
G%520 = call float @llvm.fmuladd.f32(float %517, float %519, float %499)
*float8B

	full_text


float %517
*float8B

	full_text


float %519
*float8B

	full_text


float %499
Nload8BD
B
	full_text5
3
1%521 = load float, float* %97, align 4, !tbaa !14
+float*8B

	full_text


float* %97
Nload8BD
B
	full_text5
3
1%522 = load float, float* %73, align 4, !tbaa !14
+float*8B

	full_text


float* %73
9fadd8B/
-
	full_text 

%523 = fadd float %521, %522
*float8B

	full_text


float %521
*float8B

	full_text


float %522
dcall8BZ
X
	full_textK
I
G%524 = call float @llvm.fmuladd.f32(float %517, float %523, float %495)
*float8B

	full_text


float %517
*float8B

	full_text


float %523
*float8B

	full_text


float %495
Nload8BD
B
	full_text5
3
1%525 = load float, float* %98, align 4, !tbaa !14
+float*8B

	full_text


float* %98
Nload8BD
B
	full_text5
3
1%526 = load float, float* %75, align 4, !tbaa !14
+float*8B

	full_text


float* %75
9fadd8B/
-
	full_text 

%527 = fadd float %525, %526
*float8B

	full_text


float %525
*float8B

	full_text


float %526
dcall8BZ
X
	full_textK
I
G%528 = call float @llvm.fmuladd.f32(float %517, float %527, float %496)
*float8B

	full_text


float %517
*float8B

	full_text


float %527
*float8B

	full_text


float %496
Nload8BD
B
	full_text5
3
1%529 = load float, float* %99, align 4, !tbaa !14
+float*8B

	full_text


float* %99
Nload8BD
B
	full_text5
3
1%530 = load float, float* %77, align 4, !tbaa !14
+float*8B

	full_text


float* %77
9fadd8B/
-
	full_text 

%531 = fadd float %529, %530
*float8B

	full_text


float %529
*float8B

	full_text


float %530
dcall8BZ
X
	full_textK
I
G%532 = call float @llvm.fmuladd.f32(float %517, float %531, float %497)
*float8B

	full_text


float %517
*float8B

	full_text


float %531
*float8B

	full_text


float %497
Oload8BE
C
	full_text6
4
2%533 = load float, float* %100, align 4, !tbaa !14
,float*8B

	full_text

float* %100
Nload8BD
B
	full_text5
3
1%534 = load float, float* %79, align 4, !tbaa !14
+float*8B

	full_text


float* %79
9fadd8B/
-
	full_text 

%535 = fadd float %533, %534
*float8B

	full_text


float %533
*float8B

	full_text


float %534
dcall8BZ
X
	full_textK
I
G%536 = call float @llvm.fmuladd.f32(float %517, float %535, float %498)
*float8B

	full_text


float %517
*float8B

	full_text


float %535
*float8B

	full_text


float %498
Afmul8B7
5
	full_text(
&
$%537 = fmul float %509, 5.000000e-01
*float8B

	full_text


float %509
Nload8BD
B
	full_text5
3
1%538 = load float, float* %101, align 4, !tbaa !8
,float*8B

	full_text

float* %101
8fadd8B.
,
	full_text

%539 = fadd float %38, %538
)float8B

	full_text

	float %38
*float8B

	full_text


float %538
dcall8BZ
X
	full_textK
I
G%540 = call float @llvm.fmuladd.f32(float %537, float %539, float %520)
*float8B

	full_text


float %537
*float8B

	full_text


float %539
*float8B

	full_text


float %520
Oload8BE
C
	full_text6
4
2%541 = load float, float* %102, align 4, !tbaa !16
,float*8B

	full_text

float* %102
Nload8BD
B
	full_text5
3
1%542 = load float, float* %81, align 4, !tbaa !16
+float*8B

	full_text


float* %81
9fadd8B/
-
	full_text 

%543 = fadd float %541, %542
*float8B

	full_text


float %541
*float8B

	full_text


float %542
dcall8BZ
X
	full_textK
I
G%544 = call float @llvm.fmuladd.f32(float %537, float %543, float %524)
*float8B

	full_text


float %537
*float8B

	full_text


float %543
*float8B

	full_text


float %524
Oload8BE
C
	full_text6
4
2%545 = load float, float* %103, align 4, !tbaa !16
,float*8B

	full_text

float* %103
Nload8BD
B
	full_text5
3
1%546 = load float, float* %83, align 4, !tbaa !16
+float*8B

	full_text


float* %83
9fadd8B/
-
	full_text 

%547 = fadd float %545, %546
*float8B

	full_text


float %545
*float8B

	full_text


float %546
dcall8BZ
X
	full_textK
I
G%548 = call float @llvm.fmuladd.f32(float %537, float %547, float %528)
*float8B

	full_text


float %537
*float8B

	full_text


float %547
*float8B

	full_text


float %528
Oload8BE
C
	full_text6
4
2%549 = load float, float* %104, align 4, !tbaa !16
,float*8B

	full_text

float* %104
Nload8BD
B
	full_text5
3
1%550 = load float, float* %85, align 4, !tbaa !16
+float*8B

	full_text


float* %85
9fadd8B/
-
	full_text 

%551 = fadd float %549, %550
*float8B

	full_text


float %549
*float8B

	full_text


float %550
dcall8BZ
X
	full_textK
I
G%552 = call float @llvm.fmuladd.f32(float %537, float %551, float %532)
*float8B

	full_text


float %537
*float8B

	full_text


float %551
*float8B

	full_text


float %532
Oload8BE
C
	full_text6
4
2%553 = load float, float* %105, align 4, !tbaa !16
,float*8B

	full_text

float* %105
Nload8BD
B
	full_text5
3
1%554 = load float, float* %87, align 4, !tbaa !16
+float*8B

	full_text


float* %87
9fadd8B/
-
	full_text 

%555 = fadd float %553, %554
*float8B

	full_text


float %553
*float8B

	full_text


float %554
dcall8BZ
X
	full_textK
I
G%556 = call float @llvm.fmuladd.f32(float %537, float %555, float %536)
*float8B

	full_text


float %537
*float8B

	full_text


float %555
*float8B

	full_text


float %536
Afmul8B7
5
	full_text(
&
$%557 = fmul float %513, 5.000000e-01
*float8B

	full_text


float %513
Nload8BD
B
	full_text5
3
1%558 = load float, float* %106, align 4, !tbaa !8
,float*8B

	full_text

float* %106
8fadd8B.
,
	full_text

%559 = fadd float %44, %558
)float8B

	full_text

	float %44
*float8B

	full_text


float %558
dcall8BZ
X
	full_textK
I
G%560 = call float @llvm.fmuladd.f32(float %557, float %559, float %540)
*float8B

	full_text


float %557
*float8B

	full_text


float %559
*float8B

	full_text


float %540
Oload8BE
C
	full_text6
4
2%561 = load float, float* %107, align 4, !tbaa !17
,float*8B

	full_text

float* %107
Nload8BD
B
	full_text5
3
1%562 = load float, float* %89, align 4, !tbaa !17
+float*8B

	full_text


float* %89
9fadd8B/
-
	full_text 

%563 = fadd float %561, %562
*float8B

	full_text


float %561
*float8B

	full_text


float %562
dcall8BZ
X
	full_textK
I
G%564 = call float @llvm.fmuladd.f32(float %557, float %563, float %544)
*float8B

	full_text


float %557
*float8B

	full_text


float %563
*float8B

	full_text


float %544
Oload8BE
C
	full_text6
4
2%565 = load float, float* %108, align 4, !tbaa !17
,float*8B

	full_text

float* %108
Nload8BD
B
	full_text5
3
1%566 = load float, float* %91, align 4, !tbaa !17
+float*8B

	full_text


float* %91
9fadd8B/
-
	full_text 

%567 = fadd float %565, %566
*float8B

	full_text


float %565
*float8B

	full_text


float %566
dcall8BZ
X
	full_textK
I
G%568 = call float @llvm.fmuladd.f32(float %557, float %567, float %548)
*float8B

	full_text


float %557
*float8B

	full_text


float %567
*float8B

	full_text


float %548
Oload8BE
C
	full_text6
4
2%569 = load float, float* %109, align 4, !tbaa !17
,float*8B

	full_text

float* %109
Nload8BD
B
	full_text5
3
1%570 = load float, float* %93, align 4, !tbaa !17
+float*8B

	full_text


float* %93
9fadd8B/
-
	full_text 

%571 = fadd float %569, %570
*float8B

	full_text


float %569
*float8B

	full_text


float %570
dcall8BZ
X
	full_textK
I
G%572 = call float @llvm.fmuladd.f32(float %557, float %571, float %552)
*float8B

	full_text


float %557
*float8B

	full_text


float %571
*float8B

	full_text


float %552
Oload8BE
C
	full_text6
4
2%573 = load float, float* %110, align 4, !tbaa !17
,float*8B

	full_text

float* %110
Nload8BD
B
	full_text5
3
1%574 = load float, float* %95, align 4, !tbaa !17
+float*8B

	full_text


float* %95
9fadd8B/
-
	full_text 

%575 = fadd float %573, %574
*float8B

	full_text


float %573
*float8B

	full_text


float %574
dcall8BZ
X
	full_textK
I
G%576 = call float @llvm.fmuladd.f32(float %557, float %575, float %556)
*float8B

	full_text


float %557
*float8B

	full_text


float %575
*float8B

	full_text


float %556
(br8B 

	full_text

br label %688
ccall8BY
W
	full_textJ
H
F%578 = call float @llvm.fmuladd.f32(float %505, float %57, float %496)
*float8B

	full_text


float %505
)float8B

	full_text

	float %57
*float8B

	full_text


float %496
ccall8BY
W
	full_textJ
H
F%579 = call float @llvm.fmuladd.f32(float %509, float %57, float %497)
*float8B

	full_text


float %509
)float8B

	full_text

	float %57
*float8B

	full_text


float %497
ccall8BY
W
	full_textJ
H
F%580 = call float @llvm.fmuladd.f32(float %513, float %57, float %498)
*float8B

	full_text


float %513
)float8B

	full_text

	float %57
*float8B

	full_text


float %498
(br8B 

	full_text

br label %688
9fmul8B/
-
	full_text 

%582 = fmul float %509, %509
*float8B

	full_text


float %509
*float8B

	full_text


float %509
dcall8BZ
X
	full_textK
I
G%583 = call float @llvm.fmuladd.f32(float %505, float %505, float %582)
*float8B

	full_text


float %505
*float8B

	full_text


float %505
*float8B

	full_text


float %582
dcall8BZ
X
	full_textK
I
G%584 = call float @llvm.fmuladd.f32(float %513, float %513, float %583)
*float8B

	full_text


float %513
*float8B

	full_text


float %513
*float8B

	full_text


float %583
Gcall8B=
;
	full_text.
,
*%585 = call float @_Z4sqrtf(float %584) #5
*float8B

	full_text


float %584
8sext8B.
,
	full_text

%586 = sext i32 %503 to i64
&i328B

	full_text


i32 %503
^getelementptr8BK
I
	full_text<
:
8%587 = getelementptr inbounds float, float* %2, i64 %586
&i648B

	full_text


i64 %586
Nload8BD
B
	full_text5
3
1%588 = load float, float* %587, align 4, !tbaa !8
,float*8B

	full_text

float* %587
7add8B.
,
	full_text

%589 = add nsw i32 %503, %9
&i328B

	full_text


i32 %503
8sext8B.
,
	full_text

%590 = sext i32 %589 to i64
&i328B

	full_text


i32 %589
^getelementptr8BK
I
	full_text<
:
8%591 = getelementptr inbounds float, float* %2, i64 %590
&i648B

	full_text


i64 %590
Nload8BD
B
	full_text5
3
1%592 = load float, float* %591, align 4, !tbaa !8
,float*8B

	full_text

float* %591
_insertelement8BL
J
	full_text=
;
9%593 = insertelement <2 x float> undef, float %592, i32 0
*float8B

	full_text


float %592
8add8B/
-
	full_text 

%594 = add nsw i32 %503, %34
&i328B

	full_text


i32 %503
%i328B

	full_text
	
i32 %34
8sext8B.
,
	full_text

%595 = sext i32 %594 to i64
&i328B

	full_text


i32 %594
^getelementptr8BK
I
	full_text<
:
8%596 = getelementptr inbounds float, float* %2, i64 %595
&i648B

	full_text


i64 %595
Nload8BD
B
	full_text5
3
1%597 = load float, float* %596, align 4, !tbaa !8
,float*8B

	full_text

float* %596
^insertelement8BK
I
	full_text<
:
8%598 = insertelement <2 x float> %593, float %597, i32 1
6<2 x float>8B#
!
	full_text

<2 x float> %593
*float8B

	full_text


float %597
8add8B/
-
	full_text 

%599 = add nsw i32 %503, %40
&i328B

	full_text


i32 %503
%i328B

	full_text
	
i32 %40
8sext8B.
,
	full_text

%600 = sext i32 %599 to i64
&i328B

	full_text


i32 %599
^getelementptr8BK
I
	full_text<
:
8%601 = getelementptr inbounds float, float* %2, i64 %600
&i648B

	full_text


i64 %600
Nload8BD
B
	full_text5
3
1%602 = load float, float* %601, align 4, !tbaa !8
,float*8B

	full_text

float* %601
8add8B/
-
	full_text 

%603 = add nsw i32 %503, %45
&i328B

	full_text


i32 %503
%i328B

	full_text
	
i32 %45
8sext8B.
,
	full_text

%604 = sext i32 %603 to i64
&i328B

	full_text


i32 %603
^getelementptr8BK
I
	full_text<
:
8%605 = getelementptr inbounds float, float* %2, i64 %604
&i648B

	full_text


i64 %604
Nload8BD
B
	full_text5
3
1%606 = load float, float* %605, align 4, !tbaa !8
,float*8B

	full_text

float* %605
Çcall8Bx
v
	full_texti
g
ecall void @compute_velocity(float %588, <2 x float> %598, float %602, %struct.FLOAT3* nonnull %16) #6
*float8B

	full_text


float %588
6<2 x float>8B#
!
	full_text

<2 x float> %598
*float8B

	full_text


float %602
-struct*8B

	full_text

struct* %16
Oload8BE
C
	full_text6
4
2%607 = load <2 x float>, <2 x float>* %70, align 8
7<2 x float>*8B#
!
	full_text

<2 x float>* %70
Cload8B9
7
	full_text*
(
&%608 = load float, float* %71, align 8
+float*8B

	full_text


float* %71
bcall8BX
V
	full_textI
G
E%609 = call float @compute_speed_sqd(<2 x float> %607, float %608) #6
6<2 x float>8B#
!
	full_text

<2 x float> %607
*float8B

	full_text


float %608
gcall8B]
[
	full_textN
L
J%610 = call float @compute_pressure(float %588, float %606, float %609) #6
*float8B

	full_text


float %588
*float8B

	full_text


float %606
*float8B

	full_text


float %609
acall8BW
U
	full_textH
F
D%611 = call float @compute_speed_of_sound(float %588, float %610) #6
*float8B

	full_text


float %588
*float8B

	full_text


float %610
Oload8BE
C
	full_text6
4
2%612 = load <2 x float>, <2 x float>* %70, align 8
7<2 x float>*8B#
!
	full_text

<2 x float>* %70
Cload8B9
7
	full_text*
(
&%613 = load float, float* %71, align 8
+float*8B

	full_text


float* %71
ùcall8Bí
è
	full_textÅ
˛
˚call void @compute_flux_contribution(float %588, <2 x float> %598, float %602, float %606, float %610, <2 x float> %612, float %613, %struct.FLOAT3* nonnull %17, %struct.FLOAT3* nonnull %18, %struct.FLOAT3* nonnull %19, %struct.FLOAT3* nonnull %20) #6
*float8B

	full_text


float %588
6<2 x float>8B#
!
	full_text

<2 x float> %598
*float8B

	full_text


float %602
*float8B

	full_text


float %606
*float8B

	full_text


float %610
6<2 x float>8B#
!
	full_text

<2 x float> %612
*float8B

	full_text


float %613
-struct*8B

	full_text

struct* %17
-struct*8B

	full_text

struct* %18
-struct*8B

	full_text

struct* %19
-struct*8B

	full_text

struct* %20
Gfmul8B=
;
	full_text.
,
*%614 = fmul float %585, 0xBFC99999A0000000
*float8B

	full_text


float %585
Afmul8B7
5
	full_text(
&
$%615 = fmul float %614, 5.000000e-01
*float8B

	full_text


float %614
Gcall8B=
;
	full_text.
,
*%616 = call float @_Z4sqrtf(float %609) #5
*float8B

	full_text


float %609
8fadd8B.
,
	full_text

%617 = fadd float %56, %616
)float8B

	full_text

	float %56
*float8B

	full_text


float %616
8fadd8B.
,
	full_text

%618 = fadd float %58, %617
)float8B

	full_text

	float %58
*float8B

	full_text


float %617
9fadd8B/
-
	full_text 

%619 = fadd float %611, %618
*float8B

	full_text


float %611
*float8B

	full_text


float %618
9fmul8B/
-
	full_text 

%620 = fmul float %615, %619
*float8B

	full_text


float %615
*float8B

	full_text


float %619
8fsub8B.
,
	full_text

%621 = fsub float %28, %588
)float8B

	full_text

	float %28
*float8B

	full_text


float %588
dcall8BZ
X
	full_textK
I
G%622 = call float @llvm.fmuladd.f32(float %620, float %621, float %499)
*float8B

	full_text


float %620
*float8B

	full_text


float %621
*float8B

	full_text


float %499
8fsub8B.
,
	full_text

%623 = fsub float %49, %606
)float8B

	full_text

	float %49
*float8B

	full_text


float %606
dcall8BZ
X
	full_textK
I
G%624 = call float @llvm.fmuladd.f32(float %620, float %623, float %495)
*float8B

	full_text


float %620
*float8B

	full_text


float %623
*float8B

	full_text


float %495
8fsub8B.
,
	full_text

%625 = fsub float %32, %592
)float8B

	full_text

	float %32
*float8B

	full_text


float %592
dcall8BZ
X
	full_textK
I
G%626 = call float @llvm.fmuladd.f32(float %620, float %625, float %496)
*float8B

	full_text


float %620
*float8B

	full_text


float %625
*float8B

	full_text


float %496
8fsub8B.
,
	full_text

%627 = fsub float %38, %597
)float8B

	full_text

	float %38
*float8B

	full_text


float %597
dcall8BZ
X
	full_textK
I
G%628 = call float @llvm.fmuladd.f32(float %620, float %627, float %497)
*float8B

	full_text


float %620
*float8B

	full_text


float %627
*float8B

	full_text


float %497
8fsub8B.
,
	full_text

%629 = fsub float %44, %602
)float8B

	full_text

	float %44
*float8B

	full_text


float %602
dcall8BZ
X
	full_textK
I
G%630 = call float @llvm.fmuladd.f32(float %620, float %629, float %498)
*float8B

	full_text


float %620
*float8B

	full_text


float %629
*float8B

	full_text


float %498
Afmul8B7
5
	full_text(
&
$%631 = fmul float %505, 5.000000e-01
*float8B

	full_text


float %505
8fadd8B.
,
	full_text

%632 = fadd float %32, %592
)float8B

	full_text

	float %32
*float8B

	full_text


float %592
dcall8BZ
X
	full_textK
I
G%633 = call float @llvm.fmuladd.f32(float %631, float %632, float %622)
*float8B

	full_text


float %631
*float8B

	full_text


float %632
*float8B

	full_text


float %622
Nload8BD
B
	full_text5
3
1%634 = load float, float* %72, align 4, !tbaa !14
+float*8B

	full_text


float* %72
Nload8BD
B
	full_text5
3
1%635 = load float, float* %73, align 4, !tbaa !14
+float*8B

	full_text


float* %73
9fadd8B/
-
	full_text 

%636 = fadd float %634, %635
*float8B

	full_text


float %634
*float8B

	full_text


float %635
dcall8BZ
X
	full_textK
I
G%637 = call float @llvm.fmuladd.f32(float %631, float %636, float %624)
*float8B

	full_text


float %631
*float8B

	full_text


float %636
*float8B

	full_text


float %624
Nload8BD
B
	full_text5
3
1%638 = load float, float* %74, align 4, !tbaa !14
+float*8B

	full_text


float* %74
Nload8BD
B
	full_text5
3
1%639 = load float, float* %75, align 4, !tbaa !14
+float*8B

	full_text


float* %75
9fadd8B/
-
	full_text 

%640 = fadd float %638, %639
*float8B

	full_text


float %638
*float8B

	full_text


float %639
dcall8BZ
X
	full_textK
I
G%641 = call float @llvm.fmuladd.f32(float %631, float %640, float %626)
*float8B

	full_text


float %631
*float8B

	full_text


float %640
*float8B

	full_text


float %626
Nload8BD
B
	full_text5
3
1%642 = load float, float* %76, align 4, !tbaa !14
+float*8B

	full_text


float* %76
Nload8BD
B
	full_text5
3
1%643 = load float, float* %77, align 4, !tbaa !14
+float*8B

	full_text


float* %77
9fadd8B/
-
	full_text 

%644 = fadd float %642, %643
*float8B

	full_text


float %642
*float8B

	full_text


float %643
dcall8BZ
X
	full_textK
I
G%645 = call float @llvm.fmuladd.f32(float %631, float %644, float %628)
*float8B

	full_text


float %631
*float8B

	full_text


float %644
*float8B

	full_text


float %628
Nload8BD
B
	full_text5
3
1%646 = load float, float* %78, align 4, !tbaa !14
+float*8B

	full_text


float* %78
Nload8BD
B
	full_text5
3
1%647 = load float, float* %79, align 4, !tbaa !14
+float*8B

	full_text


float* %79
9fadd8B/
-
	full_text 

%648 = fadd float %646, %647
*float8B

	full_text


float %646
*float8B

	full_text


float %647
dcall8BZ
X
	full_textK
I
G%649 = call float @llvm.fmuladd.f32(float %631, float %648, float %630)
*float8B

	full_text


float %631
*float8B

	full_text


float %648
*float8B

	full_text


float %630
Afmul8B7
5
	full_text(
&
$%650 = fmul float %509, 5.000000e-01
*float8B

	full_text


float %509
8fadd8B.
,
	full_text

%651 = fadd float %38, %597
)float8B

	full_text

	float %38
*float8B

	full_text


float %597
dcall8BZ
X
	full_textK
I
G%652 = call float @llvm.fmuladd.f32(float %650, float %651, float %633)
*float8B

	full_text


float %650
*float8B

	full_text


float %651
*float8B

	full_text


float %633
Nload8BD
B
	full_text5
3
1%653 = load float, float* %80, align 4, !tbaa !16
+float*8B

	full_text


float* %80
Nload8BD
B
	full_text5
3
1%654 = load float, float* %81, align 4, !tbaa !16
+float*8B

	full_text


float* %81
9fadd8B/
-
	full_text 

%655 = fadd float %653, %654
*float8B

	full_text


float %653
*float8B

	full_text


float %654
dcall8BZ
X
	full_textK
I
G%656 = call float @llvm.fmuladd.f32(float %650, float %655, float %637)
*float8B

	full_text


float %650
*float8B

	full_text


float %655
*float8B

	full_text


float %637
Nload8BD
B
	full_text5
3
1%657 = load float, float* %82, align 4, !tbaa !16
+float*8B

	full_text


float* %82
Nload8BD
B
	full_text5
3
1%658 = load float, float* %83, align 4, !tbaa !16
+float*8B

	full_text


float* %83
9fadd8B/
-
	full_text 

%659 = fadd float %657, %658
*float8B

	full_text


float %657
*float8B

	full_text


float %658
dcall8BZ
X
	full_textK
I
G%660 = call float @llvm.fmuladd.f32(float %650, float %659, float %641)
*float8B

	full_text


float %650
*float8B

	full_text


float %659
*float8B

	full_text


float %641
Nload8BD
B
	full_text5
3
1%661 = load float, float* %84, align 4, !tbaa !16
+float*8B

	full_text


float* %84
Nload8BD
B
	full_text5
3
1%662 = load float, float* %85, align 4, !tbaa !16
+float*8B

	full_text


float* %85
9fadd8B/
-
	full_text 

%663 = fadd float %661, %662
*float8B

	full_text


float %661
*float8B

	full_text


float %662
dcall8BZ
X
	full_textK
I
G%664 = call float @llvm.fmuladd.f32(float %650, float %663, float %645)
*float8B

	full_text


float %650
*float8B

	full_text


float %663
*float8B

	full_text


float %645
Nload8BD
B
	full_text5
3
1%665 = load float, float* %86, align 4, !tbaa !16
+float*8B

	full_text


float* %86
Nload8BD
B
	full_text5
3
1%666 = load float, float* %87, align 4, !tbaa !16
+float*8B

	full_text


float* %87
9fadd8B/
-
	full_text 

%667 = fadd float %665, %666
*float8B

	full_text


float %665
*float8B

	full_text


float %666
dcall8BZ
X
	full_textK
I
G%668 = call float @llvm.fmuladd.f32(float %650, float %667, float %649)
*float8B

	full_text


float %650
*float8B

	full_text


float %667
*float8B

	full_text


float %649
Afmul8B7
5
	full_text(
&
$%669 = fmul float %513, 5.000000e-01
*float8B

	full_text


float %513
8fadd8B.
,
	full_text

%670 = fadd float %44, %602
)float8B

	full_text

	float %44
*float8B

	full_text


float %602
dcall8BZ
X
	full_textK
I
G%671 = call float @llvm.fmuladd.f32(float %669, float %670, float %652)
*float8B

	full_text


float %669
*float8B

	full_text


float %670
*float8B

	full_text


float %652
Nload8BD
B
	full_text5
3
1%672 = load float, float* %88, align 4, !tbaa !17
+float*8B

	full_text


float* %88
Nload8BD
B
	full_text5
3
1%673 = load float, float* %89, align 4, !tbaa !17
+float*8B

	full_text


float* %89
9fadd8B/
-
	full_text 

%674 = fadd float %672, %673
*float8B

	full_text


float %672
*float8B

	full_text


float %673
dcall8BZ
X
	full_textK
I
G%675 = call float @llvm.fmuladd.f32(float %669, float %674, float %656)
*float8B

	full_text


float %669
*float8B

	full_text


float %674
*float8B

	full_text


float %656
Nload8BD
B
	full_text5
3
1%676 = load float, float* %90, align 4, !tbaa !17
+float*8B

	full_text


float* %90
Nload8BD
B
	full_text5
3
1%677 = load float, float* %91, align 4, !tbaa !17
+float*8B

	full_text


float* %91
9fadd8B/
-
	full_text 

%678 = fadd float %676, %677
*float8B

	full_text


float %676
*float8B

	full_text


float %677
dcall8BZ
X
	full_textK
I
G%679 = call float @llvm.fmuladd.f32(float %669, float %678, float %660)
*float8B

	full_text


float %669
*float8B

	full_text


float %678
*float8B

	full_text


float %660
Nload8BD
B
	full_text5
3
1%680 = load float, float* %92, align 4, !tbaa !17
+float*8B

	full_text


float* %92
Nload8BD
B
	full_text5
3
1%681 = load float, float* %93, align 4, !tbaa !17
+float*8B

	full_text


float* %93
9fadd8B/
-
	full_text 

%682 = fadd float %680, %681
*float8B

	full_text


float %680
*float8B

	full_text


float %681
dcall8BZ
X
	full_textK
I
G%683 = call float @llvm.fmuladd.f32(float %669, float %682, float %664)
*float8B

	full_text


float %669
*float8B

	full_text


float %682
*float8B

	full_text


float %664
Nload8BD
B
	full_text5
3
1%684 = load float, float* %94, align 4, !tbaa !17
+float*8B

	full_text


float* %94
Nload8BD
B
	full_text5
3
1%685 = load float, float* %95, align 4, !tbaa !17
+float*8B

	full_text


float* %95
9fadd8B/
-
	full_text 

%686 = fadd float %684, %685
*float8B

	full_text


float %684
*float8B

	full_text


float %685
dcall8BZ
X
	full_textK
I
G%687 = call float @llvm.fmuladd.f32(float %669, float %686, float %668)
*float8B

	full_text


float %669
*float8B

	full_text


float %686
*float8B

	full_text


float %668
(br8B 

	full_text

br label %688
kphi8Bb
`
	full_textS
Q
O%689 = phi float [ %675, %581 ], [ %495, %515 ], [ %495, %577 ], [ %564, %516 ]
*float8B

	full_text


float %675
*float8B

	full_text


float %495
*float8B

	full_text


float %495
*float8B

	full_text


float %564
kphi8Bb
`
	full_textS
Q
O%690 = phi float [ %679, %581 ], [ %496, %515 ], [ %578, %577 ], [ %568, %516 ]
*float8B

	full_text


float %679
*float8B

	full_text


float %496
*float8B

	full_text


float %578
*float8B

	full_text


float %568
kphi8Bb
`
	full_textS
Q
O%691 = phi float [ %683, %581 ], [ %497, %515 ], [ %579, %577 ], [ %572, %516 ]
*float8B

	full_text


float %683
*float8B

	full_text


float %497
*float8B

	full_text


float %579
*float8B

	full_text


float %572
kphi8Bb
`
	full_textS
Q
O%692 = phi float [ %687, %581 ], [ %498, %515 ], [ %580, %577 ], [ %576, %516 ]
*float8B

	full_text


float %687
*float8B

	full_text


float %498
*float8B

	full_text


float %580
*float8B

	full_text


float %576
kphi8Bb
`
	full_textS
Q
O%693 = phi float [ %671, %581 ], [ %499, %515 ], [ %499, %577 ], [ %560, %516 ]
*float8B

	full_text


float %671
*float8B

	full_text


float %499
*float8B

	full_text


float %499
*float8B

	full_text


float %560
6mul8B-
+
	full_text

%694 = mul nsw i64 %111, 3
&i648B

	full_text


i64 %111
9add8B0
.
	full_text!

%695 = add nsw i64 %694, %113
&i648B

	full_text


i64 %694
&i648B

	full_text


i64 %113
Zgetelementptr8BG
E
	full_text8
6
4%696 = getelementptr inbounds i32, i32* %0, i64 %695
&i648B

	full_text


i64 %695
Kload8BA
?
	full_text2
0
.%697 = load i32, i32* %696, align 4, !tbaa !12
(i32*8B

	full_text

	i32* %696
^getelementptr8BK
I
	full_text<
:
8%698 = getelementptr inbounds float, float* %1, i64 %695
&i648B

	full_text


i64 %695
Nload8BD
B
	full_text5
3
1%699 = load float, float* %698, align 4, !tbaa !8
,float*8B

	full_text

float* %698
6mul8B-
+
	full_text

%700 = mul nsw i64 %111, 7
&i648B

	full_text


i64 %111
9add8B0
.
	full_text!

%701 = add nsw i64 %700, %113
&i648B

	full_text


i64 %700
&i648B

	full_text


i64 %113
^getelementptr8BK
I
	full_text<
:
8%702 = getelementptr inbounds float, float* %1, i64 %701
&i648B

	full_text


i64 %701
Nload8BD
B
	full_text5
3
1%703 = load float, float* %702, align 4, !tbaa !8
,float*8B

	full_text

float* %702
7mul8B.
,
	full_text

%704 = mul nsw i64 %111, 11
&i648B

	full_text


i64 %111
9add8B0
.
	full_text!

%705 = add nsw i64 %704, %113
&i648B

	full_text


i64 %704
&i648B

	full_text


i64 %113
^getelementptr8BK
I
	full_text<
:
8%706 = getelementptr inbounds float, float* %1, i64 %705
&i648B

	full_text


i64 %705
Nload8BD
B
	full_text5
3
1%707 = load float, float* %706, align 4, !tbaa !8
,float*8B

	full_text

float* %706
9icmp8B/
-
	full_text 

%708 = icmp sgt i32 %697, -1
&i328B

	full_text


i32 %697
=br8B5
3
	full_text&
$
"br i1 %708, label %775, label %709
$i18B

	full_text
	
i1 %708
nswitch8Bb
`
	full_textS
Q
Oswitch i32 %697, label %882 [
    i32 -1, label %771
    i32 -2, label %710
  ]
&i328B

	full_text


i32 %697
Afmul8B7
5
	full_text(
&
$%711 = fmul float %699, 5.000000e-01
*float8B

	full_text


float %699
Mload8BC
A
	full_text4
2
0%712 = load float, float* %96, align 4, !tbaa !8
+float*8B

	full_text


float* %96
8fadd8B.
,
	full_text

%713 = fadd float %32, %712
)float8B

	full_text

	float %32
*float8B

	full_text


float %712
dcall8BZ
X
	full_textK
I
G%714 = call float @llvm.fmuladd.f32(float %711, float %713, float %693)
*float8B

	full_text


float %711
*float8B

	full_text


float %713
*float8B

	full_text


float %693
Nload8BD
B
	full_text5
3
1%715 = load float, float* %97, align 4, !tbaa !14
+float*8B

	full_text


float* %97
Nload8BD
B
	full_text5
3
1%716 = load float, float* %73, align 4, !tbaa !14
+float*8B

	full_text


float* %73
9fadd8B/
-
	full_text 

%717 = fadd float %715, %716
*float8B

	full_text


float %715
*float8B

	full_text


float %716
dcall8BZ
X
	full_textK
I
G%718 = call float @llvm.fmuladd.f32(float %711, float %717, float %689)
*float8B

	full_text


float %711
*float8B

	full_text


float %717
*float8B

	full_text


float %689
Nload8BD
B
	full_text5
3
1%719 = load float, float* %98, align 4, !tbaa !14
+float*8B

	full_text


float* %98
Nload8BD
B
	full_text5
3
1%720 = load float, float* %75, align 4, !tbaa !14
+float*8B

	full_text


float* %75
9fadd8B/
-
	full_text 

%721 = fadd float %719, %720
*float8B

	full_text


float %719
*float8B

	full_text


float %720
dcall8BZ
X
	full_textK
I
G%722 = call float @llvm.fmuladd.f32(float %711, float %721, float %690)
*float8B

	full_text


float %711
*float8B

	full_text


float %721
*float8B

	full_text


float %690
Nload8BD
B
	full_text5
3
1%723 = load float, float* %99, align 4, !tbaa !14
+float*8B

	full_text


float* %99
Nload8BD
B
	full_text5
3
1%724 = load float, float* %77, align 4, !tbaa !14
+float*8B

	full_text


float* %77
9fadd8B/
-
	full_text 

%725 = fadd float %723, %724
*float8B

	full_text


float %723
*float8B

	full_text


float %724
dcall8BZ
X
	full_textK
I
G%726 = call float @llvm.fmuladd.f32(float %711, float %725, float %691)
*float8B

	full_text


float %711
*float8B

	full_text


float %725
*float8B

	full_text


float %691
Oload8BE
C
	full_text6
4
2%727 = load float, float* %100, align 4, !tbaa !14
,float*8B

	full_text

float* %100
Nload8BD
B
	full_text5
3
1%728 = load float, float* %79, align 4, !tbaa !14
+float*8B

	full_text


float* %79
9fadd8B/
-
	full_text 

%729 = fadd float %727, %728
*float8B

	full_text


float %727
*float8B

	full_text


float %728
dcall8BZ
X
	full_textK
I
G%730 = call float @llvm.fmuladd.f32(float %711, float %729, float %692)
*float8B

	full_text


float %711
*float8B

	full_text


float %729
*float8B

	full_text


float %692
Afmul8B7
5
	full_text(
&
$%731 = fmul float %703, 5.000000e-01
*float8B

	full_text


float %703
Nload8BD
B
	full_text5
3
1%732 = load float, float* %101, align 4, !tbaa !8
,float*8B

	full_text

float* %101
8fadd8B.
,
	full_text

%733 = fadd float %38, %732
)float8B

	full_text

	float %38
*float8B

	full_text


float %732
dcall8BZ
X
	full_textK
I
G%734 = call float @llvm.fmuladd.f32(float %731, float %733, float %714)
*float8B

	full_text


float %731
*float8B

	full_text


float %733
*float8B

	full_text


float %714
Oload8BE
C
	full_text6
4
2%735 = load float, float* %102, align 4, !tbaa !16
,float*8B

	full_text

float* %102
Nload8BD
B
	full_text5
3
1%736 = load float, float* %81, align 4, !tbaa !16
+float*8B

	full_text


float* %81
9fadd8B/
-
	full_text 

%737 = fadd float %735, %736
*float8B

	full_text


float %735
*float8B

	full_text


float %736
dcall8BZ
X
	full_textK
I
G%738 = call float @llvm.fmuladd.f32(float %731, float %737, float %718)
*float8B

	full_text


float %731
*float8B

	full_text


float %737
*float8B

	full_text


float %718
Oload8BE
C
	full_text6
4
2%739 = load float, float* %103, align 4, !tbaa !16
,float*8B

	full_text

float* %103
Nload8BD
B
	full_text5
3
1%740 = load float, float* %83, align 4, !tbaa !16
+float*8B

	full_text


float* %83
9fadd8B/
-
	full_text 

%741 = fadd float %739, %740
*float8B

	full_text


float %739
*float8B

	full_text


float %740
dcall8BZ
X
	full_textK
I
G%742 = call float @llvm.fmuladd.f32(float %731, float %741, float %722)
*float8B

	full_text


float %731
*float8B

	full_text


float %741
*float8B

	full_text


float %722
Oload8BE
C
	full_text6
4
2%743 = load float, float* %104, align 4, !tbaa !16
,float*8B

	full_text

float* %104
Nload8BD
B
	full_text5
3
1%744 = load float, float* %85, align 4, !tbaa !16
+float*8B

	full_text


float* %85
9fadd8B/
-
	full_text 

%745 = fadd float %743, %744
*float8B

	full_text


float %743
*float8B

	full_text


float %744
dcall8BZ
X
	full_textK
I
G%746 = call float @llvm.fmuladd.f32(float %731, float %745, float %726)
*float8B

	full_text


float %731
*float8B

	full_text


float %745
*float8B

	full_text


float %726
Oload8BE
C
	full_text6
4
2%747 = load float, float* %105, align 4, !tbaa !16
,float*8B

	full_text

float* %105
Nload8BD
B
	full_text5
3
1%748 = load float, float* %87, align 4, !tbaa !16
+float*8B

	full_text


float* %87
9fadd8B/
-
	full_text 

%749 = fadd float %747, %748
*float8B

	full_text


float %747
*float8B

	full_text


float %748
dcall8BZ
X
	full_textK
I
G%750 = call float @llvm.fmuladd.f32(float %731, float %749, float %730)
*float8B

	full_text


float %731
*float8B

	full_text


float %749
*float8B

	full_text


float %730
Afmul8B7
5
	full_text(
&
$%751 = fmul float %707, 5.000000e-01
*float8B

	full_text


float %707
Nload8BD
B
	full_text5
3
1%752 = load float, float* %106, align 4, !tbaa !8
,float*8B

	full_text

float* %106
8fadd8B.
,
	full_text

%753 = fadd float %44, %752
)float8B

	full_text

	float %44
*float8B

	full_text


float %752
dcall8BZ
X
	full_textK
I
G%754 = call float @llvm.fmuladd.f32(float %751, float %753, float %734)
*float8B

	full_text


float %751
*float8B

	full_text


float %753
*float8B

	full_text


float %734
Oload8BE
C
	full_text6
4
2%755 = load float, float* %107, align 4, !tbaa !17
,float*8B

	full_text

float* %107
Nload8BD
B
	full_text5
3
1%756 = load float, float* %89, align 4, !tbaa !17
+float*8B

	full_text


float* %89
9fadd8B/
-
	full_text 

%757 = fadd float %755, %756
*float8B

	full_text


float %755
*float8B

	full_text


float %756
dcall8BZ
X
	full_textK
I
G%758 = call float @llvm.fmuladd.f32(float %751, float %757, float %738)
*float8B

	full_text


float %751
*float8B

	full_text


float %757
*float8B

	full_text


float %738
Oload8BE
C
	full_text6
4
2%759 = load float, float* %108, align 4, !tbaa !17
,float*8B

	full_text

float* %108
Nload8BD
B
	full_text5
3
1%760 = load float, float* %91, align 4, !tbaa !17
+float*8B

	full_text


float* %91
9fadd8B/
-
	full_text 

%761 = fadd float %759, %760
*float8B

	full_text


float %759
*float8B

	full_text


float %760
dcall8BZ
X
	full_textK
I
G%762 = call float @llvm.fmuladd.f32(float %751, float %761, float %742)
*float8B

	full_text


float %751
*float8B

	full_text


float %761
*float8B

	full_text


float %742
Oload8BE
C
	full_text6
4
2%763 = load float, float* %109, align 4, !tbaa !17
,float*8B

	full_text

float* %109
Nload8BD
B
	full_text5
3
1%764 = load float, float* %93, align 4, !tbaa !17
+float*8B

	full_text


float* %93
9fadd8B/
-
	full_text 

%765 = fadd float %763, %764
*float8B

	full_text


float %763
*float8B

	full_text


float %764
dcall8BZ
X
	full_textK
I
G%766 = call float @llvm.fmuladd.f32(float %751, float %765, float %746)
*float8B

	full_text


float %751
*float8B

	full_text


float %765
*float8B

	full_text


float %746
Oload8BE
C
	full_text6
4
2%767 = load float, float* %110, align 4, !tbaa !17
,float*8B

	full_text

float* %110
Nload8BD
B
	full_text5
3
1%768 = load float, float* %95, align 4, !tbaa !17
+float*8B

	full_text


float* %95
9fadd8B/
-
	full_text 

%769 = fadd float %767, %768
*float8B

	full_text


float %767
*float8B

	full_text


float %768
dcall8BZ
X
	full_textK
I
G%770 = call float @llvm.fmuladd.f32(float %751, float %769, float %750)
*float8B

	full_text


float %751
*float8B

	full_text


float %769
*float8B

	full_text


float %750
(br8B 

	full_text

br label %882
ccall8BY
W
	full_textJ
H
F%772 = call float @llvm.fmuladd.f32(float %699, float %57, float %690)
*float8B

	full_text


float %699
)float8B

	full_text

	float %57
*float8B

	full_text


float %690
ccall8BY
W
	full_textJ
H
F%773 = call float @llvm.fmuladd.f32(float %703, float %57, float %691)
*float8B

	full_text


float %703
)float8B

	full_text

	float %57
*float8B

	full_text


float %691
ccall8BY
W
	full_textJ
H
F%774 = call float @llvm.fmuladd.f32(float %707, float %57, float %692)
*float8B

	full_text


float %707
)float8B

	full_text

	float %57
*float8B

	full_text


float %692
(br8B 

	full_text

br label %882
9fmul8B/
-
	full_text 

%776 = fmul float %703, %703
*float8B

	full_text


float %703
*float8B

	full_text


float %703
dcall8BZ
X
	full_textK
I
G%777 = call float @llvm.fmuladd.f32(float %699, float %699, float %776)
*float8B

	full_text


float %699
*float8B

	full_text


float %699
*float8B

	full_text


float %776
dcall8BZ
X
	full_textK
I
G%778 = call float @llvm.fmuladd.f32(float %707, float %707, float %777)
*float8B

	full_text


float %707
*float8B

	full_text


float %707
*float8B

	full_text


float %777
Gcall8B=
;
	full_text.
,
*%779 = call float @_Z4sqrtf(float %778) #5
*float8B

	full_text


float %778
8sext8B.
,
	full_text

%780 = sext i32 %697 to i64
&i328B

	full_text


i32 %697
^getelementptr8BK
I
	full_text<
:
8%781 = getelementptr inbounds float, float* %2, i64 %780
&i648B

	full_text


i64 %780
Nload8BD
B
	full_text5
3
1%782 = load float, float* %781, align 4, !tbaa !8
,float*8B

	full_text

float* %781
7add8B.
,
	full_text

%783 = add nsw i32 %697, %9
&i328B

	full_text


i32 %697
8sext8B.
,
	full_text

%784 = sext i32 %783 to i64
&i328B

	full_text


i32 %783
^getelementptr8BK
I
	full_text<
:
8%785 = getelementptr inbounds float, float* %2, i64 %784
&i648B

	full_text


i64 %784
Nload8BD
B
	full_text5
3
1%786 = load float, float* %785, align 4, !tbaa !8
,float*8B

	full_text

float* %785
_insertelement8BL
J
	full_text=
;
9%787 = insertelement <2 x float> undef, float %786, i32 0
*float8B

	full_text


float %786
8add8B/
-
	full_text 

%788 = add nsw i32 %697, %34
&i328B

	full_text


i32 %697
%i328B

	full_text
	
i32 %34
8sext8B.
,
	full_text

%789 = sext i32 %788 to i64
&i328B

	full_text


i32 %788
^getelementptr8BK
I
	full_text<
:
8%790 = getelementptr inbounds float, float* %2, i64 %789
&i648B

	full_text


i64 %789
Nload8BD
B
	full_text5
3
1%791 = load float, float* %790, align 4, !tbaa !8
,float*8B

	full_text

float* %790
^insertelement8BK
I
	full_text<
:
8%792 = insertelement <2 x float> %787, float %791, i32 1
6<2 x float>8B#
!
	full_text

<2 x float> %787
*float8B

	full_text


float %791
8add8B/
-
	full_text 

%793 = add nsw i32 %697, %40
&i328B

	full_text


i32 %697
%i328B

	full_text
	
i32 %40
8sext8B.
,
	full_text

%794 = sext i32 %793 to i64
&i328B

	full_text


i32 %793
^getelementptr8BK
I
	full_text<
:
8%795 = getelementptr inbounds float, float* %2, i64 %794
&i648B

	full_text


i64 %794
Nload8BD
B
	full_text5
3
1%796 = load float, float* %795, align 4, !tbaa !8
,float*8B

	full_text

float* %795
8add8B/
-
	full_text 

%797 = add nsw i32 %697, %45
&i328B

	full_text


i32 %697
%i328B

	full_text
	
i32 %45
8sext8B.
,
	full_text

%798 = sext i32 %797 to i64
&i328B

	full_text


i32 %797
^getelementptr8BK
I
	full_text<
:
8%799 = getelementptr inbounds float, float* %2, i64 %798
&i648B

	full_text


i64 %798
Nload8BD
B
	full_text5
3
1%800 = load float, float* %799, align 4, !tbaa !8
,float*8B

	full_text

float* %799
Çcall8Bx
v
	full_texti
g
ecall void @compute_velocity(float %782, <2 x float> %792, float %796, %struct.FLOAT3* nonnull %16) #6
*float8B

	full_text


float %782
6<2 x float>8B#
!
	full_text

<2 x float> %792
*float8B

	full_text


float %796
-struct*8B

	full_text

struct* %16
Oload8BE
C
	full_text6
4
2%801 = load <2 x float>, <2 x float>* %70, align 8
7<2 x float>*8B#
!
	full_text

<2 x float>* %70
Cload8B9
7
	full_text*
(
&%802 = load float, float* %71, align 8
+float*8B

	full_text


float* %71
bcall8BX
V
	full_textI
G
E%803 = call float @compute_speed_sqd(<2 x float> %801, float %802) #6
6<2 x float>8B#
!
	full_text

<2 x float> %801
*float8B

	full_text


float %802
gcall8B]
[
	full_textN
L
J%804 = call float @compute_pressure(float %782, float %800, float %803) #6
*float8B

	full_text


float %782
*float8B

	full_text


float %800
*float8B

	full_text


float %803
acall8BW
U
	full_textH
F
D%805 = call float @compute_speed_of_sound(float %782, float %804) #6
*float8B

	full_text


float %782
*float8B

	full_text


float %804
Oload8BE
C
	full_text6
4
2%806 = load <2 x float>, <2 x float>* %70, align 8
7<2 x float>*8B#
!
	full_text

<2 x float>* %70
Cload8B9
7
	full_text*
(
&%807 = load float, float* %71, align 8
+float*8B

	full_text


float* %71
ùcall8Bí
è
	full_textÅ
˛
˚call void @compute_flux_contribution(float %782, <2 x float> %792, float %796, float %800, float %804, <2 x float> %806, float %807, %struct.FLOAT3* nonnull %17, %struct.FLOAT3* nonnull %18, %struct.FLOAT3* nonnull %19, %struct.FLOAT3* nonnull %20) #6
*float8B

	full_text


float %782
6<2 x float>8B#
!
	full_text

<2 x float> %792
*float8B

	full_text


float %796
*float8B

	full_text


float %800
*float8B

	full_text


float %804
6<2 x float>8B#
!
	full_text

<2 x float> %806
*float8B

	full_text


float %807
-struct*8B

	full_text

struct* %17
-struct*8B

	full_text

struct* %18
-struct*8B

	full_text

struct* %19
-struct*8B

	full_text

struct* %20
Gfmul8B=
;
	full_text.
,
*%808 = fmul float %779, 0xBFC99999A0000000
*float8B

	full_text


float %779
Afmul8B7
5
	full_text(
&
$%809 = fmul float %808, 5.000000e-01
*float8B

	full_text


float %808
Gcall8B=
;
	full_text.
,
*%810 = call float @_Z4sqrtf(float %803) #5
*float8B

	full_text


float %803
8fadd8B.
,
	full_text

%811 = fadd float %56, %810
)float8B

	full_text

	float %56
*float8B

	full_text


float %810
8fadd8B.
,
	full_text

%812 = fadd float %58, %811
)float8B

	full_text

	float %58
*float8B

	full_text


float %811
9fadd8B/
-
	full_text 

%813 = fadd float %805, %812
*float8B

	full_text


float %805
*float8B

	full_text


float %812
9fmul8B/
-
	full_text 

%814 = fmul float %809, %813
*float8B

	full_text


float %809
*float8B

	full_text


float %813
8fsub8B.
,
	full_text

%815 = fsub float %28, %782
)float8B

	full_text

	float %28
*float8B

	full_text


float %782
dcall8BZ
X
	full_textK
I
G%816 = call float @llvm.fmuladd.f32(float %814, float %815, float %693)
*float8B

	full_text


float %814
*float8B

	full_text


float %815
*float8B

	full_text


float %693
8fsub8B.
,
	full_text

%817 = fsub float %49, %800
)float8B

	full_text

	float %49
*float8B

	full_text


float %800
dcall8BZ
X
	full_textK
I
G%818 = call float @llvm.fmuladd.f32(float %814, float %817, float %689)
*float8B

	full_text


float %814
*float8B

	full_text


float %817
*float8B

	full_text


float %689
8fsub8B.
,
	full_text

%819 = fsub float %32, %786
)float8B

	full_text

	float %32
*float8B

	full_text


float %786
dcall8BZ
X
	full_textK
I
G%820 = call float @llvm.fmuladd.f32(float %814, float %819, float %690)
*float8B

	full_text


float %814
*float8B

	full_text


float %819
*float8B

	full_text


float %690
8fsub8B.
,
	full_text

%821 = fsub float %38, %791
)float8B

	full_text

	float %38
*float8B

	full_text


float %791
dcall8BZ
X
	full_textK
I
G%822 = call float @llvm.fmuladd.f32(float %814, float %821, float %691)
*float8B

	full_text


float %814
*float8B

	full_text


float %821
*float8B

	full_text


float %691
8fsub8B.
,
	full_text

%823 = fsub float %44, %796
)float8B

	full_text

	float %44
*float8B

	full_text


float %796
dcall8BZ
X
	full_textK
I
G%824 = call float @llvm.fmuladd.f32(float %814, float %823, float %692)
*float8B

	full_text


float %814
*float8B

	full_text


float %823
*float8B

	full_text


float %692
Afmul8B7
5
	full_text(
&
$%825 = fmul float %699, 5.000000e-01
*float8B

	full_text


float %699
8fadd8B.
,
	full_text

%826 = fadd float %32, %786
)float8B

	full_text

	float %32
*float8B

	full_text


float %786
dcall8BZ
X
	full_textK
I
G%827 = call float @llvm.fmuladd.f32(float %825, float %826, float %816)
*float8B

	full_text


float %825
*float8B

	full_text


float %826
*float8B

	full_text


float %816
Nload8BD
B
	full_text5
3
1%828 = load float, float* %72, align 4, !tbaa !14
+float*8B

	full_text


float* %72
Nload8BD
B
	full_text5
3
1%829 = load float, float* %73, align 4, !tbaa !14
+float*8B

	full_text


float* %73
9fadd8B/
-
	full_text 

%830 = fadd float %828, %829
*float8B

	full_text


float %828
*float8B

	full_text


float %829
dcall8BZ
X
	full_textK
I
G%831 = call float @llvm.fmuladd.f32(float %825, float %830, float %818)
*float8B

	full_text


float %825
*float8B

	full_text


float %830
*float8B

	full_text


float %818
Nload8BD
B
	full_text5
3
1%832 = load float, float* %74, align 4, !tbaa !14
+float*8B

	full_text


float* %74
Nload8BD
B
	full_text5
3
1%833 = load float, float* %75, align 4, !tbaa !14
+float*8B

	full_text


float* %75
9fadd8B/
-
	full_text 

%834 = fadd float %832, %833
*float8B

	full_text


float %832
*float8B

	full_text


float %833
dcall8BZ
X
	full_textK
I
G%835 = call float @llvm.fmuladd.f32(float %825, float %834, float %820)
*float8B

	full_text


float %825
*float8B

	full_text


float %834
*float8B

	full_text


float %820
Nload8BD
B
	full_text5
3
1%836 = load float, float* %76, align 4, !tbaa !14
+float*8B

	full_text


float* %76
Nload8BD
B
	full_text5
3
1%837 = load float, float* %77, align 4, !tbaa !14
+float*8B

	full_text


float* %77
9fadd8B/
-
	full_text 

%838 = fadd float %836, %837
*float8B

	full_text


float %836
*float8B

	full_text


float %837
dcall8BZ
X
	full_textK
I
G%839 = call float @llvm.fmuladd.f32(float %825, float %838, float %822)
*float8B

	full_text


float %825
*float8B

	full_text


float %838
*float8B

	full_text


float %822
Nload8BD
B
	full_text5
3
1%840 = load float, float* %78, align 4, !tbaa !14
+float*8B

	full_text


float* %78
Nload8BD
B
	full_text5
3
1%841 = load float, float* %79, align 4, !tbaa !14
+float*8B

	full_text


float* %79
9fadd8B/
-
	full_text 

%842 = fadd float %840, %841
*float8B

	full_text


float %840
*float8B

	full_text


float %841
dcall8BZ
X
	full_textK
I
G%843 = call float @llvm.fmuladd.f32(float %825, float %842, float %824)
*float8B

	full_text


float %825
*float8B

	full_text


float %842
*float8B

	full_text


float %824
Afmul8B7
5
	full_text(
&
$%844 = fmul float %703, 5.000000e-01
*float8B

	full_text


float %703
8fadd8B.
,
	full_text

%845 = fadd float %38, %791
)float8B

	full_text

	float %38
*float8B

	full_text


float %791
dcall8BZ
X
	full_textK
I
G%846 = call float @llvm.fmuladd.f32(float %844, float %845, float %827)
*float8B

	full_text


float %844
*float8B

	full_text


float %845
*float8B

	full_text


float %827
Nload8BD
B
	full_text5
3
1%847 = load float, float* %80, align 4, !tbaa !16
+float*8B

	full_text


float* %80
Nload8BD
B
	full_text5
3
1%848 = load float, float* %81, align 4, !tbaa !16
+float*8B

	full_text


float* %81
9fadd8B/
-
	full_text 

%849 = fadd float %847, %848
*float8B

	full_text


float %847
*float8B

	full_text


float %848
dcall8BZ
X
	full_textK
I
G%850 = call float @llvm.fmuladd.f32(float %844, float %849, float %831)
*float8B

	full_text


float %844
*float8B

	full_text


float %849
*float8B

	full_text


float %831
Nload8BD
B
	full_text5
3
1%851 = load float, float* %82, align 4, !tbaa !16
+float*8B

	full_text


float* %82
Nload8BD
B
	full_text5
3
1%852 = load float, float* %83, align 4, !tbaa !16
+float*8B

	full_text


float* %83
9fadd8B/
-
	full_text 

%853 = fadd float %851, %852
*float8B

	full_text


float %851
*float8B

	full_text


float %852
dcall8BZ
X
	full_textK
I
G%854 = call float @llvm.fmuladd.f32(float %844, float %853, float %835)
*float8B

	full_text


float %844
*float8B

	full_text


float %853
*float8B

	full_text


float %835
Nload8BD
B
	full_text5
3
1%855 = load float, float* %84, align 4, !tbaa !16
+float*8B

	full_text


float* %84
Nload8BD
B
	full_text5
3
1%856 = load float, float* %85, align 4, !tbaa !16
+float*8B

	full_text


float* %85
9fadd8B/
-
	full_text 

%857 = fadd float %855, %856
*float8B

	full_text


float %855
*float8B

	full_text


float %856
dcall8BZ
X
	full_textK
I
G%858 = call float @llvm.fmuladd.f32(float %844, float %857, float %839)
*float8B

	full_text


float %844
*float8B

	full_text


float %857
*float8B

	full_text


float %839
Nload8BD
B
	full_text5
3
1%859 = load float, float* %86, align 4, !tbaa !16
+float*8B

	full_text


float* %86
Nload8BD
B
	full_text5
3
1%860 = load float, float* %87, align 4, !tbaa !16
+float*8B

	full_text


float* %87
9fadd8B/
-
	full_text 

%861 = fadd float %859, %860
*float8B

	full_text


float %859
*float8B

	full_text


float %860
dcall8BZ
X
	full_textK
I
G%862 = call float @llvm.fmuladd.f32(float %844, float %861, float %843)
*float8B

	full_text


float %844
*float8B

	full_text


float %861
*float8B

	full_text


float %843
Afmul8B7
5
	full_text(
&
$%863 = fmul float %707, 5.000000e-01
*float8B

	full_text


float %707
8fadd8B.
,
	full_text

%864 = fadd float %44, %796
)float8B

	full_text

	float %44
*float8B

	full_text


float %796
dcall8BZ
X
	full_textK
I
G%865 = call float @llvm.fmuladd.f32(float %863, float %864, float %846)
*float8B

	full_text


float %863
*float8B

	full_text


float %864
*float8B

	full_text


float %846
Nload8BD
B
	full_text5
3
1%866 = load float, float* %88, align 4, !tbaa !17
+float*8B

	full_text


float* %88
Nload8BD
B
	full_text5
3
1%867 = load float, float* %89, align 4, !tbaa !17
+float*8B

	full_text


float* %89
9fadd8B/
-
	full_text 

%868 = fadd float %866, %867
*float8B

	full_text


float %866
*float8B

	full_text


float %867
dcall8BZ
X
	full_textK
I
G%869 = call float @llvm.fmuladd.f32(float %863, float %868, float %850)
*float8B

	full_text


float %863
*float8B

	full_text


float %868
*float8B

	full_text


float %850
Nload8BD
B
	full_text5
3
1%870 = load float, float* %90, align 4, !tbaa !17
+float*8B

	full_text


float* %90
Nload8BD
B
	full_text5
3
1%871 = load float, float* %91, align 4, !tbaa !17
+float*8B

	full_text


float* %91
9fadd8B/
-
	full_text 

%872 = fadd float %870, %871
*float8B

	full_text


float %870
*float8B

	full_text


float %871
dcall8BZ
X
	full_textK
I
G%873 = call float @llvm.fmuladd.f32(float %863, float %872, float %854)
*float8B

	full_text


float %863
*float8B

	full_text


float %872
*float8B

	full_text


float %854
Nload8BD
B
	full_text5
3
1%874 = load float, float* %92, align 4, !tbaa !17
+float*8B

	full_text


float* %92
Nload8BD
B
	full_text5
3
1%875 = load float, float* %93, align 4, !tbaa !17
+float*8B

	full_text


float* %93
9fadd8B/
-
	full_text 

%876 = fadd float %874, %875
*float8B

	full_text


float %874
*float8B

	full_text


float %875
dcall8BZ
X
	full_textK
I
G%877 = call float @llvm.fmuladd.f32(float %863, float %876, float %858)
*float8B

	full_text


float %863
*float8B

	full_text


float %876
*float8B

	full_text


float %858
Nload8BD
B
	full_text5
3
1%878 = load float, float* %94, align 4, !tbaa !17
+float*8B

	full_text


float* %94
Nload8BD
B
	full_text5
3
1%879 = load float, float* %95, align 4, !tbaa !17
+float*8B

	full_text


float* %95
9fadd8B/
-
	full_text 

%880 = fadd float %878, %879
*float8B

	full_text


float %878
*float8B

	full_text


float %879
dcall8BZ
X
	full_textK
I
G%881 = call float @llvm.fmuladd.f32(float %863, float %880, float %862)
*float8B

	full_text


float %863
*float8B

	full_text


float %880
*float8B

	full_text


float %862
(br8B 

	full_text

br label %882
kphi8Bb
`
	full_textS
Q
O%883 = phi float [ %869, %775 ], [ %689, %709 ], [ %689, %771 ], [ %758, %710 ]
*float8B

	full_text


float %869
*float8B

	full_text


float %689
*float8B

	full_text


float %689
*float8B

	full_text


float %758
kphi8Bb
`
	full_textS
Q
O%884 = phi float [ %873, %775 ], [ %690, %709 ], [ %772, %771 ], [ %762, %710 ]
*float8B

	full_text


float %873
*float8B

	full_text


float %690
*float8B

	full_text


float %772
*float8B

	full_text


float %762
kphi8Bb
`
	full_textS
Q
O%885 = phi float [ %877, %775 ], [ %691, %709 ], [ %773, %771 ], [ %766, %710 ]
*float8B

	full_text


float %877
*float8B

	full_text


float %691
*float8B

	full_text


float %773
*float8B

	full_text


float %766
kphi8Bb
`
	full_textS
Q
O%886 = phi float [ %881, %775 ], [ %692, %709 ], [ %774, %771 ], [ %770, %710 ]
*float8B

	full_text


float %881
*float8B

	full_text


float %692
*float8B

	full_text


float %774
*float8B

	full_text


float %770
kphi8Bb
`
	full_textS
Q
O%887 = phi float [ %865, %775 ], [ %693, %709 ], [ %693, %771 ], [ %754, %710 ]
*float8B

	full_text


float %865
*float8B

	full_text


float %693
*float8B

	full_text


float %693
*float8B

	full_text


float %754
]getelementptr8BJ
H
	full_text;
9
7%888 = getelementptr inbounds float, float* %4, i64 %26
%i648B

	full_text
	
i64 %26
Nstore8BC
A
	full_text4
2
0store float %887, float* %888, align 4, !tbaa !8
*float8B

	full_text


float %887
,float*8B

	full_text

float* %888
]getelementptr8BJ
H
	full_text;
9
7%889 = getelementptr inbounds float, float* %4, i64 %30
%i648B

	full_text
	
i64 %30
Nstore8BC
A
	full_text4
2
0store float %884, float* %889, align 4, !tbaa !8
*float8B

	full_text


float %884
,float*8B

	full_text

float* %889
]getelementptr8BJ
H
	full_text;
9
7%890 = getelementptr inbounds float, float* %4, i64 %36
%i648B

	full_text
	
i64 %36
Nstore8BC
A
	full_text4
2
0store float %885, float* %890, align 4, !tbaa !8
*float8B

	full_text


float %885
,float*8B

	full_text

float* %890
]getelementptr8BJ
H
	full_text;
9
7%891 = getelementptr inbounds float, float* %4, i64 %42
%i648B

	full_text
	
i64 %42
Nstore8BC
A
	full_text4
2
0store float %886, float* %891, align 4, !tbaa !8
*float8B

	full_text


float %886
,float*8B

	full_text

float* %891
]getelementptr8BJ
H
	full_text;
9
7%892 = getelementptr inbounds float, float* %4, i64 %47
%i648B

	full_text
	
i64 %47
Nstore8BC
A
	full_text4
2
0store float %883, float* %892, align 4, !tbaa !8
*float8B

	full_text


float %883
,float*8B

	full_text

float* %892
Zcall8BP
N
	full_textA
?
=call void @llvm.lifetime.end.p0i8(i64 12, i8* nonnull %69) #6
%i8*8B

	full_text
	
i8* %69
Zcall8BP
N
	full_textA
?
=call void @llvm.lifetime.end.p0i8(i64 12, i8* nonnull %68) #6
%i8*8B

	full_text
	
i8* %68
Zcall8BP
N
	full_textA
?
=call void @llvm.lifetime.end.p0i8(i64 12, i8* nonnull %67) #6
%i8*8B

	full_text
	
i8* %67
Zcall8BP
N
	full_textA
?
=call void @llvm.lifetime.end.p0i8(i64 12, i8* nonnull %66) #6
%i8*8B

	full_text
	
i8* %66
Zcall8BP
N
	full_textA
?
=call void @llvm.lifetime.end.p0i8(i64 12, i8* nonnull %65) #6
%i8*8B

	full_text
	
i8* %65
Zcall8BP
N
	full_textA
?
=call void @llvm.lifetime.end.p0i8(i64 12, i8* nonnull %62) #6
%i8*8B

	full_text
	
i8* %62
Zcall8BP
N
	full_textA
?
=call void @llvm.lifetime.end.p0i8(i64 12, i8* nonnull %61) #6
%i8*8B

	full_text
	
i8* %61
Zcall8BP
N
	full_textA
?
=call void @llvm.lifetime.end.p0i8(i64 12, i8* nonnull %60) #6
%i8*8B

	full_text
	
i8* %60
Zcall8BP
N
	full_textA
?
=call void @llvm.lifetime.end.p0i8(i64 12, i8* nonnull %59) #6
%i8*8B

	full_text
	
i8* %59
Zcall8BP
N
	full_textA
?
=call void @llvm.lifetime.end.p0i8(i64 12, i8* nonnull %50) #6
%i8*8B

	full_text
	
i8* %50
(br8B 

	full_text

br label %320
4struct*8B%
#
	full_text

%struct.FLOAT3* %8
4struct*8B%
#
	full_text

%struct.FLOAT3* %7
*float*8B

	full_text

	float* %3
4struct*8B%
#
	full_text

%struct.FLOAT3* %6
$i328B

	full_text


i32 %9
*float*8B

	full_text

	float* %2
4struct*8B%
#
	full_text

%struct.FLOAT3* %5
*float*8B

	full_text

	float* %4
&i32*8B

	full_text
	
i32* %0
*float*8B
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
-; undefined function B

	full_text

 
-; undefined function B

	full_text

 
-; undefined function B

	full_text

 
-; undefined function B

	full_text

 
-; undefined function 	B

	full_text

 
-; undefined function 
B

	full_text

 
$i328B

	full_text


i32 -1
#i648B

	full_text	

i64 0
7<2 x float>8B$
"
	full_text

<2 x float> undef
2float8B%
#
	full_text

float 5.000000e-01
2float8B%
#
	full_text

float 0.000000e+00
$i328B

	full_text


i32 -2
#i648B

	full_text	

i64 3
#i328B

	full_text	

i32 1
#i648B

	full_text	

i64 9
#i328B

	full_text	

i32 2
#i648B

	full_text	

i64 6
#i648B

	full_text	

i64 7
$i648B

	full_text


i64 10
$i648B

	full_text


i64 32
8float8B+
)
	full_text

float 0xBFC99999A0000000
#i328B

	full_text	

i32 3
$i648B

	full_text


i64 12
#i328B

	full_text	

i32 0
#i648B

	full_text	

i64 2
#i648B

	full_text	

i64 5
$i648B

	full_text


i64 11
#i648B

	full_text	

i64 1        		 

                      !    "# "" $$ %& %' %% () (( *+ ** ,- ,, ./ .0 .. 11 23 24 22 56 55 78 77 9: 99 ;; <= <> << ?@ ?? AB AA CD CC EF EE GH GG IJ IK IL IM II NO NN PQ PP RS RR TU TT VW VX VV YZ YY [\ [] [^ [[ _` _a __ bc bb de dd fg ff hi hh jk jj lm ll no nn pq pp rs rr tu tt vw vx vy vz v{ v| v} v~ v v		Ä v	
Å vv ÇÉ ÇÇ Ñ
Ö ÑÑ Üá ÜÜ à
â àà äã ää å
ç åå éè éé ê
ë êê íì íí î
ï îî ñó ññ òô òò öõ öö úù úú ûü ûû †° †† ¢£ ¢¢ §• §§ ¶ß ¶¶ ®© ®® ™´ ™™ ¨≠ ¨¨ ÆØ ÆÆ ∞± ∞∞ ≤≥ ≤≤ ¥µ ¥¥ ∂∑ ∂∂ ∏π ∏∏ ∫ª ∫∫ ºΩ ºº æø ææ ¿¡ ¿¿ ¬√ ¬¬ ƒ≈ ƒƒ ∆« ∆∆ »… »»    ÀÀ ÃÃ ÕÕ ŒŒ œœ –– —— ““ ”” ‘‘ ’’ ÷÷ ◊◊ ÿÿ ŸŸ ⁄€ ⁄⁄ ‹› ‹‹ ﬁ
ﬂ ﬁﬁ ‡· ‡‡ ‚
„ ‚‚ ‰Â ‰‰ ÊÁ ÊÊ ËÈ Ë
Í ËË Î
Ï ÎÎ ÌÓ ÌÌ Ô ÔÔ ÒÚ Ò
Û ÒÒ Ù
ı ÙÙ ˆ˜ ˆˆ ¯˘ ¯¯ ˙˚ ˙˝ ¸
˛ ¸¸ ˇÄ ˇ
Å ˇ
Ç ˇˇ ÉÑ É
Ö É
Ü ÉÉ áà áá âä ââ ã
å ãã çé çç èê èè ëí ëë ì
î ìì ïñ ïï ó
ò óó ôö ô
õ ôô úù úú û
ü ûû †° †† ¢£ ¢
§ ¢¢ •¶ •
ß •• ®© ®® ™
´ ™™ ¨≠ ¨¨ ÆØ Æ
∞ ÆÆ ±≤ ±± ≥
¥ ≥≥ µ∂ µµ ∑∏ ∑
π ∑
∫ ∑
ª ∑∑ ºΩ ºº æø ææ ¿¡ ¿
¬ ¿¿ √ƒ √
≈ √
∆ √√ «» «
… ««  À    ÃÕ ÃÃ Œœ Œ
– Œ
— Œ
“ Œ
” Œ
‘ Œ
’ Œ
÷ Œ
◊ Œ
	ÿ Œ

Ÿ ŒŒ ⁄€ ⁄⁄ ‹› ‹‹ ﬁﬂ ﬁﬁ ‡· ‡
‚ ‡‡ „‰ „
Â „„ ÊÁ Ê
Ë ÊÊ ÈÍ È
Î ÈÈ ÏÌ Ï
Ó ÏÏ Ô Ô
Ò ÔÔ ÚÛ Ú
Ù ÚÚ ıˆ ı
˜ ıı ¯˘ ¯
˙ ¯¯ ˚¸ ˚
˝ ˚˚ ˛ˇ ˛
Ä ˛˛ ÅÇ Å
É ÅÅ ÑÖ Ñ
Ü ÑÑ áà á
â áá äã ää åç å
é åå èê è
ë è
í èè ìî ìì ïñ ïï óò ó
ô óó öõ ö
ú ö
ù öö ûü ûû †° †† ¢£ ¢
§ ¢¢ •¶ •
ß •
® •• ©™ ©© ´¨ ´´ ≠Æ ≠
Ø ≠≠ ∞± ∞
≤ ∞
≥ ∞∞ ¥µ ¥¥ ∂∑ ∂∂ ∏π ∏
∫ ∏∏ ªº ª
Ω ª
æ ªª ø¿ øø ¡¬ ¡
√ ¡¡ ƒ≈ ƒ
∆ ƒ
« ƒƒ »… »»  À    ÃÕ Ã
Œ ÃÃ œ– œ
— œ
“ œœ ”‘ ”” ’÷ ’’ ◊ÿ ◊
Ÿ ◊◊ ⁄€ ⁄
‹ ⁄
› ⁄⁄ ﬁﬂ ﬁﬁ ‡· ‡‡ ‚„ ‚
‰ ‚‚ ÂÊ Â
Á Â
Ë ÂÂ ÈÍ ÈÈ ÎÏ ÎÎ ÌÓ Ì
Ô ÌÌ Ò 
Ú 
Û  Ùı ÙÙ ˆ˜ ˆ
¯ ˆˆ ˘˙ ˘
˚ ˘
¸ ˘˘ ˝˛ ˝˝ ˇÄ ˇˇ ÅÇ Å
É ÅÅ ÑÖ Ñ
Ü Ñ
á ÑÑ àâ àà äã ää åç å
é åå èê è
ë è
í èè ìî ìì ïñ ïï óò ó
ô óó öõ ö
ú ö
ù öö ûü ûû †° †† ¢£ ¢
§ ¢¢ •¶ •
ß •
® •• ©´ ™≠ ¨
Æ ¨¨ Ø∞ Ø
± ØØ ≤≥ ≤
¥ ≤≤ µ∑ ∂∂ ∏π ∏∏ ∫ª ∫
º ∫∫ Ωæ Ω
ø ΩΩ ¿¡ ¿¿ ¬√ ¬¬ ƒ≈ ƒ
∆ ƒƒ «» «
… ««  À    ÃÕ ÃÃ Œœ Œ
– ŒŒ —“ —
” —— ‘’ ‘‘ ÷◊ ÷÷ ÿŸ ÿ
⁄ ÿÿ €‹ €
› €€ ﬁﬂ ﬁﬁ ‡· ‡‡ ‚„ ‚
‰ ‚‚ ÂÊ Â
Á ÂÂ ËÈ ËË ÍÎ ÍÍ ÏÌ Ï
Ó ÏÏ Ô Ô
Ò Ô
Ú ÔÔ ÛÙ ÛÛ ıˆ ıı ˜¯ ˜
˘ ˜˜ ˙˚ ˙
¸ ˙
˝ ˙˙ ˛ˇ ˛˛ ÄÅ ÄÄ ÇÉ Ç
Ñ ÇÇ ÖÜ Ö
á Ö
à ÖÖ âä ââ ãå ãã çé ç
è çç êë ê
í ê
ì êê îï îî ñó ññ òô ò
ö òò õú õ
ù õ
û õõ ü† üü °¢ °° £§ £
• ££ ¶ß ¶
® ¶
© ¶¶ ™´ ™™ ¨≠ ¨¨ ÆØ Æ
∞ ÆÆ ±≤ ±
≥ ±
¥ ±± µ∂ µµ ∑∏ ∑∑ π∫ π
ª ππ ºΩ º
æ º
ø ºº ¿¡ ¿¿ ¬√ ¬¬ ƒ≈ ƒ
∆ ƒƒ «» «
… «
  «« ÀÃ ÀÀ ÕŒ ÕÕ œ– œ
— œœ “” “
‘ “
’ ““ ÷ÿ ◊
Ÿ ◊◊ ⁄€ ⁄
‹ ⁄
› ⁄⁄ ﬁﬂ ﬁ
‡ ﬁ
· ﬁﬁ ‚„ ‚
‰ ‚
Â ‚‚ ÊÁ Ê
Ë ÊÊ ÈÍ È
Î ÈÈ Ï
Ì ÏÏ ÓÔ ÓÓ 
Ò  ÚÛ ÚÚ Ùı ÙÙ ˆ˜ ˆ
¯ ˆˆ ˘
˙ ˘˘ ˚¸ ˚˚ ˝˛ ˝˝ ˇÄ ˇ
Å ˇˇ Ç
É ÇÇ ÑÖ ÑÑ Üá ÜÜ àâ àå ãé çç èê èè ëí ë
ì ëë îï î
ñ î
ó îî òô òò öõ öö úù ú
û úú ü† ü
° ü
¢ üü £§ ££ •¶ •• ß® ß
© ßß ™´ ™
¨ ™
≠ ™™ ÆØ ÆÆ ∞± ∞∞ ≤≥ ≤
¥ ≤≤ µ∂ µ
∑ µ
∏ µµ π∫ ππ ªº ªª Ωæ Ω
ø ΩΩ ¿¡ ¿
¬ ¿
√ ¿¿ ƒ≈ ƒƒ ∆« ∆∆ »… »
  »» ÀÃ À
Õ À
Œ ÀÀ œ– œœ —“ —— ”‘ ”
’ ”” ÷◊ ÷
ÿ ÷
Ÿ ÷÷ ⁄€ ⁄⁄ ‹› ‹‹ ﬁﬂ ﬁ
‡ ﬁﬁ ·‚ ·
„ ·
‰ ·· ÂÊ ÂÂ ÁË ÁÁ ÈÍ È
Î ÈÈ ÏÌ Ï
Ó Ï
Ô ÏÏ Ò  ÚÛ ÚÚ Ùı Ù
ˆ ÙÙ ˜¯ ˜
˘ ˜
˙ ˜˜ ˚¸ ˚˚ ˝˛ ˝˝ ˇÄ ˇ
Å ˇˇ ÇÉ Ç
Ñ Ç
Ö ÇÇ Üá ÜÜ àâ àà äã ä
å ää çé ç
è ç
ê çç ëí ëë ìî ìì ïñ ï
ó ïï òô ò
ö ò
õ òò úù úú ûü ûû †° †
¢ †† £§ £
• £
¶ ££ ß® ßß ©™ ©© ´¨ ´
≠ ´´ ÆØ Æ
∞ Æ
± ÆÆ ≤¥ ≥
µ ≥
∂ ≥≥ ∑∏ ∑
π ∑
∫ ∑∑ ªº ª
Ω ª
æ ªª ø¡ ¿
¬ ¿¿ √ƒ √
≈ √
∆ √√ «» «
… «
  «« ÀÃ ÀÀ ÕŒ ÕÕ œ
– œœ —“ —— ”‘ ”” ’÷ ’’ ◊
ÿ ◊◊ Ÿ⁄ ŸŸ €
‹ €€ ›ﬁ ›
ﬂ ›› ‡· ‡‡ ‚
„ ‚‚ ‰Â ‰‰ ÊÁ Ê
Ë ÊÊ ÈÍ È
Î ÈÈ ÏÌ ÏÏ Ó
Ô ÓÓ Ò  ÚÛ Ú
Ù ÚÚ ıˆ ıı ˜
¯ ˜˜ ˘˙ ˘˘ ˚¸ ˚
˝ ˚
˛ ˚
ˇ ˚˚ ÄÅ ÄÄ ÇÉ ÇÇ ÑÖ Ñ
Ü ÑÑ áà á
â á
ä áá ãå ã
ç ãã éè éé êë êê íì í
î í
ï í
ñ í
ó í
ò í
ô í
ö í
õ í
	ú í

ù íí ûü ûû †° †† ¢£ ¢¢ §• §
¶ §§ ß® ß
© ßß ™´ ™
¨ ™™ ≠Æ ≠
Ø ≠≠ ∞± ∞
≤ ∞∞ ≥¥ ≥
µ ≥
∂ ≥≥ ∑∏ ∑
π ∑∑ ∫ª ∫
º ∫
Ω ∫∫ æø æ
¿ ææ ¡¬ ¡
√ ¡
ƒ ¡¡ ≈∆ ≈
« ≈≈ »… »
  »
À »» ÃÕ Ã
Œ ÃÃ œ– œ
— œ
“ œœ ”‘ ”” ’÷ ’
◊ ’’ ÿŸ ÿ
⁄ ÿ
€ ÿÿ ‹› ‹‹ ﬁﬂ ﬁﬁ ‡· ‡
‚ ‡‡ „‰ „
Â „
Ê „„ ÁË ÁÁ ÈÍ ÈÈ ÎÏ Î
Ì ÎÎ ÓÔ Ó
 Ó
Ò ÓÓ ÚÛ ÚÚ Ùı ÙÙ ˆ˜ ˆ
¯ ˆˆ ˘˙ ˘
˚ ˘
¸ ˘˘ ˝˛ ˝˝ ˇÄ	 ˇˇ Å	Ç	 Å	
É	 Å	Å	 Ñ	Ö	 Ñ	
Ü	 Ñ	
á	 Ñ	Ñ	 à	â	 à	à	 ä	ã	 ä	
å	 ä	ä	 ç	é	 ç	
è	 ç	
ê	 ç	ç	 ë	í	 ë	ë	 ì	î	 ì	ì	 ï	ñ	 ï	
ó	 ï	ï	 ò	ô	 ò	
ö	 ò	
õ	 ò	ò	 ú	ù	 ú	ú	 û	ü	 û	û	 †	°	 †	
¢	 †	†	 £	§	 £	
•	 £	
¶	 £	£	 ß	®	 ß	ß	 ©	™	 ©	©	 ´	¨	 ´	
≠	 ´	´	 Æ	Ø	 Æ	
∞	 Æ	
±	 Æ	Æ	 ≤	≥	 ≤	≤	 ¥	µ	 ¥	¥	 ∂	∑	 ∂	
∏	 ∂	∂	 π	∫	 π	
ª	 π	
º	 π	π	 Ω	æ	 Ω	Ω	 ø	¿	 ø	
¡	 ø	ø	 ¬	√	 ¬	
ƒ	 ¬	
≈	 ¬	¬	 ∆	«	 ∆	∆	 »	…	 »	»	  	À	  	
Ã	  	 	 Õ	Œ	 Õ	
œ	 Õ	
–	 Õ	Õ	 —	“	 —	—	 ”	‘	 ”	”	 ’	÷	 ’	
◊	 ’	’	 ÿ	Ÿ	 ÿ	
⁄	 ÿ	
€	 ÿ	ÿ	 ‹	›	 ‹	‹	 ﬁ	ﬂ	 ﬁ	ﬁ	 ‡	·	 ‡	
‚	 ‡	‡	 „	‰	 „	
Â	 „	
Ê	 „	„	 Á	Ë	 Á	Á	 È	Í	 È	È	 Î	Ï	 Î	
Ì	 Î	Î	 Ó	Ô	 Ó	
	 Ó	
Ò	 Ó	Ó	 Ú	Ù	 Û	
ı	 Û	
ˆ	 Û	
˜	 Û	Û	 ¯	˘	 ¯	
˙	 ¯	
˚	 ¯	
¸	 ¯	¯	 ˝	˛	 ˝	
ˇ	 ˝	
Ä
 ˝	
Å
 ˝	˝	 Ç
É
 Ç

Ñ
 Ç

Ö
 Ç

Ü
 Ç
Ç
 á
à
 á

â
 á

ä
 á

ã
 á
á
 å
ç
 å
å
 é
è
 é

ê
 é
é
 ë

í
 ë
ë
 ì
î
 ì
ì
 ï

ñ
 ï
ï
 ó
ò
 ó
ó
 ô
ö
 ô
ô
 õ
ú
 õ

ù
 õ
õ
 û

ü
 û
û
 †
°
 †
†
 ¢
£
 ¢
¢
 §
•
 §

¶
 §
§
 ß

®
 ß
ß
 ©
™
 ©
©
 ´
¨
 ´
´
 ≠
Æ
 ≠
∞
 Ø
≤
 ±
±
 ≥
¥
 ≥
≥
 µ
∂
 µ

∑
 µ
µ
 ∏
π
 ∏

∫
 ∏

ª
 ∏
∏
 º
Ω
 º
º
 æ
ø
 æ
æ
 ¿
¡
 ¿

¬
 ¿
¿
 √
ƒ
 √

≈
 √

∆
 √
√
 «
»
 «
«
 …
 
 …
…
 À
Ã
 À

Õ
 À
À
 Œ
œ
 Œ

–
 Œ

—
 Œ
Œ
 “
”
 “
“
 ‘
’
 ‘
‘
 ÷
◊
 ÷

ÿ
 ÷
÷
 Ÿ
⁄
 Ÿ

€
 Ÿ

‹
 Ÿ
Ÿ
 ›
ﬁ
 ›
›
 ﬂ
‡
 ﬂ
ﬂ
 ·
‚
 ·

„
 ·
·
 ‰
Â
 ‰

Ê
 ‰

Á
 ‰
‰
 Ë
È
 Ë
Ë
 Í
Î
 Í
Í
 Ï
Ì
 Ï

Ó
 Ï
Ï
 Ô

 Ô

Ò
 Ô

Ú
 Ô
Ô
 Û
Ù
 Û
Û
 ı
ˆ
 ı
ı
 ˜
¯
 ˜

˘
 ˜
˜
 ˙
˚
 ˙

¸
 ˙

˝
 ˙
˙
 ˛
ˇ
 ˛
˛
 ÄÅ ÄÄ ÇÉ Ç
Ñ ÇÇ ÖÜ Ö
á Ö
à ÖÖ âä ââ ãå ãã çé ç
è çç êë ê
í ê
ì êê îï îî ñó ññ òô ò
ö òò õú õ
ù õ
û õõ ü† üü °¢ °° £§ £
• ££ ¶ß ¶
® ¶
© ¶¶ ™´ ™™ ¨≠ ¨¨ ÆØ Æ
∞ ÆÆ ±≤ ±
≥ ±
¥ ±± µ∂ µµ ∑∏ ∑∑ π∫ π
ª ππ ºΩ º
æ º
ø ºº ¿¡ ¿¿ ¬√ ¬¬ ƒ≈ ƒ
∆ ƒƒ «» «
… «
  «« ÀÃ ÀÀ ÕŒ ÕÕ œ– œ
— œœ “” “
‘ “
’ ““ ÷ÿ ◊
Ÿ ◊
⁄ ◊◊ €‹ €
› €
ﬁ €€ ﬂ‡ ﬂ
· ﬂ
‚ ﬂﬂ „Â ‰
Ê ‰‰ ÁË Á
È Á
Í ÁÁ ÎÏ Î
Ì Î
Ó ÎÎ Ô ÔÔ ÒÚ ÒÒ Û
Ù ÛÛ ıˆ ıı ˜¯ ˜˜ ˘˙ ˘˘ ˚
¸ ˚˚ ˝˛ ˝˝ ˇ
Ä ˇˇ ÅÇ Å
É ÅÅ ÑÖ ÑÑ Ü
á ÜÜ àâ àà äã ä
å ää çé ç
è çç êë êê í
ì íí îï îî ñó ñ
ò ññ ôö ôô õ
ú õõ ùû ùù ü† ü
° ü
¢ ü
£ üü §• §§ ¶ß ¶¶ ®© ®
™ ®® ´¨ ´
≠ ´
Æ ´´ Ø∞ Ø
± ØØ ≤≥ ≤≤ ¥µ ¥¥ ∂∑ ∂
∏ ∂
π ∂
∫ ∂
ª ∂
º ∂
Ω ∂
æ ∂
ø ∂
	¿ ∂

¡ ∂∂ ¬√ ¬¬ ƒ≈ ƒƒ ∆« ∆∆ »… »
  »» ÀÃ À
Õ ÀÀ Œœ Œ
– ŒŒ —“ —
” —— ‘’ ‘
÷ ‘‘ ◊ÿ ◊
Ÿ ◊
⁄ ◊◊ €‹ €
› €€ ﬁﬂ ﬁ
‡ ﬁ
· ﬁﬁ ‚„ ‚
‰ ‚‚ ÂÊ Â
Á Â
Ë ÂÂ ÈÍ È
Î ÈÈ ÏÌ Ï
Ó Ï
Ô ÏÏ Ò 
Ú  ÛÙ Û
ı Û
ˆ ÛÛ ˜¯ ˜˜ ˘˙ ˘
˚ ˘˘ ¸˝ ¸
˛ ¸
ˇ ¸¸ ÄÅ ÄÄ ÇÉ ÇÇ ÑÖ Ñ
Ü ÑÑ áà á
â á
ä áá ãå ãã çé çç èê è
ë èè íì í
î í
ï íí ñó ññ òô òò öõ ö
ú öö ùû ù
ü ù
† ùù °¢ °° £§ ££ •¶ •
ß •• ®© ®
™ ®
´ ®® ¨≠ ¨¨ ÆØ Æ
∞ ÆÆ ±≤ ±
≥ ±
¥ ±± µ∂ µµ ∑∏ ∑∑ π∫ π
ª ππ ºΩ º
æ º
ø ºº ¿¡ ¿¿ ¬√ ¬¬ ƒ≈ ƒ
∆ ƒƒ «» «
… «
  «« ÀÃ ÀÀ ÕŒ ÕÕ œ– œ
— œœ “” “
‘ “
’ ““ ÷◊ ÷÷ ÿŸ ÿÿ ⁄€ ⁄
‹ ⁄⁄ ›ﬁ ›
ﬂ ›
‡ ›› ·‚ ·· „‰ „
Â „„ ÊÁ Ê
Ë Ê
È ÊÊ ÍÎ ÍÍ ÏÌ ÏÏ ÓÔ Ó
 ÓÓ ÒÚ Ò
Û Ò
Ù ÒÒ ıˆ ıı ˜¯ ˜˜ ˘˙ ˘
˚ ˘˘ ¸˝ ¸
˛ ¸
ˇ ¸¸ ÄÅ ÄÄ ÇÉ ÇÇ ÑÖ Ñ
Ü ÑÑ áà á
â á
ä áá ãå ãã çé çç èê è
ë èè íì í
î í
ï íí ñò ó
ô ó
ö ó
õ óó úù ú
û ú
ü ú
† úú °¢ °
£ °
§ °
• °° ¶ß ¶
® ¶
© ¶
™ ¶¶ ´¨ ´
≠ ´
Æ ´
Ø ´´ ∞± ∞∞ ≤≥ ≤
¥ ≤≤ µ
∂ µµ ∑∏ ∑∑ π
∫ ππ ªº ªª Ωæ ΩΩ ø¿ ø
¡ øø ¬
√ ¬¬ ƒ≈ ƒƒ ∆« ∆∆ »… »
  »» À
Ã ÀÀ ÕŒ ÕÕ œ– œœ —“ —‘ ”÷ ’’ ◊ÿ ◊◊ Ÿ⁄ Ÿ
€ ŸŸ ‹› ‹
ﬁ ‹
ﬂ ‹‹ ‡· ‡‡ ‚„ ‚‚ ‰Â ‰
Ê ‰‰ ÁË Á
È Á
Í ÁÁ ÎÏ ÎÎ ÌÓ ÌÌ Ô Ô
Ò ÔÔ ÚÛ Ú
Ù Ú
ı ÚÚ ˆ˜ ˆˆ ¯˘ ¯¯ ˙˚ ˙
¸ ˙˙ ˝˛ ˝
ˇ ˝
Ä ˝˝ ÅÇ ÅÅ ÉÑ ÉÉ ÖÜ Ö
á ÖÖ àâ à
ä à
ã àà åç åå éè éé êë ê
í êê ìî ì
ï ì
ñ ìì óò óó ôö ôô õú õ
ù õõ ûü û
† û
° ûû ¢£ ¢¢ §• §§ ¶ß ¶
® ¶¶ ©™ ©
´ ©
¨ ©© ≠Æ ≠≠ Ø∞ ØØ ±≤ ±
≥ ±± ¥µ ¥
∂ ¥
∑ ¥¥ ∏π ∏∏ ∫ª ∫∫ ºΩ º
æ ºº ø¿ ø
¡ ø
¬ øø √ƒ √√ ≈∆ ≈≈ «» «
… ««  À  
Ã  
Õ    Œœ ŒŒ –— –– “” “
‘ ““ ’÷ ’
◊ ’
ÿ ’’ Ÿ⁄ ŸŸ €‹ €€ ›ﬁ ›
ﬂ ›› ‡· ‡
‚ ‡
„ ‡‡ ‰Â ‰‰ ÊÁ ÊÊ ËÈ Ë
Í ËË ÎÏ Î
Ì Î
Ó ÎÎ Ô ÔÔ ÒÚ ÒÒ ÛÙ Û
ı ÛÛ ˆ˜ ˆ
¯ ˆ
˘ ˆˆ ˙¸ ˚
˝ ˚
˛ ˚˚ ˇÄ ˇ
Å ˇ
Ç ˇˇ ÉÑ É
Ö É
Ü ÉÉ áâ à
ä àà ãå ã
ç ã
é ãã èê è
ë è
í èè ìî ìì ïñ ïï ó
ò óó ôö ôô õú õõ ùû ùù ü
† üü °¢ °° £
§ ££ •¶ •
ß •• ®© ®® ™
´ ™™ ¨≠ ¨¨ ÆØ Æ
∞ ÆÆ ±≤ ±
≥ ±± ¥µ ¥¥ ∂
∑ ∂∂ ∏π ∏∏ ∫ª ∫
º ∫∫ Ωæ ΩΩ ø
¿ øø ¡¬ ¡¡ √ƒ √
≈ √
∆ √
« √√ »… »»  À    ÃÕ Ã
Œ ÃÃ œ– œ
— œ
“ œœ ”‘ ”
’ ”” ÷◊ ÷÷ ÿŸ ÿÿ ⁄€ ⁄
‹ ⁄
› ⁄
ﬁ ⁄
ﬂ ⁄
‡ ⁄
· ⁄
‚ ⁄
„ ⁄
	‰ ⁄

Â ⁄⁄ ÊÁ ÊÊ ËÈ ËË ÍÎ ÍÍ ÏÌ Ï
Ó ÏÏ Ô Ô
Ò ÔÔ ÚÛ Ú
Ù ÚÚ ıˆ ı
˜ ıı ¯˘ ¯
˙ ¯¯ ˚¸ ˚
˝ ˚
˛ ˚˚ ˇÄ ˇ
Å ˇˇ ÇÉ Ç
Ñ Ç
Ö ÇÇ Üá Ü
à ÜÜ âä â
ã â
å ââ çé ç
è çç êë ê
í ê
ì êê îï î
ñ îî óò ó
ô ó
ö óó õú õõ ùû ù
ü ùù †° †
¢ †
£ †† §• §§ ¶ß ¶¶ ®© ®
™ ®® ´¨ ´
≠ ´
Æ ´´ Ø∞ ØØ ±≤ ±± ≥¥ ≥
µ ≥≥ ∂∑ ∂
∏ ∂
π ∂∂ ∫ª ∫∫ ºΩ ºº æø æ
¿ ææ ¡¬ ¡
√ ¡
ƒ ¡¡ ≈∆ ≈≈ «» «« …  …
À …… ÃÕ Ã
Œ Ã
œ ÃÃ –— –– “” “
‘ ““ ’÷ ’
◊ ’
ÿ ’’ Ÿ⁄ ŸŸ €‹ €€ ›ﬁ ›
ﬂ ›› ‡· ‡
‚ ‡
„ ‡‡ ‰Â ‰‰ ÊÁ ÊÊ ËÈ Ë
Í ËË ÎÏ Î
Ì Î
Ó ÎÎ Ô ÔÔ ÒÚ ÒÒ ÛÙ Û
ı ÛÛ ˆ˜ ˆ
¯ ˆ
˘ ˆˆ ˙˚ ˙˙ ¸˝ ¸¸ ˛ˇ ˛
Ä ˛˛ ÅÇ Å
É Å
Ñ ÅÅ ÖÜ ÖÖ áà á
â áá äã ä
å ä
ç ää éè éé êë êê íì í
î íí ïñ ï
ó ï
ò ïï ôö ôô õú õõ ùû ù
ü ùù †° †
¢ †
£ †† §• §§ ¶ß ¶¶ ®© ®
™ ®® ´¨ ´
≠ ´
Æ ´´ Ø∞ ØØ ±≤ ±± ≥¥ ≥
µ ≥≥ ∂∑ ∂
∏ ∂
π ∂∂ ∫º ª
Ω ª
æ ª
ø ªª ¿¡ ¿
¬ ¿
√ ¿
ƒ ¿¿ ≈∆ ≈
« ≈
» ≈
… ≈≈  À  
Ã  
Õ  
Œ    œ– œ
— œ
“ œ
” œœ ‘
’ ‘‘ ÷◊ ÷
ÿ ÷÷ Ÿ
⁄ ŸŸ €‹ €
› €€ ﬁ
ﬂ ﬁﬁ ‡· ‡
‚ ‡‡ „
‰ „„ ÂÊ Â
Á ÂÂ Ë
È ËË ÍÎ Í
Ï ÍÍ Ì
Ó ÌÌ Ô
 ÔÔ Ò
Ú ÒÒ Û
Ù ÛÛ ı
ˆ ıı ˜
¯ ˜˜ ˘
˙ ˘˘ ˚
¸ ˚˚ ˝
˛ ˝˝ ˇ
Ä ˇˇ ÅÇ ŒÇ ”Ç ÿÉ ÕÉ “É ◊Ñ  Ñ œÑ ‘Ö ÃÖ —Ö ÷	Ü 	Ü Ü $Ü 1Ü ;Ü Ÿ
Ü è
Ü ”
Ü ˜
Ü õá á á *á 7á Aá ãá ìá ûá ™á ≥á œá ◊á ‚á Óá ˜á Ûá ˚á Üá íá õá óá üá ™á ∂á øà Àà –à ’â ‘â Ÿâ ﬁâ „â Ëä ﬁä Ïä ë
ä µã ‚ã Îã Ùã ã ˘ã Çã ï
ã û
ã ß
ã πã ¬ã À           !  #$ & '% )( +* -" /, 01 3 42 65 87 :; = >< @? BA D FE H J. K9 L M ON Q SR UP WT XV Z \C ]V ^ `[ a cb e gf i kj m on qN sR u w. x9 yC z[ {r |t } ~  Ä Å ÉÇ Ö áÜ â ãä ç	 èé ë
 ìí ï ó ô
 õ ù ü ° £ •	 ß ©
 ´ ≠ Ø ± ≥ µ	 ∑ π
 ª Ω ø ¡ √ ≈	 « … €⁄ ›‹ ﬂﬁ ·‹ „‚ ÂŸ ÁÊ È‹ ÍË ÏÎ ÓŸ Ô Ú‹ ÛÒ ıÙ ˜‡ ˘¯ ˚Ì ˝Ì ˛‰ Ä‰ Å¸ Çˆ Ñˆ Öˇ ÜÉ à‡ äâ åã é‡ êè íë îì ñï ò‡ ö$ õô ùú üû °ó £† §‡ ¶1 ß• ©® ´™ ≠‡ Ø; ∞Æ ≤± ¥≥ ∂ç ∏¢ π¨ ∫ ªñ Ωò øº ¡æ ¬ç ƒµ ≈¿ ∆ç »√ …ñ Àò Õç œ¢ –¨ —µ “√ ”  ‘Ã ’ ÷ ◊	 ÿ
 Ÿá €⁄ ›¿ ﬂY ·ﬁ ‚_ ‰‡ Â« Á„ Ë‹ ÍÊ Î Ìç ÓÈ Ï ÒC Ûµ ÙÈ ˆÚ ˜  ˘ï ˙È ¸¯ ˝, ˇ† ÄÈ Ç˛ É9 Ö¨ ÜÈ àÑ â‰ ã  çï éä êå ëÔ íö îú ñì òï ôä õó úı ùû ü† °û £† §ä ¶¢ ß˚ ®¢ ™§ ¨© Æ´ Øä ±≠ ≤Å ≥¶ µ® ∑¥ π∂ ∫ä º∏ Ωá æÌ ¿, ¬† √ø ≈¡ ∆è «™ …¨ À» Õ  Œø –Ã —ö “Æ ‘∞ ÷” ÿ’ Ÿø €◊ ‹• ›≤ ﬂ¥ ·ﬁ „‡ ‰ø Ê‚ Á∞ Ë∂ Í∏ ÏÈ ÓÎ Ôø ÒÌ Úª Ûˆ ı9 ˜¨ ¯Ù ˙ˆ ˚ƒ ¸∫ ˛º Ä˝ Çˇ ÉÙ ÖÅ Üœ áæ â¿ ãà çä éÙ êå ë⁄ í¬ îƒ ñì òï ôÙ õó úÂ ù∆ ü» °û £† §Ù ¶¢ ß ®‡ ´‰ ≠[ ÆÌ ∞[ ±ˆ ≥[ ¥‰ ∑  π  ª∏ º∂ æ∫ øÀ ¡ú √¿ ≈¬ ∆∂ »ƒ …Ã À† Õ  œÃ –∂ “Œ ”Õ ’§ ◊‘ Ÿ÷ ⁄∂ ‹ÿ ›Œ ﬂ® ·ﬁ „‡ ‰∂ Ê‚ ÁÌ Èœ Î, ÌÍ ÓË Ï ÒΩ Ú– Ù¨ ˆÛ ¯ı ˘Ë ˚˜ ¸« ˝— ˇ∞ Å˛ ÉÄ ÑË ÜÇ á— à“ ä¥ åâ éã èË ëç í€ ì” ï∏ óî ôñ öË úò ùÂ ûˆ †‘ ¢9 §° •ü ß£ ®Ô ©’ ´º ≠™ Ø¨ ∞ü ≤Æ ≥˙ ¥÷ ∂¿ ∏µ ∫∑ ªü Ωπ æÖ ø◊ ¡ƒ √¿ ≈¬ ∆ü »ƒ …ê  ÿ Ã» ŒÀ –Õ —ü ”œ ‘õ ’Ñ ÿ± Ÿè €¨ ‹º ›ö ﬂØ ‡« ·• „≤ ‰“ Â˘ Á¶ Ë‹ ÍŸ ÎÈ ÌÏ ÔÈ Ò ÛŸ ıÙ ˜‹ ¯ˆ ˙˘ ¸Ÿ ˛˝ Ä‹ Åˇ ÉÇ ÖÓ áÜ âÓ åÚ é  ê  íè ìç ïë ñÊ óÀ ôú õò ùö ûç †ú °◊ ¢Ã §† ¶£ ®• ©ç ´ß ¨⁄ ≠Õ Ø§ ±Æ ≥∞ ¥ç ∂≤ ∑ﬁ ∏Œ ∫® ºπ æª øç ¡Ω ¬‚ √˚ ≈œ «, …∆  ƒ Ã» Õî Œ– –¨ “œ ‘— ’ƒ ◊” ÿü Ÿ— €∞ ›⁄ ﬂ‹ ‡ƒ ‚ﬁ „™ ‰“ Ê¥ ËÂ ÍÁ Îƒ ÌÈ Óµ Ô” Ò∏ Û ıÚ ˆƒ ¯Ù ˘¿ ˙Ñ ¸‘ ˛9 Ä˝ Å˚ Éˇ ÑÀ Ö’ áº âÜ ãà å˚ éä è÷ ê÷ í¿ îë ñì ó˚ ôï ö· õ◊ ùƒ üú °û ¢˚ §† •Ï ¶ÿ ®» ™ß ¨© ≠˚ Ø´ ∞˜ ±Ú ¥[ µ⁄ ∂˚ ∏[ πﬁ ∫Ñ º[ Ω‚ æ˚ ¡˚ ¬Ú ƒÚ ≈¿ ∆Ñ »Ñ …√  « ÃÓ ŒÕ –œ “Ó ‘” ÷’ ÿ◊ ⁄Ÿ ‹Ó ﬁ$ ﬂ› ·‡ „‚ Â€ Á‰ ËÓ Í1 ÎÈ ÌÏ ÔÓ ÒÓ Û; ÙÚ ˆı ¯˜ ˙— ¸Ê ˝ ˛ ˇñ Åò ÉÄ ÖÇ Ü— à˘ âÑ ä— åá çñ èò ë— ìÊ î ï˘ ñá óé òê ô ö õ	 ú
 ùÀ üû °Ñ £Y •¢ ¶_ ®§ ©ã ´ß ¨† Æ™ Ø ±— ≤≠ ¥∞ µÊ ∂C ∏˘ π≠ ª∑ º◊ Ω  øŸ ¿≠ ¬æ √⁄ ƒ, ∆‰ «≠ …≈  ﬁ À9 Õ Œ≠ –Ã —‚ “Ú ‘  ÷Ÿ ◊” Ÿ’ ⁄≥ €ö ›ú ﬂ‹ ·ﬁ ‚” ‰‡ Â∫ Êû Ë† ÍÁ ÏÈ Ì” ÔÎ ¡ Ò¢ Û§ ıÚ ˜Ù ¯” ˙ˆ ˚» ¸¶ ˛® Ä	˝ Ç	ˇ É	” Ö	Å	 Ü	œ á	˚ â	, ã	‰ å	à	 é	ä	 è	ÿ ê	™ í	¨ î	ë	 ñ	ì	 ó	à	 ô	ï	 ö	„ õ	Æ ù	∞ ü	ú	 °	û	 ¢	à	 §	†	 •	Ó ¶	≤ ®	¥ ™	ß	 ¨	©	 ≠	à	 Ø	´	 ∞	˘ ±	∂ ≥	∏ µ	≤	 ∑	¥	 ∏	à	 ∫	∂	 ª	Ñ	 º	Ñ æ	9 ¿	 ¡	Ω	 √	ø	 ƒ	ç	 ≈	∫ «	º …	∆	 À	»	 Ã	Ω	 Œ	 	 œ	ò	 –	æ “	¿ ‘	—	 ÷	”	 ◊	Ω	 Ÿ	’	 ⁄	£	 €	¬ ›	ƒ ﬂ	‹	 ·	ﬁ	 ‚	Ω	 ‰	‡	 Â	Æ	 Ê	∆ Ë	» Í	Á	 Ï	È	 Ì	Ω	 Ô	Î	 	π	 Ò	Õ	 Ù	◊ ı	◊ ˆ	ç ˜	ÿ	 ˘	⁄ ˙	≥ ˚	ò ¸	„	 ˛	ﬁ ˇ	∑ Ä
£ Å
Ó	 É
‚ Ñ
ª Ö
Æ Ü
¬	 à
Ê â
Ê ä
Ç ã
Ÿ ç
å
 è
‹ ê
é
 í
ë
 î
é
 ñ
ï
 ò
Ÿ ö
ô
 ú
‹ ù
õ
 ü
û
 °
Ÿ £
¢
 •
‹ ¶
§
 ®
ß
 ™
ì
 ¨
´
 Æ
ì
 ∞
ó
 ≤
  ¥
  ∂
≥
 ∑
±
 π
µ
 ∫
á
 ª
À Ω
ú ø
º
 ¡
æ
 ¬
±
 ƒ
¿
 ≈
Û	 ∆
Ã »
†  
«
 Ã
…
 Õ
±
 œ
À
 –
¯	 —
Õ ”
§ ’
“
 ◊
‘
 ÿ
±
 ⁄
÷
 €
˝	 ‹
Œ ﬁ
® ‡
›
 ‚
ﬂ
 „
±
 Â
·
 Ê
Ç
 Á
†
 È
œ Î
, Ì
Í
 Ó
Ë
 
Ï
 Ò
∏
 Ú
– Ù
¨ ˆ
Û
 ¯
ı
 ˘
Ë
 ˚
˜
 ¸
√
 ˝
— ˇ
∞ Å˛
 ÉÄ ÑË
 ÜÇ áŒ
 à“ ä¥ åâ éã èË
 ëç íŸ
 ì” ï∏ óî ôñ öË
 úò ù‰
 û©
 †‘ ¢9 §° •ü ß£ ®Ô
 ©’ ´º ≠™ Ø¨ ∞ü ≤Æ ≥˙
 ¥÷ ∂¿ ∏µ ∫∑ ªü Ωπ æÖ ø◊ ¡ƒ √¿ ≈¬ ∆ü »ƒ …ê  ÿ Ã» ŒÀ –Õ —ü ”œ ‘õ ’ó
 ÿ[ Ÿ¯	 ⁄†
 ‹[ ›˝	 ﬁ©
 ‡[ ·Ç
 ‚†
 Â†
 Êó
 Ëó
 È‰ Í©
 Ï©
 ÌÁ ÓÎ ì
 ÚÒ ÙÛ ˆì
 ¯˜ ˙˘ ¸˚ ˛˝ Äì
 Ç$ ÉÅ ÖÑ áÜ âˇ ãà åì
 é1 èç ëê ìí ïì
 ó; òñ öô úõ ûı †ä °î ¢ £ñ •ò ß§ ©¶ ™ı ¨ù ≠® Æı ∞´ ±ñ ≥ò µı ∑ä ∏î πù ∫´ ª≤ º¥ Ω æ ø	 ¿
 ¡Ô √¬ ≈® «Y …∆  _ Ã» ÕØ œÀ –ƒ “Œ ” ’ı ÷— ÿ‘ Ÿá
 ⁄C ‹ù ›— ﬂ€ ‡Û	 ·  „˝ ‰— Ê‚ Á¯	 Ë, Íà Î— ÌÈ Ó˝	 Ô9 Òî Ú— Ù ıÇ
 ˆó
 ¯  ˙˝ ˚˜ ˝˘ ˛◊ ˇö Åú ÉÄ ÖÇ Ü˜ àÑ âﬁ äû å† éã êç ë˜ ìè îÂ ï¢ ó§ ôñ õò ú˜ ûö üÏ †¶ ¢® §° ¶£ ß˜ ©• ™Û ´†
 ≠, Øà ∞¨ ≤Æ ≥¸ ¥™ ∂¨ ∏µ ∫∑ ª¨ Ωπ æá øÆ ¡∞ √¿ ≈¬ ∆¨ »ƒ …í  ≤ Ã¥ ŒÀ –Õ —¨ ”œ ‘ù ’∂ ◊∏ Ÿ÷ €ÿ ‹¨ ﬁ⁄ ﬂ® ‡©
 ‚9 ‰î Â· Á„ Ë± È∫ Îº ÌÍ ÔÏ · ÚÓ Ûº Ùæ ˆ¿ ¯ı ˙˜ ˚· ˝˘ ˛« ˇ¬ Åƒ ÉÄ ÖÇ Ü· àÑ â“ ä∆ å» éã êç ë· ìè î› ïÒ òÛ	 ôÛ	 ö± õ¸ ù¯	 û◊ üº †á ¢˝	 £€ §« •í ßÇ
 ®ﬂ ©“ ™Ê ¨á
 ≠á
 Æ¶ ØŸ ±∞ ≥‹ ¥≤ ∂µ ∏≤ ∫π ºŸ æΩ ¿‹ ¡ø √¬ ≈Ÿ «∆ …‹  » ÃÀ Œ∑ –œ “∑ ‘ª ÷  ÿ  ⁄◊ €’ ›Ÿ ﬁ´ ﬂÀ ·ú „‡ Â‚ Ê’ Ë‰ Èó ÍÃ Ï† ÓÎ Ì Ò’ ÛÔ Ùú ıÕ ˜§ ˘ˆ ˚¯ ¸’ ˛˙ ˇ° ÄŒ Ç® ÑÅ ÜÉ á’ âÖ ä¶ ãƒ çœ è, ëé íå îê ï‹ ñ– ò¨ öó úô ùå üõ †Á °— £∞ •¢ ß§ ®å ™¶ ´Ú ¨“ Æ¥ ∞≠ ≤Ø ≥å µ± ∂˝ ∑” π∏ ª∏ Ω∫ æå ¿º ¡à ¬Õ ƒ‘ ∆9 »≈ …√ À« Ãì Õ’ œº —Œ ”– ‘√ ÷“ ◊û ÿ÷ ⁄¿ ‹Ÿ ﬁ€ ﬂ√ ·› ‚© „◊ Âƒ Á‰ ÈÊ Í√ ÏË Ì¥ Óÿ » ÚÔ ÙÒ ı√ ˜Û ¯ø ˘ª ¸[ ˝ú ˛ƒ Ä[ Å° ÇÕ Ñ[ Ö¶ Üƒ âƒ äª åª çà éÕ êÕ ëã íè î∑ ñï òó ö∑ úõ ûù †ü ¢° §∑ ¶$ ß• ©® ´™ ≠£ Ø¨ ∞∑ ≤1 ≥± µ¥ ∑∂ π∑ ª; º∫ æΩ ¿ø ¬ô ƒÆ ≈∏ ∆ «ñ …ò À» Õ  Œô –¡ —Ã “ô ‘œ ’ñ ◊ò Ÿô €Æ ‹∏ ›¡ ﬁœ ﬂ÷ ‡ÿ · ‚ „	 ‰
 Âì ÁÊ ÈÃ ÎY ÌÍ Ó_ Ï Ò” ÛÔ ÙË ˆÚ ˜ ˘ô ˙ı ¸¯ ˝´ ˛C Ä¡ Åı Éˇ Ñó Ö  á° àı äÜ ãú å, é¨ èı ëç í° ì9 ï∏ ñı òî ô¶ öª ú  û° üõ °ù ¢˚ £ö •ú ß§ ©¶ ™õ ¨® ≠Ç Æû ∞† ≤Ø ¥± µõ ∑≥ ∏â π¢ ª§ Ω∫ øº ¿õ ¬æ √ê ƒ¶ ∆® »≈  « Àõ Õ… Œó œƒ —, ”¨ ‘– ÷“ ◊† ÿ™ ⁄¨ ‹Ÿ ﬁ€ ﬂ– ·› ‚´ „Æ Â∞ Á‰ ÈÊ Í– ÏË Ì∂ Ó≤ ¥ ÚÔ ÙÒ ı– ˜Û ¯¡ ˘∂ ˚∏ ˝˙ ˇ¸ Ä– Ç˛ ÉÃ ÑÕ Ü9 à∏ âÖ ãá å’ ç∫ èº ëé ìê îÖ ñí ó‡ òæ ö¿ úô ûõ üÖ °ù ¢Î £¬ •ƒ ß§ ©¶ ™Ö ¨® ≠ˆ Æ∆ ∞» ≤Ø ¥± µÖ ∑≥ ∏Å πï ºó Ωó æ’ ø† ¡ú ¬˚ √‡ ƒ´ ∆° «ˇ »Î …∂ À¶ ÃÉ Õˆ Œä –´ —´ “  ” ’œ ◊‘ ÿ ⁄¿ ‹Ÿ ›( ﬂ≈ ·ﬁ ‚5 ‰  Ê„ Á? Èª ÎË Ïí Óé ä ÚÜ ÙÇ ˆn ¯j ˙f ¸b ˛E Ä  ä˙ ¸˙ ™© ◊™ ◊™ ¨™ ∂à ¿à ãµ ◊÷ ◊Ú	 Û	ã Û	ã ≥ã ç≠
 ‰≠
 Ø
ø Û	≤ Û	ñ óØ
 óØ
 ◊Ø
 ±
— à— ”„ ó÷ ó∫ ª” ª” ˚” ’Å äá ª˙ ª ä êê ìì íí åå ïï îî éé èè çç ëëG åå G« îî «Ï îî Ïá ëë áÿ îî ÿπ	 îî π	• îî •Ô
 îî Ô
Â îî Â˝ îî ˝à îî à† îî †í ìì íª îî ª√ îî √¿ èè ¿ê îî êÆ îî Æ◊ îî ◊® îî ®« îî «™ îî ™â îî âÅ îî Å∑ éé ∑p åå p´ ëë ´« îî «‰
 îî ‰
Â îî Âˆ îî ˆ_ íí _Ö îî Öõ îî õÛ îî Ûﬂ îî ﬂ∂ îî ∂´ îî ´∂ îî ∂˝ ïï ˝º îî º† îî †µ îî µ≤ îî ≤ƒ îî ƒ£ îî £À êê À˚ éé ˚Ÿ
 îî Ÿ
© îî ©¡ îî ¡´ îî ´Ì ïï Ìå åå åı ïï ıØ íí Ø± îî ±¿ îî ¿¸ îî ¸¬	 îî ¬	‡ îî ‡ó îî óõ îî õ∑ îî ∑Î îî Î◊ îî ◊∂ ìì ∂˘ ïï ˘« îî «Ò ïï Ò[ ëë [á îî á€ îî €√ éé √á îî áˇ îî ˇÁ îî Á± îî ±” íí ”“ îî “ä îî äù îî ùá êê á∞ îî ∞ˆ îî ˆ± îî ±⁄ ìì ⁄Å îî Åì îî ìè îî èÑ îî Ñò îî òÃ èè Ã« íí «¨ îî ¨— îî —Ñ	 îî Ñ	Õ	 îî Õ	∆ êê ∆ç îî çç	 îî ç	Ó	 îî Ó	˙ îî ˙Ø îî ØÊ îî Ê‹ îî ‹˜ ïï ˜É îî Éí îî íê îî ê„ îî „Æ	 îî Æ	“ îî “¥ îî ¥î îî î çç Â îî ÂÚ îî Ú˙
 îî ˙
í îî íœ ëë œ˘ îî ˘î åå î« îî «  îî  · îî ·è îî è îî h åå hÍ êê ÍÎ îî ÎÉ îî É’ îî ’¶ îî ¶ı îî ı’ îî ’Û ïï ÛV èè VŒ ìì ŒÑ èè Ñ˚ îî ˚ã íí ã≥ îî ≥Ω îî ΩÔ îî Ô» îî »Á îî ÁÎ îî Î¸ îî ¸¢ êê ¢√
 îî √
Ò îî Òì êê ì‡ îî ‡á îî áº îî ºd åå d∫ îî ∫÷ îî ÷¡ îî ¡Œ
 îî Œ
I éé I˘ îî ˘Ô ïï Ô• îî •ﬁ îî ﬁö îî öò	 îî ò	⁄ îî ⁄ﬁ êê ﬁö îî ö˚ îî ˚ÿ	 îî ÿ	“ îî “Ç îî Çü éé üª îî ª› îî ›Ç îî Ç∏
 îî ∏
Ö îî Öû îî û√ ëë √€ îî €À îî À® èè ®ø îî øÔ îî Ôˇ îî ˇº îî ºè îî èœ îî œÓ îî Ó„	 îî „	˚ îî ˚Ï îî Ï≥ îî ≥£	 îî £	v ìì vY êê Yã îî ãê îî êl åå lÃ îî Ã¶ îî ¶ï îî ï˚ ïï ˚Ô êê Ôˇ ïï ˇü îî üÑ åå Ñ˜ îî ˜à åå àê åå êœ îî œ
ñ ¯
ñ ™
ñ Ü
ñ ã
ñ ´

ñ Ø

ñ œ
ñ ”	ó R
ó ò
ó ö
ó ú
ó û
ó †
ó ¢
ó §
ó ¶
ó ®
ó ™
ó ¨
ó Æ
ó ∞
ó ≤
ó ¥
ó ∂
ó ∏
ó ∫
ó º
ó æ
ó ¿
ó ¬
ó ƒ
ó ∆
ó »
ó À
ó Ã
ó Õ
ó Œ
ó –
ó —
ó “
ó ”
ó ’
ó ÷
ó ◊
ó ÿò "ò óò €ò ˇò £
ô ‹
ô ä
ô ø
ô Ù
ô ∂
ô Ë
ô ü
ô ç
ô ƒ
ô ˚
ô †
ô ”
ô à	
ô Ω	
ô ±

ô Ë

ô ü
ô ƒ
ô ˜
ô ¨
ô ·
ô ’
ô å
ô √
ô Ë
ô õ
ô –
ô Ö
ö Ô
ö ı
ö ˚
ö Å
ö á
ö ¨
ö Ø
ö ≤
ö Ω
ö «
ö —
ö €
ö Â
ö ◊
ö ◊
ö ⁄
ö ﬁ
ö ‚
ö Ê
ö Ê
õ ™
õ ã
õ Ø

õ ”
ú ‘
ú Ô
ú ∞ù ù ù ù ù ù ù ù ù 	ù 
	ù $	ù .
ù ™
ù ¨
ù Æ
ù ∞
ù ≤
ù ¥
ù ∂
ù ∏
ù –
ù —
ù “
ù ”
ù ¢
ù Ê
ù ä
ù Æ
û ˝	ü ;	ü R
ü ò
ü ∫
ü º
ü æ
ü ¿
ü ¬
ü ƒ
ü ∆
ü »
ü ’
ü ÷
ü ◊
ü ÿ
† ô

° Ω
¢ ¢
	£ 	£ 
£ ⁄
£ ‹
§ ⁄
§ û
§ ¬
§ Ê	• 1¶ G¶ d¶ h¶ l¶ p¶ Ñ¶ à¶ å¶ ê¶ î¶ Ì¶ Ô¶ Ò¶ Û¶ ı¶ ˜¶ ˘¶ ˚¶ ˝¶ ˇß 	ß "
ß ö
ß ú
ß û
ß †
ß ¢
ß §
ß ¶
ß ®
ß À
ß Ã
ß Õ
ß Œ
ß ó
ß €
ß ˇ
ß £
® œ
® Ê
© Ù
™ ∆
´  
´ å
"
compute_flux"
llvm.lifetime.start.p0i8"
_Z13get_global_idj"
compute_velocity"
compute_speed_sqd"

_Z4sqrtf"
compute_pressure"
compute_speed_of_sound"
compute_flux_contribution"
llvm.fmuladd.f32"
llvm.lifetime.end.p0i8*ò
rodinia-3.1-cfd-compute_flux.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282Å

wgsize_log1p
IÔÇA
 
transfer_bytes_log1p
IÔÇA

wgsize
¿

devmap_label


transfer_bytes
ƒ‹é