

[external]
?allocaB5
3
	full_text&
$
"%10 = alloca [4 x float], align 16
CbitcastB8
6
	full_text)
'
%%11 = bitcast [4 x float]* %10 to i8*
5[4 x float]*B#
!
	full_text

[4 x float]* %10
@allocaB6
4
	full_text'
%
#%12 = alloca [16 x float], align 16
?allocaB5
3
	full_text&
$
"%13 = alloca [4 x float], align 16
KcallBC
A
	full_text4
2
0%14 = tail call i64 @_Z12get_local_idj(i32 0) #5
6truncB-
+
	full_text

%15 = trunc i64 %14 to i32
#i64B

	full_text
	
i64 %14
KcallBC
A
	full_text4
2
0%16 = tail call i64 @_Z12get_local_idj(i32 1) #5
6truncB-
+
	full_text

%17 = trunc i64 %16 to i32
#i64B

	full_text
	
i64 %16
KcallBC
A
	full_text4
2
0%18 = tail call i64 @_Z12get_group_idj(i32 0) #5
6truncB-
+
	full_text

%19 = trunc i64 %18 to i32
#i64B

	full_text
	
i64 %18
.shlB'
%
	full_text

%20 = shl i32 %19, 6
#i32B

	full_text
	
i32 %19
KcallBC
A
	full_text4
2
0%21 = tail call i64 @_Z12get_group_idj(i32 1) #5
6truncB-
+
	full_text

%22 = trunc i64 %21 to i32
#i64B

	full_text
	
i64 %21
.shlB'
%
	full_text

%23 = shl i32 %22, 4
#i32B

	full_text
	
i32 %22
.shlB'
%
	full_text

%24 = shl i32 %17, 4
#i32B

	full_text
	
i32 %17
4addB-
+
	full_text

%25 = add nsw i32 %24, %15
#i32B

	full_text
	
i32 %24
#i32B

	full_text
	
i32 %15
4addB-
+
	full_text

%26 = add nsw i32 %25, %20
#i32B

	full_text
	
i32 %25
#i32B

	full_text
	
i32 %20
4sextB,
*
	full_text

%27 = sext i32 %26 to i64
#i32B

	full_text
	
i32 %26
ZgetelementptrBI
G
	full_text:
8
6%28 = getelementptr inbounds float, float* %0, i64 %27
#i64B

	full_text
	
i64 %27
3mulB,
*
	full_text

%29 = mul nsw i32 %17, %3
#i32B

	full_text
	
i32 %17
0addB)
'
	full_text

%30 = add i32 %29, %15
#i32B

	full_text
	
i32 %29
#i32B

	full_text
	
i32 %15
0addB)
'
	full_text

%31 = add i32 %30, %23
#i32B

	full_text
	
i32 %30
#i32B

	full_text
	
i32 %23
4sextB,
*
	full_text

%32 = sext i32 %31 to i64
#i32B

	full_text
	
i32 %31
3mulB,
*
	full_text

%33 = mul nsw i32 %23, %5
#i32B

	full_text
	
i32 %23
4addB-
+
	full_text

%34 = add nsw i32 %33, %26
#i32B

	full_text
	
i32 %33
#i32B

	full_text
	
i32 %26
4sextB,
*
	full_text

%35 = sext i32 %34 to i64
#i32B

	full_text
	
i32 %34
ZgetelementptrBI
G
	full_text:
8
6%36 = getelementptr inbounds float, float* %4, i64 %35
#i64B

	full_text
	
i64 %35
ZcallBR
P
	full_textC
A
?call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %11) #6
#i8*B

	full_text
	
i8* %11
3sextB+
)
	full_text

%37 = sext i32 %1 to i64
>bitcastB3
1
	full_text$
"
 %38 = bitcast float* %28 to i32*
)float*B

	full_text


float* %28
FloadB>
<
	full_text/
-
+%39 = load i32, i32* %38, align 4, !tbaa !8
%i32*B

	full_text


i32* %38
lgetelementptrB[
Y
	full_textL
J
H%40 = getelementptr inbounds [4 x float], [4 x float]* %10, i64 0, i64 0
5[4 x float]*B#
!
	full_text

[4 x float]* %10
DbitcastB9
7
	full_text*
(
&%41 = bitcast [4 x float]* %10 to i32*
5[4 x float]*B#
!
	full_text

[4 x float]* %10
GstoreB>
<
	full_text/
-
+store i32 %39, i32* %41, align 16, !tbaa !8
#i32B

	full_text
	
i32 %39
%i32*B

	full_text


i32* %41
[getelementptrBJ
H
	full_text;
9
7%42 = getelementptr inbounds float, float* %28, i64 %37
)float*B

	full_text


float* %28
#i64B

	full_text
	
i64 %37
>bitcastB3
1
	full_text$
"
 %43 = bitcast float* %42 to i32*
)float*B

	full_text


float* %42
FloadB>
<
	full_text/
-
+%44 = load i32, i32* %43, align 4, !tbaa !8
%i32*B

	full_text


i32* %43
lgetelementptrB[
Y
	full_textL
J
H%45 = getelementptr inbounds [4 x float], [4 x float]* %10, i64 0, i64 1
5[4 x float]*B#
!
	full_text

[4 x float]* %10
>bitcastB3
1
	full_text$
"
 %46 = bitcast float* %45 to i32*
)float*B

	full_text


float* %45
FstoreB=
;
	full_text.
,
*store i32 %44, i32* %46, align 4, !tbaa !8
#i32B

	full_text
	
i32 %44
%i32*B

	full_text


i32* %46
2shlB+
)
	full_text

%47 = shl nsw i64 %37, 1
#i64B

	full_text
	
i64 %37
[getelementptrBJ
H
	full_text;
9
7%48 = getelementptr inbounds float, float* %28, i64 %47
)float*B

	full_text


float* %28
#i64B

	full_text
	
i64 %47
>bitcastB3
1
	full_text$
"
 %49 = bitcast float* %48 to i32*
)float*B

	full_text


float* %48
FloadB>
<
	full_text/
-
+%50 = load i32, i32* %49, align 4, !tbaa !8
%i32*B

	full_text


i32* %49
lgetelementptrB[
Y
	full_textL
J
H%51 = getelementptr inbounds [4 x float], [4 x float]* %10, i64 0, i64 2
5[4 x float]*B#
!
	full_text

[4 x float]* %10
>bitcastB3
1
	full_text$
"
 %52 = bitcast float* %51 to i32*
)float*B

	full_text


float* %51
FstoreB=
;
	full_text.
,
*store i32 %50, i32* %52, align 8, !tbaa !8
#i32B

	full_text
	
i32 %50
%i32*B

	full_text


i32* %52
2mulB+
)
	full_text

%53 = mul nsw i64 %37, 3
#i64B

	full_text
	
i64 %37
[getelementptrBJ
H
	full_text;
9
7%54 = getelementptr inbounds float, float* %28, i64 %53
)float*B

	full_text


float* %28
#i64B

	full_text
	
i64 %53
>bitcastB3
1
	full_text$
"
 %55 = bitcast float* %54 to i32*
)float*B

	full_text


float* %54
FloadB>
<
	full_text/
-
+%56 = load i32, i32* %55, align 4, !tbaa !8
%i32*B

	full_text


i32* %55
lgetelementptrB[
Y
	full_textL
J
H%57 = getelementptr inbounds [4 x float], [4 x float]* %10, i64 0, i64 3
5[4 x float]*B#
!
	full_text

[4 x float]* %10
>bitcastB3
1
	full_text$
"
 %58 = bitcast float* %57 to i32*
)float*B

	full_text


float* %57
FstoreB=
;
	full_text.
,
*store i32 %56, i32* %58, align 4, !tbaa !8
#i32B

	full_text
	
i32 %56
%i32*B

	full_text


i32* %58
DbitcastB9
7
	full_text*
(
&%59 = bitcast [16 x float]* %12 to i8*
7[16 x float]*B$
"
	full_text

[16 x float]* %12
CbitcastB8
6
	full_text)
'
%%60 = bitcast [4 x float]* %13 to i8*
5[4 x float]*B#
!
	full_text

[4 x float]* %13
ZgetelementptrBI
G
	full_text:
8
6%61 = getelementptr inbounds float, float* %2, i64 %32
#i64B

	full_text
	
i64 %32
JloadBB
@
	full_text3
1
/%62 = load float, float* %61, align 4, !tbaa !8
)float*B

	full_text


float* %61
1shlB*
(
	full_text

%63 = shl nsw i32 %1, 2
1shlB*
(
	full_text

%64 = shl nsw i32 %3, 2
ZcallBR
P
	full_textC
A
?call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %59) #6
#i8*B

	full_text
	
i8* %59
dcallB\
Z
	full_textM
K
Icall void @llvm.memset.p0i8.i64(i8* align 16 %59, i8 0, i64 64, i1 false)
#i8*B

	full_text
	
i8* %59
4sextB,
*
	full_text

%65 = sext i32 %63 to i64
#i32B

	full_text
	
i32 %63
4sextB,
*
	full_text

%66 = sext i32 %64 to i64
#i32B

	full_text
	
i32 %64
/shlB(
&
	full_text

%67 = shl i64 %16, 32
#i64B

	full_text
	
i64 %16
7ashrB/
-
	full_text 

%68 = ashr exact i64 %67, 32
#i64B

	full_text
	
i64 %67
/shlB(
&
	full_text

%69 = shl i64 %14, 32
#i64B

	full_text
	
i64 %14
7ashrB/
-
	full_text 

%70 = ashr exact i64 %69, 32
#i64B

	full_text
	
i64 %69
�getelementptrB|
z
	full_textm
k
i%71 = getelementptr inbounds [4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 %68, i64 %70
#i64B

	full_text
	
i64 %68
#i64B

	full_text
	
i64 %70
DbitcastB9
7
	full_text*
(
&%72 = bitcast [4 x float]* %10 to i32*
5[4 x float]*B#
!
	full_text

[4 x float]* %10
1shlB*
(
	full_text

%73 = shl nsw i32 %1, 1
4sextB,
*
	full_text

%74 = sext i32 %73 to i64
#i32B

	full_text
	
i32 %73
1mulB*
(
	full_text

%75 = mul nsw i32 %1, 3
4sextB,
*
	full_text

%76 = sext i32 %75 to i64
#i32B

	full_text
	
i32 %75
lgetelementptrB[
Y
	full_textL
J
H%77 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 0
5[4 x float]*B#
!
	full_text

[4 x float]* %13
ngetelementptrB]
[
	full_textN
L
J%78 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 0
7[16 x float]*B$
"
	full_text

[16 x float]* %12
ngetelementptrB]
[
	full_textN
L
J%79 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 1
7[16 x float]*B$
"
	full_text

[16 x float]* %12
ngetelementptrB]
[
	full_textN
L
J%80 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 2
7[16 x float]*B$
"
	full_text

[16 x float]* %12
ngetelementptrB]
[
	full_textN
L
J%81 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 3
7[16 x float]*B$
"
	full_text

[16 x float]* %12
ngetelementptrB]
[
	full_textN
L
J%82 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 4
7[16 x float]*B$
"
	full_text

[16 x float]* %12
ngetelementptrB]
[
	full_textN
L
J%83 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 5
7[16 x float]*B$
"
	full_text

[16 x float]* %12
ngetelementptrB]
[
	full_textN
L
J%84 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 6
7[16 x float]*B$
"
	full_text

[16 x float]* %12
ngetelementptrB]
[
	full_textN
L
J%85 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 7
7[16 x float]*B$
"
	full_text

[16 x float]* %12
ngetelementptrB]
[
	full_textN
L
J%86 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 8
7[16 x float]*B$
"
	full_text

[16 x float]* %12
ngetelementptrB]
[
	full_textN
L
J%87 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 9
7[16 x float]*B$
"
	full_text

[16 x float]* %12
ogetelementptrB^
\
	full_textO
M
K%88 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 10
7[16 x float]*B$
"
	full_text

[16 x float]* %12
ogetelementptrB^
\
	full_textO
M
K%89 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 11
7[16 x float]*B$
"
	full_text

[16 x float]* %12
ogetelementptrB^
\
	full_textO
M
K%90 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 12
7[16 x float]*B$
"
	full_text

[16 x float]* %12
ogetelementptrB^
\
	full_textO
M
K%91 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 13
7[16 x float]*B$
"
	full_text

[16 x float]* %12
ogetelementptrB^
\
	full_textO
M
K%92 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 14
7[16 x float]*B$
"
	full_text

[16 x float]* %12
ogetelementptrB^
\
	full_textO
M
K%93 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 15
7[16 x float]*B$
"
	full_text

[16 x float]* %12
lgetelementptrB[
Y
	full_textL
J
H%94 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 1
5[4 x float]*B#
!
	full_text

[4 x float]* %13
lgetelementptrB[
Y
	full_textL
J
H%95 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 2
5[4 x float]*B#
!
	full_text

[4 x float]* %13
lgetelementptrB[
Y
	full_textL
J
H%96 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 3
5[4 x float]*B#
!
	full_text

[4 x float]* %13
2mulB+
)
	full_text

%97 = mul nsw i32 %6, %3
%brB

	full_text

br label %98
Ophi8BF
D
	full_text7
5
3%99 = phi float [ 0.000000e+00, %9 ], [ %264, %98 ]
*float8B

	full_text


float %264
Pphi8BG
E
	full_text8
6
4%100 = phi float [ 0.000000e+00, %9 ], [ %262, %98 ]
*float8B

	full_text


float %262
Pphi8BG
E
	full_text8
6
4%101 = phi float [ 0.000000e+00, %9 ], [ %260, %98 ]
*float8B

	full_text


float %260
Pphi8BG
E
	full_text8
6
4%102 = phi float [ 0.000000e+00, %9 ], [ %258, %98 ]
*float8B

	full_text


float %258
Pphi8BG
E
	full_text8
6
4%103 = phi float [ 0.000000e+00, %9 ], [ %256, %98 ]
*float8B

	full_text


float %256
Pphi8BG
E
	full_text8
6
4%104 = phi float [ 0.000000e+00, %9 ], [ %254, %98 ]
*float8B

	full_text


float %254
Pphi8BG
E
	full_text8
6
4%105 = phi float [ 0.000000e+00, %9 ], [ %252, %98 ]
*float8B

	full_text


float %252
Pphi8BG
E
	full_text8
6
4%106 = phi float [ 0.000000e+00, %9 ], [ %250, %98 ]
*float8B

	full_text


float %250
Pphi8BG
E
	full_text8
6
4%107 = phi float [ 0.000000e+00, %9 ], [ %248, %98 ]
*float8B

	full_text


float %248
Pphi8BG
E
	full_text8
6
4%108 = phi float [ 0.000000e+00, %9 ], [ %246, %98 ]
*float8B

	full_text


float %246
Pphi8BG
E
	full_text8
6
4%109 = phi float [ 0.000000e+00, %9 ], [ %244, %98 ]
*float8B

	full_text


float %244
Pphi8BG
E
	full_text8
6
4%110 = phi float [ 0.000000e+00, %9 ], [ %242, %98 ]
*float8B

	full_text


float %242
Pphi8BG
E
	full_text8
6
4%111 = phi float [ 0.000000e+00, %9 ], [ %240, %98 ]
*float8B

	full_text


float %240
Pphi8BG
E
	full_text8
6
4%112 = phi float [ 0.000000e+00, %9 ], [ %238, %98 ]
*float8B

	full_text


float %238
Pphi8BG
E
	full_text8
6
4%113 = phi float [ 0.000000e+00, %9 ], [ %236, %98 ]
*float8B

	full_text


float %236
Pphi8BG
E
	full_text8
6
4%114 = phi float [ 0.000000e+00, %9 ], [ %234, %98 ]
*float8B

	full_text


float %234
Hphi8B?
=
	full_text0
.
,%115 = phi float* [ %61, %9 ], [ %120, %98 ]
+float*8B

	full_text


float* %61
,float*8B

	full_text

float* %120
Hphi8B?
=
	full_text0
.
,%116 = phi float* [ %28, %9 ], [ %119, %98 ]
+float*8B

	full_text


float* %28
,float*8B

	full_text

float* %119
Ephi8B<
:
	full_text-
+
)%117 = phi i32 [ %64, %9 ], [ %265, %98 ]
%i328B

	full_text
	
i32 %64
&i328B

	full_text


i32 %265
Gphi8B>
<
	full_text/
-
+%118 = phi float [ %62, %9 ], [ %132, %98 ]
)float8B

	full_text

	float %62
*float8B

	full_text


float %132
_getelementptr8BL
J
	full_text=
;
9%119 = getelementptr inbounds float, float* %116, i64 %65
,float*8B

	full_text

float* %116
%i648B

	full_text
	
i64 %65
\call8BR
P
	full_textC
A
?call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %60) #6
%i8*8B

	full_text
	
i8* %60
wcall8Bm
k
	full_text^
\
Zcall void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %60, i8* align 16 %11, i64 16, i1 false)
%i8*8B

	full_text
	
i8* %60
%i8*8B

	full_text
	
i8* %11
_getelementptr8BL
J
	full_text=
;
9%120 = getelementptr inbounds float, float* %115, i64 %66
,float*8B

	full_text

float* %115
%i648B

	full_text
	
i64 %66
Mstore8BB
@
	full_text3
1
/store float %118, float* %71, align 4, !tbaa !8
*float8B

	full_text


float %118
+float*8B

	full_text


float* %71
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #7
Bbitcast8B5
3
	full_text&
$
"%121 = bitcast float* %119 to i32*
,float*8B

	full_text

float* %119
Jload8B@
>
	full_text1
/
-%122 = load i32, i32* %121, align 4, !tbaa !8
(i32*8B

	full_text

	i32* %121
Jstore8B?
=
	full_text0
.
,store i32 %122, i32* %72, align 16, !tbaa !8
&i328B

	full_text


i32 %122
'i32*8B

	full_text


i32* %72
_getelementptr8BL
J
	full_text=
;
9%123 = getelementptr inbounds float, float* %119, i64 %37
,float*8B

	full_text

float* %119
%i648B

	full_text
	
i64 %37
Bbitcast8B5
3
	full_text&
$
"%124 = bitcast float* %123 to i32*
,float*8B

	full_text

float* %123
Jload8B@
>
	full_text1
/
-%125 = load i32, i32* %124, align 4, !tbaa !8
(i32*8B

	full_text

	i32* %124
Istore8B>
<
	full_text/
-
+store i32 %125, i32* %46, align 4, !tbaa !8
&i328B

	full_text


i32 %125
'i32*8B

	full_text


i32* %46
_getelementptr8BL
J
	full_text=
;
9%126 = getelementptr inbounds float, float* %119, i64 %74
,float*8B

	full_text

float* %119
%i648B

	full_text
	
i64 %74
Bbitcast8B5
3
	full_text&
$
"%127 = bitcast float* %126 to i32*
,float*8B

	full_text

float* %126
Jload8B@
>
	full_text1
/
-%128 = load i32, i32* %127, align 4, !tbaa !8
(i32*8B

	full_text

	i32* %127
Istore8B>
<
	full_text/
-
+store i32 %128, i32* %52, align 8, !tbaa !8
&i328B

	full_text


i32 %128
'i32*8B

	full_text


i32* %52
_getelementptr8BL
J
	full_text=
;
9%129 = getelementptr inbounds float, float* %119, i64 %76
,float*8B

	full_text

float* %119
%i648B

	full_text
	
i64 %76
Bbitcast8B5
3
	full_text&
$
"%130 = bitcast float* %129 to i32*
,float*8B

	full_text

float* %129
Jload8B@
>
	full_text1
/
-%131 = load i32, i32* %130, align 4, !tbaa !8
(i32*8B

	full_text

	i32* %130
Istore8B>
<
	full_text/
-
+store i32 %131, i32* %58, align 4, !tbaa !8
&i328B

	full_text


i32 %131
'i32*8B

	full_text


i32* %58
Nload8BD
B
	full_text5
3
1%132 = load float, float* %120, align 4, !tbaa !8
,float*8B

	full_text

float* %120
Nload8BD
B
	full_text5
3
1%133 = load float, float* %77, align 16, !tbaa !8
+float*8B

	full_text


float* %77
�load8B�
�
	full_text�
�
�%134 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 0, i64 0), align 16, !tbaa !8
icall8B_
]
	full_textP
N
L%135 = tail call float @llvm.fmuladd.f32(float %133, float %134, float %114)
*float8B

	full_text


float %133
*float8B

	full_text


float %134
*float8B

	full_text


float %114
�load8B�
�
	full_text�
�
�%136 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 0, i64 1), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%137 = tail call float @llvm.fmuladd.f32(float %133, float %136, float %113)
*float8B

	full_text


float %133
*float8B

	full_text


float %136
*float8B

	full_text


float %113
�load8B�
�
	full_text�
�
�%138 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 0, i64 2), align 8, !tbaa !8
icall8B_
]
	full_textP
N
L%139 = tail call float @llvm.fmuladd.f32(float %133, float %138, float %112)
*float8B

	full_text


float %133
*float8B

	full_text


float %138
*float8B

	full_text


float %112
�load8B�
�
	full_text�
�
�%140 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 0, i64 3), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%141 = tail call float @llvm.fmuladd.f32(float %133, float %140, float %111)
*float8B

	full_text


float %133
*float8B

	full_text


float %140
*float8B

	full_text


float %111
�load8B�
�
	full_text�
�
�%142 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 0, i64 4), align 16, !tbaa !8
icall8B_
]
	full_textP
N
L%143 = tail call float @llvm.fmuladd.f32(float %133, float %142, float %110)
*float8B

	full_text


float %133
*float8B

	full_text


float %142
*float8B

	full_text


float %110
�load8B�
�
	full_text�
�
�%144 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 0, i64 5), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%145 = tail call float @llvm.fmuladd.f32(float %133, float %144, float %109)
*float8B

	full_text


float %133
*float8B

	full_text


float %144
*float8B

	full_text


float %109
�load8B�
�
	full_text�
�
�%146 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 0, i64 6), align 8, !tbaa !8
icall8B_
]
	full_textP
N
L%147 = tail call float @llvm.fmuladd.f32(float %133, float %146, float %108)
*float8B

	full_text


float %133
*float8B

	full_text


float %146
*float8B

	full_text


float %108
�load8B�
�
	full_text�
�
�%148 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 0, i64 7), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%149 = tail call float @llvm.fmuladd.f32(float %133, float %148, float %107)
*float8B

	full_text


float %133
*float8B

	full_text


float %148
*float8B

	full_text


float %107
�load8B�
�
	full_text�
�
�%150 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 0, i64 8), align 16, !tbaa !8
icall8B_
]
	full_textP
N
L%151 = tail call float @llvm.fmuladd.f32(float %133, float %150, float %106)
*float8B

	full_text


float %133
*float8B

	full_text


float %150
*float8B

	full_text


float %106
�load8B�
�
	full_text�
�
�%152 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 0, i64 9), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%153 = tail call float @llvm.fmuladd.f32(float %133, float %152, float %105)
*float8B

	full_text


float %133
*float8B

	full_text


float %152
*float8B

	full_text


float %105
�load8B�
�
	full_text�
�
�%154 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 0, i64 10), align 8, !tbaa !8
icall8B_
]
	full_textP
N
L%155 = tail call float @llvm.fmuladd.f32(float %133, float %154, float %104)
*float8B

	full_text


float %133
*float8B

	full_text


float %154
*float8B

	full_text


float %104
�load8B�
�
	full_text�
�
�%156 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 0, i64 11), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%157 = tail call float @llvm.fmuladd.f32(float %133, float %156, float %103)
*float8B

	full_text


float %133
*float8B

	full_text


float %156
*float8B

	full_text


float %103
�load8B�
�
	full_text�
�
�%158 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 0, i64 12), align 16, !tbaa !8
icall8B_
]
	full_textP
N
L%159 = tail call float @llvm.fmuladd.f32(float %133, float %158, float %102)
*float8B

	full_text


float %133
*float8B

	full_text


float %158
*float8B

	full_text


float %102
�load8B�
�
	full_text�
�
�%160 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 0, i64 13), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%161 = tail call float @llvm.fmuladd.f32(float %133, float %160, float %101)
*float8B

	full_text


float %133
*float8B

	full_text


float %160
*float8B

	full_text


float %101
�load8B�
�
	full_text�
�
�%162 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 0, i64 14), align 8, !tbaa !8
icall8B_
]
	full_textP
N
L%163 = tail call float @llvm.fmuladd.f32(float %133, float %162, float %100)
*float8B

	full_text


float %133
*float8B

	full_text


float %162
*float8B

	full_text


float %100
�load8B�
�
	full_text�
�
�%164 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 0, i64 15), align 4, !tbaa !8
hcall8B^
\
	full_textO
M
K%165 = tail call float @llvm.fmuladd.f32(float %133, float %164, float %99)
*float8B

	full_text


float %133
*float8B

	full_text


float %164
)float8B

	full_text

	float %99
Mload8BC
A
	full_text4
2
0%166 = load float, float* %94, align 4, !tbaa !8
+float*8B

	full_text


float* %94
�load8B�
�
	full_text�
�
�%167 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 1, i64 0), align 16, !tbaa !8
icall8B_
]
	full_textP
N
L%168 = tail call float @llvm.fmuladd.f32(float %166, float %167, float %135)
*float8B

	full_text


float %166
*float8B

	full_text


float %167
*float8B

	full_text


float %135
�load8B�
�
	full_text�
�
�%169 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 1, i64 1), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%170 = tail call float @llvm.fmuladd.f32(float %166, float %169, float %137)
*float8B

	full_text


float %166
*float8B

	full_text


float %169
*float8B

	full_text


float %137
�load8B�
�
	full_text�
�
�%171 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 1, i64 2), align 8, !tbaa !8
icall8B_
]
	full_textP
N
L%172 = tail call float @llvm.fmuladd.f32(float %166, float %171, float %139)
*float8B

	full_text


float %166
*float8B

	full_text


float %171
*float8B

	full_text


float %139
�load8B�
�
	full_text�
�
�%173 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 1, i64 3), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%174 = tail call float @llvm.fmuladd.f32(float %166, float %173, float %141)
*float8B

	full_text


float %166
*float8B

	full_text


float %173
*float8B

	full_text


float %141
�load8B�
�
	full_text�
�
�%175 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 1, i64 4), align 16, !tbaa !8
icall8B_
]
	full_textP
N
L%176 = tail call float @llvm.fmuladd.f32(float %166, float %175, float %143)
*float8B

	full_text


float %166
*float8B

	full_text


float %175
*float8B

	full_text


float %143
�load8B�
�
	full_text�
�
�%177 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 1, i64 5), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%178 = tail call float @llvm.fmuladd.f32(float %166, float %177, float %145)
*float8B

	full_text


float %166
*float8B

	full_text


float %177
*float8B

	full_text


float %145
�load8B�
�
	full_text�
�
�%179 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 1, i64 6), align 8, !tbaa !8
icall8B_
]
	full_textP
N
L%180 = tail call float @llvm.fmuladd.f32(float %166, float %179, float %147)
*float8B

	full_text


float %166
*float8B

	full_text


float %179
*float8B

	full_text


float %147
�load8B�
�
	full_text�
�
�%181 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 1, i64 7), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%182 = tail call float @llvm.fmuladd.f32(float %166, float %181, float %149)
*float8B

	full_text


float %166
*float8B

	full_text


float %181
*float8B

	full_text


float %149
�load8B�
�
	full_text�
�
�%183 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 1, i64 8), align 16, !tbaa !8
icall8B_
]
	full_textP
N
L%184 = tail call float @llvm.fmuladd.f32(float %166, float %183, float %151)
*float8B

	full_text


float %166
*float8B

	full_text


float %183
*float8B

	full_text


float %151
�load8B�
�
	full_text�
�
�%185 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 1, i64 9), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%186 = tail call float @llvm.fmuladd.f32(float %166, float %185, float %153)
*float8B

	full_text


float %166
*float8B

	full_text


float %185
*float8B

	full_text


float %153
�load8B�
�
	full_text�
�
�%187 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 1, i64 10), align 8, !tbaa !8
icall8B_
]
	full_textP
N
L%188 = tail call float @llvm.fmuladd.f32(float %166, float %187, float %155)
*float8B

	full_text


float %166
*float8B

	full_text


float %187
*float8B

	full_text


float %155
�load8B�
�
	full_text�
�
�%189 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 1, i64 11), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%190 = tail call float @llvm.fmuladd.f32(float %166, float %189, float %157)
*float8B

	full_text


float %166
*float8B

	full_text


float %189
*float8B

	full_text


float %157
�load8B�
�
	full_text�
�
�%191 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 1, i64 12), align 16, !tbaa !8
icall8B_
]
	full_textP
N
L%192 = tail call float @llvm.fmuladd.f32(float %166, float %191, float %159)
*float8B

	full_text


float %166
*float8B

	full_text


float %191
*float8B

	full_text


float %159
�load8B�
�
	full_text�
�
�%193 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 1, i64 13), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%194 = tail call float @llvm.fmuladd.f32(float %166, float %193, float %161)
*float8B

	full_text


float %166
*float8B

	full_text


float %193
*float8B

	full_text


float %161
�load8B�
�
	full_text�
�
�%195 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 1, i64 14), align 8, !tbaa !8
icall8B_
]
	full_textP
N
L%196 = tail call float @llvm.fmuladd.f32(float %166, float %195, float %163)
*float8B

	full_text


float %166
*float8B

	full_text


float %195
*float8B

	full_text


float %163
�load8B�
�
	full_text�
�
�%197 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 1, i64 15), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%198 = tail call float @llvm.fmuladd.f32(float %166, float %197, float %165)
*float8B

	full_text


float %166
*float8B

	full_text


float %197
*float8B

	full_text


float %165
Mload8BC
A
	full_text4
2
0%199 = load float, float* %95, align 8, !tbaa !8
+float*8B

	full_text


float* %95
�load8B�
�
	full_text�
�
�%200 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 2, i64 0), align 16, !tbaa !8
icall8B_
]
	full_textP
N
L%201 = tail call float @llvm.fmuladd.f32(float %199, float %200, float %168)
*float8B

	full_text


float %199
*float8B

	full_text


float %200
*float8B

	full_text


float %168
�load8B�
�
	full_text�
�
�%202 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 2, i64 1), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%203 = tail call float @llvm.fmuladd.f32(float %199, float %202, float %170)
*float8B

	full_text


float %199
*float8B

	full_text


float %202
*float8B

	full_text


float %170
�load8B�
�
	full_text�
�
�%204 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 2, i64 2), align 8, !tbaa !8
icall8B_
]
	full_textP
N
L%205 = tail call float @llvm.fmuladd.f32(float %199, float %204, float %172)
*float8B

	full_text


float %199
*float8B

	full_text


float %204
*float8B

	full_text


float %172
�load8B�
�
	full_text�
�
�%206 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 2, i64 3), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%207 = tail call float @llvm.fmuladd.f32(float %199, float %206, float %174)
*float8B

	full_text


float %199
*float8B

	full_text


float %206
*float8B

	full_text


float %174
�load8B�
�
	full_text�
�
�%208 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 2, i64 4), align 16, !tbaa !8
icall8B_
]
	full_textP
N
L%209 = tail call float @llvm.fmuladd.f32(float %199, float %208, float %176)
*float8B

	full_text


float %199
*float8B

	full_text


float %208
*float8B

	full_text


float %176
�load8B�
�
	full_text�
�
�%210 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 2, i64 5), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%211 = tail call float @llvm.fmuladd.f32(float %199, float %210, float %178)
*float8B

	full_text


float %199
*float8B

	full_text


float %210
*float8B

	full_text


float %178
�load8B�
�
	full_text�
�
�%212 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 2, i64 6), align 8, !tbaa !8
icall8B_
]
	full_textP
N
L%213 = tail call float @llvm.fmuladd.f32(float %199, float %212, float %180)
*float8B

	full_text


float %199
*float8B

	full_text


float %212
*float8B

	full_text


float %180
�load8B�
�
	full_text�
�
�%214 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 2, i64 7), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%215 = tail call float @llvm.fmuladd.f32(float %199, float %214, float %182)
*float8B

	full_text


float %199
*float8B

	full_text


float %214
*float8B

	full_text


float %182
�load8B�
�
	full_text�
�
�%216 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 2, i64 8), align 16, !tbaa !8
icall8B_
]
	full_textP
N
L%217 = tail call float @llvm.fmuladd.f32(float %199, float %216, float %184)
*float8B

	full_text


float %199
*float8B

	full_text


float %216
*float8B

	full_text


float %184
�load8B�
�
	full_text�
�
�%218 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 2, i64 9), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%219 = tail call float @llvm.fmuladd.f32(float %199, float %218, float %186)
*float8B

	full_text


float %199
*float8B

	full_text


float %218
*float8B

	full_text


float %186
�load8B�
�
	full_text�
�
�%220 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 2, i64 10), align 8, !tbaa !8
icall8B_
]
	full_textP
N
L%221 = tail call float @llvm.fmuladd.f32(float %199, float %220, float %188)
*float8B

	full_text


float %199
*float8B

	full_text


float %220
*float8B

	full_text


float %188
�load8B�
�
	full_text�
�
�%222 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 2, i64 11), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%223 = tail call float @llvm.fmuladd.f32(float %199, float %222, float %190)
*float8B

	full_text


float %199
*float8B

	full_text


float %222
*float8B

	full_text


float %190
�load8B�
�
	full_text�
�
�%224 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 2, i64 12), align 16, !tbaa !8
icall8B_
]
	full_textP
N
L%225 = tail call float @llvm.fmuladd.f32(float %199, float %224, float %192)
*float8B

	full_text


float %199
*float8B

	full_text


float %224
*float8B

	full_text


float %192
�load8B�
�
	full_text�
�
�%226 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 2, i64 13), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%227 = tail call float @llvm.fmuladd.f32(float %199, float %226, float %194)
*float8B

	full_text


float %199
*float8B

	full_text


float %226
*float8B

	full_text


float %194
�load8B�
�
	full_text�
�
�%228 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 2, i64 14), align 8, !tbaa !8
icall8B_
]
	full_textP
N
L%229 = tail call float @llvm.fmuladd.f32(float %199, float %228, float %196)
*float8B

	full_text


float %199
*float8B

	full_text


float %228
*float8B

	full_text


float %196
�load8B�
�
	full_text�
�
�%230 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 2, i64 15), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%231 = tail call float @llvm.fmuladd.f32(float %199, float %230, float %198)
*float8B

	full_text


float %199
*float8B

	full_text


float %230
*float8B

	full_text


float %198
Mload8BC
A
	full_text4
2
0%232 = load float, float* %96, align 4, !tbaa !8
+float*8B

	full_text


float* %96
�load8B�
�
	full_text�
�
�%233 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 3, i64 0), align 16, !tbaa !8
icall8B_
]
	full_textP
N
L%234 = tail call float @llvm.fmuladd.f32(float %232, float %233, float %201)
*float8B

	full_text


float %232
*float8B

	full_text


float %233
*float8B

	full_text


float %201
Nstore8BC
A
	full_text4
2
0store float %234, float* %78, align 16, !tbaa !8
*float8B

	full_text


float %234
+float*8B

	full_text


float* %78
�load8B�
�
	full_text�
�
�%235 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 3, i64 1), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%236 = tail call float @llvm.fmuladd.f32(float %232, float %235, float %203)
*float8B

	full_text


float %232
*float8B

	full_text


float %235
*float8B

	full_text


float %203
Mstore8BB
@
	full_text3
1
/store float %236, float* %79, align 4, !tbaa !8
*float8B

	full_text


float %236
+float*8B

	full_text


float* %79
�load8B�
�
	full_text�
�
�%237 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 3, i64 2), align 8, !tbaa !8
icall8B_
]
	full_textP
N
L%238 = tail call float @llvm.fmuladd.f32(float %232, float %237, float %205)
*float8B

	full_text


float %232
*float8B

	full_text


float %237
*float8B

	full_text


float %205
Mstore8BB
@
	full_text3
1
/store float %238, float* %80, align 8, !tbaa !8
*float8B

	full_text


float %238
+float*8B

	full_text


float* %80
�load8B�
�
	full_text�
�
�%239 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 3, i64 3), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%240 = tail call float @llvm.fmuladd.f32(float %232, float %239, float %207)
*float8B

	full_text


float %232
*float8B

	full_text


float %239
*float8B

	full_text


float %207
Mstore8BB
@
	full_text3
1
/store float %240, float* %81, align 4, !tbaa !8
*float8B

	full_text


float %240
+float*8B

	full_text


float* %81
�load8B�
�
	full_text�
�
�%241 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 3, i64 4), align 16, !tbaa !8
icall8B_
]
	full_textP
N
L%242 = tail call float @llvm.fmuladd.f32(float %232, float %241, float %209)
*float8B

	full_text


float %232
*float8B

	full_text


float %241
*float8B

	full_text


float %209
Nstore8BC
A
	full_text4
2
0store float %242, float* %82, align 16, !tbaa !8
*float8B

	full_text


float %242
+float*8B

	full_text


float* %82
�load8B�
�
	full_text�
�
�%243 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 3, i64 5), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%244 = tail call float @llvm.fmuladd.f32(float %232, float %243, float %211)
*float8B

	full_text


float %232
*float8B

	full_text


float %243
*float8B

	full_text


float %211
Mstore8BB
@
	full_text3
1
/store float %244, float* %83, align 4, !tbaa !8
*float8B

	full_text


float %244
+float*8B

	full_text


float* %83
�load8B�
�
	full_text�
�
�%245 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 3, i64 6), align 8, !tbaa !8
icall8B_
]
	full_textP
N
L%246 = tail call float @llvm.fmuladd.f32(float %232, float %245, float %213)
*float8B

	full_text


float %232
*float8B

	full_text


float %245
*float8B

	full_text


float %213
Mstore8BB
@
	full_text3
1
/store float %246, float* %84, align 8, !tbaa !8
*float8B

	full_text


float %246
+float*8B

	full_text


float* %84
�load8B�
�
	full_text�
�
�%247 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 3, i64 7), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%248 = tail call float @llvm.fmuladd.f32(float %232, float %247, float %215)
*float8B

	full_text


float %232
*float8B

	full_text


float %247
*float8B

	full_text


float %215
Mstore8BB
@
	full_text3
1
/store float %248, float* %85, align 4, !tbaa !8
*float8B

	full_text


float %248
+float*8B

	full_text


float* %85
�load8B�
�
	full_text�
�
�%249 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 3, i64 8), align 16, !tbaa !8
icall8B_
]
	full_textP
N
L%250 = tail call float @llvm.fmuladd.f32(float %232, float %249, float %217)
*float8B

	full_text


float %232
*float8B

	full_text


float %249
*float8B

	full_text


float %217
Nstore8BC
A
	full_text4
2
0store float %250, float* %86, align 16, !tbaa !8
*float8B

	full_text


float %250
+float*8B

	full_text


float* %86
�load8B�
�
	full_text�
�
�%251 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 3, i64 9), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%252 = tail call float @llvm.fmuladd.f32(float %232, float %251, float %219)
*float8B

	full_text


float %232
*float8B

	full_text


float %251
*float8B

	full_text


float %219
Mstore8BB
@
	full_text3
1
/store float %252, float* %87, align 4, !tbaa !8
*float8B

	full_text


float %252
+float*8B

	full_text


float* %87
�load8B�
�
	full_text�
�
�%253 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 3, i64 10), align 8, !tbaa !8
icall8B_
]
	full_textP
N
L%254 = tail call float @llvm.fmuladd.f32(float %232, float %253, float %221)
*float8B

	full_text


float %232
*float8B

	full_text


float %253
*float8B

	full_text


float %221
Mstore8BB
@
	full_text3
1
/store float %254, float* %88, align 8, !tbaa !8
*float8B

	full_text


float %254
+float*8B

	full_text


float* %88
�load8B�
�
	full_text�
�
�%255 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 3, i64 11), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%256 = tail call float @llvm.fmuladd.f32(float %232, float %255, float %223)
*float8B

	full_text


float %232
*float8B

	full_text


float %255
*float8B

	full_text


float %223
Mstore8BB
@
	full_text3
1
/store float %256, float* %89, align 4, !tbaa !8
*float8B

	full_text


float %256
+float*8B

	full_text


float* %89
�load8B�
�
	full_text�
�
�%257 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 3, i64 12), align 16, !tbaa !8
icall8B_
]
	full_textP
N
L%258 = tail call float @llvm.fmuladd.f32(float %232, float %257, float %225)
*float8B

	full_text


float %232
*float8B

	full_text


float %257
*float8B

	full_text


float %225
Nstore8BC
A
	full_text4
2
0store float %258, float* %90, align 16, !tbaa !8
*float8B

	full_text


float %258
+float*8B

	full_text


float* %90
�load8B�
�
	full_text�
�
�%259 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 3, i64 13), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%260 = tail call float @llvm.fmuladd.f32(float %232, float %259, float %227)
*float8B

	full_text


float %232
*float8B

	full_text


float %259
*float8B

	full_text


float %227
Mstore8BB
@
	full_text3
1
/store float %260, float* %91, align 4, !tbaa !8
*float8B

	full_text


float %260
+float*8B

	full_text


float* %91
�load8B�
�
	full_text�
�
�%261 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 3, i64 14), align 8, !tbaa !8
icall8B_
]
	full_textP
N
L%262 = tail call float @llvm.fmuladd.f32(float %232, float %261, float %229)
*float8B

	full_text


float %232
*float8B

	full_text


float %261
*float8B

	full_text


float %229
Mstore8BB
@
	full_text3
1
/store float %262, float* %92, align 8, !tbaa !8
*float8B

	full_text


float %262
+float*8B

	full_text


float* %92
�load8B�
�
	full_text�
�
�%263 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 3, i64 15), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%264 = tail call float @llvm.fmuladd.f32(float %232, float %263, float %231)
*float8B

	full_text


float %232
*float8B

	full_text


float %263
*float8B

	full_text


float %231
Mstore8BB
@
	full_text3
1
/store float %264, float* %93, align 4, !tbaa !8
*float8B

	full_text


float %264
+float*8B

	full_text


float* %93
8add8B/
-
	full_text 

%265 = add nsw i32 %117, %64
&i328B

	full_text


i32 %117
%i328B

	full_text
	
i32 %64
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #7
Zcall8BP
N
	full_textA
?
=call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %60) #6
%i8*8B

	full_text
	
i8* %60
:icmp8B0
.
	full_text!

%266 = icmp slt i32 %265, %97
&i328B

	full_text


i32 %265
%i328B

	full_text
	
i32 %97
<br8B4
2
	full_text%
#
!br i1 %266, label %98, label %267
$i18B

	full_text
	
i1 %266
Mstore8BB
@
	full_text3
1
/store float %132, float* %71, align 4, !tbaa !8
*float8B

	full_text


float %132
+float*8B

	full_text


float* %71
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #7
Nload8BD
B
	full_text5
3
1%268 = load float, float* %40, align 16, !tbaa !8
+float*8B

	full_text


float* %40
�load8B�
�
	full_text�
�
�%269 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 0, i64 0), align 16, !tbaa !8
icall8B_
]
	full_textP
N
L%270 = tail call float @llvm.fmuladd.f32(float %268, float %269, float %234)
*float8B

	full_text


float %268
*float8B

	full_text


float %269
*float8B

	full_text


float %234
�load8B�
�
	full_text�
�
�%271 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 0, i64 1), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%272 = tail call float @llvm.fmuladd.f32(float %268, float %271, float %236)
*float8B

	full_text


float %268
*float8B

	full_text


float %271
*float8B

	full_text


float %236
�load8B�
�
	full_text�
�
�%273 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 0, i64 2), align 8, !tbaa !8
icall8B_
]
	full_textP
N
L%274 = tail call float @llvm.fmuladd.f32(float %268, float %273, float %238)
*float8B

	full_text


float %268
*float8B

	full_text


float %273
*float8B

	full_text


float %238
�load8B�
�
	full_text�
�
�%275 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 0, i64 3), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%276 = tail call float @llvm.fmuladd.f32(float %268, float %275, float %240)
*float8B

	full_text


float %268
*float8B

	full_text


float %275
*float8B

	full_text


float %240
�load8B�
�
	full_text�
�
�%277 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 0, i64 4), align 16, !tbaa !8
icall8B_
]
	full_textP
N
L%278 = tail call float @llvm.fmuladd.f32(float %268, float %277, float %242)
*float8B

	full_text


float %268
*float8B

	full_text


float %277
*float8B

	full_text


float %242
�load8B�
�
	full_text�
�
�%279 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 0, i64 5), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%280 = tail call float @llvm.fmuladd.f32(float %268, float %279, float %244)
*float8B

	full_text


float %268
*float8B

	full_text


float %279
*float8B

	full_text


float %244
�load8B�
�
	full_text�
�
�%281 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 0, i64 6), align 8, !tbaa !8
icall8B_
]
	full_textP
N
L%282 = tail call float @llvm.fmuladd.f32(float %268, float %281, float %246)
*float8B

	full_text


float %268
*float8B

	full_text


float %281
*float8B

	full_text


float %246
�load8B�
�
	full_text�
�
�%283 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 0, i64 7), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%284 = tail call float @llvm.fmuladd.f32(float %268, float %283, float %248)
*float8B

	full_text


float %268
*float8B

	full_text


float %283
*float8B

	full_text


float %248
�load8B�
�
	full_text�
�
�%285 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 0, i64 8), align 16, !tbaa !8
icall8B_
]
	full_textP
N
L%286 = tail call float @llvm.fmuladd.f32(float %268, float %285, float %250)
*float8B

	full_text


float %268
*float8B

	full_text


float %285
*float8B

	full_text


float %250
�load8B�
�
	full_text�
�
�%287 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 0, i64 9), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%288 = tail call float @llvm.fmuladd.f32(float %268, float %287, float %252)
*float8B

	full_text


float %268
*float8B

	full_text


float %287
*float8B

	full_text


float %252
�load8B�
�
	full_text�
�
�%289 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 0, i64 10), align 8, !tbaa !8
icall8B_
]
	full_textP
N
L%290 = tail call float @llvm.fmuladd.f32(float %268, float %289, float %254)
*float8B

	full_text


float %268
*float8B

	full_text


float %289
*float8B

	full_text


float %254
�load8B�
�
	full_text�
�
�%291 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 0, i64 11), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%292 = tail call float @llvm.fmuladd.f32(float %268, float %291, float %256)
*float8B

	full_text


float %268
*float8B

	full_text


float %291
*float8B

	full_text


float %256
�load8B�
�
	full_text�
�
�%293 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 0, i64 12), align 16, !tbaa !8
icall8B_
]
	full_textP
N
L%294 = tail call float @llvm.fmuladd.f32(float %268, float %293, float %258)
*float8B

	full_text


float %268
*float8B

	full_text


float %293
*float8B

	full_text


float %258
�load8B�
�
	full_text�
�
�%295 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 0, i64 13), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%296 = tail call float @llvm.fmuladd.f32(float %268, float %295, float %260)
*float8B

	full_text


float %268
*float8B

	full_text


float %295
*float8B

	full_text


float %260
�load8B�
�
	full_text�
�
�%297 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 0, i64 14), align 8, !tbaa !8
icall8B_
]
	full_textP
N
L%298 = tail call float @llvm.fmuladd.f32(float %268, float %297, float %262)
*float8B

	full_text


float %268
*float8B

	full_text


float %297
*float8B

	full_text


float %262
�load8B�
�
	full_text�
�
�%299 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 0, i64 15), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%300 = tail call float @llvm.fmuladd.f32(float %268, float %299, float %264)
*float8B

	full_text


float %268
*float8B

	full_text


float %299
*float8B

	full_text


float %264
Mload8BC
A
	full_text4
2
0%301 = load float, float* %45, align 4, !tbaa !8
+float*8B

	full_text


float* %45
�load8B�
�
	full_text�
�
�%302 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 1, i64 0), align 16, !tbaa !8
icall8B_
]
	full_textP
N
L%303 = tail call float @llvm.fmuladd.f32(float %301, float %302, float %270)
*float8B

	full_text


float %301
*float8B

	full_text


float %302
*float8B

	full_text


float %270
�load8B�
�
	full_text�
�
�%304 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 1, i64 1), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%305 = tail call float @llvm.fmuladd.f32(float %301, float %304, float %272)
*float8B

	full_text


float %301
*float8B

	full_text


float %304
*float8B

	full_text


float %272
�load8B�
�
	full_text�
�
�%306 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 1, i64 2), align 8, !tbaa !8
icall8B_
]
	full_textP
N
L%307 = tail call float @llvm.fmuladd.f32(float %301, float %306, float %274)
*float8B

	full_text


float %301
*float8B

	full_text


float %306
*float8B

	full_text


float %274
�load8B�
�
	full_text�
�
�%308 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 1, i64 3), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%309 = tail call float @llvm.fmuladd.f32(float %301, float %308, float %276)
*float8B

	full_text


float %301
*float8B

	full_text


float %308
*float8B

	full_text


float %276
�load8B�
�
	full_text�
�
�%310 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 1, i64 4), align 16, !tbaa !8
icall8B_
]
	full_textP
N
L%311 = tail call float @llvm.fmuladd.f32(float %301, float %310, float %278)
*float8B

	full_text


float %301
*float8B

	full_text


float %310
*float8B

	full_text


float %278
�load8B�
�
	full_text�
�
�%312 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 1, i64 5), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%313 = tail call float @llvm.fmuladd.f32(float %301, float %312, float %280)
*float8B

	full_text


float %301
*float8B

	full_text


float %312
*float8B

	full_text


float %280
�load8B�
�
	full_text�
�
�%314 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 1, i64 6), align 8, !tbaa !8
icall8B_
]
	full_textP
N
L%315 = tail call float @llvm.fmuladd.f32(float %301, float %314, float %282)
*float8B

	full_text


float %301
*float8B

	full_text


float %314
*float8B

	full_text


float %282
�load8B�
�
	full_text�
�
�%316 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 1, i64 7), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%317 = tail call float @llvm.fmuladd.f32(float %301, float %316, float %284)
*float8B

	full_text


float %301
*float8B

	full_text


float %316
*float8B

	full_text


float %284
�load8B�
�
	full_text�
�
�%318 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 1, i64 8), align 16, !tbaa !8
icall8B_
]
	full_textP
N
L%319 = tail call float @llvm.fmuladd.f32(float %301, float %318, float %286)
*float8B

	full_text


float %301
*float8B

	full_text


float %318
*float8B

	full_text


float %286
�load8B�
�
	full_text�
�
�%320 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 1, i64 9), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%321 = tail call float @llvm.fmuladd.f32(float %301, float %320, float %288)
*float8B

	full_text


float %301
*float8B

	full_text


float %320
*float8B

	full_text


float %288
�load8B�
�
	full_text�
�
�%322 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 1, i64 10), align 8, !tbaa !8
icall8B_
]
	full_textP
N
L%323 = tail call float @llvm.fmuladd.f32(float %301, float %322, float %290)
*float8B

	full_text


float %301
*float8B

	full_text


float %322
*float8B

	full_text


float %290
�load8B�
�
	full_text�
�
�%324 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 1, i64 11), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%325 = tail call float @llvm.fmuladd.f32(float %301, float %324, float %292)
*float8B

	full_text


float %301
*float8B

	full_text


float %324
*float8B

	full_text


float %292
�load8B�
�
	full_text�
�
�%326 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 1, i64 12), align 16, !tbaa !8
icall8B_
]
	full_textP
N
L%327 = tail call float @llvm.fmuladd.f32(float %301, float %326, float %294)
*float8B

	full_text


float %301
*float8B

	full_text


float %326
*float8B

	full_text


float %294
�load8B�
�
	full_text�
�
�%328 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 1, i64 13), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%329 = tail call float @llvm.fmuladd.f32(float %301, float %328, float %296)
*float8B

	full_text


float %301
*float8B

	full_text


float %328
*float8B

	full_text


float %296
�load8B�
�
	full_text�
�
�%330 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 1, i64 14), align 8, !tbaa !8
icall8B_
]
	full_textP
N
L%331 = tail call float @llvm.fmuladd.f32(float %301, float %330, float %298)
*float8B

	full_text


float %301
*float8B

	full_text


float %330
*float8B

	full_text


float %298
�load8B�
�
	full_text�
�
�%332 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 1, i64 15), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%333 = tail call float @llvm.fmuladd.f32(float %301, float %332, float %300)
*float8B

	full_text


float %301
*float8B

	full_text


float %332
*float8B

	full_text


float %300
Mload8BC
A
	full_text4
2
0%334 = load float, float* %51, align 8, !tbaa !8
+float*8B

	full_text


float* %51
�load8B�
�
	full_text�
�
�%335 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 2, i64 0), align 16, !tbaa !8
icall8B_
]
	full_textP
N
L%336 = tail call float @llvm.fmuladd.f32(float %334, float %335, float %303)
*float8B

	full_text


float %334
*float8B

	full_text


float %335
*float8B

	full_text


float %303
�load8B�
�
	full_text�
�
�%337 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 2, i64 1), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%338 = tail call float @llvm.fmuladd.f32(float %334, float %337, float %305)
*float8B

	full_text


float %334
*float8B

	full_text


float %337
*float8B

	full_text


float %305
�load8B�
�
	full_text�
�
�%339 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 2, i64 2), align 8, !tbaa !8
icall8B_
]
	full_textP
N
L%340 = tail call float @llvm.fmuladd.f32(float %334, float %339, float %307)
*float8B

	full_text


float %334
*float8B

	full_text


float %339
*float8B

	full_text


float %307
�load8B�
�
	full_text�
�
�%341 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 2, i64 3), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%342 = tail call float @llvm.fmuladd.f32(float %334, float %341, float %309)
*float8B

	full_text


float %334
*float8B

	full_text


float %341
*float8B

	full_text


float %309
�load8B�
�
	full_text�
�
�%343 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 2, i64 4), align 16, !tbaa !8
icall8B_
]
	full_textP
N
L%344 = tail call float @llvm.fmuladd.f32(float %334, float %343, float %311)
*float8B

	full_text


float %334
*float8B

	full_text


float %343
*float8B

	full_text


float %311
�load8B�
�
	full_text�
�
�%345 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 2, i64 5), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%346 = tail call float @llvm.fmuladd.f32(float %334, float %345, float %313)
*float8B

	full_text


float %334
*float8B

	full_text


float %345
*float8B

	full_text


float %313
�load8B�
�
	full_text�
�
�%347 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 2, i64 6), align 8, !tbaa !8
icall8B_
]
	full_textP
N
L%348 = tail call float @llvm.fmuladd.f32(float %334, float %347, float %315)
*float8B

	full_text


float %334
*float8B

	full_text


float %347
*float8B

	full_text


float %315
�load8B�
�
	full_text�
�
�%349 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 2, i64 7), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%350 = tail call float @llvm.fmuladd.f32(float %334, float %349, float %317)
*float8B

	full_text


float %334
*float8B

	full_text


float %349
*float8B

	full_text


float %317
�load8B�
�
	full_text�
�
�%351 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 2, i64 8), align 16, !tbaa !8
icall8B_
]
	full_textP
N
L%352 = tail call float @llvm.fmuladd.f32(float %334, float %351, float %319)
*float8B

	full_text


float %334
*float8B

	full_text


float %351
*float8B

	full_text


float %319
�load8B�
�
	full_text�
�
�%353 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 2, i64 9), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%354 = tail call float @llvm.fmuladd.f32(float %334, float %353, float %321)
*float8B

	full_text


float %334
*float8B

	full_text


float %353
*float8B

	full_text


float %321
�load8B�
�
	full_text�
�
�%355 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 2, i64 10), align 8, !tbaa !8
icall8B_
]
	full_textP
N
L%356 = tail call float @llvm.fmuladd.f32(float %334, float %355, float %323)
*float8B

	full_text


float %334
*float8B

	full_text


float %355
*float8B

	full_text


float %323
�load8B�
�
	full_text�
�
�%357 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 2, i64 11), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%358 = tail call float @llvm.fmuladd.f32(float %334, float %357, float %325)
*float8B

	full_text


float %334
*float8B

	full_text


float %357
*float8B

	full_text


float %325
�load8B�
�
	full_text�
�
�%359 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 2, i64 12), align 16, !tbaa !8
icall8B_
]
	full_textP
N
L%360 = tail call float @llvm.fmuladd.f32(float %334, float %359, float %327)
*float8B

	full_text


float %334
*float8B

	full_text


float %359
*float8B

	full_text


float %327
�load8B�
�
	full_text�
�
�%361 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 2, i64 13), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%362 = tail call float @llvm.fmuladd.f32(float %334, float %361, float %329)
*float8B

	full_text


float %334
*float8B

	full_text


float %361
*float8B

	full_text


float %329
�load8B�
�
	full_text�
�
�%363 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 2, i64 14), align 8, !tbaa !8
icall8B_
]
	full_textP
N
L%364 = tail call float @llvm.fmuladd.f32(float %334, float %363, float %331)
*float8B

	full_text


float %334
*float8B

	full_text


float %363
*float8B

	full_text


float %331
�load8B�
�
	full_text�
�
�%365 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 2, i64 15), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%366 = tail call float @llvm.fmuladd.f32(float %334, float %365, float %333)
*float8B

	full_text


float %334
*float8B

	full_text


float %365
*float8B

	full_text


float %333
Mload8BC
A
	full_text4
2
0%367 = load float, float* %57, align 4, !tbaa !8
+float*8B

	full_text


float* %57
�load8B�
�
	full_text�
�
�%368 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 3, i64 0), align 16, !tbaa !8
icall8B_
]
	full_textP
N
L%369 = tail call float @llvm.fmuladd.f32(float %367, float %368, float %336)
*float8B

	full_text


float %367
*float8B

	full_text


float %368
*float8B

	full_text


float %336
Nstore8BC
A
	full_text4
2
0store float %369, float* %78, align 16, !tbaa !8
*float8B

	full_text


float %369
+float*8B

	full_text


float* %78
�load8B�
�
	full_text�
�
�%370 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 3, i64 1), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%371 = tail call float @llvm.fmuladd.f32(float %367, float %370, float %338)
*float8B

	full_text


float %367
*float8B

	full_text


float %370
*float8B

	full_text


float %338
Mstore8BB
@
	full_text3
1
/store float %371, float* %79, align 4, !tbaa !8
*float8B

	full_text


float %371
+float*8B

	full_text


float* %79
�load8B�
�
	full_text�
�
�%372 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 3, i64 2), align 8, !tbaa !8
icall8B_
]
	full_textP
N
L%373 = tail call float @llvm.fmuladd.f32(float %367, float %372, float %340)
*float8B

	full_text


float %367
*float8B

	full_text


float %372
*float8B

	full_text


float %340
Mstore8BB
@
	full_text3
1
/store float %373, float* %80, align 8, !tbaa !8
*float8B

	full_text


float %373
+float*8B

	full_text


float* %80
�load8B�
�
	full_text�
�
�%374 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 3, i64 3), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%375 = tail call float @llvm.fmuladd.f32(float %367, float %374, float %342)
*float8B

	full_text


float %367
*float8B

	full_text


float %374
*float8B

	full_text


float %342
Mstore8BB
@
	full_text3
1
/store float %375, float* %81, align 4, !tbaa !8
*float8B

	full_text


float %375
+float*8B

	full_text


float* %81
�load8B�
�
	full_text�
�
�%376 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 3, i64 4), align 16, !tbaa !8
icall8B_
]
	full_textP
N
L%377 = tail call float @llvm.fmuladd.f32(float %367, float %376, float %344)
*float8B

	full_text


float %367
*float8B

	full_text


float %376
*float8B

	full_text


float %344
Nstore8BC
A
	full_text4
2
0store float %377, float* %82, align 16, !tbaa !8
*float8B

	full_text


float %377
+float*8B

	full_text


float* %82
�load8B�
�
	full_text�
�
�%378 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 3, i64 5), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%379 = tail call float @llvm.fmuladd.f32(float %367, float %378, float %346)
*float8B

	full_text


float %367
*float8B

	full_text


float %378
*float8B

	full_text


float %346
Mstore8BB
@
	full_text3
1
/store float %379, float* %83, align 4, !tbaa !8
*float8B

	full_text


float %379
+float*8B

	full_text


float* %83
�load8B�
�
	full_text�
�
�%380 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 3, i64 6), align 8, !tbaa !8
icall8B_
]
	full_textP
N
L%381 = tail call float @llvm.fmuladd.f32(float %367, float %380, float %348)
*float8B

	full_text


float %367
*float8B

	full_text


float %380
*float8B

	full_text


float %348
Mstore8BB
@
	full_text3
1
/store float %381, float* %84, align 8, !tbaa !8
*float8B

	full_text


float %381
+float*8B

	full_text


float* %84
�load8B�
�
	full_text�
�
�%382 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 3, i64 7), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%383 = tail call float @llvm.fmuladd.f32(float %367, float %382, float %350)
*float8B

	full_text


float %367
*float8B

	full_text


float %382
*float8B

	full_text


float %350
Mstore8BB
@
	full_text3
1
/store float %383, float* %85, align 4, !tbaa !8
*float8B

	full_text


float %383
+float*8B

	full_text


float* %85
�load8B�
�
	full_text�
�
�%384 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 3, i64 8), align 16, !tbaa !8
icall8B_
]
	full_textP
N
L%385 = tail call float @llvm.fmuladd.f32(float %367, float %384, float %352)
*float8B

	full_text


float %367
*float8B

	full_text


float %384
*float8B

	full_text


float %352
Nstore8BC
A
	full_text4
2
0store float %385, float* %86, align 16, !tbaa !8
*float8B

	full_text


float %385
+float*8B

	full_text


float* %86
�load8B�
�
	full_text�
�
�%386 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 3, i64 9), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%387 = tail call float @llvm.fmuladd.f32(float %367, float %386, float %354)
*float8B

	full_text


float %367
*float8B

	full_text


float %386
*float8B

	full_text


float %354
Mstore8BB
@
	full_text3
1
/store float %387, float* %87, align 4, !tbaa !8
*float8B

	full_text


float %387
+float*8B

	full_text


float* %87
�load8B�
�
	full_text�
�
�%388 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 3, i64 10), align 8, !tbaa !8
icall8B_
]
	full_textP
N
L%389 = tail call float @llvm.fmuladd.f32(float %367, float %388, float %356)
*float8B

	full_text


float %367
*float8B

	full_text


float %388
*float8B

	full_text


float %356
Mstore8BB
@
	full_text3
1
/store float %389, float* %88, align 8, !tbaa !8
*float8B

	full_text


float %389
+float*8B

	full_text


float* %88
�load8B�
�
	full_text�
�
�%390 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 3, i64 11), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%391 = tail call float @llvm.fmuladd.f32(float %367, float %390, float %358)
*float8B

	full_text


float %367
*float8B

	full_text


float %390
*float8B

	full_text


float %358
Mstore8BB
@
	full_text3
1
/store float %391, float* %89, align 4, !tbaa !8
*float8B

	full_text


float %391
+float*8B

	full_text


float* %89
�load8B�
�
	full_text�
�
�%392 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 3, i64 12), align 16, !tbaa !8
icall8B_
]
	full_textP
N
L%393 = tail call float @llvm.fmuladd.f32(float %367, float %392, float %360)
*float8B

	full_text


float %367
*float8B

	full_text


float %392
*float8B

	full_text


float %360
Nstore8BC
A
	full_text4
2
0store float %393, float* %90, align 16, !tbaa !8
*float8B

	full_text


float %393
+float*8B

	full_text


float* %90
�load8B�
�
	full_text�
�
�%394 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 3, i64 13), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%395 = tail call float @llvm.fmuladd.f32(float %367, float %394, float %362)
*float8B

	full_text


float %367
*float8B

	full_text


float %394
*float8B

	full_text


float %362
Mstore8BB
@
	full_text3
1
/store float %395, float* %91, align 4, !tbaa !8
*float8B

	full_text


float %395
+float*8B

	full_text


float* %91
�load8B�
�
	full_text�
�
�%396 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 3, i64 14), align 8, !tbaa !8
icall8B_
]
	full_textP
N
L%397 = tail call float @llvm.fmuladd.f32(float %367, float %396, float %364)
*float8B

	full_text


float %367
*float8B

	full_text


float %396
*float8B

	full_text


float %364
Mstore8BB
@
	full_text3
1
/store float %397, float* %92, align 8, !tbaa !8
*float8B

	full_text


float %397
+float*8B

	full_text


float* %92
�load8B�
�
	full_text�
�
�%398 = load float, float* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 3, i64 15), align 4, !tbaa !8
icall8B_
]
	full_textP
N
L%399 = tail call float @llvm.fmuladd.f32(float %367, float %398, float %366)
*float8B

	full_text


float %367
*float8B

	full_text


float %398
*float8B

	full_text


float %366
Mstore8BB
@
	full_text3
1
/store float %399, float* %93, align 4, !tbaa !8
*float8B

	full_text


float %399
+float*8B

	full_text


float* %93
6sext8B,
*
	full_text

%400 = sext i32 %5 to i64
Mload8BC
A
	full_text4
2
0%401 = load float, float* %36, align 4, !tbaa !8
+float*8B

	full_text


float* %36
7fmul8B-
+
	full_text

%402 = fmul float %401, %8
*float8B

	full_text


float %401
gcall8B]
[
	full_textN
L
J%403 = tail call float @llvm.fmuladd.f32(float %7, float %369, float %402)
*float8B

	full_text


float %369
*float8B

	full_text


float %402
Mstore8BB
@
	full_text3
1
/store float %403, float* %36, align 4, !tbaa !8
*float8B

	full_text


float %403
+float*8B

	full_text


float* %36
_getelementptr8BL
J
	full_text=
;
9%404 = getelementptr inbounds float, float* %36, i64 %400
+float*8B

	full_text


float* %36
&i648B

	full_text


i64 %400
Nload8BD
B
	full_text5
3
1%405 = load float, float* %404, align 4, !tbaa !8
,float*8B

	full_text

float* %404
7fmul8B-
+
	full_text

%406 = fmul float %405, %8
*float8B

	full_text


float %405
gcall8B]
[
	full_textN
L
J%407 = tail call float @llvm.fmuladd.f32(float %7, float %371, float %406)
*float8B

	full_text


float %371
*float8B

	full_text


float %406
Nstore8BC
A
	full_text4
2
0store float %407, float* %404, align 4, !tbaa !8
*float8B

	full_text


float %407
,float*8B

	full_text

float* %404
`getelementptr8BM
K
	full_text>
<
:%408 = getelementptr inbounds float, float* %404, i64 %400
,float*8B

	full_text

float* %404
&i648B

	full_text


i64 %400
Nload8BD
B
	full_text5
3
1%409 = load float, float* %408, align 4, !tbaa !8
,float*8B

	full_text

float* %408
7fmul8B-
+
	full_text

%410 = fmul float %409, %8
*float8B

	full_text


float %409
gcall8B]
[
	full_textN
L
J%411 = tail call float @llvm.fmuladd.f32(float %7, float %373, float %410)
*float8B

	full_text


float %373
*float8B

	full_text


float %410
Nstore8BC
A
	full_text4
2
0store float %411, float* %408, align 4, !tbaa !8
*float8B

	full_text


float %411
,float*8B

	full_text

float* %408
`getelementptr8BM
K
	full_text>
<
:%412 = getelementptr inbounds float, float* %408, i64 %400
,float*8B

	full_text

float* %408
&i648B

	full_text


i64 %400
Nload8BD
B
	full_text5
3
1%413 = load float, float* %412, align 4, !tbaa !8
,float*8B

	full_text

float* %412
7fmul8B-
+
	full_text

%414 = fmul float %413, %8
*float8B

	full_text


float %413
gcall8B]
[
	full_textN
L
J%415 = tail call float @llvm.fmuladd.f32(float %7, float %375, float %414)
*float8B

	full_text


float %375
*float8B

	full_text


float %414
Nstore8BC
A
	full_text4
2
0store float %415, float* %412, align 4, !tbaa !8
*float8B

	full_text


float %415
,float*8B

	full_text

float* %412
`getelementptr8BM
K
	full_text>
<
:%416 = getelementptr inbounds float, float* %412, i64 %400
,float*8B

	full_text

float* %412
&i648B

	full_text


i64 %400
Nload8BD
B
	full_text5
3
1%417 = load float, float* %416, align 4, !tbaa !8
,float*8B

	full_text

float* %416
7fmul8B-
+
	full_text

%418 = fmul float %417, %8
*float8B

	full_text


float %417
gcall8B]
[
	full_textN
L
J%419 = tail call float @llvm.fmuladd.f32(float %7, float %377, float %418)
*float8B

	full_text


float %377
*float8B

	full_text


float %418
Nstore8BC
A
	full_text4
2
0store float %419, float* %416, align 4, !tbaa !8
*float8B

	full_text


float %419
,float*8B

	full_text

float* %416
`getelementptr8BM
K
	full_text>
<
:%420 = getelementptr inbounds float, float* %416, i64 %400
,float*8B

	full_text

float* %416
&i648B

	full_text


i64 %400
Nload8BD
B
	full_text5
3
1%421 = load float, float* %420, align 4, !tbaa !8
,float*8B

	full_text

float* %420
7fmul8B-
+
	full_text

%422 = fmul float %421, %8
*float8B

	full_text


float %421
gcall8B]
[
	full_textN
L
J%423 = tail call float @llvm.fmuladd.f32(float %7, float %379, float %422)
*float8B

	full_text


float %379
*float8B

	full_text


float %422
Nstore8BC
A
	full_text4
2
0store float %423, float* %420, align 4, !tbaa !8
*float8B

	full_text


float %423
,float*8B

	full_text

float* %420
`getelementptr8BM
K
	full_text>
<
:%424 = getelementptr inbounds float, float* %420, i64 %400
,float*8B

	full_text

float* %420
&i648B

	full_text


i64 %400
Nload8BD
B
	full_text5
3
1%425 = load float, float* %424, align 4, !tbaa !8
,float*8B

	full_text

float* %424
7fmul8B-
+
	full_text

%426 = fmul float %425, %8
*float8B

	full_text


float %425
gcall8B]
[
	full_textN
L
J%427 = tail call float @llvm.fmuladd.f32(float %7, float %381, float %426)
*float8B

	full_text


float %381
*float8B

	full_text


float %426
Nstore8BC
A
	full_text4
2
0store float %427, float* %424, align 4, !tbaa !8
*float8B

	full_text


float %427
,float*8B

	full_text

float* %424
`getelementptr8BM
K
	full_text>
<
:%428 = getelementptr inbounds float, float* %424, i64 %400
,float*8B

	full_text

float* %424
&i648B

	full_text


i64 %400
Nload8BD
B
	full_text5
3
1%429 = load float, float* %428, align 4, !tbaa !8
,float*8B

	full_text

float* %428
7fmul8B-
+
	full_text

%430 = fmul float %429, %8
*float8B

	full_text


float %429
gcall8B]
[
	full_textN
L
J%431 = tail call float @llvm.fmuladd.f32(float %7, float %383, float %430)
*float8B

	full_text


float %383
*float8B

	full_text


float %430
Nstore8BC
A
	full_text4
2
0store float %431, float* %428, align 4, !tbaa !8
*float8B

	full_text


float %431
,float*8B

	full_text

float* %428
`getelementptr8BM
K
	full_text>
<
:%432 = getelementptr inbounds float, float* %428, i64 %400
,float*8B

	full_text

float* %428
&i648B

	full_text


i64 %400
Nload8BD
B
	full_text5
3
1%433 = load float, float* %432, align 4, !tbaa !8
,float*8B

	full_text

float* %432
7fmul8B-
+
	full_text

%434 = fmul float %433, %8
*float8B

	full_text


float %433
gcall8B]
[
	full_textN
L
J%435 = tail call float @llvm.fmuladd.f32(float %7, float %385, float %434)
*float8B

	full_text


float %385
*float8B

	full_text


float %434
Nstore8BC
A
	full_text4
2
0store float %435, float* %432, align 4, !tbaa !8
*float8B

	full_text


float %435
,float*8B

	full_text

float* %432
`getelementptr8BM
K
	full_text>
<
:%436 = getelementptr inbounds float, float* %432, i64 %400
,float*8B

	full_text

float* %432
&i648B

	full_text


i64 %400
Nload8BD
B
	full_text5
3
1%437 = load float, float* %436, align 4, !tbaa !8
,float*8B

	full_text

float* %436
7fmul8B-
+
	full_text

%438 = fmul float %437, %8
*float8B

	full_text


float %437
gcall8B]
[
	full_textN
L
J%439 = tail call float @llvm.fmuladd.f32(float %7, float %387, float %438)
*float8B

	full_text


float %387
*float8B

	full_text


float %438
Nstore8BC
A
	full_text4
2
0store float %439, float* %436, align 4, !tbaa !8
*float8B

	full_text


float %439
,float*8B

	full_text

float* %436
`getelementptr8BM
K
	full_text>
<
:%440 = getelementptr inbounds float, float* %436, i64 %400
,float*8B

	full_text

float* %436
&i648B

	full_text


i64 %400
Nload8BD
B
	full_text5
3
1%441 = load float, float* %440, align 4, !tbaa !8
,float*8B

	full_text

float* %440
7fmul8B-
+
	full_text

%442 = fmul float %441, %8
*float8B

	full_text


float %441
gcall8B]
[
	full_textN
L
J%443 = tail call float @llvm.fmuladd.f32(float %7, float %389, float %442)
*float8B

	full_text


float %389
*float8B

	full_text


float %442
Nstore8BC
A
	full_text4
2
0store float %443, float* %440, align 4, !tbaa !8
*float8B

	full_text


float %443
,float*8B

	full_text

float* %440
`getelementptr8BM
K
	full_text>
<
:%444 = getelementptr inbounds float, float* %440, i64 %400
,float*8B

	full_text

float* %440
&i648B

	full_text


i64 %400
Nload8BD
B
	full_text5
3
1%445 = load float, float* %444, align 4, !tbaa !8
,float*8B

	full_text

float* %444
7fmul8B-
+
	full_text

%446 = fmul float %445, %8
*float8B

	full_text


float %445
gcall8B]
[
	full_textN
L
J%447 = tail call float @llvm.fmuladd.f32(float %7, float %391, float %446)
*float8B

	full_text


float %391
*float8B

	full_text


float %446
Nstore8BC
A
	full_text4
2
0store float %447, float* %444, align 4, !tbaa !8
*float8B

	full_text


float %447
,float*8B

	full_text

float* %444
`getelementptr8BM
K
	full_text>
<
:%448 = getelementptr inbounds float, float* %444, i64 %400
,float*8B

	full_text

float* %444
&i648B

	full_text


i64 %400
Nload8BD
B
	full_text5
3
1%449 = load float, float* %448, align 4, !tbaa !8
,float*8B

	full_text

float* %448
7fmul8B-
+
	full_text

%450 = fmul float %449, %8
*float8B

	full_text


float %449
gcall8B]
[
	full_textN
L
J%451 = tail call float @llvm.fmuladd.f32(float %7, float %393, float %450)
*float8B

	full_text


float %393
*float8B

	full_text


float %450
Nstore8BC
A
	full_text4
2
0store float %451, float* %448, align 4, !tbaa !8
*float8B

	full_text


float %451
,float*8B

	full_text

float* %448
`getelementptr8BM
K
	full_text>
<
:%452 = getelementptr inbounds float, float* %448, i64 %400
,float*8B

	full_text

float* %448
&i648B

	full_text


i64 %400
Mload8BC
A
	full_text4
2
0%453 = load float, float* %91, align 4, !tbaa !8
+float*8B

	full_text


float* %91
Nload8BD
B
	full_text5
3
1%454 = load float, float* %452, align 4, !tbaa !8
,float*8B

	full_text

float* %452
7fmul8B-
+
	full_text

%455 = fmul float %454, %8
*float8B

	full_text


float %454
gcall8B]
[
	full_textN
L
J%456 = tail call float @llvm.fmuladd.f32(float %7, float %453, float %455)
*float8B

	full_text


float %453
*float8B

	full_text


float %455
Nstore8BC
A
	full_text4
2
0store float %456, float* %452, align 4, !tbaa !8
*float8B

	full_text


float %456
,float*8B

	full_text

float* %452
`getelementptr8BM
K
	full_text>
<
:%457 = getelementptr inbounds float, float* %452, i64 %400
,float*8B

	full_text

float* %452
&i648B

	full_text


i64 %400
Mload8BC
A
	full_text4
2
0%458 = load float, float* %92, align 8, !tbaa !8
+float*8B

	full_text


float* %92
Nload8BD
B
	full_text5
3
1%459 = load float, float* %457, align 4, !tbaa !8
,float*8B

	full_text

float* %457
7fmul8B-
+
	full_text

%460 = fmul float %459, %8
*float8B

	full_text


float %459
gcall8B]
[
	full_textN
L
J%461 = tail call float @llvm.fmuladd.f32(float %7, float %458, float %460)
*float8B

	full_text


float %458
*float8B

	full_text


float %460
Nstore8BC
A
	full_text4
2
0store float %461, float* %457, align 4, !tbaa !8
*float8B

	full_text


float %461
,float*8B

	full_text

float* %457
`getelementptr8BM
K
	full_text>
<
:%462 = getelementptr inbounds float, float* %457, i64 %400
,float*8B

	full_text

float* %457
&i648B

	full_text


i64 %400
Mload8BC
A
	full_text4
2
0%463 = load float, float* %93, align 4, !tbaa !8
+float*8B

	full_text


float* %93
Nload8BD
B
	full_text5
3
1%464 = load float, float* %462, align 4, !tbaa !8
,float*8B

	full_text

float* %462
7fmul8B-
+
	full_text

%465 = fmul float %464, %8
*float8B

	full_text


float %464
gcall8B]
[
	full_textN
L
J%466 = tail call float @llvm.fmuladd.f32(float %7, float %463, float %465)
*float8B

	full_text


float %463
*float8B

	full_text


float %465
Nstore8BC
A
	full_text4
2
0store float %466, float* %462, align 4, !tbaa !8
*float8B

	full_text


float %466
,float*8B

	full_text

float* %462
Zcall8BP
N
	full_textA
?
=call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %59) #6
%i8*8B

	full_text
	
i8* %59
Zcall8BP
N
	full_textA
?
=call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %11) #6
%i8*8B

	full_text
	
i8* %11
$ret8B

	full_text


ret void
*float*8B

	full_text

	float* %0
*float*8B

	full_text

	float* %2
$i328B

	full_text


i32 %1
$i328B

	full_text


i32 %6
(float8B

	full_text


float %7
$i328B

	full_text


i32 %5
$i328B

	full_text


i32 %3
*float*8B

	full_text

	float* %4
(float8B

	full_text


float %8
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
�float*8B{
y
	full_textl
j
hfloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 0, i64 7)
�float*8B{
y
	full_textl
j
hfloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 1, i64 6)
$i648B

	full_text


i64 10
�float*8B|
z
	full_textm
k
ifloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 1, i64 12)
�float*8B{
y
	full_textl
j
hfloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 2, i64 2)
�float*8B{
y
	full_textl
j
hfloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 2, i64 7)
�float*8B{
y
	full_textl
j
hfloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 2, i64 8)
�float*8B|
z
	full_textm
k
ifloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 2, i64 11)
�float*8B{
y
	full_textl
j
hfloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 3, i64 1)
�float*8B{
y
	full_textl
j
hfloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 3, i64 9)
�float*8B|
z
	full_textm
k
ifloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 1, i64 11)
�float*8B|
z
	full_textm
k
ifloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 0, i64 15)
#i648B

	full_text	

i64 8
�float*8B{
y
	full_textl
j
hfloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 1, i64 2)
#i648B

	full_text	

i64 9
�float*8B|
z
	full_textm
k
ifloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 0, i64 12)
�float*8B|
z
	full_textm
k
ifloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 1, i64 15)
�float*8B{
y
	full_textl
j
hfloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 0, i64 0)
%i18B

	full_text


i1 false
�float*8B{
y
	full_textl
j
hfloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 0, i64 3)
�float*8B|
z
	full_textm
k
ifloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 1, i64 13)
�float*8B{
y
	full_textl
j
hfloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 3, i64 7)
�float*8B{
y
	full_textl
j
hfloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 3, i64 8)
�float*8B|
z
	full_textm
k
ifloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 3, i64 10)
�float*8B{
y
	full_textl
j
hfloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 1, i64 4)
2float8B%
#
	full_text

float 0.000000e+00
�float*8B{
y
	full_textl
j
hfloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 3, i64 4)
�float*8B{
y
	full_textl
j
hfloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 0, i64 9)
�float*8B{
y
	full_textl
j
hfloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 2, i64 1)
�float*8B{
y
	full_textl
j
hfloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 3, i64 3)
#i328B

	full_text	

i32 2
�float*8B{
y
	full_textl
j
hfloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 0, i64 2)
�float*8B{
y
	full_textl
j
hfloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 0, i64 6)
#i328B

	full_text	

i32 6
�float*8B{
y
	full_textl
j
hfloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 1, i64 8)
�float*8B|
z
	full_textm
k
ifloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 3, i64 11)
�float*8B{
y
	full_textl
j
hfloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 0, i64 8)
�float*8B|
z
	full_textm
k
ifloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 2, i64 12)
$i648B

	full_text


i64 15
#i648B

	full_text	

i64 1
�float*8B|
z
	full_textm
k
ifloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 2, i64 13)
$i648B

	full_text


i64 16
�float*8B{
y
	full_textl
j
hfloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 1, i64 1)
�float*8B{
y
	full_textl
j
hfloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 3, i64 6)
�float*8B|
z
	full_textm
k
ifloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 3, i64 14)
�float*8B{
y
	full_textl
j
hfloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 3, i64 0)
�float*8B{
y
	full_textl
j
hfloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 2, i64 6)
�float*8B|
z
	full_textm
k
ifloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 3, i64 15)
�float*8B{
y
	full_textl
j
hfloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 0, i64 4)
$i648B

	full_text


i64 32
#i648B

	full_text	

i64 7
�float*8B|
z
	full_textm
k
ifloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 0, i64 14)
�float*8B{
y
	full_textl
j
hfloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 2, i64 5)
$i648B

	full_text


i64 14
�float*8B|
z
	full_textm
k
ifloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 3, i64 13)
$i648B

	full_text


i64 11
�float*8B{
y
	full_textl
j
hfloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 0, i64 1)
�float*8B{
y
	full_textl
j
hfloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 3, i64 2)
#i328B

	full_text	

i32 1
#i648B

	full_text	

i64 4
�float*8B{
y
	full_textl
j
hfloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 1, i64 3)
�float*8B{
y
	full_textl
j
hfloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 1, i64 0)
#i328B

	full_text	

i32 4
#i648B

	full_text	

i64 0
#i328B

	full_text	

i32 3
�float*8B|
z
	full_textm
k
ifloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 0, i64 10)
�float*8B|
z
	full_textm
k
ifloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 2, i64 14)
#i648B

	full_text	

i64 6
�float*8B|
z
	full_textm
k
ifloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 2, i64 15)
�float*8B{
y
	full_textl
j
hfloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 1, i64 9)
�float*8B{
y
	full_textl
j
hfloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 3, i64 5)
�float*8B{
y
	full_textl
j
hfloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 2, i64 3)
!i88B

	full_text

i8 0
�float*8B{
y
	full_textl
j
hfloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 2, i64 0)
�float*8B{
y
	full_textl
j
hfloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 0, i64 5)
{[4 x [16 x float]]*8B`
^
	full_textQ
O
M@sgemmNT.bs = internal unnamed_addr global [4 x [16 x float]] undef, align 16
�float*8B|
z
	full_textm
k
ifloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 0, i64 11)
�float*8B|
z
	full_textm
k
ifloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 1, i64 14)
$i648B

	full_text


i64 12
#i328B

	full_text	

i32 0
�float*8B|
z
	full_textm
k
ifloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 3, i64 12)
�float*8B|
z
	full_textm
k
ifloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 1, i64 10)
$i648B

	full_text


i64 64
�float*8B|
z
	full_textm
k
ifloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 0, i64 13)
�float*8B{
y
	full_textl
j
hfloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 2, i64 4)
�float*8B{
y
	full_textl
j
hfloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 2, i64 9)
�float*8B{
y
	full_textl
j
hfloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 1, i64 5)
#i648B

	full_text	

i64 2
�float*8B|
z
	full_textm
k
ifloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 2, i64 10)
$i648B

	full_text


i64 13
#i648B

	full_text	

i64 3
�float*8B{
y
	full_textl
j
hfloat* getelementptr inbounds ([4 x [16 x float]], [4 x [16 x float]]* @sgemmNT.bs, i64 0, i64 1, i64 7)
#i648B

	full_text	

i64 5        		 
 

                      !    "# "" $% $& $$ '( ') '' *+ ** ,- ,, ./ .0 .. 12 11 34 33 56 55 77 89 88 :; :: <= << >? >> @A @B @@ CD CE CC FG FF HI HH JK JJ LM LL NO NP NN QR QQ ST SU SS VW VV XY XX Z[ ZZ \] \\ ^_ ^` ^^ ab aa cd ce cc fg ff hi hh jk jj lm ll no np nn qr qq st ss uv uu wx ww yy zz {| {{ }~ }} �  �� �� �� �� �� �� �� �� �� �� �
� �
� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �
� �� �
� �� �
� �� �
� �� �
� �� �
� �� �
� �� �
� �� �
� �� �
� �� �
� �� �
� �� �
� �� �
� �� �
� �� �
� �� �� �
� �� �� �
� �� �� �
� �� �� �
� �� �� �
� �� �
� �� �� �
� �� �� �
� �� �� �
� �� �� �� �� �� �� �� �
� �� �� �
� �� �� �� �� �� �� �
� �� �� �
� �� �� �� �� �� �� �
� �� �� �
� �� �� �� �� �� �� �
� �� �� �� �� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �� �� �
� �
� �� �� �
� �� �� �� �
� �
� �� �� �
� �� �� �� �
� �
� �� �� �
� �� �� �� �
� �
� �� �� �
� �� �� �� �
� �
� �� �� �
� �� �� �� �
� �
� �� �� �
� �� �� �� �
� �
� �� �� �
� �� �� �� �
� �
� �� �� �
� �� �� �� �
� �
� �� �� �
� �� �� �� �
� �
� �� �� �
� �� �� �� �
� �
� �� �� �
� �� �� �� �
� �
� �� �� �
� �� �� �� �
� �
� �� �� �
� �� �� �� �
� �
� �� �� �
� �� �� �� �
� �
� �� �� �
� �� �� �� �
� �
� �� �� �
� �� �� �
� �� �� �
� �� �� �
� �� �� �� �
� �� �� �� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �
� �
� �� �� �� �� �� �
� �
� �� �� �
� �� �� �� �
� �
� �� �� �
� �� �� �� �
� �
� �� �� �
� �� �� �� �
� �
� �� �� �
� �� �� �� �
� �
� �� �� �
� �� �� �� �
� �
� �� �� �
� �� �� �� �
� �
� �� �� �
� �� �� �� �
� �
� �� �� �
� �� �� �� �
� �
� �� �� �
� �� �� �� �
� �
� �� �� �
� �� �� �� �
� �
� �� �� �
� �� �� �� �
� �
� �� �� �
� �� �� �� �
� �
� �� �� �
� �� �� �� �
� �
� �� �� �
� �� �� �� �
� �
� �� �� �
� �� �� �� �
� �
� �� �� �
� �� �� �� �� �� �� �
� �
� �� �� �
� �� �� �
� �� �� �� �� �� �
� �
� �� �� �
� �� �� �
� �� �� �� �� �� �
� �
� �� �� �
� �� �� �
� �� �� �� �� �� �
� �
� �� �� �
� �� �� �
� �� �� �� �� �� �
� �
� �� �� �
� �� �� �
� �� �� �� �� �� �
� �
� �� �� �
� �� �� �
� �� �� �� �� �� �
� �
� �� �� �
� �� �� �
� �� �� �� ��	 �� �	
�	 �	
�	 �	�	 �	�	 �	
�	 �	�	 �	�	 �	
�	 �	�	 �	�	 �	�	 �	�	 �	�	 �	
�	 �	
�	 �	�	 �	�	 �	
�	 �	�	 �	�	 �	
�	 �	�	 �	�	 �	�	 �	�	 �	�	 �	
�	 �	
�	 �	�	 �	�	 �	
�	 �	�	 �	�	 �	
�	 �	�	 �	�	 �	�	 �	�	 �	�	 �	
�	 �	
�	 �	�	 �	�	 �	
�	 �	�	 �	�	 �	
�	 �	�	 �	�	 �	�	 �	�	 �	�	 �	
�	 �	
�	 �	�	 �	�	 �	
�	 �	�	 �	�	 �	
�	 �	�	 �	�	 �	�	 �	�	 �	�	 �	
�	 �	
�	 �	�	 �	�	 �	
�	 �	�	 �	�	 �	
�	 �	�	 �	�	 �	�	 �	�	 �	�	 �	�	 �	�	 �	
�	 �	
�	 �	�	 �	�	 �	
�	 �	�	 �	�	 �	
�	 �	�	 �	�	 �	�	 �	�	 �	�	 �	�	 �	�	 �	
�	 �	
�	 �	�	 �	�	 �	
�	 �	�	 �	�	 �	
�	 �	�	 �	�	 �	�	 �	�	 �	�	 �	�	 �	�	 �	
�	 �	
�	 �	�	 �	�	 �	
�	 �	�	 �	
�	 �	�	 �	
�	 �	�	 �	�	  �	 u�	 7�	 y�	 ��	 ��	 ��	 ��	 ��	 ��	 ��	 ��	 ��	 ��	 �	�	 �	�	 �	�	 �	�	 �	�	 �	�	 �	�	 �	�	 �		�	 ,�	 �	�
 "�
 z
�
 ��
 3
�
 �
�
 �
�
 �
�
 �
�
 �
�
 �
�
 �
�
 �
�
 �	
�
 �	
�
 �	
�
 �	
�
 �	
�
 �	
�
 �	
�
 �	  	     
       !
 #" % &$ ( )' + -, / 0. 21 4 6  98 ; = ?: A> B  D7 EC GF I KJ MH OL P7 R  TQ US WV Y [Z ]X _\ `7 b  da ec gf i kj mh ol p r t* vu xq |q ~y �z �	 �� � �� �� �� � �� �� � � � � � � � � � � � � � � � � � � � � �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �u �� �  �� �z �� �w �� �� � �s �s � �� �� �� �� �� �� �� �� �� �7 �� �� �� �L �� �� �� �� �� �\ �� �� �� �� �� �l �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �z �s �� �� �� �� �� �< �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �J �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �Z �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �j �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �3 �� �� �� �� �3 �3 �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �	� �	� �	�	 �	� �	� �	� �	�	 �	�	 �	� �	�	 �	�	 �	�	 �	�	 �	� �	�	 �	�	 �	� �	�	 �	�	 �	�	 �	�	 �	� �	�	 �	�	 �	� �	�	 �	�	 �	�	 �	�	 �	� �	�	 �	�	 �	� �	�	 �	�	 �	�	 �	�	 �	� �	�	 �	�	 �	� �	�	 �	�	 �	�	 �	�	 �	� �	� �	�	 �	�	 �	�	 �	�	 �	�	 �	�	 �	�	 �	� �	� �	�	 �	�	 �	�	 �	�	 �	�	 �	�	 �	�	 �	� �	� �	�	 �	�	 �	�	 �	�	 �	�	 �	�	 �	q �	 �	� �� �� � �
�
 �
�
 �
�
 �
�
 �
�
 �	 �
�
 �
�
 �
�
� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 � �
�
 � �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 ��	 �
�
 �	 �
�
 � �
�
 ��	 �
�
 �	� �
�
 �� �
�
 �� �
�
 �	 �
�
 	� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �{ �
�
 {� �
�
 ��	 �
�
 �	� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 ��	 �
�
 �	� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 ��	 �
�
 �	�	 �
�
 �	� �
�
 �� �
�
 �� �
�
 �� �
�
 ��	 �
�
 �	� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 ��	 �
�
 �	� �
�
 �� �
�
 �� �
�
 ��	 �
�
 �	�	 �
�
 �	� �
�
 �� �
�
 �� �
�
 � �
�
 � �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 ��	 �
�
 �	� �
�
 �� �
�
 �� �
�
 �5 �
�
 5� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �} �
�
 }� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 �� �
�
 ��
 ��
 ��
 ��
 �
�
 ��
 ��
 ��
 ��
 ��
 ��
 ��
 ��
 ��
 ��
 ��
 ��
 ��
 ��
 ��
 ��
 ��
 ��
 �
�
 ��
 ��
 �
�
 ��
 ��
 ��
 ��
 ��
 ��
 �	�
 }
�
 ��
 ��
 ��
 ��
 ��
 ��
 ��
 ��
 ��
 ��
 ��
 ��
 ��
 ��
 ��
 ��
 ��
 ��
 ��
 ��
 ��
 ��
 ��
 ��
 ��
 ��
 ��
 ��
 ��
 ��
 ��
 ��
 ��
 ��
 ��
 ��
 �	�
 y	�
 z�
 ��
 ��
 ��
 �	�
 �
 ��
 ��
 ��
 ��
 ��
 ��
 ��
 �
�
 �	�
 J	�
 Q
�
 �
�
 ��
 ��
 ��
 5�
 �
�
 ��
 ��
 �	�
 ��
 ��
 ��
 ��
 ��
 ��
 ��
 ��
 ��
 ��
 ��
 ��
 ��
 �
�
 �
�
 �
�
 �
�
 �
�
 ��
 ��
 ��
 ��
 �
�
 ��
 ��
 �
�
 ��
 ��
 ��
 ��
 ��
 �
 �
 �
 	�
 
�
 ��
 ��
 ��
 �
�
 ��
 ��
 ��
 ��
 �	�
 	�
 	�
 <	�
 <	�
 J	�
 Z	�
 j
�
 �
�
 �
�
 �
�
 �
�
 �
�
 �
�
 �
�
 �
�
 �
�
 �
�
 �
�
 �
�
 �
�
 �
�
 �
�
 �
�
 �
�
 �
�
 �
�
 �
�
 �
�
 �
�
 �
�
 ��
 ��
 ��
 ��
 �
�
 ��
 ��
 ��
 ��
 ��
 ��
 ��
 ��
 �	�
 }�
 ��
 ��
 ��
 ��
 ��
 ��
 ��
 ��
 �
�
 ��
 �
 �
 ��
 ��
 ��
 ��
 {	�
 }�
 �	�
 ��
 ��
 ��
 ��
 ��
 ��
 ��
 �	�
 Z
�
 �
�
 ��
 ��
 �
�
 �	�
 a	�
 j
�
 �
�
 ��
 ��
 �
�
 �"	
sgemmNT"
llvm.lifetime.start.p0i8"
_Z12get_local_idj"
_Z12get_group_idj"
_Z7barrierj"
llvm.fmuladd.f32"
llvm.lifetime.end.p0i8"
llvm.memcpy.p0i8.p0i8.i64"
llvm.memset.p0i8.i64*�
shoc-1.1.5-GEMM-sgemmNT.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282�

transfer_bytes
���

wgsize
@
 
transfer_bytes_log1p
0�jA

wgsize_log1p
0�jA

devmap_label
 