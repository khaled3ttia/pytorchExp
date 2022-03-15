

[external]
KcallBC
A
	full_text4
2
0%14 = tail call i64 @_Z12get_group_idj(i32 0) #4
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
0%16 = tail call i64 @_Z12get_group_idj(i32 1) #4
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
0%18 = tail call i64 @_Z12get_local_idj(i32 0) #4
6truncB-
+
	full_text

%19 = trunc i64 %18 to i32
#i64B

	full_text
	
i64 %18
KcallBC
A
	full_text4
2
0%20 = tail call i64 @_Z12get_local_idj(i32 1) #4
6truncB-
+
	full_text

%21 = trunc i64 %20 to i32
#i64B

	full_text
	
i64 %20
?fdivB7
5
	full_text(
&
$%22 = fdiv float %12, %8, !fpmath !8
HfdivB@
>
	full_text1
/
-%23 = fdiv float 1.000000e+00, %9, !fpmath !8
IfdivBA
?
	full_text2
0
.%24 = fdiv float 1.000000e+00, %10, !fpmath !8
IfdivBA
?
	full_text2
0
.%25 = fdiv float 1.000000e+00, %11, !fpmath !8
1shlB*
(
	full_text

%26 = shl nsw i32 %0, 1
3subB,
*
	full_text

%27 = sub nsw i32 64, %26
#i32B

	full_text
	
i32 %26
4mulB-
+
	full_text

%28 = mul nsw i32 %27, %17
#i32B

	full_text
	
i32 %27
#i32B

	full_text
	
i32 %17
3subB,
*
	full_text

%29 = sub nsw i32 %28, %7
#i32B

	full_text
	
i32 %28
4mulB-
+
	full_text

%30 = mul nsw i32 %27, %15
#i32B

	full_text
	
i32 %27
#i32B

	full_text
	
i32 %15
3subB,
*
	full_text

%31 = sub nsw i32 %30, %6
#i32B

	full_text
	
i32 %30
3addB,
*
	full_text

%32 = add nsw i32 %29, 63
#i32B

	full_text
	
i32 %29
3addB,
*
	full_text

%33 = add nsw i32 %31, 63
#i32B

	full_text
	
i32 %31
4addB-
+
	full_text

%34 = add nsw i32 %29, %21
#i32B

	full_text
	
i32 %29
#i32B

	full_text
	
i32 %21
4addB-
+
	full_text

%35 = add nsw i32 %31, %19
#i32B

	full_text
	
i32 %31
#i32B

	full_text
	
i32 %19
3mulB,
*
	full_text

%36 = mul nsw i32 %34, %4
#i32B

	full_text
	
i32 %34
4addB-
+
	full_text

%37 = add nsw i32 %36, %35
#i32B

	full_text
	
i32 %36
#i32B

	full_text
	
i32 %35
5icmpB-
+
	full_text

%38 = icmp sgt i32 %34, -1
#i32B

	full_text
	
i32 %34
8brB2
0
	full_text#
!
br i1 %38, label %39, label %61
!i1B

	full_text


i1 %38
7icmp8B-
+
	full_text

%40 = icmp slt i32 %34, %5
%i328B

	full_text
	
i32 %34
7icmp8B-
+
	full_text

%41 = icmp sgt i32 %35, -1
%i328B

	full_text
	
i32 %35
1and8B(
&
	full_text

%42 = and i1 %41, %40
#i18B

	full_text


i1 %41
#i18B

	full_text


i1 %40
7icmp8B-
+
	full_text

%43 = icmp slt i32 %35, %4
%i328B

	full_text
	
i32 %35
1and8B(
&
	full_text

%44 = and i1 %43, %42
#i18B

	full_text


i1 %43
#i18B

	full_text


i1 %42
:br8B2
0
	full_text#
!
br i1 %44, label %45, label %61
#i18B

	full_text


i1 %44
6sext8B,
*
	full_text

%46 = sext i32 %37 to i64
%i328B

	full_text
	
i32 %37
\getelementptr8BI
G
	full_text:
8
6%47 = getelementptr inbounds float, float* %2, i64 %46
%i648B

	full_text
	
i64 %46
@bitcast8B3
1
	full_text$
"
 %48 = bitcast float* %47 to i32*
+float*8B

	full_text


float* %47
Hload8B>
<
	full_text/
-
+%49 = load i32, i32* %48, align 4, !tbaa !9
'i32*8B

	full_text


i32* %48
1shl8B(
&
	full_text

%50 = shl i64 %20, 32
%i648B

	full_text
	
i64 %20
9ashr8B/
-
	full_text 

%51 = ashr exact i64 %50, 32
%i648B

	full_text
	
i64 %50
1shl8B(
&
	full_text

%52 = shl i64 %18, 32
%i648B

	full_text
	
i64 %18
9ashr8B/
-
	full_text 

%53 = ashr exact i64 %52, 32
%i648B

	full_text
	
i64 %52
ùgetelementptr8Bâ
Ü
	full_texty
w
u%54 = getelementptr inbounds [64 x [64 x float]], [64 x [64 x float]]* @hotspot.temp_on_cuda, i64 0, i64 %51, i64 %53
%i648B

	full_text
	
i64 %51
%i648B

	full_text
	
i64 %53
@bitcast8B3
1
	full_text$
"
 %55 = bitcast float* %54 to i32*
+float*8B

	full_text


float* %54
Hstore8B=
;
	full_text.
,
*store i32 %49, i32* %55, align 4, !tbaa !9
%i328B

	full_text
	
i32 %49
'i32*8B

	full_text


i32* %55
\getelementptr8BI
G
	full_text:
8
6%56 = getelementptr inbounds float, float* %1, i64 %46
%i648B

	full_text
	
i64 %46
@bitcast8B3
1
	full_text$
"
 %57 = bitcast float* %56 to i32*
+float*8B

	full_text


float* %56
Hload8B>
<
	full_text/
-
+%58 = load i32, i32* %57, align 4, !tbaa !9
'i32*8B

	full_text


i32* %57
ûgetelementptr8Bä
á
	full_textz
x
v%59 = getelementptr inbounds [64 x [64 x float]], [64 x [64 x float]]* @hotspot.power_on_cuda, i64 0, i64 %51, i64 %53
%i648B

	full_text
	
i64 %51
%i648B

	full_text
	
i64 %53
@bitcast8B3
1
	full_text$
"
 %60 = bitcast float* %59 to i32*
+float*8B

	full_text


float* %59
Hstore8B=
;
	full_text.
,
*store i32 %58, i32* %60, align 4, !tbaa !9
%i328B

	full_text
	
i32 %58
'i32*8B

	full_text


i32* %60
'br8B

	full_text

br label %61
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #5
6icmp8B,
*
	full_text

%62 = icmp slt i32 %29, 0
%i328B

	full_text
	
i32 %29
4sub8B+
)
	full_text

%63 = sub nsw i32 0, %29
%i328B

	full_text
	
i32 %29
Bselect8B6
4
	full_text'
%
#%64 = select i1 %62, i32 %63, i32 0
#i18B

	full_text


i1 %62
%i328B

	full_text
	
i32 %63
7icmp8B-
+
	full_text

%65 = icmp slt i32 %32, %5
%i328B

	full_text
	
i32 %32
2sub8B)
'
	full_text

%66 = sub i32 -63, %29
%i328B

	full_text
	
i32 %29
0add8B'
%
	full_text

%67 = add i32 %5, 62
2add8B)
'
	full_text

%68 = add i32 %67, %66
%i328B

	full_text
	
i32 %67
%i328B

	full_text
	
i32 %66
Cselect8B7
5
	full_text(
&
$%69 = select i1 %65, i32 63, i32 %68
#i18B

	full_text


i1 %65
%i328B

	full_text
	
i32 %68
6icmp8B,
*
	full_text

%70 = icmp slt i32 %31, 0
%i328B

	full_text
	
i32 %31
4sub8B+
)
	full_text

%71 = sub nsw i32 0, %31
%i328B

	full_text
	
i32 %31
Bselect8B6
4
	full_text'
%
#%72 = select i1 %70, i32 %71, i32 0
#i18B

	full_text


i1 %70
%i328B

	full_text
	
i32 %71
7icmp8B-
+
	full_text

%73 = icmp slt i32 %33, %4
%i328B

	full_text
	
i32 %33
2sub8B)
'
	full_text

%74 = sub i32 -63, %31
%i328B

	full_text
	
i32 %31
0add8B'
%
	full_text

%75 = add i32 %4, 62
2add8B)
'
	full_text

%76 = add i32 %75, %74
%i328B

	full_text
	
i32 %75
%i328B

	full_text
	
i32 %74
Cselect8B7
5
	full_text(
&
$%77 = select i1 %73, i32 63, i32 %76
#i18B

	full_text


i1 %73
%i328B

	full_text
	
i32 %76
5add8B,
*
	full_text

%78 = add nsw i32 %21, -1
%i328B

	full_text
	
i32 %21
4add8B+
)
	full_text

%79 = add nsw i32 %21, 1
%i328B

	full_text
	
i32 %21
5add8B,
*
	full_text

%80 = add nsw i32 %19, -1
%i328B

	full_text
	
i32 %19
4add8B+
)
	full_text

%81 = add nsw i32 %19, 1
%i328B

	full_text
	
i32 %19
5icmp8B+
)
	full_text

%82 = icmp sgt i32 %0, 0
;br8B3
1
	full_text$
"
 br i1 %82, label %83, label %168
#i18B

	full_text


i1 %82
8icmp8B.
,
	full_text

%84 = icmp sgt i32 %81, %77
%i328B

	full_text
	
i32 %81
%i328B

	full_text
	
i32 %77
Dselect8B8
6
	full_text)
'
%%85 = select i1 %84, i32 %77, i32 %81
#i18B

	full_text


i1 %84
%i328B

	full_text
	
i32 %77
%i328B

	full_text
	
i32 %81
8icmp8B.
,
	full_text

%86 = icmp slt i32 %80, %72
%i328B

	full_text
	
i32 %80
%i328B

	full_text
	
i32 %72
Dselect8B8
6
	full_text)
'
%%87 = select i1 %86, i32 %72, i32 %80
#i18B

	full_text


i1 %86
%i328B

	full_text
	
i32 %72
%i328B

	full_text
	
i32 %80
8icmp8B.
,
	full_text

%88 = icmp sgt i32 %79, %69
%i328B

	full_text
	
i32 %79
%i328B

	full_text
	
i32 %69
Dselect8B8
6
	full_text)
'
%%89 = select i1 %88, i32 %69, i32 %79
#i18B

	full_text


i1 %88
%i328B

	full_text
	
i32 %69
%i328B

	full_text
	
i32 %79
8icmp8B.
,
	full_text

%90 = icmp slt i32 %78, %64
%i328B

	full_text
	
i32 %78
%i328B

	full_text
	
i32 %64
Dselect8B8
6
	full_text)
'
%%91 = select i1 %90, i32 %64, i32 %78
#i18B

	full_text


i1 %90
%i328B

	full_text
	
i32 %64
%i328B

	full_text
	
i32 %78
8icmp8B.
,
	full_text

%92 = icmp sgt i32 %72, %19
%i328B

	full_text
	
i32 %72
%i328B

	full_text
	
i32 %19
8icmp8B.
,
	full_text

%93 = icmp slt i32 %77, %19
%i328B

	full_text
	
i32 %77
%i328B

	full_text
	
i32 %19
8icmp8B.
,
	full_text

%94 = icmp sgt i32 %64, %21
%i328B

	full_text
	
i32 %64
%i328B

	full_text
	
i32 %21
8icmp8B.
,
	full_text

%95 = icmp slt i32 %69, %21
%i328B

	full_text
	
i32 %69
%i328B

	full_text
	
i32 %21
1shl8B(
&
	full_text

%96 = shl i64 %20, 32
%i648B

	full_text
	
i64 %20
9ashr8B/
-
	full_text 

%97 = ashr exact i64 %96, 32
%i648B

	full_text
	
i64 %96
1shl8B(
&
	full_text

%98 = shl i64 %18, 32
%i648B

	full_text
	
i64 %18
9ashr8B/
-
	full_text 

%99 = ashr exact i64 %98, 32
%i648B

	full_text
	
i64 %98
ûgetelementptr8Bä
á
	full_textz
x
v%100 = getelementptr inbounds [64 x [64 x float]], [64 x [64 x float]]* @hotspot.temp_on_cuda, i64 0, i64 %97, i64 %99
%i648B

	full_text
	
i64 %97
%i648B

	full_text
	
i64 %99
ügetelementptr8Bã
à
	full_text{
y
w%101 = getelementptr inbounds [64 x [64 x float]], [64 x [64 x float]]* @hotspot.power_on_cuda, i64 0, i64 %97, i64 %99
%i648B

	full_text
	
i64 %97
%i648B

	full_text
	
i64 %99
7sext8B-
+
	full_text

%102 = sext i32 %89 to i64
%i328B

	full_text
	
i32 %89
ügetelementptr8Bã
à
	full_text{
y
w%103 = getelementptr inbounds [64 x [64 x float]], [64 x [64 x float]]* @hotspot.temp_on_cuda, i64 0, i64 %102, i64 %99
&i648B

	full_text


i64 %102
%i648B

	full_text
	
i64 %99
7sext8B-
+
	full_text

%104 = sext i32 %91 to i64
%i328B

	full_text
	
i32 %91
ügetelementptr8Bã
à
	full_text{
y
w%105 = getelementptr inbounds [64 x [64 x float]], [64 x [64 x float]]* @hotspot.temp_on_cuda, i64 0, i64 %104, i64 %99
&i648B

	full_text


i64 %104
%i648B

	full_text
	
i64 %99
7sext8B-
+
	full_text

%106 = sext i32 %85 to i64
%i328B

	full_text
	
i32 %85
ügetelementptr8Bã
à
	full_text{
y
w%107 = getelementptr inbounds [64 x [64 x float]], [64 x [64 x float]]* @hotspot.temp_on_cuda, i64 0, i64 %97, i64 %106
%i648B

	full_text
	
i64 %97
&i648B

	full_text


i64 %106
7sext8B-
+
	full_text

%108 = sext i32 %87 to i64
%i328B

	full_text
	
i32 %87
ügetelementptr8Bã
à
	full_text{
y
w%109 = getelementptr inbounds [64 x [64 x float]], [64 x [64 x float]]* @hotspot.temp_on_cuda, i64 0, i64 %97, i64 %108
%i648B

	full_text
	
i64 %97
&i648B

	full_text


i64 %108
ògetelementptr8BÑ
Å
	full_textt
r
p%110 = getelementptr inbounds [64 x [64 x float]], [64 x [64 x float]]* @hotspot.temp_t, i64 0, i64 %97, i64 %99
%i648B

	full_text
	
i64 %97
%i648B

	full_text
	
i64 %99
5add8B,
*
	full_text

%111 = add nsw i32 %0, -1
Bbitcast8B5
3
	full_text&
$
"%112 = bitcast float* %110 to i32*
,float*8B

	full_text

float* %110
Bbitcast8B5
3
	full_text&
$
"%113 = bitcast float* %100 to i32*
,float*8B

	full_text

float* %100
(br8B 

	full_text

br label %114
Ephi8B<
:
	full_text-
+
)%115 = phi i32 [ 0, %83 ], [ %116, %153 ]
&i328B

	full_text


i32 %116
:add8B1
/
	full_text"
 
%116 = add nuw nsw i32 %115, 1
&i328B

	full_text


i32 %115
:icmp8B0
.
	full_text!

%117 = icmp slt i32 %115, %19
&i328B

	full_text


i32 %115
%i328B

	full_text
	
i32 %19
=br8B5
3
	full_text&
$
"br i1 %117, label %118, label %146
$i18B

	full_text
	
i1 %117
7sub8B.
,
	full_text

%119 = sub nsw i32 64, %115
&i328B

	full_text


i32 %115
7add8B.
,
	full_text

%120 = add nsw i32 %119, -2
&i328B

	full_text


i32 %119
:icmp8B0
.
	full_text!

%121 = icmp slt i32 %120, %19
&i328B

	full_text


i32 %120
%i328B

	full_text
	
i32 %19
:icmp8B0
.
	full_text!

%122 = icmp sge i32 %115, %21
&i328B

	full_text


i32 %115
%i328B

	full_text
	
i32 %21
2or8B*
(
	full_text

%123 = or i1 %122, %121
$i18B

	full_text
	
i1 %122
$i18B

	full_text
	
i1 %121
:icmp8B0
.
	full_text!

%124 = icmp slt i32 %120, %21
&i328B

	full_text


i32 %120
%i328B

	full_text
	
i32 %21
2or8B*
(
	full_text

%125 = or i1 %124, %123
$i18B

	full_text
	
i1 %124
$i18B

	full_text
	
i1 %123
1or8B)
'
	full_text

%126 = or i1 %92, %125
#i18B

	full_text


i1 %92
$i18B

	full_text
	
i1 %125
1or8B)
'
	full_text

%127 = or i1 %93, %126
#i18B

	full_text


i1 %93
$i18B

	full_text
	
i1 %126
1or8B)
'
	full_text

%128 = or i1 %94, %127
#i18B

	full_text


i1 %94
$i18B

	full_text
	
i1 %127
1or8B)
'
	full_text

%129 = or i1 %95, %128
#i18B

	full_text


i1 %95
$i18B

	full_text
	
i1 %128
=br8B5
3
	full_text&
$
"br i1 %129, label %146, label %130
$i18B

	full_text
	
i1 %129
Nload8BD
B
	full_text5
3
1%131 = load float, float* %100, align 4, !tbaa !9
,float*8B

	full_text

float* %100
Nload8BD
B
	full_text5
3
1%132 = load float, float* %101, align 4, !tbaa !9
,float*8B

	full_text

float* %101
Nload8BD
B
	full_text5
3
1%133 = load float, float* %103, align 4, !tbaa !9
,float*8B

	full_text

float* %103
Nload8BD
B
	full_text5
3
1%134 = load float, float* %105, align 4, !tbaa !9
,float*8B

	full_text

float* %105
9fadd8B/
-
	full_text 

%135 = fadd float %133, %134
*float8B

	full_text


float %133
*float8B

	full_text


float %134
rcall8Bh
f
	full_textY
W
U%136 = tail call float @llvm.fmuladd.f32(float %131, float -2.000000e+00, float %135)
*float8B

	full_text


float %131
*float8B

	full_text


float %135
hcall8B^
\
	full_textO
M
K%137 = tail call float @llvm.fmuladd.f32(float %136, float %24, float %132)
*float8B

	full_text


float %136
)float8B

	full_text

	float %24
*float8B

	full_text


float %132
Nload8BD
B
	full_text5
3
1%138 = load float, float* %107, align 4, !tbaa !9
,float*8B

	full_text

float* %107
Nload8BD
B
	full_text5
3
1%139 = load float, float* %109, align 4, !tbaa !9
,float*8B

	full_text

float* %109
9fadd8B/
-
	full_text 

%140 = fadd float %138, %139
*float8B

	full_text


float %138
*float8B

	full_text


float %139
rcall8Bh
f
	full_textY
W
U%141 = tail call float @llvm.fmuladd.f32(float %131, float -2.000000e+00, float %140)
*float8B

	full_text


float %131
*float8B

	full_text


float %140
hcall8B^
\
	full_textO
M
K%142 = tail call float @llvm.fmuladd.f32(float %141, float %23, float %137)
*float8B

	full_text


float %141
)float8B

	full_text

	float %23
*float8B

	full_text


float %137
Afsub8B7
5
	full_text(
&
$%143 = fsub float 8.000000e+01, %131
*float8B

	full_text


float %131
hcall8B^
\
	full_textO
M
K%144 = tail call float @llvm.fmuladd.f32(float %143, float %25, float %142)
*float8B

	full_text


float %143
)float8B

	full_text

	float %25
*float8B

	full_text


float %142
hcall8B^
\
	full_textO
M
K%145 = tail call float @llvm.fmuladd.f32(float %22, float %144, float %131)
)float8B

	full_text

	float %22
*float8B

	full_text


float %144
*float8B

	full_text


float %131
Nstore8BC
A
	full_text4
2
0store float %145, float* %110, align 4, !tbaa !9
*float8B

	full_text


float %145
,float*8B

	full_text

float* %110
(br8B 

	full_text

br label %146
Ophi8BF
D
	full_text7
5
3%147 = phi i8 [ 1, %130 ], [ 0, %118 ], [ 0, %114 ]
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #5
:icmp8B0
.
	full_text!

%148 = icmp eq i32 %115, %111
&i328B

	full_text


i32 %115
&i328B

	full_text


i32 %111
=br8B5
3
	full_text&
$
"br i1 %148, label %155, label %149
$i18B

	full_text
	
i1 %148
6icmp8	B,
*
	full_text

%150 = icmp eq i8 %147, 0
$i88	B

	full_text
	
i8 %147
=br8	B5
3
	full_text&
$
"br i1 %150, label %153, label %151
$i18	B

	full_text
	
i1 %150
Jload8
B@
>
	full_text1
/
-%152 = load i32, i32* %112, align 4, !tbaa !9
(i32*8
B

	full_text

	i32* %112
Jstore8
B?
=
	full_text0
.
,store i32 %152, i32* %113, align 4, !tbaa !9
&i328
B

	full_text


i32 %152
(i32*8
B

	full_text

	i32* %113
(br8
B 

	full_text

br label %153
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #5
9icmp8B/
-
	full_text 

%154 = icmp slt i32 %116, %0
&i328B

	full_text


i32 %116
=br8B5
3
	full_text&
$
"br i1 %154, label %114, label %155
$i18B

	full_text
	
i1 %154
6icmp8B,
*
	full_text

%156 = icmp eq i8 %147, 0
$i88B

	full_text
	
i8 %147
=br8B5
3
	full_text&
$
"br i1 %156, label %168, label %157
$i18B

	full_text
	
i1 %156
2shl8B)
'
	full_text

%158 = shl i64 %20, 32
%i648B

	full_text
	
i64 %20
;ashr8B1
/
	full_text"
 
%159 = ashr exact i64 %158, 32
&i648B

	full_text


i64 %158
2shl8B)
'
	full_text

%160 = shl i64 %18, 32
%i648B

	full_text
	
i64 %18
;ashr8B1
/
	full_text"
 
%161 = ashr exact i64 %160, 32
&i648B

	full_text


i64 %160
ögetelementptr8BÜ
É
	full_textv
t
r%162 = getelementptr inbounds [64 x [64 x float]], [64 x [64 x float]]* @hotspot.temp_t, i64 0, i64 %159, i64 %161
&i648B

	full_text


i64 %159
&i648B

	full_text


i64 %161
Bbitcast8B5
3
	full_text&
$
"%163 = bitcast float* %162 to i32*
,float*8B

	full_text

float* %162
Jload8B@
>
	full_text1
/
-%164 = load i32, i32* %163, align 4, !tbaa !9
(i32*8B

	full_text

	i32* %163
7sext8B-
+
	full_text

%165 = sext i32 %37 to i64
%i328B

	full_text
	
i32 %37
^getelementptr8BK
I
	full_text<
:
8%166 = getelementptr inbounds float, float* %3, i64 %165
&i648B

	full_text


i64 %165
Bbitcast8B5
3
	full_text&
$
"%167 = bitcast float* %166 to i32*
,float*8B

	full_text

float* %166
Jstore8B?
=
	full_text0
.
,store i32 %164, i32* %167, align 4, !tbaa !9
&i328B

	full_text


i32 %164
(i32*8B

	full_text

	i32* %167
(br8B 

	full_text

br label %168
$ret8B

	full_text


ret void
*float*8B

	full_text

	float* %1
(float8B

	full_text


float %9
$i328B

	full_text


i32 %6
(float8B

	full_text


float %8
$i328B

	full_text


i32 %0
$i328B

	full_text


i32 %7
)float8B

	full_text

	float %11
)float8B

	full_text

	float %12
$i328B

	full_text


i32 %4
$i328B

	full_text


i32 %5
)float8B

	full_text

	float %10
*float*8B

	full_text

	float* %2
*float*8B
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
$i648B

	full_text


i64 32
$i328B

	full_text


i32 62
Å[64 x [64 x float]]*8Be
c
	full_textV
T
R@hotspot.temp_t = internal unnamed_addr global [64 x [64 x float]] undef, align 16
3float8B&
$
	full_text

float -2.000000e+00
2float8B%
#
	full_text

float 8.000000e+01
!i88B

	full_text

i8 0
$i328B

	full_text


i32 -1
$i328B

	full_text


i32 64
2float8B%
#
	full_text

float 1.000000e+00
!i88B

	full_text

i8 1
$i328B

	full_text


i32 -2
#i648B

	full_text	

i64 0
á[64 x [64 x float]]*8Bk
i
	full_text\
Z
X@hotspot.temp_on_cuda = internal unnamed_addr global [64 x [64 x float]] undef, align 16
#i328B

	full_text	

i32 1
#i328B

	full_text	

i32 0
$i328B

	full_text


i32 63
%i328B

	full_text
	
i32 -63
à[64 x [64 x float]]*8Bl
j
	full_text]
[
Y@hotspot.power_on_cuda = internal unnamed_addr global [64 x [64 x float]] undef, align 16       	  

                       !    "# "$ "" %& %' %% () (( *+ *, ** -. -- /0 /2 11 34 33 56 57 55 89 88 :; :< :: => =@ ?? AB AA CD CC EF EE GH GG IJ II KL KK MN MM OP OQ OO RS RR TU TV TT WX WW YZ YY [\ [[ ]^ ]_ ]] `a `` bc bd bb ef gh gg ij ii kl km kk no nn pq pp rr st su ss vw vx vv yz yy {| {{ }~ } }} ÄÅ ÄÄ Ç
É ÇÇ ÑÑ ÖÜ Ö
á ÖÖ àâ à
ä àà ãå ãã çé çç èê èè ëí ëë ìì îï îó ñ
ò ññ ôö ô
õ ô
ú ôô ùû ù
ü ùù †° †
¢ †
£ †† §• §
¶ §§ ß® ß
© ß
™ ßß ´¨ ´
≠ ´´ ÆØ Æ
∞ Æ
± ÆÆ ≤≥ ≤
¥ ≤≤ µ∂ µ
∑ µµ ∏π ∏
∫ ∏∏ ªº ª
Ω ªª æø ææ ¿¡ ¿¿ ¬√ ¬¬ ƒ≈ ƒƒ ∆
« ∆
» ∆∆ …
  …
À …… ÃÕ ÃÃ Œ
œ Œ
– ŒŒ —“ —— ”
‘ ”
’ ”” ÷◊ ÷÷ ÿ
Ÿ ÿ
⁄ ÿÿ €‹ €€ ›
ﬁ ›
ﬂ ›› ‡
· ‡
‚ ‡‡ „„ ‰Â ‰‰ ÊÁ ÊÊ Ë
Í ÈÈ ÎÏ ÎÎ ÌÓ Ì
Ô ÌÌ Ò 
Û ÚÚ Ùı ÙÙ ˆ˜ ˆ
¯ ˆˆ ˘˙ ˘
˚ ˘˘ ¸˝ ¸
˛ ¸¸ ˇÄ ˇ
Å ˇˇ ÇÉ Ç
Ñ ÇÇ ÖÜ Ö
á ÖÖ àâ à
ä àà ãå ã
ç ãã éè é
ê éé ëí ëî ìì ïñ ïï óò óó ôö ôô õú õ
ù õõ ûü û
† ûû °¢ °
£ °
§ °° •¶ •• ß® ßß ©™ ©
´ ©© ¨≠ ¨
Æ ¨¨ Ø∞ Ø
± Ø
≤ ØØ ≥
¥ ≥≥ µ∂ µ
∑ µ
∏ µµ π∫ π
ª π
º ππ Ωæ Ω
ø ΩΩ ¿¡ ¬¬ √ƒ √
≈ √√ ∆« ∆… »»  À  Õ ÃÃ Œœ Œ
– ŒŒ —“ ”‘ ”” ’÷ ’ÿ ◊◊ Ÿ⁄ Ÿ‹ €€ ›ﬁ ›› ﬂ‡ ﬂﬂ ·‚ ·· „
‰ „
Â „„ ÊÁ ÊÊ ËÈ ËË ÍÎ ÍÍ Ï
Ì ÏÏ ÓÔ ÓÓ Ò 
Ú  Ûı W	ˆ 	˜ 	¯ ˘ ˘ ì˘ „
˘ ”	˙ 	˚ ¸ 	˝ (	˝ 8
˝ Ä˝ Ñ	˛ 1	˛ n˛ r	ˇ Ä AÅ Ï   	
          ! # $ & '" )( +% ," .- 0" 2% 43 61 7% 98 ;5 <: >* @? BA DC F
 HG J LK NI PM QO SE UR V? XW ZY \I ^M _] a[ c` d h jg li m o qr tp un ws x z |y ~{   Å ÉÑ ÜÇ áÄ âÖ ä å é ê íì ïë óà òñ öà õë úè û} üù °} ¢è £ç •v ¶§ ®v ©ç ™ã ¨k ≠´ Øk ∞ã ±} ≥ ¥à ∂ ∑k π ∫v º Ω
 øæ ¡ √¬ ≈¿ «ƒ »¿  ƒ Àß ÕÃ œƒ –Æ “— ‘ƒ ’ô ◊¿ Ÿ÷ ⁄† ‹¿ ﬁ€ ﬂ¿ ·ƒ ‚‡ Â∆ ÁÎ ÍÈ ÏÈ Ó ÔÌ ÒÈ ÛÚ ıÙ ˜ ¯È ˙ ˚˘ ˝ˆ ˛Ù Ä Åˇ É¸ Ñ≤ ÜÇ áµ âÖ ä∏ åà çª èã êé í∆ î… ñŒ ò” öó úô ùì üõ †û ¢ £ï §ÿ ¶› ®• ™ß ´ì ≠© Æ¨ ∞ ±° ≤ì ¥≥ ∂ ∑Ø ∏ ∫µ ªì ºπ æ‡ øÈ ƒ„ ≈√ «¡ …» À‰ ÕÃ œÊ –Î ‘” ÷¡ ÿ◊ ⁄
 ‹€ ﬁ ‡ﬂ ‚› ‰· Â„ ÁÊ È* ÎÍ ÌÏ ÔË ÒÓ Ú/ 1/ f= ?= fî ñî Ùe fË È Ú ¡ë ¡ë ì∆ ◊∆ »¿ ¡Ÿ ÙŸ €  “  ÃÛ Ù’ È’ ◊— “ Ù ÑÑ ÇÇ ÖÖ ÉÉ¬ ÑÑ ¬π ÖÖ πf ÑÑ f ÉÉ  ÇÇ 
 ÉÉ 
¨ ÖÖ ¨µ ÖÖ µû ÖÖ û“ ÑÑ “° ÖÖ ° ÇÇ Ø ÖÖ Ø	Ü G	Ü I	Ü K	Ü M
Ü æ
Ü ¿
Ü ¬
Ü ƒ
Ü €
Ü ›
Ü ﬂ
Ü ·	á r
á Ñà ‡à „
â û
â ¨ä ≥
ã ¡
ã ¡
ã »
ã ◊	å -	å 3
å ã
å è
å „ç ç Úé é é è ¡
ê Ù	ë O	ë ]
ë ∆
ë …
ë Œ
ë ”
ë ÿ
ë ›
ë ‡
ë „í Oí ∆í Œí ”í ÿí ›ì ì 
	ì ì f
ì ç
ì ë
ì Îì ¬ì “î î 	î gî i	î k	î yî {	î }
î ìî È	ï 	ï  	ï v
ï àñ pñ Çó ]ó …"	
hotspot"
_Z12get_group_idj"
_Z12get_local_idj"
_Z7barrierj"
llvm.fmuladd.f32*ó
rodinia-3.1-hotspot-hotspot.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282Å

wgsize
Ä

wgsize_log1p
âboA
 
transfer_bytes_log1p
âboA

devmap_label


transfer_bytes
ÄÄ¿