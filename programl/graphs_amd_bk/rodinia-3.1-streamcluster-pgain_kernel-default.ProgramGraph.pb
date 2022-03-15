

[external]
LcallBD
B
	full_text5
3
1%11 = tail call i64 @_Z13get_global_idj(i32 0) #4
6truncB-
+
	full_text

%12 = trunc i64 %11 to i32
#i64B

	full_text
	
i64 %11
5icmpB-
+
	full_text

%13 = icmp slt i32 %12, %6
#i32B

	full_text
	
i32 %12
9brB3
1
	full_text$
"
 br i1 %13, label %14, label %161
!i1B

	full_text


i1 %13
Mcall8BC
A
	full_text4
2
0%15 = tail call i64 @_Z12get_local_idj(i32 0) #4
8trunc8B-
+
	full_text

%16 = trunc i64 %15 to i32
%i648B

	full_text
	
i64 %15
5icmp8B+
)
	full_text

%17 = icmp eq i32 %16, 0
%i328B

	full_text
	
i32 %16
5icmp8B+
)
	full_text

%18 = icmp sgt i32 %7, 0
1and8B(
&
	full_text

%19 = and i1 %17, %18
#i18B

	full_text


i1 %17
#i18B

	full_text


i1 %18
:br8B2
0
	full_text#
!
br i1 %19, label %20, label %82
#i18B

	full_text


i1 %19
5sext8B+
)
	full_text

%21 = sext i32 %6 to i64
5zext8B+
)
	full_text

%22 = zext i32 %7 to i64
5add8B,
*
	full_text

%23 = add nsw i64 %22, -1
%i648B

	full_text
	
i64 %22
0and8B'
%
	full_text

%24 = and i64 %22, 3
%i648B

	full_text
	
i64 %22
6icmp8B,
*
	full_text

%25 = icmp ult i64 %23, 3
%i648B

	full_text
	
i64 %23
:br8B2
0
	full_text#
!
br i1 %25, label %65, label %26
#i18B

	full_text


i1 %25
6sub8B-
+
	full_text

%27 = sub nsw i64 %22, %24
%i648B

	full_text
	
i64 %22
%i648B

	full_text
	
i64 %24
'br8B

	full_text

br label %28
Bphi8B9
7
	full_text*
(
&%29 = phi i64 [ 0, %26 ], [ %62, %28 ]
%i648B

	full_text
	
i64 %62
Dphi8B;
9
	full_text,
*
(%30 = phi i64 [ %27, %26 ], [ %63, %28 ]
%i648B

	full_text
	
i64 %27
%i648B

	full_text
	
i64 %63
6mul8B-
+
	full_text

%31 = mul nsw i64 %29, %21
%i648B

	full_text
	
i64 %29
%i648B

	full_text
	
i64 %21
5add8B,
*
	full_text

%32 = add nsw i64 %31, %8
%i648B

	full_text
	
i64 %31
\getelementptr8BI
G
	full_text:
8
6%33 = getelementptr inbounds float, float* %1, i64 %32
%i648B

	full_text
	
i64 %32
@bitcast8B3
1
	full_text$
"
 %34 = bitcast float* %33 to i32*
+float*8B

	full_text


float* %33
Hload8B>
<
	full_text/
-
+%35 = load i32, i32* %34, align 4, !tbaa !8
'i32*8B

	full_text


i32* %34
\getelementptr8BI
G
	full_text:
8
6%36 = getelementptr inbounds float, float* %5, i64 %29
%i648B

	full_text
	
i64 %29
@bitcast8B3
1
	full_text$
"
 %37 = bitcast float* %36 to i32*
+float*8B

	full_text


float* %36
Hstore8B=
;
	full_text.
,
*store i32 %35, i32* %37, align 4, !tbaa !8
%i328B

	full_text
	
i32 %35
'i32*8B

	full_text


i32* %37
.or8B&
$
	full_text

%38 = or i64 %29, 1
%i648B

	full_text
	
i64 %29
6mul8B-
+
	full_text

%39 = mul nsw i64 %38, %21
%i648B

	full_text
	
i64 %38
%i648B

	full_text
	
i64 %21
5add8B,
*
	full_text

%40 = add nsw i64 %39, %8
%i648B

	full_text
	
i64 %39
\getelementptr8BI
G
	full_text:
8
6%41 = getelementptr inbounds float, float* %1, i64 %40
%i648B

	full_text
	
i64 %40
@bitcast8B3
1
	full_text$
"
 %42 = bitcast float* %41 to i32*
+float*8B

	full_text


float* %41
Hload8B>
<
	full_text/
-
+%43 = load i32, i32* %42, align 4, !tbaa !8
'i32*8B

	full_text


i32* %42
\getelementptr8BI
G
	full_text:
8
6%44 = getelementptr inbounds float, float* %5, i64 %38
%i648B

	full_text
	
i64 %38
@bitcast8B3
1
	full_text$
"
 %45 = bitcast float* %44 to i32*
+float*8B

	full_text


float* %44
Hstore8B=
;
	full_text.
,
*store i32 %43, i32* %45, align 4, !tbaa !8
%i328B

	full_text
	
i32 %43
'i32*8B

	full_text


i32* %45
.or8B&
$
	full_text

%46 = or i64 %29, 2
%i648B

	full_text
	
i64 %29
6mul8B-
+
	full_text

%47 = mul nsw i64 %46, %21
%i648B

	full_text
	
i64 %46
%i648B

	full_text
	
i64 %21
5add8B,
*
	full_text

%48 = add nsw i64 %47, %8
%i648B

	full_text
	
i64 %47
\getelementptr8BI
G
	full_text:
8
6%49 = getelementptr inbounds float, float* %1, i64 %48
%i648B

	full_text
	
i64 %48
@bitcast8B3
1
	full_text$
"
 %50 = bitcast float* %49 to i32*
+float*8B

	full_text


float* %49
Hload8B>
<
	full_text/
-
+%51 = load i32, i32* %50, align 4, !tbaa !8
'i32*8B

	full_text


i32* %50
\getelementptr8BI
G
	full_text:
8
6%52 = getelementptr inbounds float, float* %5, i64 %46
%i648B

	full_text
	
i64 %46
@bitcast8B3
1
	full_text$
"
 %53 = bitcast float* %52 to i32*
+float*8B

	full_text


float* %52
Hstore8B=
;
	full_text.
,
*store i32 %51, i32* %53, align 4, !tbaa !8
%i328B

	full_text
	
i32 %51
'i32*8B

	full_text


i32* %53
.or8B&
$
	full_text

%54 = or i64 %29, 3
%i648B

	full_text
	
i64 %29
6mul8B-
+
	full_text

%55 = mul nsw i64 %54, %21
%i648B

	full_text
	
i64 %54
%i648B

	full_text
	
i64 %21
5add8B,
*
	full_text

%56 = add nsw i64 %55, %8
%i648B

	full_text
	
i64 %55
\getelementptr8BI
G
	full_text:
8
6%57 = getelementptr inbounds float, float* %1, i64 %56
%i648B

	full_text
	
i64 %56
@bitcast8B3
1
	full_text$
"
 %58 = bitcast float* %57 to i32*
+float*8B

	full_text


float* %57
Hload8B>
<
	full_text/
-
+%59 = load i32, i32* %58, align 4, !tbaa !8
'i32*8B

	full_text


i32* %58
\getelementptr8BI
G
	full_text:
8
6%60 = getelementptr inbounds float, float* %5, i64 %54
%i648B

	full_text
	
i64 %54
@bitcast8B3
1
	full_text$
"
 %61 = bitcast float* %60 to i32*
+float*8B

	full_text


float* %60
Hstore8B=
;
	full_text.
,
*store i32 %59, i32* %61, align 4, !tbaa !8
%i328B

	full_text
	
i32 %59
'i32*8B

	full_text


i32* %61
4add8B+
)
	full_text

%62 = add nsw i64 %29, 4
%i648B

	full_text
	
i64 %29
1add8B(
&
	full_text

%63 = add i64 %30, -4
%i648B

	full_text
	
i64 %30
5icmp8B+
)
	full_text

%64 = icmp eq i64 %63, 0
%i648B

	full_text
	
i64 %63
:br8B2
0
	full_text#
!
br i1 %64, label %65, label %28
#i18B

	full_text


i1 %64
Bphi8B9
7
	full_text*
(
&%66 = phi i64 [ 0, %20 ], [ %62, %28 ]
%i648B

	full_text
	
i64 %62
5icmp8B+
)
	full_text

%67 = icmp eq i64 %24, 0
%i648B

	full_text
	
i64 %24
:br8B2
0
	full_text#
!
br i1 %67, label %82, label %68
#i18B

	full_text


i1 %67
'br8B

	full_text

br label %69
Dphi8B;
9
	full_text,
*
(%70 = phi i64 [ %66, %68 ], [ %79, %69 ]
%i648B

	full_text
	
i64 %66
%i648B

	full_text
	
i64 %79
Dphi8B;
9
	full_text,
*
(%71 = phi i64 [ %24, %68 ], [ %80, %69 ]
%i648B

	full_text
	
i64 %24
%i648B

	full_text
	
i64 %80
6mul8B-
+
	full_text

%72 = mul nsw i64 %70, %21
%i648B

	full_text
	
i64 %70
%i648B

	full_text
	
i64 %21
5add8B,
*
	full_text

%73 = add nsw i64 %72, %8
%i648B

	full_text
	
i64 %72
\getelementptr8BI
G
	full_text:
8
6%74 = getelementptr inbounds float, float* %1, i64 %73
%i648B

	full_text
	
i64 %73
@bitcast8B3
1
	full_text$
"
 %75 = bitcast float* %74 to i32*
+float*8B

	full_text


float* %74
Hload8B>
<
	full_text/
-
+%76 = load i32, i32* %75, align 4, !tbaa !8
'i32*8B

	full_text


i32* %75
\getelementptr8BI
G
	full_text:
8
6%77 = getelementptr inbounds float, float* %5, i64 %70
%i648B

	full_text
	
i64 %70
@bitcast8B3
1
	full_text$
"
 %78 = bitcast float* %77 to i32*
+float*8B

	full_text


float* %77
Hstore8B=
;
	full_text.
,
*store i32 %76, i32* %78, align 4, !tbaa !8
%i328B

	full_text
	
i32 %76
'i32*8B

	full_text


i32* %78
8add8B/
-
	full_text 

%79 = add nuw nsw i64 %70, 1
%i648B

	full_text
	
i64 %70
1add8B(
&
	full_text

%80 = add i64 %71, -1
%i648B

	full_text
	
i64 %71
5icmp8B+
)
	full_text

%81 = icmp eq i64 %80, 0
%i648B

	full_text
	
i64 %80
Jbr8BB
@
	full_text3
1
/br i1 %81, label %82, label %69, !llvm.loop !12
#i18B

	full_text


i1 %81
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #5
;br8B3
1
	full_text$
"
 br i1 %18, label %83, label %106
#i18B

	full_text


i1 %18
5sext8	B+
)
	full_text

%84 = sext i32 %6 to i64
1shl8	B(
&
	full_text

%85 = shl i64 %11, 32
%i648	B

	full_text
	
i64 %11
9ashr8	B/
-
	full_text 

%86 = ashr exact i64 %85, 32
%i648	B

	full_text
	
i64 %85
5zext8	B+
)
	full_text

%87 = zext i32 %7 to i64
0and8	B'
%
	full_text

%88 = and i64 %87, 1
%i648	B

	full_text
	
i64 %87
4icmp8	B*
(
	full_text

%89 = icmp eq i32 %7, 1
:br8	B2
0
	full_text#
!
br i1 %89, label %92, label %90
#i18	B

	full_text


i1 %89
6sub8
B-
+
	full_text

%91 = sub nsw i64 %87, %88
%i648
B

	full_text
	
i64 %87
%i648
B

	full_text
	
i64 %88
(br8
B 

	full_text

br label %118
Jphi8BA
?
	full_text2
0
.%93 = phi float [ undef, %83 ], [ %138, %118 ]
*float8B

	full_text


float %138
Dphi8B;
9
	full_text,
*
(%94 = phi i64 [ 0, %83 ], [ %139, %118 ]
&i648B

	full_text


i64 %139
Qphi8BH
F
	full_text9
7
5%95 = phi float [ 0.000000e+00, %83 ], [ %138, %118 ]
*float8B

	full_text


float %138
5icmp8B+
)
	full_text

%96 = icmp eq i64 %88, 0
%i648B

	full_text
	
i64 %88
;br8B3
1
	full_text$
"
 br i1 %96, label %106, label %97
#i18B

	full_text


i1 %96
6mul8B-
+
	full_text

%98 = mul nsw i64 %94, %84
%i648B

	full_text
	
i64 %94
%i648B

	full_text
	
i64 %84
6add8B-
+
	full_text

%99 = add nsw i64 %98, %86
%i648B

	full_text
	
i64 %98
%i648B

	full_text
	
i64 %86
]getelementptr8BJ
H
	full_text;
9
7%100 = getelementptr inbounds float, float* %1, i64 %99
%i648B

	full_text
	
i64 %99
Nload8BD
B
	full_text5
3
1%101 = load float, float* %100, align 4, !tbaa !8
,float*8B

	full_text

float* %100
]getelementptr8BJ
H
	full_text;
9
7%102 = getelementptr inbounds float, float* %5, i64 %94
%i648B

	full_text
	
i64 %94
Nload8BD
B
	full_text5
3
1%103 = load float, float* %102, align 4, !tbaa !8
,float*8B

	full_text

float* %102
9fsub8B/
-
	full_text 

%104 = fsub float %101, %103
*float8B

	full_text


float %101
*float8B

	full_text


float %103
hcall8B^
\
	full_textO
M
K%105 = tail call float @llvm.fmuladd.f32(float %104, float %104, float %95)
*float8B

	full_text


float %104
*float8B

	full_text


float %104
)float8B

	full_text

	float %95
(br8B 

	full_text

br label %106
_phi8BV
T
	full_textG
E
C%107 = phi float [ 0.000000e+00, %82 ], [ %93, %92 ], [ %105, %97 ]
)float8B

	full_text

	float %93
*float8B

	full_text


float %105
2shl8B)
'
	full_text

%108 = shl i64 %11, 32
%i648B

	full_text
	
i64 %11
;ashr8B1
/
	full_text"
 
%109 = ashr exact i64 %108, 32
&i648B

	full_text


i64 %108
ƒgetelementptr8Bp
n
	full_texta
_
]%110 = getelementptr inbounds %struct.Point_Struct, %struct.Point_Struct* %0, i64 %109, i32 0
&i648B

	full_text


i64 %109
Oload8BE
C
	full_text6
4
2%111 = load float, float* %110, align 8, !tbaa !14
,float*8B

	full_text

float* %110
9fmul8B/
-
	full_text 

%112 = fmul float %107, %111
*float8B

	full_text


float %107
*float8B

	full_text


float %111
ƒgetelementptr8Bp
n
	full_texta
_
]%113 = getelementptr inbounds %struct.Point_Struct, %struct.Point_Struct* %0, i64 %109, i32 2
&i648B

	full_text


i64 %109
Oload8BE
C
	full_text6
4
2%114 = load float, float* %113, align 8, !tbaa !17
,float*8B

	full_text

float* %113
4add8B+
)
	full_text

%115 = add nsw i32 %9, 1
8mul8B/
-
	full_text 

%116 = mul nsw i32 %115, %12
&i328B

	full_text


i32 %115
%i328B

	full_text
	
i32 %12
=fcmp8B3
1
	full_text$
"
 %117 = fcmp olt float %112, %114
*float8B

	full_text


float %112
*float8B

	full_text


float %114
=br8B5
3
	full_text&
$
"br i1 %117, label %142, label %148
$i18B

	full_text
	
i1 %117
Ephi8B<
:
	full_text-
+
)%119 = phi i64 [ 0, %90 ], [ %139, %118 ]
&i648B

	full_text


i64 %139
Rphi8BI
G
	full_text:
8
6%120 = phi float [ 0.000000e+00, %90 ], [ %138, %118 ]
*float8B

	full_text


float %138
Gphi8B>
<
	full_text/
-
+%121 = phi i64 [ %91, %90 ], [ %140, %118 ]
%i648B

	full_text
	
i64 %91
&i648B

	full_text


i64 %140
8mul8B/
-
	full_text 

%122 = mul nsw i64 %119, %84
&i648B

	full_text


i64 %119
%i648B

	full_text
	
i64 %84
8add8B/
-
	full_text 

%123 = add nsw i64 %122, %86
&i648B

	full_text


i64 %122
%i648B

	full_text
	
i64 %86
^getelementptr8BK
I
	full_text<
:
8%124 = getelementptr inbounds float, float* %1, i64 %123
&i648B

	full_text


i64 %123
Nload8BD
B
	full_text5
3
1%125 = load float, float* %124, align 4, !tbaa !8
,float*8B

	full_text

float* %124
^getelementptr8BK
I
	full_text<
:
8%126 = getelementptr inbounds float, float* %5, i64 %119
&i648B

	full_text


i64 %119
Nload8BD
B
	full_text5
3
1%127 = load float, float* %126, align 4, !tbaa !8
,float*8B

	full_text

float* %126
9fsub8B/
-
	full_text 

%128 = fsub float %125, %127
*float8B

	full_text


float %125
*float8B

	full_text


float %127
icall8B_
]
	full_textP
N
L%129 = tail call float @llvm.fmuladd.f32(float %128, float %128, float %120)
*float8B

	full_text


float %128
*float8B

	full_text


float %128
*float8B

	full_text


float %120
0or8B(
&
	full_text

%130 = or i64 %119, 1
&i648B

	full_text


i64 %119
8mul8B/
-
	full_text 

%131 = mul nsw i64 %130, %84
&i648B

	full_text


i64 %130
%i648B

	full_text
	
i64 %84
8add8B/
-
	full_text 

%132 = add nsw i64 %131, %86
&i648B

	full_text


i64 %131
%i648B

	full_text
	
i64 %86
^getelementptr8BK
I
	full_text<
:
8%133 = getelementptr inbounds float, float* %1, i64 %132
&i648B

	full_text


i64 %132
Nload8BD
B
	full_text5
3
1%134 = load float, float* %133, align 4, !tbaa !8
,float*8B

	full_text

float* %133
^getelementptr8BK
I
	full_text<
:
8%135 = getelementptr inbounds float, float* %5, i64 %130
&i648B

	full_text


i64 %130
Nload8BD
B
	full_text5
3
1%136 = load float, float* %135, align 4, !tbaa !8
,float*8B

	full_text

float* %135
9fsub8B/
-
	full_text 

%137 = fsub float %134, %136
*float8B

	full_text


float %134
*float8B

	full_text


float %136
icall8B_
]
	full_textP
N
L%138 = tail call float @llvm.fmuladd.f32(float %137, float %137, float %129)
*float8B

	full_text


float %137
*float8B

	full_text


float %137
*float8B

	full_text


float %129
6add8B-
+
	full_text

%139 = add nsw i64 %119, 2
&i648B

	full_text


i64 %119
3add8B*
(
	full_text

%140 = add i64 %121, -2
&i648B

	full_text


i64 %121
7icmp8B-
+
	full_text

%141 = icmp eq i64 %140, 0
&i648B

	full_text


i64 %140
<br8B4
2
	full_text%
#
!br i1 %141, label %92, label %118
$i18B

	full_text
	
i1 %141
Xgetelementptr8BE
C
	full_text6
4
2%143 = getelementptr inbounds i8, i8* %4, i64 %109
&i648B

	full_text


i64 %109
Gstore8B<
:
	full_text-
+
)store i8 49, i8* %143, align 1, !tbaa !18
&i8*8B

	full_text


i8* %143
7add8B.
,
	full_text

%144 = add nsw i32 %116, %9
&i328B

	full_text


i32 %116
9fsub8B/
-
	full_text 

%145 = fsub float %112, %114
*float8B

	full_text


float %112
*float8B

	full_text


float %114
8sext8B.
,
	full_text

%146 = sext i32 %144 to i64
&i328B

	full_text


i32 %144
^getelementptr8BK
I
	full_text<
:
8%147 = getelementptr inbounds float, float* %2, i64 %146
&i648B

	full_text


i64 %146
Nstore8BC
A
	full_text4
2
0store float %145, float* %147, align 4, !tbaa !8
*float8B

	full_text


float %145
,float*8B

	full_text

float* %147
(br8B 

	full_text

br label %161
ƒgetelementptr8Bp
n
	full_texta
_
]%149 = getelementptr inbounds %struct.Point_Struct, %struct.Point_Struct* %0, i64 %109, i32 1
&i648B

	full_text


i64 %109
Kload8BA
?
	full_text2
0
.%150 = load i64, i64* %149, align 8, !tbaa !19
(i64*8B

	full_text

	i64* %149
3shl8B*
(
	full_text

%151 = shl i64 %150, 32
&i648B

	full_text


i64 %150
;ashr8B1
/
	full_text"
 
%152 = ashr exact i64 %151, 32
&i648B

	full_text


i64 %151
Zgetelementptr8BG
E
	full_text8
6
4%153 = getelementptr inbounds i32, i32* %3, i64 %152
&i648B

	full_text


i64 %152
Kload8BA
?
	full_text2
0
.%154 = load i32, i32* %153, align 4, !tbaa !20
(i32*8B

	full_text

	i32* %153
9add8B0
.
	full_text!

%155 = add nsw i32 %154, %116
&i328B

	full_text


i32 %154
&i328B

	full_text


i32 %116
9fsub8B/
-
	full_text 

%156 = fsub float %114, %112
*float8B

	full_text


float %114
*float8B

	full_text


float %112
8sext8B.
,
	full_text

%157 = sext i32 %155 to i64
&i328B

	full_text


i32 %155
^getelementptr8BK
I
	full_text<
:
8%158 = getelementptr inbounds float, float* %2, i64 %157
&i648B

	full_text


i64 %157
Nload8BD
B
	full_text5
3
1%159 = load float, float* %158, align 4, !tbaa !8
,float*8B

	full_text

float* %158
9fadd8B/
-
	full_text 

%160 = fadd float %156, %159
*float8B

	full_text


float %156
*float8B

	full_text


float %159
Nstore8BC
A
	full_text4
2
0store float %160, float* %158, align 4, !tbaa !8
*float8B

	full_text


float %160
,float*8B

	full_text

float* %158
(br8B 

	full_text

br label %161
$ret8B

	full_text


ret void
*float*8B

	full_text

	float* %2
&i32*8B

	full_text
	
i32* %3
$i328B

	full_text


i32 %7
$i328B

	full_text


i32 %6
*float*8B

	full_text

	float* %1
*float*8B

	full_text

	float* %5
$i328B

	full_text


i32 %9
$i8*8B

	full_text


i8* %4
$i648B

	full_text


i64 %8
:struct*8B+
)
	full_text

%struct.Point_Struct* %0
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
"i88B

	full_text	

i8 49
#i648B

	full_text	

i64 3
$i648B

	full_text


i64 -4
+float8B

	full_text

float undef
$i648B

	full_text


i64 32
#i328B

	full_text	

i32 0
#i648B

	full_text	

i64 1
#i648B

	full_text	

i64 0
$i648B

	full_text


i64 -2
#i648B

	full_text	

i64 2
#i328B

	full_text	

i32 1
#i648B

	full_text	

i64 4
2float8B%
#
	full_text

float 0.000000e+00
#i328B

	full_text	

i32 2
$i648B

	full_text


i64 -1       	
 		                     " !! #$ #% ## &' &( && )* )) +, ++ -. -- /0 // 12 11 34 33 56 57 55 89 88 :; :< :: => == ?@ ?? AB AA CD CC EF EE GH GG IJ IK II LM LL NO NP NN QR QQ ST SS UV UU WX WW YZ YY [\ [[ ]^ ]_ ]] `a `` bc bd bb ef ee gh gg ij ii kl kk mn mm op oo qr qs qq tu tt vw vv xy xx z{ z} || ~ ~~ € €„ ƒ
… ƒƒ †‡ †
ˆ †† ‰Š ‰
‹ ‰‰ Œ ŒŒ 
  ‘  ’“ ’’ ”
• ”” –— –– ˜™ ˜
š ˜˜ ›œ ››   Ÿ  ŸŸ ¡¢ ¡£ ¤¥ ¤¦ §¨ §§ ©ª ©© «« ¬­ ¬¬ ®® ¯° ¯² ±
³ ±± ´
¶ µµ ·
¸ ·· ¹
º ¹¹ »¼ »» ½¾ ½À ¿
Á ¿¿ ÂÃ Â
Ä ÂÂ Å
Æ ÅÅ ÇÈ ÇÇ É
Ê ÉÉ ËÌ ËË ÍÎ Í
Ï ÍÍ ĞÑ Ğ
Ò Ğ
Ó ĞĞ Ô
Ö Õ
× ÕÕ ØÙ ØØ ÚÛ ÚÚ Ü
İ ÜÜ Şß ŞŞ àá à
â àà ã
ä ãã åæ åå çç èé è
ê èè ëì ë
í ëë îï î
ñ ğğ ò
ó òò ôõ ô
ö ôô ÷ø ÷
ù ÷÷ úû ú
ü úú ı
ş ıı ÿ€ ÿÿ 
‚  ƒ„ ƒƒ …† …
‡ …… ˆ‰ ˆ
Š ˆ
‹ ˆˆ Œ ŒŒ  
  ‘’ ‘
“ ‘‘ ”
• ”” –— –– ˜
™ ˜˜ š› šš œ œ
 œœ Ÿ  Ÿ
¡ Ÿ
¢ ŸŸ £¤ ££ ¥¦ ¥¥ §¨ §§ ©ª ©
¬ «« ­
® ­­ ¯° ¯¯ ±² ±
³ ±± ´µ ´´ ¶
· ¶¶ ¸¹ ¸
º ¸¸ »
½ ¼¼ ¾¿ ¾¾ ÀÁ ÀÀ ÂÃ ÂÂ Ä
Å ÄÄ ÆÇ ÆÆ ÈÉ È
Ê ÈÈ ËÌ Ë
Í ËË ÎÏ ÎÎ Ğ
Ñ ĞĞ ÒÓ ÒÒ ÔÕ Ô
Ö ÔÔ ×Ø ×
Ù ×× ÚÜ ¶Ü Ğİ ÄŞ Ş Ş «Ş ®	ß ß ß ¦à +à ?à Sà gà à Åà ıà ”á 1á Eá Yá má ”á Éá á ˜â ç
â ¯ã «	ä )	ä =	ä Q	ä e
ä Œå Üå ãå ¼    
	          t " $v %! ' (& *) ,+ .- 0! 21 4/ 63 7! 98 ; <: >= @? BA D8 FE HC JG K! ML O PN RQ TS VU XL ZY \W ^[ _! a` c db fe hg ji l` nm pk ro s! u# wv yx {t } ~ | „› … ‡ ˆƒ Š ‹‰ Œ  ‘ “ƒ •” —’ ™– šƒ œ†   Ÿ ¢ ¥ ¨§ ª« ­® °« ²¬ ³Ÿ ¶£ ¸Ÿ º¬ ¼» ¾· À¦ Á¿ Ã© ÄÂ ÆÅ È· ÊÉ ÌÇ ÎË ÏÍ ÑÍ Ò¹ Óµ ÖĞ × ÙØ ÛÚ İÜ ßÕ áŞ âÚ äã æç é êà ìå íë ï£ ñŸ ó± õ¥ öğ ø¦ ù÷ û© üú şı €ğ ‚ „ÿ †ƒ ‡… ‰… Šò ‹ğ Œ ¦  ’© “‘ •” —Œ ™˜ ›– š œ  œ ¡ˆ ¢ğ ¤ô ¦¥ ¨§ ªÚ ¬« ®è °à ²å ³¯ µ´ ·± ¹¶ ºÚ ½¼ ¿¾ ÁÀ ÃÂ ÅÄ ÇÆ Éè Êå Ìà ÍÈ ÏÎ ÑĞ ÓË ÕÒ ÖÔ ØĞ Ù  Û  £ | ¤ ¦¤ Õ€ £€ ‚  !¯ µ¯ ±î «î ¼‚ ƒz |z !½ Õ½ ¿´ ğ» ÛÚ Û¡ £¡ ƒÔ Õ© µ© ğ èè éé Û çç ææ çç £ èè £Ğ éé ĞŸ éé Ÿˆ éé ˆ ææ ê ­	ë 	ë 	ë `	ì ví µ
î §
î ©
î Ø
î Ú
î À
î Âï ï 	ï 	ï 
ï Ü	ğ 8
ğ ›
ğ ¬
ğ Œñ !	ñ xñ |	ñ ~
ñ Ÿñ ·
ñ »ñ ğ
ñ §
ò ¥	ó L
ó £ô £
ô ®
ô ç
ô ¼	õ tö ¹ö Õö ò
÷ ã	ø 
ø "
pgain_kernel"
_Z13get_global_idj"
_Z12get_local_idj"
_Z7barrierj"
llvm.fmuladd.f32*¢
)rodinia-3.1-streamcluster-pgain_kernel.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02
 
transfer_bytes_log1p
n¢‘A

devmap_label
 

wgsize
€

wgsize_log1p
n¢‘A

transfer_bytes
€€´&