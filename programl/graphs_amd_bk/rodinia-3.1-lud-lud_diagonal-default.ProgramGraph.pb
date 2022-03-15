

[external]
JcallBB
@
	full_text3
1
/%5 = tail call i64 @_Z12get_local_idj(i32 0) #4
4truncB+
)
	full_text

%6 = trunc i64 %5 to i32
"i64B

	full_text


i64 %5
1mulB*
(
	full_text

%7 = mul nsw i32 %3, %2
1addB*
(
	full_text

%8 = add nsw i32 %7, %3
"i32B

	full_text


i32 %7
2sextB*
(
	full_text

%9 = sext i32 %8 to i64
"i32B

	full_text


i32 %8
3sextB+
)
	full_text

%10 = sext i32 %2 to i64
.shlB'
%
	full_text

%11 = shl i64 %5, 32
"i64B

	full_text


i64 %5
7ashrB/
-
	full_text 

%12 = ashr exact i64 %11, 32
#i64B

	full_text
	
i64 %11
%brB

	full_text

br label %13
Aphi8B8
6
	full_text)
'
%%14 = phi i64 [ 0, %4 ], [ %35, %13 ]
%i648B

	full_text
	
i64 %35
Bphi8B9
7
	full_text*
(
&%15 = phi i64 [ %9, %4 ], [ %34, %13 ]
$i648B

	full_text


i64 %9
%i648B

	full_text
	
i64 %34
6add8B-
+
	full_text

%16 = add nsw i64 %15, %12
%i648B

	full_text
	
i64 %15
%i648B

	full_text
	
i64 %12
\getelementptr8BI
G
	full_text:
8
6%17 = getelementptr inbounds float, float* %0, i64 %16
%i648B

	full_text
	
i64 %16
@bitcast8B3
1
	full_text$
"
 %18 = bitcast float* %17 to i32*
+float*8B

	full_text


float* %17
Hload8B>
<
	full_text/
-
+%19 = load i32, i32* %18, align 4, !tbaa !8
'i32*8B

	full_text


i32* %18
0shl8B'
%
	full_text

%20 = shl i64 %14, 6
%i648B

	full_text
	
i64 %14
6add8B-
+
	full_text

%21 = add nsw i64 %20, %12
%i648B

	full_text
	
i64 %20
%i648B

	full_text
	
i64 %12
\getelementptr8BI
G
	full_text:
8
6%22 = getelementptr inbounds float, float* %1, i64 %21
%i648B

	full_text
	
i64 %21
@bitcast8B3
1
	full_text$
"
 %23 = bitcast float* %22 to i32*
+float*8B

	full_text


float* %22
Hstore8B=
;
	full_text.
,
*store i32 %19, i32* %23, align 4, !tbaa !8
%i328B

	full_text
	
i32 %19
'i32*8B

	full_text


i32* %23
6add8B-
+
	full_text

%24 = add nsw i64 %15, %10
%i648B

	full_text
	
i64 %15
%i648B

	full_text
	
i64 %10
6add8B-
+
	full_text

%25 = add nsw i64 %24, %12
%i648B

	full_text
	
i64 %24
%i648B

	full_text
	
i64 %12
\getelementptr8BI
G
	full_text:
8
6%26 = getelementptr inbounds float, float* %0, i64 %25
%i648B

	full_text
	
i64 %25
@bitcast8B3
1
	full_text$
"
 %27 = bitcast float* %26 to i32*
+float*8B

	full_text


float* %26
Hload8B>
<
	full_text/
-
+%28 = load i32, i32* %27, align 4, !tbaa !8
'i32*8B

	full_text


i32* %27
0shl8B'
%
	full_text

%29 = shl i64 %14, 6
%i648B

	full_text
	
i64 %14
/or8B'
%
	full_text

%30 = or i64 %29, 64
%i648B

	full_text
	
i64 %29
6add8B-
+
	full_text

%31 = add nsw i64 %30, %12
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %12
\getelementptr8BI
G
	full_text:
8
6%32 = getelementptr inbounds float, float* %1, i64 %31
%i648B

	full_text
	
i64 %31
@bitcast8B3
1
	full_text$
"
 %33 = bitcast float* %32 to i32*
+float*8B

	full_text


float* %32
Hstore8B=
;
	full_text.
,
*store i32 %28, i32* %33, align 4, !tbaa !8
%i328B

	full_text
	
i32 %28
'i32*8B

	full_text


i32* %33
6add8B-
+
	full_text

%34 = add nsw i64 %24, %10
%i648B

	full_text
	
i64 %24
%i648B

	full_text
	
i64 %10
4add8B+
)
	full_text

%35 = add nsw i64 %14, 2
%i648B

	full_text
	
i64 %14
6icmp8B,
*
	full_text

%36 = icmp eq i64 %35, 64
%i648B

	full_text
	
i64 %35
:br8B2
0
	full_text#
!
br i1 %36, label %37, label %13
#i18B

	full_text


i1 %36
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #5
/shl8B&
$
	full_text

%38 = shl i32 %6, 6
$i328B

	full_text


i32 %6
6sext8B,
*
	full_text

%39 = sext i32 %38 to i64
%i328B

	full_text
	
i32 %38
'br8B

	full_text

br label %40
Dphi8B;
9
	full_text,
*
(%41 = phi i64 [ 0, %37 ], [ %153, %152 ]
&i648B

	full_text


i64 %153
0add8B'
%
	full_text

%42 = add i64 %41, 1
%i648B

	full_text
	
i64 %41
8icmp8B.
,
	full_text

%43 = icmp slt i64 %41, %12
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %12
;br8B3
1
	full_text$
"
 br i1 %43, label %44, label %100
#i18B

	full_text


i1 %43
5icmp8B+
)
	full_text

%45 = icmp eq i64 %41, 0
%i648B

	full_text
	
i64 %41
:add8B1
/
	full_text"
 
%46 = add nuw nsw i64 %41, %39
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %39
\getelementptr8BI
G
	full_text:
8
6%47 = getelementptr inbounds float, float* %1, i64 %46
%i648B

	full_text
	
i64 %46
:br8B2
0
	full_text#
!
br i1 %45, label %94, label %48
#i18B

	full_text


i1 %45
Lload8BB
@
	full_text3
1
/%49 = load float, float* %47, align 4, !tbaa !8
+float*8B

	full_text


float* %47
0and8B'
%
	full_text

%50 = and i64 %41, 1
%i648B

	full_text
	
i64 %41
5icmp8B+
)
	full_text

%51 = icmp eq i64 %41, 1
%i648B

	full_text
	
i64 %41
:br8B2
0
	full_text#
!
br i1 %51, label %80, label %52
#i18B

	full_text


i1 %51
2sub8B)
'
	full_text

%53 = sub i64 %41, %50
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %50
'br8B

	full_text

br label %54
Fphi8B=
;
	full_text.
,
*%55 = phi float [ %49, %52 ], [ %76, %54 ]
)float8B

	full_text

	float %49
)float8B

	full_text

	float %76
Bphi8B9
7
	full_text*
(
&%56 = phi i64 [ 0, %52 ], [ %77, %54 ]
%i648B

	full_text
	
i64 %77
Dphi8B;
9
	full_text,
*
(%57 = phi i64 [ %53, %52 ], [ %78, %54 ]
%i648B

	full_text
	
i64 %53
%i648B

	full_text
	
i64 %78
:add8B1
/
	full_text"
 
%58 = add nuw nsw i64 %56, %39
%i648B

	full_text
	
i64 %56
%i648B

	full_text
	
i64 %39
\getelementptr8BI
G
	full_text:
8
6%59 = getelementptr inbounds float, float* %1, i64 %58
%i648B

	full_text
	
i64 %58
Lload8BB
@
	full_text3
1
/%60 = load float, float* %59, align 4, !tbaa !8
+float*8B

	full_text


float* %59
0shl8B'
%
	full_text

%61 = shl i64 %56, 6
%i648B

	full_text
	
i64 %56
:add8B1
/
	full_text"
 
%62 = add nuw nsw i64 %61, %41
%i648B

	full_text
	
i64 %61
%i648B

	full_text
	
i64 %41
\getelementptr8BI
G
	full_text:
8
6%63 = getelementptr inbounds float, float* %1, i64 %62
%i648B

	full_text
	
i64 %62
Lload8BB
@
	full_text3
1
/%64 = load float, float* %63, align 4, !tbaa !8
+float*8B

	full_text


float* %63
@fsub8B6
4
	full_text'
%
#%65 = fsub float -0.000000e+00, %60
)float8B

	full_text

	float %60
ecall8B[
Y
	full_textL
J
H%66 = tail call float @llvm.fmuladd.f32(float %65, float %64, float %55)
)float8B

	full_text

	float %65
)float8B

	full_text

	float %64
)float8B

	full_text

	float %55
Lstore8BA
?
	full_text2
0
.store float %66, float* %47, align 4, !tbaa !8
)float8B

	full_text

	float %66
+float*8B

	full_text


float* %47
.or8B&
$
	full_text

%67 = or i64 %56, 1
%i648B

	full_text
	
i64 %56
:add8B1
/
	full_text"
 
%68 = add nuw nsw i64 %67, %39
%i648B

	full_text
	
i64 %67
%i648B

	full_text
	
i64 %39
\getelementptr8BI
G
	full_text:
8
6%69 = getelementptr inbounds float, float* %1, i64 %68
%i648B

	full_text
	
i64 %68
Lload8BB
@
	full_text3
1
/%70 = load float, float* %69, align 4, !tbaa !8
+float*8B

	full_text


float* %69
0shl8B'
%
	full_text

%71 = shl i64 %67, 6
%i648B

	full_text
	
i64 %67
:add8B1
/
	full_text"
 
%72 = add nuw nsw i64 %71, %41
%i648B

	full_text
	
i64 %71
%i648B

	full_text
	
i64 %41
\getelementptr8BI
G
	full_text:
8
6%73 = getelementptr inbounds float, float* %1, i64 %72
%i648B

	full_text
	
i64 %72
Lload8BB
@
	full_text3
1
/%74 = load float, float* %73, align 4, !tbaa !8
+float*8B

	full_text


float* %73
@fsub8B6
4
	full_text'
%
#%75 = fsub float -0.000000e+00, %70
)float8B

	full_text

	float %70
ecall8B[
Y
	full_textL
J
H%76 = tail call float @llvm.fmuladd.f32(float %75, float %74, float %66)
)float8B

	full_text

	float %75
)float8B

	full_text

	float %74
)float8B

	full_text

	float %66
Lstore8BA
?
	full_text2
0
.store float %76, float* %47, align 4, !tbaa !8
)float8B

	full_text

	float %76
+float*8B

	full_text


float* %47
4add8B+
)
	full_text

%77 = add nsw i64 %56, 2
%i648B

	full_text
	
i64 %56
1add8B(
&
	full_text

%78 = add i64 %57, -2
%i648B

	full_text
	
i64 %57
5icmp8B+
)
	full_text

%79 = icmp eq i64 %78, 0
%i648B

	full_text
	
i64 %78
:br8B2
0
	full_text#
!
br i1 %79, label %80, label %54
#i18B

	full_text


i1 %79
Fphi8B=
;
	full_text.
,
*%81 = phi float [ %49, %48 ], [ %76, %54 ]
)float8B

	full_text

	float %49
)float8B

	full_text

	float %76
Bphi8B9
7
	full_text*
(
&%82 = phi i64 [ 0, %48 ], [ %77, %54 ]
%i648B

	full_text
	
i64 %77
5icmp8B+
)
	full_text

%83 = icmp eq i64 %50, 0
%i648B

	full_text
	
i64 %50
:br8B2
0
	full_text#
!
br i1 %83, label %94, label %84
#i18B

	full_text


i1 %83
:add8	B1
/
	full_text"
 
%85 = add nuw nsw i64 %82, %39
%i648	B

	full_text
	
i64 %82
%i648	B

	full_text
	
i64 %39
\getelementptr8	BI
G
	full_text:
8
6%86 = getelementptr inbounds float, float* %1, i64 %85
%i648	B

	full_text
	
i64 %85
Lload8	BB
@
	full_text3
1
/%87 = load float, float* %86, align 4, !tbaa !8
+float*8	B

	full_text


float* %86
0shl8	B'
%
	full_text

%88 = shl i64 %82, 6
%i648	B

	full_text
	
i64 %82
:add8	B1
/
	full_text"
 
%89 = add nuw nsw i64 %88, %41
%i648	B

	full_text
	
i64 %88
%i648	B

	full_text
	
i64 %41
\getelementptr8	BI
G
	full_text:
8
6%90 = getelementptr inbounds float, float* %1, i64 %89
%i648	B

	full_text
	
i64 %89
Lload8	BB
@
	full_text3
1
/%91 = load float, float* %90, align 4, !tbaa !8
+float*8	B

	full_text


float* %90
@fsub8	B6
4
	full_text'
%
#%92 = fsub float -0.000000e+00, %87
)float8	B

	full_text

	float %87
ecall8	B[
Y
	full_textL
J
H%93 = tail call float @llvm.fmuladd.f32(float %92, float %91, float %81)
)float8	B

	full_text

	float %92
)float8	B

	full_text

	float %91
)float8	B

	full_text

	float %81
Lstore8	BA
?
	full_text2
0
.store float %93, float* %47, align 4, !tbaa !8
)float8	B

	full_text

	float %93
+float*8	B

	full_text


float* %47
'br8	B

	full_text

br label %94
9mul8
B0
.
	full_text!

%95 = mul nuw nsw i64 %41, 65
%i648
B

	full_text
	
i64 %41
\getelementptr8
BI
G
	full_text:
8
6%96 = getelementptr inbounds float, float* %1, i64 %95
%i648
B

	full_text
	
i64 %95
Lload8
BB
@
	full_text3
1
/%97 = load float, float* %96, align 4, !tbaa !8
+float*8
B

	full_text


float* %96
Lload8
BB
@
	full_text3
1
/%98 = load float, float* %47, align 4, !tbaa !8
+float*8
B

	full_text


float* %47
Cfdiv8
B9
7
	full_text*
(
&%99 = fdiv float %98, %97, !fpmath !12
)float8
B

	full_text

	float %98
)float8
B

	full_text

	float %97
Lstore8
BA
?
	full_text2
0
.store float %99, float* %47, align 4, !tbaa !8
)float8
B

	full_text

	float %99
+float*8
B

	full_text


float* %47
(br8
B 

	full_text

br label %100
Gphi8B>
<
	full_text/
-
+%101 = phi i1 [ false, %94 ], [ true, %40 ]
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #5
=br8B5
3
	full_text&
$
"br i1 %101, label %152, label %102
$i18B

	full_text
	
i1 %101
1shl8B(
&
	full_text

%103 = shl i64 %41, 6
%i648B

	full_text
	
i64 %41
;add8B2
0
	full_text#
!
%104 = add nuw nsw i64 %103, 64
&i648B

	full_text


i64 %103
8add8B/
-
	full_text 

%105 = add nsw i64 %104, %12
&i648B

	full_text


i64 %104
%i648B

	full_text
	
i64 %12
^getelementptr8BK
I
	full_text<
:
8%106 = getelementptr inbounds float, float* %1, i64 %105
&i648B

	full_text


i64 %105
Nload8BD
B
	full_text5
3
1%107 = load float, float* %106, align 4, !tbaa !8
,float*8B

	full_text

float* %106
1and8B(
&
	full_text

%108 = and i64 %42, 1
%i648B

	full_text
	
i64 %42
6icmp8B,
*
	full_text

%109 = icmp eq i64 %41, 0
%i648B

	full_text
	
i64 %41
=br8B5
3
	full_text&
$
"br i1 %109, label %138, label %110
$i18B

	full_text
	
i1 %109
4sub8B+
)
	full_text

%111 = sub i64 %42, %108
%i648B

	full_text
	
i64 %42
&i648B

	full_text


i64 %108
(br8B 

	full_text

br label %112
Kphi8BB
@
	full_text3
1
/%113 = phi float [ %107, %110 ], [ %134, %112 ]
*float8B

	full_text


float %107
*float8B

	full_text


float %134
Fphi8B=
;
	full_text.
,
*%114 = phi i64 [ 0, %110 ], [ %135, %112 ]
&i648B

	full_text


i64 %135
Iphi8B@
>
	full_text1
/
-%115 = phi i64 [ %111, %110 ], [ %136, %112 ]
&i648B

	full_text


i64 %111
&i648B

	full_text


i64 %136
=add8B4
2
	full_text%
#
!%116 = add nuw nsw i64 %114, %104
&i648B

	full_text


i64 %114
&i648B

	full_text


i64 %104
^getelementptr8BK
I
	full_text<
:
8%117 = getelementptr inbounds float, float* %1, i64 %116
&i648B

	full_text


i64 %116
Nload8BD
B
	full_text5
3
1%118 = load float, float* %117, align 4, !tbaa !8
,float*8B

	full_text

float* %117
2shl8B)
'
	full_text

%119 = shl i64 %114, 6
&i648B

	full_text


i64 %114
8add8B/
-
	full_text 

%120 = add nsw i64 %119, %12
&i648B

	full_text


i64 %119
%i648B

	full_text
	
i64 %12
^getelementptr8BK
I
	full_text<
:
8%121 = getelementptr inbounds float, float* %1, i64 %120
&i648B

	full_text


i64 %120
Nload8BD
B
	full_text5
3
1%122 = load float, float* %121, align 4, !tbaa !8
,float*8B

	full_text

float* %121
Bfsub8B8
6
	full_text)
'
%%123 = fsub float -0.000000e+00, %118
*float8B

	full_text


float %118
icall8B_
]
	full_textP
N
L%124 = tail call float @llvm.fmuladd.f32(float %123, float %122, float %113)
*float8B

	full_text


float %123
*float8B

	full_text


float %122
*float8B

	full_text


float %113
Nstore8BC
A
	full_text4
2
0store float %124, float* %106, align 4, !tbaa !8
*float8B

	full_text


float %124
,float*8B

	full_text

float* %106
0or8B(
&
	full_text

%125 = or i64 %114, 1
&i648B

	full_text


i64 %114
=add8B4
2
	full_text%
#
!%126 = add nuw nsw i64 %125, %104
&i648B

	full_text


i64 %125
&i648B

	full_text


i64 %104
^getelementptr8BK
I
	full_text<
:
8%127 = getelementptr inbounds float, float* %1, i64 %126
&i648B

	full_text


i64 %126
Nload8BD
B
	full_text5
3
1%128 = load float, float* %127, align 4, !tbaa !8
,float*8B

	full_text

float* %127
2shl8B)
'
	full_text

%129 = shl i64 %125, 6
&i648B

	full_text


i64 %125
8add8B/
-
	full_text 

%130 = add nsw i64 %129, %12
&i648B

	full_text


i64 %129
%i648B

	full_text
	
i64 %12
^getelementptr8BK
I
	full_text<
:
8%131 = getelementptr inbounds float, float* %1, i64 %130
&i648B

	full_text


i64 %130
Nload8BD
B
	full_text5
3
1%132 = load float, float* %131, align 4, !tbaa !8
,float*8B

	full_text

float* %131
Bfsub8B8
6
	full_text)
'
%%133 = fsub float -0.000000e+00, %128
*float8B

	full_text


float %128
icall8B_
]
	full_textP
N
L%134 = tail call float @llvm.fmuladd.f32(float %133, float %132, float %124)
*float8B

	full_text


float %133
*float8B

	full_text


float %132
*float8B

	full_text


float %124
Nstore8BC
A
	full_text4
2
0store float %134, float* %106, align 4, !tbaa !8
*float8B

	full_text


float %134
,float*8B

	full_text

float* %106
6add8B-
+
	full_text

%135 = add nsw i64 %114, 2
&i648B

	full_text


i64 %114
3add8B*
(
	full_text

%136 = add i64 %115, -2
&i648B

	full_text


i64 %115
7icmp8B-
+
	full_text

%137 = icmp eq i64 %136, 0
&i648B

	full_text


i64 %136
=br8B5
3
	full_text&
$
"br i1 %137, label %138, label %112
$i18B

	full_text
	
i1 %137
Kphi8BB
@
	full_text3
1
/%139 = phi float [ %107, %102 ], [ %134, %112 ]
*float8B

	full_text


float %107
*float8B

	full_text


float %134
Fphi8B=
;
	full_text.
,
*%140 = phi i64 [ 0, %102 ], [ %135, %112 ]
&i648B

	full_text


i64 %135
7icmp8B-
+
	full_text

%141 = icmp eq i64 %108, 0
&i648B

	full_text


i64 %108
=br8B5
3
	full_text&
$
"br i1 %141, label %152, label %142
$i18B

	full_text
	
i1 %141
=add8B4
2
	full_text%
#
!%143 = add nuw nsw i64 %140, %104
&i648B

	full_text


i64 %140
&i648B

	full_text


i64 %104
^getelementptr8BK
I
	full_text<
:
8%144 = getelementptr inbounds float, float* %1, i64 %143
&i648B

	full_text


i64 %143
Nload8BD
B
	full_text5
3
1%145 = load float, float* %144, align 4, !tbaa !8
,float*8B

	full_text

float* %144
2shl8B)
'
	full_text

%146 = shl i64 %140, 6
&i648B

	full_text


i64 %140
8add8B/
-
	full_text 

%147 = add nsw i64 %146, %12
&i648B

	full_text


i64 %146
%i648B

	full_text
	
i64 %12
^getelementptr8BK
I
	full_text<
:
8%148 = getelementptr inbounds float, float* %1, i64 %147
&i648B

	full_text


i64 %147
Nload8BD
B
	full_text5
3
1%149 = load float, float* %148, align 4, !tbaa !8
,float*8B

	full_text

float* %148
Bfsub8B8
6
	full_text)
'
%%150 = fsub float -0.000000e+00, %145
*float8B

	full_text


float %145
icall8B_
]
	full_textP
N
L%151 = tail call float @llvm.fmuladd.f32(float %150, float %149, float %139)
*float8B

	full_text


float %150
*float8B

	full_text


float %149
*float8B

	full_text


float %139
Nstore8BC
A
	full_text4
2
0store float %151, float* %106, align 4, !tbaa !8
*float8B

	full_text


float %151
,float*8B

	full_text

float* %106
(br8B 

	full_text

br label %152
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #5
9add8B0
.
	full_text!

%153 = add nuw nsw i64 %41, 1
%i648B

	full_text
	
i64 %41
8icmp8B.
,
	full_text

%154 = icmp eq i64 %153, 63
&i648B

	full_text


i64 %153
<br8B4
2
	full_text%
#
!br i1 %154, label %155, label %40
$i18B

	full_text
	
i1 %154
4add8B+
)
	full_text

%156 = add nsw i32 %3, 1
7mul8B.
,
	full_text

%157 = mul nsw i32 %156, %2
&i328B

	full_text


i32 %156
7add8B.
,
	full_text

%158 = add nsw i32 %157, %3
&i328B

	full_text


i32 %157
8sext8B.
,
	full_text

%159 = sext i32 %158 to i64
&i328B

	full_text


i32 %158
(br8B 

	full_text

br label %160
Fphi8B=
;
	full_text.
,
*%161 = phi i64 [ 1, %155 ], [ %192, %160 ]
&i648B

	full_text


i64 %192
Iphi8B@
>
	full_text1
/
-%162 = phi i64 [ %159, %155 ], [ %191, %160 ]
&i648B

	full_text


i64 %159
&i648B

	full_text


i64 %191
2shl8B)
'
	full_text

%163 = shl i64 %161, 6
&i648B

	full_text


i64 %161
8add8B/
-
	full_text 

%164 = add nsw i64 %163, %12
&i648B

	full_text


i64 %163
%i648B

	full_text
	
i64 %12
^getelementptr8BK
I
	full_text<
:
8%165 = getelementptr inbounds float, float* %1, i64 %164
&i648B

	full_text


i64 %164
Bbitcast8B5
3
	full_text&
$
"%166 = bitcast float* %165 to i32*
,float*8B

	full_text

float* %165
Jload8B@
>
	full_text1
/
-%167 = load i32, i32* %166, align 4, !tbaa !8
(i32*8B

	full_text

	i32* %166
8add8B/
-
	full_text 

%168 = add nsw i64 %162, %12
&i648B

	full_text


i64 %162
%i648B

	full_text
	
i64 %12
^getelementptr8BK
I
	full_text<
:
8%169 = getelementptr inbounds float, float* %0, i64 %168
&i648B

	full_text


i64 %168
Bbitcast8B5
3
	full_text&
$
"%170 = bitcast float* %169 to i32*
,float*8B

	full_text

float* %169
Jstore8B?
=
	full_text0
.
,store i32 %167, i32* %170, align 4, !tbaa !8
&i328B

	full_text


i32 %167
(i32*8B

	full_text

	i32* %170
8add8B/
-
	full_text 

%171 = add nsw i64 %162, %10
&i648B

	full_text


i64 %162
%i648B

	full_text
	
i64 %10
2shl8B)
'
	full_text

%172 = shl i64 %161, 6
&i648B

	full_text


i64 %161
3add8B*
(
	full_text

%173 = add i64 %172, 64
&i648B

	full_text


i64 %172
8add8B/
-
	full_text 

%174 = add nsw i64 %173, %12
&i648B

	full_text


i64 %173
%i648B

	full_text
	
i64 %12
^getelementptr8BK
I
	full_text<
:
8%175 = getelementptr inbounds float, float* %1, i64 %174
&i648B

	full_text


i64 %174
Bbitcast8B5
3
	full_text&
$
"%176 = bitcast float* %175 to i32*
,float*8B

	full_text

float* %175
Jload8B@
>
	full_text1
/
-%177 = load i32, i32* %176, align 4, !tbaa !8
(i32*8B

	full_text

	i32* %176
8add8B/
-
	full_text 

%178 = add nsw i64 %171, %12
&i648B

	full_text


i64 %171
%i648B

	full_text
	
i64 %12
^getelementptr8BK
I
	full_text<
:
8%179 = getelementptr inbounds float, float* %0, i64 %178
&i648B

	full_text


i64 %178
Bbitcast8B5
3
	full_text&
$
"%180 = bitcast float* %179 to i32*
,float*8B

	full_text

float* %179
Jstore8B?
=
	full_text0
.
,store i32 %177, i32* %180, align 4, !tbaa !8
&i328B

	full_text


i32 %177
(i32*8B

	full_text

	i32* %180
8add8B/
-
	full_text 

%181 = add nsw i64 %171, %10
&i648B

	full_text


i64 %171
%i648B

	full_text
	
i64 %10
2shl8B)
'
	full_text

%182 = shl i64 %161, 6
&i648B

	full_text


i64 %161
4add8B+
)
	full_text

%183 = add i64 %182, 128
&i648B

	full_text


i64 %182
8add8B/
-
	full_text 

%184 = add nsw i64 %183, %12
&i648B

	full_text


i64 %183
%i648B

	full_text
	
i64 %12
^getelementptr8BK
I
	full_text<
:
8%185 = getelementptr inbounds float, float* %1, i64 %184
&i648B

	full_text


i64 %184
Bbitcast8B5
3
	full_text&
$
"%186 = bitcast float* %185 to i32*
,float*8B

	full_text

float* %185
Jload8B@
>
	full_text1
/
-%187 = load i32, i32* %186, align 4, !tbaa !8
(i32*8B

	full_text

	i32* %186
8add8B/
-
	full_text 

%188 = add nsw i64 %181, %12
&i648B

	full_text


i64 %181
%i648B

	full_text
	
i64 %12
^getelementptr8BK
I
	full_text<
:
8%189 = getelementptr inbounds float, float* %0, i64 %188
&i648B

	full_text


i64 %188
Bbitcast8B5
3
	full_text&
$
"%190 = bitcast float* %189 to i32*
,float*8B

	full_text

float* %189
Jstore8B?
=
	full_text0
.
,store i32 %187, i32* %190, align 4, !tbaa !8
&i328B

	full_text


i32 %187
(i32*8B

	full_text

	i32* %190
8add8B/
-
	full_text 

%191 = add nsw i64 %181, %10
&i648B

	full_text


i64 %181
%i648B

	full_text
	
i64 %10
6add8B-
+
	full_text

%192 = add nsw i64 %161, 3
&i648B

	full_text


i64 %161
8icmp8B.
,
	full_text

%193 = icmp eq i64 %192, 64
&i648B

	full_text


i64 %192
=br8B5
3
	full_text&
$
"br i1 %193, label %194, label %160
$i18B

	full_text
	
i1 %193
$ret8B

	full_text


ret void
$i328B

	full_text


i32 %3
*float*8B

	full_text

	float* %0
$i328B

	full_text


i32 %2
*float*8B
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
3float8B&
$
	full_text

float -0.000000e+00
$i648B

	full_text


i64 64
%i648B

	full_text
	
i64 128
#i328B

	full_text	

i32 1
$i648B

	full_text


i64 63
#i648B

	full_text	

i64 6
#i648B

	full_text	

i64 2
$i648B

	full_text


i64 -2
%i18B

	full_text


i1 false
$i648B

	full_text


i64 32
$i18B

	full_text
	
i1 true
$i648B

	full_text


i64 65
#i328B

	full_text	

i32 6
#i648B

	full_text	

i64 0
#i328B

	full_text	

i32 0
#i648B

	full_text	

i64 3
#i648B

	full_text	

i64 1        		 
 

                     !  "# "" $% $$ &' &( && )* )+ )) ,- ,. ,, /0 // 12 11 34 33 56 55 78 77 9: 9; 99 <= << >? >> @A @B @@ CD CE CC FG FF HI HH JK JL MN MM OP OO QS RR TU TT VW VX VV YZ Y\ [[ ]^ ]_ ]] `a `` bc be dd fg ff hi hh jk jm ln ll oq pr pp st ss uv uw uu xy xz xx {| {{ }~ }} Ä  ÅÇ Å
É ÅÅ Ñ
Ö ÑÑ Üá ÜÜ à
â àà äã ä
å ä
ç ää éè é
ê éé ëí ëë ìî ì
ï ìì ñ
ó ññ òô òò öõ öö úù ú
û úú ü
† üü °¢ °° £
§ ££ •¶ •
ß •
® •• ©™ ©
´ ©© ¨≠ ¨¨ ÆØ ÆÆ ∞± ∞∞ ≤≥ ≤µ ¥
∂ ¥¥ ∑
∏ ∑∑ π∫ ππ ªº ªæ Ω
ø ΩΩ ¿
¡ ¿¿ ¬√ ¬¬ ƒ≈ ƒƒ ∆« ∆
» ∆∆ …
  …… ÀÃ ÀÀ Õ
Œ ÕÕ œ– œ
— œ
“ œœ ”‘ ”
’ ”” ÷ÿ ◊◊ Ÿ
⁄ ŸŸ €‹ €€ ›ﬁ ›› ﬂ‡ ﬂ
· ﬂﬂ ‚„ ‚
‰ ‚‚ ÂÊ ÁÁ ËÈ ËÎ ÍÍ ÏÌ ÏÏ ÓÔ Ó
 ÓÓ Ò
Ú ÒÒ ÛÙ ÛÛ ıˆ ıı ˜¯ ˜˜ ˘˙ ˘¸ ˚
˝ ˚˚ ˛Ä ˇ
Å ˇˇ Ç
É ÇÇ ÑÖ Ñ
Ü ÑÑ áà á
â áá ä
ã ää åç åå éè éé êë ê
í êê ì
î ìì ïñ ïï ó
ò óó ôö ô
õ ô
ú ôô ùû ù
ü ùù †° †† ¢£ ¢
§ ¢¢ •
¶ •• ß® ßß ©™ ©© ´¨ ´
≠ ´´ Æ
Ø ÆÆ ∞± ∞∞ ≤
≥ ≤≤ ¥µ ¥
∂ ¥
∑ ¥¥ ∏π ∏
∫ ∏∏ ªº ªª Ωæ ΩΩ ø¿ øø ¡¬ ¡ƒ √
≈ √√ ∆
« ∆∆ »… »»  À  Õ Ã
Œ ÃÃ œ
– œœ —“ —— ”‘ ”” ’÷ ’
◊ ’’ ÿ
Ÿ ÿÿ ⁄€ ⁄⁄ ‹
› ‹‹ ﬁﬂ ﬁ
‡ ﬁ
· ﬁﬁ ‚„ ‚
‰ ‚‚ ÂÊ ÁË ÁÁ ÈÍ ÈÈ ÎÏ ÎÌ ÓÔ ÓÓ Ò  ÚÛ ÚÚ Ù
ˆ ıı ˜¯ ˜
˘ ˜˜ ˙˚ ˙˙ ¸˝ ¸
˛ ¸¸ ˇ
Ä ˇˇ ÅÇ ÅÅ ÉÑ ÉÉ ÖÜ Ö
á ÖÖ à
â àà äã ää åç å
é åå èê è
ë èè íì íí îï îî ñó ñ
ò ññ ô
ö ôô õú õõ ùû ùù ü† ü
° üü ¢
£ ¢¢ §• §§ ¶ß ¶
® ¶¶ ©™ ©
´ ©© ¨≠ ¨¨ ÆØ ÆÆ ∞± ∞
≤ ∞∞ ≥
¥ ≥≥ µ∂ µµ ∑∏ ∑∑ π∫ π
ª ππ º
Ω ºº æø ææ ¿¡ ¿
¬ ¿¿ √ƒ √
≈ √√ ∆« ∆∆ »… »»  À  Õ 	Õ Õ Ì
Õ Œ Œ /Œ àŒ ¢Œ º	œ œ 	
œ Ó– "– <– `– {– Ñ– ñ– ü– ¿– …– Ÿ– Ò– ä– ì– •– Æ– œ– ÿ– ˇ– ô– ≥    
 F  C          ! #" % '$ ( *	 +) - ., 0/ 21 4 65 87 : ;9 =< ?3 A> B) D	 E GF IH K NM PÁ SR UR W XV ZR \R ^O _] a[ c` eR gR ih kR mf nd q• r¨ tl vÆ ws yO zx |{ ~s Ä ÇR ÉÅ ÖÑ á} âà ãÜ åp çä è` ês íë îO ïì óñ ôë õö ùR ûú †ü ¢ò §£ ¶° ßä ®• ™` ´s ≠u ØÆ ±∞ ≥d µ• ∂¨ ∏f ∫π º∑ æO øΩ ¡¿ √∑ ≈ƒ «R »∆  … Ã¬ ŒÕ –À —¥ “œ ‘` ’R ÿ◊ ⁄Ÿ ‹` ﬁ› ‡€ ·ﬂ „` ‰Ê ÈR ÎÍ ÌÏ Ô Ó ÚÒ ÙT ˆR ¯˜ ˙T ¸ı ˝Û Ä¥ Åª É˚ ÖΩ ÜÇ àÏ âá ãä çÇ èé ë íê îì ñå òó öï õˇ úô ûÒ üÇ °† £Ï §¢ ¶• ®† ™© ¨ ≠´ ØÆ ±ß ≥≤ µ∞ ∂ô ∑¥ πÒ ∫Ç ºÑ æΩ ¿ø ¬Û ƒ¥ ≈ª «ı …» À∆ ÕÏ ŒÃ –œ “∆ ‘” ÷ ◊’ Ÿÿ €— ›‹ ﬂ⁄ ‡√ ·ﬁ „Ò ‰R ËÁ ÍÈ ÏÌ ÔÓ Ò Û∆ ˆÚ ¯√ ˘ı ˚˙ ˝ ˛¸ Äˇ ÇÅ Ñ˜ Ü áÖ âà ãÉ çä é˜ ê	 ëı ìí ïî ó òñ öô úõ ûè † °ü £¢ •ù ß§ ®è ™	 ´ı ≠¨ ØÆ ± ≤∞ ¥≥ ∂µ ∏© ∫ ªπ Ωº ø∑ ¡æ ¬© ƒ	 ≈ı «∆ …» À J LJ Q RY [Y Êb ◊b dË ÊË ÍÂ Êj ¥j lÎ ÌÎ R˘ √˘ ˚ª ◊ª Ωo pÙ ı  Ê  Ã˛ ˇ÷ ◊≤ ¥≤ p  Ã  ıÂ Ê¡ √¡ ˇ —— ““ ”” Ãä ”” ä —— ﬁ ”” ﬁô ”” ôÁ ““ ÁÊ ““ Êœ ”” œ• ”” •¥ ”” ¥L ““ L‘ à‘ £‘ Õ‘ ó‘ ≤‘ ‹	’ 7	’ H
’ Ï
’ î
’ »
÷ Æ◊ L◊ Á◊ Ê
◊ Ì
ÿ È	Ÿ 	Ÿ 5	Ÿ 
Ÿ ö
Ÿ ƒ
Ÿ Í
Ÿ é
Ÿ ©
Ÿ ”
Ÿ ˙
Ÿ í
Ÿ ¨	⁄ F
⁄ ¨
⁄ ª
€ Æ
€ Ω‹ Ê	› 
	› 
ﬁ Ê
ﬂ ◊	‡ M· · R	· [· s
· ∞· ∑
· π
· ˜· Ç
· ø· ∆
· »‚ 
„ ∆	‰ T	‰ f	‰ h
‰ ë
‰ ı
‰ †
‰ Á‰ ı"
lud_diagonal"
_Z12get_local_idj"
_Z7barrierj"
llvm.fmuladd.f32*ò
rodinia-3.1-lud-lud_diagonal.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282Ä

wgsize


transfer_bytes
ÄÄÄ

wgsize_log1p
·¸sA

devmap_label
 
 
transfer_bytes_log1p
·¸sA