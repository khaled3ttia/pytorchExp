

[external]
KcallBC
A
	full_text4
2
0%9 = tail call i64 @_Z13get_global_idj(i32 0) #5
-addB&
$
	full_text

%10 = add i64 %9, 1
"i64B

	full_text


i64 %9
KcallBC
A
	full_text4
2
0%11 = tail call i64 @_Z12get_local_idj(i32 0) #5
8mulB1
/
	full_text"
 
%12 = mul i64 %11, 21474836480
#i64B

	full_text
	
i64 %11
7ashrB/
-
	full_text 

%13 = ashr exact i64 %12, 32
#i64B

	full_text
	
i64 %12
\getelementptrBK
I
	full_text<
:
8%14 = getelementptr inbounds double, double* %2, i64 %13
#i64B

	full_text
	
i64 %13
>bitcastB3
1
	full_text$
"
 %15 = bitcast double* %14 to i8*
+double*B

	full_text

double* %14
ccallB[
Y
	full_textL
J
Hcall void @llvm.memset.p0i8.i64(i8* align 8 %15, i8 0, i64 40, i1 false)
#i8*B

	full_text
	
i8* %15
6truncB-
+
	full_text

%16 = trunc i64 %10 to i32
#i64B

	full_text
	
i64 %10
6truncB-
+
	full_text

%17 = trunc i64 %11 to i32
#i64B

	full_text
	
i64 %11
6icmpB.
,
	full_text

%18 = icmp slt i32 %16, 101
#i32B

	full_text
	
i32 %16
8brB2
0
	full_text#
!
br i1 %18, label %19, label %74
!i1B

	full_text


i1 %18
Ybitcast8BL
J
	full_text=
;
9%20 = bitcast double* %0 to [103 x [103 x [5 x double]]]*
6icmp8B,
*
	full_text

%21 = icmp slt i32 %6, %7
:br8B2
0
	full_text#
!
br i1 %21, label %22, label %74
#i18B

	full_text


i1 %21
6icmp8B,
*
	full_text

%23 = icmp slt i32 %4, %5
1shl8B(
&
	full_text

%24 = shl i64 %10, 32
%i648B

	full_text
	
i64 %10
9ashr8B/
-
	full_text 

%25 = ashr exact i64 %24, 32
%i648B

	full_text
	
i64 %24
5sext8B+
)
	full_text

%26 = sext i32 %4 to i64
5sext8B+
)
	full_text

%27 = sext i32 %6 to i64
]getelementptr8BJ
H
	full_text;
9
7%28 = getelementptr inbounds double, double* %14, i64 1
-double*8B

	full_text

double* %14
]getelementptr8BJ
H
	full_text;
9
7%29 = getelementptr inbounds double, double* %14, i64 2
-double*8B

	full_text

double* %14
]getelementptr8BJ
H
	full_text;
9
7%30 = getelementptr inbounds double, double* %14, i64 3
-double*8B

	full_text

double* %14
]getelementptr8BJ
H
	full_text;
9
7%31 = getelementptr inbounds double, double* %14, i64 4
-double*8B

	full_text

double* %14
5sext8B+
)
	full_text

%32 = sext i32 %5 to i64
5sext8B+
)
	full_text

%33 = sext i32 %7 to i64
'br8B

	full_text

br label %34
Pphi8BG
E
	full_text8
6
4%35 = phi double [ 0.000000e+00, %22 ], [ %67, %66 ]
+double8B

	full_text


double %67
Pphi8BG
E
	full_text8
6
4%36 = phi double [ 0.000000e+00, %22 ], [ %68, %66 ]
+double8B

	full_text


double %68
Pphi8BG
E
	full_text8
6
4%37 = phi double [ 0.000000e+00, %22 ], [ %69, %66 ]
+double8B

	full_text


double %69
Pphi8BG
E
	full_text8
6
4%38 = phi double [ 0.000000e+00, %22 ], [ %70, %66 ]
+double8B

	full_text


double %70
Pphi8BG
E
	full_text8
6
4%39 = phi double [ 0.000000e+00, %22 ], [ %71, %66 ]
+double8B

	full_text


double %71
Dphi8B;
9
	full_text,
*
(%40 = phi i64 [ %27, %22 ], [ %72, %66 ]
%i648B

	full_text
	
i64 %27
%i648B

	full_text
	
i64 %72
:br8B2
0
	full_text#
!
br i1 %23, label %41, label %66
#i18B

	full_text


i1 %23
'br8B

	full_text

br label %42
Gphi8B>
<
	full_text/
-
+%43 = phi double [ %63, %42 ], [ %35, %41 ]
+double8B

	full_text


double %63
+double8B

	full_text


double %35
Gphi8B>
<
	full_text/
-
+%44 = phi double [ %60, %42 ], [ %36, %41 ]
+double8B

	full_text


double %60
+double8B

	full_text


double %36
Gphi8B>
<
	full_text/
-
+%45 = phi double [ %57, %42 ], [ %37, %41 ]
+double8B

	full_text


double %57
+double8B

	full_text


double %37
Gphi8B>
<
	full_text/
-
+%46 = phi double [ %54, %42 ], [ %38, %41 ]
+double8B

	full_text


double %54
+double8B

	full_text


double %38
Gphi8B>
<
	full_text/
-
+%47 = phi double [ %51, %42 ], [ %39, %41 ]
+double8B

	full_text


double %51
+double8B

	full_text


double %39
Dphi8B;
9
	full_text,
*
(%48 = phi i64 [ %64, %42 ], [ %26, %41 ]
%i648B

	full_text
	
i64 %64
%i648B

	full_text
	
i64 %26
®getelementptr8Bî
ë
	full_textÉ
Ä
~%49 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %20, i64 %25, i64 %40, i64 %48, i64 0
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %20
%i648B

	full_text
	
i64 %25
%i648B

	full_text
	
i64 %40
%i648B

	full_text
	
i64 %48
Nload8BD
B
	full_text5
3
1%50 = load double, double* %49, align 8, !tbaa !8
-double*8B

	full_text

double* %49
icall8B_
]
	full_textP
N
L%51 = tail call double @llvm.fmuladd.f64(double %50, double %50, double %47)
+double8B

	full_text


double %50
+double8B

	full_text


double %50
+double8B

	full_text


double %47
Nstore8BC
A
	full_text4
2
0store double %51, double* %14, align 8, !tbaa !8
+double8B

	full_text


double %51
-double*8B

	full_text

double* %14
®getelementptr8Bî
ë
	full_textÉ
Ä
~%52 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %20, i64 %25, i64 %40, i64 %48, i64 1
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %20
%i648B

	full_text
	
i64 %25
%i648B

	full_text
	
i64 %40
%i648B

	full_text
	
i64 %48
Nload8BD
B
	full_text5
3
1%53 = load double, double* %52, align 8, !tbaa !8
-double*8B

	full_text

double* %52
icall8B_
]
	full_textP
N
L%54 = tail call double @llvm.fmuladd.f64(double %53, double %53, double %46)
+double8B

	full_text


double %53
+double8B

	full_text


double %53
+double8B

	full_text


double %46
Nstore8BC
A
	full_text4
2
0store double %54, double* %28, align 8, !tbaa !8
+double8B

	full_text


double %54
-double*8B

	full_text

double* %28
®getelementptr8Bî
ë
	full_textÉ
Ä
~%55 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %20, i64 %25, i64 %40, i64 %48, i64 2
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %20
%i648B

	full_text
	
i64 %25
%i648B

	full_text
	
i64 %40
%i648B

	full_text
	
i64 %48
Nload8BD
B
	full_text5
3
1%56 = load double, double* %55, align 8, !tbaa !8
-double*8B

	full_text

double* %55
icall8B_
]
	full_textP
N
L%57 = tail call double @llvm.fmuladd.f64(double %56, double %56, double %45)
+double8B

	full_text


double %56
+double8B

	full_text


double %56
+double8B

	full_text


double %45
Nstore8BC
A
	full_text4
2
0store double %57, double* %29, align 8, !tbaa !8
+double8B

	full_text


double %57
-double*8B

	full_text

double* %29
®getelementptr8Bî
ë
	full_textÉ
Ä
~%58 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %20, i64 %25, i64 %40, i64 %48, i64 3
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %20
%i648B

	full_text
	
i64 %25
%i648B

	full_text
	
i64 %40
%i648B

	full_text
	
i64 %48
Nload8BD
B
	full_text5
3
1%59 = load double, double* %58, align 8, !tbaa !8
-double*8B

	full_text

double* %58
icall8B_
]
	full_textP
N
L%60 = tail call double @llvm.fmuladd.f64(double %59, double %59, double %44)
+double8B

	full_text


double %59
+double8B

	full_text


double %59
+double8B

	full_text


double %44
Nstore8BC
A
	full_text4
2
0store double %60, double* %30, align 8, !tbaa !8
+double8B

	full_text


double %60
-double*8B

	full_text

double* %30
®getelementptr8Bî
ë
	full_textÉ
Ä
~%61 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %20, i64 %25, i64 %40, i64 %48, i64 4
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %20
%i648B

	full_text
	
i64 %25
%i648B

	full_text
	
i64 %40
%i648B

	full_text
	
i64 %48
Nload8BD
B
	full_text5
3
1%62 = load double, double* %61, align 8, !tbaa !8
-double*8B

	full_text

double* %61
icall8B_
]
	full_textP
N
L%63 = tail call double @llvm.fmuladd.f64(double %62, double %62, double %43)
+double8B

	full_text


double %62
+double8B

	full_text


double %62
+double8B

	full_text


double %43
Nstore8BC
A
	full_text4
2
0store double %63, double* %31, align 8, !tbaa !8
+double8B

	full_text


double %63
-double*8B

	full_text

double* %31
4add8B+
)
	full_text

%64 = add nsw i64 %48, 1
%i648B

	full_text
	
i64 %48
7icmp8B-
+
	full_text

%65 = icmp eq i64 %64, %32
%i648B

	full_text
	
i64 %64
%i648B

	full_text
	
i64 %32
:br8B2
0
	full_text#
!
br i1 %65, label %66, label %42
#i18B

	full_text


i1 %65
Gphi8B>
<
	full_text/
-
+%67 = phi double [ %35, %34 ], [ %63, %42 ]
+double8B

	full_text


double %35
+double8B

	full_text


double %63
Gphi8B>
<
	full_text/
-
+%68 = phi double [ %36, %34 ], [ %60, %42 ]
+double8B

	full_text


double %36
+double8B

	full_text


double %60
Gphi8B>
<
	full_text/
-
+%69 = phi double [ %37, %34 ], [ %57, %42 ]
+double8B

	full_text


double %37
+double8B

	full_text


double %57
Gphi8B>
<
	full_text/
-
+%70 = phi double [ %38, %34 ], [ %54, %42 ]
+double8B

	full_text


double %38
+double8B

	full_text


double %54
Gphi8B>
<
	full_text/
-
+%71 = phi double [ %39, %34 ], [ %51, %42 ]
+double8B

	full_text


double %39
+double8B

	full_text


double %51
4add8B+
)
	full_text

%72 = add nsw i64 %40, 1
%i648B

	full_text
	
i64 %40
7icmp8B-
+
	full_text

%73 = icmp eq i64 %72, %33
%i648B

	full_text
	
i64 %72
%i648B

	full_text
	
i64 %33
:br8B2
0
	full_text#
!
br i1 %73, label %74, label %34
#i18B

	full_text


i1 %73
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #6
5icmp8B+
)
	full_text

%75 = icmp eq i32 %17, 0
%i328B

	full_text
	
i32 %17
;br8B3
1
	full_text$
"
 br i1 %75, label %76, label %149
#i18B

	full_text


i1 %75
Ocall8BE
C
	full_text6
4
2%77 = tail call i64 @_Z14get_local_sizej(i32 0) #5
6icmp8B,
*
	full_text

%78 = icmp ugt i64 %77, 1
%i648B

	full_text
	
i64 %77
:br8B2
0
	full_text#
!
br i1 %78, label %86, label %79
#i18B

	full_text


i1 %78
Abitcast8	B4
2
	full_text%
#
!%80 = bitcast double* %14 to i64*
-double*8	B

	full_text

double* %14
Hload8	B>
<
	full_text/
-
+%81 = load i64, i64* %80, align 8, !tbaa !8
'i64*8	B

	full_text


i64* %80
]getelementptr8	BJ
H
	full_text;
9
7%82 = getelementptr inbounds double, double* %14, i64 1
-double*8	B

	full_text

double* %14
]getelementptr8	BJ
H
	full_text;
9
7%83 = getelementptr inbounds double, double* %14, i64 2
-double*8	B

	full_text

double* %14
]getelementptr8	BJ
H
	full_text;
9
7%84 = getelementptr inbounds double, double* %14, i64 3
-double*8	B

	full_text

double* %14
]getelementptr8	BJ
H
	full_text;
9
7%85 = getelementptr inbounds double, double* %14, i64 4
-double*8	B

	full_text

double* %14
(br8	B 

	full_text

br label %123
Nload8
BD
B
	full_text5
3
1%87 = load double, double* %14, align 8, !tbaa !8
-double*8
B

	full_text

double* %14
]getelementptr8
BJ
H
	full_text;
9
7%88 = getelementptr inbounds double, double* %14, i64 1
-double*8
B

	full_text

double* %14
Nload8
BD
B
	full_text5
3
1%89 = load double, double* %88, align 8, !tbaa !8
-double*8
B

	full_text

double* %88
]getelementptr8
BJ
H
	full_text;
9
7%90 = getelementptr inbounds double, double* %14, i64 2
-double*8
B

	full_text

double* %14
Nload8
BD
B
	full_text5
3
1%91 = load double, double* %90, align 8, !tbaa !8
-double*8
B

	full_text

double* %90
]getelementptr8
BJ
H
	full_text;
9
7%92 = getelementptr inbounds double, double* %14, i64 3
-double*8
B

	full_text

double* %14
Nload8
BD
B
	full_text5
3
1%93 = load double, double* %92, align 8, !tbaa !8
-double*8
B

	full_text

double* %92
]getelementptr8
BJ
H
	full_text;
9
7%94 = getelementptr inbounds double, double* %14, i64 4
-double*8
B

	full_text

double* %14
Nload8
BD
B
	full_text5
3
1%95 = load double, double* %94, align 8, !tbaa !8
-double*8
B

	full_text

double* %94
'br8
B

	full_text

br label %96
Hphi8B?
=
	full_text0
.
,%97 = phi double [ %95, %86 ], [ %118, %96 ]
+double8B

	full_text


double %95
,double8B

	full_text

double %118
Hphi8B?
=
	full_text0
.
,%98 = phi double [ %93, %86 ], [ %115, %96 ]
+double8B

	full_text


double %93
,double8B

	full_text

double %115
Hphi8B?
=
	full_text0
.
,%99 = phi double [ %91, %86 ], [ %112, %96 ]
+double8B

	full_text


double %91
,double8B

	full_text

double %112
Iphi8B@
>
	full_text1
/
-%100 = phi double [ %89, %86 ], [ %109, %96 ]
+double8B

	full_text


double %89
,double8B

	full_text

double %109
Iphi8B@
>
	full_text1
/
-%101 = phi double [ %87, %86 ], [ %106, %96 ]
+double8B

	full_text


double %87
,double8B

	full_text

double %106
Dphi8B;
9
	full_text,
*
(%102 = phi i64 [ 1, %86 ], [ %119, %96 ]
&i648B

	full_text


i64 %119
:mul8B1
/
	full_text"
 
%103 = mul nuw nsw i64 %102, 5
&i648B

	full_text


i64 %102
`getelementptr8BM
K
	full_text>
<
:%104 = getelementptr inbounds double, double* %2, i64 %103
&i648B

	full_text


i64 %103
Pload8BF
D
	full_text7
5
3%105 = load double, double* %104, align 8, !tbaa !8
.double*8B

	full_text

double* %104
:fadd8B0
.
	full_text!

%106 = fadd double %105, %101
,double8B

	full_text

double %105
,double8B

	full_text

double %101
Ostore8BD
B
	full_text5
3
1store double %106, double* %14, align 8, !tbaa !8
,double8B

	full_text

double %106
-double*8B

	full_text

double* %14
_getelementptr8BL
J
	full_text=
;
9%107 = getelementptr inbounds double, double* %104, i64 1
.double*8B

	full_text

double* %104
Pload8BF
D
	full_text7
5
3%108 = load double, double* %107, align 8, !tbaa !8
.double*8B

	full_text

double* %107
:fadd8B0
.
	full_text!

%109 = fadd double %108, %100
,double8B

	full_text

double %108
,double8B

	full_text

double %100
Ostore8BD
B
	full_text5
3
1store double %109, double* %88, align 8, !tbaa !8
,double8B

	full_text

double %109
-double*8B

	full_text

double* %88
_getelementptr8BL
J
	full_text=
;
9%110 = getelementptr inbounds double, double* %104, i64 2
.double*8B

	full_text

double* %104
Pload8BF
D
	full_text7
5
3%111 = load double, double* %110, align 8, !tbaa !8
.double*8B

	full_text

double* %110
9fadd8B/
-
	full_text 

%112 = fadd double %111, %99
,double8B

	full_text

double %111
+double8B

	full_text


double %99
Ostore8BD
B
	full_text5
3
1store double %112, double* %90, align 8, !tbaa !8
,double8B

	full_text

double %112
-double*8B

	full_text

double* %90
_getelementptr8BL
J
	full_text=
;
9%113 = getelementptr inbounds double, double* %104, i64 3
.double*8B

	full_text

double* %104
Pload8BF
D
	full_text7
5
3%114 = load double, double* %113, align 8, !tbaa !8
.double*8B

	full_text

double* %113
9fadd8B/
-
	full_text 

%115 = fadd double %114, %98
,double8B

	full_text

double %114
+double8B

	full_text


double %98
Ostore8BD
B
	full_text5
3
1store double %115, double* %92, align 8, !tbaa !8
,double8B

	full_text

double %115
-double*8B

	full_text

double* %92
_getelementptr8BL
J
	full_text=
;
9%116 = getelementptr inbounds double, double* %104, i64 4
.double*8B

	full_text

double* %104
Pload8BF
D
	full_text7
5
3%117 = load double, double* %116, align 8, !tbaa !8
.double*8B

	full_text

double* %116
9fadd8B/
-
	full_text 

%118 = fadd double %117, %97
,double8B

	full_text

double %117
+double8B

	full_text


double %97
Ostore8BD
B
	full_text5
3
1store double %118, double* %94, align 8, !tbaa !8
,double8B

	full_text

double %118
-double*8B

	full_text

double* %94
:add8B1
/
	full_text"
 
%119 = add nuw nsw i64 %102, 1
&i648B

	full_text


i64 %102
9icmp8B/
-
	full_text 

%120 = icmp eq i64 %119, %77
&i648B

	full_text


i64 %119
%i648B

	full_text
	
i64 %77
<br8B4
2
	full_text%
#
!br i1 %120, label %121, label %96
$i18B

	full_text
	
i1 %120
Abitcast8B4
2
	full_text%
#
!%122 = bitcast double %106 to i64
,double8B

	full_text

double %106
(br8B 

	full_text

br label %123
Jphi8BA
?
	full_text2
0
.%124 = phi double* [ %85, %79 ], [ %94, %121 ]
-double*8B

	full_text

double* %85
-double*8B

	full_text

double* %94
Jphi8BA
?
	full_text2
0
.%125 = phi double* [ %84, %79 ], [ %92, %121 ]
-double*8B

	full_text

double* %84
-double*8B

	full_text

double* %92
Jphi8BA
?
	full_text2
0
.%126 = phi double* [ %83, %79 ], [ %90, %121 ]
-double*8B

	full_text

double* %83
-double*8B

	full_text

double* %90
Jphi8BA
?
	full_text2
0
.%127 = phi double* [ %82, %79 ], [ %88, %121 ]
-double*8B

	full_text

double* %82
-double*8B

	full_text

double* %88
Gphi8B>
<
	full_text/
-
+%128 = phi i64 [ %81, %79 ], [ %122, %121 ]
%i648B

	full_text
	
i64 %81
&i648B

	full_text


i64 %122
Ncall8BD
B
	full_text5
3
1%129 = tail call i64 @_Z12get_group_idj(i32 0) #5
2mul8B)
'
	full_text

%130 = mul i64 %129, 5
&i648B

	full_text


i64 %129
`getelementptr8BM
K
	full_text>
<
:%131 = getelementptr inbounds double, double* %1, i64 %130
&i648B

	full_text


i64 %130
Cbitcast8B6
4
	full_text'
%
#%132 = bitcast double* %131 to i64*
.double*8B

	full_text

double* %131
Jstore8B?
=
	full_text0
.
,store i64 %128, i64* %132, align 8, !tbaa !8
&i648B

	full_text


i64 %128
(i64*8B

	full_text

	i64* %132
Cbitcast8B6
4
	full_text'
%
#%133 = bitcast double* %127 to i64*
.double*8B

	full_text

double* %127
Jload8B@
>
	full_text1
/
-%134 = load i64, i64* %133, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %133
_getelementptr8BL
J
	full_text=
;
9%135 = getelementptr inbounds double, double* %131, i64 1
.double*8B

	full_text

double* %131
Cbitcast8B6
4
	full_text'
%
#%136 = bitcast double* %135 to i64*
.double*8B

	full_text

double* %135
Jstore8B?
=
	full_text0
.
,store i64 %134, i64* %136, align 8, !tbaa !8
&i648B

	full_text


i64 %134
(i64*8B

	full_text

	i64* %136
Cbitcast8B6
4
	full_text'
%
#%137 = bitcast double* %126 to i64*
.double*8B

	full_text

double* %126
Jload8B@
>
	full_text1
/
-%138 = load i64, i64* %137, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %137
_getelementptr8BL
J
	full_text=
;
9%139 = getelementptr inbounds double, double* %131, i64 2
.double*8B

	full_text

double* %131
Cbitcast8B6
4
	full_text'
%
#%140 = bitcast double* %139 to i64*
.double*8B

	full_text

double* %139
Jstore8B?
=
	full_text0
.
,store i64 %138, i64* %140, align 8, !tbaa !8
&i648B

	full_text


i64 %138
(i64*8B

	full_text

	i64* %140
Cbitcast8B6
4
	full_text'
%
#%141 = bitcast double* %125 to i64*
.double*8B

	full_text

double* %125
Jload8B@
>
	full_text1
/
-%142 = load i64, i64* %141, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %141
_getelementptr8BL
J
	full_text=
;
9%143 = getelementptr inbounds double, double* %131, i64 3
.double*8B

	full_text

double* %131
Cbitcast8B6
4
	full_text'
%
#%144 = bitcast double* %143 to i64*
.double*8B

	full_text

double* %143
Jstore8B?
=
	full_text0
.
,store i64 %142, i64* %144, align 8, !tbaa !8
&i648B

	full_text


i64 %142
(i64*8B

	full_text

	i64* %144
Cbitcast8B6
4
	full_text'
%
#%145 = bitcast double* %124 to i64*
.double*8B

	full_text

double* %124
Jload8B@
>
	full_text1
/
-%146 = load i64, i64* %145, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %145
_getelementptr8BL
J
	full_text=
;
9%147 = getelementptr inbounds double, double* %131, i64 4
.double*8B

	full_text

double* %131
Cbitcast8B6
4
	full_text'
%
#%148 = bitcast double* %147 to i64*
.double*8B

	full_text

double* %147
Jstore8B?
=
	full_text0
.
,store i64 %146, i64* %148, align 8, !tbaa !8
&i648B

	full_text


i64 %146
(i64*8B

	full_text

	i64* %148
(br8B 

	full_text

br label %149
$ret8B

	full_text


ret void
,double*8B

	full_text


double* %0
,double*8B

	full_text


double* %2
$i328B

	full_text


i32 %6
$i328B

	full_text


i32 %5
$i328B

	full_text


i32 %4
,double*8B

	full_text


double* %1
$i328B

	full_text


i32 %7
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
4double8B&
$
	full_text

double 0.000000e+00
$i648B

	full_text


i64 40
#i648B

	full_text	

i64 1
%i18B

	full_text


i1 false
#i648B

	full_text	

i64 2
#i328B

	full_text	

i32 0
#i648B

	full_text	

i64 5
-i648B"
 
	full_text

i64 21474836480
#i648B

	full_text	

i64 3
$i648B

	full_text


i64 32
#i328B

	full_text	

i32 1
#i648B

	full_text	

i64 4
%i328B

	full_text
	
i32 101
#i648B

	full_text	

i64 0
!i88B

	full_text

i8 0        	
 		                       !! "# "" $% $$ &' && () (( ** ++ ,. -- /0 // 12 11 34 33 56 55 78 79 77 :; :> =? == @A @B @@ CD CE CC FG FH FF IJ IK II LM LN LL OP OQ OR OS OO TU TT VW VX VY VV Z[ Z\ ZZ ]^ ]_ ]` ]a ]] bc bb de df dg dd hi hj hh kl km kn ko kk pq pp rs rt ru rr vw vx vv yz y{ y| y} yy ~ ~~ ÄÅ Ä
Ç Ä
É ÄÄ ÑÖ Ñ
Ü ÑÑ áà á
â á
ä á
ã áá åç åå éè é
ê é
ë éé íì í
î íí ïñ ïï óò ó
ô óó öõ öù ú
û úú ü† ü
° üü ¢£ ¢
§ ¢¢ •¶ •
ß •• ®© ®
™ ®® ´¨ ´´ ≠Æ ≠
Ø ≠≠ ∞± ∞≤ ≥¥ ≥≥ µ∂ µ∑ ∏π ∏∏ ∫ª ∫Ω ºº æø ææ ¿¡ ¿¿ ¬√ ¬¬ ƒ≈ ƒƒ ∆« ∆∆ »  …… ÀÃ ÀÀ ÕŒ ÕÕ œ– œœ —“ —— ”‘ ”” ’÷ ’’ ◊ÿ ◊◊ Ÿ⁄ ŸŸ €› ‹
ﬁ ‹‹ ﬂ‡ ﬂ
· ﬂﬂ ‚„ ‚
‰ ‚‚ ÂÊ Â
Á ÂÂ ËÈ Ë
Í ËË Î
Ï ÎÎ ÌÓ ÌÌ Ô
 ÔÔ ÒÚ ÒÒ ÛÙ Û
ı ÛÛ ˆ˜ ˆ
¯ ˆˆ ˘˙ ˘˘ ˚¸ ˚˚ ˝˛ ˝
ˇ ˝˝ ÄÅ Ä
Ç ÄÄ ÉÑ ÉÉ ÖÜ ÖÖ áà á
â áá äã ä
å ää çé çç èê èè ëí ë
ì ëë îï î
ñ îî óò óó ôö ôô õú õ
ù õõ ûü û
† ûû °¢ °° £§ £
• ££ ¶ß ¶© ®® ™¨ ´
≠ ´´ ÆØ Æ
∞ ÆÆ ±≤ ±
≥ ±± ¥µ ¥
∂ ¥¥ ∑∏ ∑
π ∑∑ ∫∫ ªº ªª Ω
æ ΩΩ ø¿ øø ¡¬ ¡
√ ¡¡ ƒ≈ ƒƒ ∆« ∆∆ »… »»  À    ÃÕ Ã
Œ ÃÃ œ– œœ —“ —— ”‘ ”” ’÷ ’’ ◊ÿ ◊
Ÿ ◊◊ ⁄€ ⁄⁄ ‹› ‹‹ ﬁﬂ ﬁﬁ ‡· ‡‡ ‚„ ‚
‰ ‚‚ ÂÊ ÂÂ ÁË ÁÁ ÈÍ ÈÈ ÎÏ ÎÎ ÌÓ Ì
Ô ÌÌ Ú Û 	Û ÔÙ Ù !	ı ı *ˆ ˆ  ˜ Ω	¯ ¯ +    
	         	 #	 %	 '	 )ú .ü 0¢ 2• 4® 6! 8´ 9 ;é >- ?Ä A/ Br D1 Ed G3 HV J5 Kï M  N P Q7 RL SO UT WT XI YV [	 \ ^ _7 `L a] cb eb fF gd i" j l m7 nL ok qp sp tC ur w$ x z {7 |L }y ~ Å~ Ç@ ÉÄ Ö& Ü à â7 äL ãá çå èå ê= ëé ì( îL ñï ò* ôó õ- ùé û/ †Ä °1 £r §3 ¶d ß5 ©V ™7 ¨´ Æ+ Ø≠ ± ¥≥ ∂∑ π∏ ª	 Ωº ø	 ¡	 √	 ≈	 «	  	 ÃÀ Œ	 –œ “	 ‘” ÷	 ÿ◊ ⁄Ÿ ›õ ﬁ’ ‡ë ·— „á ‰Õ Ê˝ Á… ÈÛ Í° ÏÎ ÓÌ Ô ÚÒ ÙË ıÛ ˜	 ¯Ô ˙˘ ¸˚ ˛Â ˇ˝ ÅÀ ÇÔ ÑÉ ÜÖ à‚ âá ãœ åÔ éç êè íﬂ ìë ï” ñÔ òó öô ú‹ ùõ ü◊ †Î ¢° §∑ •£ ßÛ ©∆ ¨◊ ≠ƒ Ø” ∞¬ ≤œ ≥¿ µÀ ∂æ ∏® π∫ ºª æΩ ¿∑ ¬ø √¥ ≈ƒ «Ω …» À∆ Õ  Œ± –œ “Ω ‘” ÷— ÿ’ ŸÆ €⁄ ›Ω ﬂﬁ ·‹ „‡ ‰´ ÊÂ ËΩ ÍÈ ÏÁ ÓÎ Ô  ≤  ≤µ ∑µ Ò, -∫ …∫ º: <: ú€ ‹» ´< =∞ ≤∞ -¶ ®¶ ‹ Òö úö =™ ´ Ò ˙˙ ˘˘ ˚˚ ˇˇ ˝˝ ¸¸ ˛˛≤ ¸¸ ≤∫ ˛˛ ∫r ˚˚ r∑ ˝˝ ∑ ˙˙  ˇˇ Ä ˚˚ Äé ˚˚ éV ˚˚ Vd ˚˚ d ˘˘ Ä -Ä /Ä 1Ä 3Ä 5	Å 	Ç 	Ç "	Ç ]
Ç ï
Ç ´
Ç ∏
Ç ¿
Ç ÀÇ Î
Ç ˘
Ç °
Ç »	É 	Ñ $	Ñ k
Ñ ¬
Ñ œ
Ñ É
Ñ ”Ö Ö 
Ö ≥Ö ∑Ö ∫
Ü Ì
Ü ª	á 	à &	à y
à ƒ
à ”
à ç
à ﬁ	â 	â 	â ä ≤	ã (
ã á
ã ∆
ã ◊
ã ó
ã È	å 	ç O	é "
l2norm"
_Z13get_global_idj"
_Z12get_local_idj"
llvm.fmuladd.f64"
_Z7barrierj"
_Z14get_local_sizej"
_Z12get_group_idj"
llvm.memset.p0i8.i64*â
npb-LU-l2norm.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282Ä
 
transfer_bytes_log1p
⁄}òA

devmap_label
 

wgsize


transfer_bytes
»ê¿Z

wgsize_log1p
⁄}òA