

[external]
KcallBC
A
	full_text4
2
0%7 = tail call i64 @_Z13get_global_idj(i32 0) #5
,addB%
#
	full_text

%8 = add i64 %7, 1
"i64B

	full_text


i64 %7
JcallBB
@
	full_text3
1
/%9 = tail call i64 @_Z12get_local_idj(i32 0) #5
7mulB0
.
	full_text!

%10 = mul i64 %9, 21474836480
"i64B

	full_text


i64 %9
7ashrB/
-
	full_text 

%11 = ashr exact i64 %10, 32
#i64B

	full_text
	
i64 %10
\getelementptrBK
I
	full_text<
:
8%12 = getelementptr inbounds double, double* %2, i64 %11
#i64B

	full_text
	
i64 %11
>bitcastB3
1
	full_text$
"
 %13 = bitcast double* %12 to i8*
+double*B

	full_text

double* %12
ccallB[
Y
	full_textL
J
Hcall void @llvm.memset.p0i8.i64(i8* align 8 %13, i8 0, i64 40, i1 false)
#i8*B

	full_text
	
i8* %13
5truncB,
*
	full_text

%14 = trunc i64 %8 to i32
"i64B

	full_text


i64 %8
5truncB,
*
	full_text

%15 = trunc i64 %9 to i32
"i64B

	full_text


i64 %9
2addB+
)
	full_text

%16 = add nsw i32 %5, -2
6icmpB.
,
	full_text

%17 = icmp slt i32 %16, %14
#i32B

	full_text
	
i32 %16
#i32B

	full_text
	
i32 %14
8brB2
0
	full_text#
!
br i1 %17, label %73, label %18
!i1B

	full_text


i1 %17
Ybitcast8BL
J
	full_text=
;
9%19 = bitcast double* %0 to [103 x [103 x [5 x double]]]*
5icmp8B+
)
	full_text

%20 = icmp slt i32 %4, 3
:br8B2
0
	full_text#
!
br i1 %20, label %73, label %21
#i18B

	full_text


i1 %20
5icmp8B+
)
	full_text

%22 = icmp slt i32 %3, 3
0shl8B'
%
	full_text

%23 = shl i64 %8, 32
$i648B

	full_text


i64 %8
9ashr8B/
-
	full_text 

%24 = ashr exact i64 %23, 32
%i648B

	full_text
	
i64 %23
0add8B'
%
	full_text

%25 = add i32 %3, -1
0add8B'
%
	full_text

%26 = add i32 %4, -1
6zext8B,
*
	full_text

%27 = zext i32 %26 to i64
%i328B

	full_text
	
i32 %26
]getelementptr8BJ
H
	full_text;
9
7%28 = getelementptr inbounds double, double* %12, i64 1
-double*8B

	full_text

double* %12
]getelementptr8BJ
H
	full_text;
9
7%29 = getelementptr inbounds double, double* %12, i64 2
-double*8B

	full_text

double* %12
]getelementptr8BJ
H
	full_text;
9
7%30 = getelementptr inbounds double, double* %12, i64 3
-double*8B

	full_text

double* %12
]getelementptr8BJ
H
	full_text;
9
7%31 = getelementptr inbounds double, double* %12, i64 4
-double*8B

	full_text

double* %12
6zext8B,
*
	full_text

%32 = zext i32 %25 to i64
%i328B

	full_text
	
i32 %25
'br8B

	full_text

br label %33
Pphi8BG
E
	full_text8
6
4%34 = phi double [ %66, %65 ], [ 0.000000e+00, %21 ]
+double8B

	full_text


double %66
Pphi8BG
E
	full_text8
6
4%35 = phi double [ %67, %65 ], [ 0.000000e+00, %21 ]
+double8B

	full_text


double %67
Pphi8BG
E
	full_text8
6
4%36 = phi double [ %68, %65 ], [ 0.000000e+00, %21 ]
+double8B

	full_text


double %68
Pphi8BG
E
	full_text8
6
4%37 = phi double [ %69, %65 ], [ 0.000000e+00, %21 ]
+double8B

	full_text


double %69
Pphi8BG
E
	full_text8
6
4%38 = phi double [ %70, %65 ], [ 0.000000e+00, %21 ]
+double8B

	full_text


double %70
Bphi8B9
7
	full_text*
(
&%39 = phi i64 [ %71, %65 ], [ 1, %21 ]
%i648B

	full_text
	
i64 %71
:br8B2
0
	full_text#
!
br i1 %22, label %65, label %40
#i18B

	full_text


i1 %22
'br8B

	full_text

br label %41
Gphi8B>
<
	full_text/
-
+%42 = phi double [ %62, %41 ], [ %34, %40 ]
+double8B

	full_text


double %62
+double8B

	full_text


double %34
Gphi8B>
<
	full_text/
-
+%43 = phi double [ %59, %41 ], [ %35, %40 ]
+double8B

	full_text


double %59
+double8B

	full_text


double %35
Gphi8B>
<
	full_text/
-
+%44 = phi double [ %56, %41 ], [ %36, %40 ]
+double8B

	full_text


double %56
+double8B

	full_text


double %36
Gphi8B>
<
	full_text/
-
+%45 = phi double [ %53, %41 ], [ %37, %40 ]
+double8B

	full_text


double %53
+double8B

	full_text


double %37
Gphi8B>
<
	full_text/
-
+%46 = phi double [ %50, %41 ], [ %38, %40 ]
+double8B

	full_text


double %50
+double8B

	full_text


double %38
Bphi8B9
7
	full_text*
(
&%47 = phi i64 [ %63, %41 ], [ 1, %40 ]
%i648B

	full_text
	
i64 %63
®getelementptr8Bî
ë
	full_textÉ
Ä
~%48 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %19, i64 %24, i64 %39, i64 %47, i64 0
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %19
%i648B

	full_text
	
i64 %24
%i648B

	full_text
	
i64 %39
%i648B

	full_text
	
i64 %47
Nload8BD
B
	full_text5
3
1%49 = load double, double* %48, align 8, !tbaa !8
-double*8B

	full_text

double* %48
icall8B_
]
	full_textP
N
L%50 = tail call double @llvm.fmuladd.f64(double %49, double %49, double %46)
+double8B

	full_text


double %49
+double8B

	full_text


double %49
+double8B

	full_text


double %46
Nstore8BC
A
	full_text4
2
0store double %50, double* %12, align 8, !tbaa !8
+double8B

	full_text


double %50
-double*8B

	full_text

double* %12
®getelementptr8Bî
ë
	full_textÉ
Ä
~%51 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %19, i64 %24, i64 %39, i64 %47, i64 1
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %19
%i648B

	full_text
	
i64 %24
%i648B

	full_text
	
i64 %39
%i648B

	full_text
	
i64 %47
Nload8BD
B
	full_text5
3
1%52 = load double, double* %51, align 8, !tbaa !8
-double*8B

	full_text

double* %51
icall8B_
]
	full_textP
N
L%53 = tail call double @llvm.fmuladd.f64(double %52, double %52, double %45)
+double8B

	full_text


double %52
+double8B

	full_text


double %52
+double8B

	full_text


double %45
Nstore8BC
A
	full_text4
2
0store double %53, double* %28, align 8, !tbaa !8
+double8B

	full_text


double %53
-double*8B

	full_text

double* %28
®getelementptr8Bî
ë
	full_textÉ
Ä
~%54 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %19, i64 %24, i64 %39, i64 %47, i64 2
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %19
%i648B

	full_text
	
i64 %24
%i648B

	full_text
	
i64 %39
%i648B

	full_text
	
i64 %47
Nload8BD
B
	full_text5
3
1%55 = load double, double* %54, align 8, !tbaa !8
-double*8B

	full_text

double* %54
icall8B_
]
	full_textP
N
L%56 = tail call double @llvm.fmuladd.f64(double %55, double %55, double %44)
+double8B

	full_text


double %55
+double8B

	full_text


double %55
+double8B

	full_text


double %44
Nstore8BC
A
	full_text4
2
0store double %56, double* %29, align 8, !tbaa !8
+double8B

	full_text


double %56
-double*8B

	full_text

double* %29
®getelementptr8Bî
ë
	full_textÉ
Ä
~%57 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %19, i64 %24, i64 %39, i64 %47, i64 3
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %19
%i648B

	full_text
	
i64 %24
%i648B

	full_text
	
i64 %39
%i648B

	full_text
	
i64 %47
Nload8BD
B
	full_text5
3
1%58 = load double, double* %57, align 8, !tbaa !8
-double*8B

	full_text

double* %57
icall8B_
]
	full_textP
N
L%59 = tail call double @llvm.fmuladd.f64(double %58, double %58, double %43)
+double8B

	full_text


double %58
+double8B

	full_text


double %58
+double8B

	full_text


double %43
Nstore8BC
A
	full_text4
2
0store double %59, double* %30, align 8, !tbaa !8
+double8B

	full_text


double %59
-double*8B

	full_text

double* %30
®getelementptr8Bî
ë
	full_textÉ
Ä
~%60 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %19, i64 %24, i64 %39, i64 %47, i64 4
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %19
%i648B

	full_text
	
i64 %24
%i648B

	full_text
	
i64 %39
%i648B

	full_text
	
i64 %47
Nload8BD
B
	full_text5
3
1%61 = load double, double* %60, align 8, !tbaa !8
-double*8B

	full_text

double* %60
icall8B_
]
	full_textP
N
L%62 = tail call double @llvm.fmuladd.f64(double %61, double %61, double %42)
+double8B

	full_text


double %61
+double8B

	full_text


double %61
+double8B

	full_text


double %42
Nstore8BC
A
	full_text4
2
0store double %62, double* %31, align 8, !tbaa !8
+double8B

	full_text


double %62
-double*8B

	full_text

double* %31
8add8B/
-
	full_text 

%63 = add nuw nsw i64 %47, 1
%i648B

	full_text
	
i64 %47
7icmp8B-
+
	full_text

%64 = icmp eq i64 %63, %32
%i648B

	full_text
	
i64 %63
%i648B

	full_text
	
i64 %32
:br8B2
0
	full_text#
!
br i1 %64, label %65, label %41
#i18B

	full_text


i1 %64
Gphi8B>
<
	full_text/
-
+%66 = phi double [ %34, %33 ], [ %62, %41 ]
+double8B

	full_text


double %34
+double8B

	full_text


double %62
Gphi8B>
<
	full_text/
-
+%67 = phi double [ %35, %33 ], [ %59, %41 ]
+double8B

	full_text


double %35
+double8B

	full_text


double %59
Gphi8B>
<
	full_text/
-
+%68 = phi double [ %36, %33 ], [ %56, %41 ]
+double8B

	full_text


double %36
+double8B

	full_text


double %56
Gphi8B>
<
	full_text/
-
+%69 = phi double [ %37, %33 ], [ %53, %41 ]
+double8B

	full_text


double %37
+double8B

	full_text


double %53
Gphi8B>
<
	full_text/
-
+%70 = phi double [ %38, %33 ], [ %50, %41 ]
+double8B

	full_text


double %38
+double8B

	full_text


double %50
8add8B/
-
	full_text 

%71 = add nuw nsw i64 %39, 1
%i648B

	full_text
	
i64 %39
7icmp8B-
+
	full_text

%72 = icmp eq i64 %71, %27
%i648B

	full_text
	
i64 %71
%i648B

	full_text
	
i64 %27
:br8B2
0
	full_text#
!
br i1 %72, label %73, label %33
#i18B

	full_text


i1 %72
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #6
5icmp8B+
)
	full_text

%74 = icmp eq i32 %15, 0
%i328B

	full_text
	
i32 %15
;br8B3
1
	full_text$
"
 br i1 %74, label %75, label %148
#i18B

	full_text


i1 %74
Ocall8BE
C
	full_text6
4
2%76 = tail call i64 @_Z14get_local_sizej(i32 0) #5
6icmp8B,
*
	full_text

%77 = icmp ugt i64 %76, 1
%i648B

	full_text
	
i64 %76
:br8B2
0
	full_text#
!
br i1 %77, label %85, label %78
#i18B

	full_text


i1 %77
Abitcast8	B4
2
	full_text%
#
!%79 = bitcast double* %12 to i64*
-double*8	B

	full_text

double* %12
Hload8	B>
<
	full_text/
-
+%80 = load i64, i64* %79, align 8, !tbaa !8
'i64*8	B

	full_text


i64* %79
]getelementptr8	BJ
H
	full_text;
9
7%81 = getelementptr inbounds double, double* %12, i64 1
-double*8	B

	full_text

double* %12
]getelementptr8	BJ
H
	full_text;
9
7%82 = getelementptr inbounds double, double* %12, i64 2
-double*8	B

	full_text

double* %12
]getelementptr8	BJ
H
	full_text;
9
7%83 = getelementptr inbounds double, double* %12, i64 3
-double*8	B

	full_text

double* %12
]getelementptr8	BJ
H
	full_text;
9
7%84 = getelementptr inbounds double, double* %12, i64 4
-double*8	B

	full_text

double* %12
(br8	B 

	full_text

br label %122
Nload8
BD
B
	full_text5
3
1%86 = load double, double* %12, align 8, !tbaa !8
-double*8
B

	full_text

double* %12
]getelementptr8
BJ
H
	full_text;
9
7%87 = getelementptr inbounds double, double* %12, i64 1
-double*8
B

	full_text

double* %12
Nload8
BD
B
	full_text5
3
1%88 = load double, double* %87, align 8, !tbaa !8
-double*8
B

	full_text

double* %87
]getelementptr8
BJ
H
	full_text;
9
7%89 = getelementptr inbounds double, double* %12, i64 2
-double*8
B

	full_text

double* %12
Nload8
BD
B
	full_text5
3
1%90 = load double, double* %89, align 8, !tbaa !8
-double*8
B

	full_text

double* %89
]getelementptr8
BJ
H
	full_text;
9
7%91 = getelementptr inbounds double, double* %12, i64 3
-double*8
B

	full_text

double* %12
Nload8
BD
B
	full_text5
3
1%92 = load double, double* %91, align 8, !tbaa !8
-double*8
B

	full_text

double* %91
]getelementptr8
BJ
H
	full_text;
9
7%93 = getelementptr inbounds double, double* %12, i64 4
-double*8
B

	full_text

double* %12
Nload8
BD
B
	full_text5
3
1%94 = load double, double* %93, align 8, !tbaa !8
-double*8
B

	full_text

double* %93
'br8
B

	full_text

br label %95
Hphi8B?
=
	full_text0
.
,%96 = phi double [ %94, %85 ], [ %117, %95 ]
+double8B

	full_text


double %94
,double8B

	full_text

double %117
Hphi8B?
=
	full_text0
.
,%97 = phi double [ %92, %85 ], [ %114, %95 ]
+double8B

	full_text


double %92
,double8B

	full_text

double %114
Hphi8B?
=
	full_text0
.
,%98 = phi double [ %90, %85 ], [ %111, %95 ]
+double8B

	full_text


double %90
,double8B

	full_text

double %111
Hphi8B?
=
	full_text0
.
,%99 = phi double [ %88, %85 ], [ %108, %95 ]
+double8B

	full_text


double %88
,double8B

	full_text

double %108
Iphi8B@
>
	full_text1
/
-%100 = phi double [ %86, %85 ], [ %105, %95 ]
+double8B

	full_text


double %86
,double8B

	full_text

double %105
Dphi8B;
9
	full_text,
*
(%101 = phi i64 [ 1, %85 ], [ %118, %95 ]
&i648B

	full_text


i64 %118
:mul8B1
/
	full_text"
 
%102 = mul nuw nsw i64 %101, 5
&i648B

	full_text


i64 %101
`getelementptr8BM
K
	full_text>
<
:%103 = getelementptr inbounds double, double* %2, i64 %102
&i648B

	full_text


i64 %102
Pload8BF
D
	full_text7
5
3%104 = load double, double* %103, align 8, !tbaa !8
.double*8B

	full_text

double* %103
:fadd8B0
.
	full_text!

%105 = fadd double %104, %100
,double8B

	full_text

double %104
,double8B

	full_text

double %100
Ostore8BD
B
	full_text5
3
1store double %105, double* %12, align 8, !tbaa !8
,double8B

	full_text

double %105
-double*8B

	full_text

double* %12
_getelementptr8BL
J
	full_text=
;
9%106 = getelementptr inbounds double, double* %103, i64 1
.double*8B

	full_text

double* %103
Pload8BF
D
	full_text7
5
3%107 = load double, double* %106, align 8, !tbaa !8
.double*8B

	full_text

double* %106
9fadd8B/
-
	full_text 

%108 = fadd double %107, %99
,double8B

	full_text

double %107
+double8B

	full_text


double %99
Ostore8BD
B
	full_text5
3
1store double %108, double* %87, align 8, !tbaa !8
,double8B

	full_text

double %108
-double*8B

	full_text

double* %87
_getelementptr8BL
J
	full_text=
;
9%109 = getelementptr inbounds double, double* %103, i64 2
.double*8B

	full_text

double* %103
Pload8BF
D
	full_text7
5
3%110 = load double, double* %109, align 8, !tbaa !8
.double*8B

	full_text

double* %109
9fadd8B/
-
	full_text 

%111 = fadd double %110, %98
,double8B

	full_text

double %110
+double8B

	full_text


double %98
Ostore8BD
B
	full_text5
3
1store double %111, double* %89, align 8, !tbaa !8
,double8B

	full_text

double %111
-double*8B

	full_text

double* %89
_getelementptr8BL
J
	full_text=
;
9%112 = getelementptr inbounds double, double* %103, i64 3
.double*8B

	full_text

double* %103
Pload8BF
D
	full_text7
5
3%113 = load double, double* %112, align 8, !tbaa !8
.double*8B

	full_text

double* %112
9fadd8B/
-
	full_text 

%114 = fadd double %113, %97
,double8B

	full_text

double %113
+double8B

	full_text


double %97
Ostore8BD
B
	full_text5
3
1store double %114, double* %91, align 8, !tbaa !8
,double8B

	full_text

double %114
-double*8B

	full_text

double* %91
_getelementptr8BL
J
	full_text=
;
9%115 = getelementptr inbounds double, double* %103, i64 4
.double*8B

	full_text

double* %103
Pload8BF
D
	full_text7
5
3%116 = load double, double* %115, align 8, !tbaa !8
.double*8B

	full_text

double* %115
9fadd8B/
-
	full_text 

%117 = fadd double %116, %96
,double8B

	full_text

double %116
+double8B

	full_text


double %96
Ostore8BD
B
	full_text5
3
1store double %117, double* %93, align 8, !tbaa !8
,double8B

	full_text

double %117
-double*8B

	full_text

double* %93
:add8B1
/
	full_text"
 
%118 = add nuw nsw i64 %101, 1
&i648B

	full_text


i64 %101
9icmp8B/
-
	full_text 

%119 = icmp eq i64 %118, %76
&i648B

	full_text


i64 %118
%i648B

	full_text
	
i64 %76
<br8B4
2
	full_text%
#
!br i1 %119, label %120, label %95
$i18B

	full_text
	
i1 %119
Abitcast8B4
2
	full_text%
#
!%121 = bitcast double %105 to i64
,double8B

	full_text

double %105
(br8B 

	full_text

br label %122
Jphi8BA
?
	full_text2
0
.%123 = phi double* [ %84, %78 ], [ %93, %120 ]
-double*8B

	full_text

double* %84
-double*8B

	full_text

double* %93
Jphi8BA
?
	full_text2
0
.%124 = phi double* [ %83, %78 ], [ %91, %120 ]
-double*8B

	full_text

double* %83
-double*8B

	full_text

double* %91
Jphi8BA
?
	full_text2
0
.%125 = phi double* [ %82, %78 ], [ %89, %120 ]
-double*8B

	full_text

double* %82
-double*8B

	full_text

double* %89
Jphi8BA
?
	full_text2
0
.%126 = phi double* [ %81, %78 ], [ %87, %120 ]
-double*8B

	full_text

double* %81
-double*8B

	full_text

double* %87
Gphi8B>
<
	full_text/
-
+%127 = phi i64 [ %80, %78 ], [ %121, %120 ]
%i648B

	full_text
	
i64 %80
&i648B

	full_text


i64 %121
Ncall8BD
B
	full_text5
3
1%128 = tail call i64 @_Z12get_group_idj(i32 0) #5
2mul8B)
'
	full_text

%129 = mul i64 %128, 5
&i648B

	full_text


i64 %128
`getelementptr8BM
K
	full_text>
<
:%130 = getelementptr inbounds double, double* %1, i64 %129
&i648B

	full_text


i64 %129
Cbitcast8B6
4
	full_text'
%
#%131 = bitcast double* %130 to i64*
.double*8B

	full_text

double* %130
Jstore8B?
=
	full_text0
.
,store i64 %127, i64* %131, align 8, !tbaa !8
&i648B

	full_text


i64 %127
(i64*8B

	full_text

	i64* %131
Cbitcast8B6
4
	full_text'
%
#%132 = bitcast double* %126 to i64*
.double*8B

	full_text

double* %126
Jload8B@
>
	full_text1
/
-%133 = load i64, i64* %132, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %132
_getelementptr8BL
J
	full_text=
;
9%134 = getelementptr inbounds double, double* %130, i64 1
.double*8B

	full_text

double* %130
Cbitcast8B6
4
	full_text'
%
#%135 = bitcast double* %134 to i64*
.double*8B

	full_text

double* %134
Jstore8B?
=
	full_text0
.
,store i64 %133, i64* %135, align 8, !tbaa !8
&i648B

	full_text


i64 %133
(i64*8B

	full_text

	i64* %135
Cbitcast8B6
4
	full_text'
%
#%136 = bitcast double* %125 to i64*
.double*8B

	full_text

double* %125
Jload8B@
>
	full_text1
/
-%137 = load i64, i64* %136, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %136
_getelementptr8BL
J
	full_text=
;
9%138 = getelementptr inbounds double, double* %130, i64 2
.double*8B

	full_text

double* %130
Cbitcast8B6
4
	full_text'
%
#%139 = bitcast double* %138 to i64*
.double*8B

	full_text

double* %138
Jstore8B?
=
	full_text0
.
,store i64 %137, i64* %139, align 8, !tbaa !8
&i648B

	full_text


i64 %137
(i64*8B

	full_text

	i64* %139
Cbitcast8B6
4
	full_text'
%
#%140 = bitcast double* %124 to i64*
.double*8B

	full_text

double* %124
Jload8B@
>
	full_text1
/
-%141 = load i64, i64* %140, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %140
_getelementptr8BL
J
	full_text=
;
9%142 = getelementptr inbounds double, double* %130, i64 3
.double*8B

	full_text

double* %130
Cbitcast8B6
4
	full_text'
%
#%143 = bitcast double* %142 to i64*
.double*8B

	full_text

double* %142
Jstore8B?
=
	full_text0
.
,store i64 %141, i64* %143, align 8, !tbaa !8
&i648B

	full_text


i64 %141
(i64*8B

	full_text

	i64* %143
Cbitcast8B6
4
	full_text'
%
#%144 = bitcast double* %123 to i64*
.double*8B

	full_text

double* %123
Jload8B@
>
	full_text1
/
-%145 = load i64, i64* %144, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %144
_getelementptr8BL
J
	full_text=
;
9%146 = getelementptr inbounds double, double* %130, i64 4
.double*8B

	full_text

double* %130
Cbitcast8B6
4
	full_text'
%
#%147 = bitcast double* %146 to i64*
.double*8B

	full_text

double* %146
Jstore8B?
=
	full_text0
.
,store i64 %145, i64* %147, align 8, !tbaa !8
&i648B

	full_text


i64 %145
(i64*8B

	full_text

	i64* %147
(br8B 

	full_text

br label %148
$ret8B

	full_text


ret void
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
,double*8B

	full_text


double* %2
,double*8B

	full_text


double* %1
,double*8B

	full_text


double* %0
$i328B

	full_text


i32 %3
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
#i648B

	full_text	

i64 4
#i328B

	full_text	

i32 1
-i648B"
 
	full_text

i64 21474836480
$i328B

	full_text


i32 -1
#i648B

	full_text	

i64 2
!i88B

	full_text

i8 0
4double8B&
$
	full_text

double 0.000000e+00
#i648B

	full_text	

i64 1
$i328B

	full_text


i32 -2
#i328B

	full_text	

i32 3
#i648B

	full_text	

i64 0
#i648B

	full_text	

i64 5
#i328B

	full_text	

i32 0
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
$i648B

	full_text


i64 40
%i18B

	full_text


i1 false        	
 		                     !    "" ## $% $$ &' && () (( *+ ** ,- ,, ./ .. 02 11 34 33 56 55 78 77 9: 99 ;< ;; => =A @B @@ CD CE CC FG FH FF IJ IK II LM LN LL OP OO QR QS QT QU QQ VW VV XY XZ X[ XX \] \^ \\ _` _a _b _c __ de dd fg fh fi ff jk jl jj mn mo mp mq mm rs rr tu tv tw tt xy xz xx {| {} {~ { {{ ÄÅ ÄÄ ÇÉ Ç
Ñ Ç
Ö ÇÇ Üá Ü
à ÜÜ âä â
ã â
å â
ç ââ éè éé êë ê
í ê
ì êê îï î
ñ îî óò óó ôö ô
õ ôô úù úü û
† ûû °¢ °
£ °° §• §
¶ §§ ß® ß
© ßß ™´ ™
¨ ™™ ≠Æ ≠≠ Ø∞ Ø
± ØØ ≤≥ ≤¥ µ∂ µµ ∑∏ ∑π ∫ª ∫∫ ºΩ ºø ææ ¿¡ ¿¿ ¬√ ¬¬ ƒ≈ ƒƒ ∆« ∆∆ »… »»  Ã ÀÀ ÕŒ ÕÕ œ– œœ —“ —— ”‘ ”” ’÷ ’’ ◊ÿ ◊◊ Ÿ⁄ ŸŸ €‹ €€ ›ﬂ ﬁ
‡ ﬁﬁ ·‚ ·
„ ·· ‰Â ‰
Ê ‰‰ ÁË Á
È ÁÁ ÍÎ Í
Ï ÍÍ Ì
Ó ÌÌ Ô ÔÔ Ò
Ú ÒÒ ÛÙ ÛÛ ıˆ ı
˜ ıı ¯˘ ¯
˙ ¯¯ ˚¸ ˚˚ ˝˛ ˝˝ ˇÄ ˇ
Å ˇˇ ÇÉ Ç
Ñ ÇÇ ÖÜ ÖÖ áà áá âä â
ã ââ åç å
é åå èê èè ëí ëë ìî ì
ï ìì ñó ñ
ò ññ ôö ôô õú õõ ùû ù
ü ùù †° †
¢ †† £§ ££ •¶ •
ß •• ®© ®´ ™™ ¨Æ ≠
Ø ≠≠ ∞± ∞
≤ ∞∞ ≥¥ ≥
µ ≥≥ ∂∑ ∂
∏ ∂∂ π∫ π
ª ππ ºº Ωæ ΩΩ ø
¿ øø ¡¬ ¡¡ √ƒ √
≈ √√ ∆« ∆∆ »… »»  À    ÃÕ ÃÃ Œœ Œ
– ŒŒ —“ —— ”‘ ”” ’÷ ’’ ◊ÿ ◊◊ Ÿ⁄ Ÿ
€ ŸŸ ‹› ‹‹ ﬁﬂ ﬁﬁ ‡· ‡‡ ‚„ ‚‚ ‰Â ‰
Ê ‰‰ ÁË ÁÁ ÈÍ ÈÈ ÎÏ ÎÎ ÌÓ ÌÌ Ô Ô
Ò ÔÔ ÚÙ Ù #ı ˆ 	ˆ Ò˜ ø¯ ˘ ˘ "    
	          !# %	 '	 )	 +	 -" /û 2° 4§ 6ß 8™ :≠ < >ê A1 BÇ D3 Et G5 Hf J7 KX M9 Nó P R  S; TO UQ WV YV ZL [X ]	 ^ `  a; bO c_ ed gd hI if k& l n  o; pO qm sr ur vF wt y( z |  }; ~O { ÅÄ ÉÄ ÑC ÖÇ á* à ä  ã; åO çâ èé ëé í@ ìê ï, ñO òó ö. õô ù1 üê †3 ¢Ç £5 •t ¶7 ®f ©9 ´X ¨; Æ≠ ∞$ ±Ø ≥ ∂µ ∏π ª∫ Ω	 øæ ¡	 √	 ≈	 «	 …	 Ã	 ŒÕ –	 “— ‘	 ÷’ ÿ	 ⁄Ÿ ‹€ ﬂù ‡◊ ‚ì „” Ââ Êœ Ëˇ ÈÀ Îı Ï£ ÓÌ Ô ÚÒ ÙÛ ˆÍ ˜ı ˘	 ˙Ò ¸˚ ˛˝ ÄÁ Åˇ ÉÕ ÑÒ ÜÖ àá ä‰ ãâ ç— éÒ êè íë î· ïì ó’ òÒ öô úõ ûﬁ üù °Ÿ ¢Ì §£ ¶π ß• ©ı ´» ÆŸ Ø∆ ±’ ≤ƒ ¥— µ¬ ∑Õ ∏¿ ∫™ ªº æΩ ¿ø ¬π ƒ¡ ≈∂ «∆ …ø À  Õ» œÃ –≥ “— ‘ø ÷’ ÿ” ⁄◊ €∞ ›‹ ﬂø ·‡ „ﬁ Â‚ Ê≠ ËÁ Íø ÏÎ ÓÈ Ì Ò ¥ ∑ π∑ Û ¥ º Àº æ0 1› ﬁ  ≠= û= ?® ™® ﬁÚ Û≤ ¥≤ 1? @¨ ≠ú ûú @ ˛˛ ¸¸ ˝˝ Û ˙˙ ˇˇ ˚˚ ÄÄt ¸¸ tÇ ¸¸ Ç ÄÄ ¥ ˝˝ ¥X ¸¸ Xf ¸¸ fê ¸¸ ê ˙˙ º ˇˇ ºπ ˛˛ π ˚˚ 	Å ,
Å â
Å »
Å Ÿ
Å ô
Å ÎÇ ¥	É 	Ñ "	Ñ #	Ö (	Ö m
Ö ƒ
Ö —
Ö Ö
Ö ’	Ü 	á 1	á 3	á 5	á 7	á 9	à 	à &	à ;	à O	à _
à ó
à ≠
à ∫
à ¬
à Õà Ì
à ˚
à £
à  	â 	ä 	ä 	ã Q
å Ô
å Ωç ç 
ç µç πç º	é *	é {
é ∆
é ’
é è
é ‡	è 	è 	è  	ê 	ë "

rhs_norm"
_Z13get_global_idj"
_Z12get_local_idj"
llvm.fmuladd.f64"
_Z7barrierj"
_Z14get_local_sizej"
_Z12get_group_idj"
llvm.memset.p0i8.i64*ã
npb-BT-rhs_norm.clu
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

wgsize

 
transfer_bytes_log1p
 ¯ßA

transfer_bytes	
»û¥Ú

wgsize_log1p
 ¯ßA