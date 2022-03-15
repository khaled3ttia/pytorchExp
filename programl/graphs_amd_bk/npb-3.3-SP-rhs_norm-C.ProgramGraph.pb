
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
5icmpB-
+
	full_text

%16 = icmp sgt i32 %14, %5
#i32B

	full_text
	
i32 %14
8brB2
0
	full_text#
!
br i1 %16, label %72, label %17
!i1B

	full_text


i1 %16
Ybitcast8BL
J
	full_text=
;
9%18 = bitcast double* %0 to [163 x [163 x [5 x double]]]*
5icmp8B+
)
	full_text

%19 = icmp slt i32 %4, 1
:br8B2
0
	full_text#
!
br i1 %19, label %72, label %20
#i18B

	full_text


i1 %19
5icmp8B+
)
	full_text

%21 = icmp slt i32 %3, 1
0shl8B'
%
	full_text

%22 = shl i64 %8, 32
$i648B

	full_text


i64 %8
9ashr8B/
-
	full_text 

%23 = ashr exact i64 %22, 32
%i648B

	full_text
	
i64 %22
/add8B&
$
	full_text

%24 = add i32 %3, 1
/add8B&
$
	full_text

%25 = add i32 %4, 1
6zext8B,
*
	full_text

%26 = zext i32 %25 to i64
%i328B

	full_text
	
i32 %25
]getelementptr8BJ
H
	full_text;
9
7%27 = getelementptr inbounds double, double* %12, i64 1
-double*8B

	full_text

double* %12
]getelementptr8BJ
H
	full_text;
9
7%28 = getelementptr inbounds double, double* %12, i64 2
-double*8B

	full_text

double* %12
]getelementptr8BJ
H
	full_text;
9
7%29 = getelementptr inbounds double, double* %12, i64 3
-double*8B

	full_text

double* %12
]getelementptr8BJ
H
	full_text;
9
7%30 = getelementptr inbounds double, double* %12, i64 4
-double*8B

	full_text

double* %12
6zext8B,
*
	full_text

%31 = zext i32 %24 to i64
%i328B

	full_text
	
i32 %24
'br8B

	full_text

br label %32
Pphi8BG
E
	full_text8
6
4%33 = phi double [ %65, %64 ], [ 0.000000e+00, %20 ]
+double8B

	full_text


double %65
Pphi8BG
E
	full_text8
6
4%34 = phi double [ %66, %64 ], [ 0.000000e+00, %20 ]
+double8B

	full_text


double %66
Pphi8BG
E
	full_text8
6
4%35 = phi double [ %67, %64 ], [ 0.000000e+00, %20 ]
+double8B

	full_text


double %67
Pphi8BG
E
	full_text8
6
4%36 = phi double [ %68, %64 ], [ 0.000000e+00, %20 ]
+double8B

	full_text


double %68
Pphi8BG
E
	full_text8
6
4%37 = phi double [ %69, %64 ], [ 0.000000e+00, %20 ]
+double8B

	full_text


double %69
Bphi8B9
7
	full_text*
(
&%38 = phi i64 [ %70, %64 ], [ 1, %20 ]
%i648B

	full_text
	
i64 %70
:br8B2
0
	full_text#
!
br i1 %21, label %64, label %39
#i18B

	full_text


i1 %21
'br8B

	full_text

br label %40
Gphi8B>
<
	full_text/
-
+%41 = phi double [ %61, %40 ], [ %33, %39 ]
+double8B

	full_text


double %61
+double8B

	full_text


double %33
Gphi8B>
<
	full_text/
-
+%42 = phi double [ %58, %40 ], [ %34, %39 ]
+double8B

	full_text


double %58
+double8B

	full_text


double %34
Gphi8B>
<
	full_text/
-
+%43 = phi double [ %55, %40 ], [ %35, %39 ]
+double8B

	full_text


double %55
+double8B

	full_text


double %35
Gphi8B>
<
	full_text/
-
+%44 = phi double [ %52, %40 ], [ %36, %39 ]
+double8B

	full_text


double %52
+double8B

	full_text


double %36
Gphi8B>
<
	full_text/
-
+%45 = phi double [ %49, %40 ], [ %37, %39 ]
+double8B

	full_text


double %49
+double8B

	full_text


double %37
Bphi8B9
7
	full_text*
(
&%46 = phi i64 [ %62, %40 ], [ 1, %39 ]
%i648B

	full_text
	
i64 %62
¨getelementptr8B”
‘
	full_textƒ
€
~%47 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %18, i64 %23, i64 %38, i64 %46, i64 0
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %18
%i648B

	full_text
	
i64 %23
%i648B

	full_text
	
i64 %38
%i648B

	full_text
	
i64 %46
Nload8BD
B
	full_text5
3
1%48 = load double, double* %47, align 8, !tbaa !8
-double*8B

	full_text

double* %47
icall8B_
]
	full_textP
N
L%49 = tail call double @llvm.fmuladd.f64(double %48, double %48, double %45)
+double8B

	full_text


double %48
+double8B

	full_text


double %48
+double8B

	full_text


double %45
Nstore8BC
A
	full_text4
2
0store double %49, double* %12, align 8, !tbaa !8
+double8B

	full_text


double %49
-double*8B

	full_text

double* %12
¨getelementptr8B”
‘
	full_textƒ
€
~%50 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %18, i64 %23, i64 %38, i64 %46, i64 1
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %18
%i648B

	full_text
	
i64 %23
%i648B

	full_text
	
i64 %38
%i648B

	full_text
	
i64 %46
Nload8BD
B
	full_text5
3
1%51 = load double, double* %50, align 8, !tbaa !8
-double*8B

	full_text

double* %50
icall8B_
]
	full_textP
N
L%52 = tail call double @llvm.fmuladd.f64(double %51, double %51, double %44)
+double8B

	full_text


double %51
+double8B

	full_text


double %51
+double8B

	full_text


double %44
Nstore8BC
A
	full_text4
2
0store double %52, double* %27, align 8, !tbaa !8
+double8B

	full_text


double %52
-double*8B

	full_text

double* %27
¨getelementptr8B”
‘
	full_textƒ
€
~%53 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %18, i64 %23, i64 %38, i64 %46, i64 2
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %18
%i648B

	full_text
	
i64 %23
%i648B

	full_text
	
i64 %38
%i648B

	full_text
	
i64 %46
Nload8BD
B
	full_text5
3
1%54 = load double, double* %53, align 8, !tbaa !8
-double*8B

	full_text

double* %53
icall8B_
]
	full_textP
N
L%55 = tail call double @llvm.fmuladd.f64(double %54, double %54, double %43)
+double8B

	full_text


double %54
+double8B

	full_text


double %54
+double8B

	full_text


double %43
Nstore8BC
A
	full_text4
2
0store double %55, double* %28, align 8, !tbaa !8
+double8B

	full_text


double %55
-double*8B

	full_text

double* %28
¨getelementptr8B”
‘
	full_textƒ
€
~%56 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %18, i64 %23, i64 %38, i64 %46, i64 3
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %18
%i648B

	full_text
	
i64 %23
%i648B

	full_text
	
i64 %38
%i648B

	full_text
	
i64 %46
Nload8BD
B
	full_text5
3
1%57 = load double, double* %56, align 8, !tbaa !8
-double*8B

	full_text

double* %56
icall8B_
]
	full_textP
N
L%58 = tail call double @llvm.fmuladd.f64(double %57, double %57, double %42)
+double8B

	full_text


double %57
+double8B

	full_text


double %57
+double8B

	full_text


double %42
Nstore8BC
A
	full_text4
2
0store double %58, double* %29, align 8, !tbaa !8
+double8B

	full_text


double %58
-double*8B

	full_text

double* %29
¨getelementptr8B”
‘
	full_textƒ
€
~%59 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %18, i64 %23, i64 %38, i64 %46, i64 4
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %18
%i648B

	full_text
	
i64 %23
%i648B

	full_text
	
i64 %38
%i648B

	full_text
	
i64 %46
Nload8BD
B
	full_text5
3
1%60 = load double, double* %59, align 8, !tbaa !8
-double*8B

	full_text

double* %59
icall8B_
]
	full_textP
N
L%61 = tail call double @llvm.fmuladd.f64(double %60, double %60, double %41)
+double8B

	full_text


double %60
+double8B

	full_text


double %60
+double8B

	full_text


double %41
Nstore8BC
A
	full_text4
2
0store double %61, double* %30, align 8, !tbaa !8
+double8B

	full_text


double %61
-double*8B

	full_text

double* %30
8add8B/
-
	full_text 

%62 = add nuw nsw i64 %46, 1
%i648B

	full_text
	
i64 %46
7icmp8B-
+
	full_text

%63 = icmp eq i64 %62, %31
%i648B

	full_text
	
i64 %62
%i648B

	full_text
	
i64 %31
:br8B2
0
	full_text#
!
br i1 %63, label %64, label %40
#i18B

	full_text


i1 %63
Gphi8B>
<
	full_text/
-
+%65 = phi double [ %33, %32 ], [ %61, %40 ]
+double8B

	full_text


double %33
+double8B

	full_text


double %61
Gphi8B>
<
	full_text/
-
+%66 = phi double [ %34, %32 ], [ %58, %40 ]
+double8B

	full_text


double %34
+double8B

	full_text


double %58
Gphi8B>
<
	full_text/
-
+%67 = phi double [ %35, %32 ], [ %55, %40 ]
+double8B

	full_text


double %35
+double8B

	full_text


double %55
Gphi8B>
<
	full_text/
-
+%68 = phi double [ %36, %32 ], [ %52, %40 ]
+double8B

	full_text


double %36
+double8B

	full_text


double %52
Gphi8B>
<
	full_text/
-
+%69 = phi double [ %37, %32 ], [ %49, %40 ]
+double8B

	full_text


double %37
+double8B

	full_text


double %49
8add8B/
-
	full_text 

%70 = add nuw nsw i64 %38, 1
%i648B

	full_text
	
i64 %38
7icmp8B-
+
	full_text

%71 = icmp eq i64 %70, %26
%i648B

	full_text
	
i64 %70
%i648B

	full_text
	
i64 %26
:br8B2
0
	full_text#
!
br i1 %71, label %72, label %32
#i18B

	full_text


i1 %71
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #6
5icmp8B+
)
	full_text

%73 = icmp eq i32 %15, 0
%i328B

	full_text
	
i32 %15
;br8B3
1
	full_text$
"
 br i1 %73, label %74, label %147
#i18B

	full_text


i1 %73
Ocall8BE
C
	full_text6
4
2%75 = tail call i64 @_Z14get_local_sizej(i32 0) #5
6icmp8B,
*
	full_text

%76 = icmp ugt i64 %75, 1
%i648B

	full_text
	
i64 %75
:br8B2
0
	full_text#
!
br i1 %76, label %84, label %77
#i18B

	full_text


i1 %76
Abitcast8	B4
2
	full_text%
#
!%78 = bitcast double* %12 to i64*
-double*8	B

	full_text

double* %12
Hload8	B>
<
	full_text/
-
+%79 = load i64, i64* %78, align 8, !tbaa !8
'i64*8	B

	full_text


i64* %78
]getelementptr8	BJ
H
	full_text;
9
7%80 = getelementptr inbounds double, double* %12, i64 1
-double*8	B

	full_text

double* %12
]getelementptr8	BJ
H
	full_text;
9
7%81 = getelementptr inbounds double, double* %12, i64 2
-double*8	B

	full_text

double* %12
]getelementptr8	BJ
H
	full_text;
9
7%82 = getelementptr inbounds double, double* %12, i64 3
-double*8	B

	full_text

double* %12
]getelementptr8	BJ
H
	full_text;
9
7%83 = getelementptr inbounds double, double* %12, i64 4
-double*8	B

	full_text

double* %12
(br8	B 

	full_text

br label %121
Nload8
BD
B
	full_text5
3
1%85 = load double, double* %12, align 8, !tbaa !8
-double*8
B

	full_text

double* %12
]getelementptr8
BJ
H
	full_text;
9
7%86 = getelementptr inbounds double, double* %12, i64 1
-double*8
B

	full_text

double* %12
Nload8
BD
B
	full_text5
3
1%87 = load double, double* %86, align 8, !tbaa !8
-double*8
B

	full_text

double* %86
]getelementptr8
BJ
H
	full_text;
9
7%88 = getelementptr inbounds double, double* %12, i64 2
-double*8
B

	full_text

double* %12
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
7%90 = getelementptr inbounds double, double* %12, i64 3
-double*8
B

	full_text

double* %12
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
7%92 = getelementptr inbounds double, double* %12, i64 4
-double*8
B

	full_text

double* %12
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
'br8
B

	full_text

br label %94
Hphi8B?
=
	full_text0
.
,%95 = phi double [ %93, %84 ], [ %116, %94 ]
+double8B

	full_text


double %93
,double8B

	full_text

double %116
Hphi8B?
=
	full_text0
.
,%96 = phi double [ %91, %84 ], [ %113, %94 ]
+double8B

	full_text


double %91
,double8B

	full_text

double %113
Hphi8B?
=
	full_text0
.
,%97 = phi double [ %89, %84 ], [ %110, %94 ]
+double8B

	full_text


double %89
,double8B

	full_text

double %110
Hphi8B?
=
	full_text0
.
,%98 = phi double [ %87, %84 ], [ %107, %94 ]
+double8B

	full_text


double %87
,double8B

	full_text

double %107
Hphi8B?
=
	full_text0
.
,%99 = phi double [ %85, %84 ], [ %104, %94 ]
+double8B

	full_text


double %85
,double8B

	full_text

double %104
Dphi8B;
9
	full_text,
*
(%100 = phi i64 [ 1, %84 ], [ %117, %94 ]
&i648B

	full_text


i64 %117
:mul8B1
/
	full_text"
 
%101 = mul nuw nsw i64 %100, 5
&i648B

	full_text


i64 %100
`getelementptr8BM
K
	full_text>
<
:%102 = getelementptr inbounds double, double* %2, i64 %101
&i648B

	full_text


i64 %101
Pload8BF
D
	full_text7
5
3%103 = load double, double* %102, align 8, !tbaa !8
.double*8B

	full_text

double* %102
9fadd8B/
-
	full_text 

%104 = fadd double %103, %99
,double8B

	full_text

double %103
+double8B

	full_text


double %99
Ostore8BD
B
	full_text5
3
1store double %104, double* %12, align 8, !tbaa !8
,double8B

	full_text

double %104
-double*8B

	full_text

double* %12
_getelementptr8BL
J
	full_text=
;
9%105 = getelementptr inbounds double, double* %102, i64 1
.double*8B

	full_text

double* %102
Pload8BF
D
	full_text7
5
3%106 = load double, double* %105, align 8, !tbaa !8
.double*8B

	full_text

double* %105
9fadd8B/
-
	full_text 

%107 = fadd double %106, %98
,double8B

	full_text

double %106
+double8B

	full_text


double %98
Ostore8BD
B
	full_text5
3
1store double %107, double* %86, align 8, !tbaa !8
,double8B

	full_text

double %107
-double*8B

	full_text

double* %86
_getelementptr8BL
J
	full_text=
;
9%108 = getelementptr inbounds double, double* %102, i64 2
.double*8B

	full_text

double* %102
Pload8BF
D
	full_text7
5
3%109 = load double, double* %108, align 8, !tbaa !8
.double*8B

	full_text

double* %108
9fadd8B/
-
	full_text 

%110 = fadd double %109, %97
,double8B

	full_text

double %109
+double8B

	full_text


double %97
Ostore8BD
B
	full_text5
3
1store double %110, double* %88, align 8, !tbaa !8
,double8B

	full_text

double %110
-double*8B

	full_text

double* %88
_getelementptr8BL
J
	full_text=
;
9%111 = getelementptr inbounds double, double* %102, i64 3
.double*8B

	full_text

double* %102
Pload8BF
D
	full_text7
5
3%112 = load double, double* %111, align 8, !tbaa !8
.double*8B

	full_text

double* %111
9fadd8B/
-
	full_text 

%113 = fadd double %112, %96
,double8B

	full_text

double %112
+double8B

	full_text


double %96
Ostore8BD
B
	full_text5
3
1store double %113, double* %90, align 8, !tbaa !8
,double8B

	full_text

double %113
-double*8B

	full_text

double* %90
_getelementptr8BL
J
	full_text=
;
9%114 = getelementptr inbounds double, double* %102, i64 4
.double*8B

	full_text

double* %102
Pload8BF
D
	full_text7
5
3%115 = load double, double* %114, align 8, !tbaa !8
.double*8B

	full_text

double* %114
9fadd8B/
-
	full_text 

%116 = fadd double %115, %95
,double8B

	full_text

double %115
+double8B

	full_text


double %95
Ostore8BD
B
	full_text5
3
1store double %116, double* %92, align 8, !tbaa !8
,double8B

	full_text

double %116
-double*8B

	full_text

double* %92
:add8B1
/
	full_text"
 
%117 = add nuw nsw i64 %100, 1
&i648B

	full_text


i64 %100
9icmp8B/
-
	full_text 

%118 = icmp eq i64 %117, %75
&i648B

	full_text


i64 %117
%i648B

	full_text
	
i64 %75
<br8B4
2
	full_text%
#
!br i1 %118, label %119, label %94
$i18B

	full_text
	
i1 %118
Abitcast8B4
2
	full_text%
#
!%120 = bitcast double %104 to i64
,double8B

	full_text

double %104
(br8B 

	full_text

br label %121
Jphi8BA
?
	full_text2
0
.%122 = phi double* [ %83, %77 ], [ %92, %119 ]
-double*8B

	full_text

double* %83
-double*8B

	full_text

double* %92
Jphi8BA
?
	full_text2
0
.%123 = phi double* [ %82, %77 ], [ %90, %119 ]
-double*8B

	full_text

double* %82
-double*8B

	full_text

double* %90
Jphi8BA
?
	full_text2
0
.%124 = phi double* [ %81, %77 ], [ %88, %119 ]
-double*8B

	full_text

double* %81
-double*8B

	full_text

double* %88
Jphi8BA
?
	full_text2
0
.%125 = phi double* [ %80, %77 ], [ %86, %119 ]
-double*8B

	full_text

double* %80
-double*8B

	full_text

double* %86
Gphi8B>
<
	full_text/
-
+%126 = phi i64 [ %79, %77 ], [ %120, %119 ]
%i648B

	full_text
	
i64 %79
&i648B

	full_text


i64 %120
Ncall8BD
B
	full_text5
3
1%127 = tail call i64 @_Z12get_group_idj(i32 0) #5
2mul8B)
'
	full_text

%128 = mul i64 %127, 5
&i648B

	full_text


i64 %127
`getelementptr8BM
K
	full_text>
<
:%129 = getelementptr inbounds double, double* %1, i64 %128
&i648B

	full_text


i64 %128
Cbitcast8B6
4
	full_text'
%
#%130 = bitcast double* %129 to i64*
.double*8B

	full_text

double* %129
Jstore8B?
=
	full_text0
.
,store i64 %126, i64* %130, align 8, !tbaa !8
&i648B

	full_text


i64 %126
(i64*8B

	full_text

	i64* %130
Cbitcast8B6
4
	full_text'
%
#%131 = bitcast double* %125 to i64*
.double*8B

	full_text

double* %125
Jload8B@
>
	full_text1
/
-%132 = load i64, i64* %131, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %131
_getelementptr8BL
J
	full_text=
;
9%133 = getelementptr inbounds double, double* %129, i64 1
.double*8B

	full_text

double* %129
Cbitcast8B6
4
	full_text'
%
#%134 = bitcast double* %133 to i64*
.double*8B

	full_text

double* %133
Jstore8B?
=
	full_text0
.
,store i64 %132, i64* %134, align 8, !tbaa !8
&i648B

	full_text


i64 %132
(i64*8B

	full_text

	i64* %134
Cbitcast8B6
4
	full_text'
%
#%135 = bitcast double* %124 to i64*
.double*8B

	full_text

double* %124
Jload8B@
>
	full_text1
/
-%136 = load i64, i64* %135, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %135
_getelementptr8BL
J
	full_text=
;
9%137 = getelementptr inbounds double, double* %129, i64 2
.double*8B

	full_text

double* %129
Cbitcast8B6
4
	full_text'
%
#%138 = bitcast double* %137 to i64*
.double*8B

	full_text

double* %137
Jstore8B?
=
	full_text0
.
,store i64 %136, i64* %138, align 8, !tbaa !8
&i648B

	full_text


i64 %136
(i64*8B

	full_text

	i64* %138
Cbitcast8B6
4
	full_text'
%
#%139 = bitcast double* %123 to i64*
.double*8B

	full_text

double* %123
Jload8B@
>
	full_text1
/
-%140 = load i64, i64* %139, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %139
_getelementptr8BL
J
	full_text=
;
9%141 = getelementptr inbounds double, double* %129, i64 3
.double*8B

	full_text

double* %129
Cbitcast8B6
4
	full_text'
%
#%142 = bitcast double* %141 to i64*
.double*8B

	full_text

double* %141
Jstore8B?
=
	full_text0
.
,store i64 %140, i64* %142, align 8, !tbaa !8
&i648B

	full_text


i64 %140
(i64*8B

	full_text

	i64* %142
Cbitcast8B6
4
	full_text'
%
#%143 = bitcast double* %122 to i64*
.double*8B

	full_text

double* %122
Jload8B@
>
	full_text1
/
-%144 = load i64, i64* %143, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %143
_getelementptr8BL
J
	full_text=
;
9%145 = getelementptr inbounds double, double* %129, i64 4
.double*8B

	full_text

double* %129
Cbitcast8B6
4
	full_text'
%
#%146 = bitcast double* %145 to i64*
.double*8B

	full_text

double* %145
Jstore8B?
=
	full_text0
.
,store i64 %144, i64* %146, align 8, !tbaa !8
&i648B

	full_text


i64 %144
(i64*8B

	full_text

	i64* %146
(br8B 

	full_text

br label %147
$ret8B

	full_text


ret void
$i328B

	full_text


i32 %3
,double*8B

	full_text


double* %2
,double*8B

	full_text


double* %0
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
i32 %5
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
i64 0
#i328B

	full_text	

i32 1
#i648B

	full_text	

i64 1
$i648B

	full_text


i64 32
!i88B

	full_text

i8 0
#i648B

	full_text	

i64 3
#i648B

	full_text	

i64 4
%i18B

	full_text


i1 false
#i648B

	full_text	

i64 2
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
i64 21474836480        	
 		                       !! "# "" $% $$ &' && () (( *+ ** ,- ,, .0 // 12 11 34 33 56 55 78 77 9: 99 ;< ;? >@ >> AB AC AA DE DF DD GH GI GG JK JL JJ MN MM OP OQ OR OS OO TU TT VW VX VY VV Z[ Z\ ZZ ]^ ]_ ]` ]a ]] bc bb de df dg dd hi hj hh kl km kn ko kk pq pp rs rt ru rr vw vx vv yz y{ y| y} yy ~ ~~ € €
‚ €
ƒ €€ „… „
† „„ ‡ˆ ‡
‰ ‡
Š ‡
‹ ‡‡ Œ ŒŒ  
 
‘  ’“ ’
” ’’ •– •• —˜ —
™ —— š› š œ
 œœ Ÿ  Ÿ
¡ ŸŸ ¢£ ¢
¤ ¢¢ ¥¦ ¥
§ ¥¥ ¨© ¨
ª ¨¨ «¬ «« ­® ­
¯ ­­ °± °² ³´ ³³ µ¶ µ· ¸¹ ¸¸ º» º½ ¼¼ ¾¿ ¾¾ ÀÁ ÀÀ ÂÃ ÂÂ ÄÅ ÄÄ ÆÇ ÆÆ ÈÊ ÉÉ ËÌ ËË ÍÎ ÍÍ ÏĞ ÏÏ ÑÒ ÑÑ ÓÔ ÓÓ ÕÖ ÕÕ ×Ø ×× ÙÚ ÙÙ Ûİ Ü
Ş ÜÜ ßà ß
á ßß âã â
ä ââ åæ å
ç åå èé è
ê èè ë
ì ëë íî íí ï
ğ ïï ñò ññ óô ó
õ óó ö÷ ö
ø öö ùú ùù ûü ûû ış ı
ÿ ıı € €
‚ €€ ƒ„ ƒƒ …† …… ‡ˆ ‡
‰ ‡‡ Š‹ Š
Œ ŠŠ     ‘’ ‘
“ ‘‘ ”• ”
– ”” —˜ —— ™š ™™ ›œ ›
 ›› Ÿ 
   ¡¢ ¡¡ £¤ £
¥ ££ ¦§ ¦© ¨¨ ª¬ «
­ «« ®¯ ®
° ®® ±² ±
³ ±± ´µ ´
¶ ´´ ·¸ ·
¹ ·· ºº »¼ »» ½
¾ ½½ ¿À ¿¿ ÁÂ Á
Ã ÁÁ ÄÅ ÄÄ ÆÇ ÆÆ ÈÉ ÈÈ ÊË ÊÊ ÌÍ Ì
Î ÌÌ ÏĞ ÏÏ ÑÒ ÑÑ ÓÔ ÓÓ ÕÖ ÕÕ ×Ø ×
Ù ×× ÚÛ ÚÚ Üİ ÜÜ Şß ŞŞ àá àà âã â
ä ââ åæ åå çè çç éê éé ëì ëë íî í
ï íí ğò ò  ó 	ó ïô õ õ !ö ½	÷     
	         ! #	 %	 '	 )	 +  -œ 0Ÿ 2¢ 4¥ 6¨ 8« : < ?/ @€ B1 Cr E3 Fd H5 IV K7 L• N P Q9 RM SO UT WT XJ YV [	 \ ^ _9 `M a] cb eb fG gd i$ j l m9 nM ok qp sp tD ur w& x z {9 |M }y ~ ~ ‚A ƒ€ …( † ˆ ‰9 ŠM ‹‡ Œ Œ > ‘ “* ”M –• ˜, ™— ›/  1  € ¡3 £r ¤5 ¦d §7 ©V ª9 ¬« ®" ¯­ ± ´³ ¶· ¹¸ »	 ½¼ ¿	 Á	 Ã	 Å	 Ç	 Ê	 ÌË Î	 ĞÏ Ò	 ÔÓ Ö	 Ø× ÚÙ İ› ŞÕ à‘ áÑ ã‡ äÍ æı çÉ éó ê¡ ìë îí ğï òñ ôè õó ÷	 øï úù üû şå ÿı Ë ‚ï „ƒ †… ˆâ ‰‡ ‹Ï Œï   ’ß “‘ •Ó –ï ˜— š™ œÜ › Ÿ×  ë ¢¡ ¤· ¥£ §ó ©Æ ¬× ­Ä ¯Ó °Â ²Ï ³À µË ¶¾ ¸¨ ¹º ¼» ¾½ À· Â¿ Ã´ ÅÄ Ç½ ÉÈ ËÆ ÍÊ Î± ĞÏ Ò½ ÔÓ ÖÑ ØÕ Ù® ÛÚ İ½ ßŞ áÜ ãà ä« æå è½ êé ìç îë ï ² µ ·µ ñ ² º Éº ¼. /Û ÜÈ «; œ; =¦ ¨¦ Üğ ñ° ²° /= >ª «š œš > øø üü şş úú ıı ûû ñ ùù· üü · úú ² ûû ² øø  ùù V úú Vd úú dr úú rº ıı º şş € úú €	ÿ O	€ 	€ 	€  	€ !€ ²	 	 $	 9	 M	 ]
 •
 «
 ¸
 À
 Ë ë
 ù
 ¡
 È	‚ 	‚ 	‚ 	ƒ 	„ (	„ y
„ Ä
„ Ó
„ 
„ Ş	… *
… ‡
… Æ
… ×
… —
… é	† 	‡ &	‡ k
‡ Â
‡ Ï
‡ ƒ
‡ Ó	ˆ /	ˆ 1	ˆ 3	ˆ 5	ˆ 7	‰ Š Š 
Š ³Š ·Š º
‹ í
‹ »	Œ "

rhs_norm"
_Z13get_global_idj"
_Z12get_local_idj"
llvm.fmuladd.f64"
_Z7barrierj"
_Z14get_local_sizej"
_Z12get_group_idj"
llvm.memset.p0i8.i64*‹
npb-SP-rhs_norm.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282
 
transfer_bytes_log1p
ªA

wgsize


transfer_bytes	
°á¾á

devmap_label
 

wgsize_log1p
ªA