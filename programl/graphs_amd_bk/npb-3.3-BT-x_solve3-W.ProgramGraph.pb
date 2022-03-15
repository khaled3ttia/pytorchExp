

[external]
KcallBC
A
	full_text4
2
0%7 = tail call i64 @_Z13get_global_idj(i32 2) #3
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
4truncB+
)
	full_text

%9 = trunc i64 %8 to i32
"i64B

	full_text


i64 %8
LcallBD
B
	full_text5
3
1%10 = tail call i64 @_Z13get_global_idj(i32 1) #3
.addB'
%
	full_text

%11 = add i64 %10, 1
#i64B

	full_text
	
i64 %10
6truncB-
+
	full_text

%12 = trunc i64 %11 to i32
#i64B

	full_text
	
i64 %11
LcallBD
B
	full_text5
3
1%13 = tail call i64 @_Z13get_global_idj(i32 0) #3
.addB'
%
	full_text

%14 = add i64 %13, 1
#i64B

	full_text
	
i64 %13
2addB+
)
	full_text

%15 = add nsw i32 %5, -2
5icmpB-
+
	full_text

%16 = icmp slt i32 %15, %9
#i32B

	full_text
	
i32 %15
"i32B

	full_text


i32 %9
9brB3
1
	full_text$
"
 br i1 %16, label %560, label %17
!i1B

	full_text


i1 %16
8trunc8B-
+
	full_text

%18 = trunc i64 %14 to i32
%i648B

	full_text
	
i64 %14
4add8B+
)
	full_text

%19 = add nsw i32 %4, -2
8icmp8B.
,
	full_text

%20 = icmp slt i32 %19, %12
%i328B

	full_text
	
i32 %19
%i328B

	full_text
	
i32 %12
4add8B+
)
	full_text

%21 = add nsw i32 %3, -2
8icmp8B.
,
	full_text

%22 = icmp slt i32 %21, %18
%i328B

	full_text
	
i32 %21
%i328B

	full_text
	
i32 %18
/or8B'
%
	full_text

%23 = or i1 %20, %22
#i18B

	full_text


i1 %20
#i18B

	full_text


i1 %22
;br8B3
1
	full_text$
"
 br i1 %23, label %560, label %24
#i18B

	full_text


i1 %23
4add8B+
)
	full_text

%25 = add nsw i32 %9, -1
$i328B

	full_text


i32 %9
6mul8B-
+
	full_text

%26 = mul nsw i32 %25, %19
%i328B

	full_text
	
i32 %25
%i328B

	full_text
	
i32 %19
5add8B,
*
	full_text

%27 = add nsw i32 %12, -1
%i328B

	full_text
	
i32 %12
6add8B-
+
	full_text

%28 = add nsw i32 %27, %26
%i328B

	full_text
	
i32 %27
%i328B

	full_text
	
i32 %26
2mul8B)
'
	full_text

%29 = mul i32 %28, 625
%i328B

	full_text
	
i32 %28
6sext8B,
*
	full_text

%30 = sext i32 %29 to i64
%i328B

	full_text
	
i32 %29
^getelementptr8BK
I
	full_text<
:
8%31 = getelementptr inbounds double, double* %0, i64 %30
%i648B

	full_text
	
i64 %30
Pbitcast8BC
A
	full_text4
2
0%32 = bitcast double* %31 to [5 x [5 x double]]*
-double*8B

	full_text

double* %31
^getelementptr8BK
I
	full_text<
:
8%33 = getelementptr inbounds double, double* %1, i64 %30
%i648B

	full_text
	
i64 %30
Pbitcast8BC
A
	full_text4
2
0%34 = bitcast double* %33 to [5 x [5 x double]]*
-double*8B

	full_text

double* %33
3mul8B*
(
	full_text

%35 = mul i32 %28, 1875
%i328B

	full_text
	
i32 %28
6sext8B,
*
	full_text

%36 = sext i32 %35 to i64
%i328B

	full_text
	
i32 %35
^getelementptr8BK
I
	full_text<
:
8%37 = getelementptr inbounds double, double* %2, i64 %36
%i648B

	full_text
	
i64 %36
Vbitcast8BI
G
	full_text:
8
6%38 = bitcast double* %37 to [3 x [5 x [5 x double]]]*
-double*8B

	full_text

double* %37
1shl8B(
&
	full_text

%39 = shl i64 %14, 32
%i648B

	full_text
	
i64 %14
:add8B1
/
	full_text"
 
%40 = add i64 %39, -4294967296
%i648B

	full_text
	
i64 %39
9ashr8B/
-
	full_text 

%41 = ashr exact i64 %40, 32
%i648B

	full_text
	
i64 %40
…getelementptr8Br
p
	full_textc
a
_%42 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %32, i64 %41, i64 0, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %32
%i648B

	full_text
	
i64 %41
Nload8BD
B
	full_text5
3
1%43 = load double, double* %42, align 8, !tbaa !8
-double*8B

	full_text

double* %42
…getelementptr8Br
p
	full_textc
a
_%44 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %41, i64 0, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
%i648B

	full_text
	
i64 %41
Nload8BD
B
	full_text5
3
1%45 = load double, double* %44, align 8, !tbaa !8
-double*8B

	full_text

double* %44
@fmul8B6
4
	full_text'
%
#%46 = fmul double %45, 4.232000e-01
+double8B

	full_text


double %45
Afsub8B7
5
	full_text(
&
$%47 = fsub double -0.000000e+00, %46
+double8B

	full_text


double %46
xcall8Bn
l
	full_text_
]
[%48 = tail call double @llvm.fmuladd.f64(double %43, double 0xBF82D77318FC5048, double %47)
+double8B

	full_text


double %43
+double8B

	full_text


double %47
|call8Br
p
	full_textc
a
_%49 = tail call double @llvm.fmuladd.f64(double -4.232000e-01, double 7.500000e-01, double %48)
+double8B

	full_text


double %48
9ashr8B/
-
	full_text 

%50 = ashr exact i64 %39, 32
%i648B

	full_text
	
i64 %39
šgetelementptr8B†
ƒ
	full_textv
t
r%51 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 0, i64 0, i64 0
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Nstore8BC
A
	full_text4
2
0store double %49, double* %51, align 8, !tbaa !8
+double8B

	full_text


double %49
-double*8B

	full_text

double* %51
…getelementptr8Br
p
	full_textc
a
_%52 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %32, i64 %41, i64 1, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %32
%i648B

	full_text
	
i64 %41
Nload8BD
B
	full_text5
3
1%53 = load double, double* %52, align 8, !tbaa !8
-double*8B

	full_text

double* %52
…getelementptr8Br
p
	full_textc
a
_%54 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %41, i64 1, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
%i648B

	full_text
	
i64 %41
Nload8BD
B
	full_text5
3
1%55 = load double, double* %54, align 8, !tbaa !8
-double*8B

	full_text

double* %54
@fmul8B6
4
	full_text'
%
#%56 = fmul double %55, 4.232000e-01
+double8B

	full_text


double %55
Afsub8B7
5
	full_text(
&
$%57 = fsub double -0.000000e+00, %56
+double8B

	full_text


double %56
xcall8Bn
l
	full_text_
]
[%58 = tail call double @llvm.fmuladd.f64(double %53, double 0xBF82D77318FC5048, double %57)
+double8B

	full_text


double %53
+double8B

	full_text


double %57
šgetelementptr8B†
ƒ
	full_textv
t
r%59 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 0, i64 1, i64 0
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Nstore8BC
A
	full_text4
2
0store double %58, double* %59, align 8, !tbaa !8
+double8B

	full_text


double %58
-double*8B

	full_text

double* %59
…getelementptr8Br
p
	full_textc
a
_%60 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %32, i64 %41, i64 2, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %32
%i648B

	full_text
	
i64 %41
Nload8BD
B
	full_text5
3
1%61 = load double, double* %60, align 8, !tbaa !8
-double*8B

	full_text

double* %60
…getelementptr8Br
p
	full_textc
a
_%62 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %41, i64 2, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
%i648B

	full_text
	
i64 %41
Nload8BD
B
	full_text5
3
1%63 = load double, double* %62, align 8, !tbaa !8
-double*8B

	full_text

double* %62
@fmul8B6
4
	full_text'
%
#%64 = fmul double %63, 4.232000e-01
+double8B

	full_text


double %63
Afsub8B7
5
	full_text(
&
$%65 = fsub double -0.000000e+00, %64
+double8B

	full_text


double %64
xcall8Bn
l
	full_text_
]
[%66 = tail call double @llvm.fmuladd.f64(double %61, double 0xBF82D77318FC5048, double %65)
+double8B

	full_text


double %61
+double8B

	full_text


double %65
šgetelementptr8B†
ƒ
	full_textv
t
r%67 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 0, i64 2, i64 0
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Nstore8BC
A
	full_text4
2
0store double %66, double* %67, align 8, !tbaa !8
+double8B

	full_text


double %66
-double*8B

	full_text

double* %67
…getelementptr8Br
p
	full_textc
a
_%68 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %32, i64 %41, i64 3, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %32
%i648B

	full_text
	
i64 %41
Nload8BD
B
	full_text5
3
1%69 = load double, double* %68, align 8, !tbaa !8
-double*8B

	full_text

double* %68
…getelementptr8Br
p
	full_textc
a
_%70 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %41, i64 3, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
%i648B

	full_text
	
i64 %41
Nload8BD
B
	full_text5
3
1%71 = load double, double* %70, align 8, !tbaa !8
-double*8B

	full_text

double* %70
@fmul8B6
4
	full_text'
%
#%72 = fmul double %71, 4.232000e-01
+double8B

	full_text


double %71
Afsub8B7
5
	full_text(
&
$%73 = fsub double -0.000000e+00, %72
+double8B

	full_text


double %72
xcall8Bn
l
	full_text_
]
[%74 = tail call double @llvm.fmuladd.f64(double %69, double 0xBF82D77318FC5048, double %73)
+double8B

	full_text


double %69
+double8B

	full_text


double %73
šgetelementptr8B†
ƒ
	full_textv
t
r%75 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 0, i64 3, i64 0
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Nstore8BC
A
	full_text4
2
0store double %74, double* %75, align 8, !tbaa !8
+double8B

	full_text


double %74
-double*8B

	full_text

double* %75
…getelementptr8Br
p
	full_textc
a
_%76 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %32, i64 %41, i64 4, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %32
%i648B

	full_text
	
i64 %41
Nload8BD
B
	full_text5
3
1%77 = load double, double* %76, align 8, !tbaa !8
-double*8B

	full_text

double* %76
…getelementptr8Br
p
	full_textc
a
_%78 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %41, i64 4, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
%i648B

	full_text
	
i64 %41
Nload8BD
B
	full_text5
3
1%79 = load double, double* %78, align 8, !tbaa !8
-double*8B

	full_text

double* %78
@fmul8B6
4
	full_text'
%
#%80 = fmul double %79, 4.232000e-01
+double8B

	full_text


double %79
Afsub8B7
5
	full_text(
&
$%81 = fsub double -0.000000e+00, %80
+double8B

	full_text


double %80
xcall8Bn
l
	full_text_
]
[%82 = tail call double @llvm.fmuladd.f64(double %77, double 0xBF82D77318FC5048, double %81)
+double8B

	full_text


double %77
+double8B

	full_text


double %81
šgetelementptr8B†
ƒ
	full_textv
t
r%83 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 0, i64 4, i64 0
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Nstore8BC
A
	full_text4
2
0store double %82, double* %83, align 8, !tbaa !8
+double8B

	full_text


double %82
-double*8B

	full_text

double* %83
…getelementptr8Br
p
	full_textc
a
_%84 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %32, i64 %41, i64 0, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %32
%i648B

	full_text
	
i64 %41
Nload8BD
B
	full_text5
3
1%85 = load double, double* %84, align 8, !tbaa !8
-double*8B

	full_text

double* %84
…getelementptr8Br
p
	full_textc
a
_%86 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %41, i64 0, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
%i648B

	full_text
	
i64 %41
Nload8BD
B
	full_text5
3
1%87 = load double, double* %86, align 8, !tbaa !8
-double*8B

	full_text

double* %86
@fmul8B6
4
	full_text'
%
#%88 = fmul double %87, 4.232000e-01
+double8B

	full_text


double %87
Afsub8B7
5
	full_text(
&
$%89 = fsub double -0.000000e+00, %88
+double8B

	full_text


double %88
xcall8Bn
l
	full_text_
]
[%90 = tail call double @llvm.fmuladd.f64(double %85, double 0xBF82D77318FC5048, double %89)
+double8B

	full_text


double %85
+double8B

	full_text


double %89
šgetelementptr8B†
ƒ
	full_textv
t
r%91 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 0, i64 0, i64 1
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Nstore8BC
A
	full_text4
2
0store double %90, double* %91, align 8, !tbaa !8
+double8B

	full_text


double %90
-double*8B

	full_text

double* %91
…getelementptr8Br
p
	full_textc
a
_%92 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %32, i64 %41, i64 1, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %32
%i648B

	full_text
	
i64 %41
Nload8BD
B
	full_text5
3
1%93 = load double, double* %92, align 8, !tbaa !8
-double*8B

	full_text

double* %92
…getelementptr8Br
p
	full_textc
a
_%94 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %41, i64 1, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
%i648B

	full_text
	
i64 %41
Nload8BD
B
	full_text5
3
1%95 = load double, double* %94, align 8, !tbaa !8
-double*8B

	full_text

double* %94
@fmul8B6
4
	full_text'
%
#%96 = fmul double %95, 4.232000e-01
+double8B

	full_text


double %95
Afsub8B7
5
	full_text(
&
$%97 = fsub double -0.000000e+00, %96
+double8B

	full_text


double %96
xcall8Bn
l
	full_text_
]
[%98 = tail call double @llvm.fmuladd.f64(double %93, double 0xBF82D77318FC5048, double %97)
+double8B

	full_text


double %93
+double8B

	full_text


double %97
|call8Br
p
	full_textc
a
_%99 = tail call double @llvm.fmuladd.f64(double -4.232000e-01, double 7.500000e-01, double %98)
+double8B

	full_text


double %98
›getelementptr8B‡
„
	full_textw
u
s%100 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 0, i64 1, i64 1
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Ostore8BD
B
	full_text5
3
1store double %99, double* %100, align 8, !tbaa !8
+double8B

	full_text


double %99
.double*8B

	full_text

double* %100
†getelementptr8Bs
q
	full_textd
b
`%101 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %32, i64 %41, i64 2, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %32
%i648B

	full_text
	
i64 %41
Pload8BF
D
	full_text7
5
3%102 = load double, double* %101, align 8, !tbaa !8
.double*8B

	full_text

double* %101
†getelementptr8Bs
q
	full_textd
b
`%103 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %41, i64 2, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
%i648B

	full_text
	
i64 %41
Pload8BF
D
	full_text7
5
3%104 = load double, double* %103, align 8, !tbaa !8
.double*8B

	full_text

double* %103
Bfmul8B8
6
	full_text)
'
%%105 = fmul double %104, 4.232000e-01
,double8B

	full_text

double %104
Cfsub8B9
7
	full_text*
(
&%106 = fsub double -0.000000e+00, %105
,double8B

	full_text

double %105
{call8Bq
o
	full_textb
`
^%107 = tail call double @llvm.fmuladd.f64(double %102, double 0xBF82D77318FC5048, double %106)
,double8B

	full_text

double %102
,double8B

	full_text

double %106
›getelementptr8B‡
„
	full_textw
u
s%108 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 0, i64 2, i64 1
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %107, double* %108, align 8, !tbaa !8
,double8B

	full_text

double %107
.double*8B

	full_text

double* %108
†getelementptr8Bs
q
	full_textd
b
`%109 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %32, i64 %41, i64 3, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %32
%i648B

	full_text
	
i64 %41
Pload8BF
D
	full_text7
5
3%110 = load double, double* %109, align 8, !tbaa !8
.double*8B

	full_text

double* %109
†getelementptr8Bs
q
	full_textd
b
`%111 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %41, i64 3, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
%i648B

	full_text
	
i64 %41
Pload8BF
D
	full_text7
5
3%112 = load double, double* %111, align 8, !tbaa !8
.double*8B

	full_text

double* %111
Bfmul8B8
6
	full_text)
'
%%113 = fmul double %112, 4.232000e-01
,double8B

	full_text

double %112
Cfsub8B9
7
	full_text*
(
&%114 = fsub double -0.000000e+00, %113
,double8B

	full_text

double %113
{call8Bq
o
	full_textb
`
^%115 = tail call double @llvm.fmuladd.f64(double %110, double 0xBF82D77318FC5048, double %114)
,double8B

	full_text

double %110
,double8B

	full_text

double %114
›getelementptr8B‡
„
	full_textw
u
s%116 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 0, i64 3, i64 1
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %115, double* %116, align 8, !tbaa !8
,double8B

	full_text

double %115
.double*8B

	full_text

double* %116
†getelementptr8Bs
q
	full_textd
b
`%117 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %32, i64 %41, i64 4, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %32
%i648B

	full_text
	
i64 %41
Pload8BF
D
	full_text7
5
3%118 = load double, double* %117, align 8, !tbaa !8
.double*8B

	full_text

double* %117
†getelementptr8Bs
q
	full_textd
b
`%119 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %41, i64 4, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
%i648B

	full_text
	
i64 %41
Pload8BF
D
	full_text7
5
3%120 = load double, double* %119, align 8, !tbaa !8
.double*8B

	full_text

double* %119
Bfmul8B8
6
	full_text)
'
%%121 = fmul double %120, 4.232000e-01
,double8B

	full_text

double %120
Cfsub8B9
7
	full_text*
(
&%122 = fsub double -0.000000e+00, %121
,double8B

	full_text

double %121
{call8Bq
o
	full_textb
`
^%123 = tail call double @llvm.fmuladd.f64(double %118, double 0xBF82D77318FC5048, double %122)
,double8B

	full_text

double %118
,double8B

	full_text

double %122
›getelementptr8B‡
„
	full_textw
u
s%124 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 0, i64 4, i64 1
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %123, double* %124, align 8, !tbaa !8
,double8B

	full_text

double %123
.double*8B

	full_text

double* %124
†getelementptr8Bs
q
	full_textd
b
`%125 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %32, i64 %41, i64 0, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %32
%i648B

	full_text
	
i64 %41
Pload8BF
D
	full_text7
5
3%126 = load double, double* %125, align 8, !tbaa !8
.double*8B

	full_text

double* %125
†getelementptr8Bs
q
	full_textd
b
`%127 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %41, i64 0, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
%i648B

	full_text
	
i64 %41
Pload8BF
D
	full_text7
5
3%128 = load double, double* %127, align 8, !tbaa !8
.double*8B

	full_text

double* %127
Bfmul8B8
6
	full_text)
'
%%129 = fmul double %128, 4.232000e-01
,double8B

	full_text

double %128
Cfsub8B9
7
	full_text*
(
&%130 = fsub double -0.000000e+00, %129
,double8B

	full_text

double %129
{call8Bq
o
	full_textb
`
^%131 = tail call double @llvm.fmuladd.f64(double %126, double 0xBF82D77318FC5048, double %130)
,double8B

	full_text

double %126
,double8B

	full_text

double %130
›getelementptr8B‡
„
	full_textw
u
s%132 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 0, i64 0, i64 2
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %131, double* %132, align 8, !tbaa !8
,double8B

	full_text

double %131
.double*8B

	full_text

double* %132
†getelementptr8Bs
q
	full_textd
b
`%133 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %32, i64 %41, i64 1, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %32
%i648B

	full_text
	
i64 %41
Pload8BF
D
	full_text7
5
3%134 = load double, double* %133, align 8, !tbaa !8
.double*8B

	full_text

double* %133
†getelementptr8Bs
q
	full_textd
b
`%135 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %41, i64 1, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
%i648B

	full_text
	
i64 %41
Pload8BF
D
	full_text7
5
3%136 = load double, double* %135, align 8, !tbaa !8
.double*8B

	full_text

double* %135
Bfmul8B8
6
	full_text)
'
%%137 = fmul double %136, 4.232000e-01
,double8B

	full_text

double %136
Cfsub8B9
7
	full_text*
(
&%138 = fsub double -0.000000e+00, %137
,double8B

	full_text

double %137
{call8Bq
o
	full_textb
`
^%139 = tail call double @llvm.fmuladd.f64(double %134, double 0xBF82D77318FC5048, double %138)
,double8B

	full_text

double %134
,double8B

	full_text

double %138
›getelementptr8B‡
„
	full_textw
u
s%140 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 0, i64 1, i64 2
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %139, double* %140, align 8, !tbaa !8
,double8B

	full_text

double %139
.double*8B

	full_text

double* %140
†getelementptr8Bs
q
	full_textd
b
`%141 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %32, i64 %41, i64 2, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %32
%i648B

	full_text
	
i64 %41
Pload8BF
D
	full_text7
5
3%142 = load double, double* %141, align 8, !tbaa !8
.double*8B

	full_text

double* %141
†getelementptr8Bs
q
	full_textd
b
`%143 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %41, i64 2, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
%i648B

	full_text
	
i64 %41
Pload8BF
D
	full_text7
5
3%144 = load double, double* %143, align 8, !tbaa !8
.double*8B

	full_text

double* %143
Bfmul8B8
6
	full_text)
'
%%145 = fmul double %144, 4.232000e-01
,double8B

	full_text

double %144
Cfsub8B9
7
	full_text*
(
&%146 = fsub double -0.000000e+00, %145
,double8B

	full_text

double %145
{call8Bq
o
	full_textb
`
^%147 = tail call double @llvm.fmuladd.f64(double %142, double 0xBF82D77318FC5048, double %146)
,double8B

	full_text

double %142
,double8B

	full_text

double %146
~call8Bt
r
	full_texte
c
a%148 = tail call double @llvm.fmuladd.f64(double -4.232000e-01, double 7.500000e-01, double %147)
,double8B

	full_text

double %147
›getelementptr8B‡
„
	full_textw
u
s%149 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 0, i64 2, i64 2
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %148, double* %149, align 8, !tbaa !8
,double8B

	full_text

double %148
.double*8B

	full_text

double* %149
†getelementptr8Bs
q
	full_textd
b
`%150 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %32, i64 %41, i64 3, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %32
%i648B

	full_text
	
i64 %41
Pload8BF
D
	full_text7
5
3%151 = load double, double* %150, align 8, !tbaa !8
.double*8B

	full_text

double* %150
†getelementptr8Bs
q
	full_textd
b
`%152 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %41, i64 3, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
%i648B

	full_text
	
i64 %41
Pload8BF
D
	full_text7
5
3%153 = load double, double* %152, align 8, !tbaa !8
.double*8B

	full_text

double* %152
Bfmul8B8
6
	full_text)
'
%%154 = fmul double %153, 4.232000e-01
,double8B

	full_text

double %153
Cfsub8B9
7
	full_text*
(
&%155 = fsub double -0.000000e+00, %154
,double8B

	full_text

double %154
{call8Bq
o
	full_textb
`
^%156 = tail call double @llvm.fmuladd.f64(double %151, double 0xBF82D77318FC5048, double %155)
,double8B

	full_text

double %151
,double8B

	full_text

double %155
›getelementptr8B‡
„
	full_textw
u
s%157 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 0, i64 3, i64 2
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %156, double* %157, align 8, !tbaa !8
,double8B

	full_text

double %156
.double*8B

	full_text

double* %157
†getelementptr8Bs
q
	full_textd
b
`%158 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %32, i64 %41, i64 4, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %32
%i648B

	full_text
	
i64 %41
Pload8BF
D
	full_text7
5
3%159 = load double, double* %158, align 8, !tbaa !8
.double*8B

	full_text

double* %158
†getelementptr8Bs
q
	full_textd
b
`%160 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %41, i64 4, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
%i648B

	full_text
	
i64 %41
Pload8BF
D
	full_text7
5
3%161 = load double, double* %160, align 8, !tbaa !8
.double*8B

	full_text

double* %160
Bfmul8B8
6
	full_text)
'
%%162 = fmul double %161, 4.232000e-01
,double8B

	full_text

double %161
Cfsub8B9
7
	full_text*
(
&%163 = fsub double -0.000000e+00, %162
,double8B

	full_text

double %162
{call8Bq
o
	full_textb
`
^%164 = tail call double @llvm.fmuladd.f64(double %159, double 0xBF82D77318FC5048, double %163)
,double8B

	full_text

double %159
,double8B

	full_text

double %163
›getelementptr8B‡
„
	full_textw
u
s%165 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 0, i64 4, i64 2
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %164, double* %165, align 8, !tbaa !8
,double8B

	full_text

double %164
.double*8B

	full_text

double* %165
†getelementptr8Bs
q
	full_textd
b
`%166 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %32, i64 %41, i64 0, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %32
%i648B

	full_text
	
i64 %41
Pload8BF
D
	full_text7
5
3%167 = load double, double* %166, align 8, !tbaa !8
.double*8B

	full_text

double* %166
†getelementptr8Bs
q
	full_textd
b
`%168 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %41, i64 0, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
%i648B

	full_text
	
i64 %41
Pload8BF
D
	full_text7
5
3%169 = load double, double* %168, align 8, !tbaa !8
.double*8B

	full_text

double* %168
Bfmul8B8
6
	full_text)
'
%%170 = fmul double %169, 4.232000e-01
,double8B

	full_text

double %169
Cfsub8B9
7
	full_text*
(
&%171 = fsub double -0.000000e+00, %170
,double8B

	full_text

double %170
{call8Bq
o
	full_textb
`
^%172 = tail call double @llvm.fmuladd.f64(double %167, double 0xBF82D77318FC5048, double %171)
,double8B

	full_text

double %167
,double8B

	full_text

double %171
›getelementptr8B‡
„
	full_textw
u
s%173 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 0, i64 0, i64 3
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %172, double* %173, align 8, !tbaa !8
,double8B

	full_text

double %172
.double*8B

	full_text

double* %173
†getelementptr8Bs
q
	full_textd
b
`%174 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %32, i64 %41, i64 1, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %32
%i648B

	full_text
	
i64 %41
Pload8BF
D
	full_text7
5
3%175 = load double, double* %174, align 8, !tbaa !8
.double*8B

	full_text

double* %174
†getelementptr8Bs
q
	full_textd
b
`%176 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %41, i64 1, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
%i648B

	full_text
	
i64 %41
Pload8BF
D
	full_text7
5
3%177 = load double, double* %176, align 8, !tbaa !8
.double*8B

	full_text

double* %176
Bfmul8B8
6
	full_text)
'
%%178 = fmul double %177, 4.232000e-01
,double8B

	full_text

double %177
Cfsub8B9
7
	full_text*
(
&%179 = fsub double -0.000000e+00, %178
,double8B

	full_text

double %178
{call8Bq
o
	full_textb
`
^%180 = tail call double @llvm.fmuladd.f64(double %175, double 0xBF82D77318FC5048, double %179)
,double8B

	full_text

double %175
,double8B

	full_text

double %179
›getelementptr8B‡
„
	full_textw
u
s%181 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 0, i64 1, i64 3
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %180, double* %181, align 8, !tbaa !8
,double8B

	full_text

double %180
.double*8B

	full_text

double* %181
†getelementptr8Bs
q
	full_textd
b
`%182 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %32, i64 %41, i64 2, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %32
%i648B

	full_text
	
i64 %41
Pload8BF
D
	full_text7
5
3%183 = load double, double* %182, align 8, !tbaa !8
.double*8B

	full_text

double* %182
†getelementptr8Bs
q
	full_textd
b
`%184 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %41, i64 2, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
%i648B

	full_text
	
i64 %41
Pload8BF
D
	full_text7
5
3%185 = load double, double* %184, align 8, !tbaa !8
.double*8B

	full_text

double* %184
Bfmul8B8
6
	full_text)
'
%%186 = fmul double %185, 4.232000e-01
,double8B

	full_text

double %185
Cfsub8B9
7
	full_text*
(
&%187 = fsub double -0.000000e+00, %186
,double8B

	full_text

double %186
{call8Bq
o
	full_textb
`
^%188 = tail call double @llvm.fmuladd.f64(double %183, double 0xBF82D77318FC5048, double %187)
,double8B

	full_text

double %183
,double8B

	full_text

double %187
›getelementptr8B‡
„
	full_textw
u
s%189 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 0, i64 2, i64 3
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %188, double* %189, align 8, !tbaa !8
,double8B

	full_text

double %188
.double*8B

	full_text

double* %189
†getelementptr8Bs
q
	full_textd
b
`%190 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %32, i64 %41, i64 3, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %32
%i648B

	full_text
	
i64 %41
Pload8BF
D
	full_text7
5
3%191 = load double, double* %190, align 8, !tbaa !8
.double*8B

	full_text

double* %190
†getelementptr8Bs
q
	full_textd
b
`%192 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %41, i64 3, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
%i648B

	full_text
	
i64 %41
Pload8BF
D
	full_text7
5
3%193 = load double, double* %192, align 8, !tbaa !8
.double*8B

	full_text

double* %192
Bfmul8B8
6
	full_text)
'
%%194 = fmul double %193, 4.232000e-01
,double8B

	full_text

double %193
Cfsub8B9
7
	full_text*
(
&%195 = fsub double -0.000000e+00, %194
,double8B

	full_text

double %194
{call8Bq
o
	full_textb
`
^%196 = tail call double @llvm.fmuladd.f64(double %191, double 0xBF82D77318FC5048, double %195)
,double8B

	full_text

double %191
,double8B

	full_text

double %195
~call8Bt
r
	full_texte
c
a%197 = tail call double @llvm.fmuladd.f64(double -4.232000e-01, double 7.500000e-01, double %196)
,double8B

	full_text

double %196
›getelementptr8B‡
„
	full_textw
u
s%198 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 0, i64 3, i64 3
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %197, double* %198, align 8, !tbaa !8
,double8B

	full_text

double %197
.double*8B

	full_text

double* %198
†getelementptr8Bs
q
	full_textd
b
`%199 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %32, i64 %41, i64 4, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %32
%i648B

	full_text
	
i64 %41
Pload8BF
D
	full_text7
5
3%200 = load double, double* %199, align 8, !tbaa !8
.double*8B

	full_text

double* %199
†getelementptr8Bs
q
	full_textd
b
`%201 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %41, i64 4, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
%i648B

	full_text
	
i64 %41
Pload8BF
D
	full_text7
5
3%202 = load double, double* %201, align 8, !tbaa !8
.double*8B

	full_text

double* %201
Bfmul8B8
6
	full_text)
'
%%203 = fmul double %202, 4.232000e-01
,double8B

	full_text

double %202
Cfsub8B9
7
	full_text*
(
&%204 = fsub double -0.000000e+00, %203
,double8B

	full_text

double %203
{call8Bq
o
	full_textb
`
^%205 = tail call double @llvm.fmuladd.f64(double %200, double 0xBF82D77318FC5048, double %204)
,double8B

	full_text

double %200
,double8B

	full_text

double %204
›getelementptr8B‡
„
	full_textw
u
s%206 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 0, i64 4, i64 3
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %205, double* %206, align 8, !tbaa !8
,double8B

	full_text

double %205
.double*8B

	full_text

double* %206
†getelementptr8Bs
q
	full_textd
b
`%207 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %32, i64 %41, i64 0, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %32
%i648B

	full_text
	
i64 %41
Pload8BF
D
	full_text7
5
3%208 = load double, double* %207, align 8, !tbaa !8
.double*8B

	full_text

double* %207
†getelementptr8Bs
q
	full_textd
b
`%209 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %41, i64 0, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
%i648B

	full_text
	
i64 %41
Pload8BF
D
	full_text7
5
3%210 = load double, double* %209, align 8, !tbaa !8
.double*8B

	full_text

double* %209
Bfmul8B8
6
	full_text)
'
%%211 = fmul double %210, 4.232000e-01
,double8B

	full_text

double %210
Cfsub8B9
7
	full_text*
(
&%212 = fsub double -0.000000e+00, %211
,double8B

	full_text

double %211
{call8Bq
o
	full_textb
`
^%213 = tail call double @llvm.fmuladd.f64(double %208, double 0xBF82D77318FC5048, double %212)
,double8B

	full_text

double %208
,double8B

	full_text

double %212
›getelementptr8B‡
„
	full_textw
u
s%214 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 0, i64 0, i64 4
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %213, double* %214, align 8, !tbaa !8
,double8B

	full_text

double %213
.double*8B

	full_text

double* %214
†getelementptr8Bs
q
	full_textd
b
`%215 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %32, i64 %41, i64 1, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %32
%i648B

	full_text
	
i64 %41
Pload8BF
D
	full_text7
5
3%216 = load double, double* %215, align 8, !tbaa !8
.double*8B

	full_text

double* %215
†getelementptr8Bs
q
	full_textd
b
`%217 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %41, i64 1, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
%i648B

	full_text
	
i64 %41
Pload8BF
D
	full_text7
5
3%218 = load double, double* %217, align 8, !tbaa !8
.double*8B

	full_text

double* %217
Bfmul8B8
6
	full_text)
'
%%219 = fmul double %218, 4.232000e-01
,double8B

	full_text

double %218
Cfsub8B9
7
	full_text*
(
&%220 = fsub double -0.000000e+00, %219
,double8B

	full_text

double %219
{call8Bq
o
	full_textb
`
^%221 = tail call double @llvm.fmuladd.f64(double %216, double 0xBF82D77318FC5048, double %220)
,double8B

	full_text

double %216
,double8B

	full_text

double %220
›getelementptr8B‡
„
	full_textw
u
s%222 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 0, i64 1, i64 4
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %221, double* %222, align 8, !tbaa !8
,double8B

	full_text

double %221
.double*8B

	full_text

double* %222
†getelementptr8Bs
q
	full_textd
b
`%223 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %32, i64 %41, i64 2, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %32
%i648B

	full_text
	
i64 %41
Pload8BF
D
	full_text7
5
3%224 = load double, double* %223, align 8, !tbaa !8
.double*8B

	full_text

double* %223
†getelementptr8Bs
q
	full_textd
b
`%225 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %41, i64 2, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
%i648B

	full_text
	
i64 %41
Pload8BF
D
	full_text7
5
3%226 = load double, double* %225, align 8, !tbaa !8
.double*8B

	full_text

double* %225
Bfmul8B8
6
	full_text)
'
%%227 = fmul double %226, 4.232000e-01
,double8B

	full_text

double %226
Cfsub8B9
7
	full_text*
(
&%228 = fsub double -0.000000e+00, %227
,double8B

	full_text

double %227
{call8Bq
o
	full_textb
`
^%229 = tail call double @llvm.fmuladd.f64(double %224, double 0xBF82D77318FC5048, double %228)
,double8B

	full_text

double %224
,double8B

	full_text

double %228
›getelementptr8B‡
„
	full_textw
u
s%230 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 0, i64 2, i64 4
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %229, double* %230, align 8, !tbaa !8
,double8B

	full_text

double %229
.double*8B

	full_text

double* %230
†getelementptr8Bs
q
	full_textd
b
`%231 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %32, i64 %41, i64 3, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %32
%i648B

	full_text
	
i64 %41
Pload8BF
D
	full_text7
5
3%232 = load double, double* %231, align 8, !tbaa !8
.double*8B

	full_text

double* %231
†getelementptr8Bs
q
	full_textd
b
`%233 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %41, i64 3, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
%i648B

	full_text
	
i64 %41
Pload8BF
D
	full_text7
5
3%234 = load double, double* %233, align 8, !tbaa !8
.double*8B

	full_text

double* %233
Bfmul8B8
6
	full_text)
'
%%235 = fmul double %234, 4.232000e-01
,double8B

	full_text

double %234
Cfsub8B9
7
	full_text*
(
&%236 = fsub double -0.000000e+00, %235
,double8B

	full_text

double %235
{call8Bq
o
	full_textb
`
^%237 = tail call double @llvm.fmuladd.f64(double %232, double 0xBF82D77318FC5048, double %236)
,double8B

	full_text

double %232
,double8B

	full_text

double %236
›getelementptr8B‡
„
	full_textw
u
s%238 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 0, i64 3, i64 4
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %237, double* %238, align 8, !tbaa !8
,double8B

	full_text

double %237
.double*8B

	full_text

double* %238
†getelementptr8Bs
q
	full_textd
b
`%239 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %32, i64 %41, i64 4, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %32
%i648B

	full_text
	
i64 %41
Pload8BF
D
	full_text7
5
3%240 = load double, double* %239, align 8, !tbaa !8
.double*8B

	full_text

double* %239
†getelementptr8Bs
q
	full_textd
b
`%241 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %41, i64 4, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
%i648B

	full_text
	
i64 %41
Pload8BF
D
	full_text7
5
3%242 = load double, double* %241, align 8, !tbaa !8
.double*8B

	full_text

double* %241
Bfmul8B8
6
	full_text)
'
%%243 = fmul double %242, 4.232000e-01
,double8B

	full_text

double %242
Cfsub8B9
7
	full_text*
(
&%244 = fsub double -0.000000e+00, %243
,double8B

	full_text

double %243
{call8Bq
o
	full_textb
`
^%245 = tail call double @llvm.fmuladd.f64(double %240, double 0xBF82D77318FC5048, double %244)
,double8B

	full_text

double %240
,double8B

	full_text

double %244
~call8Bt
r
	full_texte
c
a%246 = tail call double @llvm.fmuladd.f64(double -4.232000e-01, double 7.500000e-01, double %245)
,double8B

	full_text

double %245
›getelementptr8B‡
„
	full_textw
u
s%247 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 0, i64 4, i64 4
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %246, double* %247, align 8, !tbaa !8
,double8B

	full_text

double %246
.double*8B

	full_text

double* %247
†getelementptr8Bs
q
	full_textd
b
`%248 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %50, i64 0, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
%i648B

	full_text
	
i64 %50
Pload8BF
D
	full_text7
5
3%249 = load double, double* %248, align 8, !tbaa !8
.double*8B

	full_text

double* %248
}call8Bs
q
	full_textd
b
`%250 = tail call double @llvm.fmuladd.f64(double %249, double 8.464000e-01, double 1.000000e+00)
,double8B

	full_text

double %249
}call8Bs
q
	full_textd
b
`%251 = tail call double @llvm.fmuladd.f64(double 8.464000e-01, double 7.500000e-01, double %250)
,double8B

	full_text

double %250
›getelementptr8B‡
„
	full_textw
u
s%252 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 1, i64 0, i64 0
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %251, double* %252, align 8, !tbaa !8
,double8B

	full_text

double %251
.double*8B

	full_text

double* %252
†getelementptr8Bs
q
	full_textd
b
`%253 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %50, i64 1, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
%i648B

	full_text
	
i64 %50
Pload8BF
D
	full_text7
5
3%254 = load double, double* %253, align 8, !tbaa !8
.double*8B

	full_text

double* %253
Bfmul8B8
6
	full_text)
'
%%255 = fmul double %254, 8.464000e-01
,double8B

	full_text

double %254
›getelementptr8B‡
„
	full_textw
u
s%256 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 1, i64 1, i64 0
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %255, double* %256, align 8, !tbaa !8
,double8B

	full_text

double %255
.double*8B

	full_text

double* %256
†getelementptr8Bs
q
	full_textd
b
`%257 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %50, i64 2, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
%i648B

	full_text
	
i64 %50
Pload8BF
D
	full_text7
5
3%258 = load double, double* %257, align 8, !tbaa !8
.double*8B

	full_text

double* %257
Bfmul8B8
6
	full_text)
'
%%259 = fmul double %258, 8.464000e-01
,double8B

	full_text

double %258
›getelementptr8B‡
„
	full_textw
u
s%260 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 1, i64 2, i64 0
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %259, double* %260, align 8, !tbaa !8
,double8B

	full_text

double %259
.double*8B

	full_text

double* %260
†getelementptr8Bs
q
	full_textd
b
`%261 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %50, i64 3, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
%i648B

	full_text
	
i64 %50
Pload8BF
D
	full_text7
5
3%262 = load double, double* %261, align 8, !tbaa !8
.double*8B

	full_text

double* %261
Bfmul8B8
6
	full_text)
'
%%263 = fmul double %262, 8.464000e-01
,double8B

	full_text

double %262
›getelementptr8B‡
„
	full_textw
u
s%264 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 1, i64 3, i64 0
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %263, double* %264, align 8, !tbaa !8
,double8B

	full_text

double %263
.double*8B

	full_text

double* %264
†getelementptr8Bs
q
	full_textd
b
`%265 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %50, i64 4, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
%i648B

	full_text
	
i64 %50
Pload8BF
D
	full_text7
5
3%266 = load double, double* %265, align 8, !tbaa !8
.double*8B

	full_text

double* %265
Bfmul8B8
6
	full_text)
'
%%267 = fmul double %266, 8.464000e-01
,double8B

	full_text

double %266
›getelementptr8B‡
„
	full_textw
u
s%268 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 1, i64 4, i64 0
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %267, double* %268, align 8, !tbaa !8
,double8B

	full_text

double %267
.double*8B

	full_text

double* %268
†getelementptr8Bs
q
	full_textd
b
`%269 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %50, i64 0, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
%i648B

	full_text
	
i64 %50
Pload8BF
D
	full_text7
5
3%270 = load double, double* %269, align 8, !tbaa !8
.double*8B

	full_text

double* %269
Bfmul8B8
6
	full_text)
'
%%271 = fmul double %270, 8.464000e-01
,double8B

	full_text

double %270
›getelementptr8B‡
„
	full_textw
u
s%272 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 1, i64 0, i64 1
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %271, double* %272, align 8, !tbaa !8
,double8B

	full_text

double %271
.double*8B

	full_text

double* %272
†getelementptr8Bs
q
	full_textd
b
`%273 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %50, i64 1, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
%i648B

	full_text
	
i64 %50
Pload8BF
D
	full_text7
5
3%274 = load double, double* %273, align 8, !tbaa !8
.double*8B

	full_text

double* %273
}call8Bs
q
	full_textd
b
`%275 = tail call double @llvm.fmuladd.f64(double %274, double 8.464000e-01, double 1.000000e+00)
,double8B

	full_text

double %274
}call8Bs
q
	full_textd
b
`%276 = tail call double @llvm.fmuladd.f64(double 8.464000e-01, double 7.500000e-01, double %275)
,double8B

	full_text

double %275
›getelementptr8B‡
„
	full_textw
u
s%277 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 1, i64 1, i64 1
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %276, double* %277, align 8, !tbaa !8
,double8B

	full_text

double %276
.double*8B

	full_text

double* %277
†getelementptr8Bs
q
	full_textd
b
`%278 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %50, i64 2, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
%i648B

	full_text
	
i64 %50
Pload8BF
D
	full_text7
5
3%279 = load double, double* %278, align 8, !tbaa !8
.double*8B

	full_text

double* %278
Bfmul8B8
6
	full_text)
'
%%280 = fmul double %279, 8.464000e-01
,double8B

	full_text

double %279
›getelementptr8B‡
„
	full_textw
u
s%281 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 1, i64 2, i64 1
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %280, double* %281, align 8, !tbaa !8
,double8B

	full_text

double %280
.double*8B

	full_text

double* %281
†getelementptr8Bs
q
	full_textd
b
`%282 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %50, i64 3, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
%i648B

	full_text
	
i64 %50
Pload8BF
D
	full_text7
5
3%283 = load double, double* %282, align 8, !tbaa !8
.double*8B

	full_text

double* %282
Bfmul8B8
6
	full_text)
'
%%284 = fmul double %283, 8.464000e-01
,double8B

	full_text

double %283
›getelementptr8B‡
„
	full_textw
u
s%285 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 1, i64 3, i64 1
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %284, double* %285, align 8, !tbaa !8
,double8B

	full_text

double %284
.double*8B

	full_text

double* %285
†getelementptr8Bs
q
	full_textd
b
`%286 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %50, i64 4, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
%i648B

	full_text
	
i64 %50
Pload8BF
D
	full_text7
5
3%287 = load double, double* %286, align 8, !tbaa !8
.double*8B

	full_text

double* %286
Bfmul8B8
6
	full_text)
'
%%288 = fmul double %287, 8.464000e-01
,double8B

	full_text

double %287
›getelementptr8B‡
„
	full_textw
u
s%289 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 1, i64 4, i64 1
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %288, double* %289, align 8, !tbaa !8
,double8B

	full_text

double %288
.double*8B

	full_text

double* %289
†getelementptr8Bs
q
	full_textd
b
`%290 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %50, i64 0, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
%i648B

	full_text
	
i64 %50
Pload8BF
D
	full_text7
5
3%291 = load double, double* %290, align 8, !tbaa !8
.double*8B

	full_text

double* %290
Bfmul8B8
6
	full_text)
'
%%292 = fmul double %291, 8.464000e-01
,double8B

	full_text

double %291
›getelementptr8B‡
„
	full_textw
u
s%293 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 1, i64 0, i64 2
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %292, double* %293, align 8, !tbaa !8
,double8B

	full_text

double %292
.double*8B

	full_text

double* %293
†getelementptr8Bs
q
	full_textd
b
`%294 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %50, i64 1, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
%i648B

	full_text
	
i64 %50
Pload8BF
D
	full_text7
5
3%295 = load double, double* %294, align 8, !tbaa !8
.double*8B

	full_text

double* %294
Bfmul8B8
6
	full_text)
'
%%296 = fmul double %295, 8.464000e-01
,double8B

	full_text

double %295
›getelementptr8B‡
„
	full_textw
u
s%297 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 1, i64 1, i64 2
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %296, double* %297, align 8, !tbaa !8
,double8B

	full_text

double %296
.double*8B

	full_text

double* %297
†getelementptr8Bs
q
	full_textd
b
`%298 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %50, i64 2, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
%i648B

	full_text
	
i64 %50
Pload8BF
D
	full_text7
5
3%299 = load double, double* %298, align 8, !tbaa !8
.double*8B

	full_text

double* %298
}call8Bs
q
	full_textd
b
`%300 = tail call double @llvm.fmuladd.f64(double %299, double 8.464000e-01, double 1.000000e+00)
,double8B

	full_text

double %299
}call8Bs
q
	full_textd
b
`%301 = tail call double @llvm.fmuladd.f64(double 8.464000e-01, double 7.500000e-01, double %300)
,double8B

	full_text

double %300
›getelementptr8B‡
„
	full_textw
u
s%302 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 1, i64 2, i64 2
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %301, double* %302, align 8, !tbaa !8
,double8B

	full_text

double %301
.double*8B

	full_text

double* %302
†getelementptr8Bs
q
	full_textd
b
`%303 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %50, i64 3, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
%i648B

	full_text
	
i64 %50
Pload8BF
D
	full_text7
5
3%304 = load double, double* %303, align 8, !tbaa !8
.double*8B

	full_text

double* %303
Bfmul8B8
6
	full_text)
'
%%305 = fmul double %304, 8.464000e-01
,double8B

	full_text

double %304
›getelementptr8B‡
„
	full_textw
u
s%306 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 1, i64 3, i64 2
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %305, double* %306, align 8, !tbaa !8
,double8B

	full_text

double %305
.double*8B

	full_text

double* %306
†getelementptr8Bs
q
	full_textd
b
`%307 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %50, i64 4, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
%i648B

	full_text
	
i64 %50
Pload8BF
D
	full_text7
5
3%308 = load double, double* %307, align 8, !tbaa !8
.double*8B

	full_text

double* %307
Bfmul8B8
6
	full_text)
'
%%309 = fmul double %308, 8.464000e-01
,double8B

	full_text

double %308
›getelementptr8B‡
„
	full_textw
u
s%310 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 1, i64 4, i64 2
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %309, double* %310, align 8, !tbaa !8
,double8B

	full_text

double %309
.double*8B

	full_text

double* %310
†getelementptr8Bs
q
	full_textd
b
`%311 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %50, i64 0, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
%i648B

	full_text
	
i64 %50
Pload8BF
D
	full_text7
5
3%312 = load double, double* %311, align 8, !tbaa !8
.double*8B

	full_text

double* %311
Bfmul8B8
6
	full_text)
'
%%313 = fmul double %312, 8.464000e-01
,double8B

	full_text

double %312
›getelementptr8B‡
„
	full_textw
u
s%314 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 1, i64 0, i64 3
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %313, double* %314, align 8, !tbaa !8
,double8B

	full_text

double %313
.double*8B

	full_text

double* %314
†getelementptr8Bs
q
	full_textd
b
`%315 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %50, i64 1, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
%i648B

	full_text
	
i64 %50
Pload8BF
D
	full_text7
5
3%316 = load double, double* %315, align 8, !tbaa !8
.double*8B

	full_text

double* %315
Bfmul8B8
6
	full_text)
'
%%317 = fmul double %316, 8.464000e-01
,double8B

	full_text

double %316
›getelementptr8B‡
„
	full_textw
u
s%318 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 1, i64 1, i64 3
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %317, double* %318, align 8, !tbaa !8
,double8B

	full_text

double %317
.double*8B

	full_text

double* %318
†getelementptr8Bs
q
	full_textd
b
`%319 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %50, i64 2, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
%i648B

	full_text
	
i64 %50
Pload8BF
D
	full_text7
5
3%320 = load double, double* %319, align 8, !tbaa !8
.double*8B

	full_text

double* %319
Bfmul8B8
6
	full_text)
'
%%321 = fmul double %320, 8.464000e-01
,double8B

	full_text

double %320
›getelementptr8B‡
„
	full_textw
u
s%322 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 1, i64 2, i64 3
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %321, double* %322, align 8, !tbaa !8
,double8B

	full_text

double %321
.double*8B

	full_text

double* %322
†getelementptr8Bs
q
	full_textd
b
`%323 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %50, i64 3, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
%i648B

	full_text
	
i64 %50
Pload8BF
D
	full_text7
5
3%324 = load double, double* %323, align 8, !tbaa !8
.double*8B

	full_text

double* %323
}call8Bs
q
	full_textd
b
`%325 = tail call double @llvm.fmuladd.f64(double %324, double 8.464000e-01, double 1.000000e+00)
,double8B

	full_text

double %324
}call8Bs
q
	full_textd
b
`%326 = tail call double @llvm.fmuladd.f64(double 8.464000e-01, double 7.500000e-01, double %325)
,double8B

	full_text

double %325
›getelementptr8B‡
„
	full_textw
u
s%327 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 1, i64 3, i64 3
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %326, double* %327, align 8, !tbaa !8
,double8B

	full_text

double %326
.double*8B

	full_text

double* %327
†getelementptr8Bs
q
	full_textd
b
`%328 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %50, i64 4, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
%i648B

	full_text
	
i64 %50
Pload8BF
D
	full_text7
5
3%329 = load double, double* %328, align 8, !tbaa !8
.double*8B

	full_text

double* %328
Bfmul8B8
6
	full_text)
'
%%330 = fmul double %329, 8.464000e-01
,double8B

	full_text

double %329
›getelementptr8B‡
„
	full_textw
u
s%331 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 1, i64 4, i64 3
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %330, double* %331, align 8, !tbaa !8
,double8B

	full_text

double %330
.double*8B

	full_text

double* %331
†getelementptr8Bs
q
	full_textd
b
`%332 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %50, i64 0, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
%i648B

	full_text
	
i64 %50
Pload8BF
D
	full_text7
5
3%333 = load double, double* %332, align 8, !tbaa !8
.double*8B

	full_text

double* %332
Bfmul8B8
6
	full_text)
'
%%334 = fmul double %333, 8.464000e-01
,double8B

	full_text

double %333
›getelementptr8B‡
„
	full_textw
u
s%335 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 1, i64 0, i64 4
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %334, double* %335, align 8, !tbaa !8
,double8B

	full_text

double %334
.double*8B

	full_text

double* %335
†getelementptr8Bs
q
	full_textd
b
`%336 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %50, i64 1, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
%i648B

	full_text
	
i64 %50
Pload8BF
D
	full_text7
5
3%337 = load double, double* %336, align 8, !tbaa !8
.double*8B

	full_text

double* %336
Bfmul8B8
6
	full_text)
'
%%338 = fmul double %337, 8.464000e-01
,double8B

	full_text

double %337
›getelementptr8B‡
„
	full_textw
u
s%339 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 1, i64 1, i64 4
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %338, double* %339, align 8, !tbaa !8
,double8B

	full_text

double %338
.double*8B

	full_text

double* %339
†getelementptr8Bs
q
	full_textd
b
`%340 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %50, i64 2, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
%i648B

	full_text
	
i64 %50
Pload8BF
D
	full_text7
5
3%341 = load double, double* %340, align 8, !tbaa !8
.double*8B

	full_text

double* %340
Bfmul8B8
6
	full_text)
'
%%342 = fmul double %341, 8.464000e-01
,double8B

	full_text

double %341
›getelementptr8B‡
„
	full_textw
u
s%343 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 1, i64 2, i64 4
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %342, double* %343, align 8, !tbaa !8
,double8B

	full_text

double %342
.double*8B

	full_text

double* %343
†getelementptr8Bs
q
	full_textd
b
`%344 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %50, i64 3, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
%i648B

	full_text
	
i64 %50
Pload8BF
D
	full_text7
5
3%345 = load double, double* %344, align 8, !tbaa !8
.double*8B

	full_text

double* %344
Bfmul8B8
6
	full_text)
'
%%346 = fmul double %345, 8.464000e-01
,double8B

	full_text

double %345
›getelementptr8B‡
„
	full_textw
u
s%347 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 1, i64 3, i64 4
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %346, double* %347, align 8, !tbaa !8
,double8B

	full_text

double %346
.double*8B

	full_text

double* %347
†getelementptr8Bs
q
	full_textd
b
`%348 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %50, i64 4, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
%i648B

	full_text
	
i64 %50
Pload8BF
D
	full_text7
5
3%349 = load double, double* %348, align 8, !tbaa !8
.double*8B

	full_text

double* %348
}call8Bs
q
	full_textd
b
`%350 = tail call double @llvm.fmuladd.f64(double %349, double 8.464000e-01, double 1.000000e+00)
,double8B

	full_text

double %349
}call8Bs
q
	full_textd
b
`%351 = tail call double @llvm.fmuladd.f64(double 8.464000e-01, double 7.500000e-01, double %350)
,double8B

	full_text

double %350
›getelementptr8B‡
„
	full_textw
u
s%352 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 1, i64 4, i64 4
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %351, double* %352, align 8, !tbaa !8
,double8B

	full_text

double %351
.double*8B

	full_text

double* %352
:add8B1
/
	full_text"
 
%353 = add i64 %39, 4294967296
%i648B

	full_text
	
i64 %39
;ashr8B1
/
	full_text"
 
%354 = ashr exact i64 %353, 32
&i648B

	full_text


i64 %353
‡getelementptr8Bt
r
	full_texte
c
a%355 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %32, i64 %354, i64 0, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %32
&i648B

	full_text


i64 %354
Pload8BF
D
	full_text7
5
3%356 = load double, double* %355, align 8, !tbaa !8
.double*8B

	full_text

double* %355
‡getelementptr8Bt
r
	full_texte
c
a%357 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %354, i64 0, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
&i648B

	full_text


i64 %354
Pload8BF
D
	full_text7
5
3%358 = load double, double* %357, align 8, !tbaa !8
.double*8B

	full_text

double* %357
Bfmul8B8
6
	full_text)
'
%%359 = fmul double %358, 4.232000e-01
,double8B

	full_text

double %358
Cfsub8B9
7
	full_text*
(
&%360 = fsub double -0.000000e+00, %359
,double8B

	full_text

double %359
{call8Bq
o
	full_textb
`
^%361 = tail call double @llvm.fmuladd.f64(double %356, double 0x3F82D77318FC5048, double %360)
,double8B

	full_text

double %356
,double8B

	full_text

double %360
~call8Bt
r
	full_texte
c
a%362 = tail call double @llvm.fmuladd.f64(double -4.232000e-01, double 7.500000e-01, double %361)
,double8B

	full_text

double %361
›getelementptr8B‡
„
	full_textw
u
s%363 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 2, i64 0, i64 0
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %362, double* %363, align 8, !tbaa !8
,double8B

	full_text

double %362
.double*8B

	full_text

double* %363
‡getelementptr8Bt
r
	full_texte
c
a%364 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %32, i64 %354, i64 1, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %32
&i648B

	full_text


i64 %354
Pload8BF
D
	full_text7
5
3%365 = load double, double* %364, align 8, !tbaa !8
.double*8B

	full_text

double* %364
‡getelementptr8Bt
r
	full_texte
c
a%366 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %354, i64 1, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
&i648B

	full_text


i64 %354
Pload8BF
D
	full_text7
5
3%367 = load double, double* %366, align 8, !tbaa !8
.double*8B

	full_text

double* %366
Bfmul8B8
6
	full_text)
'
%%368 = fmul double %367, 4.232000e-01
,double8B

	full_text

double %367
Cfsub8B9
7
	full_text*
(
&%369 = fsub double -0.000000e+00, %368
,double8B

	full_text

double %368
{call8Bq
o
	full_textb
`
^%370 = tail call double @llvm.fmuladd.f64(double %365, double 0x3F82D77318FC5048, double %369)
,double8B

	full_text

double %365
,double8B

	full_text

double %369
›getelementptr8B‡
„
	full_textw
u
s%371 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 2, i64 1, i64 0
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %370, double* %371, align 8, !tbaa !8
,double8B

	full_text

double %370
.double*8B

	full_text

double* %371
‡getelementptr8Bt
r
	full_texte
c
a%372 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %32, i64 %354, i64 2, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %32
&i648B

	full_text


i64 %354
Pload8BF
D
	full_text7
5
3%373 = load double, double* %372, align 8, !tbaa !8
.double*8B

	full_text

double* %372
‡getelementptr8Bt
r
	full_texte
c
a%374 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %354, i64 2, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
&i648B

	full_text


i64 %354
Pload8BF
D
	full_text7
5
3%375 = load double, double* %374, align 8, !tbaa !8
.double*8B

	full_text

double* %374
Bfmul8B8
6
	full_text)
'
%%376 = fmul double %375, 4.232000e-01
,double8B

	full_text

double %375
Cfsub8B9
7
	full_text*
(
&%377 = fsub double -0.000000e+00, %376
,double8B

	full_text

double %376
{call8Bq
o
	full_textb
`
^%378 = tail call double @llvm.fmuladd.f64(double %373, double 0x3F82D77318FC5048, double %377)
,double8B

	full_text

double %373
,double8B

	full_text

double %377
›getelementptr8B‡
„
	full_textw
u
s%379 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 2, i64 2, i64 0
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %378, double* %379, align 8, !tbaa !8
,double8B

	full_text

double %378
.double*8B

	full_text

double* %379
‡getelementptr8Bt
r
	full_texte
c
a%380 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %32, i64 %354, i64 3, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %32
&i648B

	full_text


i64 %354
Pload8BF
D
	full_text7
5
3%381 = load double, double* %380, align 8, !tbaa !8
.double*8B

	full_text

double* %380
‡getelementptr8Bt
r
	full_texte
c
a%382 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %354, i64 3, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
&i648B

	full_text


i64 %354
Pload8BF
D
	full_text7
5
3%383 = load double, double* %382, align 8, !tbaa !8
.double*8B

	full_text

double* %382
Bfmul8B8
6
	full_text)
'
%%384 = fmul double %383, 4.232000e-01
,double8B

	full_text

double %383
Cfsub8B9
7
	full_text*
(
&%385 = fsub double -0.000000e+00, %384
,double8B

	full_text

double %384
{call8Bq
o
	full_textb
`
^%386 = tail call double @llvm.fmuladd.f64(double %381, double 0x3F82D77318FC5048, double %385)
,double8B

	full_text

double %381
,double8B

	full_text

double %385
›getelementptr8B‡
„
	full_textw
u
s%387 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 2, i64 3, i64 0
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %386, double* %387, align 8, !tbaa !8
,double8B

	full_text

double %386
.double*8B

	full_text

double* %387
‡getelementptr8Bt
r
	full_texte
c
a%388 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %32, i64 %354, i64 4, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %32
&i648B

	full_text


i64 %354
Pload8BF
D
	full_text7
5
3%389 = load double, double* %388, align 8, !tbaa !8
.double*8B

	full_text

double* %388
‡getelementptr8Bt
r
	full_texte
c
a%390 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %354, i64 4, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
&i648B

	full_text


i64 %354
Pload8BF
D
	full_text7
5
3%391 = load double, double* %390, align 8, !tbaa !8
.double*8B

	full_text

double* %390
Bfmul8B8
6
	full_text)
'
%%392 = fmul double %391, 4.232000e-01
,double8B

	full_text

double %391
Cfsub8B9
7
	full_text*
(
&%393 = fsub double -0.000000e+00, %392
,double8B

	full_text

double %392
{call8Bq
o
	full_textb
`
^%394 = tail call double @llvm.fmuladd.f64(double %389, double 0x3F82D77318FC5048, double %393)
,double8B

	full_text

double %389
,double8B

	full_text

double %393
›getelementptr8B‡
„
	full_textw
u
s%395 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 2, i64 4, i64 0
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %394, double* %395, align 8, !tbaa !8
,double8B

	full_text

double %394
.double*8B

	full_text

double* %395
‡getelementptr8Bt
r
	full_texte
c
a%396 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %32, i64 %354, i64 0, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %32
&i648B

	full_text


i64 %354
Pload8BF
D
	full_text7
5
3%397 = load double, double* %396, align 8, !tbaa !8
.double*8B

	full_text

double* %396
‡getelementptr8Bt
r
	full_texte
c
a%398 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %354, i64 0, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
&i648B

	full_text


i64 %354
Pload8BF
D
	full_text7
5
3%399 = load double, double* %398, align 8, !tbaa !8
.double*8B

	full_text

double* %398
Bfmul8B8
6
	full_text)
'
%%400 = fmul double %399, 4.232000e-01
,double8B

	full_text

double %399
Cfsub8B9
7
	full_text*
(
&%401 = fsub double -0.000000e+00, %400
,double8B

	full_text

double %400
{call8Bq
o
	full_textb
`
^%402 = tail call double @llvm.fmuladd.f64(double %397, double 0x3F82D77318FC5048, double %401)
,double8B

	full_text

double %397
,double8B

	full_text

double %401
›getelementptr8B‡
„
	full_textw
u
s%403 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 2, i64 0, i64 1
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %402, double* %403, align 8, !tbaa !8
,double8B

	full_text

double %402
.double*8B

	full_text

double* %403
‡getelementptr8Bt
r
	full_texte
c
a%404 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %32, i64 %354, i64 1, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %32
&i648B

	full_text


i64 %354
Pload8BF
D
	full_text7
5
3%405 = load double, double* %404, align 8, !tbaa !8
.double*8B

	full_text

double* %404
‡getelementptr8Bt
r
	full_texte
c
a%406 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %354, i64 1, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
&i648B

	full_text


i64 %354
Pload8BF
D
	full_text7
5
3%407 = load double, double* %406, align 8, !tbaa !8
.double*8B

	full_text

double* %406
Bfmul8B8
6
	full_text)
'
%%408 = fmul double %407, 4.232000e-01
,double8B

	full_text

double %407
Cfsub8B9
7
	full_text*
(
&%409 = fsub double -0.000000e+00, %408
,double8B

	full_text

double %408
{call8Bq
o
	full_textb
`
^%410 = tail call double @llvm.fmuladd.f64(double %405, double 0x3F82D77318FC5048, double %409)
,double8B

	full_text

double %405
,double8B

	full_text

double %409
~call8Bt
r
	full_texte
c
a%411 = tail call double @llvm.fmuladd.f64(double -4.232000e-01, double 7.500000e-01, double %410)
,double8B

	full_text

double %410
›getelementptr8B‡
„
	full_textw
u
s%412 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 2, i64 1, i64 1
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %411, double* %412, align 8, !tbaa !8
,double8B

	full_text

double %411
.double*8B

	full_text

double* %412
‡getelementptr8Bt
r
	full_texte
c
a%413 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %32, i64 %354, i64 2, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %32
&i648B

	full_text


i64 %354
Pload8BF
D
	full_text7
5
3%414 = load double, double* %413, align 8, !tbaa !8
.double*8B

	full_text

double* %413
‡getelementptr8Bt
r
	full_texte
c
a%415 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %354, i64 2, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
&i648B

	full_text


i64 %354
Pload8BF
D
	full_text7
5
3%416 = load double, double* %415, align 8, !tbaa !8
.double*8B

	full_text

double* %415
Bfmul8B8
6
	full_text)
'
%%417 = fmul double %416, 4.232000e-01
,double8B

	full_text

double %416
Cfsub8B9
7
	full_text*
(
&%418 = fsub double -0.000000e+00, %417
,double8B

	full_text

double %417
{call8Bq
o
	full_textb
`
^%419 = tail call double @llvm.fmuladd.f64(double %414, double 0x3F82D77318FC5048, double %418)
,double8B

	full_text

double %414
,double8B

	full_text

double %418
›getelementptr8B‡
„
	full_textw
u
s%420 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 2, i64 2, i64 1
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %419, double* %420, align 8, !tbaa !8
,double8B

	full_text

double %419
.double*8B

	full_text

double* %420
‡getelementptr8Bt
r
	full_texte
c
a%421 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %32, i64 %354, i64 3, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %32
&i648B

	full_text


i64 %354
Pload8BF
D
	full_text7
5
3%422 = load double, double* %421, align 8, !tbaa !8
.double*8B

	full_text

double* %421
‡getelementptr8Bt
r
	full_texte
c
a%423 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %354, i64 3, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
&i648B

	full_text


i64 %354
Pload8BF
D
	full_text7
5
3%424 = load double, double* %423, align 8, !tbaa !8
.double*8B

	full_text

double* %423
Bfmul8B8
6
	full_text)
'
%%425 = fmul double %424, 4.232000e-01
,double8B

	full_text

double %424
Cfsub8B9
7
	full_text*
(
&%426 = fsub double -0.000000e+00, %425
,double8B

	full_text

double %425
{call8Bq
o
	full_textb
`
^%427 = tail call double @llvm.fmuladd.f64(double %422, double 0x3F82D77318FC5048, double %426)
,double8B

	full_text

double %422
,double8B

	full_text

double %426
›getelementptr8B‡
„
	full_textw
u
s%428 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 2, i64 3, i64 1
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %427, double* %428, align 8, !tbaa !8
,double8B

	full_text

double %427
.double*8B

	full_text

double* %428
‡getelementptr8Bt
r
	full_texte
c
a%429 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %32, i64 %354, i64 4, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %32
&i648B

	full_text


i64 %354
Pload8BF
D
	full_text7
5
3%430 = load double, double* %429, align 8, !tbaa !8
.double*8B

	full_text

double* %429
‡getelementptr8Bt
r
	full_texte
c
a%431 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %354, i64 4, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
&i648B

	full_text


i64 %354
Pload8BF
D
	full_text7
5
3%432 = load double, double* %431, align 8, !tbaa !8
.double*8B

	full_text

double* %431
Bfmul8B8
6
	full_text)
'
%%433 = fmul double %432, 4.232000e-01
,double8B

	full_text

double %432
Cfsub8B9
7
	full_text*
(
&%434 = fsub double -0.000000e+00, %433
,double8B

	full_text

double %433
{call8Bq
o
	full_textb
`
^%435 = tail call double @llvm.fmuladd.f64(double %430, double 0x3F82D77318FC5048, double %434)
,double8B

	full_text

double %430
,double8B

	full_text

double %434
›getelementptr8B‡
„
	full_textw
u
s%436 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 2, i64 4, i64 1
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %435, double* %436, align 8, !tbaa !8
,double8B

	full_text

double %435
.double*8B

	full_text

double* %436
‡getelementptr8Bt
r
	full_texte
c
a%437 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %32, i64 %354, i64 0, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %32
&i648B

	full_text


i64 %354
Pload8BF
D
	full_text7
5
3%438 = load double, double* %437, align 8, !tbaa !8
.double*8B

	full_text

double* %437
‡getelementptr8Bt
r
	full_texte
c
a%439 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %354, i64 0, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
&i648B

	full_text


i64 %354
Pload8BF
D
	full_text7
5
3%440 = load double, double* %439, align 8, !tbaa !8
.double*8B

	full_text

double* %439
Bfmul8B8
6
	full_text)
'
%%441 = fmul double %440, 4.232000e-01
,double8B

	full_text

double %440
Cfsub8B9
7
	full_text*
(
&%442 = fsub double -0.000000e+00, %441
,double8B

	full_text

double %441
{call8Bq
o
	full_textb
`
^%443 = tail call double @llvm.fmuladd.f64(double %438, double 0x3F82D77318FC5048, double %442)
,double8B

	full_text

double %438
,double8B

	full_text

double %442
›getelementptr8B‡
„
	full_textw
u
s%444 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 2, i64 0, i64 2
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %443, double* %444, align 8, !tbaa !8
,double8B

	full_text

double %443
.double*8B

	full_text

double* %444
‡getelementptr8Bt
r
	full_texte
c
a%445 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %32, i64 %354, i64 1, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %32
&i648B

	full_text


i64 %354
Pload8BF
D
	full_text7
5
3%446 = load double, double* %445, align 8, !tbaa !8
.double*8B

	full_text

double* %445
‡getelementptr8Bt
r
	full_texte
c
a%447 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %354, i64 1, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
&i648B

	full_text


i64 %354
Pload8BF
D
	full_text7
5
3%448 = load double, double* %447, align 8, !tbaa !8
.double*8B

	full_text

double* %447
Bfmul8B8
6
	full_text)
'
%%449 = fmul double %448, 4.232000e-01
,double8B

	full_text

double %448
Cfsub8B9
7
	full_text*
(
&%450 = fsub double -0.000000e+00, %449
,double8B

	full_text

double %449
{call8Bq
o
	full_textb
`
^%451 = tail call double @llvm.fmuladd.f64(double %446, double 0x3F82D77318FC5048, double %450)
,double8B

	full_text

double %446
,double8B

	full_text

double %450
›getelementptr8B‡
„
	full_textw
u
s%452 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 2, i64 1, i64 2
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %451, double* %452, align 8, !tbaa !8
,double8B

	full_text

double %451
.double*8B

	full_text

double* %452
‡getelementptr8Bt
r
	full_texte
c
a%453 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %32, i64 %354, i64 2, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %32
&i648B

	full_text


i64 %354
Pload8BF
D
	full_text7
5
3%454 = load double, double* %453, align 8, !tbaa !8
.double*8B

	full_text

double* %453
‡getelementptr8Bt
r
	full_texte
c
a%455 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %354, i64 2, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
&i648B

	full_text


i64 %354
Pload8BF
D
	full_text7
5
3%456 = load double, double* %455, align 8, !tbaa !8
.double*8B

	full_text

double* %455
Bfmul8B8
6
	full_text)
'
%%457 = fmul double %456, 4.232000e-01
,double8B

	full_text

double %456
Cfsub8B9
7
	full_text*
(
&%458 = fsub double -0.000000e+00, %457
,double8B

	full_text

double %457
{call8Bq
o
	full_textb
`
^%459 = tail call double @llvm.fmuladd.f64(double %454, double 0x3F82D77318FC5048, double %458)
,double8B

	full_text

double %454
,double8B

	full_text

double %458
~call8Bt
r
	full_texte
c
a%460 = tail call double @llvm.fmuladd.f64(double -4.232000e-01, double 7.500000e-01, double %459)
,double8B

	full_text

double %459
›getelementptr8B‡
„
	full_textw
u
s%461 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 2, i64 2, i64 2
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %460, double* %461, align 8, !tbaa !8
,double8B

	full_text

double %460
.double*8B

	full_text

double* %461
‡getelementptr8Bt
r
	full_texte
c
a%462 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %32, i64 %354, i64 3, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %32
&i648B

	full_text


i64 %354
Pload8BF
D
	full_text7
5
3%463 = load double, double* %462, align 8, !tbaa !8
.double*8B

	full_text

double* %462
‡getelementptr8Bt
r
	full_texte
c
a%464 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %354, i64 3, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
&i648B

	full_text


i64 %354
Pload8BF
D
	full_text7
5
3%465 = load double, double* %464, align 8, !tbaa !8
.double*8B

	full_text

double* %464
Bfmul8B8
6
	full_text)
'
%%466 = fmul double %465, 4.232000e-01
,double8B

	full_text

double %465
Cfsub8B9
7
	full_text*
(
&%467 = fsub double -0.000000e+00, %466
,double8B

	full_text

double %466
{call8Bq
o
	full_textb
`
^%468 = tail call double @llvm.fmuladd.f64(double %463, double 0x3F82D77318FC5048, double %467)
,double8B

	full_text

double %463
,double8B

	full_text

double %467
›getelementptr8B‡
„
	full_textw
u
s%469 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 2, i64 3, i64 2
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %468, double* %469, align 8, !tbaa !8
,double8B

	full_text

double %468
.double*8B

	full_text

double* %469
‡getelementptr8Bt
r
	full_texte
c
a%470 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %32, i64 %354, i64 4, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %32
&i648B

	full_text


i64 %354
Pload8BF
D
	full_text7
5
3%471 = load double, double* %470, align 8, !tbaa !8
.double*8B

	full_text

double* %470
‡getelementptr8Bt
r
	full_texte
c
a%472 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %354, i64 4, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
&i648B

	full_text


i64 %354
Pload8BF
D
	full_text7
5
3%473 = load double, double* %472, align 8, !tbaa !8
.double*8B

	full_text

double* %472
Bfmul8B8
6
	full_text)
'
%%474 = fmul double %473, 4.232000e-01
,double8B

	full_text

double %473
Cfsub8B9
7
	full_text*
(
&%475 = fsub double -0.000000e+00, %474
,double8B

	full_text

double %474
{call8Bq
o
	full_textb
`
^%476 = tail call double @llvm.fmuladd.f64(double %471, double 0x3F82D77318FC5048, double %475)
,double8B

	full_text

double %471
,double8B

	full_text

double %475
›getelementptr8B‡
„
	full_textw
u
s%477 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 2, i64 4, i64 2
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %476, double* %477, align 8, !tbaa !8
,double8B

	full_text

double %476
.double*8B

	full_text

double* %477
‡getelementptr8Bt
r
	full_texte
c
a%478 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %32, i64 %354, i64 0, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %32
&i648B

	full_text


i64 %354
Pload8BF
D
	full_text7
5
3%479 = load double, double* %478, align 8, !tbaa !8
.double*8B

	full_text

double* %478
‡getelementptr8Bt
r
	full_texte
c
a%480 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %354, i64 0, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
&i648B

	full_text


i64 %354
Pload8BF
D
	full_text7
5
3%481 = load double, double* %480, align 8, !tbaa !8
.double*8B

	full_text

double* %480
Bfmul8B8
6
	full_text)
'
%%482 = fmul double %481, 4.232000e-01
,double8B

	full_text

double %481
Cfsub8B9
7
	full_text*
(
&%483 = fsub double -0.000000e+00, %482
,double8B

	full_text

double %482
{call8Bq
o
	full_textb
`
^%484 = tail call double @llvm.fmuladd.f64(double %479, double 0x3F82D77318FC5048, double %483)
,double8B

	full_text

double %479
,double8B

	full_text

double %483
›getelementptr8B‡
„
	full_textw
u
s%485 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 2, i64 0, i64 3
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %484, double* %485, align 8, !tbaa !8
,double8B

	full_text

double %484
.double*8B

	full_text

double* %485
‡getelementptr8Bt
r
	full_texte
c
a%486 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %32, i64 %354, i64 1, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %32
&i648B

	full_text


i64 %354
Pload8BF
D
	full_text7
5
3%487 = load double, double* %486, align 8, !tbaa !8
.double*8B

	full_text

double* %486
‡getelementptr8Bt
r
	full_texte
c
a%488 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %354, i64 1, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
&i648B

	full_text


i64 %354
Pload8BF
D
	full_text7
5
3%489 = load double, double* %488, align 8, !tbaa !8
.double*8B

	full_text

double* %488
Bfmul8B8
6
	full_text)
'
%%490 = fmul double %489, 4.232000e-01
,double8B

	full_text

double %489
Cfsub8B9
7
	full_text*
(
&%491 = fsub double -0.000000e+00, %490
,double8B

	full_text

double %490
{call8Bq
o
	full_textb
`
^%492 = tail call double @llvm.fmuladd.f64(double %487, double 0x3F82D77318FC5048, double %491)
,double8B

	full_text

double %487
,double8B

	full_text

double %491
›getelementptr8B‡
„
	full_textw
u
s%493 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 2, i64 1, i64 3
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %492, double* %493, align 8, !tbaa !8
,double8B

	full_text

double %492
.double*8B

	full_text

double* %493
‡getelementptr8Bt
r
	full_texte
c
a%494 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %32, i64 %354, i64 2, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %32
&i648B

	full_text


i64 %354
Pload8BF
D
	full_text7
5
3%495 = load double, double* %494, align 8, !tbaa !8
.double*8B

	full_text

double* %494
‡getelementptr8Bt
r
	full_texte
c
a%496 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %354, i64 2, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
&i648B

	full_text


i64 %354
Pload8BF
D
	full_text7
5
3%497 = load double, double* %496, align 8, !tbaa !8
.double*8B

	full_text

double* %496
Bfmul8B8
6
	full_text)
'
%%498 = fmul double %497, 4.232000e-01
,double8B

	full_text

double %497
Cfsub8B9
7
	full_text*
(
&%499 = fsub double -0.000000e+00, %498
,double8B

	full_text

double %498
{call8Bq
o
	full_textb
`
^%500 = tail call double @llvm.fmuladd.f64(double %495, double 0x3F82D77318FC5048, double %499)
,double8B

	full_text

double %495
,double8B

	full_text

double %499
›getelementptr8B‡
„
	full_textw
u
s%501 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 2, i64 2, i64 3
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %500, double* %501, align 8, !tbaa !8
,double8B

	full_text

double %500
.double*8B

	full_text

double* %501
‡getelementptr8Bt
r
	full_texte
c
a%502 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %32, i64 %354, i64 3, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %32
&i648B

	full_text


i64 %354
Pload8BF
D
	full_text7
5
3%503 = load double, double* %502, align 8, !tbaa !8
.double*8B

	full_text

double* %502
‡getelementptr8Bt
r
	full_texte
c
a%504 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %354, i64 3, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
&i648B

	full_text


i64 %354
Pload8BF
D
	full_text7
5
3%505 = load double, double* %504, align 8, !tbaa !8
.double*8B

	full_text

double* %504
Bfmul8B8
6
	full_text)
'
%%506 = fmul double %505, 4.232000e-01
,double8B

	full_text

double %505
Cfsub8B9
7
	full_text*
(
&%507 = fsub double -0.000000e+00, %506
,double8B

	full_text

double %506
{call8Bq
o
	full_textb
`
^%508 = tail call double @llvm.fmuladd.f64(double %503, double 0x3F82D77318FC5048, double %507)
,double8B

	full_text

double %503
,double8B

	full_text

double %507
~call8Bt
r
	full_texte
c
a%509 = tail call double @llvm.fmuladd.f64(double -4.232000e-01, double 7.500000e-01, double %508)
,double8B

	full_text

double %508
›getelementptr8B‡
„
	full_textw
u
s%510 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 2, i64 3, i64 3
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %509, double* %510, align 8, !tbaa !8
,double8B

	full_text

double %509
.double*8B

	full_text

double* %510
‡getelementptr8Bt
r
	full_texte
c
a%511 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %32, i64 %354, i64 4, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %32
&i648B

	full_text


i64 %354
Pload8BF
D
	full_text7
5
3%512 = load double, double* %511, align 8, !tbaa !8
.double*8B

	full_text

double* %511
‡getelementptr8Bt
r
	full_texte
c
a%513 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %354, i64 4, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
&i648B

	full_text


i64 %354
Pload8BF
D
	full_text7
5
3%514 = load double, double* %513, align 8, !tbaa !8
.double*8B

	full_text

double* %513
Bfmul8B8
6
	full_text)
'
%%515 = fmul double %514, 4.232000e-01
,double8B

	full_text

double %514
Cfsub8B9
7
	full_text*
(
&%516 = fsub double -0.000000e+00, %515
,double8B

	full_text

double %515
{call8Bq
o
	full_textb
`
^%517 = tail call double @llvm.fmuladd.f64(double %512, double 0x3F82D77318FC5048, double %516)
,double8B

	full_text

double %512
,double8B

	full_text

double %516
›getelementptr8B‡
„
	full_textw
u
s%518 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 2, i64 4, i64 3
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %517, double* %518, align 8, !tbaa !8
,double8B

	full_text

double %517
.double*8B

	full_text

double* %518
‡getelementptr8Bt
r
	full_texte
c
a%519 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %32, i64 %354, i64 0, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %32
&i648B

	full_text


i64 %354
Pload8BF
D
	full_text7
5
3%520 = load double, double* %519, align 8, !tbaa !8
.double*8B

	full_text

double* %519
‡getelementptr8Bt
r
	full_texte
c
a%521 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %354, i64 0, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
&i648B

	full_text


i64 %354
Pload8BF
D
	full_text7
5
3%522 = load double, double* %521, align 8, !tbaa !8
.double*8B

	full_text

double* %521
Bfmul8B8
6
	full_text)
'
%%523 = fmul double %522, 4.232000e-01
,double8B

	full_text

double %522
Cfsub8B9
7
	full_text*
(
&%524 = fsub double -0.000000e+00, %523
,double8B

	full_text

double %523
{call8Bq
o
	full_textb
`
^%525 = tail call double @llvm.fmuladd.f64(double %520, double 0x3F82D77318FC5048, double %524)
,double8B

	full_text

double %520
,double8B

	full_text

double %524
›getelementptr8B‡
„
	full_textw
u
s%526 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 2, i64 0, i64 4
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %525, double* %526, align 8, !tbaa !8
,double8B

	full_text

double %525
.double*8B

	full_text

double* %526
‡getelementptr8Bt
r
	full_texte
c
a%527 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %32, i64 %354, i64 1, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %32
&i648B

	full_text


i64 %354
Pload8BF
D
	full_text7
5
3%528 = load double, double* %527, align 8, !tbaa !8
.double*8B

	full_text

double* %527
‡getelementptr8Bt
r
	full_texte
c
a%529 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %354, i64 1, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
&i648B

	full_text


i64 %354
Pload8BF
D
	full_text7
5
3%530 = load double, double* %529, align 8, !tbaa !8
.double*8B

	full_text

double* %529
Bfmul8B8
6
	full_text)
'
%%531 = fmul double %530, 4.232000e-01
,double8B

	full_text

double %530
Cfsub8B9
7
	full_text*
(
&%532 = fsub double -0.000000e+00, %531
,double8B

	full_text

double %531
{call8Bq
o
	full_textb
`
^%533 = tail call double @llvm.fmuladd.f64(double %528, double 0x3F82D77318FC5048, double %532)
,double8B

	full_text

double %528
,double8B

	full_text

double %532
›getelementptr8B‡
„
	full_textw
u
s%534 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 2, i64 1, i64 4
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %533, double* %534, align 8, !tbaa !8
,double8B

	full_text

double %533
.double*8B

	full_text

double* %534
‡getelementptr8Bt
r
	full_texte
c
a%535 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %32, i64 %354, i64 2, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %32
&i648B

	full_text


i64 %354
Pload8BF
D
	full_text7
5
3%536 = load double, double* %535, align 8, !tbaa !8
.double*8B

	full_text

double* %535
‡getelementptr8Bt
r
	full_texte
c
a%537 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %354, i64 2, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
&i648B

	full_text


i64 %354
Pload8BF
D
	full_text7
5
3%538 = load double, double* %537, align 8, !tbaa !8
.double*8B

	full_text

double* %537
Bfmul8B8
6
	full_text)
'
%%539 = fmul double %538, 4.232000e-01
,double8B

	full_text

double %538
Cfsub8B9
7
	full_text*
(
&%540 = fsub double -0.000000e+00, %539
,double8B

	full_text

double %539
{call8Bq
o
	full_textb
`
^%541 = tail call double @llvm.fmuladd.f64(double %536, double 0x3F82D77318FC5048, double %540)
,double8B

	full_text

double %536
,double8B

	full_text

double %540
›getelementptr8B‡
„
	full_textw
u
s%542 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 2, i64 2, i64 4
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %541, double* %542, align 8, !tbaa !8
,double8B

	full_text

double %541
.double*8B

	full_text

double* %542
‡getelementptr8Bt
r
	full_texte
c
a%543 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %32, i64 %354, i64 3, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %32
&i648B

	full_text


i64 %354
Pload8BF
D
	full_text7
5
3%544 = load double, double* %543, align 8, !tbaa !8
.double*8B

	full_text

double* %543
‡getelementptr8Bt
r
	full_texte
c
a%545 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %354, i64 3, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
&i648B

	full_text


i64 %354
Pload8BF
D
	full_text7
5
3%546 = load double, double* %545, align 8, !tbaa !8
.double*8B

	full_text

double* %545
Bfmul8B8
6
	full_text)
'
%%547 = fmul double %546, 4.232000e-01
,double8B

	full_text

double %546
Cfsub8B9
7
	full_text*
(
&%548 = fsub double -0.000000e+00, %547
,double8B

	full_text

double %547
{call8Bq
o
	full_textb
`
^%549 = tail call double @llvm.fmuladd.f64(double %544, double 0x3F82D77318FC5048, double %548)
,double8B

	full_text

double %544
,double8B

	full_text

double %548
›getelementptr8B‡
„
	full_textw
u
s%550 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 2, i64 3, i64 4
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %549, double* %550, align 8, !tbaa !8
,double8B

	full_text

double %549
.double*8B

	full_text

double* %550
‡getelementptr8Bt
r
	full_texte
c
a%551 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %32, i64 %354, i64 4, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %32
&i648B

	full_text


i64 %354
Pload8BF
D
	full_text7
5
3%552 = load double, double* %551, align 8, !tbaa !8
.double*8B

	full_text

double* %551
‡getelementptr8Bt
r
	full_texte
c
a%553 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %34, i64 %354, i64 4, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %34
&i648B

	full_text


i64 %354
Pload8BF
D
	full_text7
5
3%554 = load double, double* %553, align 8, !tbaa !8
.double*8B

	full_text

double* %553
Bfmul8B8
6
	full_text)
'
%%555 = fmul double %554, 4.232000e-01
,double8B

	full_text

double %554
Cfsub8B9
7
	full_text*
(
&%556 = fsub double -0.000000e+00, %555
,double8B

	full_text

double %555
{call8Bq
o
	full_textb
`
^%557 = tail call double @llvm.fmuladd.f64(double %552, double 0x3F82D77318FC5048, double %556)
,double8B

	full_text

double %552
,double8B

	full_text

double %556
~call8Bt
r
	full_texte
c
a%558 = tail call double @llvm.fmuladd.f64(double -4.232000e-01, double 7.500000e-01, double %557)
,double8B

	full_text

double %557
›getelementptr8B‡
„
	full_textw
u
s%559 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %50, i64 2, i64 4, i64 4
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %50
Pstore8BE
C
	full_text6
4
2store double %558, double* %559, align 8, !tbaa !8
,double8B

	full_text

double %558
.double*8B

	full_text

double* %559
(br8B 

	full_text

br label %560
$ret8B

	full_text


ret void
$i328B

	full_text


i32 %3
,double*8B

	full_text


double* %0
,double*8B

	full_text


double* %2
$i328B

	full_text


i32 %5
$i328B

	full_text


i32 %4
,double*8B

	full_text


double* %1
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
5double8B'
%
	full_text

double -4.232000e-01
$i648B

	full_text


i64 32
#i648B

	full_text	

i64 0
#i328B

	full_text	

i32 0
-i648B"
 
	full_text

i64 -4294967296
#i648B

	full_text	

i64 3
#i328B

	full_text	

i32 2
:double8B,
*
	full_text

double 0xBF82D77318FC5048
#i328B

	full_text	

i32 1
%i328B

	full_text
	
i32 625
4double8B&
$
	full_text

double 8.464000e-01
4double8B&
$
	full_text

double 4.232000e-01
4double8B&
$
	full_text

double 1.000000e+00
5double8B'
%
	full_text

double -0.000000e+00
,i648B!

	full_text

i64 4294967296
$i328B

	full_text


i32 -2
#i648B

	full_text	

i64 1
$i328B

	full_text


i32 -1
#i648B

	full_text	

i64 2
4double8B&
$
	full_text

double 7.500000e-01
#i648B

	full_text	

i64 4
:double8B,
*
	full_text

double 0x3F82D77318FC5048
&i328B

	full_text


i32 1875        	
 		                       !" !$ ## %& %' %% () (( *+ *, ** -. -- /0 // 12 11 34 33 56 55 78 77 9: 99 ;< ;; => == ?@ ?? AB AA CD CC EF EE GH GI GG JK JJ LM LN LL OP OO QR QQ ST SS UV UW UU XY XX Z[ ZZ \] \^ \\ _` _a __ bc bd bb ef ee gh gi gg jk jj lm ll no nn pq pr pp st su ss vw vx vv yz y{ yy |} || ~ ~	€ ~~ ‚  ƒ„ ƒƒ …
† …… ‡ˆ ‡
‰ ‡‡ Š‹ Š
Œ ŠŠ Ž 
  ‘ 
’  “” ““ •– •
— •• ˜™ ˜˜ š› šš œ
 œœ žŸ ž
  žž ¡¢ ¡
£ ¡¡ ¤¥ ¤
¦ ¤¤ §¨ §
© §§ ª« ªª ¬­ ¬
® ¬¬ ¯° ¯¯ ±² ±± ³
´ ³³ µ¶ µ
· µµ ¸¹ ¸
º ¸¸ »¼ »
½ »» ¾¿ ¾
À ¾¾ ÁÂ ÁÁ ÃÄ Ã
Å ÃÃ ÆÇ ÆÆ ÈÉ ÈÈ Ê
Ë ÊÊ ÌÍ Ì
Î ÌÌ ÏÐ Ï
Ñ ÏÏ ÒÓ Ò
Ô ÒÒ ÕÖ Õ
× ÕÕ ØÙ ØØ ÚÛ Ú
Ü ÚÚ ÝÞ ÝÝ ßà ßß á
â áá ãä ã
å ãã æ
ç ææ èé è
ê èè ëì ë
í ëë îï î
ð îî ñò ññ óô ó
õ óó ö÷ öö øù øø ú
û úú üý ü
þ üü ÿ€ ÿ
 ÿÿ ‚ƒ ‚
„ ‚‚ …† …
‡ …… ˆ‰ ˆˆ Š‹ Š
Œ ŠŠ Ž    ‘
’ ‘‘ “” “
• ““ –— –
˜ –– ™š ™
› ™™ œ œ
ž œœ Ÿ  ŸŸ ¡¢ ¡
£ ¡¡ ¤¥ ¤¤ ¦§ ¦¦ ¨
© ¨¨ ª« ª
¬ ªª ­® ­
¯ ­­ °± °
² °° ³´ ³
µ ³³ ¶· ¶¶ ¸¹ ¸
º ¸¸ »¼ »» ½¾ ½½ ¿
À ¿¿ ÁÂ Á
Ã ÁÁ ÄÅ Ä
Æ ÄÄ ÇÈ Ç
É ÇÇ ÊË Ê
Ì ÊÊ ÍÎ ÍÍ ÏÐ Ï
Ñ ÏÏ ÒÓ ÒÒ ÔÕ ÔÔ Ö
× ÖÖ ØÙ Ø
Ú ØØ ÛÜ Û
Ý ÛÛ Þß Þ
à ÞÞ áâ á
ã áá äå ää æç æ
è ææ éê éé ëì ëë í
î íí ïð ï
ñ ïï ò
ó òò ôõ ô
ö ôô ÷ø ÷
ù ÷÷ úû ú
ü úú ýþ ýý ÿ€ ÿ
 ÿÿ ‚ƒ ‚‚ „… „„ †
‡ †† ˆ‰ ˆ
Š ˆˆ ‹Œ ‹
 ‹‹ Ž Ž
 ŽŽ ‘’ ‘
“ ‘‘ ”• ”” –— –
˜ –– ™š ™™ ›œ ›› 
ž  Ÿ  Ÿ
¡ ŸŸ ¢£ ¢
¤ ¢¢ ¥¦ ¥
§ ¥¥ ¨© ¨
ª ¨¨ «¬ «« ­® ­
¯ ­­ °± °° ²³ ²² ´
µ ´´ ¶· ¶
¸ ¶¶ ¹º ¹
» ¹¹ ¼½ ¼
¾ ¼¼ ¿À ¿
Á ¿¿ ÂÃ ÂÂ ÄÅ Ä
Æ ÄÄ ÇÈ ÇÇ ÉÊ ÉÉ Ë
Ì ËË ÍÎ Í
Ï ÍÍ ÐÑ Ð
Ò ÐÐ ÓÔ Ó
Õ ÓÓ Ö× Ö
Ø ÖÖ ÙÚ ÙÙ ÛÜ Û
Ý ÛÛ Þß ÞÞ àá àà â
ã ââ äå ä
æ ää çè ç
é çç êë ê
ì êê íî í
ï íí ðñ ðð òó ò
ô òò õö õõ ÷ø ÷÷ ù
ú ùù ûü û
ý ûû þ
ÿ þþ € €
‚ €€ ƒ„ ƒ
… ƒƒ †‡ †
ˆ †† ‰Š ‰‰ ‹Œ ‹
 ‹‹ Ž ŽŽ ‘  ’
“ ’’ ”• ”
– ”” —˜ —
™ —— š› š
œ šš ž 
Ÿ   ¡    ¢£ ¢
¤ ¢¢ ¥¦ ¥¥ §¨ §§ ©
ª ©© «¬ «
­ «« ®¯ ®
° ®® ±² ±
³ ±± ´µ ´
¶ ´´ ·¸ ·· ¹º ¹
» ¹¹ ¼½ ¼¼ ¾¿ ¾¾ À
Á ÀÀ ÂÃ Â
Ä ÂÂ ÅÆ Å
Ç ÅÅ ÈÉ È
Ê ÈÈ ËÌ Ë
Í ËË ÎÏ ÎÎ ÐÑ Ð
Ò ÐÐ ÓÔ ÓÓ ÕÖ ÕÕ ×
Ø ×× ÙÚ Ù
Û ÙÙ ÜÝ Ü
Þ ÜÜ ßà ß
á ßß âã â
ä ââ åæ åå çè ç
é çç êë êê ìí ìì î
ï îî ðñ ð
ò ðð óô ó
õ óó ö÷ ö
ø öö ùú ù
û ùù üý üü þÿ þ
€ þþ ‚  ƒ„ ƒƒ …
† …… ‡ˆ ‡
‰ ‡‡ Š
‹ ŠŠ Œ Œ
Ž ŒŒ  
‘  ’“ ’
” ’’ •– •• —˜ —— ™
š ™™ ›œ ›
 ›› žŸ ž
  žž ¡¢ ¡
£ ¡¡ ¤¥ ¤¤ ¦§ ¦¦ ¨© ¨
ª ¨¨ «¬ «
­ «« ®¯ ®
° ®® ±² ±± ³´ ³³ µ¶ µ
· µµ ¸¹ ¸
º ¸¸ »¼ »
½ »» ¾¿ ¾¾ ÀÁ ÀÀ ÂÃ Â
Ä ÂÂ ÅÆ Å
Ç ÅÅ ÈÉ È
Ê ÈÈ ËÌ ËË ÍÎ ÍÍ ÏÐ Ï
Ñ ÏÏ ÒÓ Ò
Ô ÒÒ ÕÖ Õ
× ÕÕ ØÙ ØØ ÚÛ ÚÚ ÜÝ Ü
Þ ÜÜ ßà ß
á ßß âã â
ä ââ åæ åå çè çç é
ê éé ëì ë
í ëë îï î
ð îî ñò ñ
ó ññ ôõ ôô ö÷ öö øù ø
ú øø ûü û
ý ûû þÿ þ
€ þþ ‚  ƒ„ ƒƒ …† …
‡ …… ˆ‰ ˆ
Š ˆˆ ‹Œ ‹
 ‹‹ Ž ŽŽ ‘  ’“ ’
” ’’ •– •
— •• ˜™ ˜
š ˜˜ ›œ ›› ž  Ÿ  Ÿ
¡ ŸŸ ¢£ ¢
¤ ¢¢ ¥¦ ¥
§ ¥¥ ¨© ¨¨ ª« ªª ¬­ ¬
® ¬¬ ¯° ¯
± ¯¯ ²³ ²
´ ²² µ¶ µµ ·¸ ·· ¹
º ¹¹ »¼ »
½ »» ¾¿ ¾
À ¾¾ ÁÂ Á
Ã ÁÁ ÄÅ ÄÄ ÆÇ ÆÆ ÈÉ È
Ê ÈÈ ËÌ Ë
Í ËË ÎÏ Î
Ð ÎÎ ÑÒ ÑÑ ÓÔ ÓÓ ÕÖ Õ
× ÕÕ ØÙ Ø
Ú ØØ ÛÜ Û
Ý ÛÛ Þß ÞÞ àá àà âã â
ä ââ åæ å
ç åå èé è
ê èè ëì ëë íî íí ïð ï
ñ ïï òó ò
ô òò õö õ
÷ õõ øù øø úû úú üý ü
þ üü ÿ€ ÿ
 ÿÿ ‚ƒ ‚
„ ‚‚ …† …… ‡ˆ ‡‡ ‰
Š ‰‰ ‹Œ ‹
 ‹‹ Ž Ž
 ŽŽ ‘’ ‘
“ ‘‘ ”• ”” –— –– ˜™ ˜
š ˜˜ ›œ ›
 ›› žŸ ž
  žž ¡¢ ¡¡ £¤ ££ ¥¦ ¥
§ ¥¥ ¨© ¨
ª ¨¨ «¬ «
­ «« ®¯ ®® °± °° ²³ ²
´ ²² µ¶ µ
· µµ ¸¹ ¸
º ¸¸ »¼ »» ½¾ ½½ ¿À ¿
Á ¿¿ ÂÃ Â
Ä ÂÂ ÅÆ Å
Ç ÅÅ ÈÉ ÈÈ ÊË ÊÊ ÌÍ Ì
Î ÌÌ ÏÐ Ï
Ñ ÏÏ ÒÓ Ò
Ô ÒÒ ÕÖ ÕÕ ×Ø ×× Ù
Ú ÙÙ ÛÜ Û
Ý ÛÛ Þß Þ
à ÞÞ áâ áá ãä ãã åæ å
ç åå èé èè êë ê
ì êê íî íí ïð ïï ñ
ò ññ óô ó
õ óó ö
÷ öö øù ø
ú øø ûü û
ý ûû þÿ þ
€ þþ ‚  ƒ„ ƒ
… ƒƒ †‡ †† ˆ‰ ˆˆ Š
‹ ŠŠ Œ Œ
Ž ŒŒ  
‘  ’“ ’
” ’’ •– •
— •• ˜™ ˜˜ š› š
œ šš ž  Ÿ  ŸŸ ¡
¢ ¡¡ £¤ £
¥ ££ ¦§ ¦
¨ ¦¦ ©ª ©
« ©© ¬­ ¬
® ¬¬ ¯° ¯¯ ±² ±
³ ±± ´µ ´´ ¶· ¶¶ ¸
¹ ¸¸ º» º
¼ ºº ½¾ ½
¿ ½½ ÀÁ À
Â ÀÀ ÃÄ Ã
Å ÃÃ ÆÇ ÆÆ ÈÉ È
Ê ÈÈ ËÌ ËË ÍÎ ÍÍ Ï
Ð ÏÏ ÑÒ Ñ
Ó ÑÑ ÔÕ Ô
Ö ÔÔ ×Ø ×
Ù ×× ÚÛ Ú
Ü ÚÚ ÝÞ ÝÝ ßà ß
á ßß âã ââ äå ää æ
ç ææ èé è
ê èè ëì ë
í ëë îï î
ð îî ñò ñ
ó ññ ôõ ôô ö÷ ö
ø öö ùú ùù ûü ûû ý
þ ýý ÿ€	 ÿ
	 ÿÿ ‚	
ƒ	 ‚	‚	 „	…	 „	
†	 „	„	 ‡	ˆ	 ‡	
‰	 ‡	‡	 Š	‹	 Š	
Œ	 Š	Š	 	Ž	 		 		 	
‘	 		 ’	“	 ’	’	 ”	•	 ”	”	 –	
—	 –	–	 ˜	™	 ˜	
š	 ˜	˜	 ›	œ	 ›	
	 ›	›	 ž	Ÿ	 ž	
 	 ž	ž	 ¡	¢	 ¡	
£	 ¡	¡	 ¤	¥	 ¤	¤	 ¦	§	 ¦	
¨	 ¦	¦	 ©	ª	 ©	©	 «	¬	 «	«	 ­	
®	 ­	­	 ¯	°	 ¯	
±	 ¯	¯	 ²	³	 ²	
´	 ²	²	 µ	¶	 µ	
·	 µ	µ	 ¸	¹	 ¸	
º	 ¸	¸	 »	¼	 »	»	 ½	¾	 ½	
¿	 ½	½	 À	Á	 À	À	 Â	Ã	 Â	Â	 Ä	
Å	 Ä	Ä	 Æ	Ç	 Æ	
È	 Æ	Æ	 É	Ê	 É	
Ë	 É	É	 Ì	Í	 Ì	
Î	 Ì	Ì	 Ï	Ð	 Ï	
Ñ	 Ï	Ï	 Ò	Ó	 Ò	Ò	 Ô	Õ	 Ô	
Ö	 Ô	Ô	 ×	Ø	 ×	×	 Ù	Ú	 Ù	Ù	 Û	
Ü	 Û	Û	 Ý	Þ	 Ý	
ß	 Ý	Ý	 à	á	 à	
â	 à	à	 ã	ä	 ã	
å	 ã	ã	 æ	ç	 æ	
è	 æ	æ	 é	ê	 é	é	 ë	ì	 ë	
í	 ë	ë	 î	ï	 î	î	 ð	ñ	 ð	ð	 ò	
ó	 ò	ò	 ô	õ	 ô	
ö	 ô	ô	 ÷	ø	 ÷	
ù	 ÷	÷	 ú	û	 ú	
ü	 ú	ú	 ý	þ	 ý	
ÿ	 ý	ý	 €

 €
€
 ‚
ƒ
 ‚

„
 ‚
‚
 …
†
 …
…
 ‡
ˆ
 ‡
‡
 ‰

Š
 ‰
‰
 ‹
Œ
 ‹


 ‹
‹
 Ž


 Ž
Ž
 
‘
 

’
 

 “
”
 “

•
 “
“
 –
—
 –

˜
 –
–
 ™
š
 ™
™
 ›
œ
 ›


 ›
›
 ž
Ÿ
 ž
ž
  
¡
  
 
 ¢

£
 ¢
¢
 ¤
¥
 ¤

¦
 ¤
¤
 §
¨
 §

©
 §
§
 ª
«
 ª

¬
 ª
ª
 ­
®
 ­

¯
 ­
­
 °
±
 °
°
 ²
³
 ²

´
 ²
²
 µ
¶
 µ
µ
 ·
¸
 ·
·
 ¹

º
 ¹
¹
 »
¼
 »

½
 »
»
 ¾
¿
 ¾

À
 ¾
¾
 Á
Â
 Á

Ã
 Á
Á
 Ä
Å
 Ä

Æ
 Ä
Ä
 Ç
È
 Ç
Ç
 É
Ê
 É

Ë
 É
É
 Ì
Í
 Ì
Ì
 Î
Ï
 Î
Î
 Ð

Ñ
 Ð
Ð
 Ò
Ó
 Ò

Ô
 Ò
Ò
 Õ
Ö
 Õ

×
 Õ
Õ
 Ø
Ù
 Ø

Ú
 Ø
Ø
 Û
Ü
 Û

Ý
 Û
Û
 Þ
ß
 Þ
Þ
 à
á
 à

â
 à
à
 ã
ä
 ã
ã
 å
æ
 å
å
 ç

è
 ç
ç
 é
ê
 é

ë
 é
é
 ì
í
 ì

î
 ì
ì
 ï
ð
 ï

ñ
 ï
ï
 ò
ó
 ò

ô
 ò
ò
 õ
ö
 õ
õ
 ÷
ø
 ÷

ù
 ÷
÷
 ú
û
 ú
ú
 ü
ý
 ü
ü
 þ

ÿ
 þ
þ
 € €
‚ €€ ƒ„ ƒ
… ƒƒ †‡ †
ˆ †† ‰Š ‰
‹ ‰‰ Œ ŒŒ Ž Ž
 ŽŽ ‘’ ‘‘ “” ““ •
– •• —˜ —
™ —— š
› šš œ œ
ž œœ Ÿ  Ÿ
¡ ŸŸ ¢£ ¢
¤ ¢¢ ¥¦ ¥¥ §¨ §
© §§ ª« ªª ¬­ ¬¬ ®
¯ ®® °± °
² °° ³´ ³
µ ³³ ¶· ¶
¸ ¶¶ ¹º ¹
» ¹¹ ¼½ ¼¼ ¾¿ ¾
À ¾¾ ÁÂ ÁÁ ÃÄ ÃÃ Å
Æ ÅÅ ÇÈ Ç
É ÇÇ ÊË Ê
Ì ÊÊ ÍÎ Í
Ï ÍÍ ÐÑ Ð
Ò ÐÐ ÓÔ ÓÓ ÕÖ Õ
× ÕÕ ØÙ ØØ ÚÛ ÚÚ Ü
Ý ÜÜ Þß Þ
à ÞÞ áâ á
ã áá äå ä
æ ää çè ç
é çç êë êê ìí ì
î ìì ïð ïï ñò ññ ó
ô óó õö õ
÷ õõ øù ø
ú øø ûü û
ý ûû þÿ þ
€ þþ ‚  ƒ„ ƒ
… ƒƒ †‡ †† ˆ‰ ˆˆ Š
‹ ŠŠ Œ Œ
Ž ŒŒ  
‘  ’“ ’
” ’’ •– •
— •• ˜™ ˜˜ š› š
œ šš ž  Ÿ  ŸŸ ¡
¢ ¡¡ £¤ £
¥ ££ ¦
§ ¦¦ ¨© ¨
ª ¨¨ «¬ «
­ «« ®° ± 1² =³ ´ µ 5    
      	       " $# & '	 )( +% ,* .- 0/ 21 4/ 65 8* :9 <; >= @ BA DC F3 HE IG K7 ME NL PO RQ TJ VS WU YA [? ]Z ^X `\ a3 cE db f7 hE ig kj ml oe qn r? tZ up ws x3 zE {y }7 E €~ ‚ „ƒ †| ˆ… ‰? ‹Z Œ‡ ŽŠ 3 ‘E ’ ”7 –E —• ™˜ ›š “ Ÿœ  ? ¢Z £ž ¥¡ ¦3 ¨E ©§ «7 ­E ®¬ °¯ ²± ´ª ¶³ ·? ¹Z ºµ ¼¸ ½3 ¿E À¾ Â7 ÄE ÅÃ ÇÆ ÉÈ ËÁ ÍÊ Î? ÐZ ÑÌ ÓÏ Ô3 ÖE ×Õ Ù7 ÛE ÜÚ ÞÝ àß âØ äá åã ç? éZ êæ ìè í3 ïE ðî ò7 ôE õó ÷ö ùø ûñ ýú þ? €Z ü ƒÿ „3 †E ‡… ‰7 ‹E ŒŠ Ž  ’ˆ ”‘ •? —Z ˜“ š– ›3 E žœ  7 ¢E £¡ ¥¤ §¦ ©Ÿ «¨ ¬? ®Z ¯ª ±­ ²3 ´E µ³ ·7 ¹E º¸ ¼» ¾½ À¶ Â¿ Ã? ÅZ ÆÁ ÈÄ É3 ËE ÌÊ Î7 ÐE ÑÏ ÓÒ ÕÔ ×Í ÙÖ Ú? ÜZ ÝØ ßÛ à3 âE ãá å7 çE èæ êé ìë îä ðí ñï ó? õZ öò øô ù3 ûE üú þ7 €E ÿ ƒ‚ …„ ‡ý ‰† Š? ŒZ ˆ ‹ 3 ’E “‘ •7 —E ˜– š™ œ› ž”   ¡? £Z ¤Ÿ ¦¢ §3 ©E ª¨ ¬7 ®E ¯­ ±° ³² µ« ·´ ¸? ºZ »¶ ½¹ ¾3 ÀE Á¿ Ã7 ÅE ÆÄ ÈÇ ÊÉ ÌÂ ÎË Ï? ÑZ ÒÍ ÔÐ Õ3 ×E ØÖ Ú7 ÜE ÝÛ ßÞ áà ãÙ åâ æ? èZ éä ëç ì3 îE ïí ñ7 óE ôò öõ ø÷ úð üù ýû ÿ? Z ‚þ „€ …3 ‡E ˆ† Š7 ŒE ‹ Ž ‘ “‰ •’ –? ˜Z ™” ›— œ3 žE Ÿ ¡7 £E ¤¢ ¦¥ ¨§ ª  ¬© ­? ¯Z °« ²® ³3 µE ¶´ ¸7 ºE »¹ ½¼ ¿¾ Á· ÃÀ Ä? ÆZ ÇÂ ÉÅ Ê3 ÌE ÍË Ï7 ÑE ÒÐ ÔÓ ÖÕ ØÎ Ú× Û? ÝZ ÞÙ àÜ á3 ãE äâ æ7 èE éç ëê íì ïå ñî ò? ôZ õð ÷ó ø3 úE ûù ý7 ÿE €þ ‚ „ƒ †ü ˆ… ‰‡ ‹? Z ŽŠ Œ ‘7 “Z ”’ –• ˜— š? œZ ™ Ÿ›  7 ¢Z £¡ ¥¤ §? ©Z ª¦ ¬¨ ­7 ¯Z °® ²± ´? ¶Z ·³ ¹µ º7 ¼Z ½» ¿¾ Á? ÃZ ÄÀ ÆÂ Ç7 ÉZ ÊÈ ÌË Î? ÐZ ÑÍ ÓÏ Ô7 ÖZ ×Õ ÙØ Û? ÝZ ÞÚ àÜ á7 ãZ äâ æå èç ê? ìZ íé ïë ð7 òZ óñ õô ÷? ùZ úö üø ý7 ÿZ €þ ‚ „? †Z ‡ƒ ‰… Š7 ŒZ ‹ Ž ‘? “Z ” –’ —7 ™Z š˜ œ› ž?  Z ¡ £Ÿ ¤7 ¦Z §¥ ©¨ «? ­Z ®ª °¬ ±7 ³Z ´² ¶µ ¸· º? ¼Z ½¹ ¿» À7 ÂZ ÃÁ ÅÄ Ç? ÉZ ÊÆ ÌÈ Í7 ÏZ ÐÎ ÒÑ Ô? ÖZ ×Ó ÙÕ Ú7 ÜZ ÝÛ ßÞ á? ãZ äà æâ ç7 éZ êè ìë î? ðZ ñí óï ô7 öZ ÷õ ùø û? ýZ þú €ü 7 ƒZ „‚ †… ˆ‡ Š? ŒZ ‰ ‹ 7 ’Z “‘ •” —? ™Z š– œ˜ 7 ŸZ  ž ¢¡ ¤? ¦Z §£ ©¥ ª7 ¬Z ­« ¯® ±? ³Z ´° ¶² ·7 ¹Z º¸ ¼» ¾? ÀZ Á½ Ã¿ Ä7 ÆZ ÇÅ ÉÈ Ë? ÍZ ÎÊ ÐÌ Ñ7 ÓZ ÔÒ ÖÕ Ø× Ú? ÜZ ÝÙ ßÛ àA âá ä3 æã çå é7 ëã ìê îí ðï òè ôñ õó ÷? ùZ úö üø ý3 ÿã €þ ‚7 „ã …ƒ ‡† ‰ˆ ‹ Š Ž? Z ‘Œ “ ”3 –ã —• ™7 ›ã œš ž  Ÿ ¢˜ ¤¡ ¥? §Z ¨£ ª¦ «3 ­ã ®¬ °7 ²ã ³± µ´ ·¶ ¹¯ »¸ ¼? ¾Z ¿º Á½ Â3 Äã ÅÃ Ç7 Éã ÊÈ ÌË ÎÍ ÐÆ ÒÏ Ó? ÕZ ÖÑ ØÔ Ù3 Ûã ÜÚ Þ7 àã áß ãâ åä çÝ éæ ê? ìZ íè ïë ð3 òã óñ õ7 ÷ã øö úù üû þô €	ý 	ÿ ƒ	? …	Z †	‚	 ˆ	„	 ‰	3 ‹	ã Œ	Š	 Ž	7 	ã ‘		 “	’	 •	”	 —		 ™	–	 š	? œ	Z 	˜	 Ÿ	›	  	3 ¢	ã £	¡	 ¥	7 §	ã ¨	¦	 ª	©	 ¬	«	 ®	¤	 °	­	 ±	? ³	Z ´	¯	 ¶	²	 ·	3 ¹	ã º	¸	 ¼	7 ¾	ã ¿	½	 Á	À	 Ã	Â	 Å	»	 Ç	Ä	 È	? Ê	Z Ë	Æ	 Í	É	 Î	3 Ð	ã Ñ	Ï	 Ó	7 Õ	ã Ö	Ô	 Ø	×	 Ú	Ù	 Ü	Ò	 Þ	Û	 ß	? á	Z â	Ý	 ä	à	 å	3 ç	ã è	æ	 ê	7 ì	ã í	ë	 ï	î	 ñ	ð	 ó	é	 õ	ò	 ö	? ø	Z ù	ô	 û	÷	 ü	3 þ	ã ÿ	ý	 
7 ƒ
ã „
‚
 †
…
 ˆ
‡
 Š
€
 Œ
‰
 
‹
 
? ‘
Z ’
Ž
 ”

 •
3 —
ã ˜
–
 š
7 œ
ã 
›
 Ÿ
ž
 ¡
 
 £
™
 ¥
¢
 ¦
? ¨
Z ©
¤
 «
§
 ¬
3 ®
ã ¯
­
 ±
7 ³
ã ´
²
 ¶
µ
 ¸
·
 º
°
 ¼
¹
 ½
? ¿
Z À
»
 Â
¾
 Ã
3 Å
ã Æ
Ä
 È
7 Ê
ã Ë
É
 Í
Ì
 Ï
Î
 Ñ
Ç
 Ó
Ð
 Ô
? Ö
Z ×
Ò
 Ù
Õ
 Ú
3 Ü
ã Ý
Û
 ß
7 á
ã â
à
 ä
ã
 æ
å
 è
Þ
 ê
ç
 ë
? í
Z î
é
 ð
ì
 ñ
3 ó
ã ô
ò
 ö
7 ø
ã ù
÷
 û
ú
 ý
ü
 ÿ
õ
 þ
 ‚? „Z …€ ‡ƒ ˆ3 Šã ‹‰ 7 ã Ž ’‘ ”“ –Œ ˜• ™— ›? Z žš  œ ¡3 £ã ¤¢ ¦7 ¨ã ©§ «ª ­¬ ¯¥ ±® ²? ´Z µ° ·³ ¸3 ºã »¹ ½7 ¿ã À¾ ÂÁ ÄÃ Æ¼ ÈÅ É? ËZ ÌÇ ÎÊ Ï3 Ñã ÒÐ Ô7 Öã ×Õ ÙØ ÛÚ ÝÓ ßÜ à? âZ ãÞ åá æ3 èã éç ë7 íã îì ðï òñ ôê öó ÷? ùZ úõ üø ý3 ÿã €þ ‚7 „ã …ƒ ‡† ‰ˆ ‹ Š Ž? Z ‘Œ “ ”3 –ã —• ™7 ›ã œš ž  Ÿ ¢˜ ¤¡ ¥£ §? ©Z ª¦ ¬¨ ­ ¯ ! ¯! #® ¯ ¯ ¶¶ ··Ÿ ·· Ÿô	 ·· ô	€ ·· €Ñ ·· Ñº ·· ºÙ ·· Ù£ ·· £¶ ·· ¶é ·· éU ·· Uã ·· ãæ ·· æŠ ·· ŠÝ	 ·· Ý	« ·· «” ·· ”‡ ·· ‡ó ·· óö ·· öÌ ·· ÌÍ ·· Í° ·· °µ ·· µŒ ·· Œð ·· ð ¶¶ ˆ ·· ˆü ·· üØ ·· Ø ¶¶ Ç ·· ÇÙ ·· ÙÒ
 ·· Ò
— ·· —è ·· èª ·· ªŒ ·· Œp ·· pÿ ·· ÿž ·· ž‰ ·· ‰¤
 ·· ¤
¦ ·· ¦× ·· ×‡ ·· ‡£ ·· £Ž
 ·· Ž
ò ·· òï ·· ïÁ ·· Á· ·· ·‡ ·· ‡¹ ·· ¹Â ·· Â‹
 ·· ‹
X ·· X ¶¶ û ·· û“ ·· “Æ	 ·· Æ	»
 ·· »
— ·· —é
 ·· é
õ ·· õ‚	 ·· ‚	ç ·· çä ·· äÞ ·· Þ™ ·· ™þ ·· þ¯	 ·· ¯	˜	 ·· ˜	š ·· š¸ X¸ æ¸ ò¸ þ¸ Š¸ ö¸ ‚	¸ Ž
¸ š¸ ¦	¹ A	¹ E	¹ Z
¹ ã	º G	º G	º L	º L	º \	º \	º \	º b	º g	º s	º s	º y	º ~
º Š
º Š
º 
º •
º ¡
º ¡
º §
º ¬
º ¸
º ¸
º ¾
º Ã
º Ï
º Ï
º è
º ÿ
º –
º ­
º ³
º ¸
º Ä
º Ä
º Û
º ô
º ‹
º ¢
º ¨
º ­
º ¹
º ¹
º Ð
º ç
º €
º —
º 
º ¢
º ®
º ®
º Å
º Ü
º ó
º Œ
º ’
º ’
º ›
º ›
º ¡
º ¨
º ®
º µ
º »
º Â
º È
º Ï
º Õ
º Ü
º ˜
º Ÿ
º Û
º â
º ž
º ¥
º å
º å
º ê
º ê
º ø
º ø
º þ
º ƒ
º 
º •
º š
º ¦
º ¬
º ±
º ½
º Ã
º È
º Ô
º Ú
º ß
º ë
º Ï	
º Ô	
º à	
º Ä

º É

º Õ

º ¹
º ¾
º Ê» 	¼ C
½ 
½ •
½ ¡
½ …
½ Š
½ –
½ ú
½ ÿ
½ ‹
½ ¨
½ ­
½ ¹
½ ¿
½ Ä
½ Ð
½ Ö
½ Û
½ ç
½ í
½ í
½ ò
½ ò
½ €
½ €
½ †
½ ‹
½ —
½ â
½ ç
½ ó
½ »
½ Â
½ þ
½ …
½ Á
½ È
½ Û
½ â
½ è
½ ï
½ õ
½ ü
½ ‚
½ ‚
½ ‹
½ ‹
½ ‘
½ ˜
½ Å
½ Ì
½ ¬
½ ±
½ ½
½ ¡	
½ ¦	
½ ²	
½ –

½ ›

½ §

½ Ä

½ É

½ Õ

½ Û

½ à

½ ì

½ ò

½ ÷

½ ƒ
½ ‰
½ ‰
½ Ž
½ Ž
½ œ
½ œ
½ ¢
½ §
½ ³
½ þ
½ ƒ
½ ¾ 	¿ U	¿ p
¿ ‡
¿ ž
¿ µ
¿ Ì
¿ ã
¿ ü
¿ “
¿ ª
¿ Á
¿ Ø
¿ ï
¿ ˆ
¿ Ÿ
¿ ¶
¿ Í
¿ ä
¿ û
¿ ”
¿ «
¿ Â
¿ Ù
¿ ð
¿ ‡À 	Á -
Â —Â ™
Â ¦
Â ³
Â À
Â Í
Â Ú
Â çÂ é
Â ö
Â ƒ
Â 
Â 
Â ª
Â ·Â ¹
Â Æ
Â Ó
Â à
Â í
Â ú
Â ‡Â ‰
Â –
Â £
Â °
Â ½
Â Ê
Â ×Â Ù	Ã Q	Ã l
Ã ƒ
Ã š
Ã ±
Ã È
Ã ß
Ã ø
Ã 
Ã ¦
Ã ½
Ã Ô
Ã ë
Ã „
Ã ›
Ã ²
Ã É
Ã à
Ã ÷
Ã 
Ã §
Ã ¾
Ã Õ
Ã ì
Ã ƒ
Ã ï
Ã ˆ
Ã Ÿ
Ã ¶
Ã Í
Ã ä
Ã û
Ã ”	
Ã «	
Ã Â	
Ã Ù	
Ã ð	
Ã ‡

Ã  

Ã ·

Ã Î

Ã å

Ã ü

Ã “
Ã ¬
Ã Ã
Ã Ú
Ã ñ
Ã ˆ
Ã Ÿ
Ä —
Ä ç
Ä ·
Ä ‡
Ä ×Å SÅ nÅ …Å œÅ ³Å ÊÅ áÅ úÅ ‘Å ¨Å ¿Å ÖÅ íÅ †Å Å ´Å ËÅ âÅ ùÅ ’Å ©Å ÀÅ ×Å îÅ …Å ñÅ ŠÅ ¡Å ¸Å ÏÅ æÅ ýÅ –	Å ­	Å Ä	Å Û	Å ò	Å ‰
Å ¢
Å ¹
Å Ð
Å ç
Å þ
Å •Å ®Å ÅÅ ÜÅ óÅ ŠÅ ¡
Æ á	Ç 	Ç 	Ç 	È 	È 	È 	È b	È g	È s
È ¾
È Ã
È Ï
È Õ
È Õ
È Ú
È Ú
È è
È è
È î
È ó
È ÿ
È …
È Š
È –
È œ
È ¡
È ­
È Ê
È Ï
È Û
È ¿
È Ä
È Ð
È ´
È ¹
È Å
È ›
È ¡
È ¨
È ¨
È µ
È Â
È Ï
È Õ
È Ü
È Ü
È â
È â
È ë
È ë
È ë
È ñ
È ø
È ø
È þ
È …
È …
È ‹
È ’
È ’
È Ÿ
È ¥
È ¬
È ¬
È »
È È
È Õ
È â
È è
È ï
È ï
È ü
È ‹
È ˜
È ¥
È «
È ²
È ²
È ¿
È Ì
È Û
È þ
È ƒ
È 
È Ú
È ß
È ë
È ñ
È ñ
È ö
È ö
È „	
È „	
È Š	
È 	
È ›	
È ¡	
È ¦	
È ²	
È ¸	
È ½	
È É	
È æ	
È ë	
È ÷	
È Û

È à

È ì

È Ð
È Õ
È á	É #	É (	Ê y	Ê ~
Ê Š
Ê î
Ê ó
Ê ÿ
Ê ³
Ê ¸
Ê Ä
Ê Ê
Ê Ï
Ê Û
Ê á
Ê á
Ê æ
Ê æ
Ê ô
Ê ô
Ê ú
Ê ÿ
Ê ‹
Ê ‘
Ê –
Ê ¢
Ê Ö
Ê Û
Ê ç
Ê Ë
Ê Ð
Ê Ü
Ê ®
Ê µ
Ê ñ
Ê ø
Ê ˜
Ê Ÿ
Ê ¥
Ê ¬
Ê ²
Ê ²
Ê »
Ê »
Ê Á
Ê È
Ê Î
Ê Õ
Ê õ
Ê ü
Ê ¸
Ê ¿
Ê ø
Ê 
Ê •
Ê š
Ê ¦
Ê ¦
Ê ½
Ê Ô
Ê ë
Ê „	
Ê Š	
Ê 	
Ê ›	
Ê ›	
Ê ²	
Ê É	
Ê Ï	
Ê Ô	
Ê à	
Ê à	
Ê æ	
Ê ë	
Ê ÷	
Ê ÷	
Ê ý	
Ê ý	
Ê ‚

Ê ‚

Ê 

Ê 

Ê 

Ê –

Ê ›

Ê §

Ê §

Ê ­

Ê ²

Ê ¾

Ê ¾

Ê Õ

Ê ì

Ê ò

Ê ÷

Ê ƒ
Ê ƒ
Ê œ
Ê ³
Ê Ê
Ê á
Ê ç
Ê ì
Ê ø
Ê ø
Ê 
Ê ¨	Ë X
Ë æ
Ë ò
Ë þ
Ë Š
Ë ™
Ë é
Ë ¹
Ë ‰
Ë Ù
Ë ö
Ë ‚	
Ë Ž

Ë š
Ë ¦
Ì §
Ì ¬
Ì ¸
Ì œ
Ì ¡
Ì ­
Ì ‘
Ì –
Ì ¢
Ì †
Ì ‹
Ì —
Ì 
Ì ¢
Ì ®
Ì ´
Ì ¹
Ì Å
Ì Ë
Ì Ð
Ì Ü
Ì â
Ì ç
Ì ó
Ì ù
Ì ù
Ì þ
Ì þ
Ì Œ
Ì Œ
Ì È
Ì Ï
Ì ‹
Ì ’
Ì Î
Ì Õ
Ì ‘
Ì ˜
Ì ž
Ì ¥
Ì «
Ì ²
Ì ¸
Ì ¿
Ì Å
Ì Ì
Ì Ò
Ì Ò
Ì Û
Ì Û
Ì Ã
Ì È
Ì Ô
Ì ¸	
Ì ½	
Ì É	
Ì ­

Ì ²

Ì ¾

Ì ¢
Ì §
Ì ³
Ì ¹
Ì ¾
Ì Ê
Ì Ð
Ì Õ
Ì á
Ì ç
Ì ì
Ì ø
Ì þ
Ì ƒ
Ì 
Ì •
Ì •
Ì š
Ì š
Ì ¨
Ì ¨
Í ó
Í Œ
Í £
Í º
Í Ñ
Í è
Í ÿ
Í ˜	
Í ¯	
Í Æ	
Í Ý	
Í ô	
Í ‹

Í ¤

Í »

Í Ò

Í é

Í €
Í —
Í °
Í Ç
Í Þ
Í õ
Í Œ
Í £	Î 9"

x_solve3"
_Z13get_global_idj"
llvm.fmuladd.f64*‹
npb-BT-x_solve3.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282€

transfer_bytes
à´Í

wgsize
,

devmap_label
 
 
transfer_bytes_log1p
Ï­„A

wgsize_log1p
Ï­„A