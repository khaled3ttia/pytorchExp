

[external]
LcallBD
B
	full_text5
3
1%10 = tail call i64 @_Z13get_global_idj(i32 2) #3
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
1%13 = tail call i64 @_Z13get_global_idj(i32 1) #3
.addB'
%
	full_text

%14 = add i64 %13, 1
#i64B

	full_text
	
i64 %13
6truncB-
+
	full_text

%15 = trunc i64 %14 to i32
#i64B

	full_text
	
i64 %14
LcallBD
B
	full_text5
3
1%16 = tail call i64 @_Z13get_global_idj(i32 0) #3
2addB+
)
	full_text

%17 = add nsw i32 %8, -2
6icmpB.
,
	full_text

%18 = icmp slt i32 %17, %12
#i32B

	full_text
	
i32 %17
#i32B

	full_text
	
i32 %12
9brB3
1
	full_text$
"
 br i1 %18, label %171, label %19
!i1B

	full_text


i1 %18
8trunc8B-
+
	full_text

%20 = trunc i64 %16 to i32
%i648B

	full_text
	
i64 %16
4add8B+
)
	full_text

%21 = add nsw i32 %6, -2
8icmp8B.
,
	full_text

%22 = icmp sge i32 %21, %15
%i328B

	full_text
	
i32 %21
%i328B

	full_text
	
i32 %15
7icmp8B-
+
	full_text

%23 = icmp slt i32 %20, %7
%i328B

	full_text
	
i32 %20
1and8B(
&
	full_text

%24 = and i1 %22, %23
#i18B

	full_text


i1 %22
#i18B

	full_text


i1 %23
;br8B3
1
	full_text$
"
 br i1 %24, label %25, label %171
#i18B

	full_text


i1 %24
Sbitcast8BF
D
	full_text7
5
3%26 = bitcast double* %0 to [103 x [103 x double]]*
Sbitcast8BF
D
	full_text7
5
3%27 = bitcast double* %1 to [103 x [103 x double]]*
Sbitcast8BF
D
	full_text7
5
3%28 = bitcast double* %2 to [103 x [103 x double]]*
Ybitcast8BL
J
	full_text=
;
9%29 = bitcast double* %3 to [103 x [103 x [5 x double]]]*
5add8B,
*
	full_text

%30 = add nsw i32 %12, -1
%i328B

	full_text
	
i32 %12
6mul8B-
+
	full_text

%31 = mul nsw i32 %30, %21
%i328B

	full_text
	
i32 %30
%i328B

	full_text
	
i32 %21
5add8B,
*
	full_text

%32 = add nsw i32 %15, -1
%i328B

	full_text
	
i32 %15
6add8B-
+
	full_text

%33 = add nsw i32 %32, %31
%i328B

	full_text
	
i32 %32
%i328B

	full_text
	
i32 %31
3mul8B*
(
	full_text

%34 = mul i32 %33, 2575
%i328B

	full_text
	
i32 %33
6sext8B,
*
	full_text

%35 = sext i32 %34 to i64
%i328B

	full_text
	
i32 %34
^getelementptr8BK
I
	full_text<
:
8%36 = getelementptr inbounds double, double* %4, i64 %35
%i648B

	full_text
	
i64 %35
Pbitcast8BC
A
	full_text4
2
0%37 = bitcast double* %36 to [5 x [5 x double]]*
-double*8B

	full_text

double* %36
^getelementptr8BK
I
	full_text<
:
8%38 = getelementptr inbounds double, double* %5, i64 %35
%i648B

	full_text
	
i64 %35
Pbitcast8BC
A
	full_text4
2
0%39 = bitcast double* %38 to [5 x [5 x double]]*
-double*8B

	full_text

double* %38
1shl8B(
&
	full_text

%40 = shl i64 %11, 32
%i648B

	full_text
	
i64 %11
9ashr8B/
-
	full_text 

%41 = ashr exact i64 %40, 32
%i648B

	full_text
	
i64 %40
1shl8B(
&
	full_text

%42 = shl i64 %16, 32
%i648B

	full_text
	
i64 %16
9ashr8B/
-
	full_text 

%43 = ashr exact i64 %42, 32
%i648B

	full_text
	
i64 %42
1shl8B(
&
	full_text

%44 = shl i64 %14, 32
%i648B

	full_text
	
i64 %14
9ashr8B/
-
	full_text 

%45 = ashr exact i64 %44, 32
%i648B

	full_text
	
i64 %44
�getelementptr8B~
|
	full_texto
m
k%46 = getelementptr inbounds [103 x [103 x double]], [103 x [103 x double]]* %27, i64 %41, i64 %43, i64 %45
M[103 x [103 x double]]*8B.
,
	full_text

[103 x [103 x double]]* %27
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Nload8BD
B
	full_text5
3
1%47 = load double, double* %46, align 8, !tbaa !8
-double*8B

	full_text

double* %46
7fmul8B-
+
	full_text

%48 = fmul double %47, %47
+double8B

	full_text


double %47
+double8B

	full_text


double %47
7fmul8B-
+
	full_text

%49 = fmul double %47, %48
+double8B

	full_text


double %47
+double8B

	full_text


double %48
�getelementptr8B�
�
	full_text�
�
~%50 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %29, i64 %41, i64 %43, i64 %45, i64 1
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %29
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Nload8BD
B
	full_text5
3
1%51 = load double, double* %50, align 8, !tbaa !8
-double*8B

	full_text

double* %50
�getelementptr8B�
�
	full_text�
�
~%52 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %29, i64 %41, i64 %43, i64 %45, i64 2
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %29
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Nload8BD
B
	full_text5
3
1%53 = load double, double* %52, align 8, !tbaa !8
-double*8B

	full_text

double* %52
�getelementptr8B�
�
	full_text�
�
~%54 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %29, i64 %41, i64 %43, i64 %45, i64 3
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %29
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Nload8BD
B
	full_text5
3
1%55 = load double, double* %54, align 8, !tbaa !8
-double*8B

	full_text

double* %54
�getelementptr8B�
�
	full_text�
�
~%56 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %29, i64 %41, i64 %43, i64 %45, i64 4
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %29
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Nload8BD
B
	full_text5
3
1%57 = load double, double* %56, align 8, !tbaa !8
-double*8B

	full_text

double* %56
�getelementptr8Br
p
	full_textc
a
_%58 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %43, i64 0, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %43
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %58, align 8, !tbaa !8
-double*8B

	full_text

double* %58
�getelementptr8Br
p
	full_textc
a
_%59 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %43, i64 1, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %43
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %59, align 8, !tbaa !8
-double*8B

	full_text

double* %59
�getelementptr8Br
p
	full_textc
a
_%60 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %43, i64 2, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %43
Wstore8BL
J
	full_text=
;
9store double 1.000000e+00, double* %60, align 8, !tbaa !8
-double*8B

	full_text

double* %60
�getelementptr8Br
p
	full_textc
a
_%61 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %43, i64 3, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %43
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %61, align 8, !tbaa !8
-double*8B

	full_text

double* %61
�getelementptr8Br
p
	full_textc
a
_%62 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %43, i64 4, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %43
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %62, align 8, !tbaa !8
-double*8B

	full_text

double* %62
7fmul8B-
+
	full_text

%63 = fmul double %51, %53
+double8B

	full_text


double %51
+double8B

	full_text


double %53
7fmul8B-
+
	full_text

%64 = fmul double %48, %63
+double8B

	full_text


double %48
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
�getelementptr8Br
p
	full_textc
a
_%66 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %43, i64 0, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %43
Nstore8BC
A
	full_text4
2
0store double %65, double* %66, align 8, !tbaa !8
+double8B

	full_text


double %65
-double*8B

	full_text

double* %66
7fmul8B-
+
	full_text

%67 = fmul double %47, %53
+double8B

	full_text


double %47
+double8B

	full_text


double %53
�getelementptr8Br
p
	full_textc
a
_%68 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %43, i64 1, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %43
Nstore8BC
A
	full_text4
2
0store double %67, double* %68, align 8, !tbaa !8
+double8B

	full_text


double %67
-double*8B

	full_text

double* %68
7fmul8B-
+
	full_text

%69 = fmul double %47, %51
+double8B

	full_text


double %47
+double8B

	full_text


double %51
�getelementptr8Br
p
	full_textc
a
_%70 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %43, i64 2, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %43
Nstore8BC
A
	full_text4
2
0store double %69, double* %70, align 8, !tbaa !8
+double8B

	full_text


double %69
-double*8B

	full_text

double* %70
�getelementptr8Br
p
	full_textc
a
_%71 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %43, i64 3, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %43
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %71, align 8, !tbaa !8
-double*8B

	full_text

double* %71
�getelementptr8Br
p
	full_textc
a
_%72 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %43, i64 4, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %43
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %72, align 8, !tbaa !8
-double*8B

	full_text

double* %72
7fmul8B-
+
	full_text

%73 = fmul double %53, %53
+double8B

	full_text


double %53
+double8B

	full_text


double %53
7fmul8B-
+
	full_text

%74 = fmul double %48, %73
+double8B

	full_text


double %48
+double8B

	full_text


double %73
Afsub8B7
5
	full_text(
&
$%75 = fsub double -0.000000e+00, %74
+double8B

	full_text


double %74
�getelementptr8B~
|
	full_texto
m
k%76 = getelementptr inbounds [103 x [103 x double]], [103 x [103 x double]]* %26, i64 %41, i64 %43, i64 %45
M[103 x [103 x double]]*8B.
,
	full_text

[103 x [103 x double]]* %26
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Nload8BD
B
	full_text5
3
1%77 = load double, double* %76, align 8, !tbaa !8
-double*8B

	full_text

double* %76
rcall8Bh
f
	full_textY
W
U%78 = tail call double @llvm.fmuladd.f64(double %77, double 4.000000e-01, double %75)
+double8B

	full_text


double %77
+double8B

	full_text


double %75
�getelementptr8Br
p
	full_textc
a
_%79 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %43, i64 0, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %43
Nstore8BC
A
	full_text4
2
0store double %78, double* %79, align 8, !tbaa !8
+double8B

	full_text


double %78
-double*8B

	full_text

double* %79
Afmul8B7
5
	full_text(
&
$%80 = fmul double %51, -4.000000e-01
+double8B

	full_text


double %51
7fmul8B-
+
	full_text

%81 = fmul double %47, %80
+double8B

	full_text


double %47
+double8B

	full_text


double %80
�getelementptr8Br
p
	full_textc
a
_%82 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %43, i64 1, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %43
Nstore8BC
A
	full_text4
2
0store double %81, double* %82, align 8, !tbaa !8
+double8B

	full_text


double %81
-double*8B

	full_text

double* %82
@fmul8B6
4
	full_text'
%
#%83 = fmul double %53, 1.600000e+00
+double8B

	full_text


double %53
7fmul8B-
+
	full_text

%84 = fmul double %47, %83
+double8B

	full_text


double %47
+double8B

	full_text


double %83
�getelementptr8Br
p
	full_textc
a
_%85 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %43, i64 2, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %43
Nstore8BC
A
	full_text4
2
0store double %84, double* %85, align 8, !tbaa !8
+double8B

	full_text


double %84
-double*8B

	full_text

double* %85
Afmul8B7
5
	full_text(
&
$%86 = fmul double %55, -4.000000e-01
+double8B

	full_text


double %55
7fmul8B-
+
	full_text

%87 = fmul double %47, %86
+double8B

	full_text


double %47
+double8B

	full_text


double %86
�getelementptr8Br
p
	full_textc
a
_%88 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %43, i64 3, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %43
Nstore8BC
A
	full_text4
2
0store double %87, double* %88, align 8, !tbaa !8
+double8B

	full_text


double %87
-double*8B

	full_text

double* %88
�getelementptr8Br
p
	full_textc
a
_%89 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %43, i64 4, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %43
Wstore8BL
J
	full_text=
;
9store double 4.000000e-01, double* %89, align 8, !tbaa !8
-double*8B

	full_text

double* %89
7fmul8B-
+
	full_text

%90 = fmul double %53, %55
+double8B

	full_text


double %53
+double8B

	full_text


double %55
7fmul8B-
+
	full_text

%91 = fmul double %48, %90
+double8B

	full_text


double %48
+double8B

	full_text


double %90
Afsub8B7
5
	full_text(
&
$%92 = fsub double -0.000000e+00, %91
+double8B

	full_text


double %91
�getelementptr8Br
p
	full_textc
a
_%93 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %43, i64 0, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %43
Nstore8BC
A
	full_text4
2
0store double %92, double* %93, align 8, !tbaa !8
+double8B

	full_text


double %92
-double*8B

	full_text

double* %93
�getelementptr8Br
p
	full_textc
a
_%94 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %43, i64 1, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %43
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %94, align 8, !tbaa !8
-double*8B

	full_text

double* %94
7fmul8B-
+
	full_text

%95 = fmul double %47, %55
+double8B

	full_text


double %47
+double8B

	full_text


double %55
�getelementptr8Br
p
	full_textc
a
_%96 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %43, i64 2, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %43
Nstore8BC
A
	full_text4
2
0store double %95, double* %96, align 8, !tbaa !8
+double8B

	full_text


double %95
-double*8B

	full_text

double* %96
�getelementptr8Br
p
	full_textc
a
_%97 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %43, i64 3, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %43
Nstore8BC
A
	full_text4
2
0store double %67, double* %97, align 8, !tbaa !8
+double8B

	full_text


double %67
-double*8B

	full_text

double* %97
�getelementptr8Br
p
	full_textc
a
_%98 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %43, i64 4, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %43
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %98, align 8, !tbaa !8
-double*8B

	full_text

double* %98
�getelementptr8B~
|
	full_texto
m
k%99 = getelementptr inbounds [103 x [103 x double]], [103 x [103 x double]]* %28, i64 %41, i64 %43, i64 %45
M[103 x [103 x double]]*8B.
,
	full_text

[103 x [103 x double]]* %28
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Oload8BE
C
	full_text6
4
2%100 = load double, double* %99, align 8, !tbaa !8
-double*8B

	full_text

double* %99
Afmul8B7
5
	full_text(
&
$%101 = fmul double %57, 1.400000e+00
+double8B

	full_text


double %57
Cfsub8B9
7
	full_text*
(
&%102 = fsub double -0.000000e+00, %101
,double8B

	full_text

double %101
ucall8Bk
i
	full_text\
Z
X%103 = tail call double @llvm.fmuladd.f64(double %100, double 8.000000e-01, double %102)
,double8B

	full_text

double %100
,double8B

	full_text

double %102
9fmul8B/
-
	full_text 

%104 = fmul double %53, %103
+double8B

	full_text


double %53
,double8B

	full_text

double %103
9fmul8B/
-
	full_text 

%105 = fmul double %48, %104
+double8B

	full_text


double %48
,double8B

	full_text

double %104
�getelementptr8Bs
q
	full_textd
b
`%106 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %43, i64 0, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %43
Pstore8BE
C
	full_text6
4
2store double %105, double* %106, align 8, !tbaa !8
,double8B

	full_text

double %105
.double*8B

	full_text

double* %106
8fmul8B.
,
	full_text

%107 = fmul double %80, %53
+double8B

	full_text


double %80
+double8B

	full_text


double %53
9fmul8B/
-
	full_text 

%108 = fmul double %48, %107
+double8B

	full_text


double %48
,double8B

	full_text

double %107
�getelementptr8Bs
q
	full_textd
b
`%109 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %43, i64 1, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %43
Pstore8BE
C
	full_text6
4
2store double %108, double* %109, align 8, !tbaa !8
,double8B

	full_text

double %108
.double*8B

	full_text

double* %109
Oload8BE
C
	full_text6
4
2%110 = load double, double* %76, align 8, !tbaa !8
-double*8B

	full_text

double* %76
kcall8Ba
_
	full_textR
P
N%111 = tail call double @llvm.fmuladd.f64(double %73, double %48, double %110)
+double8B

	full_text


double %73
+double8B

	full_text


double %48
,double8B

	full_text

double %110
Bfmul8B8
6
	full_text)
'
%%112 = fmul double %111, 4.000000e-01
,double8B

	full_text

double %111
Cfsub8B9
7
	full_text*
(
&%113 = fsub double -0.000000e+00, %112
,double8B

	full_text

double %112
lcall8Bb
`
	full_textS
Q
O%114 = tail call double @llvm.fmuladd.f64(double %101, double %47, double %113)
,double8B

	full_text

double %101
+double8B

	full_text


double %47
,double8B

	full_text

double %113
�getelementptr8Bs
q
	full_textd
b
`%115 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %43, i64 2, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %43
Pstore8BE
C
	full_text6
4
2store double %114, double* %115, align 8, !tbaa !8
,double8B

	full_text

double %114
.double*8B

	full_text

double* %115
Bfmul8B8
6
	full_text)
'
%%116 = fmul double %90, -4.000000e-01
+double8B

	full_text


double %90
9fmul8B/
-
	full_text 

%117 = fmul double %48, %116
+double8B

	full_text


double %48
,double8B

	full_text

double %116
�getelementptr8Bs
q
	full_textd
b
`%118 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %43, i64 3, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %43
Pstore8BE
C
	full_text6
4
2store double %117, double* %118, align 8, !tbaa !8
,double8B

	full_text

double %117
.double*8B

	full_text

double* %118
Afmul8B7
5
	full_text(
&
$%119 = fmul double %53, 1.400000e+00
+double8B

	full_text


double %53
9fmul8B/
-
	full_text 

%120 = fmul double %47, %119
+double8B

	full_text


double %47
,double8B

	full_text

double %119
�getelementptr8Bs
q
	full_textd
b
`%121 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %43, i64 4, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %43
Pstore8BE
C
	full_text6
4
2store double %120, double* %121, align 8, !tbaa !8
,double8B

	full_text

double %120
.double*8B

	full_text

double* %121
�getelementptr8Bs
q
	full_textd
b
`%122 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %39, i64 %43, i64 0, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %39
%i648B

	full_text
	
i64 %43
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %122, align 8, !tbaa !8
.double*8B

	full_text

double* %122
�getelementptr8Bs
q
	full_textd
b
`%123 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %39, i64 %43, i64 1, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %39
%i648B

	full_text
	
i64 %43
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %123, align 8, !tbaa !8
.double*8B

	full_text

double* %123
�getelementptr8Bs
q
	full_textd
b
`%124 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %39, i64 %43, i64 2, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %39
%i648B

	full_text
	
i64 %43
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %124, align 8, !tbaa !8
.double*8B

	full_text

double* %124
�getelementptr8Bs
q
	full_textd
b
`%125 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %39, i64 %43, i64 3, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %39
%i648B

	full_text
	
i64 %43
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %125, align 8, !tbaa !8
.double*8B

	full_text

double* %125
�getelementptr8Bs
q
	full_textd
b
`%126 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %39, i64 %43, i64 4, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %39
%i648B

	full_text
	
i64 %43
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %126, align 8, !tbaa !8
.double*8B

	full_text

double* %126
Bfmul8B8
6
	full_text)
'
%%127 = fmul double %48, -1.000000e-01
+double8B

	full_text


double %48
9fmul8B/
-
	full_text 

%128 = fmul double %51, %127
+double8B

	full_text


double %51
,double8B

	full_text

double %127
�getelementptr8Bs
q
	full_textd
b
`%129 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %39, i64 %43, i64 0, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %39
%i648B

	full_text
	
i64 %43
Pstore8BE
C
	full_text6
4
2store double %128, double* %129, align 8, !tbaa !8
,double8B

	full_text

double %128
.double*8B

	full_text

double* %129
Afmul8B7
5
	full_text(
&
$%130 = fmul double %47, 1.000000e-01
+double8B

	full_text


double %47
�getelementptr8Bs
q
	full_textd
b
`%131 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %39, i64 %43, i64 1, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %39
%i648B

	full_text
	
i64 %43
Pstore8BE
C
	full_text6
4
2store double %130, double* %131, align 8, !tbaa !8
,double8B

	full_text

double %130
.double*8B

	full_text

double* %131
�getelementptr8Bs
q
	full_textd
b
`%132 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %39, i64 %43, i64 2, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %39
%i648B

	full_text
	
i64 %43
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %132, align 8, !tbaa !8
.double*8B

	full_text

double* %132
�getelementptr8Bs
q
	full_textd
b
`%133 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %39, i64 %43, i64 3, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %39
%i648B

	full_text
	
i64 %43
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %133, align 8, !tbaa !8
.double*8B

	full_text

double* %133
�getelementptr8Bs
q
	full_textd
b
`%134 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %39, i64 %43, i64 4, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %39
%i648B

	full_text
	
i64 %43
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %134, align 8, !tbaa !8
.double*8B

	full_text

double* %134
Gfmul8B=
;
	full_text.
,
*%135 = fmul double %48, 0xBFC1111111111111
+double8B

	full_text


double %48
9fmul8B/
-
	full_text 

%136 = fmul double %135, %53
,double8B

	full_text

double %135
+double8B

	full_text


double %53
�getelementptr8Bs
q
	full_textd
b
`%137 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %39, i64 %43, i64 0, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %39
%i648B

	full_text
	
i64 %43
Pstore8BE
C
	full_text6
4
2store double %136, double* %137, align 8, !tbaa !8
,double8B

	full_text

double %136
.double*8B

	full_text

double* %137
�getelementptr8Bs
q
	full_textd
b
`%138 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %39, i64 %43, i64 1, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %39
%i648B

	full_text
	
i64 %43
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %138, align 8, !tbaa !8
.double*8B

	full_text

double* %138
Gfmul8B=
;
	full_text.
,
*%139 = fmul double %47, 0x3FC1111111111111
+double8B

	full_text


double %47
�getelementptr8Bs
q
	full_textd
b
`%140 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %39, i64 %43, i64 2, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %39
%i648B

	full_text
	
i64 %43
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
�getelementptr8Bs
q
	full_textd
b
`%141 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %39, i64 %43, i64 3, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %39
%i648B

	full_text
	
i64 %43
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %141, align 8, !tbaa !8
.double*8B

	full_text

double* %141
�getelementptr8Bs
q
	full_textd
b
`%142 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %39, i64 %43, i64 4, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %39
%i648B

	full_text
	
i64 %43
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %142, align 8, !tbaa !8
.double*8B

	full_text

double* %142
9fmul8B/
-
	full_text 

%143 = fmul double %127, %55
,double8B

	full_text

double %127
+double8B

	full_text


double %55
�getelementptr8Bs
q
	full_textd
b
`%144 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %39, i64 %43, i64 0, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %39
%i648B

	full_text
	
i64 %43
Pstore8BE
C
	full_text6
4
2store double %143, double* %144, align 8, !tbaa !8
,double8B

	full_text

double %143
.double*8B

	full_text

double* %144
�getelementptr8Bs
q
	full_textd
b
`%145 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %39, i64 %43, i64 1, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %39
%i648B

	full_text
	
i64 %43
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %145, align 8, !tbaa !8
.double*8B

	full_text

double* %145
�getelementptr8Bs
q
	full_textd
b
`%146 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %39, i64 %43, i64 2, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %39
%i648B

	full_text
	
i64 %43
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %146, align 8, !tbaa !8
.double*8B

	full_text

double* %146
�getelementptr8Bs
q
	full_textd
b
`%147 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %39, i64 %43, i64 3, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %39
%i648B

	full_text
	
i64 %43
Pstore8BE
C
	full_text6
4
2store double %130, double* %147, align 8, !tbaa !8
,double8B

	full_text

double %130
.double*8B

	full_text

double* %147
�getelementptr8Bs
q
	full_textd
b
`%148 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %39, i64 %43, i64 4, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %39
%i648B

	full_text
	
i64 %43
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %148, align 8, !tbaa !8
.double*8B

	full_text

double* %148
Gfmul8B=
;
	full_text.
,
*%149 = fmul double %49, 0x3FB89374BC6A7EF8
+double8B

	full_text


double %49
8fmul8B.
,
	full_text

%150 = fmul double %51, %51
+double8B

	full_text


double %51
+double8B

	full_text


double %51
Gfmul8B=
;
	full_text.
,
*%151 = fmul double %49, 0xBFB00AEC33E1F670
+double8B

	full_text


double %49
9fmul8B/
-
	full_text 

%152 = fmul double %151, %73
,double8B

	full_text

double %151
+double8B

	full_text


double %73
Cfsub8B9
7
	full_text*
(
&%153 = fsub double -0.000000e+00, %152
,double8B

	full_text

double %152
mcall8Bc
a
	full_textT
R
P%154 = tail call double @llvm.fmuladd.f64(double %149, double %150, double %153)
,double8B

	full_text

double %149
,double8B

	full_text

double %150
,double8B

	full_text

double %153
8fmul8B.
,
	full_text

%155 = fmul double %55, %55
+double8B

	full_text


double %55
+double8B

	full_text


double %55
mcall8Bc
a
	full_textT
R
P%156 = tail call double @llvm.fmuladd.f64(double %149, double %155, double %154)
,double8B

	full_text

double %149
,double8B

	full_text

double %155
,double8B

	full_text

double %154
Gfmul8B=
;
	full_text.
,
*%157 = fmul double %48, 0x3FC916872B020C49
+double8B

	full_text


double %48
Cfsub8B9
7
	full_text*
(
&%158 = fsub double -0.000000e+00, %157
,double8B

	full_text

double %157
lcall8Bb
`
	full_textS
Q
O%159 = tail call double @llvm.fmuladd.f64(double %158, double %57, double %156)
,double8B

	full_text

double %158
+double8B

	full_text


double %57
,double8B

	full_text

double %156
�getelementptr8Bs
q
	full_textd
b
`%160 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %39, i64 %43, i64 0, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %39
%i648B

	full_text
	
i64 %43
Pstore8BE
C
	full_text6
4
2store double %159, double* %160, align 8, !tbaa !8
,double8B

	full_text

double %159
.double*8B

	full_text

double* %160
Gfmul8B=
;
	full_text.
,
*%161 = fmul double %48, 0xBFB89374BC6A7EF8
+double8B

	full_text


double %48
9fmul8B/
-
	full_text 

%162 = fmul double %51, %161
+double8B

	full_text


double %51
,double8B

	full_text

double %161
�getelementptr8Bs
q
	full_textd
b
`%163 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %39, i64 %43, i64 1, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %39
%i648B

	full_text
	
i64 %43
Pstore8BE
C
	full_text6
4
2store double %162, double* %163, align 8, !tbaa !8
,double8B

	full_text

double %162
.double*8B

	full_text

double* %163
Gfmul8B=
;
	full_text.
,
*%164 = fmul double %48, 0xBFB00AEC33E1F670
+double8B

	full_text


double %48
9fmul8B/
-
	full_text 

%165 = fmul double %164, %53
,double8B

	full_text

double %164
+double8B

	full_text


double %53
�getelementptr8Bs
q
	full_textd
b
`%166 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %39, i64 %43, i64 2, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %39
%i648B

	full_text
	
i64 %43
Pstore8BE
C
	full_text6
4
2store double %165, double* %166, align 8, !tbaa !8
,double8B

	full_text

double %165
.double*8B

	full_text

double* %166
9fmul8B/
-
	full_text 

%167 = fmul double %161, %55
,double8B

	full_text

double %161
+double8B

	full_text


double %55
�getelementptr8Bs
q
	full_textd
b
`%168 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %39, i64 %43, i64 3, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %39
%i648B

	full_text
	
i64 %43
Pstore8BE
C
	full_text6
4
2store double %167, double* %168, align 8, !tbaa !8
,double8B

	full_text

double %167
.double*8B

	full_text

double* %168
Gfmul8B=
;
	full_text.
,
*%169 = fmul double %47, 0x3FC916872B020C49
+double8B

	full_text


double %47
�getelementptr8Bs
q
	full_textd
b
`%170 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %39, i64 %43, i64 4, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %39
%i648B

	full_text
	
i64 %43
Pstore8BE
C
	full_text6
4
2store double %169, double* %170, align 8, !tbaa !8
,double8B

	full_text

double %169
.double*8B

	full_text

double* %170
(br8B 

	full_text

br label %171
$ret8B

	full_text


ret void
$i328B

	full_text


i32 %6
,double*8B

	full_text


double* %3
,double*8B

	full_text


double* %0
$i328B

	full_text


i32 %8
,double*8B

	full_text


double* %1
,double*8B

	full_text


double* %2
,double*8B

	full_text


double* %4
,double*8B

	full_text


double* %5
$i328B
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
5double8B'
%
	full_text

double -1.000000e-01
#i648B

	full_text	

i64 3
#i648B

	full_text	

i64 2
#i648B

	full_text	

i64 4
4double8B&
$
	full_text

double 1.000000e+00
#i648B

	full_text	

i64 0
4double8B&
$
	full_text

double 1.400000e+00
4double8B&
$
	full_text

double 1.000000e-01
:double8B,
*
	full_text

double 0xBFC1111111111111
:double8B,
*
	full_text

double 0x3FC1111111111111
$i648B

	full_text


i64 32
:double8B,
*
	full_text

double 0x3FB89374BC6A7EF8
:double8B,
*
	full_text

double 0xBFB89374BC6A7EF8
#i328B

	full_text	

i32 1
:double8B,
*
	full_text

double 0xBFB00AEC33E1F670
4double8B&
$
	full_text

double 1.600000e+00
#i328B

	full_text	

i32 2
4double8B&
$
	full_text

double 0.000000e+00
&i328B

	full_text


i32 2575
#i328B

	full_text	

i32 0
:double8B,
*
	full_text

double 0x3FC916872B020C49
4double8B&
$
	full_text

double 8.000000e-01
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
4double8B&
$
	full_text

double 4.000000e-01
5double8B'
%
	full_text

double -4.000000e-01
$i328B

	full_text


i32 -2
5double8B'
%
	full_text

double -0.000000e+00        	
 		                       !! "" #$ ## %& %' %% () (( *+ *, ** -. -- /0 // 12 11 34 33 56 55 78 77 9: 99 ;< ;; => == ?@ ?? AB AA CD CC EF EG EH EI EE JK JJ LM LN LL OP OQ OO RS RT RU RV RR WX WW YZ Y[ Y\ Y] YY ^_ ^^ `a `b `c `d `` ef ee gh gi gj gk gg lm ll no np nn qr qq st su ss vw vv xy xz xx {| {{ }~ } }} �
� �� �� �
� �� �
� �� �� �
� �� �� �
� �� �
� �� �� �
� �� �� �
� �� �� �
� �� �� �
� �� �� �
� �� �� �
� �� �� �
� �� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �� �
� �� �
� �� �� �
� �
� �
� �� �� �� �� �
� �� �� �
� �� �� �
� �� �� �� �� �
� �� �� �
� �� �� �
� �� �� �� �� �
� �� �� �
� �� �� �
� �� �� �� �� �
� �� �� �
� �� �� �
� �� �� �
� �� �
� �� �� �
� �� �� �
� �� �
� �� �� �
� �� �� �
� �� �� �
� �� �
� �� �� �
� �� �� �
� �� �� �
� �� �� �
� �� �� �
� �� �� �
� �� �
� �� �� �
� �
� �
� �� �� �� �� �� �
� �� �� �
� �� �� �
� �� �� �
� �� �� �
� �� �� �
� �� �� �
� �� �� �
� �� �� �
� �� �� �
� �� �� �� �� �
� �
� �� �� �� �
� �� �� �
� �
� �� �� �
� �� �� �
� �� �� �� �� �
� �� �� �
� �� �� �
� �� �� �� �� �
� �� �� �
� �� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �� �� �
� �� �� �
� �� �� �
� �� �� �� �� �
� �� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �� �� �
� �� �� �
� �� �� �
� �� �� �
� �� �
� �� �� �� �� �
� �� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �� �
� �� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �� �
� �� �� �
� �� �
� �� �� �� �� �
� �� �� �� �� �
� �� �
� �� �� �
� �
� �� �� �
� �� �� �
� �
� �� �� �� �
� �� �� �
� �
� �� �� �
� �� �� �
� �� �� �� �� �
� �� �� �
� �� �� �
� �� �� �� �� �
� �� �� �
� �� �� �
� �� �� �
� �� �� �
� �� �� �
� �� �� �� �� �
� �� �� �
� �� �� � "� � �  � !� 1� 5	�     
     	      $# & '	 )( +% ,* .- 0/ 21 4/ 65 8 :9 < >= @ BA D  F; G? HC IE KJ MJ NJ PL Q" S; T? UC VR X" Z; [? \C ]Y _" a; b? cC d` f" h; i? jC kg m3 o? pn r3 t? us w3 y? zx |3 ~? } �3 �? �� �W �^ �L �� �� �3 �? �� �� �J �^ �3 �? �� �� �J �W �3 �? �� �� �3 �? �� �3 �? �� �^ �^ �L �� �� � �; �? �C �� �� �� �3 �? �� �� �W �J �� �3 �? �� �� �^ �J �� �3 �? �� �� �e �J �� �3 �? �� �� �3 �? �� �^ �e �L �� �� �3 �? �� �� �3 �? �� �J �e �3 �? �� �� �3 �? �� �� �3 �? �� �! �; �? �C �� �l �� �� �� �^ �� �L �� �3 �? �� �� �� �^ �L �� �3 �? �� �� �� �� �L �� �� �� �� �J �� �3 �? �� �� �� �L �� �3 �? �� �� �^ �J �� �3 �? �� �� �7 �? �� �7 �? �� �7 �? �� �7 �? �� �7 �? �� �L �W �� �7 �? �� �� �J �7 �? �� �� �7 �? �� �7 �? �� �7 �? �� �L �� �^ �7 �? �� �� �7 �? �� �J �7 �? �� �� �7 �? �� �7 �? �� �� �e �7 �? �� �� �7 �? �� �7 �? �� �7 �? �� �� �7 �? �� �O �W �W �O �� �� �� �� �� �� �e �e �� �� �� �L �� �� �l �� �7 �? �� �� �L �W �� �7 �? �� �� �L �� �^ �7 �? �� �� �� �e �7 �? �� �� �J �7 �? �� �� � �   �� � � �� ��� �� � ��  �� � �� � �� � �� �� �� �� �� �� �� �� �� �
� �	� `	� }
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �	� Y	� x
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �	� g
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �� {	� n	� n	� s	� x	� }
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �	� 9	� ;	� =	� ?	� A	� C
� �
� �� 
� �
� �
� �� � q� v� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �	� -� 
� �
� �
� �	� 	� 	� R	� s
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �	� #	� (
� �� �
� �
� �
� �
� �	� 	� � �� �� �� �� �� �� �"

y_solve1"
_Z13get_global_idj"
llvm.fmuladd.f64*�
npb-BT-y_solve1.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282�

wgsize
 

wgsize_log1p
 ��A

transfer_bytes	
����
 
transfer_bytes_log1p
 ��A

devmap_label
