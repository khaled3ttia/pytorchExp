

[external]
KcallBC
A
	full_text4
2
0%9 = tail call i64 @_Z13get_global_idj(i32 2) #3
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
6truncB-
+
	full_text

%11 = trunc i64 %10 to i32
#i64B

	full_text
	
i64 %10
LcallBD
B
	full_text5
3
1%12 = tail call i64 @_Z13get_global_idj(i32 1) #3
.addB'
%
	full_text

%13 = add i64 %12, 1
#i64B

	full_text
	
i64 %12
6truncB-
+
	full_text

%14 = trunc i64 %13 to i32
#i64B

	full_text
	
i64 %13
LcallBD
B
	full_text5
3
1%15 = tail call i64 @_Z13get_global_idj(i32 0) #3
2addB+
)
	full_text

%16 = add nsw i32 %6, -2
6icmpB.
,
	full_text

%17 = icmp slt i32 %16, %11
#i32B

	full_text
	
i32 %16
#i32B

	full_text
	
i32 %11
9brB3
1
	full_text$
"
 br i1 %17, label %172, label %18
!i1B

	full_text


i1 %17
8trunc8B-
+
	full_text

%19 = trunc i64 %15 to i32
%i648B

	full_text
	
i64 %15
4add8B+
)
	full_text

%20 = add nsw i32 %5, -2
8icmp8B.
,
	full_text

%21 = icmp sge i32 %20, %14
%i328B

	full_text
	
i32 %20
%i328B

	full_text
	
i32 %14
7icmp8B-
+
	full_text

%22 = icmp slt i32 %19, %7
%i328B

	full_text
	
i32 %19
1and8B(
&
	full_text

%23 = and i1 %21, %22
#i18B

	full_text


i1 %21
#i18B

	full_text


i1 %22
;br8B3
1
	full_text$
"
 br i1 %23, label %24, label %172
#i18B

	full_text


i1 %23
Qbitcast8BD
B
	full_text5
3
1%25 = bitcast double* %0 to [13 x [13 x double]]*
Qbitcast8BD
B
	full_text5
3
1%26 = bitcast double* %1 to [13 x [13 x double]]*
Wbitcast8BJ
H
	full_text;
9
7%27 = bitcast double* %2 to [13 x [13 x [5 x double]]]*
5add8B,
*
	full_text

%28 = add nsw i32 %11, -1
%i328B

	full_text
	
i32 %11
6mul8B-
+
	full_text

%29 = mul nsw i32 %28, %20
%i328B

	full_text
	
i32 %28
%i328B

	full_text
	
i32 %20
5add8B,
*
	full_text

%30 = add nsw i32 %14, -1
%i328B

	full_text
	
i32 %14
6add8B-
+
	full_text

%31 = add nsw i32 %30, %29
%i328B

	full_text
	
i32 %30
%i328B

	full_text
	
i32 %29
2mul8B)
'
	full_text

%32 = mul i32 %31, 325
%i328B

	full_text
	
i32 %31
6sext8B,
*
	full_text

%33 = sext i32 %32 to i64
%i328B

	full_text
	
i32 %32
^getelementptr8BK
I
	full_text<
:
8%34 = getelementptr inbounds double, double* %3, i64 %33
%i648B

	full_text
	
i64 %33
Pbitcast8BC
A
	full_text4
2
0%35 = bitcast double* %34 to [5 x [5 x double]]*
-double*8B

	full_text

double* %34
^getelementptr8BK
I
	full_text<
:
8%36 = getelementptr inbounds double, double* %4, i64 %33
%i648B

	full_text
	
i64 %33
Pbitcast8BC
A
	full_text4
2
0%37 = bitcast double* %36 to [5 x [5 x double]]*
-double*8B

	full_text

double* %36
1shl8B(
&
	full_text

%38 = shl i64 %15, 32
%i648B

	full_text
	
i64 %15
9ashr8B/
-
	full_text 

%39 = ashr exact i64 %38, 32
%i648B

	full_text
	
i64 %38
1shl8B(
&
	full_text

%40 = shl i64 %10, 32
%i648B

	full_text
	
i64 %10
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
%42 = shl i64 %13, 32
%i648B

	full_text
	
i64 %13
9ashr8B/
-
	full_text 

%43 = ashr exact i64 %42, 32
%i648B

	full_text
	
i64 %42
�getelementptr8B�
�
	full_text~
|
z%44 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %27, i64 %39, i64 %41, i64 %43, i64 0
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %27
%i648B

	full_text
	
i64 %39
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Nload8BD
B
	full_text5
3
1%45 = load double, double* %44, align 8, !tbaa !8
-double*8B

	full_text

double* %44
@fdiv8B6
4
	full_text'
%
#%46 = fdiv double 1.000000e+00, %45
+double8B

	full_text


double %45
7fmul8B-
+
	full_text

%47 = fmul double %46, %46
+double8B

	full_text


double %46
+double8B

	full_text


double %46
7fmul8B-
+
	full_text

%48 = fmul double %46, %47
+double8B

	full_text


double %46
+double8B

	full_text


double %47
�getelementptr8B�
�
	full_text~
|
z%49 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %27, i64 %39, i64 %41, i64 %43, i64 1
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %27
%i648B

	full_text
	
i64 %39
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Nload8BD
B
	full_text5
3
1%50 = load double, double* %49, align 8, !tbaa !8
-double*8B

	full_text

double* %49
�getelementptr8B�
�
	full_text~
|
z%51 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %27, i64 %39, i64 %41, i64 %43, i64 2
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %27
%i648B

	full_text
	
i64 %39
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Nload8BD
B
	full_text5
3
1%52 = load double, double* %51, align 8, !tbaa !8
-double*8B

	full_text

double* %51
�getelementptr8B�
�
	full_text~
|
z%53 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %27, i64 %39, i64 %41, i64 %43, i64 3
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %27
%i648B

	full_text
	
i64 %39
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Nload8BD
B
	full_text5
3
1%54 = load double, double* %53, align 8, !tbaa !8
-double*8B

	full_text

double* %53
�getelementptr8B�
�
	full_text~
|
z%55 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %27, i64 %39, i64 %41, i64 %43, i64 4
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %27
%i648B

	full_text
	
i64 %39
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Nload8BD
B
	full_text5
3
1%56 = load double, double* %55, align 8, !tbaa !8
-double*8B

	full_text

double* %55
�getelementptr8Br
p
	full_textc
a
_%57 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %35, i64 %39, i64 0, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %35
%i648B

	full_text
	
i64 %39
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %57, align 8, !tbaa !8
-double*8B

	full_text

double* %57
�getelementptr8Br
p
	full_textc
a
_%58 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %35, i64 %39, i64 1, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %35
%i648B

	full_text
	
i64 %39
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
_%59 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %35, i64 %39, i64 2, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %35
%i648B

	full_text
	
i64 %39
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
_%60 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %35, i64 %39, i64 3, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %35
%i648B

	full_text
	
i64 %39
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
_%61 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %35, i64 %39, i64 4, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %35
%i648B

	full_text
	
i64 %39
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %61, align 8, !tbaa !8
-double*8B

	full_text

double* %61
7fmul8B-
+
	full_text

%62 = fmul double %50, %54
+double8B

	full_text


double %50
+double8B

	full_text


double %54
7fmul8B-
+
	full_text

%63 = fmul double %47, %62
+double8B

	full_text


double %47
+double8B

	full_text


double %62
Afsub8B7
5
	full_text(
&
$%64 = fsub double -0.000000e+00, %63
+double8B

	full_text


double %63
�getelementptr8Br
p
	full_textc
a
_%65 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %35, i64 %39, i64 0, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %35
%i648B

	full_text
	
i64 %39
Nstore8BC
A
	full_text4
2
0store double %64, double* %65, align 8, !tbaa !8
+double8B

	full_text


double %64
-double*8B

	full_text

double* %65
7fmul8B-
+
	full_text

%66 = fmul double %46, %54
+double8B

	full_text


double %46
+double8B

	full_text


double %54
�getelementptr8Br
p
	full_textc
a
_%67 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %35, i64 %39, i64 1, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %35
%i648B

	full_text
	
i64 %39
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
�getelementptr8Br
p
	full_textc
a
_%68 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %35, i64 %39, i64 2, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %35
%i648B

	full_text
	
i64 %39
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %68, align 8, !tbaa !8
-double*8B

	full_text

double* %68
7fmul8B-
+
	full_text

%69 = fmul double %50, %46
+double8B

	full_text


double %50
+double8B

	full_text


double %46
�getelementptr8Br
p
	full_textc
a
_%70 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %35, i64 %39, i64 3, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %35
%i648B

	full_text
	
i64 %39
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
_%71 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %35, i64 %39, i64 4, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %35
%i648B

	full_text
	
i64 %39
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %71, align 8, !tbaa !8
-double*8B

	full_text

double* %71
7fmul8B-
+
	full_text

%72 = fmul double %52, %54
+double8B

	full_text


double %52
+double8B

	full_text


double %54
7fmul8B-
+
	full_text

%73 = fmul double %47, %72
+double8B

	full_text


double %47
+double8B

	full_text


double %72
Afsub8B7
5
	full_text(
&
$%74 = fsub double -0.000000e+00, %73
+double8B

	full_text


double %73
�getelementptr8Br
p
	full_textc
a
_%75 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %35, i64 %39, i64 0, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %35
%i648B

	full_text
	
i64 %39
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
�getelementptr8Br
p
	full_textc
a
_%76 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %35, i64 %39, i64 1, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %35
%i648B

	full_text
	
i64 %39
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %76, align 8, !tbaa !8
-double*8B

	full_text

double* %76
�getelementptr8Br
p
	full_textc
a
_%77 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %35, i64 %39, i64 2, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %35
%i648B

	full_text
	
i64 %39
Nstore8BC
A
	full_text4
2
0store double %66, double* %77, align 8, !tbaa !8
+double8B

	full_text


double %66
-double*8B

	full_text

double* %77
7fmul8B-
+
	full_text

%78 = fmul double %46, %52
+double8B

	full_text


double %46
+double8B

	full_text


double %52
�getelementptr8Br
p
	full_textc
a
_%79 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %35, i64 %39, i64 3, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %35
%i648B

	full_text
	
i64 %39
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
�getelementptr8Br
p
	full_textc
a
_%80 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %35, i64 %39, i64 4, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %35
%i648B

	full_text
	
i64 %39
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %80, align 8, !tbaa !8
-double*8B

	full_text

double* %80
7fmul8B-
+
	full_text

%81 = fmul double %54, %54
+double8B

	full_text


double %54
+double8B

	full_text


double %54
7fmul8B-
+
	full_text

%82 = fmul double %47, %81
+double8B

	full_text


double %47
+double8B

	full_text


double %81
Afsub8B7
5
	full_text(
&
$%83 = fsub double -0.000000e+00, %82
+double8B

	full_text


double %82
�getelementptr8Bz
x
	full_textk
i
g%84 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %25, i64 %39, i64 %41, i64 %43
I[13 x [13 x double]]*8B,
*
	full_text

[13 x [13 x double]]* %25
%i648B

	full_text
	
i64 %39
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Nload8BD
B
	full_text5
3
1%85 = load double, double* %84, align 8, !tbaa !8
-double*8B

	full_text

double* %84
rcall8Bh
f
	full_textY
W
U%86 = tail call double @llvm.fmuladd.f64(double %85, double 4.000000e-01, double %83)
+double8B

	full_text


double %85
+double8B

	full_text


double %83
�getelementptr8Br
p
	full_textc
a
_%87 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %35, i64 %39, i64 0, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %35
%i648B

	full_text
	
i64 %39
Nstore8BC
A
	full_text4
2
0store double %86, double* %87, align 8, !tbaa !8
+double8B

	full_text


double %86
-double*8B

	full_text

double* %87
Afmul8B7
5
	full_text(
&
$%88 = fmul double %50, -4.000000e-01
+double8B

	full_text


double %50
7fmul8B-
+
	full_text

%89 = fmul double %46, %88
+double8B

	full_text


double %46
+double8B

	full_text


double %88
�getelementptr8Br
p
	full_textc
a
_%90 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %35, i64 %39, i64 1, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %35
%i648B

	full_text
	
i64 %39
Nstore8BC
A
	full_text4
2
0store double %89, double* %90, align 8, !tbaa !8
+double8B

	full_text


double %89
-double*8B

	full_text

double* %90
Afmul8B7
5
	full_text(
&
$%91 = fmul double %52, -4.000000e-01
+double8B

	full_text


double %52
7fmul8B-
+
	full_text

%92 = fmul double %46, %91
+double8B

	full_text


double %46
+double8B

	full_text


double %91
�getelementptr8Br
p
	full_textc
a
_%93 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %35, i64 %39, i64 2, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %35
%i648B

	full_text
	
i64 %39
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
@fmul8B6
4
	full_text'
%
#%94 = fmul double %54, 1.600000e+00
+double8B

	full_text


double %54
7fmul8B-
+
	full_text

%95 = fmul double %46, %94
+double8B

	full_text


double %46
+double8B

	full_text


double %94
�getelementptr8Br
p
	full_textc
a
_%96 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %35, i64 %39, i64 3, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %35
%i648B

	full_text
	
i64 %39
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
_%97 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %35, i64 %39, i64 4, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %35
%i648B

	full_text
	
i64 %39
Wstore8BL
J
	full_text=
;
9store double 4.000000e-01, double* %97, align 8, !tbaa !8
-double*8B

	full_text

double* %97
�getelementptr8Bz
x
	full_textk
i
g%98 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %26, i64 %39, i64 %41, i64 %43
I[13 x [13 x double]]*8B,
*
	full_text

[13 x [13 x double]]* %26
%i648B

	full_text
	
i64 %39
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Nload8BD
B
	full_text5
3
1%99 = load double, double* %98, align 8, !tbaa !8
-double*8B

	full_text

double* %98
Afmul8B7
5
	full_text(
&
$%100 = fmul double %56, 1.400000e+00
+double8B

	full_text


double %56
Cfsub8B9
7
	full_text*
(
&%101 = fsub double -0.000000e+00, %100
,double8B

	full_text

double %100
tcall8Bj
h
	full_text[
Y
W%102 = tail call double @llvm.fmuladd.f64(double %99, double 8.000000e-01, double %101)
+double8B

	full_text


double %99
,double8B

	full_text

double %101
9fmul8B/
-
	full_text 

%103 = fmul double %54, %102
+double8B

	full_text


double %54
,double8B

	full_text

double %102
9fmul8B/
-
	full_text 

%104 = fmul double %47, %103
+double8B

	full_text


double %47
,double8B

	full_text

double %103
�getelementptr8Bs
q
	full_textd
b
`%105 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %35, i64 %39, i64 0, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %35
%i648B

	full_text
	
i64 %39
Pstore8BE
C
	full_text6
4
2store double %104, double* %105, align 8, !tbaa !8
,double8B

	full_text

double %104
.double*8B

	full_text

double* %105
Bfmul8B8
6
	full_text)
'
%%106 = fmul double %62, -4.000000e-01
+double8B

	full_text


double %62
9fmul8B/
-
	full_text 

%107 = fmul double %47, %106
+double8B

	full_text


double %47
,double8B

	full_text

double %106
�getelementptr8Bs
q
	full_textd
b
`%108 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %35, i64 %39, i64 1, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %35
%i648B

	full_text
	
i64 %39
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
Bfmul8B8
6
	full_text)
'
%%109 = fmul double %72, -4.000000e-01
+double8B

	full_text


double %72
9fmul8B/
-
	full_text 

%110 = fmul double %47, %109
+double8B

	full_text


double %47
,double8B

	full_text

double %109
�getelementptr8Bs
q
	full_textd
b
`%111 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %35, i64 %39, i64 2, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %35
%i648B

	full_text
	
i64 %39
Pstore8BE
C
	full_text6
4
2store double %110, double* %111, align 8, !tbaa !8
,double8B

	full_text

double %110
.double*8B

	full_text

double* %111
8fmul8B.
,
	full_text

%112 = fmul double %46, %56
+double8B

	full_text


double %46
+double8B

	full_text


double %56
Oload8BE
C
	full_text6
4
2%113 = load double, double* %84, align 8, !tbaa !8
-double*8B

	full_text

double* %84
kcall8Ba
_
	full_textR
P
N%114 = tail call double @llvm.fmuladd.f64(double %81, double %47, double %113)
+double8B

	full_text


double %81
+double8B

	full_text


double %47
,double8B

	full_text

double %113
Bfmul8B8
6
	full_text)
'
%%115 = fmul double %114, 4.000000e-01
,double8B

	full_text

double %114
Cfsub8B9
7
	full_text*
(
&%116 = fsub double -0.000000e+00, %115
,double8B

	full_text

double %115
ucall8Bk
i
	full_text\
Z
X%117 = tail call double @llvm.fmuladd.f64(double %112, double 1.400000e+00, double %116)
,double8B

	full_text

double %112
,double8B

	full_text

double %116
�getelementptr8Bs
q
	full_textd
b
`%118 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %35, i64 %39, i64 3, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %35
%i648B

	full_text
	
i64 %39
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
$%119 = fmul double %54, 1.400000e+00
+double8B

	full_text


double %54
9fmul8B/
-
	full_text 

%120 = fmul double %46, %119
+double8B

	full_text


double %46
,double8B

	full_text

double %119
�getelementptr8Bs
q
	full_textd
b
`%121 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %35, i64 %39, i64 4, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %35
%i648B

	full_text
	
i64 %39
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
`%122 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %39, i64 0, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %39
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
`%123 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %39, i64 1, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %39
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
`%124 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %39, i64 2, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %39
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
`%125 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %39, i64 3, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %39
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
`%126 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %39, i64 4, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %39
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
%%127 = fmul double %47, -1.000000e-01
+double8B

	full_text


double %47
9fmul8B/
-
	full_text 

%128 = fmul double %50, %127
+double8B

	full_text


double %50
,double8B

	full_text

double %127
�getelementptr8Bs
q
	full_textd
b
`%129 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %39, i64 0, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %39
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
$%130 = fmul double %46, 1.000000e-01
+double8B

	full_text


double %46
�getelementptr8Bs
q
	full_textd
b
`%131 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %39, i64 1, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %39
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
`%132 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %39, i64 2, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %39
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
`%133 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %39, i64 3, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %39
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
`%134 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %39, i64 4, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %39
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %134, align 8, !tbaa !8
.double*8B

	full_text

double* %134
9fmul8B/
-
	full_text 

%135 = fmul double %52, %127
+double8B

	full_text


double %52
,double8B

	full_text

double %127
�getelementptr8Bs
q
	full_textd
b
`%136 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %39, i64 0, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %39
Pstore8BE
C
	full_text6
4
2store double %135, double* %136, align 8, !tbaa !8
,double8B

	full_text

double %135
.double*8B

	full_text

double* %136
�getelementptr8Bs
q
	full_textd
b
`%137 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %39, i64 1, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %39
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %137, align 8, !tbaa !8
.double*8B

	full_text

double* %137
�getelementptr8Bs
q
	full_textd
b
`%138 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %39, i64 2, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %39
Pstore8BE
C
	full_text6
4
2store double %130, double* %138, align 8, !tbaa !8
,double8B

	full_text

double %130
.double*8B

	full_text

double* %138
�getelementptr8Bs
q
	full_textd
b
`%139 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %39, i64 3, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %39
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %139, align 8, !tbaa !8
.double*8B

	full_text

double* %139
�getelementptr8Bs
q
	full_textd
b
`%140 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %39, i64 4, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %39
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %140, align 8, !tbaa !8
.double*8B

	full_text

double* %140
Gfmul8B=
;
	full_text.
,
*%141 = fmul double %47, 0xBFC1111111111111
+double8B

	full_text


double %47
9fmul8B/
-
	full_text 

%142 = fmul double %141, %54
,double8B

	full_text

double %141
+double8B

	full_text


double %54
�getelementptr8Bs
q
	full_textd
b
`%143 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %39, i64 0, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %39
Pstore8BE
C
	full_text6
4
2store double %142, double* %143, align 8, !tbaa !8
,double8B

	full_text

double %142
.double*8B

	full_text

double* %143
�getelementptr8Bs
q
	full_textd
b
`%144 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %39, i64 1, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %39
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %144, align 8, !tbaa !8
.double*8B

	full_text

double* %144
�getelementptr8Bs
q
	full_textd
b
`%145 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %39, i64 2, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %39
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %145, align 8, !tbaa !8
.double*8B

	full_text

double* %145
Gfmul8B=
;
	full_text.
,
*%146 = fmul double %46, 0x3FC1111111111111
+double8B

	full_text


double %46
�getelementptr8Bs
q
	full_textd
b
`%147 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %39, i64 3, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %39
Pstore8BE
C
	full_text6
4
2store double %146, double* %147, align 8, !tbaa !8
,double8B

	full_text

double %146
.double*8B

	full_text

double* %147
�getelementptr8Bs
q
	full_textd
b
`%148 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %39, i64 4, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %39
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
*%149 = fmul double %48, 0x3FB89374BC6A7EF8
+double8B

	full_text


double %48
8fmul8B.
,
	full_text

%150 = fmul double %50, %50
+double8B

	full_text


double %50
+double8B

	full_text


double %50
Gfmul8B=
;
	full_text.
,
*%151 = fmul double %48, 0xBFB89374BC6A7EF8
+double8B

	full_text


double %48
8fmul8B.
,
	full_text

%152 = fmul double %52, %52
+double8B

	full_text


double %52
+double8B

	full_text


double %52
:fmul8B0
.
	full_text!

%153 = fmul double %152, %151
,double8B

	full_text

double %152
,double8B

	full_text

double %151
Cfsub8B9
7
	full_text*
(
&%154 = fsub double -0.000000e+00, %153
,double8B

	full_text

double %153
mcall8Bc
a
	full_textT
R
P%155 = tail call double @llvm.fmuladd.f64(double %149, double %150, double %154)
,double8B

	full_text

double %149
,double8B

	full_text

double %150
,double8B

	full_text

double %154
Gfmul8B=
;
	full_text.
,
*%156 = fmul double %48, 0x3FB00AEC33E1F670
+double8B

	full_text


double %48
lcall8Bb
`
	full_textS
Q
O%157 = tail call double @llvm.fmuladd.f64(double %156, double %81, double %155)
,double8B

	full_text

double %156
+double8B

	full_text


double %81
,double8B

	full_text

double %155
Gfmul8B=
;
	full_text.
,
*%158 = fmul double %47, 0x3FC916872B020C49
+double8B

	full_text


double %47
Cfsub8B9
7
	full_text*
(
&%159 = fsub double -0.000000e+00, %158
,double8B

	full_text

double %158
lcall8Bb
`
	full_textS
Q
O%160 = tail call double @llvm.fmuladd.f64(double %159, double %56, double %157)
,double8B

	full_text

double %159
+double8B

	full_text


double %56
,double8B

	full_text

double %157
�getelementptr8Bs
q
	full_textd
b
`%161 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %39, i64 0, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %39
Pstore8BE
C
	full_text6
4
2store double %160, double* %161, align 8, !tbaa !8
,double8B

	full_text

double %160
.double*8B

	full_text

double* %161
Gfmul8B=
;
	full_text.
,
*%162 = fmul double %47, 0xBFB89374BC6A7EF8
+double8B

	full_text


double %47
9fmul8B/
-
	full_text 

%163 = fmul double %50, %162
+double8B

	full_text


double %50
,double8B

	full_text

double %162
�getelementptr8Bs
q
	full_textd
b
`%164 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %39, i64 1, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %39
Pstore8BE
C
	full_text6
4
2store double %163, double* %164, align 8, !tbaa !8
,double8B

	full_text

double %163
.double*8B

	full_text

double* %164
9fmul8B/
-
	full_text 

%165 = fmul double %52, %162
+double8B

	full_text


double %52
,double8B

	full_text

double %162
�getelementptr8Bs
q
	full_textd
b
`%166 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %39, i64 2, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %39
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
Gfmul8B=
;
	full_text.
,
*%167 = fmul double %47, 0xBFB00AEC33E1F670
+double8B

	full_text


double %47
9fmul8B/
-
	full_text 

%168 = fmul double %167, %54
,double8B

	full_text

double %167
+double8B

	full_text


double %54
�getelementptr8Bs
q
	full_textd
b
`%169 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %39, i64 3, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %39
Pstore8BE
C
	full_text6
4
2store double %168, double* %169, align 8, !tbaa !8
,double8B

	full_text

double %168
.double*8B

	full_text

double* %169
Gfmul8B=
;
	full_text.
,
*%170 = fmul double %46, 0x3FC916872B020C49
+double8B

	full_text


double %46
�getelementptr8Bs
q
	full_textd
b
`%171 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %39, i64 4, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %39
Pstore8BE
C
	full_text6
4
2store double %170, double* %171, align 8, !tbaa !8
,double8B

	full_text

double %170
.double*8B

	full_text

double* %171
(br8B 

	full_text

br label %172
$ret8B

	full_text


ret void
,double*8B

	full_text


double* %4
$i328B

	full_text


i32 %6
,double*8B

	full_text


double* %3
$i328B

	full_text


i32 %5
,double*8B

	full_text


double* %0
,double*8B

	full_text


double* %2
,double*8B

	full_text


double* %1
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
$i328B

	full_text


i32 -1
4double8B&
$
	full_text

double 0.000000e+00
4double8B&
$
	full_text

double 1.600000e+00
5double8B'
%
	full_text

double -0.000000e+00
4double8B&
$
	full_text

double 8.000000e-01
#i328B

	full_text	

i32 2
#i648B

	full_text	

i64 0
:double8B,
*
	full_text

double 0xBFB00AEC33E1F670
:double8B,
*
	full_text

double 0xBFC1111111111111
:double8B,
*
	full_text

double 0x3FB89374BC6A7EF8
4double8B&
$
	full_text

double 1.000000e+00
#i328B

	full_text	

i32 1
4double8B&
$
	full_text

double 1.400000e+00
5double8B'
%
	full_text

double -1.000000e-01
:double8B,
*
	full_text

double 0x3FC916872B020C49
$i328B

	full_text


i32 -2
#i648B

	full_text	

i64 3
%i328B

	full_text
	
i32 325
:double8B,
*
	full_text

double 0xBFB89374BC6A7EF8
4double8B&
$
	full_text

double 4.000000e-01
#i648B

	full_text	

i64 4
:double8B,
*
	full_text

double 0x3FB00AEC33E1F670
:double8B,
*
	full_text

double 0x3FC1111111111111
5double8B'
%
	full_text

double -4.000000e-01
#i648B

	full_text	

i64 1
4double8B&
$
	full_text

double 1.000000e-01
$i648B

	full_text


i64 32
#i648B

	full_text	

i64 2
#i328B

	full_text	

i32 0        	
 		                       !! "# "" $% $& $$ '( '' )* )+ )) ,- ,, ./ .. 01 00 23 22 45 44 67 66 89 88 :; :: <= << >? >> @A @@ BC BB DE DF DG DH DD IJ II KL KK MN MO MM PQ PR PP ST SU SV SW SS XY XX Z[ Z\ Z] Z^ ZZ _` __ ab ac ad ae aa fg ff hi hj hk hl hh mn mm op oq oo rs rr tu tv tt wx ww yz y{ yy |} || ~ ~	� ~~ �
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
� �� �
� �� �� �
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
� �� �
� �� �� �
� �� �� �
� �� �� �
� �� �� �
� �� �� �
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
� �� �� �
� �� �� �
� �� �� �� �� �
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
� �� �� �� �� �
� �� �� �
� �� �� �
� �� �� �� �� �
� �� �� �
� �� �� �
� �� �� �
� �� �� �� �� �
� �
� �� �� �� �
� �� �� �
� �� �� �
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
� �� �� �� �� �
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
� �� �� �
� �� �� �
� �� �� �
� �� �� �
� �� �
� �� �� �
� �� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �� �� �
� �� �� �
� �� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �� �� �
� �� �� �
� �� �� �
� �� �
� �� �� �� �� �
� �� �� �� �� �
� �� �� �
� �� �
� �� �� �
� �
� �� �� �� �� �
� �
� �� �� �� �
� �� �� �
� �
� �� �� �
� �� �� �
� �� �� �� �� �
� �� �� �
� �� �� �
� �� �� �
� �� �� �
� �� �� �
� �� �� �� �� �
� �� �� �
� �� �� �
� �� �� �� �� �
� �� �� �
� �� �� 4� � 0� � � !�  	�     
     	      #" % &	 (' *$ +) -, /. 10 3. 54 7 98 ; =< ? A@ C! E: F> GB HD JI LK NK OK QM R! T: U> VB WS Y! [: \> ]B ^Z `! b: c> dB ea g! i: j> kB lh n2 p: qo s2 u: vt x2 z: {y }2 : �~ �2 �: �� �X �f �M �� �� �2 �: �� �� �K �f �2 �: �� �� �2 �: �� �X �K �2 �: �� �� �2 �: �� �_ �f �M �� �� �2 �: �� �� �2 �: �� �2 �: �� �� �K �_ �2 �: �� �� �2 �: �� �f �f �M �� �� � �: �> �B �� �� �� �2 �: �� �� �X �K �� �2 �: �� �� �_ �K �� �2 �: �� �� �f �K �� �2 �: �� �� �2 �: �� �  �: �> �B �� �m �� �� �� �f �� �M �� �2 �: �� �� �� �M �� �2 �: �� �� �� �M �� �2 �: �� �� �K �m �� �� �M �� �� �� �� �� �2 �: �� �� �f �K �� �2 �: �� �� �6 �: �� �6 �: �� �6 �: �� �6 �: �� �6 �: �� �M �X �� �6 �: �� �� �K �6 �: �� �� �6 �: �� �6 �: �� �6 �: �� �_ �� �6 �: �� �� �6 �: �� �6 �: �� �� �6 �: �� �6 �: �� �M �� �f �6 �: �� �� �6 �: �� �6 �: �� �K �6 �: �� �� �6 �: �� �P �X �X �P �_ �_ �� �� �� �� �� �� �P �� �� �� �M �� �� �m �� �6 �: �� �� �M �X �� �6 �: �� �� �_ �� �6 �: �� �� �M �� �f �6 �: �� �� �K �6 �: �� �� � �   �� � �� � ��� �� �� �� �� �� �� �� � ��  �� � �� �� �� �� �� � �� 	� "	� '� r� w� |� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �
� �� �� �� �� �� �� �� �
� �� 	� D	� o	� o	� t	� y	� ~
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
� �
� �
� �� K� �� 
� �
� �
� �
� �
� �
� �	� 	� 	� a	� ~
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
� �	� ,
� �
� �
� �� �
� �	� h
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
� �
� �
� �
� �
� �
� �
� �	� 	� 	� S	� t
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
� �
� �	� 8	� :	� <	� >	� @	� B	� Z	� y
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
� �� "

z_solve1"
_Z13get_global_idj"
llvm.fmuladd.f64*�
npb-BT-z_solve1.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282

wgsize
<

transfer_bytes
��n

devmap_label
 

wgsize_log1p
�fA
 
transfer_bytes_log1p
�fA