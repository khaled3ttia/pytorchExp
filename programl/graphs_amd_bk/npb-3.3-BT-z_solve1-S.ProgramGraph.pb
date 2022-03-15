
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
¢getelementptr8Bé
ã
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
¢getelementptr8Bé
ã
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
¢getelementptr8Bé
ã
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
¢getelementptr8Bé
ã
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
¢getelementptr8Bé
ã
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
Ögetelementptr8Br
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
Ögetelementptr8Br
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
Ögetelementptr8Br
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
Ögetelementptr8Br
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
Ögetelementptr8Br
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
Ögetelementptr8Br
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
Ögetelementptr8Br
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
Ögetelementptr8Br
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
Ögetelementptr8Br
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
Ögetelementptr8Br
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
Ögetelementptr8Br
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
Ögetelementptr8Br
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
Ögetelementptr8Br
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
Ögetelementptr8Br
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
Ögetelementptr8Br
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
çgetelementptr8Bz
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
Ögetelementptr8Br
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
Ögetelementptr8Br
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
Ögetelementptr8Br
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
Ögetelementptr8Br
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
Ögetelementptr8Br
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
çgetelementptr8Bz
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
Ügetelementptr8Bs
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
Ügetelementptr8Bs
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
Ügetelementptr8Bs
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
Ügetelementptr8Bs
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
Ügetelementptr8Bs
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
Ügetelementptr8Bs
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
Ügetelementptr8Bs
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
Ügetelementptr8Bs
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
Ügetelementptr8Bs
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
Ügetelementptr8Bs
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
Ügetelementptr8Bs
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
Ügetelementptr8Bs
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
Ügetelementptr8Bs
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
Ügetelementptr8Bs
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
Ügetelementptr8Bs
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
Ügetelementptr8Bs
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
Ügetelementptr8Bs
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
Ügetelementptr8Bs
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
Ügetelementptr8Bs
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
Ügetelementptr8Bs
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
Ügetelementptr8Bs
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
Ügetelementptr8Bs
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
Ügetelementptr8Bs
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
Ügetelementptr8Bs
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
Ügetelementptr8Bs
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
Ügetelementptr8Bs
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
Ügetelementptr8Bs
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
Ügetelementptr8Bs
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
Ügetelementptr8Bs
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
Ügetelementptr8Bs
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
 		                       !! "# "" $% $& $$ '( '' )* )+ )) ,- ,, ./ .. 01 00 23 22 45 44 67 66 89 88 :; :: <= << >? >> @A @@ BC BB DE DF DG DH DD IJ II KL KK MN MO MM PQ PR PP ST SU SV SW SS XY XX Z[ Z\ Z] Z^ ZZ _` __ ab ac ad ae aa fg ff hi hj hk hl hh mn mm op oq oo rs rr tu tv tt wx ww yz y{ yy |} || ~ ~	Ä ~~ Å
Ç ÅÅ ÉÑ É
Ö ÉÉ Ü
á ÜÜ àâ à
ä àà ãå ã
ç ãã é
è éé êë ê
í êê ìî ì
ï ìì ñó ñ
ò ññ ôö ô
õ ôô úù ú
û úú ü† ü
° üü ¢
£ ¢¢ §• §
¶ §§ ß® ß
© ßß ™´ ™
¨ ™™ ≠Æ ≠
Ø ≠≠ ∞
± ∞∞ ≤≥ ≤
¥ ≤≤ µ∂ µ
∑ µµ ∏
π ∏∏ ∫ª ∫
º ∫∫ Ωæ Ω
ø ΩΩ ¿¡ ¿
¬ ¿¿ √
ƒ √√ ≈∆ ≈
« ≈≈ »… »
  »» ÀÃ À
Õ ÀÀ Œœ Œ
– ŒŒ —“ —
” —— ‘’ ‘
÷ ‘‘ ◊
ÿ ◊◊ Ÿ⁄ Ÿ
€ ŸŸ ‹› ‹
ﬁ ‹‹ ﬂ
‡ ﬂﬂ ·‚ ·
„ ·
‰ ·
Â ·· ÊÁ ÊÊ ËÈ Ë
Í ËË ÎÏ Î
Ì ÎÎ ÓÔ Ó
 ÓÓ ÒÚ ÒÒ ÛÙ Û
ı ÛÛ ˆ˜ ˆ
¯ ˆˆ ˘˙ ˘
˚ ˘˘ ¸˝ ¸¸ ˛ˇ ˛
Ä ˛˛ ÅÇ Å
É ÅÅ ÑÖ Ñ
Ü ÑÑ áà áá âä â
ã ââ åç å
é åå èê è
ë èè íì í
î íí ï
ñ ïï óò ó
ô ó
ö ó
õ óó úù úú ûü ûû †
° †† ¢£ ¢
§ ¢¢ •¶ •
ß •• ®© ®
™ ®® ´¨ ´
≠ ´´ ÆØ Æ
∞ ÆÆ ±≤ ±± ≥¥ ≥
µ ≥≥ ∂∑ ∂
∏ ∂∂ π∫ π
ª ππ ºΩ ºº æø æ
¿ ææ ¡¬ ¡
√ ¡¡ ƒ≈ ƒ
∆ ƒƒ «» «
… ««  À    ÃÕ Ã
Œ Ã
œ ÃÃ –— –– “
” ““ ‘’ ‘
÷ ‘‘ ◊ÿ ◊
Ÿ ◊◊ ⁄€ ⁄
‹ ⁄⁄ ›ﬁ ›› ﬂ‡ ﬂ
· ﬂﬂ ‚„ ‚
‰ ‚‚ ÂÊ Â
Á ÂÂ ËÈ Ë
Í ËË Î
Ï ÎÎ ÌÓ Ì
Ô ÌÌ 
Ò  ÚÛ Ú
Ù ÚÚ ı
ˆ ıı ˜¯ ˜
˘ ˜˜ ˙
˚ ˙˙ ¸˝ ¸
˛ ¸¸ ˇ
Ä ˇˇ ÅÇ ÅÅ ÉÑ É
Ö ÉÉ Üá Ü
à ÜÜ âä â
ã ââ åç åå éè é
ê éé ëí ë
ì ëë îï î
ñ îî ó
ò óó ôö ô
õ ôô ú
ù úú ûü û
† ûû °
¢ °° £§ £
• ££ ¶ß ¶
® ¶¶ ©™ ©
´ ©© ¨≠ ¨
Æ ¨¨ Ø
∞ ØØ ±≤ ±
≥ ±± ¥µ ¥
∂ ¥¥ ∑∏ ∑
π ∑∑ ∫
ª ∫∫ ºΩ º
æ ºº ø
¿ øø ¡¬ ¡¡ √ƒ √
≈ √√ ∆« ∆
» ∆∆ …  …
À …… ÃÕ Ã
Œ ÃÃ œ
– œœ —“ —
” —— ‘
’ ‘‘ ÷◊ ÷÷ ÿŸ ÿ
⁄ ÿÿ €‹ €
› €€ ﬁﬂ ﬁ
‡ ﬁﬁ ·
‚ ·· „‰ „„ ÂÊ Â
Á ÂÂ ËÈ ËË ÍÎ Í
Ï ÍÍ ÌÓ Ì
Ô ÌÌ 
Ò  ÚÛ Ú
Ù Ú
ı ÚÚ ˆ˜ ˆˆ ¯˘ ¯
˙ ¯
˚ ¯¯ ¸˝ ¸¸ ˛
ˇ ˛˛ ÄÅ Ä
Ç Ä
É ÄÄ ÑÖ Ñ
Ü ÑÑ áà á
â áá äã ää åç å
é åå èê è
ë èè íì í
î íí ïñ ï
ó ïï òô ò
ö òò õú õ
ù õõ ûü ûû †° †
¢ †† £§ £
• ££ ¶ß ¶
® ¶¶ ©™ ©© ´¨ ´
≠ ´´ ÆØ Æ
∞ ÆÆ ±≥ 4¥ µ 0∂ ∑ ∏ !π  	∫     
     	      #" % &	 (' *$ +) -, /. 10 3. 54 7 98 ; =< ? A@ C! E: F> GB HD JI LK NK OK QM R! T: U> VB WS Y! [: \> ]B ^Z `! b: c> dB ea g! i: j> kB lh n2 p: qo s2 u: vt x2 z: {y }2 : Ä~ Ç2 Ñ: ÖÉ áX âf äM åà çã è2 ë: íé îê ïK óf ò2 ö: õñ ùô û2 †: °ü £X •K ¶2 ®: ©§ ´ß ¨2 Æ: Ø≠ ±_ ≥f ¥M ∂≤ ∑µ π2 ª: º∏ æ∫ ø2 ¡: ¬¿ ƒ2 ∆: «ñ …≈  K Ã_ Õ2 œ: –À “Œ ”2 ’: ÷‘ ÿf ⁄f €M ›Ÿ ﬁ‹ ‡ ‚: „> ‰B Â· ÁÊ Èﬂ Í2 Ï: ÌË ÔÎ X ÚK ÙÒ ı2 ˜: ¯Û ˙ˆ ˚_ ˝K ˇ¸ Ä2 Ç: É˛ ÖÅ Üf àK äá ã2 ç: éâ êå ë2 ì: îí ñ  ò: ô> öB õó ùm üû °ú £† §f ¶¢ ßM ©• ™2 ¨: ≠® Ø´ ∞à ≤M ¥± µ2 ∑: ∏≥ ∫∂ ª≤ ΩM øº ¿2 ¬: √æ ≈¡ ∆K »m …· ÀŸ ÕM Œ  œÃ —– ”« ’“ ÷2 ÿ: Ÿ‘ €◊ ‹f ﬁK ‡› ·2 „: ‰ﬂ Ê‚ Á6 È: ÍË Ï6 Ó: ÔÌ Ò6 Û: ÙÚ ˆ6 ¯: ˘˜ ˚6 ˝: ˛¸ ÄM ÇX ÑÅ Ö6 á: àÉ äÜ ãK ç6 è: êå íé ì6 ï: ñî ò6 ö: õô ù6 ü: †û ¢_ §Å •6 ß: ®£ ™¶ ´6 ≠: Æ¨ ∞6 ≤: ≥å µ± ∂6 ∏: π∑ ª6 Ω: æº ¿M ¬¡ ƒf ≈6 «: »√  ∆ À6 Õ: ŒÃ –6 “: ”— ’K ◊6 Ÿ: ⁄÷ ‹ÿ ›6 ﬂ: ‡ﬁ ‚P ‰X ÊX ÁP È_ Î_ ÏÍ ÓË ÔÌ Ò„ ÛÂ Ù ıP ˜ˆ ˘Ÿ ˙Ú ˚M ˝¸ ˇ˛ Åm Ç¯ É6 Ö: ÜÄ àÑ âM ãX çä é6 ê: ëå ìè î_ ñä ó6 ô: öï úò ùM üû °f ¢6 §: •† ß£ ®K ™6 ¨: ≠© Ø´ ∞ ≤   ≤± ≤ ªª ≤ ºº‘ ºº ‘Ä ºº ÄË ºº Ë¢ ºº ¢ ªª  ªª Ã ºº ÃÚ ºº Ú¯ ºº ¯ ªª 	Ω "	Ω 'æ ræ wæ |æ Üæ ¢æ ∞æ √æ ◊æ Îæ æ ıæ ˙æ ˇæ óæ úæ °æ Øæ ∫æ øæ œæ ‘æ ·
ø á¿ é¿ ∏¿ ﬂ¿ †¿ “¿ ¿ ˛
¡ ¢¬ 	√ D	√ o	√ o	√ t	√ y	√ ~
√ É
√ ê
√ ∫
√ Î
√ ´
√ Ë
√ Ë
√ Ì
√ Ú
√ ˜
√ ¸
√ Ü
√ ¶
√ ∆
√ Ñ
ƒ û
≈ ¡
∆ „« K« Å» 
… û
… ‘
… ›
  Å
À ¸
À ©	Ã 	Ã 	Õ a	Õ ~
Õ ß
Õ Œ
Õ Î
Õ ˆ
Õ Å
Õ å
Õ å
Õ í
Õ ◊
Õ ˜
Õ ô
Õ ∑
Õ ∆
Õ Ã
Õ —
Õ ÿ
Õ ÿ
Õ ﬁ
Õ £	Œ ,
œ Ë
œ ä
– Ë– ï
– –	— h
— É
— ≠
— ‘
— í
— ´
— ∂
— ¡
— ◊
— ‚
— ‚
— ¸
— û
— º
— ﬁ
— Ñ
— è
— ò
— £
— ´
— ´
“ ˆ
” ÷
‘ Ò
‘ ¸
‘ ±
‘ º	’ 	’ 	’ S	’ t
’ ê
’ ô
’ ô
’ ü
’ ß
’ ≠
’ ¿
’ ˆ
’ ∂
’ Ì
’ Ü
’ é
’ é
’ î
’ ô
’ û
’ ¨
’ Ã
’ è
÷ å	◊ 8	◊ :	◊ <	◊ >	◊ @	◊ B	ÿ Z	ÿ y
ÿ ü
ÿ ∫
ÿ ¿
ÿ ≈
ÿ ≈
ÿ Œ
ÿ ‘
ÿ Å
ÿ ¡
ÿ Ú
ÿ î
ÿ ¶
ÿ ¨
ÿ ±
ÿ ±
ÿ ∑
ÿ º
ÿ —
ÿ òŸ "

z_solve1"
_Z13get_global_idj"
llvm.fmuladd.f64*ã
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
¯¨n

devmap_label
 

wgsize_log1p
ÜfA
 
transfer_bytes_log1p
ÜfA