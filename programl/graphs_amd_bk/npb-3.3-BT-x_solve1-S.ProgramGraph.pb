
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
%21 = add nsw i32 %7, -2
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

%23 = icmp slt i32 %20, %6
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
Qbitcast8BD
B
	full_text5
3
1%26 = bitcast double* %0 to [13 x [13 x double]]*
Qbitcast8BD
B
	full_text5
3
1%27 = bitcast double* %1 to [13 x [13 x double]]*
Qbitcast8BD
B
	full_text5
3
1%28 = bitcast double* %2 to [13 x [13 x double]]*
Wbitcast8BJ
H
	full_text;
9
7%29 = bitcast double* %3 to [13 x [13 x [5 x double]]]*
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
2mul8B)
'
	full_text

%34 = mul i32 %33, 325
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
%42 = shl i64 %14, 32
%i648B

	full_text
	
i64 %14
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
%44 = shl i64 %16, 32
%i648B

	full_text
	
i64 %16
9ashr8B/
-
	full_text 

%45 = ashr exact i64 %44, 32
%i648B

	full_text
	
i64 %44
çgetelementptr8Bz
x
	full_textk
i
g%46 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %27, i64 %41, i64 %43, i64 %45
I[13 x [13 x double]]*8B,
*
	full_text

[13 x [13 x double]]* %27
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
¢getelementptr8Bé
ã
	full_text~
|
z%50 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %29, i64 %41, i64 %43, i64 %45, i64 0
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %29
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
¢getelementptr8Bé
ã
	full_text~
|
z%52 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %29, i64 %41, i64 %43, i64 %45, i64 1
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %29
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
¢getelementptr8Bé
ã
	full_text~
|
z%54 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %29, i64 %41, i64 %43, i64 %45, i64 2
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %29
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
¢getelementptr8Bé
ã
	full_text~
|
z%56 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %29, i64 %41, i64 %43, i64 %45, i64 3
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %29
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
¢getelementptr8Bé
ã
	full_text~
|
z%58 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %29, i64 %41, i64 %43, i64 %45, i64 4
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %29
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
1%59 = load double, double* %58, align 8, !tbaa !8
-double*8B

	full_text

double* %58
Ögetelementptr8Br
p
	full_textc
a
_%60 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %45, i64 0, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %45
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %60, align 8, !tbaa !8
-double*8B

	full_text

double* %60
Ögetelementptr8Br
p
	full_textc
a
_%61 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %45, i64 1, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %45
Wstore8BL
J
	full_text=
;
9store double 1.000000e+00, double* %61, align 8, !tbaa !8
-double*8B

	full_text

double* %61
Ögetelementptr8Br
p
	full_textc
a
_%62 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %45, i64 2, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %45
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %62, align 8, !tbaa !8
-double*8B

	full_text

double* %62
Ögetelementptr8Br
p
	full_textc
a
_%63 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %45, i64 3, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %45
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %63, align 8, !tbaa !8
-double*8B

	full_text

double* %63
Ögetelementptr8Br
p
	full_textc
a
_%64 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %45, i64 4, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %45
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %64, align 8, !tbaa !8
-double*8B

	full_text

double* %64
7fmul8B-
+
	full_text

%65 = fmul double %48, %53
+double8B

	full_text


double %48
+double8B

	full_text


double %53
7fmul8B-
+
	full_text

%66 = fmul double %53, %65
+double8B

	full_text


double %53
+double8B

	full_text


double %65
Afsub8B7
5
	full_text(
&
$%67 = fsub double -0.000000e+00, %66
+double8B

	full_text


double %66
çgetelementptr8Bz
x
	full_textk
i
g%68 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %26, i64 %41, i64 %43, i64 %45
I[13 x [13 x double]]*8B,
*
	full_text

[13 x [13 x double]]* %26
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
1%69 = load double, double* %68, align 8, !tbaa !8
-double*8B

	full_text

double* %68
rcall8Bh
f
	full_textY
W
U%70 = tail call double @llvm.fmuladd.f64(double %69, double 4.000000e-01, double %67)
+double8B

	full_text


double %69
+double8B

	full_text


double %67
Ögetelementptr8Br
p
	full_textc
a
_%71 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %45, i64 0, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %45
Nstore8BC
A
	full_text4
2
0store double %70, double* %71, align 8, !tbaa !8
+double8B

	full_text


double %70
-double*8B

	full_text

double* %71
7fdiv8B-
+
	full_text

%72 = fdiv double %53, %51
+double8B

	full_text


double %53
+double8B

	full_text


double %51
@fmul8B6
4
	full_text'
%
#%73 = fmul double %72, 1.600000e+00
+double8B

	full_text


double %72
Ögetelementptr8Br
p
	full_textc
a
_%74 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %45, i64 1, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %45
Nstore8BC
A
	full_text4
2
0store double %73, double* %74, align 8, !tbaa !8
+double8B

	full_text


double %73
-double*8B

	full_text

double* %74
7fmul8B-
+
	full_text

%75 = fmul double %47, %55
+double8B

	full_text


double %47
+double8B

	full_text


double %55
Afmul8B7
5
	full_text(
&
$%76 = fmul double %75, -4.000000e-01
+double8B

	full_text


double %75
Ögetelementptr8Br
p
	full_textc
a
_%77 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %45, i64 2, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %45
Nstore8BC
A
	full_text4
2
0store double %76, double* %77, align 8, !tbaa !8
+double8B

	full_text


double %76
-double*8B

	full_text

double* %77
7fmul8B-
+
	full_text

%78 = fmul double %47, %57
+double8B

	full_text


double %47
+double8B

	full_text


double %57
Afmul8B7
5
	full_text(
&
$%79 = fmul double %78, -4.000000e-01
+double8B

	full_text


double %78
Ögetelementptr8Br
p
	full_textc
a
_%80 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %45, i64 3, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %45
Nstore8BC
A
	full_text4
2
0store double %79, double* %80, align 8, !tbaa !8
+double8B

	full_text


double %79
-double*8B

	full_text

double* %80
Ögetelementptr8Br
p
	full_textc
a
_%81 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %45, i64 4, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %45
Wstore8BL
J
	full_text=
;
9store double 4.000000e-01, double* %81, align 8, !tbaa !8
-double*8B

	full_text

double* %81
7fmul8B-
+
	full_text

%82 = fmul double %53, %55
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

%83 = fmul double %48, %82
+double8B

	full_text


double %48
+double8B

	full_text


double %82
Afsub8B7
5
	full_text(
&
$%84 = fsub double -0.000000e+00, %83
+double8B

	full_text


double %83
Ögetelementptr8Br
p
	full_textc
a
_%85 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %45, i64 0, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %45
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
Ögetelementptr8Br
p
	full_textc
a
_%86 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %45, i64 1, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %45
Nstore8BC
A
	full_text4
2
0store double %75, double* %86, align 8, !tbaa !8
+double8B

	full_text


double %75
-double*8B

	full_text

double* %86
7fmul8B-
+
	full_text

%87 = fmul double %47, %53
+double8B

	full_text


double %47
+double8B

	full_text


double %53
Ögetelementptr8Br
p
	full_textc
a
_%88 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %45, i64 2, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %45
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
Ögetelementptr8Br
p
	full_textc
a
_%89 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %45, i64 3, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %45
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %89, align 8, !tbaa !8
-double*8B

	full_text

double* %89
Ögetelementptr8Br
p
	full_textc
a
_%90 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %45, i64 4, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %45
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %90, align 8, !tbaa !8
-double*8B

	full_text

double* %90
7fmul8B-
+
	full_text

%91 = fmul double %53, %57
+double8B

	full_text


double %53
+double8B

	full_text


double %57
7fmul8B-
+
	full_text

%92 = fmul double %48, %91
+double8B

	full_text


double %48
+double8B

	full_text


double %91
Afsub8B7
5
	full_text(
&
$%93 = fsub double -0.000000e+00, %92
+double8B

	full_text


double %92
Ögetelementptr8Br
p
	full_textc
a
_%94 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %45, i64 0, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %45
Nstore8BC
A
	full_text4
2
0store double %93, double* %94, align 8, !tbaa !8
+double8B

	full_text


double %93
-double*8B

	full_text

double* %94
Ögetelementptr8Br
p
	full_textc
a
_%95 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %45, i64 1, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %45
Nstore8BC
A
	full_text4
2
0store double %78, double* %95, align 8, !tbaa !8
+double8B

	full_text


double %78
-double*8B

	full_text

double* %95
Ögetelementptr8Br
p
	full_textc
a
_%96 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %45, i64 2, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %45
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %96, align 8, !tbaa !8
-double*8B

	full_text

double* %96
Ögetelementptr8Br
p
	full_textc
a
_%97 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %45, i64 3, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %45
Nstore8BC
A
	full_text4
2
0store double %87, double* %97, align 8, !tbaa !8
+double8B

	full_text


double %87
-double*8B

	full_text

double* %97
Ögetelementptr8Br
p
	full_textc
a
_%98 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %45, i64 4, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %45
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %98, align 8, !tbaa !8
-double*8B

	full_text

double* %98
çgetelementptr8Bz
x
	full_textk
i
g%99 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %28, i64 %41, i64 %43, i64 %45
I[13 x [13 x double]]*8B,
*
	full_text

[13 x [13 x double]]* %28
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
$%101 = fmul double %59, 1.400000e+00
+double8B

	full_text


double %59
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

%104 = fmul double %65, %103
+double8B

	full_text


double %65
,double8B

	full_text

double %103
Ügetelementptr8Bs
q
	full_textd
b
`%105 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %45, i64 0, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %45
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
8fmul8B.
,
	full_text

%106 = fmul double %53, %53
+double8B

	full_text


double %53
+double8B

	full_text


double %53
Oload8BE
C
	full_text6
4
2%107 = load double, double* %68, align 8, !tbaa !8
-double*8B

	full_text

double* %68
lcall8Bb
`
	full_textS
Q
O%108 = tail call double @llvm.fmuladd.f64(double %106, double %48, double %107)
,double8B

	full_text

double %106
+double8B

	full_text


double %48
,double8B

	full_text

double %107
Bfmul8B8
6
	full_text)
'
%%109 = fmul double %108, 4.000000e-01
,double8B

	full_text

double %108
Cfsub8B9
7
	full_text*
(
&%110 = fsub double -0.000000e+00, %109
,double8B

	full_text

double %109
lcall8Bb
`
	full_textS
Q
O%111 = tail call double @llvm.fmuladd.f64(double %101, double %47, double %110)
,double8B

	full_text

double %101
+double8B

	full_text


double %47
,double8B

	full_text

double %110
Ügetelementptr8Bs
q
	full_textd
b
`%112 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %45, i64 1, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %45
Pstore8BE
C
	full_text6
4
2store double %111, double* %112, align 8, !tbaa !8
,double8B

	full_text

double %111
.double*8B

	full_text

double* %112
Bfmul8B8
6
	full_text)
'
%%113 = fmul double %82, -4.000000e-01
+double8B

	full_text


double %82
9fmul8B/
-
	full_text 

%114 = fmul double %48, %113
+double8B

	full_text


double %48
,double8B

	full_text

double %113
Ügetelementptr8Bs
q
	full_textd
b
`%115 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %45, i64 2, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %45
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
%%116 = fmul double %91, -4.000000e-01
+double8B

	full_text


double %91
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
Ügetelementptr8Bs
q
	full_textd
b
`%118 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %45, i64 3, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %45
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
$%119 = fmul double %87, 1.400000e+00
+double8B

	full_text


double %87
Ügetelementptr8Bs
q
	full_textd
b
`%120 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %37, i64 %45, i64 4, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %37
%i648B

	full_text
	
i64 %45
Pstore8BE
C
	full_text6
4
2store double %119, double* %120, align 8, !tbaa !8
,double8B

	full_text

double %119
.double*8B

	full_text

double* %120
Ügetelementptr8Bs
q
	full_textd
b
`%121 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %39, i64 %45, i64 0, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %39
%i648B

	full_text
	
i64 %45
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %121, align 8, !tbaa !8
.double*8B

	full_text

double* %121
Ügetelementptr8Bs
q
	full_textd
b
`%122 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %39, i64 %45, i64 1, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %39
%i648B

	full_text
	
i64 %45
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
`%123 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %39, i64 %45, i64 2, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %39
%i648B

	full_text
	
i64 %45
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
`%124 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %39, i64 %45, i64 3, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %39
%i648B

	full_text
	
i64 %45
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
`%125 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %39, i64 %45, i64 4, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %39
%i648B

	full_text
	
i64 %45
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %125, align 8, !tbaa !8
.double*8B

	full_text

double* %125
Gfmul8B=
;
	full_text.
,
*%126 = fmul double %48, 0xBFC1111111111111
+double8B

	full_text


double %48
9fmul8B/
-
	full_text 

%127 = fmul double %126, %53
,double8B

	full_text

double %126
+double8B

	full_text


double %53
Ügetelementptr8Bs
q
	full_textd
b
`%128 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %39, i64 %45, i64 0, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %39
%i648B

	full_text
	
i64 %45
Pstore8BE
C
	full_text6
4
2store double %127, double* %128, align 8, !tbaa !8
,double8B

	full_text

double %127
.double*8B

	full_text

double* %128
Gfmul8B=
;
	full_text.
,
*%129 = fmul double %47, 0x3FC1111111111111
+double8B

	full_text


double %47
Ügetelementptr8Bs
q
	full_textd
b
`%130 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %39, i64 %45, i64 1, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %39
%i648B

	full_text
	
i64 %45
Pstore8BE
C
	full_text6
4
2store double %129, double* %130, align 8, !tbaa !8
,double8B

	full_text

double %129
.double*8B

	full_text

double* %130
Ügetelementptr8Bs
q
	full_textd
b
`%131 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %39, i64 %45, i64 2, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %39
%i648B

	full_text
	
i64 %45
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %131, align 8, !tbaa !8
.double*8B

	full_text

double* %131
Ügetelementptr8Bs
q
	full_textd
b
`%132 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %39, i64 %45, i64 3, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %39
%i648B

	full_text
	
i64 %45
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
`%133 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %39, i64 %45, i64 4, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %39
%i648B

	full_text
	
i64 %45
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %133, align 8, !tbaa !8
.double*8B

	full_text

double* %133
Bfmul8B8
6
	full_text)
'
%%134 = fmul double %48, -1.000000e-01
+double8B

	full_text


double %48
9fmul8B/
-
	full_text 

%135 = fmul double %134, %55
,double8B

	full_text

double %134
+double8B

	full_text


double %55
Ügetelementptr8Bs
q
	full_textd
b
`%136 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %39, i64 %45, i64 0, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %39
%i648B

	full_text
	
i64 %45
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
`%137 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %39, i64 %45, i64 1, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %39
%i648B

	full_text
	
i64 %45
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %137, align 8, !tbaa !8
.double*8B

	full_text

double* %137
Afmul8B7
5
	full_text(
&
$%138 = fmul double %47, 1.000000e-01
+double8B

	full_text


double %47
Ügetelementptr8Bs
q
	full_textd
b
`%139 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %39, i64 %45, i64 2, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %39
%i648B

	full_text
	
i64 %45
Pstore8BE
C
	full_text6
4
2store double %138, double* %139, align 8, !tbaa !8
,double8B

	full_text

double %138
.double*8B

	full_text

double* %139
Ügetelementptr8Bs
q
	full_textd
b
`%140 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %39, i64 %45, i64 3, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %39
%i648B

	full_text
	
i64 %45
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %140, align 8, !tbaa !8
.double*8B

	full_text

double* %140
Ügetelementptr8Bs
q
	full_textd
b
`%141 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %39, i64 %45, i64 4, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %39
%i648B

	full_text
	
i64 %45
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %141, align 8, !tbaa !8
.double*8B

	full_text

double* %141
9fmul8B/
-
	full_text 

%142 = fmul double %134, %57
,double8B

	full_text

double %134
+double8B

	full_text


double %57
Ügetelementptr8Bs
q
	full_textd
b
`%143 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %39, i64 %45, i64 0, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %39
%i648B

	full_text
	
i64 %45
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
`%144 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %39, i64 %45, i64 1, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %39
%i648B

	full_text
	
i64 %45
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
`%145 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %39, i64 %45, i64 2, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %39
%i648B

	full_text
	
i64 %45
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %145, align 8, !tbaa !8
.double*8B

	full_text

double* %145
Ügetelementptr8Bs
q
	full_textd
b
`%146 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %39, i64 %45, i64 3, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %39
%i648B

	full_text
	
i64 %45
Pstore8BE
C
	full_text6
4
2store double %138, double* %146, align 8, !tbaa !8
,double8B

	full_text

double %138
.double*8B

	full_text

double* %146
Ügetelementptr8Bs
q
	full_textd
b
`%147 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %39, i64 %45, i64 4, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %39
%i648B

	full_text
	
i64 %45
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %147, align 8, !tbaa !8
.double*8B

	full_text

double* %147
Gfmul8B=
;
	full_text.
,
*%148 = fmul double %49, 0x3FB00AEC33E1F670
+double8B

	full_text


double %49
Gfmul8B=
;
	full_text.
,
*%149 = fmul double %49, 0xBFB89374BC6A7EF8
+double8B

	full_text


double %49
8fmul8B.
,
	full_text

%150 = fmul double %55, %55
+double8B

	full_text


double %55
+double8B

	full_text


double %55
:fmul8B0
.
	full_text!

%151 = fmul double %149, %150
,double8B

	full_text

double %149
,double8B

	full_text

double %150
Cfsub8B9
7
	full_text*
(
&%152 = fsub double -0.000000e+00, %151
,double8B

	full_text

double %151
mcall8Bc
a
	full_textT
R
P%153 = tail call double @llvm.fmuladd.f64(double %148, double %106, double %152)
,double8B

	full_text

double %148
,double8B

	full_text

double %106
,double8B

	full_text

double %152
8fmul8B.
,
	full_text

%154 = fmul double %57, %57
+double8B

	full_text


double %57
+double8B

	full_text


double %57
Cfsub8B9
7
	full_text*
(
&%155 = fsub double -0.000000e+00, %149
,double8B

	full_text

double %149
mcall8Bc
a
	full_textT
R
P%156 = tail call double @llvm.fmuladd.f64(double %155, double %154, double %153)
,double8B

	full_text

double %155
,double8B

	full_text

double %154
,double8B

	full_text

double %153
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
O%159 = tail call double @llvm.fmuladd.f64(double %158, double %59, double %156)
,double8B

	full_text

double %158
+double8B

	full_text


double %59
,double8B

	full_text

double %156
Ügetelementptr8Bs
q
	full_textd
b
`%160 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %39, i64 %45, i64 0, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %39
%i648B

	full_text
	
i64 %45
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
*%161 = fmul double %48, 0xBFB00AEC33E1F670
+double8B

	full_text


double %48
9fmul8B/
-
	full_text 

%162 = fmul double %161, %53
,double8B

	full_text

double %161
+double8B

	full_text


double %53
Ügetelementptr8Bs
q
	full_textd
b
`%163 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %39, i64 %45, i64 1, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %39
%i648B

	full_text
	
i64 %45
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
*%164 = fmul double %48, 0xBFB89374BC6A7EF8
+double8B

	full_text


double %48
9fmul8B/
-
	full_text 

%165 = fmul double %164, %55
,double8B

	full_text

double %164
+double8B

	full_text


double %55
Ügetelementptr8Bs
q
	full_textd
b
`%166 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %39, i64 %45, i64 2, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %39
%i648B

	full_text
	
i64 %45
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

%167 = fmul double %164, %57
,double8B

	full_text

double %164
+double8B

	full_text


double %57
Ügetelementptr8Bs
q
	full_textd
b
`%168 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %39, i64 %45, i64 3, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %39
%i648B

	full_text
	
i64 %45
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
Ügetelementptr8Bs
q
	full_textd
b
`%170 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %39, i64 %45, i64 4, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %39
%i648B

	full_text
	
i64 %45
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
i32 %7
$i328B

	full_text


i32 %6
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


double* %3
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
#i648B

	full_text	

i64 3
5double8B'
%
	full_text

double -0.000000e+00
4double8B&
$
	full_text

double 4.000000e-01
:double8B,
*
	full_text

double 0x3FC1111111111111
4double8B&
$
	full_text

double 1.000000e-01
#i328B

	full_text	

i32 1
:double8B,
*
	full_text

double 0x3FB00AEC33E1F670
#i648B

	full_text	

i64 2
4double8B&
$
	full_text

double 8.000000e-01
4double8B&
$
	full_text

double 0.000000e+00
4double8B&
$
	full_text

double 1.400000e+00
$i648B

	full_text


i64 32
$i328B

	full_text


i32 -1
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
#i328B

	full_text	

i32 2
%i328B

	full_text
	
i32 325
:double8B,
*
	full_text

double 0xBFC1111111111111
4double8B&
$
	full_text

double 1.600000e+00
#i648B

	full_text	

i64 0
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
double 1.000000e+00
:double8B,
*
	full_text

double 0xBFB89374BC6A7EF8
:double8B,
*
	full_text

double 0xBFB00AEC33E1F670
#i328B

	full_text	

i32 0
#i648B

	full_text	

i64 4
$i328B

	full_text


i32 -2        	
 		                       !! "" #$ ## %& %' %% () (( *+ *, ** -. -- /0 // 12 11 34 33 56 55 78 77 9: 99 ;< ;; => == ?@ ?? AB AA CD CC EF EG EH EI EE JK JJ LM LN LL OP OQ OO RS RT RU RV RR WX WW YZ Y[ Y\ Y] YY ^_ ^^ `a `b `c `d `` ef ee gh gi gj gk gg lm ll no np nq nr nn st ss uv uw uu xy xx z{ z| zz }~ }} Ä 	Å  Ç
É ÇÇ ÑÖ Ñ
Ü ÑÑ á
à áá âä â
ã ââ å
ç åå éè é
ê éé ëí ë
ì ëë î
ï îî ñó ñ
ò ñ
ô ñ
ö ññ õú õõ ùû ù
ü ùù †° †
¢ †† £§ £
• ££ ¶ß ¶
® ¶¶ ©™ ©© ´¨ ´
≠ ´´ ÆØ Æ
∞ ÆÆ ±≤ ±
≥ ±± ¥µ ¥¥ ∂∑ ∂
∏ ∂∂ π∫ π
ª ππ ºΩ º
æ ºº ø¿ øø ¡¬ ¡
√ ¡¡ ƒ≈ ƒ
∆ ƒƒ «» «
… ««  
À    ÃÕ Ã
Œ ÃÃ œ– œ
— œœ “
” ““ ‘’ ‘
÷ ‘‘ ◊ÿ ◊
Ÿ ◊◊ ⁄€ ⁄
‹ ⁄⁄ ›ﬁ ›
ﬂ ›› ‡· ‡
‚ ‡‡ „‰ „
Â „„ ÊÁ Ê
Ë ÊÊ ÈÍ È
Î ÈÈ Ï
Ì ÏÏ ÓÔ Ó
 ÓÓ Ò
Ú ÒÒ ÛÙ Û
ı ÛÛ ˆ˜ ˆ
¯ ˆˆ ˘
˙ ˘˘ ˚¸ ˚
˝ ˚˚ ˛ˇ ˛
Ä ˛˛ ÅÇ Å
É ÅÅ ÑÖ Ñ
Ü ÑÑ áà á
â áá ä
ã ää åç å
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
µ ≥
∂ ≥≥ ∑∏ ∑∑ π
∫ ππ ªº ª
Ω ª
æ ªª ø¿ ø
¡ øø ¬√ ¬
ƒ ¬¬ ≈∆ ≈≈ «» «
… ««  À  
Ã    ÕŒ Õ
œ ÕÕ –— –– “” “
‘ ““ ’÷ ’
◊ ’’ ÿŸ ÿ
⁄ ÿÿ €‹ €€ ›ﬁ ›
ﬂ ›› ‡· ‡
‚ ‡‡ „‰ „
Â „„ Ê
Á ÊÊ ËÈ Ë
Í ËË Î
Ï ÎÎ ÌÓ Ì
Ô ÌÌ 
Ò  ÚÛ Ú
Ù ÚÚ ı
ˆ ıı ˜¯ ˜
˘ ˜˜ ˙
˚ ˙˙ ¸˝ ¸¸ ˛ˇ ˛
Ä ˛˛ ÅÇ Å
É ÅÅ ÑÖ Ñ
Ü ÑÑ áà áá âä â
ã ââ åç å
é åå èê è
ë èè í
ì íí îï î
ñ îî ó
ò óó ôö ô
õ ôô ú
ù úú ûü ûû †° †
¢ †† £§ £
• ££ ¶ß ¶
® ¶¶ ©™ ©
´ ©© ¨
≠ ¨¨ ÆØ ÆÆ ∞± ∞
≤ ∞∞ ≥¥ ≥
µ ≥≥ ∂∑ ∂
∏ ∂∂ π
∫ ππ ªº ª
Ω ªª æ
ø ææ ¿¡ ¿
¬ ¿¿ √ƒ √
≈ √√ ∆« ∆
» ∆∆ …  …
À …… Ã
Õ ÃÃ Œœ Œ
– ŒŒ —
“ —— ”‘ ”
’ ”” ÷◊ ÷
ÿ ÷÷ Ÿ⁄ Ÿ
€ ŸŸ ‹
› ‹‹ ﬁﬂ ﬁﬁ ‡· ‡‡ ‚„ ‚
‰ ‚‚ ÂÊ Â
Á ÂÂ Ë
È ËË ÍÎ Í
Ï Í
Ì ÍÍ ÓÔ Ó
 ÓÓ Ò
Ú ÒÒ ÛÙ Û
ı Û
ˆ ÛÛ ˜¯ ˜˜ ˘
˙ ˘˘ ˚¸ ˚
˝ ˚
˛ ˚˚ ˇÄ ˇ
Å ˇˇ ÇÉ Ç
Ñ ÇÇ ÖÜ ÖÖ áà á
â áá äã ä
å ää çé ç
è çç êë êê íì í
î íí ïñ ï
ó ïï òô ò
ö òò õú õ
ù õõ ûü û
† ûû °¢ °
£ °° §• §§ ¶ß ¶
® ¶¶ ©™ ©
´ ©© ¨Æ 	Ø ∞ ± ≤ "≥ !¥ 1µ 5∂      
     	      $# & '	 )( +% ,* .- 0/ 21 4/ 65 8 :9 < >= @ BA D  F; G? HC IE KJ MJ NJ PL Q" S; T? UC VR X" Z; [? \C ]Y _" a; b? cC d` f" h; i? jC kg m" o; p? qC rn t3 vC wu y3 {C |z ~3 ÄC Å É3 ÖC ÜÑ à3 äC ãâ çL è^ ê^ íé ìë ï ó; ò? ôC öñ úõ ûî ü3 °C ¢ù §† •^ ßW ®¶ ™3 ¨C ≠© Ø´ ∞J ≤e ≥± µ3 ∑C ∏¥ ∫∂ ªJ Ωl æº ¿3 ¬C √ø ≈¡ ∆3 »C …« À^ Õe ŒL –Ã —œ ”3 ’C ÷“ ÿ‘ Ÿ3 €C ‹± ﬁ⁄ ﬂJ ·^ ‚3 ‰C Â‡ Á„ Ë3 ÍC ÎÈ Ì3 ÔC Ó Ú^ Ùl ıL ˜Û ¯ˆ ˙3 ¸C ˝˘ ˇ˚ Ä3 ÇC Éº ÖÅ Ü3 àC âá ã3 çC é‡ êå ë3 ìC îí ñ! ò; ô? öC õó ùs üû °ú £† §é ¶¢ ß3 ©C ™• ¨® ≠^ Ø^ ∞ñ ≤Æ ¥L µ± ∂≥ ∏∑ ∫û ºJ Ωπ æ3 ¿C ¡ª √ø ƒÃ ∆L »≈ …3 ÀC Ã« Œ  œÛ —L ”– ‘3 ÷C ◊“ Ÿ’ ⁄‡ ‹3 ﬁC ﬂ€ ·› ‚7 ‰C Â„ Á7 ÈC ÍË Ï7 ÓC ÔÌ Ò7 ÛC ÙÚ ˆ7 ¯C ˘˜ ˚L ˝¸ ˇ^ Ä7 ÇC É˛ ÖÅ ÜJ à7 äC ãá çâ é7 êC ëè ì7 ïC ñî ò7 öC õô ùL üû °e ¢7 §C •† ß£ ®7 ™C ´© ≠J Ø7 ±C ≤Æ ¥∞ µ7 ∑C ∏∂ ∫7 ºC Ωª øû ¡l ¬7 ƒC ≈¿ «√ »7  C À… Õ7 œC –Œ “7 ‘C ’Æ ◊” ÿ7 ⁄C €Ÿ ›O ﬂO ·e „e ‰‡ Ê‚ ÁÂ Èﬁ ÎÆ ÏË Ìl Ôl ‡ ÚÒ ÙÓ ıÍ ˆL ¯˜ ˙˘ ¸s ˝Û ˛7 ÄC Å˚ Éˇ ÑL ÜÖ à^ â7 ãC åá éä èL ëê ìe î7 ñC óí ôï öê úl ù7 üC †õ ¢û £J •7 ßC ®§ ™¶ ´ ≠   ≠¨ ≠ ∑∑ ∏∏ ≠≥ ∏∏ ≥ ∑∑ ù ∏∏ ùÍ ∏∏ ÍÛ ∏∏ Û˚ ∏∏ ˚ª ∏∏ ª ∑∑ ¢ ∏∏ ¢ ∑∑ 	π g
π Ñ
π ¡
π È
π ˚
π Å
π á
π å
π å
π í
π ’
π Ú
π î
π ∂
π √
π …
π Œ
π ”
π ”
π Ÿ
π û∫ î∫ “∫ ˘∫ †∫ π∫ Ë∫ Ò∫ ˘
ª ùª  
ª ∑
º á
Ω Ææ 
ø ﬁ	¿ `	¿ 
¿ ∂
¿ ‘
¿ ⁄
¿ „
¿ „
¿ È
¿ Ó
¿ á
¿  
¿ Ì
¿ è
¿ £
¿ ©
¿ ∞
¿ ∞
¿ ∂
¿ ª
¿ Œ
¿ ï
¡ ¢¬ x¬ Ç¬ á¬ å¬ Ï¬ Ò¬ ä¬ ï¬ Ê¬ Î¬ ¬ ı¬ ˙¬ í¬ ó¬ ú¬ ¨¬ π¬ æ¬ Ã¬ —¬ ‹
√ û
√ €	ƒ 9	ƒ ;	ƒ =	ƒ ?	ƒ A	ƒ C	≈ #	≈ (
∆ û
« ˜
« §» 	… -
  ¸
À ©	Ã R	Ã u	Ã u	Ã z	Ã 
Ã Ñ
Ã â
Ã †
Ã ‘
Ã ˚
Ã ®
Ã „
Ã „
Ã Ë
Ã Ì
Ã Ú
Ã ˜
Ã Å
Ã £
Ã √
Ã ˇ
Õ ¥
Õ ø
Õ ≈
Õ –	Œ 	Œ 	Œ Y	Œ z
Œ †
Œ ´
Œ ´
Œ ∂
Œ ¡
Œ «
Œ ⁄
Œ Å
Œ ø
Œ Ë
Œ Å
Œ â
Œ â
Œ è
Œ î
Œ ô
Œ ©
Œ …
Œ äœ }
– ‡
– ê
— Ö“ 	” n
” â
” «
” Ó
” í
” ®
” ø
”  
” ’
” ›
” ›
” ˜
” ô
” ª
” Ÿ
” ˇ
” ä
” ï
” û
” ¶
” ¶	‘ 	‘ "

x_solve1"
_Z13get_global_idj"
llvm.fmuladd.f64*ã
npb-BT-x_solve1.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02

transfer_bytes
¯¨n

wgsize_log1p
ÜfA
 
transfer_bytes_log1p
ÜfA

wgsize
<

devmap_label
 