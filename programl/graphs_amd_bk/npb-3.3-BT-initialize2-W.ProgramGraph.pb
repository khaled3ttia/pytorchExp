

[external]
KallocaBA
?
	full_text2
0
.%6 = alloca [2 x [3 x [5 x double]]], align 16
NbitcastBC
A
	full_text4
2
0%7 = bitcast [2 x [3 x [5 x double]]]* %6 to i8*
N[2 x [3 x [5 x double]]]*B/
-
	full_text 

[2 x [3 x [5 x double]]]* %6
ZcallBR
P
	full_textC
A
?call void @llvm.lifetime.start.p0i8(i64 240, i8* nonnull %7) #5
"i8*B

	full_text


i8* %7
KcallBC
A
	full_text4
2
0%8 = tail call i64 @_Z13get_global_idj(i32 2) #6
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
1%10 = tail call i64 @_Z13get_global_idj(i32 1) #6
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
1%12 = tail call i64 @_Z13get_global_idj(i32 0) #6
6truncB-
+
	full_text

%13 = trunc i64 %12 to i32
#i64B

	full_text
	
i64 %12
4icmpB,
*
	full_text

%14 = icmp slt i32 %9, %4
"i32B

	full_text


i32 %9
5icmpB-
+
	full_text

%15 = icmp slt i32 %11, %3
#i32B

	full_text
	
i32 %11
/andB(
&
	full_text

%16 = and i1 %14, %15
!i1B

	full_text


i1 %14
!i1B

	full_text


i1 %15
5icmpB-
+
	full_text

%17 = icmp slt i32 %13, %2
#i32B

	full_text
	
i32 %13
/andB(
&
	full_text

%18 = and i1 %16, %17
!i1B

	full_text


i1 %16
!i1B

	full_text


i1 %17
9brB3
1
	full_text$
"
 br i1 %18, label %19, label %176
!i1B

	full_text


i1 %18
<sitofp8B0
.
	full_text!

%20 = sitofp i32 %9 to double
$i328B

	full_text


i32 %9
Ffmul8B<
:
	full_text-
+
)%21 = fmul double %20, 0x3FA642C8590B2164
+double8B

	full_text


double %20
=sitofp8B1
/
	full_text"
 
%22 = sitofp i32 %11 to double
%i328B

	full_text
	
i32 %11
Ffmul8B<
:
	full_text-
+
)%23 = fmul double %22, 0x3FA642C8590B2164
+double8B

	full_text


double %22
=sitofp8B1
/
	full_text"
 
%24 = sitofp i32 %13 to double
%i328B

	full_text
	
i32 %13
—getelementptr8Bƒ
€
	full_texts
q
o%25 = getelementptr inbounds [2 x [3 x [5 x double]]], [2 x [3 x [5 x double]]]* %6, i64 0, i64 0, i64 0, i64 0
P[2 x [3 x [5 x double]]]*8B/
-
	full_text 

[2 x [3 x [5 x double]]]* %6
‡call8B}
{
	full_textn
l
jcall void @exact_solution(double 0.000000e+00, double %23, double %21, double* nonnull %25, double* %1) #5
+double8B

	full_text


double %23
+double8B

	full_text


double %21
-double*8B

	full_text

double* %25
—getelementptr8Bƒ
€
	full_texts
q
o%26 = getelementptr inbounds [2 x [3 x [5 x double]]], [2 x [3 x [5 x double]]]* %6, i64 0, i64 1, i64 0, i64 0
P[2 x [3 x [5 x double]]]*8B/
-
	full_text 

[2 x [3 x [5 x double]]]* %6
call8Bu
s
	full_textf
d
bcall void @exact_solution(double 1.000000e+00, double %23, double %21, double* %26, double* %1) #5
+double8B

	full_text


double %23
+double8B

	full_text


double %21
-double*8B

	full_text

double* %26
Wbitcast8BJ
H
	full_text;
9
7%27 = bitcast double* %0 to [25 x [25 x [5 x double]]]*
Ffmul8B<
:
	full_text-
+
)%28 = fmul double %24, 0x3FA642C8590B2164
+double8B

	full_text


double %24
—getelementptr8Bƒ
€
	full_texts
q
o%29 = getelementptr inbounds [2 x [3 x [5 x double]]], [2 x [3 x [5 x double]]]* %6, i64 0, i64 0, i64 1, i64 0
P[2 x [3 x [5 x double]]]*8B/
-
	full_text 

[2 x [3 x [5 x double]]]* %6
call8Bu
s
	full_textf
d
bcall void @exact_solution(double %28, double 0.000000e+00, double %21, double* %29, double* %1) #5
+double8B

	full_text


double %28
+double8B

	full_text


double %21
-double*8B

	full_text

double* %29
—getelementptr8Bƒ
€
	full_texts
q
o%30 = getelementptr inbounds [2 x [3 x [5 x double]]], [2 x [3 x [5 x double]]]* %6, i64 0, i64 1, i64 1, i64 0
P[2 x [3 x [5 x double]]]*8B/
-
	full_text 

[2 x [3 x [5 x double]]]* %6
call8Bu
s
	full_textf
d
bcall void @exact_solution(double %28, double 1.000000e+00, double %21, double* %30, double* %1) #5
+double8B

	full_text


double %28
+double8B

	full_text


double %21
-double*8B

	full_text

double* %30
—getelementptr8Bƒ
€
	full_texts
q
o%31 = getelementptr inbounds [2 x [3 x [5 x double]]], [2 x [3 x [5 x double]]]* %6, i64 0, i64 0, i64 2, i64 0
P[2 x [3 x [5 x double]]]*8B/
-
	full_text 

[2 x [3 x [5 x double]]]* %6
call8Bu
s
	full_textf
d
bcall void @exact_solution(double %28, double %23, double 0.000000e+00, double* %31, double* %1) #5
+double8B

	full_text


double %28
+double8B

	full_text


double %23
-double*8B

	full_text

double* %31
—getelementptr8Bƒ
€
	full_texts
q
o%32 = getelementptr inbounds [2 x [3 x [5 x double]]], [2 x [3 x [5 x double]]]* %6, i64 0, i64 1, i64 2, i64 0
P[2 x [3 x [5 x double]]]*8B/
-
	full_text 

[2 x [3 x [5 x double]]]* %6
call8Bu
s
	full_textf
d
bcall void @exact_solution(double %28, double %23, double 1.000000e+00, double* %32, double* %1) #5
+double8B

	full_text


double %28
+double8B

	full_text


double %23
-double*8B

	full_text

double* %32
@fsub8B6
4
	full_text'
%
#%33 = fsub double 1.000000e+00, %28
+double8B

	full_text


double %28
@fsub8B6
4
	full_text'
%
#%34 = fsub double 1.000000e+00, %23
+double8B

	full_text


double %23
@fsub8B6
4
	full_text'
%
#%35 = fsub double 1.000000e+00, %21
+double8B

	full_text


double %21
0shl8B'
%
	full_text

%36 = shl i64 %8, 32
$i648B

	full_text


i64 %8
9ashr8B/
-
	full_text 

%37 = ashr exact i64 %36, 32
%i648B

	full_text
	
i64 %36
1shl8B(
&
	full_text

%38 = shl i64 %10, 32
%i648B

	full_text
	
i64 %10
9ashr8B/
-
	full_text 

%39 = ashr exact i64 %38, 32
%i648B

	full_text
	
i64 %38
1shl8B(
&
	full_text

%40 = shl i64 %12, 32
%i648B

	full_text
	
i64 %12
9ashr8B/
-
	full_text 

%41 = ashr exact i64 %40, 32
%i648B

	full_text
	
i64 %40
Nload8BD
B
	full_text5
3
1%42 = load double, double* %26, align 8, !tbaa !8
-double*8B

	full_text

double* %26
Oload8BE
C
	full_text6
4
2%43 = load double, double* %25, align 16, !tbaa !8
-double*8B

	full_text

double* %25
7fmul8B-
+
	full_text

%44 = fmul double %33, %43
+double8B

	full_text


double %33
+double8B

	full_text


double %43
dcall8BZ
X
	full_textK
I
G%45 = call double @llvm.fmuladd.f64(double %28, double %42, double %44)
+double8B

	full_text


double %28
+double8B

	full_text


double %42
+double8B

	full_text


double %44
Nload8BD
B
	full_text5
3
1%46 = load double, double* %30, align 8, !tbaa !8
-double*8B

	full_text

double* %30
Nload8BD
B
	full_text5
3
1%47 = load double, double* %29, align 8, !tbaa !8
-double*8B

	full_text

double* %29
7fmul8B-
+
	full_text

%48 = fmul double %34, %47
+double8B

	full_text


double %34
+double8B

	full_text


double %47
dcall8BZ
X
	full_textK
I
G%49 = call double @llvm.fmuladd.f64(double %23, double %46, double %48)
+double8B

	full_text


double %23
+double8B

	full_text


double %46
+double8B

	full_text


double %48
Nload8BD
B
	full_text5
3
1%50 = load double, double* %32, align 8, !tbaa !8
-double*8B

	full_text

double* %32
Oload8BE
C
	full_text6
4
2%51 = load double, double* %31, align 16, !tbaa !8
-double*8B

	full_text

double* %31
7fmul8B-
+
	full_text

%52 = fmul double %35, %51
+double8B

	full_text


double %35
+double8B

	full_text


double %51
dcall8BZ
X
	full_textK
I
G%53 = call double @llvm.fmuladd.f64(double %21, double %50, double %52)
+double8B

	full_text


double %21
+double8B

	full_text


double %50
+double8B

	full_text


double %52
7fadd8B-
+
	full_text

%54 = fadd double %45, %49
+double8B

	full_text


double %45
+double8B

	full_text


double %49
7fadd8B-
+
	full_text

%55 = fadd double %54, %53
+double8B

	full_text


double %54
+double8B

	full_text


double %53
Afsub8B7
5
	full_text(
&
$%56 = fsub double -0.000000e+00, %45
+double8B

	full_text


double %45
dcall8BZ
X
	full_textK
I
G%57 = call double @llvm.fmuladd.f64(double %56, double %49, double %55)
+double8B

	full_text


double %56
+double8B

	full_text


double %49
+double8B

	full_text


double %55
dcall8BZ
X
	full_textK
I
G%58 = call double @llvm.fmuladd.f64(double %56, double %53, double %57)
+double8B

	full_text


double %56
+double8B

	full_text


double %53
+double8B

	full_text


double %57
Afsub8B7
5
	full_text(
&
$%59 = fsub double -0.000000e+00, %49
+double8B

	full_text


double %49
dcall8BZ
X
	full_textK
I
G%60 = call double @llvm.fmuladd.f64(double %59, double %53, double %58)
+double8B

	full_text


double %59
+double8B

	full_text


double %53
+double8B

	full_text


double %58
7fmul8B-
+
	full_text

%61 = fmul double %45, %49
+double8B

	full_text


double %45
+double8B

	full_text


double %49
dcall8BZ
X
	full_textK
I
G%62 = call double @llvm.fmuladd.f64(double %61, double %53, double %60)
+double8B

	full_text


double %61
+double8B

	full_text


double %53
+double8B

	full_text


double %60
¢getelementptr8B
‹
	full_text~
|
z%63 = getelementptr inbounds [25 x [25 x [5 x double]]], [25 x [25 x [5 x double]]]* %27, i64 %37, i64 %39, i64 %41, i64 0
U[25 x [25 x [5 x double]]]*8B2
0
	full_text#
!
[25 x [25 x [5 x double]]]* %27
%i648B

	full_text
	
i64 %37
%i648B

	full_text
	
i64 %39
%i648B

	full_text
	
i64 %41
Nstore8BC
A
	full_text4
2
0store double %62, double* %63, align 8, !tbaa !8
+double8B

	full_text


double %62
-double*8B

	full_text

double* %63
—getelementptr8Bƒ
€
	full_texts
q
o%64 = getelementptr inbounds [2 x [3 x [5 x double]]], [2 x [3 x [5 x double]]]* %6, i64 0, i64 1, i64 0, i64 1
P[2 x [3 x [5 x double]]]*8B/
-
	full_text 

[2 x [3 x [5 x double]]]* %6
Nload8BD
B
	full_text5
3
1%65 = load double, double* %64, align 8, !tbaa !8
-double*8B

	full_text

double* %64
—getelementptr8Bƒ
€
	full_texts
q
o%66 = getelementptr inbounds [2 x [3 x [5 x double]]], [2 x [3 x [5 x double]]]* %6, i64 0, i64 0, i64 0, i64 1
P[2 x [3 x [5 x double]]]*8B/
-
	full_text 

[2 x [3 x [5 x double]]]* %6
Nload8BD
B
	full_text5
3
1%67 = load double, double* %66, align 8, !tbaa !8
-double*8B

	full_text

double* %66
7fmul8B-
+
	full_text

%68 = fmul double %33, %67
+double8B

	full_text


double %33
+double8B

	full_text


double %67
dcall8BZ
X
	full_textK
I
G%69 = call double @llvm.fmuladd.f64(double %28, double %65, double %68)
+double8B

	full_text


double %28
+double8B

	full_text


double %65
+double8B

	full_text


double %68
—getelementptr8Bƒ
€
	full_texts
q
o%70 = getelementptr inbounds [2 x [3 x [5 x double]]], [2 x [3 x [5 x double]]]* %6, i64 0, i64 1, i64 1, i64 1
P[2 x [3 x [5 x double]]]*8B/
-
	full_text 

[2 x [3 x [5 x double]]]* %6
Nload8BD
B
	full_text5
3
1%71 = load double, double* %70, align 8, !tbaa !8
-double*8B

	full_text

double* %70
—getelementptr8Bƒ
€
	full_texts
q
o%72 = getelementptr inbounds [2 x [3 x [5 x double]]], [2 x [3 x [5 x double]]]* %6, i64 0, i64 0, i64 1, i64 1
P[2 x [3 x [5 x double]]]*8B/
-
	full_text 

[2 x [3 x [5 x double]]]* %6
Nload8BD
B
	full_text5
3
1%73 = load double, double* %72, align 8, !tbaa !8
-double*8B

	full_text

double* %72
7fmul8B-
+
	full_text

%74 = fmul double %34, %73
+double8B

	full_text


double %34
+double8B

	full_text


double %73
dcall8BZ
X
	full_textK
I
G%75 = call double @llvm.fmuladd.f64(double %23, double %71, double %74)
+double8B

	full_text


double %23
+double8B

	full_text


double %71
+double8B

	full_text


double %74
—getelementptr8Bƒ
€
	full_texts
q
o%76 = getelementptr inbounds [2 x [3 x [5 x double]]], [2 x [3 x [5 x double]]]* %6, i64 0, i64 1, i64 2, i64 1
P[2 x [3 x [5 x double]]]*8B/
-
	full_text 

[2 x [3 x [5 x double]]]* %6
Nload8BD
B
	full_text5
3
1%77 = load double, double* %76, align 8, !tbaa !8
-double*8B

	full_text

double* %76
—getelementptr8Bƒ
€
	full_texts
q
o%78 = getelementptr inbounds [2 x [3 x [5 x double]]], [2 x [3 x [5 x double]]]* %6, i64 0, i64 0, i64 2, i64 1
P[2 x [3 x [5 x double]]]*8B/
-
	full_text 

[2 x [3 x [5 x double]]]* %6
Nload8BD
B
	full_text5
3
1%79 = load double, double* %78, align 8, !tbaa !8
-double*8B

	full_text

double* %78
7fmul8B-
+
	full_text

%80 = fmul double %35, %79
+double8B

	full_text


double %35
+double8B

	full_text


double %79
dcall8BZ
X
	full_textK
I
G%81 = call double @llvm.fmuladd.f64(double %21, double %77, double %80)
+double8B

	full_text


double %21
+double8B

	full_text


double %77
+double8B

	full_text


double %80
7fadd8B-
+
	full_text

%82 = fadd double %69, %75
+double8B

	full_text


double %69
+double8B

	full_text


double %75
7fadd8B-
+
	full_text

%83 = fadd double %82, %81
+double8B

	full_text


double %82
+double8B

	full_text


double %81
Afsub8B7
5
	full_text(
&
$%84 = fsub double -0.000000e+00, %69
+double8B

	full_text


double %69
dcall8BZ
X
	full_textK
I
G%85 = call double @llvm.fmuladd.f64(double %84, double %75, double %83)
+double8B

	full_text


double %84
+double8B

	full_text


double %75
+double8B

	full_text


double %83
dcall8BZ
X
	full_textK
I
G%86 = call double @llvm.fmuladd.f64(double %84, double %81, double %85)
+double8B

	full_text


double %84
+double8B

	full_text


double %81
+double8B

	full_text


double %85
Afsub8B7
5
	full_text(
&
$%87 = fsub double -0.000000e+00, %75
+double8B

	full_text


double %75
dcall8BZ
X
	full_textK
I
G%88 = call double @llvm.fmuladd.f64(double %87, double %81, double %86)
+double8B

	full_text


double %87
+double8B

	full_text


double %81
+double8B

	full_text


double %86
7fmul8B-
+
	full_text

%89 = fmul double %69, %75
+double8B

	full_text


double %69
+double8B

	full_text


double %75
dcall8BZ
X
	full_textK
I
G%90 = call double @llvm.fmuladd.f64(double %89, double %81, double %88)
+double8B

	full_text


double %89
+double8B

	full_text


double %81
+double8B

	full_text


double %88
¢getelementptr8B
‹
	full_text~
|
z%91 = getelementptr inbounds [25 x [25 x [5 x double]]], [25 x [25 x [5 x double]]]* %27, i64 %37, i64 %39, i64 %41, i64 1
U[25 x [25 x [5 x double]]]*8B2
0
	full_text#
!
[25 x [25 x [5 x double]]]* %27
%i648B

	full_text
	
i64 %37
%i648B

	full_text
	
i64 %39
%i648B

	full_text
	
i64 %41
Nstore8BC
A
	full_text4
2
0store double %90, double* %91, align 8, !tbaa !8
+double8B

	full_text


double %90
-double*8B

	full_text

double* %91
—getelementptr8Bƒ
€
	full_texts
q
o%92 = getelementptr inbounds [2 x [3 x [5 x double]]], [2 x [3 x [5 x double]]]* %6, i64 0, i64 1, i64 0, i64 2
P[2 x [3 x [5 x double]]]*8B/
-
	full_text 

[2 x [3 x [5 x double]]]* %6
Nload8BD
B
	full_text5
3
1%93 = load double, double* %92, align 8, !tbaa !8
-double*8B

	full_text

double* %92
—getelementptr8Bƒ
€
	full_texts
q
o%94 = getelementptr inbounds [2 x [3 x [5 x double]]], [2 x [3 x [5 x double]]]* %6, i64 0, i64 0, i64 0, i64 2
P[2 x [3 x [5 x double]]]*8B/
-
	full_text 

[2 x [3 x [5 x double]]]* %6
Oload8BE
C
	full_text6
4
2%95 = load double, double* %94, align 16, !tbaa !8
-double*8B

	full_text

double* %94
7fmul8B-
+
	full_text

%96 = fmul double %33, %95
+double8B

	full_text


double %33
+double8B

	full_text


double %95
dcall8BZ
X
	full_textK
I
G%97 = call double @llvm.fmuladd.f64(double %28, double %93, double %96)
+double8B

	full_text


double %28
+double8B

	full_text


double %93
+double8B

	full_text


double %96
—getelementptr8Bƒ
€
	full_texts
q
o%98 = getelementptr inbounds [2 x [3 x [5 x double]]], [2 x [3 x [5 x double]]]* %6, i64 0, i64 1, i64 1, i64 2
P[2 x [3 x [5 x double]]]*8B/
-
	full_text 

[2 x [3 x [5 x double]]]* %6
Nload8BD
B
	full_text5
3
1%99 = load double, double* %98, align 8, !tbaa !8
-double*8B

	full_text

double* %98
˜getelementptr8B„

	full_textt
r
p%100 = getelementptr inbounds [2 x [3 x [5 x double]]], [2 x [3 x [5 x double]]]* %6, i64 0, i64 0, i64 1, i64 2
P[2 x [3 x [5 x double]]]*8B/
-
	full_text 

[2 x [3 x [5 x double]]]* %6
Pload8BF
D
	full_text7
5
3%101 = load double, double* %100, align 8, !tbaa !8
.double*8B

	full_text

double* %100
9fmul8B/
-
	full_text 

%102 = fmul double %34, %101
+double8B

	full_text


double %34
,double8B

	full_text

double %101
fcall8B\
Z
	full_textM
K
I%103 = call double @llvm.fmuladd.f64(double %23, double %99, double %102)
+double8B

	full_text


double %23
+double8B

	full_text


double %99
,double8B

	full_text

double %102
˜getelementptr8B„

	full_textt
r
p%104 = getelementptr inbounds [2 x [3 x [5 x double]]], [2 x [3 x [5 x double]]]* %6, i64 0, i64 1, i64 2, i64 2
P[2 x [3 x [5 x double]]]*8B/
-
	full_text 

[2 x [3 x [5 x double]]]* %6
Pload8BF
D
	full_text7
5
3%105 = load double, double* %104, align 8, !tbaa !8
.double*8B

	full_text

double* %104
˜getelementptr8B„

	full_textt
r
p%106 = getelementptr inbounds [2 x [3 x [5 x double]]], [2 x [3 x [5 x double]]]* %6, i64 0, i64 0, i64 2, i64 2
P[2 x [3 x [5 x double]]]*8B/
-
	full_text 

[2 x [3 x [5 x double]]]* %6
Qload8BG
E
	full_text8
6
4%107 = load double, double* %106, align 16, !tbaa !8
.double*8B

	full_text

double* %106
9fmul8B/
-
	full_text 

%108 = fmul double %35, %107
+double8B

	full_text


double %35
,double8B

	full_text

double %107
gcall8B]
[
	full_textN
L
J%109 = call double @llvm.fmuladd.f64(double %21, double %105, double %108)
+double8B

	full_text


double %21
,double8B

	full_text

double %105
,double8B

	full_text

double %108
9fadd8B/
-
	full_text 

%110 = fadd double %97, %103
+double8B

	full_text


double %97
,double8B

	full_text

double %103
:fadd8B0
.
	full_text!

%111 = fadd double %110, %109
,double8B

	full_text

double %110
,double8B

	full_text

double %109
Bfsub8B8
6
	full_text)
'
%%112 = fsub double -0.000000e+00, %97
+double8B

	full_text


double %97
hcall8B^
\
	full_textO
M
K%113 = call double @llvm.fmuladd.f64(double %112, double %103, double %111)
,double8B

	full_text

double %112
,double8B

	full_text

double %103
,double8B

	full_text

double %111
hcall8B^
\
	full_textO
M
K%114 = call double @llvm.fmuladd.f64(double %112, double %109, double %113)
,double8B

	full_text

double %112
,double8B

	full_text

double %109
,double8B

	full_text

double %113
Cfsub8B9
7
	full_text*
(
&%115 = fsub double -0.000000e+00, %103
,double8B

	full_text

double %103
hcall8B^
\
	full_textO
M
K%116 = call double @llvm.fmuladd.f64(double %115, double %109, double %114)
,double8B

	full_text

double %115
,double8B

	full_text

double %109
,double8B

	full_text

double %114
9fmul8B/
-
	full_text 

%117 = fmul double %97, %103
+double8B

	full_text


double %97
,double8B

	full_text

double %103
hcall8B^
\
	full_textO
M
K%118 = call double @llvm.fmuladd.f64(double %117, double %109, double %116)
,double8B

	full_text

double %117
,double8B

	full_text

double %109
,double8B

	full_text

double %116
£getelementptr8B
Œ
	full_text
}
{%119 = getelementptr inbounds [25 x [25 x [5 x double]]], [25 x [25 x [5 x double]]]* %27, i64 %37, i64 %39, i64 %41, i64 2
U[25 x [25 x [5 x double]]]*8B2
0
	full_text#
!
[25 x [25 x [5 x double]]]* %27
%i648B

	full_text
	
i64 %37
%i648B

	full_text
	
i64 %39
%i648B

	full_text
	
i64 %41
Pstore8BE
C
	full_text6
4
2store double %118, double* %119, align 8, !tbaa !8
,double8B

	full_text

double %118
.double*8B

	full_text

double* %119
˜getelementptr8B„

	full_textt
r
p%120 = getelementptr inbounds [2 x [3 x [5 x double]]], [2 x [3 x [5 x double]]]* %6, i64 0, i64 1, i64 0, i64 3
P[2 x [3 x [5 x double]]]*8B/
-
	full_text 

[2 x [3 x [5 x double]]]* %6
Pload8BF
D
	full_text7
5
3%121 = load double, double* %120, align 8, !tbaa !8
.double*8B

	full_text

double* %120
˜getelementptr8B„

	full_textt
r
p%122 = getelementptr inbounds [2 x [3 x [5 x double]]], [2 x [3 x [5 x double]]]* %6, i64 0, i64 0, i64 0, i64 3
P[2 x [3 x [5 x double]]]*8B/
-
	full_text 

[2 x [3 x [5 x double]]]* %6
Pload8BF
D
	full_text7
5
3%123 = load double, double* %122, align 8, !tbaa !8
.double*8B

	full_text

double* %122
9fmul8B/
-
	full_text 

%124 = fmul double %33, %123
+double8B

	full_text


double %33
,double8B

	full_text

double %123
gcall8B]
[
	full_textN
L
J%125 = call double @llvm.fmuladd.f64(double %28, double %121, double %124)
+double8B

	full_text


double %28
,double8B

	full_text

double %121
,double8B

	full_text

double %124
˜getelementptr8B„

	full_textt
r
p%126 = getelementptr inbounds [2 x [3 x [5 x double]]], [2 x [3 x [5 x double]]]* %6, i64 0, i64 1, i64 1, i64 3
P[2 x [3 x [5 x double]]]*8B/
-
	full_text 

[2 x [3 x [5 x double]]]* %6
Pload8BF
D
	full_text7
5
3%127 = load double, double* %126, align 8, !tbaa !8
.double*8B

	full_text

double* %126
˜getelementptr8B„

	full_textt
r
p%128 = getelementptr inbounds [2 x [3 x [5 x double]]], [2 x [3 x [5 x double]]]* %6, i64 0, i64 0, i64 1, i64 3
P[2 x [3 x [5 x double]]]*8B/
-
	full_text 

[2 x [3 x [5 x double]]]* %6
Pload8BF
D
	full_text7
5
3%129 = load double, double* %128, align 8, !tbaa !8
.double*8B

	full_text

double* %128
9fmul8B/
-
	full_text 

%130 = fmul double %34, %129
+double8B

	full_text


double %34
,double8B

	full_text

double %129
gcall8B]
[
	full_textN
L
J%131 = call double @llvm.fmuladd.f64(double %23, double %127, double %130)
+double8B

	full_text


double %23
,double8B

	full_text

double %127
,double8B

	full_text

double %130
˜getelementptr8B„

	full_textt
r
p%132 = getelementptr inbounds [2 x [3 x [5 x double]]], [2 x [3 x [5 x double]]]* %6, i64 0, i64 1, i64 2, i64 3
P[2 x [3 x [5 x double]]]*8B/
-
	full_text 

[2 x [3 x [5 x double]]]* %6
Pload8BF
D
	full_text7
5
3%133 = load double, double* %132, align 8, !tbaa !8
.double*8B

	full_text

double* %132
˜getelementptr8B„

	full_textt
r
p%134 = getelementptr inbounds [2 x [3 x [5 x double]]], [2 x [3 x [5 x double]]]* %6, i64 0, i64 0, i64 2, i64 3
P[2 x [3 x [5 x double]]]*8B/
-
	full_text 

[2 x [3 x [5 x double]]]* %6
Pload8BF
D
	full_text7
5
3%135 = load double, double* %134, align 8, !tbaa !8
.double*8B

	full_text

double* %134
9fmul8B/
-
	full_text 

%136 = fmul double %35, %135
+double8B

	full_text


double %35
,double8B

	full_text

double %135
gcall8B]
[
	full_textN
L
J%137 = call double @llvm.fmuladd.f64(double %21, double %133, double %136)
+double8B

	full_text


double %21
,double8B

	full_text

double %133
,double8B

	full_text

double %136
:fadd8B0
.
	full_text!

%138 = fadd double %125, %131
,double8B

	full_text

double %125
,double8B

	full_text

double %131
:fadd8B0
.
	full_text!

%139 = fadd double %138, %137
,double8B

	full_text

double %138
,double8B

	full_text

double %137
Cfsub8B9
7
	full_text*
(
&%140 = fsub double -0.000000e+00, %125
,double8B

	full_text

double %125
hcall8B^
\
	full_textO
M
K%141 = call double @llvm.fmuladd.f64(double %140, double %131, double %139)
,double8B

	full_text

double %140
,double8B

	full_text

double %131
,double8B

	full_text

double %139
hcall8B^
\
	full_textO
M
K%142 = call double @llvm.fmuladd.f64(double %140, double %137, double %141)
,double8B

	full_text

double %140
,double8B

	full_text

double %137
,double8B

	full_text

double %141
Cfsub8B9
7
	full_text*
(
&%143 = fsub double -0.000000e+00, %131
,double8B

	full_text

double %131
hcall8B^
\
	full_textO
M
K%144 = call double @llvm.fmuladd.f64(double %143, double %137, double %142)
,double8B

	full_text

double %143
,double8B

	full_text

double %137
,double8B

	full_text

double %142
:fmul8B0
.
	full_text!

%145 = fmul double %125, %131
,double8B

	full_text

double %125
,double8B

	full_text

double %131
hcall8B^
\
	full_textO
M
K%146 = call double @llvm.fmuladd.f64(double %145, double %137, double %144)
,double8B

	full_text

double %145
,double8B

	full_text

double %137
,double8B

	full_text

double %144
£getelementptr8B
Œ
	full_text
}
{%147 = getelementptr inbounds [25 x [25 x [5 x double]]], [25 x [25 x [5 x double]]]* %27, i64 %37, i64 %39, i64 %41, i64 3
U[25 x [25 x [5 x double]]]*8B2
0
	full_text#
!
[25 x [25 x [5 x double]]]* %27
%i648B

	full_text
	
i64 %37
%i648B

	full_text
	
i64 %39
%i648B

	full_text
	
i64 %41
Pstore8BE
C
	full_text6
4
2store double %146, double* %147, align 8, !tbaa !8
,double8B

	full_text

double %146
.double*8B

	full_text

double* %147
˜getelementptr8B„

	full_textt
r
p%148 = getelementptr inbounds [2 x [3 x [5 x double]]], [2 x [3 x [5 x double]]]* %6, i64 0, i64 1, i64 0, i64 4
P[2 x [3 x [5 x double]]]*8B/
-
	full_text 

[2 x [3 x [5 x double]]]* %6
Pload8BF
D
	full_text7
5
3%149 = load double, double* %148, align 8, !tbaa !8
.double*8B

	full_text

double* %148
˜getelementptr8B„

	full_textt
r
p%150 = getelementptr inbounds [2 x [3 x [5 x double]]], [2 x [3 x [5 x double]]]* %6, i64 0, i64 0, i64 0, i64 4
P[2 x [3 x [5 x double]]]*8B/
-
	full_text 

[2 x [3 x [5 x double]]]* %6
Qload8BG
E
	full_text8
6
4%151 = load double, double* %150, align 16, !tbaa !8
.double*8B

	full_text

double* %150
9fmul8B/
-
	full_text 

%152 = fmul double %33, %151
+double8B

	full_text


double %33
,double8B

	full_text

double %151
gcall8B]
[
	full_textN
L
J%153 = call double @llvm.fmuladd.f64(double %28, double %149, double %152)
+double8B

	full_text


double %28
,double8B

	full_text

double %149
,double8B

	full_text

double %152
˜getelementptr8B„

	full_textt
r
p%154 = getelementptr inbounds [2 x [3 x [5 x double]]], [2 x [3 x [5 x double]]]* %6, i64 0, i64 1, i64 1, i64 4
P[2 x [3 x [5 x double]]]*8B/
-
	full_text 

[2 x [3 x [5 x double]]]* %6
Pload8BF
D
	full_text7
5
3%155 = load double, double* %154, align 8, !tbaa !8
.double*8B

	full_text

double* %154
˜getelementptr8B„

	full_textt
r
p%156 = getelementptr inbounds [2 x [3 x [5 x double]]], [2 x [3 x [5 x double]]]* %6, i64 0, i64 0, i64 1, i64 4
P[2 x [3 x [5 x double]]]*8B/
-
	full_text 

[2 x [3 x [5 x double]]]* %6
Pload8BF
D
	full_text7
5
3%157 = load double, double* %156, align 8, !tbaa !8
.double*8B

	full_text

double* %156
9fmul8B/
-
	full_text 

%158 = fmul double %34, %157
+double8B

	full_text


double %34
,double8B

	full_text

double %157
gcall8B]
[
	full_textN
L
J%159 = call double @llvm.fmuladd.f64(double %23, double %155, double %158)
+double8B

	full_text


double %23
,double8B

	full_text

double %155
,double8B

	full_text

double %158
˜getelementptr8B„

	full_textt
r
p%160 = getelementptr inbounds [2 x [3 x [5 x double]]], [2 x [3 x [5 x double]]]* %6, i64 0, i64 1, i64 2, i64 4
P[2 x [3 x [5 x double]]]*8B/
-
	full_text 

[2 x [3 x [5 x double]]]* %6
Pload8BF
D
	full_text7
5
3%161 = load double, double* %160, align 8, !tbaa !8
.double*8B

	full_text

double* %160
˜getelementptr8B„

	full_textt
r
p%162 = getelementptr inbounds [2 x [3 x [5 x double]]], [2 x [3 x [5 x double]]]* %6, i64 0, i64 0, i64 2, i64 4
P[2 x [3 x [5 x double]]]*8B/
-
	full_text 

[2 x [3 x [5 x double]]]* %6
Qload8BG
E
	full_text8
6
4%163 = load double, double* %162, align 16, !tbaa !8
.double*8B

	full_text

double* %162
9fmul8B/
-
	full_text 

%164 = fmul double %35, %163
+double8B

	full_text


double %35
,double8B

	full_text

double %163
gcall8B]
[
	full_textN
L
J%165 = call double @llvm.fmuladd.f64(double %21, double %161, double %164)
+double8B

	full_text


double %21
,double8B

	full_text

double %161
,double8B

	full_text

double %164
:fadd8B0
.
	full_text!

%166 = fadd double %153, %159
,double8B

	full_text

double %153
,double8B

	full_text

double %159
:fadd8B0
.
	full_text!

%167 = fadd double %166, %165
,double8B

	full_text

double %166
,double8B

	full_text

double %165
Cfsub8B9
7
	full_text*
(
&%168 = fsub double -0.000000e+00, %153
,double8B

	full_text

double %153
hcall8B^
\
	full_textO
M
K%169 = call double @llvm.fmuladd.f64(double %168, double %159, double %167)
,double8B

	full_text

double %168
,double8B

	full_text

double %159
,double8B

	full_text

double %167
hcall8B^
\
	full_textO
M
K%170 = call double @llvm.fmuladd.f64(double %168, double %165, double %169)
,double8B

	full_text

double %168
,double8B

	full_text

double %165
,double8B

	full_text

double %169
Cfsub8B9
7
	full_text*
(
&%171 = fsub double -0.000000e+00, %159
,double8B

	full_text

double %159
hcall8B^
\
	full_textO
M
K%172 = call double @llvm.fmuladd.f64(double %171, double %165, double %170)
,double8B

	full_text

double %171
,double8B

	full_text

double %165
,double8B

	full_text

double %170
:fmul8B0
.
	full_text!

%173 = fmul double %153, %159
,double8B

	full_text

double %153
,double8B

	full_text

double %159
hcall8B^
\
	full_textO
M
K%174 = call double @llvm.fmuladd.f64(double %173, double %165, double %172)
,double8B

	full_text

double %173
,double8B

	full_text

double %165
,double8B

	full_text

double %172
£getelementptr8B
Œ
	full_text
}
{%175 = getelementptr inbounds [25 x [25 x [5 x double]]], [25 x [25 x [5 x double]]]* %27, i64 %37, i64 %39, i64 %41, i64 4
U[25 x [25 x [5 x double]]]*8B2
0
	full_text#
!
[25 x [25 x [5 x double]]]* %27
%i648B

	full_text
	
i64 %37
%i648B

	full_text
	
i64 %39
%i648B

	full_text
	
i64 %41
Pstore8BE
C
	full_text6
4
2store double %174, double* %175, align 8, !tbaa !8
,double8B

	full_text

double %174
.double*8B

	full_text

double* %175
(br8B 

	full_text

br label %176
Zcall8BP
N
	full_textA
?
=call void @llvm.lifetime.end.p0i8(i64 240, i8* nonnull %7) #5
$i8*8B

	full_text


i8* %7
$ret8B

	full_text


ret void
$i328B

	full_text


i32 %4
$i328B

	full_text


i32 %2
,double*8B

	full_text


double* %0
$i328B

	full_text


i32 %3
,double*8B
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
#i328B

	full_text	

i32 2
#i328B

	full_text	

i32 1
:double8B,
*
	full_text

double 0x3FA642C8590B2164
#i648B

	full_text	

i64 2
%i648B

	full_text
	
i64 240
#i328B

	full_text	

i32 0
#i648B

	full_text	

i64 3
#i648B

	full_text	

i64 4
4double8B&
$
	full_text

double 0.000000e+00
#i648B

	full_text	

i64 1
4double8B&
$
	full_text

double 1.000000e+00
$i648B

	full_text


i64 32
#i648B

	full_text	

i64 0
5double8B'
%
	full_text

double -0.000000e+00        		 
 

                      !" !! #$ ## %& %% '( '' )* )+ ), )) -. -- /0 /1 /2 // 33 45 44 67 66 89 8: 8; 88 <= << >? >@ >A >> BC BB DE DF DG DD HI HH JK JL JM JJ NO NN PQ PP RS RR TU TT VW VV XY XX Z[ ZZ \] \\ ^_ ^^ `a `` bc bb de df dd gh gi gj gg kl kk mn mm op oq oo rs rt ru rr vw vv xy xx z{ z| zz }~ } }	€ }} ‚ 
ƒ  „… „
† „„ ‡
ˆ ‡‡ ‰Š ‰
‹ ‰
Œ ‰‰  
 
  ‘
’ ‘‘ “” “
• “
– ““ —˜ —
™ —— š› š
œ š
 šš Ÿ 
  
¡ 
¢  £¤ £
¥ ££ ¦§ ¦¦ ¨© ¨¨ ª« ªª ¬­ ¬¬ ®¯ ®
° ®® ±² ±
³ ±
´ ±± µ¶ µµ ·¸ ·· ¹º ¹¹ »¼ »» ½¾ ½
¿ ½½ ÀÁ À
Â À
Ã ÀÀ ÄÅ ÄÄ ÆÇ ÆÆ ÈÉ ÈÈ ÊË ÊÊ ÌÍ Ì
Î ÌÌ ÏĞ Ï
Ñ Ï
Ò ÏÏ ÓÔ Ó
Õ ÓÓ Ö× Ö
Ø ÖÖ Ù
Ú ÙÙ ÛÜ Û
İ Û
Ş ÛÛ ßà ß
á ß
â ßß ã
ä ãã åæ å
ç å
è åå éê é
ë éé ìí ì
î ì
ï ìì ğñ ğ
ò ğ
ó ğ
ô ğğ õö õ
÷ õõ øù øø úû úú üı üü şÿ şş € €
‚ €€ ƒ„ ƒ
… ƒ
† ƒƒ ‡ˆ ‡‡ ‰Š ‰‰ ‹Œ ‹‹    
‘  ’“ ’
” ’
• ’’ –— –– ˜™ ˜˜ š› šš œ œœ Ÿ 
   ¡¢ ¡
£ ¡
¤ ¡¡ ¥¦ ¥
§ ¥¥ ¨© ¨
ª ¨¨ «
¬ «« ­® ­
¯ ­
° ­­ ±² ±
³ ±
´ ±± µ
¶ µµ ·¸ ·
¹ ·
º ·· »¼ »
½ »» ¾¿ ¾
À ¾
Á ¾¾ ÂÃ Â
Ä Â
Å Â
Æ ÂÂ ÇÈ Ç
É ÇÇ ÊË ÊÊ ÌÍ ÌÌ ÎÏ ÎÎ ĞÑ ĞĞ ÒÓ Ò
Ô ÒÒ ÕÖ Õ
× Õ
Ø ÕÕ ÙÚ ÙÙ ÛÜ ÛÛ İŞ İİ ßà ßß áâ á
ã áá äå ä
æ ä
ç ää èé èè êë êê ìí ìì îï îî ğñ ğ
ò ğğ óô ó
õ ó
ö óó ÷ø ÷
ù ÷÷ úû ú
ü úú ı
ş ıı ÿ€ ÿ
 ÿ
‚ ÿÿ ƒ„ ƒ
… ƒ
† ƒƒ ‡
ˆ ‡‡ ‰Š ‰
‹ ‰
Œ ‰‰  
  ‘ 
’ 
“  ”• ”
– ”
— ”
˜ ”” ™š ™
› ™™ œ œœ Ÿ   ¡    ¢£ ¢¢ ¤¥ ¤
¦ ¤¤ §¨ §
© §
ª §§ «¬ «« ­® ­­ ¯° ¯¯ ±² ±± ³´ ³
µ ³³ ¶· ¶
¸ ¶
¹ ¶¶ º» ºº ¼½ ¼¼ ¾¿ ¾¾ ÀÁ ÀÀ ÂÃ Â
Ä ÂÂ ÅÆ Å
Ç Å
È ÅÅ ÉÊ É
Ë ÉÉ ÌÍ Ì
Î ÌÌ Ï
Ğ ÏÏ ÑÒ Ñ
Ó Ñ
Ô ÑÑ ÕÖ Õ
× Õ
Ø ÕÕ Ù
Ú ÙÙ ÛÜ Û
İ Û
Ş ÛÛ ßà ß
á ßß âã â
ä â
å ââ æç æ
è æ
é æ
ê ææ ëì ë
í ëë î
ğ ïï ñ	ò 	ó ô 3	õ 	ö )	ö /	ö 8	ö >	ö D	ö J   	   
          
 "! $ & (# * +' , .# 0 1- 2% 5 74 9 :6 ; =4 ? @< A C4 E# FB G I4 K# LH M4 O# Q S UT W	 YX [ ]\ _- a' cN eb f4 h` id j< l6 nP pm q# sk to uH wB yR {x | ~v z €g ‚r ƒ …} †g ˆ‡ Šr ‹„ Œ‡ } ‰ r ’‘ ”} • –g ˜r ™— ›} œ“ 3 ŸV  Z ¡^ ¢š ¤ ¥ §¦ © «ª ­N ¯¬ °4 ²¨ ³® ´ ¶µ ¸ º¹ ¼P ¾» ¿# Á· Â½ Ã ÅÄ Ç ÉÈ ËR ÍÊ Î ĞÆ ÑÌ Ò± ÔÀ ÕÓ ×Ï Ø± ÚÙ ÜÀ İÖ ŞÙ àÏ áÛ âÀ äã æÏ çß è± êÀ ëé íÏ îå ï3 ñV òZ ó^ ôì öğ ÷ ùø û ıü ÿN ş ‚4 „ú …€ † ˆ‡ Š Œ‹ P  ‘# “‰ ” • —– ™ ›š R Ÿœ   ¢˜ £ ¤ƒ ¦’ §¥ ©¡ ªƒ ¬« ®’ ¯¨ °« ²¡ ³­ ´’ ¶µ ¸¡ ¹± ºƒ ¼’ ½» ¿¡ À· Á3 ÃV ÄZ Å^ Æ¾ ÈÂ É ËÊ Í ÏÎ ÑN ÓĞ Ô4 ÖÌ ×Ò Ø ÚÙ Ü Şİ àP âß ã# åÛ æá ç éè ë íì ïR ñî ò ôê õğ öÕ øä ù÷ ûó üÕ şı €ä ú ‚ı „ó …ÿ †ä ˆ‡ Šó ‹ƒ ŒÕ ä  ‘ó ’‰ “3 •V –Z —^ ˜ š” › œ Ÿ ¡  £N ¥¢ ¦4 ¨ ©¤ ª ¬« ® °¯ ²P ´± µ# ·­ ¸³ ¹ »º ½ ¿¾ ÁR ÃÀ Ä Æ¼ ÇÂ È§ Ê¶ ËÉ ÍÅ Î§ ĞÏ Ò¶ ÓÌ ÔÏ ÖÅ ×Ñ Ø¶ ÚÙ ÜÅ İÕ Ş§ à¶ áß ãÅ äÛ å3 çV èZ é^ êâ ìæ í ğ  ïî ï øø ÷÷ ñ ùù úú ûû‰ úú ‰À úú À“ úú “­ úú ­§ úú § úú ï ûû ï’ úú ’ß úú ß¡ úú ¡/ ùù /8 ùù 8> ùù >g úú g øø r úú r} úú }Û úú Ûå úú å¾ úú ¾Õ úú Õ øø ä úú ä úú 	 øø 	ƒ úú ƒ¶ úú ¶· úú ·Õ úú Õì úú ìƒ úú ƒÅ úú ÅD ùù D‰ úú ‰± úú ±Ï úú Ïó úú óÑ úú Ñ ÷÷ š úú šÿ úú ÿâ úú âJ ùù JÛ úú Û) ùù )± úú ±ü ı ı 		ş 	ş #	ş 4	ÿ B	ÿ H
ÿ Ä
ÿ È
ÿ ø
ÿ ü
ÿ ‡
ÿ ‹
ÿ –
ÿ –
ÿ š
ÿ š
ÿ Â
ÿ è
ÿ ì
ÿ º
ÿ ¾€ € ï 
‚ Ê
‚ Î
‚ Ù
‚ İ
‚ è
‚ ì
‚ ”
ƒ œ
ƒ  
ƒ «
ƒ ¯
ƒ º
ƒ ¾
ƒ æ„ )	„ 8	„ D	… -	… 6	… <	… <	… H
… ¦
… ¦
… ª
… µ
… µ
… µ
… ¹
… ¹
… Ä
… Ä
… È
… ğ
… ø
… ‡
… ‡
… ‹
… –
… Ê
… Ù
… Ù
… İ
… è
… œ
… «
… «
… ¯
… º† /	† >	† J† N† P† R	‡ T	‡ V	‡ X	‡ Z	‡ \	‡ ^	ˆ '	ˆ '	ˆ '	ˆ '	ˆ -	ˆ -	ˆ -	ˆ 6	ˆ 6	ˆ 6	ˆ <	ˆ <	ˆ B	ˆ B	ˆ B	ˆ H	ˆ H
ˆ 
ˆ ¦
ˆ ¦
ˆ ª
ˆ ª
ˆ ª
ˆ µ
ˆ ¹
ˆ ¹
ˆ Ä
ˆ È
ˆ È
ˆ ø
ˆ ø
ˆ ü
ˆ ü
ˆ ü
ˆ ‡
ˆ ‹
ˆ ‹
ˆ –
ˆ š
ˆ š
ˆ Ê
ˆ Ê
ˆ Î
ˆ Î
ˆ Î
ˆ Ù
ˆ İ
ˆ İ
ˆ è
ˆ ì
ˆ ì
ˆ œ
ˆ œ
ˆ  
ˆ  
ˆ  
ˆ «
ˆ ¯
ˆ ¯
ˆ º
ˆ ¾
ˆ ¾‰ ‡‰ ‘‰ Ù‰ ã‰ «‰ µ‰ ı‰ ‡‰ Ï‰ Ù"
initialize2"
llvm.lifetime.start.p0i8"
_Z13get_global_idj"
exact_solution"
llvm.fmuladd.f64"
llvm.lifetime.end.p0i8*
npb-BT-initialize2.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02€

transfer_bytes
à´Í

devmap_label
 

wgsize_log1p
Ï­„A
 
transfer_bytes_log1p
Ï­„A

wgsize
0