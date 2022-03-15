
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
)%21 = fmul double %20, 0x3F90410410410410
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
)%23 = fmul double %22, 0x3F90410410410410
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
ógetelementptr8BÉ
Ä
	full_texts
q
o%25 = getelementptr inbounds [2 x [3 x [5 x double]]], [2 x [3 x [5 x double]]]* %6, i64 0, i64 0, i64 0, i64 0
P[2 x [3 x [5 x double]]]*8B/
-
	full_text 

[2 x [3 x [5 x double]]]* %6
ácall8B}
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
ógetelementptr8BÉ
Ä
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
7%27 = bitcast double* %0 to [65 x [65 x [5 x double]]]*
Ffmul8B<
:
	full_text-
+
)%28 = fmul double %24, 0x3F90410410410410
+double8B

	full_text


double %24
ógetelementptr8BÉ
Ä
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
ógetelementptr8BÉ
Ä
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
ógetelementptr8BÉ
Ä
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
ógetelementptr8BÉ
Ä
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
¢getelementptr8Bé
ã
	full_text~
|
z%63 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %27, i64 %37, i64 %39, i64 %41, i64 0
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %27
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
ógetelementptr8BÉ
Ä
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
ógetelementptr8BÉ
Ä
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
ógetelementptr8BÉ
Ä
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
ógetelementptr8BÉ
Ä
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
ógetelementptr8BÉ
Ä
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
ógetelementptr8BÉ
Ä
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
¢getelementptr8Bé
ã
	full_text~
|
z%91 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %27, i64 %37, i64 %39, i64 %41, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %27
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
ógetelementptr8BÉ
Ä
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
ógetelementptr8BÉ
Ä
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
ógetelementptr8BÉ
Ä
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
ògetelementptr8BÑ
Å
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
ògetelementptr8BÑ
Å
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
ògetelementptr8BÑ
Å
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
£getelementptr8Bè
å
	full_text
}
{%119 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %27, i64 %37, i64 %39, i64 %41, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %27
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
ògetelementptr8BÑ
Å
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
ògetelementptr8BÑ
Å
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
ògetelementptr8BÑ
Å
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
ògetelementptr8BÑ
Å
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
ògetelementptr8BÑ
Å
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
ògetelementptr8BÑ
Å
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
£getelementptr8Bè
å
	full_text
}
{%147 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %27, i64 %37, i64 %39, i64 %41, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %27
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
ògetelementptr8BÑ
Å
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
ògetelementptr8BÑ
Å
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
ògetelementptr8BÑ
Å
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
ògetelementptr8BÑ
Å
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
ògetelementptr8BÑ
Å
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
ògetelementptr8BÑ
Å
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
£getelementptr8Bè
å
	full_text
}
{%175 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %27, i64 %37, i64 %39, i64 %41, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %27
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
i32 %2
,double*8B

	full_text


double* %1
$i328B

	full_text


i32 %4
$i328B

	full_text


i32 %3
,double*8B

	full_text


double* %0
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
#i648B

	full_text	

i64 0
#i328B

	full_text	

i32 2
%i648B

	full_text
	
i64 240
#i328B

	full_text	

i32 1
:double8B,
*
	full_text

double 0x3F90410410410410
#i328B

	full_text	

i32 0
4double8B&
$
	full_text

double 0.000000e+00
4double8B&
$
	full_text

double 1.000000e+00
#i648B

	full_text	

i64 3
#i648B

	full_text	

i64 1
#i648B

	full_text	

i64 2
5double8B'
%
	full_text

double -0.000000e+00
$i648B

	full_text


i64 32
#i648B

	full_text	

i64 4        		 
 

                      !" !! #$ ## %& %% '( '' )* )+ ), )) -. -- /0 /1 /2 // 33 45 44 67 66 89 8: 8; 88 <= << >? >@ >A >> BC BB DE DF DG DD HI HH JK JL JM JJ NO NN PQ PP RS RR TU TT VW VV XY XX Z[ ZZ \] \\ ^_ ^^ `a `` bc bb de df dd gh gi gj gg kl kk mn mm op oq oo rs rt ru rr vw vv xy xx z{ z| zz }~ } }	Ä }} ÅÇ Å
É ÅÅ ÑÖ Ñ
Ü ÑÑ á
à áá âä â
ã â
å ââ çé ç
è ç
ê çç ë
í ëë ìî ì
ï ì
ñ ìì óò ó
ô óó öõ ö
ú ö
ù öö ûü û
† û
° û
¢ ûû £§ £
• ££ ¶ß ¶¶ ®© ®® ™´ ™™ ¨≠ ¨¨ ÆØ Æ
∞ ÆÆ ±≤ ±
≥ ±
¥ ±± µ∂ µµ ∑∏ ∑∑ π∫ ππ ªº ªª Ωæ Ω
ø ΩΩ ¿¡ ¿
¬ ¿
√ ¿¿ ƒ≈ ƒƒ ∆« ∆∆ »… »»  À    ÃÕ Ã
Œ ÃÃ œ– œ
— œ
“ œœ ”‘ ”
’ ”” ÷◊ ÷
ÿ ÷÷ Ÿ
⁄ ŸŸ €‹ €
› €
ﬁ €€ ﬂ‡ ﬂ
· ﬂ
‚ ﬂﬂ „
‰ „„ ÂÊ Â
Á Â
Ë ÂÂ ÈÍ È
Î ÈÈ ÏÌ Ï
Ó Ï
Ô ÏÏ Ò 
Ú 
Û 
Ù  ıˆ ı
˜ ıı ¯˘ ¯¯ ˙˚ ˙˙ ¸˝ ¸¸ ˛ˇ ˛˛ ÄÅ Ä
Ç ÄÄ ÉÑ É
Ö É
Ü ÉÉ áà áá âä ââ ãå ãã çé çç èê è
ë èè íì í
î í
ï íí ñó ññ òô òò öõ öö úù úú ûü û
† ûû °¢ °
£ °
§ °° •¶ •
ß •• ®© ®
™ ®® ´
¨ ´´ ≠Æ ≠
Ø ≠
∞ ≠≠ ±≤ ±
≥ ±
¥ ±± µ
∂ µµ ∑∏ ∑
π ∑
∫ ∑∑ ªº ª
Ω ªª æø æ
¿ æ
¡ ææ ¬√ ¬
ƒ ¬
≈ ¬
∆ ¬¬ «» «
… ««  À    ÃÕ ÃÃ Œœ ŒŒ –— –– “” “
‘ ““ ’÷ ’
◊ ’
ÿ ’’ Ÿ⁄ ŸŸ €‹ €€ ›ﬁ ›› ﬂ‡ ﬂﬂ ·‚ ·
„ ·· ‰Â ‰
Ê ‰
Á ‰‰ ËÈ ËË ÍÎ ÍÍ ÏÌ ÏÏ ÓÔ ÓÓ Ò 
Ú  ÛÙ Û
ı Û
ˆ ÛÛ ˜¯ ˜
˘ ˜˜ ˙˚ ˙
¸ ˙˙ ˝
˛ ˝˝ ˇÄ ˇ
Å ˇ
Ç ˇˇ ÉÑ É
Ö É
Ü ÉÉ á
à áá âä â
ã â
å ââ çé ç
è çç êë ê
í ê
ì êê îï î
ñ î
ó î
ò îî ôö ô
õ ôô úù úú ûü ûû †° †† ¢£ ¢¢ §• §
¶ §§ ß® ß
© ß
™ ßß ´¨ ´´ ≠Æ ≠≠ Ø∞ ØØ ±≤ ±± ≥¥ ≥
µ ≥≥ ∂∑ ∂
∏ ∂
π ∂∂ ∫ª ∫∫ ºΩ ºº æø ææ ¿¡ ¿¿ ¬√ ¬
ƒ ¬¬ ≈∆ ≈
« ≈
» ≈≈ …  …
À …… ÃÕ Ã
Œ ÃÃ œ
– œœ —“ —
” —
‘ —— ’÷ ’
◊ ’
ÿ ’’ Ÿ
⁄ ŸŸ €‹ €
› €
ﬁ €€ ﬂ‡ ﬂ
· ﬂﬂ ‚„ ‚
‰ ‚
Â ‚‚ ÊÁ Ê
Ë Ê
È Ê
Í ÊÊ ÎÏ Î
Ì ÎÎ Ó
 ÔÔ Ò	Ú 	Û )	Û /	Û 8	Û >	Û D	Û J	Ù 	ı ˆ 3   	   
          
 "! $ & (# * +' , .# 0 1- 2% 5 74 9 :6 ; =4 ? @< A C4 E# FB G I4 K# LH M4 O# Q S UT W	 YX [ ]\ _- a' cN eb f4 h` id j< l6 nP pm q# sk to uH wB yR {x | ~v z Äg Çr ÉÅ Ö} Üg àá är ãÑ åá é} èâ êr íë î} ïç ñg òr ôó õ} úì ù3 üV †Z °^ ¢ö §û • ß¶ © ´™ ≠N Ø¨ ∞4 ≤® ≥Æ ¥ ∂µ ∏ ∫π ºP æª ø# ¡∑ ¬Ω √ ≈ƒ « …» ÀR Õ  Œ –∆ —Ã “± ‘¿ ’” ◊œ ÿ± ⁄Ÿ ‹¿ ›÷ ﬁŸ ‡œ ·€ ‚¿ ‰„ Êœ Áﬂ Ë± Í¿ ÎÈ Ìœ ÓÂ Ô3 ÒV ÚZ Û^ ÙÏ ˆ ˜ ˘¯ ˚ ˝¸ ˇN Å˛ Ç4 Ñ˙ ÖÄ Ü àá ä åã éP êç ë# ìâ îè ï óñ ô õö ùR üú † ¢ò £û §É ¶í ß• ©° ™É ¨´ Æí Ø® ∞´ ≤° ≥≠ ¥í ∂µ ∏° π± ∫É ºí Ωª ø° ¿∑ ¡3 √V ƒZ ≈^ ∆æ »¬ … À  Õ œŒ —N ”– ‘4 ÷Ã ◊“ ÿ ⁄Ÿ ‹ ﬁ› ‡P ‚ﬂ „# Â€ Ê· Á ÈË Î ÌÏ ÔR ÒÓ Ú ÙÍ ı ˆ’ ¯‰ ˘˜ ˚Û ¸’ ˛˝ Ä‰ Å˙ Ç˝ ÑÛ Öˇ Ü‰ àá äÛ ãÉ å’ é‰ èç ëÛ íâ ì3 ïV ñZ ó^ òê öî õ ùú ü °† £N •¢ ¶4 ®û ©§ ™ ¨´ Æ ∞Ø ≤P ¥± µ# ∑≠ ∏≥ π ª∫ Ω øæ ¡R √¿ ƒ ∆º «¬ »ß  ∂ À… Õ≈ Œß –œ “∂ ”Ã ‘œ ÷≈ ◊— ÿ∂ ⁄Ÿ ‹≈ ›’ ﬁß ‡∂ ·ﬂ „≈ ‰€ Â3 ÁV ËZ È^ Í‚ ÏÊ Ì   ÔÓ Ô ˜˜ ¯¯ Ò ˘˘ ˙˙ ˚˚í ˙˙ íâ ˙˙ â± ˙˙ ±€ ˙˙ €€ ˙˙ €/ ˘˘ /ö ˙˙ ö° ˙˙ °É ˙˙ ÉÔ ˚˚ Ôß ˙˙ ß’ ˙˙ ’æ ˙˙ ær ˙˙ r∂ ˙˙ ∂œ ˙˙ œD ˘˘ DÂ ˙˙ Â∑ ˙˙ ∑ﬂ ˙˙ ﬂ8 ˘˘ 8É ˙˙ É¿ ˙˙ ¿Û ˙˙ Û≈ ˙˙ ≈± ˙˙ ±ì ˙˙ ì ˜˜ Ï ˙˙ Ïg ˙˙ gˇ ˙˙ ˇâ ˙˙ â‰ ˙˙ ‰ ¯¯ ê ˙˙ ê— ˙˙ —‚ ˙˙ ‚’ ˙˙ ’> ˘˘ >J ˘˘ J ¯¯ ç ˙˙ ç} ˙˙ }) ˘˘ )	 ¯¯ 	≠ ˙˙ ≠	¸ '	¸ '	¸ '	¸ '	¸ -	¸ -	¸ -	¸ 6	¸ 6	¸ 6	¸ <	¸ <	¸ B	¸ B	¸ B	¸ H	¸ H
¸ û
¸ ¶
¸ ¶
¸ ™
¸ ™
¸ ™
¸ µ
¸ π
¸ π
¸ ƒ
¸ »
¸ »
¸ ¯
¸ ¯
¸ ¸
¸ ¸
¸ ¸
¸ á
¸ ã
¸ ã
¸ ñ
¸ ö
¸ ö
¸  
¸  
¸ Œ
¸ Œ
¸ Œ
¸ Ÿ
¸ ›
¸ ›
¸ Ë
¸ Ï
¸ Ï
¸ ú
¸ ú
¸ †
¸ †
¸ †
¸ ´
¸ Ø
¸ Ø
¸ ∫
¸ æ
¸ æ˝ ˛ ˛ Ôˇ ˇ 		Ä 	Ä #	Ä 4Å Ç )	Ç 8	Ç DÉ /	É >	É JÉ NÉ PÉ R
Ñ  
Ñ Œ
Ñ Ÿ
Ñ ›
Ñ Ë
Ñ Ï
Ñ î	Ö -	Ö 6	Ö <	Ö <	Ö H
Ö ¶
Ö ¶
Ö ™
Ö µ
Ö µ
Ö µ
Ö π
Ö π
Ö ƒ
Ö ƒ
Ö »
Ö 
Ö ¯
Ö á
Ö á
Ö ã
Ö ñ
Ö  
Ö Ÿ
Ö Ÿ
Ö ›
Ö Ë
Ö ú
Ö ´
Ö ´
Ö Ø
Ö ∫	Ü B	Ü H
Ü ƒ
Ü »
Ü ¯
Ü ¸
Ü á
Ü ã
Ü ñ
Ü ñ
Ü ö
Ü ö
Ü ¬
Ü Ë
Ü Ï
Ü ∫
Ü æá áá ëá Ÿá „á ´á µá ˝á áá œá Ÿ	à T	à V	à X	à Z	à \	à ^
â ú
â †
â ´
â Ø
â ∫
â æ
â Ê"
initialize2"
llvm.lifetime.start.p0i8"
_Z13get_global_idj"
exact_solution"
llvm.fmuladd.f64"
llvm.lifetime.end.p0i8*é
npb-BT-initialize2.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02Å

transfer_bytes	
ÿîÁò

devmap_label


wgsize_log1p
ùÆúA

wgsize
@
 
transfer_bytes_log1p
ùÆúA