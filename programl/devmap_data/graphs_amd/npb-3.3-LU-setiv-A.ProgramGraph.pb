

[external]
?allocaB5
3
	full_text&
$
"%6 = alloca [5 x double], align 16
?allocaB5
3
	full_text&
$
"%7 = alloca [5 x double], align 16
?allocaB5
3
	full_text&
$
"%8 = alloca [5 x double], align 16
?allocaB5
3
	full_text&
$
"%9 = alloca [5 x double], align 16
@allocaB6
4
	full_text'
%
#%10 = alloca [5 x double], align 16
@allocaB6
4
	full_text'
%
#%11 = alloca [5 x double], align 16
CbitcastB8
6
	full_text)
'
%%12 = bitcast [5 x double]* %6 to i8*
6[5 x double]*B#
!
	full_text

[5 x double]* %6
ZcallBR
P
	full_textC
A
?call void @llvm.lifetime.start.p0i8(i64 40, i8* nonnull %12) #5
#i8*B

	full_text
	
i8* %12
CbitcastB8
6
	full_text)
'
%%13 = bitcast [5 x double]* %7 to i8*
6[5 x double]*B#
!
	full_text

[5 x double]* %7
ZcallBR
P
	full_textC
A
?call void @llvm.lifetime.start.p0i8(i64 40, i8* nonnull %13) #5
#i8*B

	full_text
	
i8* %13
CbitcastB8
6
	full_text)
'
%%14 = bitcast [5 x double]* %8 to i8*
6[5 x double]*B#
!
	full_text

[5 x double]* %8
ZcallBR
P
	full_textC
A
?call void @llvm.lifetime.start.p0i8(i64 40, i8* nonnull %14) #5
#i8*B

	full_text
	
i8* %14
CbitcastB8
6
	full_text)
'
%%15 = bitcast [5 x double]* %9 to i8*
6[5 x double]*B#
!
	full_text

[5 x double]* %9
ZcallBR
P
	full_textC
A
?call void @llvm.lifetime.start.p0i8(i64 40, i8* nonnull %15) #5
#i8*B

	full_text
	
i8* %15
DbitcastB9
7
	full_text*
(
&%16 = bitcast [5 x double]* %10 to i8*
7[5 x double]*B$
"
	full_text

[5 x double]* %10
ZcallBR
P
	full_textC
A
?call void @llvm.lifetime.start.p0i8(i64 40, i8* nonnull %16) #5
#i8*B

	full_text
	
i8* %16
DbitcastB9
7
	full_text*
(
&%17 = bitcast [5 x double]* %11 to i8*
7[5 x double]*B$
"
	full_text

[5 x double]* %11
ZcallBR
P
	full_textC
A
?call void @llvm.lifetime.start.p0i8(i64 40, i8* nonnull %17) #5
#i8*B

	full_text
	
i8* %17
LcallBD
B
	full_text5
3
1%18 = tail call i64 @_Z13get_global_idj(i32 2) #6
.addB'
%
	full_text

%19 = add i64 %18, 1
#i64B

	full_text
	
i64 %18
6truncB-
+
	full_text

%20 = trunc i64 %19 to i32
#i64B

	full_text
	
i64 %19
LcallBD
B
	full_text5
3
1%21 = tail call i64 @_Z13get_global_idj(i32 1) #6
.addB'
%
	full_text

%22 = add i64 %21, 1
#i64B

	full_text
	
i64 %21
6truncB-
+
	full_text

%23 = trunc i64 %22 to i32
#i64B

	full_text
	
i64 %22
LcallBD
B
	full_text5
3
1%24 = tail call i64 @_Z13get_global_idj(i32 0) #6
.addB'
%
	full_text

%25 = add i64 %24, 1
#i64B

	full_text
	
i64 %24
6truncB-
+
	full_text

%26 = trunc i64 %25 to i32
#i64B

	full_text
	
i64 %25
2addB+
)
	full_text

%27 = add nsw i32 %4, -1
6icmpB.
,
	full_text

%28 = icmp sgt i32 %27, %20
#i32B

	full_text
	
i32 %27
#i32B

	full_text
	
i32 %20
2addB+
)
	full_text

%29 = add nsw i32 %3, -1
6icmpB.
,
	full_text

%30 = icmp sgt i32 %29, %23
#i32B

	full_text
	
i32 %29
#i32B

	full_text
	
i32 %23
/andB(
&
	full_text

%31 = and i1 %28, %30
!i1B

	full_text


i1 %28
!i1B

	full_text


i1 %30
2addB+
)
	full_text

%32 = add nsw i32 %2, -1
6icmpB.
,
	full_text

%33 = icmp sgt i32 %32, %26
#i32B

	full_text
	
i32 %32
#i32B

	full_text
	
i32 %26
/andB(
&
	full_text

%34 = and i1 %31, %33
!i1B

	full_text


i1 %31
!i1B

	full_text


i1 %33
9brB3
1
	full_text$
"
 br i1 %34, label %35, label %198
!i1B

	full_text


i1 %34
Wbitcast8BJ
H
	full_text;
9
7%36 = bitcast double* %0 to [65 x [65 x [5 x double]]]*
=sitofp8B1
/
	full_text"
 
%37 = sitofp i32 %20 to double
%i328B

	full_text
	
i32 %20
=sitofp8B1
/
	full_text"
 
%38 = sitofp i32 %27 to double
%i328B

	full_text
	
i32 %27
7fdiv8B-
+
	full_text

%39 = fdiv double %37, %38
+double8B

	full_text


double %37
+double8B

	full_text


double %38
=sitofp8B1
/
	full_text"
 
%40 = sitofp i32 %23 to double
%i328B

	full_text
	
i32 %23
@fdiv8B6
4
	full_text'
%
#%41 = fdiv double %40, 6.300000e+01
+double8B

	full_text


double %40
=sitofp8B1
/
	full_text"
 
%42 = sitofp i32 %26 to double
%i328B

	full_text
	
i32 %26
@fdiv8B6
4
	full_text'
%
#%43 = fdiv double %42, 6.300000e+01
+double8B

	full_text


double %42
ogetelementptr8B\
Z
	full_textM
K
I%44 = getelementptr inbounds [5 x double], [5 x double]* %6, i64 0, i64 0
8[5 x double]*8B#
!
	full_text

[5 x double]* %6
jcall8B`
^
	full_textQ
O
Mcall void @exact(i32 0, i32 %23, i32 %20, double* nonnull %44, double* %1) #5
%i328B

	full_text
	
i32 %23
%i328B

	full_text
	
i32 %20
-double*8B

	full_text

double* %44
ogetelementptr8B\
Z
	full_textM
K
I%45 = getelementptr inbounds [5 x double], [5 x double]* %7, i64 0, i64 0
8[5 x double]*8B#
!
	full_text

[5 x double]* %7
kcall8Ba
_
	full_textR
P
Ncall void @exact(i32 63, i32 %23, i32 %20, double* nonnull %45, double* %1) #5
%i328B

	full_text
	
i32 %23
%i328B

	full_text
	
i32 %20
-double*8B

	full_text

double* %45
ogetelementptr8B\
Z
	full_textM
K
I%46 = getelementptr inbounds [5 x double], [5 x double]* %8, i64 0, i64 0
8[5 x double]*8B#
!
	full_text

[5 x double]* %8
jcall8B`
^
	full_textQ
O
Mcall void @exact(i32 %26, i32 0, i32 %20, double* nonnull %46, double* %1) #5
%i328B

	full_text
	
i32 %26
%i328B

	full_text
	
i32 %20
-double*8B

	full_text

double* %46
ogetelementptr8B\
Z
	full_textM
K
I%47 = getelementptr inbounds [5 x double], [5 x double]* %9, i64 0, i64 0
8[5 x double]*8B#
!
	full_text

[5 x double]* %9
kcall8Ba
_
	full_textR
P
Ncall void @exact(i32 %26, i32 63, i32 %20, double* nonnull %47, double* %1) #5
%i328B

	full_text
	
i32 %26
%i328B

	full_text
	
i32 %20
-double*8B

	full_text

double* %47
pgetelementptr8B]
[
	full_textN
L
J%48 = getelementptr inbounds [5 x double], [5 x double]* %10, i64 0, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %10
jcall8B`
^
	full_textQ
O
Mcall void @exact(i32 %26, i32 %23, i32 0, double* nonnull %48, double* %1) #5
%i328B

	full_text
	
i32 %26
%i328B

	full_text
	
i32 %23
-double*8B

	full_text

double* %48
pgetelementptr8B]
[
	full_textN
L
J%49 = getelementptr inbounds [5 x double], [5 x double]* %11, i64 0, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %11
lcall8Bb
`
	full_textS
Q
Ocall void @exact(i32 %26, i32 %23, i32 %27, double* nonnull %49, double* %1) #5
%i328B

	full_text
	
i32 %26
%i328B

	full_text
	
i32 %23
%i328B

	full_text
	
i32 %27
-double*8B

	full_text

double* %49
@fsub8B6
4
	full_text'
%
#%50 = fsub double 1.000000e+00, %43
+double8B

	full_text


double %43
@fsub8B6
4
	full_text'
%
#%51 = fsub double 1.000000e+00, %41
+double8B

	full_text


double %41
@fsub8B6
4
	full_text'
%
#%52 = fsub double 1.000000e+00, %39
+double8B

	full_text


double %39
1shl8B(
&
	full_text

%53 = shl i64 %19, 32
%i648B

	full_text
	
i64 %19
9ashr8B/
-
	full_text 

%54 = ashr exact i64 %53, 32
%i648B

	full_text
	
i64 %53
1shl8B(
&
	full_text

%55 = shl i64 %22, 32
%i648B

	full_text
	
i64 %22
9ashr8B/
-
	full_text 

%56 = ashr exact i64 %55, 32
%i648B

	full_text
	
i64 %55
1shl8B(
&
	full_text

%57 = shl i64 %25, 32
%i648B

	full_text
	
i64 %25
9ashr8B/
-
	full_text 

%58 = ashr exact i64 %57, 32
%i648B

	full_text
	
i64 %57
Oload8BE
C
	full_text6
4
2%59 = load double, double* %44, align 16, !tbaa !8
-double*8B

	full_text

double* %44
Oload8BE
C
	full_text6
4
2%60 = load double, double* %45, align 16, !tbaa !8
-double*8B

	full_text

double* %45
7fmul8B-
+
	full_text

%61 = fmul double %43, %60
+double8B

	full_text


double %43
+double8B

	full_text


double %60
dcall8BZ
X
	full_textK
I
G%62 = call double @llvm.fmuladd.f64(double %50, double %59, double %61)
+double8B

	full_text


double %50
+double8B

	full_text


double %59
+double8B

	full_text


double %61
Oload8BE
C
	full_text6
4
2%63 = load double, double* %46, align 16, !tbaa !8
-double*8B

	full_text

double* %46
Oload8BE
C
	full_text6
4
2%64 = load double, double* %47, align 16, !tbaa !8
-double*8B

	full_text

double* %47
7fmul8B-
+
	full_text

%65 = fmul double %41, %64
+double8B

	full_text


double %41
+double8B

	full_text


double %64
dcall8BZ
X
	full_textK
I
G%66 = call double @llvm.fmuladd.f64(double %51, double %63, double %65)
+double8B

	full_text


double %51
+double8B

	full_text


double %63
+double8B

	full_text


double %65
Oload8BE
C
	full_text6
4
2%67 = load double, double* %48, align 16, !tbaa !8
-double*8B

	full_text

double* %48
Oload8BE
C
	full_text6
4
2%68 = load double, double* %49, align 16, !tbaa !8
-double*8B

	full_text

double* %49
7fmul8B-
+
	full_text

%69 = fmul double %39, %68
+double8B

	full_text


double %39
+double8B

	full_text


double %68
dcall8BZ
X
	full_textK
I
G%70 = call double @llvm.fmuladd.f64(double %52, double %67, double %69)
+double8B

	full_text


double %52
+double8B

	full_text


double %67
+double8B

	full_text


double %69
7fadd8B-
+
	full_text

%71 = fadd double %62, %66
+double8B

	full_text


double %62
+double8B

	full_text


double %66
7fadd8B-
+
	full_text

%72 = fadd double %71, %70
+double8B

	full_text


double %71
+double8B

	full_text


double %70
Afsub8B7
5
	full_text(
&
$%73 = fsub double -0.000000e+00, %62
+double8B

	full_text


double %62
dcall8BZ
X
	full_textK
I
G%74 = call double @llvm.fmuladd.f64(double %73, double %66, double %72)
+double8B

	full_text


double %73
+double8B

	full_text


double %66
+double8B

	full_text


double %72
Afsub8B7
5
	full_text(
&
$%75 = fsub double -0.000000e+00, %66
+double8B

	full_text


double %66
dcall8BZ
X
	full_textK
I
G%76 = call double @llvm.fmuladd.f64(double %75, double %70, double %74)
+double8B

	full_text


double %75
+double8B

	full_text


double %70
+double8B

	full_text


double %74
Afsub8B7
5
	full_text(
&
$%77 = fsub double -0.000000e+00, %70
+double8B

	full_text


double %70
dcall8BZ
X
	full_textK
I
G%78 = call double @llvm.fmuladd.f64(double %77, double %62, double %76)
+double8B

	full_text


double %77
+double8B

	full_text


double %62
+double8B

	full_text


double %76
7fmul8B-
+
	full_text

%79 = fmul double %62, %66
+double8B

	full_text


double %62
+double8B

	full_text


double %66
dcall8BZ
X
	full_textK
I
G%80 = call double @llvm.fmuladd.f64(double %79, double %70, double %78)
+double8B

	full_text


double %79
+double8B

	full_text


double %70
+double8B

	full_text


double %78
¢getelementptr8Bé
ã
	full_text~
|
z%81 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %36, i64 %54, i64 %56, i64 %58, i64 0
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %36
%i648B

	full_text
	
i64 %54
%i648B

	full_text
	
i64 %56
%i648B

	full_text
	
i64 %58
Nstore8BC
A
	full_text4
2
0store double %80, double* %81, align 8, !tbaa !8
+double8B

	full_text


double %80
-double*8B

	full_text

double* %81
ogetelementptr8B\
Z
	full_textM
K
I%82 = getelementptr inbounds [5 x double], [5 x double]* %6, i64 0, i64 1
8[5 x double]*8B#
!
	full_text

[5 x double]* %6
Nload8BD
B
	full_text5
3
1%83 = load double, double* %82, align 8, !tbaa !8
-double*8B

	full_text

double* %82
ogetelementptr8B\
Z
	full_textM
K
I%84 = getelementptr inbounds [5 x double], [5 x double]* %7, i64 0, i64 1
8[5 x double]*8B#
!
	full_text

[5 x double]* %7
Nload8BD
B
	full_text5
3
1%85 = load double, double* %84, align 8, !tbaa !8
-double*8B

	full_text

double* %84
7fmul8B-
+
	full_text

%86 = fmul double %43, %85
+double8B

	full_text


double %43
+double8B

	full_text


double %85
dcall8BZ
X
	full_textK
I
G%87 = call double @llvm.fmuladd.f64(double %50, double %83, double %86)
+double8B

	full_text


double %50
+double8B

	full_text


double %83
+double8B

	full_text


double %86
ogetelementptr8B\
Z
	full_textM
K
I%88 = getelementptr inbounds [5 x double], [5 x double]* %8, i64 0, i64 1
8[5 x double]*8B#
!
	full_text

[5 x double]* %8
Nload8BD
B
	full_text5
3
1%89 = load double, double* %88, align 8, !tbaa !8
-double*8B

	full_text

double* %88
ogetelementptr8B\
Z
	full_textM
K
I%90 = getelementptr inbounds [5 x double], [5 x double]* %9, i64 0, i64 1
8[5 x double]*8B#
!
	full_text

[5 x double]* %9
Nload8BD
B
	full_text5
3
1%91 = load double, double* %90, align 8, !tbaa !8
-double*8B

	full_text

double* %90
7fmul8B-
+
	full_text

%92 = fmul double %41, %91
+double8B

	full_text


double %41
+double8B

	full_text


double %91
dcall8BZ
X
	full_textK
I
G%93 = call double @llvm.fmuladd.f64(double %51, double %89, double %92)
+double8B

	full_text


double %51
+double8B

	full_text


double %89
+double8B

	full_text


double %92
pgetelementptr8B]
[
	full_textN
L
J%94 = getelementptr inbounds [5 x double], [5 x double]* %10, i64 0, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %10
Nload8BD
B
	full_text5
3
1%95 = load double, double* %94, align 8, !tbaa !8
-double*8B

	full_text

double* %94
pgetelementptr8B]
[
	full_textN
L
J%96 = getelementptr inbounds [5 x double], [5 x double]* %11, i64 0, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %11
Nload8BD
B
	full_text5
3
1%97 = load double, double* %96, align 8, !tbaa !8
-double*8B

	full_text

double* %96
7fmul8B-
+
	full_text

%98 = fmul double %39, %97
+double8B

	full_text


double %39
+double8B

	full_text


double %97
dcall8BZ
X
	full_textK
I
G%99 = call double @llvm.fmuladd.f64(double %52, double %95, double %98)
+double8B

	full_text


double %52
+double8B

	full_text


double %95
+double8B

	full_text


double %98
8fadd8B.
,
	full_text

%100 = fadd double %87, %93
+double8B

	full_text


double %87
+double8B

	full_text


double %93
9fadd8B/
-
	full_text 

%101 = fadd double %100, %99
,double8B

	full_text

double %100
+double8B

	full_text


double %99
Bfsub8B8
6
	full_text)
'
%%102 = fsub double -0.000000e+00, %87
+double8B

	full_text


double %87
gcall8B]
[
	full_textN
L
J%103 = call double @llvm.fmuladd.f64(double %102, double %93, double %101)
,double8B

	full_text

double %102
+double8B

	full_text


double %93
,double8B

	full_text

double %101
Bfsub8B8
6
	full_text)
'
%%104 = fsub double -0.000000e+00, %93
+double8B

	full_text


double %93
gcall8B]
[
	full_textN
L
J%105 = call double @llvm.fmuladd.f64(double %104, double %99, double %103)
,double8B

	full_text

double %104
+double8B

	full_text


double %99
,double8B

	full_text

double %103
Bfsub8B8
6
	full_text)
'
%%106 = fsub double -0.000000e+00, %99
+double8B

	full_text


double %99
gcall8B]
[
	full_textN
L
J%107 = call double @llvm.fmuladd.f64(double %106, double %87, double %105)
,double8B

	full_text

double %106
+double8B

	full_text


double %87
,double8B

	full_text

double %105
8fmul8B.
,
	full_text

%108 = fmul double %87, %93
+double8B

	full_text


double %87
+double8B

	full_text


double %93
gcall8B]
[
	full_textN
L
J%109 = call double @llvm.fmuladd.f64(double %108, double %99, double %107)
,double8B

	full_text

double %108
+double8B

	full_text


double %99
,double8B

	full_text

double %107
£getelementptr8Bè
å
	full_text
}
{%110 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %36, i64 %54, i64 %56, i64 %58, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %36
%i648B

	full_text
	
i64 %54
%i648B

	full_text
	
i64 %56
%i648B

	full_text
	
i64 %58
Pstore8BE
C
	full_text6
4
2store double %109, double* %110, align 8, !tbaa !8
,double8B

	full_text

double %109
.double*8B

	full_text

double* %110
pgetelementptr8B]
[
	full_textN
L
J%111 = getelementptr inbounds [5 x double], [5 x double]* %6, i64 0, i64 2
8[5 x double]*8B#
!
	full_text

[5 x double]* %6
Qload8BG
E
	full_text8
6
4%112 = load double, double* %111, align 16, !tbaa !8
.double*8B

	full_text

double* %111
pgetelementptr8B]
[
	full_textN
L
J%113 = getelementptr inbounds [5 x double], [5 x double]* %7, i64 0, i64 2
8[5 x double]*8B#
!
	full_text

[5 x double]* %7
Qload8BG
E
	full_text8
6
4%114 = load double, double* %113, align 16, !tbaa !8
.double*8B

	full_text

double* %113
9fmul8B/
-
	full_text 

%115 = fmul double %43, %114
+double8B

	full_text


double %43
,double8B

	full_text

double %114
gcall8B]
[
	full_textN
L
J%116 = call double @llvm.fmuladd.f64(double %50, double %112, double %115)
+double8B

	full_text


double %50
,double8B

	full_text

double %112
,double8B

	full_text

double %115
pgetelementptr8B]
[
	full_textN
L
J%117 = getelementptr inbounds [5 x double], [5 x double]* %8, i64 0, i64 2
8[5 x double]*8B#
!
	full_text

[5 x double]* %8
Qload8BG
E
	full_text8
6
4%118 = load double, double* %117, align 16, !tbaa !8
.double*8B

	full_text

double* %117
pgetelementptr8B]
[
	full_textN
L
J%119 = getelementptr inbounds [5 x double], [5 x double]* %9, i64 0, i64 2
8[5 x double]*8B#
!
	full_text

[5 x double]* %9
Qload8BG
E
	full_text8
6
4%120 = load double, double* %119, align 16, !tbaa !8
.double*8B

	full_text

double* %119
9fmul8B/
-
	full_text 

%121 = fmul double %41, %120
+double8B

	full_text


double %41
,double8B

	full_text

double %120
gcall8B]
[
	full_textN
L
J%122 = call double @llvm.fmuladd.f64(double %51, double %118, double %121)
+double8B

	full_text


double %51
,double8B

	full_text

double %118
,double8B

	full_text

double %121
qgetelementptr8B^
\
	full_textO
M
K%123 = getelementptr inbounds [5 x double], [5 x double]* %10, i64 0, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %10
Qload8BG
E
	full_text8
6
4%124 = load double, double* %123, align 16, !tbaa !8
.double*8B

	full_text

double* %123
qgetelementptr8B^
\
	full_textO
M
K%125 = getelementptr inbounds [5 x double], [5 x double]* %11, i64 0, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %11
Qload8BG
E
	full_text8
6
4%126 = load double, double* %125, align 16, !tbaa !8
.double*8B

	full_text

double* %125
9fmul8B/
-
	full_text 

%127 = fmul double %39, %126
+double8B

	full_text


double %39
,double8B

	full_text

double %126
gcall8B]
[
	full_textN
L
J%128 = call double @llvm.fmuladd.f64(double %52, double %124, double %127)
+double8B

	full_text


double %52
,double8B

	full_text

double %124
,double8B

	full_text

double %127
:fadd8B0
.
	full_text!

%129 = fadd double %116, %122
,double8B

	full_text

double %116
,double8B

	full_text

double %122
:fadd8B0
.
	full_text!

%130 = fadd double %129, %128
,double8B

	full_text

double %129
,double8B

	full_text

double %128
Cfsub8B9
7
	full_text*
(
&%131 = fsub double -0.000000e+00, %116
,double8B

	full_text

double %116
hcall8B^
\
	full_textO
M
K%132 = call double @llvm.fmuladd.f64(double %131, double %122, double %130)
,double8B

	full_text

double %131
,double8B

	full_text

double %122
,double8B

	full_text

double %130
Cfsub8B9
7
	full_text*
(
&%133 = fsub double -0.000000e+00, %122
,double8B

	full_text

double %122
hcall8B^
\
	full_textO
M
K%134 = call double @llvm.fmuladd.f64(double %133, double %128, double %132)
,double8B

	full_text

double %133
,double8B

	full_text

double %128
,double8B

	full_text

double %132
Cfsub8B9
7
	full_text*
(
&%135 = fsub double -0.000000e+00, %128
,double8B

	full_text

double %128
hcall8B^
\
	full_textO
M
K%136 = call double @llvm.fmuladd.f64(double %135, double %116, double %134)
,double8B

	full_text

double %135
,double8B

	full_text

double %116
,double8B

	full_text

double %134
:fmul8B0
.
	full_text!

%137 = fmul double %116, %122
,double8B

	full_text

double %116
,double8B

	full_text

double %122
hcall8B^
\
	full_textO
M
K%138 = call double @llvm.fmuladd.f64(double %137, double %128, double %136)
,double8B

	full_text

double %137
,double8B

	full_text

double %128
,double8B

	full_text

double %136
£getelementptr8Bè
å
	full_text
}
{%139 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %36, i64 %54, i64 %56, i64 %58, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %36
%i648B

	full_text
	
i64 %54
%i648B

	full_text
	
i64 %56
%i648B

	full_text
	
i64 %58
Pstore8BE
C
	full_text6
4
2store double %138, double* %139, align 8, !tbaa !8
,double8B

	full_text

double %138
.double*8B

	full_text

double* %139
pgetelementptr8B]
[
	full_textN
L
J%140 = getelementptr inbounds [5 x double], [5 x double]* %6, i64 0, i64 3
8[5 x double]*8B#
!
	full_text

[5 x double]* %6
Pload8BF
D
	full_text7
5
3%141 = load double, double* %140, align 8, !tbaa !8
.double*8B

	full_text

double* %140
pgetelementptr8B]
[
	full_textN
L
J%142 = getelementptr inbounds [5 x double], [5 x double]* %7, i64 0, i64 3
8[5 x double]*8B#
!
	full_text

[5 x double]* %7
Pload8BF
D
	full_text7
5
3%143 = load double, double* %142, align 8, !tbaa !8
.double*8B

	full_text

double* %142
9fmul8B/
-
	full_text 

%144 = fmul double %43, %143
+double8B

	full_text


double %43
,double8B

	full_text

double %143
gcall8B]
[
	full_textN
L
J%145 = call double @llvm.fmuladd.f64(double %50, double %141, double %144)
+double8B

	full_text


double %50
,double8B

	full_text

double %141
,double8B

	full_text

double %144
pgetelementptr8B]
[
	full_textN
L
J%146 = getelementptr inbounds [5 x double], [5 x double]* %8, i64 0, i64 3
8[5 x double]*8B#
!
	full_text

[5 x double]* %8
Pload8BF
D
	full_text7
5
3%147 = load double, double* %146, align 8, !tbaa !8
.double*8B

	full_text

double* %146
pgetelementptr8B]
[
	full_textN
L
J%148 = getelementptr inbounds [5 x double], [5 x double]* %9, i64 0, i64 3
8[5 x double]*8B#
!
	full_text

[5 x double]* %9
Pload8BF
D
	full_text7
5
3%149 = load double, double* %148, align 8, !tbaa !8
.double*8B

	full_text

double* %148
9fmul8B/
-
	full_text 

%150 = fmul double %41, %149
+double8B

	full_text


double %41
,double8B

	full_text

double %149
gcall8B]
[
	full_textN
L
J%151 = call double @llvm.fmuladd.f64(double %51, double %147, double %150)
+double8B

	full_text


double %51
,double8B

	full_text

double %147
,double8B

	full_text

double %150
qgetelementptr8B^
\
	full_textO
M
K%152 = getelementptr inbounds [5 x double], [5 x double]* %10, i64 0, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %10
Pload8BF
D
	full_text7
5
3%153 = load double, double* %152, align 8, !tbaa !8
.double*8B

	full_text

double* %152
qgetelementptr8B^
\
	full_textO
M
K%154 = getelementptr inbounds [5 x double], [5 x double]* %11, i64 0, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %11
Pload8BF
D
	full_text7
5
3%155 = load double, double* %154, align 8, !tbaa !8
.double*8B

	full_text

double* %154
9fmul8B/
-
	full_text 

%156 = fmul double %39, %155
+double8B

	full_text


double %39
,double8B

	full_text

double %155
gcall8B]
[
	full_textN
L
J%157 = call double @llvm.fmuladd.f64(double %52, double %153, double %156)
+double8B

	full_text


double %52
,double8B

	full_text

double %153
,double8B

	full_text

double %156
:fadd8B0
.
	full_text!

%158 = fadd double %145, %151
,double8B

	full_text

double %145
,double8B

	full_text

double %151
:fadd8B0
.
	full_text!

%159 = fadd double %158, %157
,double8B

	full_text

double %158
,double8B

	full_text

double %157
Cfsub8B9
7
	full_text*
(
&%160 = fsub double -0.000000e+00, %145
,double8B

	full_text

double %145
hcall8B^
\
	full_textO
M
K%161 = call double @llvm.fmuladd.f64(double %160, double %151, double %159)
,double8B

	full_text

double %160
,double8B

	full_text

double %151
,double8B

	full_text

double %159
Cfsub8B9
7
	full_text*
(
&%162 = fsub double -0.000000e+00, %151
,double8B

	full_text

double %151
hcall8B^
\
	full_textO
M
K%163 = call double @llvm.fmuladd.f64(double %162, double %157, double %161)
,double8B

	full_text

double %162
,double8B

	full_text

double %157
,double8B

	full_text

double %161
Cfsub8B9
7
	full_text*
(
&%164 = fsub double -0.000000e+00, %157
,double8B

	full_text

double %157
hcall8B^
\
	full_textO
M
K%165 = call double @llvm.fmuladd.f64(double %164, double %145, double %163)
,double8B

	full_text

double %164
,double8B

	full_text

double %145
,double8B

	full_text

double %163
:fmul8B0
.
	full_text!

%166 = fmul double %145, %151
,double8B

	full_text

double %145
,double8B

	full_text

double %151
hcall8B^
\
	full_textO
M
K%167 = call double @llvm.fmuladd.f64(double %166, double %157, double %165)
,double8B

	full_text

double %166
,double8B

	full_text

double %157
,double8B

	full_text

double %165
£getelementptr8Bè
å
	full_text
}
{%168 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %36, i64 %54, i64 %56, i64 %58, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %36
%i648B

	full_text
	
i64 %54
%i648B

	full_text
	
i64 %56
%i648B

	full_text
	
i64 %58
Pstore8BE
C
	full_text6
4
2store double %167, double* %168, align 8, !tbaa !8
,double8B

	full_text

double %167
.double*8B

	full_text

double* %168
pgetelementptr8B]
[
	full_textN
L
J%169 = getelementptr inbounds [5 x double], [5 x double]* %6, i64 0, i64 4
8[5 x double]*8B#
!
	full_text

[5 x double]* %6
Qload8BG
E
	full_text8
6
4%170 = load double, double* %169, align 16, !tbaa !8
.double*8B

	full_text

double* %169
pgetelementptr8B]
[
	full_textN
L
J%171 = getelementptr inbounds [5 x double], [5 x double]* %7, i64 0, i64 4
8[5 x double]*8B#
!
	full_text

[5 x double]* %7
Qload8BG
E
	full_text8
6
4%172 = load double, double* %171, align 16, !tbaa !8
.double*8B

	full_text

double* %171
9fmul8B/
-
	full_text 

%173 = fmul double %43, %172
+double8B

	full_text


double %43
,double8B

	full_text

double %172
gcall8B]
[
	full_textN
L
J%174 = call double @llvm.fmuladd.f64(double %50, double %170, double %173)
+double8B

	full_text


double %50
,double8B

	full_text

double %170
,double8B

	full_text

double %173
pgetelementptr8B]
[
	full_textN
L
J%175 = getelementptr inbounds [5 x double], [5 x double]* %8, i64 0, i64 4
8[5 x double]*8B#
!
	full_text

[5 x double]* %8
Qload8BG
E
	full_text8
6
4%176 = load double, double* %175, align 16, !tbaa !8
.double*8B

	full_text

double* %175
pgetelementptr8B]
[
	full_textN
L
J%177 = getelementptr inbounds [5 x double], [5 x double]* %9, i64 0, i64 4
8[5 x double]*8B#
!
	full_text

[5 x double]* %9
Qload8BG
E
	full_text8
6
4%178 = load double, double* %177, align 16, !tbaa !8
.double*8B

	full_text

double* %177
9fmul8B/
-
	full_text 

%179 = fmul double %41, %178
+double8B

	full_text


double %41
,double8B

	full_text

double %178
gcall8B]
[
	full_textN
L
J%180 = call double @llvm.fmuladd.f64(double %51, double %176, double %179)
+double8B

	full_text


double %51
,double8B

	full_text

double %176
,double8B

	full_text

double %179
qgetelementptr8B^
\
	full_textO
M
K%181 = getelementptr inbounds [5 x double], [5 x double]* %10, i64 0, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %10
Qload8BG
E
	full_text8
6
4%182 = load double, double* %181, align 16, !tbaa !8
.double*8B

	full_text

double* %181
qgetelementptr8B^
\
	full_textO
M
K%183 = getelementptr inbounds [5 x double], [5 x double]* %11, i64 0, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %11
Qload8BG
E
	full_text8
6
4%184 = load double, double* %183, align 16, !tbaa !8
.double*8B

	full_text

double* %183
9fmul8B/
-
	full_text 

%185 = fmul double %39, %184
+double8B

	full_text


double %39
,double8B

	full_text

double %184
gcall8B]
[
	full_textN
L
J%186 = call double @llvm.fmuladd.f64(double %52, double %182, double %185)
+double8B

	full_text


double %52
,double8B

	full_text

double %182
,double8B

	full_text

double %185
:fadd8B0
.
	full_text!

%187 = fadd double %174, %180
,double8B

	full_text

double %174
,double8B

	full_text

double %180
:fadd8B0
.
	full_text!

%188 = fadd double %187, %186
,double8B

	full_text

double %187
,double8B

	full_text

double %186
Cfsub8B9
7
	full_text*
(
&%189 = fsub double -0.000000e+00, %174
,double8B

	full_text

double %174
hcall8B^
\
	full_textO
M
K%190 = call double @llvm.fmuladd.f64(double %189, double %180, double %188)
,double8B

	full_text

double %189
,double8B

	full_text

double %180
,double8B

	full_text

double %188
Cfsub8B9
7
	full_text*
(
&%191 = fsub double -0.000000e+00, %180
,double8B

	full_text

double %180
hcall8B^
\
	full_textO
M
K%192 = call double @llvm.fmuladd.f64(double %191, double %186, double %190)
,double8B

	full_text

double %191
,double8B

	full_text

double %186
,double8B

	full_text

double %190
Cfsub8B9
7
	full_text*
(
&%193 = fsub double -0.000000e+00, %186
,double8B

	full_text

double %186
hcall8B^
\
	full_textO
M
K%194 = call double @llvm.fmuladd.f64(double %193, double %174, double %192)
,double8B

	full_text

double %193
,double8B

	full_text

double %174
,double8B

	full_text

double %192
:fmul8B0
.
	full_text!

%195 = fmul double %174, %180
,double8B

	full_text

double %174
,double8B

	full_text

double %180
hcall8B^
\
	full_textO
M
K%196 = call double @llvm.fmuladd.f64(double %195, double %186, double %194)
,double8B

	full_text

double %195
,double8B

	full_text

double %186
,double8B

	full_text

double %194
£getelementptr8Bè
å
	full_text
}
{%197 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %36, i64 %54, i64 %56, i64 %58, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %36
%i648B

	full_text
	
i64 %54
%i648B

	full_text
	
i64 %56
%i648B

	full_text
	
i64 %58
Pstore8BE
C
	full_text6
4
2store double %196, double* %197, align 8, !tbaa !8
,double8B

	full_text

double %196
.double*8B

	full_text

double* %197
(br8B 

	full_text

br label %198
Zcall8BP
N
	full_textA
?
=call void @llvm.lifetime.end.p0i8(i64 40, i8* nonnull %17) #5
%i8*8B

	full_text
	
i8* %17
Zcall8BP
N
	full_textA
?
=call void @llvm.lifetime.end.p0i8(i64 40, i8* nonnull %16) #5
%i8*8B

	full_text
	
i8* %16
Zcall8BP
N
	full_textA
?
=call void @llvm.lifetime.end.p0i8(i64 40, i8* nonnull %15) #5
%i8*8B

	full_text
	
i8* %15
Zcall8BP
N
	full_textA
?
=call void @llvm.lifetime.end.p0i8(i64 40, i8* nonnull %14) #5
%i8*8B

	full_text
	
i8* %14
Zcall8BP
N
	full_textA
?
=call void @llvm.lifetime.end.p0i8(i64 40, i8* nonnull %13) #5
%i8*8B

	full_text
	
i8* %13
Zcall8BP
N
	full_textA
?
=call void @llvm.lifetime.end.p0i8(i64 40, i8* nonnull %12) #5
%i8*8B

	full_text
	
i8* %12
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
$i328B

	full_text


i32 %4
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
$i328B

	full_text


i32 63
#i648B

	full_text	

i64 2
#i648B

	full_text	

i64 3
#i328B

	full_text	

i32 2
#i648B

	full_text	

i64 1
#i328B

	full_text	

i32 0
#i648B

	full_text	

i64 4
4double8B&
$
	full_text

double 1.000000e+00
$i648B

	full_text


i64 40
$i648B

	full_text


i64 32
$i328B

	full_text


i32 -1
#i328B

	full_text	

i32 1
#i648B

	full_text	

i64 0
5double8B'
%
	full_text

double -0.000000e+00
4double8B&
$
	full_text

double 6.300000e+01        	
 		                       !    "# "" $$ %& %% '( '' )) *+ ** ,- ,, .. /0 /1 // 22 34 35 33 67 68 66 99 :; :< :: => =? == @A @B CD CC EF EE GH GI GG JK JJ LM LL NO NN PQ PP RS RR TU TV TW TT XY XX Z[ Z\ Z] ZZ ^_ ^^ `a `b `c `` de dd fg fh fi ff jk jj lm ln lo ll pq pp rs rt ru rv rr wx ww yz yy {| {{ }~ }} Ä  ÅÇ ÅÅ ÉÑ ÉÉ ÖÜ ÖÖ áà áá âä ââ ãå ãã çé ç
è çç êë ê
í ê
ì êê îï îî ñó ññ òô ò
ö òò õú õ
ù õ
û õõ ü† üü °¢ °° £§ £
• ££ ¶ß ¶
® ¶
© ¶¶ ™´ ™
¨ ™™ ≠Æ ≠
Ø ≠≠ ∞
± ∞∞ ≤≥ ≤
¥ ≤
µ ≤≤ ∂
∑ ∂∂ ∏π ∏
∫ ∏
ª ∏∏ º
Ω ºº æø æ
¿ æ
¡ ææ ¬√ ¬
ƒ ¬¬ ≈∆ ≈
« ≈
» ≈≈ …  …
À …
Ã …
Õ …… Œœ Œ
– ŒŒ —“ —— ”‘ ”” ’÷ ’’ ◊ÿ ◊◊ Ÿ⁄ Ÿ
€ ŸŸ ‹› ‹
ﬁ ‹
ﬂ ‹‹ ‡· ‡‡ ‚„ ‚‚ ‰Â ‰‰ ÊÁ ÊÊ ËÈ Ë
Í ËË ÎÏ Î
Ì Î
Ó ÎÎ Ô ÔÔ ÒÚ ÒÒ ÛÙ ÛÛ ıˆ ıı ˜¯ ˜
˘ ˜˜ ˙˚ ˙
¸ ˙
˝ ˙˙ ˛ˇ ˛
Ä ˛˛ ÅÇ Å
É ÅÅ Ñ
Ö ÑÑ Üá Ü
à Ü
â ÜÜ ä
ã ää åç å
é å
è åå ê
ë êê íì í
î í
ï íí ñó ñ
ò ññ ôö ô
õ ô
ú ôô ùû ù
ü ù
† ù
° ùù ¢£ ¢
§ ¢¢ •¶ •• ß® ßß ©™ ©© ´¨ ´´ ≠Æ ≠
Ø ≠≠ ∞± ∞
≤ ∞
≥ ∞∞ ¥µ ¥¥ ∂∑ ∂∂ ∏π ∏∏ ∫ª ∫∫ ºΩ º
æ ºº ø¿ ø
¡ ø
¬ øø √ƒ √√ ≈∆ ≈≈ «» «« …  …… ÀÃ À
Õ ÀÀ Œœ Œ
– Œ
— ŒŒ “” “
‘ ““ ’÷ ’
◊ ’’ ÿ
Ÿ ÿÿ ⁄€ ⁄
‹ ⁄
› ⁄⁄ ﬁ
ﬂ ﬁﬁ ‡· ‡
‚ ‡
„ ‡‡ ‰
Â ‰‰ ÊÁ Ê
Ë Ê
È ÊÊ ÍÎ Í
Ï ÍÍ ÌÓ Ì
Ô Ì
 ÌÌ ÒÚ Ò
Û Ò
Ù Ò
ı ÒÒ ˆ˜ ˆ
¯ ˆˆ ˘˙ ˘˘ ˚¸ ˚˚ ˝˛ ˝˝ ˇÄ ˇˇ ÅÇ Å
É ÅÅ ÑÖ Ñ
Ü Ñ
á ÑÑ àâ àà äã ää åç åå éè éé êë ê
í êê ìî ì
ï ì
ñ ìì óò óó ôö ôô õú õõ ùû ùù ü† ü
° üü ¢£ ¢
§ ¢
• ¢¢ ¶ß ¶
® ¶¶ ©™ ©
´ ©© ¨
≠ ¨¨ ÆØ Æ
∞ Æ
± ÆÆ ≤
≥ ≤≤ ¥µ ¥
∂ ¥
∑ ¥¥ ∏
π ∏∏ ∫ª ∫
º ∫
Ω ∫∫ æø æ
¿ ææ ¡¬ ¡
√ ¡
ƒ ¡¡ ≈∆ ≈
« ≈
» ≈
… ≈≈  À  
Ã    ÕŒ ÕÕ œ– œœ —“ —— ”‘ ”” ’÷ ’
◊ ’’ ÿŸ ÿ
⁄ ÿ
€ ÿÿ ‹› ‹‹ ﬁﬂ ﬁﬁ ‡· ‡‡ ‚„ ‚‚ ‰Â ‰
Ê ‰‰ ÁË Á
È Á
Í ÁÁ ÎÏ ÎÎ ÌÓ ÌÌ Ô ÔÔ ÒÚ ÒÒ ÛÙ Û
ı ÛÛ ˆ˜ ˆ
¯ ˆ
˘ ˆˆ ˙˚ ˙
¸ ˙˙ ˝˛ ˝
ˇ ˝˝ Ä
Å ÄÄ ÇÉ Ç
Ñ Ç
Ö ÇÇ Ü
á ÜÜ àâ à
ä à
ã àà å
ç åå éè é
ê é
ë éé íì í
î íí ïñ ï
ó ï
ò ïï ôö ô
õ ô
ú ô
ù ôô ûü û
† ûû °
£ ¢¢ §
• §§ ¶
ß ¶¶ ®
© ®® ™
´ ™™ ¨
≠ ¨¨ ÆØ 9∞ B± 2	≤ T	≤ Z	≤ `	≤ f	≤ l	≤ r≥ .  
           !  #$ &% () +* -. 0" 12 4' 5/ 73 89 ;, <6 >: ?= A" D. FC HE I' KJ M, ON Q S' U" VR W Y' [" \X ] _, a" b^ c e, g" hd i k, m' nj o q, s' t. up vP xL zG |  ~} Ä% ÇÅ Ñ* ÜÖ àR äX åP éã èw ëâ íç ì^ ïd óL ôñ öy úî ùò ûj †p ¢G §° •{ ßü ®£ ©ê ´õ ¨™ Æ¶ Øê ±∞ ≥õ ¥≠ µõ ∑∂ π¶ ∫≤ ª¶ Ωº øê ¿∏ ¡ê √õ ƒ¬ ∆¶ «æ »B   ÀÉ Ãá Õ≈ œ… – “— ‘ ÷’ ÿP ⁄◊ €w ›” ﬁŸ ﬂ ·‡ „ Â‰ ÁL ÈÊ Íy Ï‚ ÌË Ó Ô Ú ÙÛ ˆG ¯ı ˘{ ˚Ò ¸˜ ˝‹ ˇÎ Ä˛ Ç˙ É‹ ÖÑ áÎ àÅ âÎ ãä ç˙ éÜ è˙ ëê ì‹ îå ï‹ óÎ òñ ö˙ õí úB û üÉ †á °ô £ù § ¶• ® ™© ¨P Æ´ Øw ±ß ≤≠ ≥ µ¥ ∑ π∏ ªL Ω∫ æy ¿∂ ¡º ¬ ƒ√ ∆ »«  G Ã… Õ{ œ≈ –À —∞ ”ø ‘“ ÷Œ ◊∞ Ÿÿ €ø ‹’ ›ø ﬂﬁ ·Œ ‚⁄ „Œ Â‰ Á∞ Ë‡ È∞ Îø ÏÍ ÓŒ ÔÊ B Ú ÛÉ Ùá ıÌ ˜Ò ¯ ˙˘ ¸ ˛˝ ÄP Çˇ Éw Ö˚ ÜÅ á âà ã çå èL ëé íy îä ïê ñ òó ö úõ ûG †ù °{ £ô §ü •Ñ ßì ®¶ ™¢ ´Ñ ≠¨ Øì ∞© ±ì ≥≤ µ¢ ∂Æ ∑¢ π∏ ªÑ º¥ ΩÑ øì ¿æ ¬¢ √∫ ƒB ∆ «É »á …¡ À≈ Ã ŒÕ – “— ‘P ÷” ◊w Ÿœ ⁄’ € ›‹ ﬂ ·‡ „L Â‚ Êy Ëﬁ È‰ Í ÏÎ Ó Ô ÚG ÙÒ ı{ ˜Ì ¯Û ˘ÿ ˚Á ¸˙ ˛ˆ ˇÿ ÅÄ ÉÁ Ñ˝ ÖÁ áÜ âˆ äÇ ãˆ çå èÿ êà ëÿ ìÁ îí ñˆ óé òB ö õÉ úá ùï üô † £ • ß © ´ ≠@ B@ ¢° ¢ ∂∂ ∑∑ ∏∏ ¥¥ µµ Æ ¥¥ ô ∑∑ ôÑ ∑∑ Ñ ¥¥ å ∑∑ å∏ ∑∑ ∏Œ ∑∑ Œì ∑∑ ìà ∑∑ à) µµ )$ µµ $¢ ∏∏ ¢Ç ∑∑ Ç§ ∏∏ §¶ ∏∏ ¶õ ∑∑ õZ ∂∂ ZÆ ∑∑ Æ¢ ∑∑ ¢‹ ∑∑ ‹¥ ∑∑ ¥æ ∑∑ æ⁄ ∑∑ ⁄f ∂∂ f¶ ∑∑ ¶Á ∑∑ Á® ∏∏ ® ¥¥  µµ l ∂∂ lí ∑∑ íˆ ∑∑ ˆT ∂∂ TÌ ∑∑ Ì¨ ∏∏ ¨¡ ∑∑ ¡é ∑∑ é ¥¥  ¥¥ r ∂∂ rï ∑∑ ï≤ ∑∑ ≤‡ ∑∑ ‡≈ ∑∑ ≈™ ∏∏ ™˙ ∑∑ ˙	 ¥¥ 	Î ∑∑ Î∫ ∑∑ ∫ÿ ∑∑ ÿ` ∂∂ `ø ∑∑ ø∞ ∑∑ ∞ê ∑∑ êÜ ∑∑ ÜÊ ∑∑ Êπ Z	π f
∫ •
∫ ©
∫ ¥
∫ ∏
∫ √
∫ «
∫ Ò
ª ˘
ª ˝
ª à
ª å
ª ó
ª õ
ª ≈º 	Ω  	Ω %	Ω *
Ω —
Ω ’
Ω ‡
Ω ‰
Ω Ô
Ω Û
Ω ùæ )æ T	æ `	æ l
ø Õ
ø —
ø ‹
ø ‡
ø Î
ø Ô
ø ô¿ w¿ y¿ {¡ 	¡ ¡ ¡ ¡ ¡ ¡ ¢¡ §¡ ¶¡ ®¡ ™¡ ¨	¬ }	¬ 
¬ Å
¬ É
¬ Ö
¬ á	√ .	√ 2	√ 9ƒ ƒ ƒ ƒ ƒ ƒ ƒ $	≈ R	≈ R	≈ X	≈ X	≈ ^	≈ ^	≈ d	≈ d	≈ j	≈ j	≈ p	≈ p
≈ …
≈ —
≈ ’
≈ ‡
≈ ‰
≈ Ô
≈ Û
≈ •
≈ ©
≈ ¥
≈ ∏
≈ √
≈ «
≈ ˘
≈ ˝
≈ à
≈ å
≈ ó
≈ õ
≈ Õ
≈ —
≈ ‹
≈ ‡
≈ Î
≈ Ô∆ ∞∆ ∂∆ º∆ Ñ∆ ä∆ ê∆ ÿ∆ ﬁ∆ ‰∆ ¨∆ ≤∆ ∏∆ Ä∆ Ü∆ å	« L	« P"
setiv"
llvm.lifetime.start.p0i8"
_Z13get_global_idj"
exact"
llvm.fmuladd.f64"
llvm.lifetime.end.p0i8*à
npb-LU-setiv.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02Ä

wgsize_log1p
⁄açA
 
transfer_bytes_log1p
⁄açA

wgsize
>

transfer_bytes
òì…

devmap_label
