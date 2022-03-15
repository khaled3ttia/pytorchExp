

[external]
DallocaB:
8
	full_text+
)
'%2 = alloca [8 x <2 x float>], align 16
JcallBB
@
	full_text3
1
/%3 = tail call i64 @_Z12get_local_idj(i32 0) #5
4truncB+
)
	full_text

%4 = trunc i64 %3 to i32
"i64B

	full_text


i64 %3
JcallBB
@
	full_text3
1
/%5 = tail call i64 @_Z12get_group_idj(i32 0) #5
,shlB%
#
	full_text

%6 = shl i64 %5, 9
"i64B

	full_text


i64 %5
-addB&
$
	full_text

%7 = add i64 %6, %3
"i64B

	full_text


i64 %6
"i64B

	full_text


i64 %3
.ashrB&
$
	full_text

%8 = ashr i32 %4, 3
"i32B

	full_text


i32 %4
GbitcastB<
:
	full_text-
+
)%9 = bitcast [8 x <2 x float>]* %2 to i8*
@[8 x <2 x float>]*B(
&
	full_text

[8 x <2 x float>]* %2
YcallBQ
O
	full_textB
@
>call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %9) #6
"i8*B

	full_text


i8* %9
.shlB'
%
	full_text

%10 = shl i64 %7, 32
"i64B

	full_text


i64 %7
7ashrB/
-
	full_text 

%11 = ashr exact i64 %10, 32
#i64B

	full_text
	
i64 %10
fgetelementptrBU
S
	full_textF
D
B%12 = getelementptr inbounds <2 x float>, <2 x float>* %0, i64 %11
#i64B

	full_text
	
i64 %11
wgetelementptrBf
d
	full_textW
U
S%13 = getelementptr inbounds [8 x <2 x float>], [8 x <2 x float>]* %2, i64 0, i64 0
@[8 x <2 x float>]*B(
&
	full_text

[8 x <2 x float>]* %2
icallBa
_
	full_textR
P
Ncall void @globalLoads8(<2 x float>* nonnull %13, <2 x float>* %12, i32 64) #6
5<2 x float>*B#
!
	full_text

<2 x float>* %13
5<2 x float>*B#
!
	full_text

<2 x float>* %12
LbitcastBA
?
	full_text2
0
.%14 = bitcast [8 x <2 x float>]* %2 to double*
@[8 x <2 x float>]*B(
&
	full_text

[8 x <2 x float>]* %2
MloadBE
C
	full_text6
4
2%15 = load double, double* %14, align 16, !tbaa !9
+double*B

	full_text

double* %14
wgetelementptrBf
d
	full_textW
U
S%16 = getelementptr inbounds [8 x <2 x float>], [8 x <2 x float>]* %2, i64 0, i64 4
@[8 x <2 x float>]*B(
&
	full_text

[8 x <2 x float>]* %2
GbitcastB<
:
	full_text-
+
)%17 = bitcast <2 x float>* %16 to double*
5<2 x float>*B#
!
	full_text

<2 x float>* %16
MloadBE
C
	full_text6
4
2%18 = load double, double* %17, align 16, !tbaa !9
+double*B

	full_text

double* %17
RcallBJ
H
	full_text;
9
7%19 = call double @cmplx_add(double %15, double %18) #6
)doubleB

	full_text


double %15
)doubleB

	full_text


double %18
MstoreBD
B
	full_text5
3
1store double %19, double* %14, align 16, !tbaa !9
)doubleB

	full_text


double %19
+double*B

	full_text

double* %14
MloadBE
C
	full_text6
4
2%20 = load double, double* %17, align 16, !tbaa !9
+double*B

	full_text

double* %17
RcallBJ
H
	full_text;
9
7%21 = call double @cmplx_sub(double %15, double %20) #6
)doubleB

	full_text


double %15
)doubleB

	full_text


double %20
MstoreBD
B
	full_text5
3
1store double %21, double* %17, align 16, !tbaa !9
)doubleB

	full_text


double %21
+double*B

	full_text

double* %17
wgetelementptrBf
d
	full_textW
U
S%22 = getelementptr inbounds [8 x <2 x float>], [8 x <2 x float>]* %2, i64 0, i64 1
@[8 x <2 x float>]*B(
&
	full_text

[8 x <2 x float>]* %2
GbitcastB<
:
	full_text-
+
)%23 = bitcast <2 x float>* %22 to double*
5<2 x float>*B#
!
	full_text

<2 x float>* %22
LloadBD
B
	full_text5
3
1%24 = load double, double* %23, align 8, !tbaa !9
+double*B

	full_text

double* %23
wgetelementptrBf
d
	full_textW
U
S%25 = getelementptr inbounds [8 x <2 x float>], [8 x <2 x float>]* %2, i64 0, i64 5
@[8 x <2 x float>]*B(
&
	full_text

[8 x <2 x float>]* %2
GbitcastB<
:
	full_text-
+
)%26 = bitcast <2 x float>* %25 to double*
5<2 x float>*B#
!
	full_text

<2 x float>* %25
LloadBD
B
	full_text5
3
1%27 = load double, double* %26, align 8, !tbaa !9
+double*B

	full_text

double* %26
RcallBJ
H
	full_text;
9
7%28 = call double @cmplx_add(double %24, double %27) #6
)doubleB

	full_text


double %24
)doubleB

	full_text


double %27
LstoreBC
A
	full_text4
2
0store double %28, double* %23, align 8, !tbaa !9
)doubleB

	full_text


double %28
+double*B

	full_text

double* %23
LloadBD
B
	full_text5
3
1%29 = load double, double* %26, align 8, !tbaa !9
+double*B

	full_text

double* %26
RcallBJ
H
	full_text;
9
7%30 = call double @cmplx_sub(double %24, double %29) #6
)doubleB

	full_text


double %24
)doubleB

	full_text


double %29
LstoreBC
A
	full_text4
2
0store double %30, double* %26, align 8, !tbaa !9
)doubleB

	full_text


double %30
+double*B

	full_text

double* %26
wgetelementptrBf
d
	full_textW
U
S%31 = getelementptr inbounds [8 x <2 x float>], [8 x <2 x float>]* %2, i64 0, i64 2
@[8 x <2 x float>]*B(
&
	full_text

[8 x <2 x float>]* %2
GbitcastB<
:
	full_text-
+
)%32 = bitcast <2 x float>* %31 to double*
5<2 x float>*B#
!
	full_text

<2 x float>* %31
MloadBE
C
	full_text6
4
2%33 = load double, double* %32, align 16, !tbaa !9
+double*B

	full_text

double* %32
wgetelementptrBf
d
	full_textW
U
S%34 = getelementptr inbounds [8 x <2 x float>], [8 x <2 x float>]* %2, i64 0, i64 6
@[8 x <2 x float>]*B(
&
	full_text

[8 x <2 x float>]* %2
GbitcastB<
:
	full_text-
+
)%35 = bitcast <2 x float>* %34 to double*
5<2 x float>*B#
!
	full_text

<2 x float>* %34
MloadBE
C
	full_text6
4
2%36 = load double, double* %35, align 16, !tbaa !9
+double*B

	full_text

double* %35
RcallBJ
H
	full_text;
9
7%37 = call double @cmplx_add(double %33, double %36) #6
)doubleB

	full_text


double %33
)doubleB

	full_text


double %36
MstoreBD
B
	full_text5
3
1store double %37, double* %32, align 16, !tbaa !9
)doubleB

	full_text


double %37
+double*B

	full_text

double* %32
MloadBE
C
	full_text6
4
2%38 = load double, double* %35, align 16, !tbaa !9
+double*B

	full_text

double* %35
RcallBJ
H
	full_text;
9
7%39 = call double @cmplx_sub(double %33, double %38) #6
)doubleB

	full_text


double %33
)doubleB

	full_text


double %38
MstoreBD
B
	full_text5
3
1store double %39, double* %35, align 16, !tbaa !9
)doubleB

	full_text


double %39
+double*B

	full_text

double* %35
wgetelementptrBf
d
	full_textW
U
S%40 = getelementptr inbounds [8 x <2 x float>], [8 x <2 x float>]* %2, i64 0, i64 3
@[8 x <2 x float>]*B(
&
	full_text

[8 x <2 x float>]* %2
GbitcastB<
:
	full_text-
+
)%41 = bitcast <2 x float>* %40 to double*
5<2 x float>*B#
!
	full_text

<2 x float>* %40
LloadBD
B
	full_text5
3
1%42 = load double, double* %41, align 8, !tbaa !9
+double*B

	full_text

double* %41
wgetelementptrBf
d
	full_textW
U
S%43 = getelementptr inbounds [8 x <2 x float>], [8 x <2 x float>]* %2, i64 0, i64 7
@[8 x <2 x float>]*B(
&
	full_text

[8 x <2 x float>]* %2
GbitcastB<
:
	full_text-
+
)%44 = bitcast <2 x float>* %43 to double*
5<2 x float>*B#
!
	full_text

<2 x float>* %43
LloadBD
B
	full_text5
3
1%45 = load double, double* %44, align 8, !tbaa !9
+double*B

	full_text

double* %44
RcallBJ
H
	full_text;
9
7%46 = call double @cmplx_add(double %42, double %45) #6
)doubleB

	full_text


double %42
)doubleB

	full_text


double %45
LstoreBC
A
	full_text4
2
0store double %46, double* %41, align 8, !tbaa !9
)doubleB

	full_text


double %46
+double*B

	full_text

double* %41
LloadBD
B
	full_text5
3
1%47 = load double, double* %44, align 8, !tbaa !9
+double*B

	full_text

double* %44
RcallBJ
H
	full_text;
9
7%48 = call double @cmplx_sub(double %42, double %47) #6
)doubleB

	full_text


double %42
)doubleB

	full_text


double %47
LstoreBC
A
	full_text4
2
0store double %48, double* %44, align 8, !tbaa !9
)doubleB

	full_text


double %48
+double*B

	full_text

double* %44
LloadBD
B
	full_text5
3
1%49 = load double, double* %26, align 8, !tbaa !9
+double*B

	full_text

double* %26
acallBY
W
	full_textJ
H
F%50 = call double @cmplx_mul(double %49, double 0xBF8000003F800000) #6
)doubleB

	full_text


double %49
`callBX
V
	full_textI
G
E%51 = call double @cm_fl_mul(double %50, float 0x3FE6A09E60000000) #6
)doubleB

	full_text


double %50
LstoreBC
A
	full_text4
2
0store double %51, double* %26, align 8, !tbaa !9
)doubleB

	full_text


double %51
+double*B

	full_text

double* %26
MloadBE
C
	full_text6
4
2%52 = load double, double* %35, align 16, !tbaa !9
+double*B

	full_text

double* %35
\callBT
R
	full_textE
C
A%53 = call double @cmplx_mul(double %52, double -7.812500e-03) #6
)doubleB

	full_text


double %52
MstoreBD
B
	full_text5
3
1store double %53, double* %35, align 16, !tbaa !9
)doubleB

	full_text


double %53
+double*B

	full_text

double* %35
LloadBD
B
	full_text5
3
1%54 = load double, double* %44, align 8, !tbaa !9
+double*B

	full_text

double* %44
acallBY
W
	full_textJ
H
F%55 = call double @cmplx_mul(double %54, double 0xBF800000BF800000) #6
)doubleB

	full_text


double %54
`callBX
V
	full_textI
G
E%56 = call double @cm_fl_mul(double %55, float 0x3FE6A09E60000000) #6
)doubleB

	full_text


double %55
LstoreBC
A
	full_text4
2
0store double %56, double* %44, align 8, !tbaa !9
)doubleB

	full_text


double %56
+double*B

	full_text

double* %44
MloadBE
C
	full_text6
4
2%57 = load double, double* %14, align 16, !tbaa !9
+double*B

	full_text

double* %14
MloadBE
C
	full_text6
4
2%58 = load double, double* %32, align 16, !tbaa !9
+double*B

	full_text

double* %32
RcallBJ
H
	full_text;
9
7%59 = call double @cmplx_add(double %57, double %58) #6
)doubleB

	full_text


double %57
)doubleB

	full_text


double %58
MstoreBD
B
	full_text5
3
1store double %59, double* %14, align 16, !tbaa !9
)doubleB

	full_text


double %59
+double*B

	full_text

double* %14
MloadBE
C
	full_text6
4
2%60 = load double, double* %32, align 16, !tbaa !9
+double*B

	full_text

double* %32
RcallBJ
H
	full_text;
9
7%61 = call double @cmplx_sub(double %57, double %60) #6
)doubleB

	full_text


double %57
)doubleB

	full_text


double %60
MstoreBD
B
	full_text5
3
1store double %61, double* %32, align 16, !tbaa !9
)doubleB

	full_text


double %61
+double*B

	full_text

double* %32
LloadBD
B
	full_text5
3
1%62 = load double, double* %23, align 8, !tbaa !9
+double*B

	full_text

double* %23
LloadBD
B
	full_text5
3
1%63 = load double, double* %41, align 8, !tbaa !9
+double*B

	full_text

double* %41
RcallBJ
H
	full_text;
9
7%64 = call double @cmplx_add(double %62, double %63) #6
)doubleB

	full_text


double %62
)doubleB

	full_text


double %63
LstoreBC
A
	full_text4
2
0store double %64, double* %23, align 8, !tbaa !9
)doubleB

	full_text


double %64
+double*B

	full_text

double* %23
LloadBD
B
	full_text5
3
1%65 = load double, double* %41, align 8, !tbaa !9
+double*B

	full_text

double* %41
RcallBJ
H
	full_text;
9
7%66 = call double @cmplx_sub(double %62, double %65) #6
)doubleB

	full_text


double %62
)doubleB

	full_text


double %65
LstoreBC
A
	full_text4
2
0store double %66, double* %41, align 8, !tbaa !9
)doubleB

	full_text


double %66
+double*B

	full_text

double* %41
\callBT
R
	full_textE
C
A%67 = call double @cmplx_mul(double %66, double -7.812500e-03) #6
)doubleB

	full_text


double %66
LstoreBC
A
	full_text4
2
0store double %67, double* %41, align 8, !tbaa !9
)doubleB

	full_text


double %67
+double*B

	full_text

double* %41
MloadBE
C
	full_text6
4
2%68 = load double, double* %14, align 16, !tbaa !9
+double*B

	full_text

double* %14
LloadBD
B
	full_text5
3
1%69 = load double, double* %23, align 8, !tbaa !9
+double*B

	full_text

double* %23
RcallBJ
H
	full_text;
9
7%70 = call double @cmplx_add(double %68, double %69) #6
)doubleB

	full_text


double %68
)doubleB

	full_text


double %69
MstoreBD
B
	full_text5
3
1store double %70, double* %14, align 16, !tbaa !9
)doubleB

	full_text


double %70
+double*B

	full_text

double* %14
LloadBD
B
	full_text5
3
1%71 = load double, double* %23, align 8, !tbaa !9
+double*B

	full_text

double* %23
RcallBJ
H
	full_text;
9
7%72 = call double @cmplx_sub(double %68, double %71) #6
)doubleB

	full_text


double %68
)doubleB

	full_text


double %71
LstoreBC
A
	full_text4
2
0store double %72, double* %23, align 8, !tbaa !9
)doubleB

	full_text


double %72
+double*B

	full_text

double* %23
MloadBE
C
	full_text6
4
2%73 = load double, double* %32, align 16, !tbaa !9
+double*B

	full_text

double* %32
LloadBD
B
	full_text5
3
1%74 = load double, double* %41, align 8, !tbaa !9
+double*B

	full_text

double* %41
RcallBJ
H
	full_text;
9
7%75 = call double @cmplx_add(double %73, double %74) #6
)doubleB

	full_text


double %73
)doubleB

	full_text


double %74
MstoreBD
B
	full_text5
3
1store double %75, double* %32, align 16, !tbaa !9
)doubleB

	full_text


double %75
+double*B

	full_text

double* %32
LloadBD
B
	full_text5
3
1%76 = load double, double* %41, align 8, !tbaa !9
+double*B

	full_text

double* %41
RcallBJ
H
	full_text;
9
7%77 = call double @cmplx_sub(double %73, double %76) #6
)doubleB

	full_text


double %73
)doubleB

	full_text


double %76
LstoreBC
A
	full_text4
2
0store double %77, double* %41, align 8, !tbaa !9
)doubleB

	full_text


double %77
+double*B

	full_text

double* %41
MloadBE
C
	full_text6
4
2%78 = load double, double* %17, align 16, !tbaa !9
+double*B

	full_text

double* %17
MloadBE
C
	full_text6
4
2%79 = load double, double* %35, align 16, !tbaa !9
+double*B

	full_text

double* %35
RcallBJ
H
	full_text;
9
7%80 = call double @cmplx_add(double %78, double %79) #6
)doubleB

	full_text


double %78
)doubleB

	full_text


double %79
MstoreBD
B
	full_text5
3
1store double %80, double* %17, align 16, !tbaa !9
)doubleB

	full_text


double %80
+double*B

	full_text

double* %17
MloadBE
C
	full_text6
4
2%81 = load double, double* %35, align 16, !tbaa !9
+double*B

	full_text

double* %35
RcallBJ
H
	full_text;
9
7%82 = call double @cmplx_sub(double %78, double %81) #6
)doubleB

	full_text


double %78
)doubleB

	full_text


double %81
MstoreBD
B
	full_text5
3
1store double %82, double* %35, align 16, !tbaa !9
)doubleB

	full_text


double %82
+double*B

	full_text

double* %35
LloadBD
B
	full_text5
3
1%83 = load double, double* %26, align 8, !tbaa !9
+double*B

	full_text

double* %26
LloadBD
B
	full_text5
3
1%84 = load double, double* %44, align 8, !tbaa !9
+double*B

	full_text

double* %44
RcallBJ
H
	full_text;
9
7%85 = call double @cmplx_add(double %83, double %84) #6
)doubleB

	full_text


double %83
)doubleB

	full_text


double %84
LstoreBC
A
	full_text4
2
0store double %85, double* %26, align 8, !tbaa !9
)doubleB

	full_text


double %85
+double*B

	full_text

double* %26
LloadBD
B
	full_text5
3
1%86 = load double, double* %44, align 8, !tbaa !9
+double*B

	full_text

double* %44
RcallBJ
H
	full_text;
9
7%87 = call double @cmplx_sub(double %83, double %86) #6
)doubleB

	full_text


double %83
)doubleB

	full_text


double %86
LstoreBC
A
	full_text4
2
0store double %87, double* %44, align 8, !tbaa !9
)doubleB

	full_text


double %87
+double*B

	full_text

double* %44
\callBT
R
	full_textE
C
A%88 = call double @cmplx_mul(double %87, double -7.812500e-03) #6
)doubleB

	full_text


double %87
LstoreBC
A
	full_text4
2
0store double %88, double* %44, align 8, !tbaa !9
)doubleB

	full_text


double %88
+double*B

	full_text

double* %44
MloadBE
C
	full_text6
4
2%89 = load double, double* %17, align 16, !tbaa !9
+double*B

	full_text

double* %17
LloadBD
B
	full_text5
3
1%90 = load double, double* %26, align 8, !tbaa !9
+double*B

	full_text

double* %26
RcallBJ
H
	full_text;
9
7%91 = call double @cmplx_add(double %89, double %90) #6
)doubleB

	full_text


double %89
)doubleB

	full_text


double %90
MstoreBD
B
	full_text5
3
1store double %91, double* %17, align 16, !tbaa !9
)doubleB

	full_text


double %91
+double*B

	full_text

double* %17
LloadBD
B
	full_text5
3
1%92 = load double, double* %26, align 8, !tbaa !9
+double*B

	full_text

double* %26
RcallBJ
H
	full_text;
9
7%93 = call double @cmplx_sub(double %89, double %92) #6
)doubleB

	full_text


double %89
)doubleB

	full_text


double %92
LstoreBC
A
	full_text4
2
0store double %93, double* %26, align 8, !tbaa !9
)doubleB

	full_text


double %93
+double*B

	full_text

double* %26
MloadBE
C
	full_text6
4
2%94 = load double, double* %35, align 16, !tbaa !9
+double*B

	full_text

double* %35
LloadBD
B
	full_text5
3
1%95 = load double, double* %44, align 8, !tbaa !9
+double*B

	full_text

double* %44
RcallBJ
H
	full_text;
9
7%96 = call double @cmplx_add(double %94, double %95) #6
)doubleB

	full_text


double %94
)doubleB

	full_text


double %95
MstoreBD
B
	full_text5
3
1store double %96, double* %35, align 16, !tbaa !9
)doubleB

	full_text


double %96
+double*B

	full_text

double* %35
LloadBD
B
	full_text5
3
1%97 = load double, double* %44, align 8, !tbaa !9
+double*B

	full_text

double* %44
RcallBJ
H
	full_text;
9
7%98 = call double @cmplx_sub(double %94, double %97) #6
)doubleB

	full_text


double %94
)doubleB

	full_text


double %97
LstoreBC
A
	full_text4
2
0store double %98, double* %44, align 8, !tbaa !9
)doubleB

	full_text


double %98
+double*B

	full_text

double* %44
9sitofpB/
-
	full_text 

%99 = sitofp i32 %4 to float
"i32B

	full_text


i32 %4
MloadBE
C
	full_text6
4
2%100 = load double, double* %23, align 8, !tbaa !9
+double*B

	full_text

double* %23
DfmulB<
:
	full_text-
+
)%101 = fmul float %99, 0xBFA921FB60000000
'floatB

	full_text

	float %99
CcallB;
9
	full_text,
*
(%102 = call double @exp_i(float %101) #6
(floatB

	full_text


float %101
UcallBM
K
	full_text>
<
:%103 = call double @cmplx_mul(double %100, double %102) #6
*doubleB

	full_text

double %100
*doubleB

	full_text

double %102
MstoreBD
B
	full_text5
3
1store double %103, double* %23, align 8, !tbaa !9
*doubleB

	full_text

double %103
+double*B

	full_text

double* %23
NloadBF
D
	full_text7
5
3%104 = load double, double* %32, align 16, !tbaa !9
+double*B

	full_text

double* %32
DfmulB<
:
	full_text-
+
)%105 = fmul float %99, 0xBF9921FB60000000
'floatB

	full_text

	float %99
CcallB;
9
	full_text,
*
(%106 = call double @exp_i(float %105) #6
(floatB

	full_text


float %105
UcallBM
K
	full_text>
<
:%107 = call double @cmplx_mul(double %104, double %106) #6
*doubleB

	full_text

double %104
*doubleB

	full_text

double %106
NstoreBE
C
	full_text6
4
2store double %107, double* %32, align 16, !tbaa !9
*doubleB

	full_text

double %107
+double*B

	full_text

double* %32
MloadBE
C
	full_text6
4
2%108 = load double, double* %41, align 8, !tbaa !9
+double*B

	full_text

double* %41
DfmulB<
:
	full_text-
+
)%109 = fmul float %99, 0xBFB2D97C80000000
'floatB

	full_text

	float %99
CcallB;
9
	full_text,
*
(%110 = call double @exp_i(float %109) #6
(floatB

	full_text


float %109
UcallBM
K
	full_text>
<
:%111 = call double @cmplx_mul(double %108, double %110) #6
*doubleB

	full_text

double %108
*doubleB

	full_text

double %110
MstoreBD
B
	full_text5
3
1store double %111, double* %41, align 8, !tbaa !9
*doubleB

	full_text

double %111
+double*B

	full_text

double* %41
NloadBF
D
	full_text7
5
3%112 = load double, double* %17, align 16, !tbaa !9
+double*B

	full_text

double* %17
DfmulB<
:
	full_text-
+
)%113 = fmul float %99, 0xBF8921FB60000000
'floatB

	full_text

	float %99
CcallB;
9
	full_text,
*
(%114 = call double @exp_i(float %113) #6
(floatB

	full_text


float %113
UcallBM
K
	full_text>
<
:%115 = call double @cmplx_mul(double %112, double %114) #6
*doubleB

	full_text

double %112
*doubleB

	full_text

double %114
NstoreBE
C
	full_text6
4
2store double %115, double* %17, align 16, !tbaa !9
*doubleB

	full_text

double %115
+double*B

	full_text

double* %17
MloadBE
C
	full_text6
4
2%116 = load double, double* %26, align 8, !tbaa !9
+double*B

	full_text

double* %26
DfmulB<
:
	full_text-
+
)%117 = fmul float %99, 0xBFAF6A7A40000000
'floatB

	full_text

	float %99
CcallB;
9
	full_text,
*
(%118 = call double @exp_i(float %117) #6
(floatB

	full_text


float %117
UcallBM
K
	full_text>
<
:%119 = call double @cmplx_mul(double %116, double %118) #6
*doubleB

	full_text

double %116
*doubleB

	full_text

double %118
MstoreBD
B
	full_text5
3
1store double %119, double* %26, align 8, !tbaa !9
*doubleB

	full_text

double %119
+double*B

	full_text

double* %26
NloadBF
D
	full_text7
5
3%120 = load double, double* %35, align 16, !tbaa !9
+double*B

	full_text

double* %35
DfmulB<
:
	full_text-
+
)%121 = fmul float %99, 0xBFA2D97C80000000
'floatB

	full_text

	float %99
CcallB;
9
	full_text,
*
(%122 = call double @exp_i(float %121) #6
(floatB

	full_text


float %121
UcallBM
K
	full_text>
<
:%123 = call double @cmplx_mul(double %120, double %122) #6
*doubleB

	full_text

double %120
*doubleB

	full_text

double %122
NstoreBE
C
	full_text6
4
2store double %123, double* %35, align 16, !tbaa !9
*doubleB

	full_text

double %123
+double*B

	full_text

double* %35
MloadBE
C
	full_text6
4
2%124 = load double, double* %44, align 8, !tbaa !9
+double*B

	full_text

double* %44
DfmulB<
:
	full_text-
+
)%125 = fmul float %99, 0xBFB5FDBC00000000
'floatB

	full_text

	float %99
CcallB;
9
	full_text,
*
(%126 = call double @exp_i(float %125) #6
(floatB

	full_text


float %125
UcallBM
K
	full_text>
<
:%127 = call double @cmplx_mul(double %124, double %126) #6
*doubleB

	full_text

double %124
*doubleB

	full_text

double %126
MstoreBD
B
	full_text5
3
1store double %127, double* %44, align 8, !tbaa !9
*doubleB

	full_text

double %127
+double*B

	full_text

double* %44
.andB'
%
	full_text

%128 = and i32 %4, 7
"i32B

	full_text


i32 %4
2shlB+
)
	full_text

%129 = shl nsw i32 %8, 3
"i32B

	full_text


i32 %8
1orB+
)
	full_text

%130 = or i32 %129, %128
$i32B

	full_text


i32 %129
$i32B

	full_text


i32 %128
6sextB.
,
	full_text

%131 = sext i32 %130 to i64
$i32B

	full_text


i32 %130
ÄgetelementptrBo
m
	full_text`
^
\%132 = getelementptr inbounds [576 x float], [576 x float]* @fft1D_512.smem, i64 0, i64 %131
$i64B

	full_text


i64 %131
_callBW
U
	full_textH
F
Dcall void @storex8(<2 x float>* nonnull %13, float* %132, i32 66) #6
5<2 x float>*B#
!
	full_text

<2 x float>* %13
*float*B

	full_text

float* %132
;callB3
1
	full_text$
"
 call void @_Z7barrierj(i32 1) #7
9mulB2
0
	full_text#
!
%133 = mul nuw nsw i32 %128, 66
$i32B

	full_text


i32 %128
5addB.
,
	full_text

%134 = add nsw i32 %133, %8
$i32B

	full_text


i32 %133
"i32B

	full_text


i32 %8
6sextB.
,
	full_text

%135 = sext i32 %134 to i64
$i32B

	full_text


i32 %134
ÄgetelementptrBo
m
	full_text`
^
\%136 = getelementptr inbounds [576 x float], [576 x float]* @fft1D_512.smem, i64 0, i64 %135
$i64B

	full_text


i64 %135
]callBU
S
	full_textF
D
Bcall void @loadx8(<2 x float>* nonnull %13, float* %136, i32 8) #6
5<2 x float>*B#
!
	full_text

<2 x float>* %13
*float*B

	full_text

float* %136
;callB3
1
	full_text$
"
 call void @_Z7barrierj(i32 1) #7
_callBW
U
	full_textH
F
Dcall void @storey8(<2 x float>* nonnull %13, float* %132, i32 66) #6
5<2 x float>*B#
!
	full_text

<2 x float>* %13
*float*B

	full_text

float* %132
;callB3
1
	full_text$
"
 call void @_Z7barrierj(i32 1) #7
]callBU
S
	full_textF
D
Bcall void @loady8(<2 x float>* nonnull %13, float* %136, i32 8) #6
5<2 x float>*B#
!
	full_text

<2 x float>* %13
*float*B

	full_text

float* %136
;callB3
1
	full_text$
"
 call void @_Z7barrierj(i32 1) #7
NloadBF
D
	full_text7
5
3%137 = load double, double* %14, align 16, !tbaa !9
+double*B

	full_text

double* %14
NloadBF
D
	full_text7
5
3%138 = load double, double* %17, align 16, !tbaa !9
+double*B

	full_text

double* %17
UcallBM
K
	full_text>
<
:%139 = call double @cmplx_add(double %137, double %138) #6
*doubleB

	full_text

double %137
*doubleB

	full_text

double %138
NstoreBE
C
	full_text6
4
2store double %139, double* %14, align 16, !tbaa !9
*doubleB

	full_text

double %139
+double*B

	full_text

double* %14
NloadBF
D
	full_text7
5
3%140 = load double, double* %17, align 16, !tbaa !9
+double*B

	full_text

double* %17
UcallBM
K
	full_text>
<
:%141 = call double @cmplx_sub(double %137, double %140) #6
*doubleB

	full_text

double %137
*doubleB

	full_text

double %140
NstoreBE
C
	full_text6
4
2store double %141, double* %17, align 16, !tbaa !9
*doubleB

	full_text

double %141
+double*B

	full_text

double* %17
MloadBE
C
	full_text6
4
2%142 = load double, double* %23, align 8, !tbaa !9
+double*B

	full_text

double* %23
MloadBE
C
	full_text6
4
2%143 = load double, double* %26, align 8, !tbaa !9
+double*B

	full_text

double* %26
UcallBM
K
	full_text>
<
:%144 = call double @cmplx_add(double %142, double %143) #6
*doubleB

	full_text

double %142
*doubleB

	full_text

double %143
MstoreBD
B
	full_text5
3
1store double %144, double* %23, align 8, !tbaa !9
*doubleB

	full_text

double %144
+double*B

	full_text

double* %23
MloadBE
C
	full_text6
4
2%145 = load double, double* %26, align 8, !tbaa !9
+double*B

	full_text

double* %26
UcallBM
K
	full_text>
<
:%146 = call double @cmplx_sub(double %142, double %145) #6
*doubleB

	full_text

double %142
*doubleB

	full_text

double %145
MstoreBD
B
	full_text5
3
1store double %146, double* %26, align 8, !tbaa !9
*doubleB

	full_text

double %146
+double*B

	full_text

double* %26
NloadBF
D
	full_text7
5
3%147 = load double, double* %32, align 16, !tbaa !9
+double*B

	full_text

double* %32
NloadBF
D
	full_text7
5
3%148 = load double, double* %35, align 16, !tbaa !9
+double*B

	full_text

double* %35
UcallBM
K
	full_text>
<
:%149 = call double @cmplx_add(double %147, double %148) #6
*doubleB

	full_text

double %147
*doubleB

	full_text

double %148
NstoreBE
C
	full_text6
4
2store double %149, double* %32, align 16, !tbaa !9
*doubleB

	full_text

double %149
+double*B

	full_text

double* %32
NloadBF
D
	full_text7
5
3%150 = load double, double* %35, align 16, !tbaa !9
+double*B

	full_text

double* %35
UcallBM
K
	full_text>
<
:%151 = call double @cmplx_sub(double %147, double %150) #6
*doubleB

	full_text

double %147
*doubleB

	full_text

double %150
NstoreBE
C
	full_text6
4
2store double %151, double* %35, align 16, !tbaa !9
*doubleB

	full_text

double %151
+double*B

	full_text

double* %35
MloadBE
C
	full_text6
4
2%152 = load double, double* %41, align 8, !tbaa !9
+double*B

	full_text

double* %41
MloadBE
C
	full_text6
4
2%153 = load double, double* %44, align 8, !tbaa !9
+double*B

	full_text

double* %44
UcallBM
K
	full_text>
<
:%154 = call double @cmplx_add(double %152, double %153) #6
*doubleB

	full_text

double %152
*doubleB

	full_text

double %153
MstoreBD
B
	full_text5
3
1store double %154, double* %41, align 8, !tbaa !9
*doubleB

	full_text

double %154
+double*B

	full_text

double* %41
MloadBE
C
	full_text6
4
2%155 = load double, double* %44, align 8, !tbaa !9
+double*B

	full_text

double* %44
UcallBM
K
	full_text>
<
:%156 = call double @cmplx_sub(double %152, double %155) #6
*doubleB

	full_text

double %152
*doubleB

	full_text

double %155
MstoreBD
B
	full_text5
3
1store double %156, double* %44, align 8, !tbaa !9
*doubleB

	full_text

double %156
+double*B

	full_text

double* %44
MloadBE
C
	full_text6
4
2%157 = load double, double* %26, align 8, !tbaa !9
+double*B

	full_text

double* %26
ccallB[
Y
	full_textL
J
H%158 = call double @cmplx_mul(double %157, double 0xBF8000003F800000) #6
*doubleB

	full_text

double %157
bcallBZ
X
	full_textK
I
G%159 = call double @cm_fl_mul(double %158, float 0x3FE6A09E60000000) #6
*doubleB

	full_text

double %158
MstoreBD
B
	full_text5
3
1store double %159, double* %26, align 8, !tbaa !9
*doubleB

	full_text

double %159
+double*B

	full_text

double* %26
NloadBF
D
	full_text7
5
3%160 = load double, double* %35, align 16, !tbaa !9
+double*B

	full_text

double* %35
^callBV
T
	full_textG
E
C%161 = call double @cmplx_mul(double %160, double -7.812500e-03) #6
*doubleB

	full_text

double %160
NstoreBE
C
	full_text6
4
2store double %161, double* %35, align 16, !tbaa !9
*doubleB

	full_text

double %161
+double*B

	full_text

double* %35
MloadBE
C
	full_text6
4
2%162 = load double, double* %44, align 8, !tbaa !9
+double*B

	full_text

double* %44
ccallB[
Y
	full_textL
J
H%163 = call double @cmplx_mul(double %162, double 0xBF800000BF800000) #6
*doubleB

	full_text

double %162
bcallBZ
X
	full_textK
I
G%164 = call double @cm_fl_mul(double %163, float 0x3FE6A09E60000000) #6
*doubleB

	full_text

double %163
MstoreBD
B
	full_text5
3
1store double %164, double* %44, align 8, !tbaa !9
*doubleB

	full_text

double %164
+double*B

	full_text

double* %44
NloadBF
D
	full_text7
5
3%165 = load double, double* %14, align 16, !tbaa !9
+double*B

	full_text

double* %14
NloadBF
D
	full_text7
5
3%166 = load double, double* %32, align 16, !tbaa !9
+double*B

	full_text

double* %32
UcallBM
K
	full_text>
<
:%167 = call double @cmplx_add(double %165, double %166) #6
*doubleB

	full_text

double %165
*doubleB

	full_text

double %166
NstoreBE
C
	full_text6
4
2store double %167, double* %14, align 16, !tbaa !9
*doubleB

	full_text

double %167
+double*B

	full_text

double* %14
NloadBF
D
	full_text7
5
3%168 = load double, double* %32, align 16, !tbaa !9
+double*B

	full_text

double* %32
UcallBM
K
	full_text>
<
:%169 = call double @cmplx_sub(double %165, double %168) #6
*doubleB

	full_text

double %165
*doubleB

	full_text

double %168
NstoreBE
C
	full_text6
4
2store double %169, double* %32, align 16, !tbaa !9
*doubleB

	full_text

double %169
+double*B

	full_text

double* %32
MloadBE
C
	full_text6
4
2%170 = load double, double* %23, align 8, !tbaa !9
+double*B

	full_text

double* %23
MloadBE
C
	full_text6
4
2%171 = load double, double* %41, align 8, !tbaa !9
+double*B

	full_text

double* %41
UcallBM
K
	full_text>
<
:%172 = call double @cmplx_add(double %170, double %171) #6
*doubleB

	full_text

double %170
*doubleB

	full_text

double %171
MstoreBD
B
	full_text5
3
1store double %172, double* %23, align 8, !tbaa !9
*doubleB

	full_text

double %172
+double*B

	full_text

double* %23
MloadBE
C
	full_text6
4
2%173 = load double, double* %41, align 8, !tbaa !9
+double*B

	full_text

double* %41
UcallBM
K
	full_text>
<
:%174 = call double @cmplx_sub(double %170, double %173) #6
*doubleB

	full_text

double %170
*doubleB

	full_text

double %173
MstoreBD
B
	full_text5
3
1store double %174, double* %41, align 8, !tbaa !9
*doubleB

	full_text

double %174
+double*B

	full_text

double* %41
^callBV
T
	full_textG
E
C%175 = call double @cmplx_mul(double %174, double -7.812500e-03) #6
*doubleB

	full_text

double %174
MstoreBD
B
	full_text5
3
1store double %175, double* %41, align 8, !tbaa !9
*doubleB

	full_text

double %175
+double*B

	full_text

double* %41
NloadBF
D
	full_text7
5
3%176 = load double, double* %14, align 16, !tbaa !9
+double*B

	full_text

double* %14
MloadBE
C
	full_text6
4
2%177 = load double, double* %23, align 8, !tbaa !9
+double*B

	full_text

double* %23
UcallBM
K
	full_text>
<
:%178 = call double @cmplx_add(double %176, double %177) #6
*doubleB

	full_text

double %176
*doubleB

	full_text

double %177
NstoreBE
C
	full_text6
4
2store double %178, double* %14, align 16, !tbaa !9
*doubleB

	full_text

double %178
+double*B

	full_text

double* %14
MloadBE
C
	full_text6
4
2%179 = load double, double* %23, align 8, !tbaa !9
+double*B

	full_text

double* %23
UcallBM
K
	full_text>
<
:%180 = call double @cmplx_sub(double %176, double %179) #6
*doubleB

	full_text

double %176
*doubleB

	full_text

double %179
MstoreBD
B
	full_text5
3
1store double %180, double* %23, align 8, !tbaa !9
*doubleB

	full_text

double %180
+double*B

	full_text

double* %23
NloadBF
D
	full_text7
5
3%181 = load double, double* %32, align 16, !tbaa !9
+double*B

	full_text

double* %32
MloadBE
C
	full_text6
4
2%182 = load double, double* %41, align 8, !tbaa !9
+double*B

	full_text

double* %41
UcallBM
K
	full_text>
<
:%183 = call double @cmplx_add(double %181, double %182) #6
*doubleB

	full_text

double %181
*doubleB

	full_text

double %182
NstoreBE
C
	full_text6
4
2store double %183, double* %32, align 16, !tbaa !9
*doubleB

	full_text

double %183
+double*B

	full_text

double* %32
MloadBE
C
	full_text6
4
2%184 = load double, double* %41, align 8, !tbaa !9
+double*B

	full_text

double* %41
UcallBM
K
	full_text>
<
:%185 = call double @cmplx_sub(double %181, double %184) #6
*doubleB

	full_text

double %181
*doubleB

	full_text

double %184
MstoreBD
B
	full_text5
3
1store double %185, double* %41, align 8, !tbaa !9
*doubleB

	full_text

double %185
+double*B

	full_text

double* %41
NloadBF
D
	full_text7
5
3%186 = load double, double* %17, align 16, !tbaa !9
+double*B

	full_text

double* %17
NloadBF
D
	full_text7
5
3%187 = load double, double* %35, align 16, !tbaa !9
+double*B

	full_text

double* %35
UcallBM
K
	full_text>
<
:%188 = call double @cmplx_add(double %186, double %187) #6
*doubleB

	full_text

double %186
*doubleB

	full_text

double %187
NstoreBE
C
	full_text6
4
2store double %188, double* %17, align 16, !tbaa !9
*doubleB

	full_text

double %188
+double*B

	full_text

double* %17
NloadBF
D
	full_text7
5
3%189 = load double, double* %35, align 16, !tbaa !9
+double*B

	full_text

double* %35
UcallBM
K
	full_text>
<
:%190 = call double @cmplx_sub(double %186, double %189) #6
*doubleB

	full_text

double %186
*doubleB

	full_text

double %189
NstoreBE
C
	full_text6
4
2store double %190, double* %35, align 16, !tbaa !9
*doubleB

	full_text

double %190
+double*B

	full_text

double* %35
MloadBE
C
	full_text6
4
2%191 = load double, double* %26, align 8, !tbaa !9
+double*B

	full_text

double* %26
MloadBE
C
	full_text6
4
2%192 = load double, double* %44, align 8, !tbaa !9
+double*B

	full_text

double* %44
UcallBM
K
	full_text>
<
:%193 = call double @cmplx_add(double %191, double %192) #6
*doubleB

	full_text

double %191
*doubleB

	full_text

double %192
MstoreBD
B
	full_text5
3
1store double %193, double* %26, align 8, !tbaa !9
*doubleB

	full_text

double %193
+double*B

	full_text

double* %26
MloadBE
C
	full_text6
4
2%194 = load double, double* %44, align 8, !tbaa !9
+double*B

	full_text

double* %44
UcallBM
K
	full_text>
<
:%195 = call double @cmplx_sub(double %191, double %194) #6
*doubleB

	full_text

double %191
*doubleB

	full_text

double %194
MstoreBD
B
	full_text5
3
1store double %195, double* %44, align 8, !tbaa !9
*doubleB

	full_text

double %195
+double*B

	full_text

double* %44
^callBV
T
	full_textG
E
C%196 = call double @cmplx_mul(double %195, double -7.812500e-03) #6
*doubleB

	full_text

double %195
MstoreBD
B
	full_text5
3
1store double %196, double* %44, align 8, !tbaa !9
*doubleB

	full_text

double %196
+double*B

	full_text

double* %44
NloadBF
D
	full_text7
5
3%197 = load double, double* %17, align 16, !tbaa !9
+double*B

	full_text

double* %17
MloadBE
C
	full_text6
4
2%198 = load double, double* %26, align 8, !tbaa !9
+double*B

	full_text

double* %26
UcallBM
K
	full_text>
<
:%199 = call double @cmplx_add(double %197, double %198) #6
*doubleB

	full_text

double %197
*doubleB

	full_text

double %198
NstoreBE
C
	full_text6
4
2store double %199, double* %17, align 16, !tbaa !9
*doubleB

	full_text

double %199
+double*B

	full_text

double* %17
MloadBE
C
	full_text6
4
2%200 = load double, double* %26, align 8, !tbaa !9
+double*B

	full_text

double* %26
UcallBM
K
	full_text>
<
:%201 = call double @cmplx_sub(double %197, double %200) #6
*doubleB

	full_text

double %197
*doubleB

	full_text

double %200
MstoreBD
B
	full_text5
3
1store double %201, double* %26, align 8, !tbaa !9
*doubleB

	full_text

double %201
+double*B

	full_text

double* %26
NloadBF
D
	full_text7
5
3%202 = load double, double* %35, align 16, !tbaa !9
+double*B

	full_text

double* %35
MloadBE
C
	full_text6
4
2%203 = load double, double* %44, align 8, !tbaa !9
+double*B

	full_text

double* %44
UcallBM
K
	full_text>
<
:%204 = call double @cmplx_add(double %202, double %203) #6
*doubleB

	full_text

double %202
*doubleB

	full_text

double %203
NstoreBE
C
	full_text6
4
2store double %204, double* %35, align 16, !tbaa !9
*doubleB

	full_text

double %204
+double*B

	full_text

double* %35
MloadBE
C
	full_text6
4
2%205 = load double, double* %44, align 8, !tbaa !9
+double*B

	full_text

double* %44
UcallBM
K
	full_text>
<
:%206 = call double @cmplx_sub(double %202, double %205) #6
*doubleB

	full_text

double %202
*doubleB

	full_text

double %205
MstoreBD
B
	full_text5
3
1store double %206, double* %44, align 8, !tbaa !9
*doubleB

	full_text

double %206
+double*B

	full_text

double* %44
:sitofpB0
.
	full_text!

%207 = sitofp i32 %8 to float
"i32B

	full_text


i32 %8
MloadBE
C
	full_text6
4
2%208 = load double, double* %23, align 8, !tbaa !9
+double*B

	full_text

double* %23
EfmulB=
;
	full_text.
,
*%209 = fmul float %207, 0xBFD921FB60000000
(floatB

	full_text


float %207
CcallB;
9
	full_text,
*
(%210 = call double @exp_i(float %209) #6
(floatB

	full_text


float %209
UcallBM
K
	full_text>
<
:%211 = call double @cmplx_mul(double %208, double %210) #6
*doubleB

	full_text

double %208
*doubleB

	full_text

double %210
MstoreBD
B
	full_text5
3
1store double %211, double* %23, align 8, !tbaa !9
*doubleB

	full_text

double %211
+double*B

	full_text

double* %23
NloadBF
D
	full_text7
5
3%212 = load double, double* %32, align 16, !tbaa !9
+double*B

	full_text

double* %32
EfmulB=
;
	full_text.
,
*%213 = fmul float %207, 0xBFC921FB60000000
(floatB

	full_text


float %207
CcallB;
9
	full_text,
*
(%214 = call double @exp_i(float %213) #6
(floatB

	full_text


float %213
UcallBM
K
	full_text>
<
:%215 = call double @cmplx_mul(double %212, double %214) #6
*doubleB

	full_text

double %212
*doubleB

	full_text

double %214
NstoreBE
C
	full_text6
4
2store double %215, double* %32, align 16, !tbaa !9
*doubleB

	full_text

double %215
+double*B

	full_text

double* %32
MloadBE
C
	full_text6
4
2%216 = load double, double* %41, align 8, !tbaa !9
+double*B

	full_text

double* %41
EfmulB=
;
	full_text.
,
*%217 = fmul float %207, 0xBFE2D97C80000000
(floatB

	full_text


float %207
CcallB;
9
	full_text,
*
(%218 = call double @exp_i(float %217) #6
(floatB

	full_text


float %217
UcallBM
K
	full_text>
<
:%219 = call double @cmplx_mul(double %216, double %218) #6
*doubleB

	full_text

double %216
*doubleB

	full_text

double %218
MstoreBD
B
	full_text5
3
1store double %219, double* %41, align 8, !tbaa !9
*doubleB

	full_text

double %219
+double*B

	full_text

double* %41
NloadBF
D
	full_text7
5
3%220 = load double, double* %17, align 16, !tbaa !9
+double*B

	full_text

double* %17
EfmulB=
;
	full_text.
,
*%221 = fmul float %207, 0xBFB921FB60000000
(floatB

	full_text


float %207
CcallB;
9
	full_text,
*
(%222 = call double @exp_i(float %221) #6
(floatB

	full_text


float %221
UcallBM
K
	full_text>
<
:%223 = call double @cmplx_mul(double %220, double %222) #6
*doubleB

	full_text

double %220
*doubleB

	full_text

double %222
NstoreBE
C
	full_text6
4
2store double %223, double* %17, align 16, !tbaa !9
*doubleB

	full_text

double %223
+double*B

	full_text

double* %17
MloadBE
C
	full_text6
4
2%224 = load double, double* %26, align 8, !tbaa !9
+double*B

	full_text

double* %26
EfmulB=
;
	full_text.
,
*%225 = fmul float %207, 0xBFDF6A7A40000000
(floatB

	full_text


float %207
CcallB;
9
	full_text,
*
(%226 = call double @exp_i(float %225) #6
(floatB

	full_text


float %225
UcallBM
K
	full_text>
<
:%227 = call double @cmplx_mul(double %224, double %226) #6
*doubleB

	full_text

double %224
*doubleB

	full_text

double %226
MstoreBD
B
	full_text5
3
1store double %227, double* %26, align 8, !tbaa !9
*doubleB

	full_text

double %227
+double*B

	full_text

double* %26
NloadBF
D
	full_text7
5
3%228 = load double, double* %35, align 16, !tbaa !9
+double*B

	full_text

double* %35
EfmulB=
;
	full_text.
,
*%229 = fmul float %207, 0xBFD2D97C80000000
(floatB

	full_text


float %207
CcallB;
9
	full_text,
*
(%230 = call double @exp_i(float %229) #6
(floatB

	full_text


float %229
UcallBM
K
	full_text>
<
:%231 = call double @cmplx_mul(double %228, double %230) #6
*doubleB

	full_text

double %228
*doubleB

	full_text

double %230
NstoreBE
C
	full_text6
4
2store double %231, double* %35, align 16, !tbaa !9
*doubleB

	full_text

double %231
+double*B

	full_text

double* %35
MloadBE
C
	full_text6
4
2%232 = load double, double* %44, align 8, !tbaa !9
+double*B

	full_text

double* %44
EfmulB=
;
	full_text.
,
*%233 = fmul float %207, 0xBFE5FDBC00000000
(floatB

	full_text


float %207
CcallB;
9
	full_text,
*
(%234 = call double @exp_i(float %233) #6
(floatB

	full_text


float %233
UcallBM
K
	full_text>
<
:%235 = call double @cmplx_mul(double %232, double %234) #6
*doubleB

	full_text

double %232
*doubleB

	full_text

double %234
MstoreBD
B
	full_text5
3
1store double %235, double* %44, align 8, !tbaa !9
*doubleB

	full_text

double %235
+double*B

	full_text

double* %44
_callBW
U
	full_textH
F
Dcall void @storex8(<2 x float>* nonnull %13, float* %132, i32 72) #6
5<2 x float>*B#
!
	full_text

<2 x float>* %13
*float*B

	full_text

float* %132
;callB3
1
	full_text$
"
 call void @_Z7barrierj(i32 1) #7
3mulB,
*
	full_text

%236 = mul nsw i32 %8, 72
"i32B

	full_text


i32 %8
1orB+
)
	full_text

%237 = or i32 %236, %128
$i32B

	full_text


i32 %236
$i32B

	full_text


i32 %128
6sextB.
,
	full_text

%238 = sext i32 %237 to i64
$i32B

	full_text


i32 %237
ÄgetelementptrBo
m
	full_text`
^
\%239 = getelementptr inbounds [576 x float], [576 x float]* @fft1D_512.smem, i64 0, i64 %238
$i64B

	full_text


i64 %238
]callBU
S
	full_textF
D
Bcall void @loadx8(<2 x float>* nonnull %13, float* %239, i32 8) #6
5<2 x float>*B#
!
	full_text

<2 x float>* %13
*float*B

	full_text

float* %239
;callB3
1
	full_text$
"
 call void @_Z7barrierj(i32 1) #7
_callBW
U
	full_textH
F
Dcall void @storey8(<2 x float>* nonnull %13, float* %132, i32 72) #6
5<2 x float>*B#
!
	full_text

<2 x float>* %13
*float*B

	full_text

float* %132
;callB3
1
	full_text$
"
 call void @_Z7barrierj(i32 1) #7
]callBU
S
	full_textF
D
Bcall void @loady8(<2 x float>* nonnull %13, float* %239, i32 8) #6
5<2 x float>*B#
!
	full_text

<2 x float>* %13
*float*B

	full_text

float* %239
NloadBF
D
	full_text7
5
3%240 = load double, double* %14, align 16, !tbaa !9
+double*B

	full_text

double* %14
NloadBF
D
	full_text7
5
3%241 = load double, double* %17, align 16, !tbaa !9
+double*B

	full_text

double* %17
UcallBM
K
	full_text>
<
:%242 = call double @cmplx_add(double %240, double %241) #6
*doubleB

	full_text

double %240
*doubleB

	full_text

double %241
NstoreBE
C
	full_text6
4
2store double %242, double* %14, align 16, !tbaa !9
*doubleB

	full_text

double %242
+double*B

	full_text

double* %14
NloadBF
D
	full_text7
5
3%243 = load double, double* %17, align 16, !tbaa !9
+double*B

	full_text

double* %17
UcallBM
K
	full_text>
<
:%244 = call double @cmplx_sub(double %240, double %243) #6
*doubleB

	full_text

double %240
*doubleB

	full_text

double %243
NstoreBE
C
	full_text6
4
2store double %244, double* %17, align 16, !tbaa !9
*doubleB

	full_text

double %244
+double*B

	full_text

double* %17
MloadBE
C
	full_text6
4
2%245 = load double, double* %23, align 8, !tbaa !9
+double*B

	full_text

double* %23
MloadBE
C
	full_text6
4
2%246 = load double, double* %26, align 8, !tbaa !9
+double*B

	full_text

double* %26
UcallBM
K
	full_text>
<
:%247 = call double @cmplx_add(double %245, double %246) #6
*doubleB

	full_text

double %245
*doubleB

	full_text

double %246
MstoreBD
B
	full_text5
3
1store double %247, double* %23, align 8, !tbaa !9
*doubleB

	full_text

double %247
+double*B

	full_text

double* %23
MloadBE
C
	full_text6
4
2%248 = load double, double* %26, align 8, !tbaa !9
+double*B

	full_text

double* %26
UcallBM
K
	full_text>
<
:%249 = call double @cmplx_sub(double %245, double %248) #6
*doubleB

	full_text

double %245
*doubleB

	full_text

double %248
MstoreBD
B
	full_text5
3
1store double %249, double* %26, align 8, !tbaa !9
*doubleB

	full_text

double %249
+double*B

	full_text

double* %26
NloadBF
D
	full_text7
5
3%250 = load double, double* %32, align 16, !tbaa !9
+double*B

	full_text

double* %32
NloadBF
D
	full_text7
5
3%251 = load double, double* %35, align 16, !tbaa !9
+double*B

	full_text

double* %35
UcallBM
K
	full_text>
<
:%252 = call double @cmplx_add(double %250, double %251) #6
*doubleB

	full_text

double %250
*doubleB

	full_text

double %251
NstoreBE
C
	full_text6
4
2store double %252, double* %32, align 16, !tbaa !9
*doubleB

	full_text

double %252
+double*B

	full_text

double* %32
NloadBF
D
	full_text7
5
3%253 = load double, double* %35, align 16, !tbaa !9
+double*B

	full_text

double* %35
UcallBM
K
	full_text>
<
:%254 = call double @cmplx_sub(double %250, double %253) #6
*doubleB

	full_text

double %250
*doubleB

	full_text

double %253
NstoreBE
C
	full_text6
4
2store double %254, double* %35, align 16, !tbaa !9
*doubleB

	full_text

double %254
+double*B

	full_text

double* %35
MloadBE
C
	full_text6
4
2%255 = load double, double* %41, align 8, !tbaa !9
+double*B

	full_text

double* %41
MloadBE
C
	full_text6
4
2%256 = load double, double* %44, align 8, !tbaa !9
+double*B

	full_text

double* %44
UcallBM
K
	full_text>
<
:%257 = call double @cmplx_add(double %255, double %256) #6
*doubleB

	full_text

double %255
*doubleB

	full_text

double %256
MstoreBD
B
	full_text5
3
1store double %257, double* %41, align 8, !tbaa !9
*doubleB

	full_text

double %257
+double*B

	full_text

double* %41
MloadBE
C
	full_text6
4
2%258 = load double, double* %44, align 8, !tbaa !9
+double*B

	full_text

double* %44
UcallBM
K
	full_text>
<
:%259 = call double @cmplx_sub(double %255, double %258) #6
*doubleB

	full_text

double %255
*doubleB

	full_text

double %258
MstoreBD
B
	full_text5
3
1store double %259, double* %44, align 8, !tbaa !9
*doubleB

	full_text

double %259
+double*B

	full_text

double* %44
MloadBE
C
	full_text6
4
2%260 = load double, double* %26, align 8, !tbaa !9
+double*B

	full_text

double* %26
ccallB[
Y
	full_textL
J
H%261 = call double @cmplx_mul(double %260, double 0xBF8000003F800000) #6
*doubleB

	full_text

double %260
bcallBZ
X
	full_textK
I
G%262 = call double @cm_fl_mul(double %261, float 0x3FE6A09E60000000) #6
*doubleB

	full_text

double %261
MstoreBD
B
	full_text5
3
1store double %262, double* %26, align 8, !tbaa !9
*doubleB

	full_text

double %262
+double*B

	full_text

double* %26
NloadBF
D
	full_text7
5
3%263 = load double, double* %35, align 16, !tbaa !9
+double*B

	full_text

double* %35
^callBV
T
	full_textG
E
C%264 = call double @cmplx_mul(double %263, double -7.812500e-03) #6
*doubleB

	full_text

double %263
NstoreBE
C
	full_text6
4
2store double %264, double* %35, align 16, !tbaa !9
*doubleB

	full_text

double %264
+double*B

	full_text

double* %35
MloadBE
C
	full_text6
4
2%265 = load double, double* %44, align 8, !tbaa !9
+double*B

	full_text

double* %44
ccallB[
Y
	full_textL
J
H%266 = call double @cmplx_mul(double %265, double 0xBF800000BF800000) #6
*doubleB

	full_text

double %265
bcallBZ
X
	full_textK
I
G%267 = call double @cm_fl_mul(double %266, float 0x3FE6A09E60000000) #6
*doubleB

	full_text

double %266
MstoreBD
B
	full_text5
3
1store double %267, double* %44, align 8, !tbaa !9
*doubleB

	full_text

double %267
+double*B

	full_text

double* %44
NloadBF
D
	full_text7
5
3%268 = load double, double* %14, align 16, !tbaa !9
+double*B

	full_text

double* %14
NloadBF
D
	full_text7
5
3%269 = load double, double* %32, align 16, !tbaa !9
+double*B

	full_text

double* %32
UcallBM
K
	full_text>
<
:%270 = call double @cmplx_add(double %268, double %269) #6
*doubleB

	full_text

double %268
*doubleB

	full_text

double %269
NstoreBE
C
	full_text6
4
2store double %270, double* %14, align 16, !tbaa !9
*doubleB

	full_text

double %270
+double*B

	full_text

double* %14
NloadBF
D
	full_text7
5
3%271 = load double, double* %32, align 16, !tbaa !9
+double*B

	full_text

double* %32
UcallBM
K
	full_text>
<
:%272 = call double @cmplx_sub(double %268, double %271) #6
*doubleB

	full_text

double %268
*doubleB

	full_text

double %271
NstoreBE
C
	full_text6
4
2store double %272, double* %32, align 16, !tbaa !9
*doubleB

	full_text

double %272
+double*B

	full_text

double* %32
MloadBE
C
	full_text6
4
2%273 = load double, double* %23, align 8, !tbaa !9
+double*B

	full_text

double* %23
MloadBE
C
	full_text6
4
2%274 = load double, double* %41, align 8, !tbaa !9
+double*B

	full_text

double* %41
UcallBM
K
	full_text>
<
:%275 = call double @cmplx_add(double %273, double %274) #6
*doubleB

	full_text

double %273
*doubleB

	full_text

double %274
MstoreBD
B
	full_text5
3
1store double %275, double* %23, align 8, !tbaa !9
*doubleB

	full_text

double %275
+double*B

	full_text

double* %23
MloadBE
C
	full_text6
4
2%276 = load double, double* %41, align 8, !tbaa !9
+double*B

	full_text

double* %41
UcallBM
K
	full_text>
<
:%277 = call double @cmplx_sub(double %273, double %276) #6
*doubleB

	full_text

double %273
*doubleB

	full_text

double %276
MstoreBD
B
	full_text5
3
1store double %277, double* %41, align 8, !tbaa !9
*doubleB

	full_text

double %277
+double*B

	full_text

double* %41
^callBV
T
	full_textG
E
C%278 = call double @cmplx_mul(double %277, double -7.812500e-03) #6
*doubleB

	full_text

double %277
MstoreBD
B
	full_text5
3
1store double %278, double* %41, align 8, !tbaa !9
*doubleB

	full_text

double %278
+double*B

	full_text

double* %41
NloadBF
D
	full_text7
5
3%279 = load double, double* %14, align 16, !tbaa !9
+double*B

	full_text

double* %14
MloadBE
C
	full_text6
4
2%280 = load double, double* %23, align 8, !tbaa !9
+double*B

	full_text

double* %23
UcallBM
K
	full_text>
<
:%281 = call double @cmplx_add(double %279, double %280) #6
*doubleB

	full_text

double %279
*doubleB

	full_text

double %280
NstoreBE
C
	full_text6
4
2store double %281, double* %14, align 16, !tbaa !9
*doubleB

	full_text

double %281
+double*B

	full_text

double* %14
MloadBE
C
	full_text6
4
2%282 = load double, double* %23, align 8, !tbaa !9
+double*B

	full_text

double* %23
UcallBM
K
	full_text>
<
:%283 = call double @cmplx_sub(double %279, double %282) #6
*doubleB

	full_text

double %279
*doubleB

	full_text

double %282
MstoreBD
B
	full_text5
3
1store double %283, double* %23, align 8, !tbaa !9
*doubleB

	full_text

double %283
+double*B

	full_text

double* %23
NloadBF
D
	full_text7
5
3%284 = load double, double* %32, align 16, !tbaa !9
+double*B

	full_text

double* %32
MloadBE
C
	full_text6
4
2%285 = load double, double* %41, align 8, !tbaa !9
+double*B

	full_text

double* %41
UcallBM
K
	full_text>
<
:%286 = call double @cmplx_add(double %284, double %285) #6
*doubleB

	full_text

double %284
*doubleB

	full_text

double %285
NstoreBE
C
	full_text6
4
2store double %286, double* %32, align 16, !tbaa !9
*doubleB

	full_text

double %286
+double*B

	full_text

double* %32
MloadBE
C
	full_text6
4
2%287 = load double, double* %41, align 8, !tbaa !9
+double*B

	full_text

double* %41
UcallBM
K
	full_text>
<
:%288 = call double @cmplx_sub(double %284, double %287) #6
*doubleB

	full_text

double %284
*doubleB

	full_text

double %287
MstoreBD
B
	full_text5
3
1store double %288, double* %41, align 8, !tbaa !9
*doubleB

	full_text

double %288
+double*B

	full_text

double* %41
NloadBF
D
	full_text7
5
3%289 = load double, double* %17, align 16, !tbaa !9
+double*B

	full_text

double* %17
NloadBF
D
	full_text7
5
3%290 = load double, double* %35, align 16, !tbaa !9
+double*B

	full_text

double* %35
UcallBM
K
	full_text>
<
:%291 = call double @cmplx_add(double %289, double %290) #6
*doubleB

	full_text

double %289
*doubleB

	full_text

double %290
NstoreBE
C
	full_text6
4
2store double %291, double* %17, align 16, !tbaa !9
*doubleB

	full_text

double %291
+double*B

	full_text

double* %17
NloadBF
D
	full_text7
5
3%292 = load double, double* %35, align 16, !tbaa !9
+double*B

	full_text

double* %35
UcallBM
K
	full_text>
<
:%293 = call double @cmplx_sub(double %289, double %292) #6
*doubleB

	full_text

double %289
*doubleB

	full_text

double %292
NstoreBE
C
	full_text6
4
2store double %293, double* %35, align 16, !tbaa !9
*doubleB

	full_text

double %293
+double*B

	full_text

double* %35
MloadBE
C
	full_text6
4
2%294 = load double, double* %26, align 8, !tbaa !9
+double*B

	full_text

double* %26
MloadBE
C
	full_text6
4
2%295 = load double, double* %44, align 8, !tbaa !9
+double*B

	full_text

double* %44
UcallBM
K
	full_text>
<
:%296 = call double @cmplx_add(double %294, double %295) #6
*doubleB

	full_text

double %294
*doubleB

	full_text

double %295
MstoreBD
B
	full_text5
3
1store double %296, double* %26, align 8, !tbaa !9
*doubleB

	full_text

double %296
+double*B

	full_text

double* %26
MloadBE
C
	full_text6
4
2%297 = load double, double* %44, align 8, !tbaa !9
+double*B

	full_text

double* %44
UcallBM
K
	full_text>
<
:%298 = call double @cmplx_sub(double %294, double %297) #6
*doubleB

	full_text

double %294
*doubleB

	full_text

double %297
MstoreBD
B
	full_text5
3
1store double %298, double* %44, align 8, !tbaa !9
*doubleB

	full_text

double %298
+double*B

	full_text

double* %44
^callBV
T
	full_textG
E
C%299 = call double @cmplx_mul(double %298, double -7.812500e-03) #6
*doubleB

	full_text

double %298
MstoreBD
B
	full_text5
3
1store double %299, double* %44, align 8, !tbaa !9
*doubleB

	full_text

double %299
+double*B

	full_text

double* %44
NloadBF
D
	full_text7
5
3%300 = load double, double* %17, align 16, !tbaa !9
+double*B

	full_text

double* %17
MloadBE
C
	full_text6
4
2%301 = load double, double* %26, align 8, !tbaa !9
+double*B

	full_text

double* %26
UcallBM
K
	full_text>
<
:%302 = call double @cmplx_add(double %300, double %301) #6
*doubleB

	full_text

double %300
*doubleB

	full_text

double %301
NstoreBE
C
	full_text6
4
2store double %302, double* %17, align 16, !tbaa !9
*doubleB

	full_text

double %302
+double*B

	full_text

double* %17
MloadBE
C
	full_text6
4
2%303 = load double, double* %26, align 8, !tbaa !9
+double*B

	full_text

double* %26
UcallBM
K
	full_text>
<
:%304 = call double @cmplx_sub(double %300, double %303) #6
*doubleB

	full_text

double %300
*doubleB

	full_text

double %303
MstoreBD
B
	full_text5
3
1store double %304, double* %26, align 8, !tbaa !9
*doubleB

	full_text

double %304
+double*B

	full_text

double* %26
NloadBF
D
	full_text7
5
3%305 = load double, double* %35, align 16, !tbaa !9
+double*B

	full_text

double* %35
MloadBE
C
	full_text6
4
2%306 = load double, double* %44, align 8, !tbaa !9
+double*B

	full_text

double* %44
UcallBM
K
	full_text>
<
:%307 = call double @cmplx_add(double %305, double %306) #6
*doubleB

	full_text

double %305
*doubleB

	full_text

double %306
NstoreBE
C
	full_text6
4
2store double %307, double* %35, align 16, !tbaa !9
*doubleB

	full_text

double %307
+double*B

	full_text

double* %35
MloadBE
C
	full_text6
4
2%308 = load double, double* %44, align 8, !tbaa !9
+double*B

	full_text

double* %44
UcallBM
K
	full_text>
<
:%309 = call double @cmplx_sub(double %305, double %308) #6
*doubleB

	full_text

double %305
*doubleB

	full_text

double %308
MstoreBD
B
	full_text5
3
1store double %309, double* %44, align 8, !tbaa !9
*doubleB

	full_text

double %309
+double*B

	full_text

double* %44
jcallBb
`
	full_textS
Q
Ocall void @globalStores8(<2 x float>* nonnull %13, <2 x float>* %12, i32 64) #6
5<2 x float>*B#
!
	full_text

<2 x float>* %13
5<2 x float>*B#
!
	full_text

<2 x float>* %12
WcallBO
M
	full_text@
>
<call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %9) #6
"i8*B

	full_text


i8* %9
"retB

	full_text


ret void
6<2 x float>*8B"
 
	full_text

<2 x float>* %0
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
-; undefined function B

	full_text

 
-; undefined function 	B

	full_text

 
-; undefined function 
B

	full_text

 
-; undefined function B

	full_text

 
-; undefined function B

	full_text

 
-; undefined function B

	full_text

 
-; undefined function B

	full_text

 
-; undefined function B

	full_text

 
-; undefined function B

	full_text

 
8float8B+
)
	full_text

float 0xBFD2D97C80000000
8float8B+
)
	full_text

float 0x3FE6A09E60000000
8float8B+
)
	full_text

float 0xBFC921FB60000000
#i648B

	full_text	

i64 0
8float8B+
)
	full_text

float 0xBFB5FDBC00000000
#i648B

	full_text	

i64 3
8float8B+
)
	full_text

float 0xBFDF6A7A40000000
$i328B

	full_text


i32 64
:double8B,
*
	full_text

double 0xBF800000BF800000
$i328B

	full_text


i32 72
$i648B

	full_text


i64 32
8float8B+
)
	full_text

float 0xBFE2D97C80000000
5double8B'
%
	full_text

double -7.812500e-03
#i648B

	full_text	

i64 6
#i328B

	full_text	

i32 3
#i648B

	full_text	

i64 5
8float8B+
)
	full_text

float 0xBFA921FB60000000
#i328B

	full_text	

i32 0
:double8B,
*
	full_text

double 0xBF8000003F800000
8float8B+
)
	full_text

float 0xBF9921FB60000000
#i648B

	full_text	

i64 4
8float8B+
)
	full_text

float 0xBFD921FB60000000
8float8B+
)
	full_text

float 0xBFB2D97C80000000
#i328B

	full_text	

i32 7
8float8B+
)
	full_text

float 0xBF8921FB60000000
$i328B

	full_text


i32 66
#i328B

	full_text	

i32 8
8float8B+
)
	full_text

float 0xBFB921FB60000000
#i648B

	full_text	

i64 1
#i648B

	full_text	

i64 2
#i648B

	full_text	

i64 7
8float8B+
)
	full_text

float 0xBFAF6A7A40000000
#i328B

	full_text	

i32 1
h[576 x float]*8BR
P
	full_textC
A
?@fft1D_512.smem = internal global [576 x float] undef, align 16
8float8B+
)
	full_text

float 0xBFE5FDBC00000000
#i648B

	full_text	

i64 9
$i648B

	full_text


i64 64
8float8B+
)
	full_text

float 0xBFA2D97C80000000       	 
                        !    "# "" $% $$ &' &( && )* )+ )) ,- ,, ./ .0 .. 12 13 11 45 44 67 66 89 88 :; :: <= << >? >> @A @B @@ CD CE CC FG FF HI HJ HH KL KM KK NO NN PQ PP RS RR TU TT VW VV XY XX Z[ Z\ ZZ ]^ ]_ ]] `a `` bc bd bb ef eg ee hi hh jk jj lm ll no nn pq pp rs rr tu tv tt wx wy ww z{ zz |} |~ || Ä 	Å  ÇÉ ÇÇ ÑÖ ÑÑ Üá ÜÜ àâ à
ä àà ãå ãã çé çç èê è
ë èè íì íí îï îî ñó ññ òô ò
ö òò õú õõ ùû ùù ü† ü
° üü ¢£ ¢
§ ¢¢ •¶ •• ß® ß
© ßß ™´ ™
¨ ™™ ≠Æ ≠≠ Ø∞ ØØ ±≤ ±
≥ ±± ¥µ ¥
∂ ¥¥ ∑∏ ∑∑ π∫ π
ª ππ ºΩ º
æ ºº ø¿ øø ¡¬ ¡
√ ¡¡ ƒ≈ ƒƒ ∆« ∆∆ »… »
  »» ÀÃ À
Õ ÀÀ Œœ ŒŒ –— –
“ –– ”‘ ”
’ ”” ÷◊ ÷÷ ÿŸ ÿÿ ⁄€ ⁄
‹ ⁄⁄ ›ﬁ ›
ﬂ ›› ‡· ‡‡ ‚„ ‚
‰ ‚‚ ÂÊ Â
Á ÂÂ ËÈ ËË ÍÎ ÍÍ ÏÌ Ï
Ó ÏÏ Ô Ô
Ò ÔÔ ÚÛ ÚÚ Ùı Ù
ˆ ÙÙ ˜¯ ˜
˘ ˜˜ ˙˚ ˙˙ ¸˝ ¸¸ ˛ˇ ˛
Ä ˛˛ ÅÇ Å
É ÅÅ ÑÖ ÑÑ Üá Ü
à ÜÜ âä â
ã ââ åç åå éè é
ê éé ëí ëë ìî ìì ïñ ï
ó ïï òô ò
ö òò õú õõ ùû ù
ü ùù †° †
¢ †† £§ ££ •¶ •• ß® ß
© ßß ™´ ™
¨ ™™ ≠Æ ≠≠ Ø∞ Ø
± ØØ ≤≥ ≤
¥ ≤≤ µ∂ µµ ∑∏ ∑∑ π∫ ππ ªº ªª Ωæ Ω
ø ΩΩ ¿¡ ¿
¬ ¿¿ √ƒ √√ ≈∆ ≈≈ «» «« …  …
À …… ÃÕ Ã
Œ ÃÃ œ– œœ —“ —— ”‘ ”” ’÷ ’
◊ ’’ ÿŸ ÿ
⁄ ÿÿ €‹ €€ ›ﬁ ›› ﬂ‡ ﬂﬂ ·‚ ·
„ ·· ‰Â ‰
Ê ‰‰ ÁË ÁÁ ÈÍ ÈÈ ÎÏ ÎÎ ÌÓ Ì
Ô ÌÌ Ò 
Ú  ÛÙ ÛÛ ıˆ ıı ˜¯ ˜˜ ˘˙ ˘
˚ ˘˘ ¸˝ ¸
˛ ¸¸ ˇÄ ˇˇ ÅÇ ÅÅ ÉÑ ÉÉ ÖÜ Ö
á ÖÖ àâ à
ä àà ãå ãã çé çç èê è
ë èè íì íí î
ï îî ñó ñ
ò ññ ôô öõ öö úù ú
û úú ü† üü °
¢ °° £§ £
• ££ ¶¶ ß® ß
© ßß ™™ ´¨ ´
≠ ´´ ÆÆ Ø∞ ØØ ±≤ ±± ≥¥ ≥
µ ≥≥ ∂∑ ∂
∏ ∂∂ π∫ ππ ªº ª
Ω ªª æø æ
¿ ææ ¡¬ ¡¡ √ƒ √√ ≈∆ ≈
« ≈≈ »… »
  »» ÀÃ ÀÀ ÕŒ Õ
œ ÕÕ –— –
“ –– ”‘ ”” ’÷ ’’ ◊ÿ ◊
Ÿ ◊◊ ⁄€ ⁄
‹ ⁄⁄ ›ﬁ ›› ﬂ‡ ﬂ
· ﬂﬂ ‚„ ‚
‰ ‚‚ ÂÊ ÂÂ ÁË ÁÁ ÈÍ È
Î ÈÈ ÏÌ Ï
Ó ÏÏ Ô ÔÔ ÒÚ Ò
Û ÒÒ Ùı Ù
ˆ ÙÙ ˜¯ ˜˜ ˘˙ ˘˘ ˚¸ ˚˚ ˝˛ ˝
ˇ ˝˝ ÄÅ ÄÄ ÇÉ ÇÇ ÑÖ Ñ
Ü ÑÑ áà áá âä ââ ãå ãã çé ç
è çç êë êê íì íí îï î
ñ îî óò ó
ô óó öõ öö úù ú
û úú ü† ü
° üü ¢£ ¢¢ §• §§ ¶ß ¶
® ¶¶ ©™ ©
´ ©© ¨≠ ¨¨ ÆØ Æ
∞ ÆÆ ±≤ ±
≥ ±± ¥µ ¥¥ ∂∑ ∂
∏ ∂∂ π∫ ππ ªº ªª Ωæ Ω
ø ΩΩ ¿¡ ¿
¬ ¿¿ √ƒ √√ ≈∆ ≈
« ≈≈ »… »
  »» ÀÃ ÀÀ ÕŒ ÕÕ œ– œ
— œœ “” “
‘ ““ ’÷ ’’ ◊ÿ ◊
Ÿ ◊◊ ⁄€ ⁄
‹ ⁄⁄ ›ﬁ ›› ﬂ‡ ﬂﬂ ·‚ ·
„ ·· ‰Â ‰
Ê ‰‰ ÁË ÁÁ ÈÍ È
Î ÈÈ ÏÌ Ï
Ó ÏÏ Ô ÔÔ ÒÚ ÒÒ ÛÙ Û
ı ÛÛ ˆ˜ ˆ
¯ ˆˆ ˘˙ ˘˘ ˚¸ ˚
˝ ˚˚ ˛ˇ ˛
Ä ˛˛ ÅÇ ÅÅ ÉÑ É
Ö ÉÉ Üá ÜÜ àâ àà äã ä
å ää çé ç
è çç êë êê íì í
î íí ïñ ï
ó ïï òô òò öõ öö úù ú
û úú ü† ü
° üü ¢£ ¢¢ §• §
¶ §§ ß® ß
© ßß ™´ ™™ ¨≠ ¨¨ ÆØ ÆÆ ∞± ∞∞ ≤≥ ≤
¥ ≤≤ µ∂ µ
∑ µµ ∏π ∏∏ ∫ª ∫∫ ºΩ ºº æø æ
¿ ææ ¡¬ ¡
√ ¡¡ ƒ≈ ƒƒ ∆« ∆∆ »… »»  À  
Ã    ÕŒ Õ
œ ÕÕ –— –– “” ““ ‘’ ‘‘ ÷◊ ÷
ÿ ÷÷ Ÿ⁄ Ÿ
€ ŸŸ ‹› ‹‹ ﬁﬂ ﬁﬁ ‡· ‡‡ ‚„ ‚
‰ ‚‚ ÂÊ Â
Á ÂÂ ËÈ ËË ÍÎ ÍÍ ÏÌ ÏÏ ÓÔ Ó
 ÓÓ ÒÚ Ò
Û ÒÒ Ùı ÙÙ ˆ˜ ˆˆ ¯˘ ¯¯ ˙˚ ˙
¸ ˙˙ ˝˛ ˝
ˇ ˝˝ ÄÅ Ä
Ç ÄÄ ÉÉ ÑÖ ÑÑ Üá Ü
à ÜÜ âä ââ ã
å ãã çé ç
è çç êê ëí ë
ì ëë îî ïñ ï
ó ïï òô òò öõ öö úù ú
û úú ü† ü
° üü ¢£ ¢¢ §• §
¶ §§ ß® ß
© ßß ™´ ™™ ¨≠ ¨¨ ÆØ Æ
∞ ÆÆ ±≤ ±
≥ ±± ¥µ ¥¥ ∂∑ ∂
∏ ∂∂ π∫ π
ª ππ ºΩ ºº æø ææ ¿¡ ¿
¬ ¿¿ √ƒ √
≈ √√ ∆« ∆∆ »… »
  »» ÀÃ À
Õ ÀÀ Œœ ŒŒ –— –– “” “
‘ ““ ’÷ ’
◊ ’’ ÿŸ ÿÿ ⁄€ ⁄
‹ ⁄⁄ ›ﬁ ›
ﬂ ›› ‡· ‡‡ ‚„ ‚‚ ‰Â ‰‰ ÊÁ Ê
Ë ÊÊ ÈÍ ÈÈ ÎÏ ÎÎ ÌÓ Ì
Ô ÌÌ Ò  ÚÛ ÚÚ Ùı ÙÙ ˆ˜ ˆ
¯ ˆˆ ˘˙ ˘˘ ˚¸ ˚˚ ˝˛ ˝
ˇ ˝˝ ÄÅ Ä
Ç ÄÄ ÉÑ ÉÉ ÖÜ Ö
á ÖÖ àâ à
ä àà ãå ãã çé çç èê è
ë èè íì í
î íí ïñ ïï óò ó
ô óó öõ ö
ú öö ùû ùù ü† ü
° üü ¢£ ¢¢ §• §§ ¶ß ¶
® ¶¶ ©™ ©
´ ©© ¨≠ ¨¨ ÆØ Æ
∞ ÆÆ ±≤ ±
≥ ±± ¥µ ¥¥ ∂∑ ∂∂ ∏π ∏
∫ ∏∏ ªº ª
Ω ªª æø ææ ¿¡ ¿
¬ ¿¿ √ƒ √
≈ √√ ∆« ∆∆ »… »»  À  
Ã    ÕŒ Õ
œ ÕÕ –— –– “” “
‘ ““ ’÷ ’
◊ ’’ ÿŸ ÿÿ ⁄€ ⁄⁄ ‹› ‹
ﬁ ‹‹ ﬂ‡ ﬂ
· ﬂﬂ ‚„ ‚‚ ‰Â ‰
Ê ‰‰ ÁË Á
È ÁÁ ÍÎ ÍÍ ÏÌ Ï
Ó ÏÏ Ô ÔÔ ÒÚ ÒÒ ÛÙ Û
ı ÛÛ ˆ˜ ˆ
¯ ˆˆ ˘˙ ˘˘ ˚¸ ˚
˝ ˚˚ ˛ˇ ˛
Ä ˛˛ ÅÇ ÅÅ ÉÑ ÉÉ ÖÜ Ö
á ÖÖ àâ à
ä àà ãå ãã çé ç
è çç êë ê
í êê ìî ì
ï ìì ñ
ó ññ òô    	 
            !  #" % '$ (& * +" - /, 0. 2" 3 54 76 9 ;: =< ?8 A> B@ D6 E< G8 IF JH L< M ON QP S UT WV YR [X \Z ^P _V aR c` db fV g ih kj m on qp sl ur vt xj yp {l }z ~| Äp Å< ÉÇ ÖÑ áÜ â< äV åã éç êV ëp ìí ïî óñ ôp ö úP ûõ †ù °ü £ §P ¶õ ®• ©ß ´P ¨6 Æj ∞≠ ≤Ø ≥± µ6 ∂j ∏≠ ∫∑ ªπ Ωj æπ ¿ø ¬j √ ≈6 «ƒ …∆  » Ã Õ6 œƒ —Œ “– ‘6 ’P ◊j Ÿ÷ €ÿ ‹⁄ ﬁP ﬂj ·÷ „‡ ‰‚ Êj Á" ÈV ÎË ÌÍ ÓÏ " ÒV ÛË ıÚ ˆÙ ¯V ˘< ˚p ˝˙ ˇ¸ Ä˛ Ç< Ép Ö˙ áÑ àÜ äp ãÜ çå èp ê" í< îë ñì óï ô" ö< úë ûõ üù °< ¢V §p ¶£ ®• ©ß ´V ¨p Æ£ ∞≠ ±Ø ≥p ¥ ∂6 ∏µ ∫π º∑ æª øΩ ¡6 ¬P ƒµ ∆≈ »√  « À… ÕP Œj –µ “— ‘œ ÷” ◊’ Ÿj ⁄" ‹µ ﬁ› ‡€ ‚ﬂ „· Â" Ê< Ëµ ÍÈ ÏÁ ÓÎ ÔÌ Ò< ÚV Ùµ ˆı ¯Û ˙˜ ˚˘ ˝V ˛p Äµ ÇÅ Ñˇ ÜÉ áÖ âp ä å éç êã ëè ìí ï óî òã õö ù ûú †ü ¢ §° • ®î © ¨° ≠ ∞" ≤Ø ¥± µ≥ ∑ ∏" ∫Ø ºπ Ωª ø" ¿6 ¬< ƒ¡ ∆√ «≈ …6  < Ã¡ ŒÀ œÕ —< “P ‘V ÷” ÿ’ Ÿ◊ €P ‹V ﬁ” ‡› ·ﬂ „V ‰j Êp ËÂ ÍÁ ÎÈ Ìj Óp Â ÚÔ ÛÒ ıp ˆ< ¯˜ ˙˘ ¸˚ ˛< ˇV ÅÄ ÉÇ ÖV Üp àá äâ åã ép è ëP ìê ïí ñî ò ôP õê ùö ûú †P °6 £j •¢ ß§ ®¶ ™6 ´j ≠¢ Ø¨ ∞Æ ≤j ≥Æ µ¥ ∑j ∏ ∫6 ºπ æª øΩ ¡ ¬6 ƒπ ∆√ «≈ …6  P Ãj ŒÀ –Õ —œ ”P ‘j ÷À ÿ’ Ÿ◊ €j ‹" ﬁV ‡› ‚ﬂ „· Â" ÊV Ë› ÍÁ ÎÈ ÌV Ó< p ÚÔ ÙÒ ıÛ ˜< ¯p ˙Ô ¸˘ ˝˚ ˇp Ä˚ ÇÅ Ñp Ö" á< âÜ ãà åä é" è< ëÜ ìê îí ñ< óV ôp õò ùö ûú †V °p £ò •¢ ¶§ ®p © ´6 ≠™ ØÆ ±¨ ≥∞ ¥≤ ∂6 ∑P π™ ª∫ Ω∏ øº ¿æ ¬P √j ≈™ «∆ …ƒ À» Ã  Œj œ" —™ ”“ ’– ◊‘ ÿ÷ ⁄" €< ›™ ﬂﬁ ·‹ „‡ ‰‚ Ê< ÁV È™ ÎÍ ÌË ÔÏ Ó ÚV Ûp ı™ ˜ˆ ˘Ù ˚¯ ¸˙ ˛p ˇ Åî Ç ÖÑ áã àÜ äâ å éã è íî ì ñã ó ô" õò ùö ûú † °" £ò •¢ ¶§ ®" ©6 ´< ≠™ Ø¨ ∞Æ ≤6 ≥< µ™ ∑¥ ∏∂ ∫< ªP ΩV øº ¡æ ¬¿ ƒP ≈V «º …∆  » ÃV Õj œp —Œ ”– ‘“ ÷j ◊p ŸŒ €ÿ ‹⁄ ﬁp ﬂ< ·‡ „‚ Â‰ Á< ËV ÍÈ ÏÎ ÓV Ôp Ò ÛÚ ıÙ ˜p ¯ ˙P ¸˘ ˛˚ ˇ˝ Å ÇP Ñ˘ ÜÉ áÖ âP ä6 åj éã êç ëè ì6 îj ñã òï ôó õj úó ûù †j ° £6 •¢ ß§ ®¶ ™ ´6 ≠¢ Ø¨ ∞Æ ≤6 ≥P µj ∑¥ π∂ ∫∏ ºP Ωj ø¥ ¡æ ¬¿ ƒj ≈" «V …∆ À» Ã  Œ" œV —∆ ”– ‘“ ÷V ◊< Ÿp €ÿ ›⁄ ﬁ‹ ‡< ·p „ÿ Â‚ Ê‰ Ëp È‰ ÎÍ Ìp Ó" < ÚÔ ÙÒ ıÛ ˜" ¯< ˙Ô ¸˘ ˝˚ ˇ< ÄV Çp ÑÅ ÜÉ áÖ âV äp åÅ éã èç ëp í î ï ó üü ®® úú ££ ßß ¢¢ ò ¶¶ öö õõ §§ ùù •• ûû †† ©© °°æ ¢¢ æ÷ ¢¢ ÷ öö È üü È  ûû  Û ûû ÛÑ ¢¢ Ñì ©© ìÙ °° ÙÈ ûû È∏ ûû ∏Ü °° ÜÕ üü Õ¶ ûû ¶˚ üü ˚ﬂ ££ ﬂ§ üü §¿ üü ¿Ò üü Òî ¢¢ îﬂ üü ﬂÎ ¢¢ Îî ûû î‚ ¢¢ ‚» ûû »ê •• êë ßß ëè ûû è“ üü “É •• É˘ ¢¢ ˘Æ üü Æ¯ ££ ¯ü ûû üœ ûû œä ûû ä´ ®® ´Ç ¢¢ Ç™ •• ™Æ üü Æ. üü .˛ ûû ˛ñ §§ ñÓ ¢¢ Ó∞ ££ ∞– üü –ã °° ã± ûû ±¶ •• ¶Ï ûû Ïï ®® ï§ üü §‡ ££ ‡ç ¶¶ ç⁄ üü ⁄≈ ûû ≈Ö üü Öº ££ º˚ °° ˚≤ ¢¢ ≤Z ûû Zô •• ôπ üü π˜ ££ ˜Ä §§ Ä∂ üü ∂» üü »◊ üü ◊⁄ ûû ⁄ª üü ª@ ûû @å ¢¢ åï ûû ï¿ ûû ¿Ö ûû Ö… ¢¢ …ñ °° ñù ¢¢ ù õõ  ùù Ø üü Ø˙ ¢¢ ˙| üü |â ¢¢ â‰ üü ‰ úú ¥ ¢¢ ¥Æ ûû ÆÏ ££ Ïù üü ùú ûû úú üü ú‰ °° ‰ñ †† ñ˚ üü ˚· ¢¢ ·  ¢¢  Û ûû ÛÉ ££ É≥ ûû ≥ç ¢¢ ç‚ ¢¢ ‚H üü Hí üü íÌ ¢¢ Ìç üü çú ûû ú· ûû ·Í ¢¢ Íb üü bª ££ ª‘ ££ ‘˝ ûû ˝Ω ¢¢ Ω& ûû &ß ûû ß’ ¢¢ ’£ ¶¶ £¶ ûû ¶ó üü ó‹ ûû ‹î •• î‚ üü ‚Î ££ Î◊ ûû ◊˘ ¢¢ ˘ß üü ß« ££ «Ö ¢¢ Ö“ ûû “Æ •• ÆÚ ¢¢ ÚÅ ¢¢ Åt ûû t» ££ »ß ßß ß” ££ ”Ü üü Ü≈ üü ≈ø ¢¢ øÙ üü ÙΩ ûû Ω
™ Í
´ Ü
´ ñ
´ ˚
´ ã
´ ‰
´ Ù
¨ ∫	≠ 	≠ 	≠  	≠ 4	≠ :	≠ N	≠ T	≠ h	≠ n
≠ î
≠ °
≠ ã
Æ Å	Ø h
∞ ﬁ	± 
± ì
≤ î
≤ â
≤ Ú
≥ Ä
≥ Ñ
≥ ë	¥ 	¥ 
µ ∆
∂ ç
∂ ø
∂ å
∂ Ç
∂ ¥
∂ Å
∂ Î
∂ ù
∂ Í	∑ T	∏ 
∏ ç	π :
∫ πª ª 
º Ñ
º ˘
º ‚
Ω ≈	æ  
ø Æ
¿ —
¡ ã
¬ ›
√ ñ
√ ö
√ ß
ƒ £
ƒ ´
ƒ ç
ƒ ï
≈ “	∆ 4	« N	» n
… È    ô  ¶  ™  Æ  É  ê  îÀ îÀ °À ã
Ã ˆ	Õ Œ Œ ñ
œ ı"
	fft1D_512"
llvm.lifetime.start.p0i8"
_Z12get_local_idj"
_Z12get_group_idj"
globalLoads8"
	cmplx_add"
	cmplx_sub"
llvm.lifetime.end.p0i8"
	cm_fl_mul"
	cmplx_mul"
exp_i"	
storex8"
_Z7barrierj"
loadx8"	
storey8"
loady8"
globalStores8*î
shoc-1.1.5-FFT-fft1D_512.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282Ä
 
transfer_bytes_log1p
´yzA

transfer_bytes
àÄÄ

devmap_label
 

wgsize_log1p
´yzA

wgsize
@