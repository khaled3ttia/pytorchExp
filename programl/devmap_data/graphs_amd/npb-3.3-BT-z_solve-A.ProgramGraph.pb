

[external]
EallocaB;
9
	full_text,
*
(%6 = alloca [5 x [5 x double]], align 16
EallocaB;
9
	full_text,
*
(%7 = alloca [5 x [5 x double]], align 16
EallocaB;
9
	full_text,
*
(%8 = alloca [5 x [5 x double]], align 16
?allocaB5
3
	full_text&
$
"%9 = alloca [5 x double], align 16
FallocaB<
:
	full_text-
+
)%10 = alloca [5 x [5 x double]], align 16
@allocaB6
4
	full_text'
%
#%11 = alloca [5 x double], align 16
DbitcastB9
7
	full_text*
(
&%12 = bitcast [5 x double]* %11 to i8*
7[5 x double]*B$
"
	full_text

[5 x double]* %11
IbitcastB>
<
	full_text/
-
+%13 = bitcast [5 x [5 x double]]* %6 to i8*
B[5 x [5 x double]]*B)
'
	full_text

[5 x [5 x double]]* %6
[callBS
Q
	full_textD
B
@call void @llvm.lifetime.start.p0i8(i64 200, i8* nonnull %13) #5
#i8*B

	full_text
	
i8* %13
IbitcastB>
<
	full_text/
-
+%14 = bitcast [5 x [5 x double]]* %7 to i8*
B[5 x [5 x double]]*B)
'
	full_text

[5 x [5 x double]]* %7
[callBS
Q
	full_textD
B
@call void @llvm.lifetime.start.p0i8(i64 200, i8* nonnull %14) #5
#i8*B

	full_text
	
i8* %14
IbitcastB>
<
	full_text/
-
+%15 = bitcast [5 x [5 x double]]* %8 to i8*
B[5 x [5 x double]]*B)
'
	full_text

[5 x [5 x double]]* %8
[callBS
Q
	full_textD
B
@call void @llvm.lifetime.start.p0i8(i64 200, i8* nonnull %15) #5
#i8*B

	full_text
	
i8* %15
CbitcastB8
6
	full_text)
'
%%16 = bitcast [5 x double]* %9 to i8*
6[5 x double]*B#
!
	full_text

[5 x double]* %9
ZcallBR
P
	full_textC
A
?call void @llvm.lifetime.start.p0i8(i64 40, i8* nonnull %16) #5
#i8*B

	full_text
	
i8* %16
JbitcastB?
=
	full_text0
.
,%17 = bitcast [5 x [5 x double]]* %10 to i8*
C[5 x [5 x double]]*B*
(
	full_text

[5 x [5 x double]]* %10
[callBS
Q
	full_textD
B
@call void @llvm.lifetime.start.p0i8(i64 200, i8* nonnull %17) #5
#i8*B

	full_text
	
i8* %17
ZcallBR
P
	full_textC
A
?call void @llvm.lifetime.start.p0i8(i64 40, i8* nonnull %12) #5
#i8*B

	full_text
	
i8* %12
LcallBD
B
	full_text5
3
1%18 = tail call i64 @_Z13get_global_idj(i32 1) #6
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
1%21 = tail call i64 @_Z13get_global_idj(i32 0) #6
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
2addB+
)
	full_text

%24 = add nsw i32 %3, -2
6icmpB.
,
	full_text

%25 = icmp slt i32 %24, %20
#i32B

	full_text
	
i32 %24
#i32B

	full_text
	
i32 %20
9brB3
1
	full_text$
"
 br i1 %25, label %204, label %26
!i1B

	full_text


i1 %25
4add8B+
)
	full_text

%27 = add nsw i32 %2, -2
8icmp8B.
,
	full_text

%28 = icmp slt i32 %27, %23
%i328B

	full_text
	
i32 %27
%i328B

	full_text
	
i32 %23
;br8B3
1
	full_text$
"
 br i1 %28, label %204, label %29
#i18B

	full_text


i1 %28
Wbitcast8BJ
H
	full_text;
9
7%30 = bitcast double* %0 to [65 x [65 x [5 x double]]]*
5add8B,
*
	full_text

%31 = add nsw i32 %20, -1
%i328B

	full_text
	
i32 %20
6mul8B-
+
	full_text

%32 = mul nsw i32 %31, %27
%i328B

	full_text
	
i32 %31
%i328B

	full_text
	
i32 %27
5add8B,
*
	full_text

%33 = add nsw i32 %23, -1
%i328B

	full_text
	
i32 %23
6add8B-
+
	full_text

%34 = add nsw i32 %33, %32
%i328B

	full_text
	
i32 %33
%i328B

	full_text
	
i32 %32
3mul8B*
(
	full_text

%35 = mul i32 %34, 4875
%i328B

	full_text
	
i32 %34
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
8%37 = getelementptr inbounds double, double* %1, i64 %36
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
0add8B'
%
	full_text

%39 = add i32 %4, -1
{getelementptr8Bh
f
	full_textY
W
U%40 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %7, i64 0, i64 0
D[5 x [5 x double]]*8B)
'
	full_text

[5 x [5 x double]]* %7
^getelementptr8BK
I
	full_text<
:
8%41 = getelementptr inbounds double, double* %37, i64 25
-double*8B

	full_text

double* %37
Jbitcast8B=
;
	full_text.
,
*%42 = bitcast double* %41 to [5 x double]*
-double*8B

	full_text

double* %41
dcall8BZ
X
	full_textK
I
Gcall void @load_matrix([5 x double]* nonnull %40, [5 x double]* %42) #5
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
9[5 x double]*8B$
"
	full_text

[5 x double]* %42
{getelementptr8Bh
f
	full_textY
W
U%43 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %8, i64 0, i64 0
D[5 x [5 x double]]*8B)
'
	full_text

[5 x [5 x double]]* %8
^getelementptr8BK
I
	full_text<
:
8%44 = getelementptr inbounds double, double* %37, i64 50
-double*8B

	full_text

double* %37
Jbitcast8B=
;
	full_text.
,
*%45 = bitcast double* %44 to [5 x double]*
-double*8B

	full_text

double* %44
dcall8BZ
X
	full_textK
I
Gcall void @load_matrix([5 x double]* nonnull %43, [5 x double]* %45) #5
9[5 x double]*8B$
"
	full_text

[5 x double]* %43
9[5 x double]*8B$
"
	full_text

[5 x double]* %45
ogetelementptr8B\
Z
	full_textM
K
I%46 = getelementptr inbounds [5 x double], [5 x double]* %9, i64 0, i64 0
8[5 x double]*8B#
!
	full_text

[5 x double]* %9
1shl8B(
&
	full_text

%47 = shl i64 %19, 32
%i648B

	full_text
	
i64 %19
9ashr8B/
-
	full_text 

%48 = ashr exact i64 %47, 32
%i648B

	full_text
	
i64 %47
1shl8B(
&
	full_text

%49 = shl i64 %22, 32
%i648B

	full_text
	
i64 %22
9ashr8B/
-
	full_text 

%50 = ashr exact i64 %49, 32
%i648B

	full_text
	
i64 %49
†getelementptr8Bå
â
	full_text|
z
x%51 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %30, i64 0, i64 %48, i64 %50, i64 0
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %30
%i648B

	full_text
	
i64 %48
%i648B

	full_text
	
i64 %50
Xcall8BN
L
	full_text?
=
;call void @load_vector(double* nonnull %46, double* %51) #5
-double*8B

	full_text

double* %46
-double*8B

	full_text

double* %51
Äcall8Bv
t
	full_textg
e
ccall void @p_binvcrhs([5 x double]* nonnull %40, [5 x double]* nonnull %43, double* nonnull %46) #5
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
9[5 x double]*8B$
"
	full_text

[5 x double]* %43
-double*8B

	full_text

double* %46
dcall8BZ
X
	full_textK
I
Gcall void @save_matrix([5 x double]* %42, [5 x double]* nonnull %40) #5
9[5 x double]*8B$
"
	full_text

[5 x double]* %42
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
dcall8BZ
X
	full_textK
I
Gcall void @save_matrix([5 x double]* %45, [5 x double]* nonnull %43) #5
9[5 x double]*8B$
"
	full_text

[5 x double]* %45
9[5 x double]*8B$
"
	full_text

[5 x double]* %43
Xcall8BN
L
	full_text?
=
;call void @save_vector(double* %51, double* nonnull %46) #5
-double*8B

	full_text

double* %51
-double*8B

	full_text

double* %46
4add8B+
)
	full_text

%52 = add nsw i32 %4, -2
5icmp8B+
)
	full_text

%53 = icmp slt i32 %4, 3
|getelementptr8Bi
g
	full_textZ
X
V%54 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %10, i64 0, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %10
lcall8Bb
`
	full_textS
Q
Ocall void @copy_matrix([5 x double]* nonnull %54, [5 x double]* nonnull %43) #5
9[5 x double]*8B$
"
	full_text

[5 x double]* %54
9[5 x double]*8B$
"
	full_text

[5 x double]* %43
pgetelementptr8B]
[
	full_textN
L
J%55 = getelementptr inbounds [5 x double], [5 x double]* %11, i64 0, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %11
`call8BV
T
	full_textG
E
Ccall void @copy_vector(double* nonnull %55, double* nonnull %46) #5
-double*8B

	full_text

double* %55
-double*8B

	full_text

double* %46
{getelementptr8Bh
f
	full_textY
W
U%56 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %6, i64 0, i64 0
D[5 x [5 x double]]*8B)
'
	full_text

[5 x [5 x double]]* %6
:br8B2
0
	full_text#
!
br i1 %53, label %67, label %57
#i18B

	full_text


i1 %53
6zext8B,
*
	full_text

%58 = zext i32 %39 to i64
%i328B

	full_text
	
i32 %39
'br8B

	full_text

br label %59
Bphi8B9
7
	full_text*
(
&%60 = phi i64 [ %65, %59 ], [ 1, %57 ]
%i648B

	full_text
	
i64 %65
ëgetelementptr8B~
|
	full_texto
m
k%61 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %60, i64 0, i64 0
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %60
dcall8BZ
X
	full_textK
I
Gcall void @load_matrix([5 x double]* nonnull %56, [5 x double]* %61) #5
9[5 x double]*8B$
"
	full_text

[5 x double]* %56
9[5 x double]*8B$
"
	full_text

[5 x double]* %61
ëgetelementptr8B~
|
	full_texto
m
k%62 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %60, i64 1, i64 0
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %60
dcall8BZ
X
	full_textK
I
Gcall void @load_matrix([5 x double]* nonnull %40, [5 x double]* %62) #5
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
9[5 x double]*8B$
"
	full_text

[5 x double]* %62
ëgetelementptr8B~
|
	full_texto
m
k%63 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %60, i64 2, i64 0
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %60
dcall8BZ
X
	full_textK
I
Gcall void @load_matrix([5 x double]* nonnull %43, [5 x double]* %63) #5
9[5 x double]*8B$
"
	full_text

[5 x double]* %43
9[5 x double]*8B$
"
	full_text

[5 x double]* %63
¢getelementptr8Bé
ã
	full_text~
|
z%64 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %30, i64 %60, i64 %48, i64 %50, i64 0
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %30
%i648B

	full_text
	
i64 %60
%i648B

	full_text
	
i64 %48
%i648B

	full_text
	
i64 %50
Xcall8BN
L
	full_text?
=
;call void @load_vector(double* nonnull %46, double* %64) #5
-double*8B

	full_text

double* %46
-double*8B

	full_text

double* %64
|call8Br
p
	full_textc
a
_call void @p_matvec_sub([5 x double]* nonnull %56, double* nonnull %55, double* nonnull %46) #5
9[5 x double]*8B$
"
	full_text

[5 x double]* %56
-double*8B

	full_text

double* %55
-double*8B

	full_text

double* %46
àcall8B~
|
	full_texto
m
kcall void @p_matmul_sub([5 x double]* nonnull %56, [5 x double]* nonnull %54, [5 x double]* nonnull %40) #5
9[5 x double]*8B$
"
	full_text

[5 x double]* %56
9[5 x double]*8B$
"
	full_text

[5 x double]* %54
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
Äcall8Bv
t
	full_textg
e
ccall void @p_binvcrhs([5 x double]* nonnull %40, [5 x double]* nonnull %43, double* nonnull %46) #5
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
9[5 x double]*8B$
"
	full_text

[5 x double]* %43
-double*8B

	full_text

double* %46
dcall8BZ
X
	full_textK
I
Gcall void @save_matrix([5 x double]* %62, [5 x double]* nonnull %40) #5
9[5 x double]*8B$
"
	full_text

[5 x double]* %62
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
dcall8BZ
X
	full_textK
I
Gcall void @save_matrix([5 x double]* %63, [5 x double]* nonnull %43) #5
9[5 x double]*8B$
"
	full_text

[5 x double]* %63
9[5 x double]*8B$
"
	full_text

[5 x double]* %43
Xcall8BN
L
	full_text?
=
;call void @save_vector(double* %64, double* nonnull %46) #5
-double*8B

	full_text

double* %64
-double*8B

	full_text

double* %46
8add8B/
-
	full_text 

%65 = add nuw nsw i64 %60, 1
%i648B

	full_text
	
i64 %60
lcall8Bb
`
	full_textS
Q
Ocall void @copy_matrix([5 x double]* nonnull %54, [5 x double]* nonnull %43) #5
9[5 x double]*8B$
"
	full_text

[5 x double]* %54
9[5 x double]*8B$
"
	full_text

[5 x double]* %43
`call8BV
T
	full_textG
E
Ccall void @copy_vector(double* nonnull %55, double* nonnull %46) #5
-double*8B

	full_text

double* %55
-double*8B

	full_text

double* %46
7icmp8B-
+
	full_text

%66 = icmp eq i64 %65, %58
%i648B

	full_text
	
i64 %65
%i648B

	full_text
	
i64 %58
:br8B2
0
	full_text#
!
br i1 %66, label %67, label %59
#i18B

	full_text


i1 %66
6sext8B,
*
	full_text

%68 = sext i32 %39 to i64
%i328B

	full_text
	
i32 %39
ëgetelementptr8B~
|
	full_texto
m
k%69 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %68, i64 0, i64 0
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %68
dcall8BZ
X
	full_textK
I
Gcall void @load_matrix([5 x double]* nonnull %56, [5 x double]* %69) #5
9[5 x double]*8B$
"
	full_text

[5 x double]* %56
9[5 x double]*8B$
"
	full_text

[5 x double]* %69
ëgetelementptr8B~
|
	full_texto
m
k%70 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %68, i64 1, i64 0
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %68
dcall8BZ
X
	full_textK
I
Gcall void @load_matrix([5 x double]* nonnull %40, [5 x double]* %70) #5
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
9[5 x double]*8B$
"
	full_text

[5 x double]* %70
ëgetelementptr8B~
|
	full_texto
m
k%71 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %68, i64 2, i64 0
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %68
dcall8BZ
X
	full_textK
I
Gcall void @load_matrix([5 x double]* nonnull %43, [5 x double]* %71) #5
9[5 x double]*8B$
"
	full_text

[5 x double]* %43
9[5 x double]*8B$
"
	full_text

[5 x double]* %71
¢getelementptr8Bé
ã
	full_text~
|
z%72 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %30, i64 %68, i64 %48, i64 %50, i64 0
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %30
%i648B

	full_text
	
i64 %68
%i648B

	full_text
	
i64 %48
%i648B

	full_text
	
i64 %50
Xcall8BN
L
	full_text?
=
;call void @load_vector(double* nonnull %46, double* %72) #5
-double*8B

	full_text

double* %46
-double*8B

	full_text

double* %72
|call8Br
p
	full_textc
a
_call void @p_matvec_sub([5 x double]* nonnull %56, double* nonnull %55, double* nonnull %46) #5
9[5 x double]*8B$
"
	full_text

[5 x double]* %56
-double*8B

	full_text

double* %55
-double*8B

	full_text

double* %46
àcall8B~
|
	full_texto
m
kcall void @p_matmul_sub([5 x double]* nonnull %56, [5 x double]* nonnull %54, [5 x double]* nonnull %40) #5
9[5 x double]*8B$
"
	full_text

[5 x double]* %56
9[5 x double]*8B$
"
	full_text

[5 x double]* %54
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
dcall8BZ
X
	full_textK
I
Gcall void @p_binvrhs([5 x double]* nonnull %40, double* nonnull %46) #5
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
-double*8B

	full_text

double* %46
dcall8BZ
X
	full_textK
I
Gcall void @save_matrix([5 x double]* %70, [5 x double]* nonnull %40) #5
9[5 x double]*8B$
"
	full_text

[5 x double]* %70
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
dcall8BZ
X
	full_textK
I
Gcall void @save_matrix([5 x double]* %71, [5 x double]* nonnull %43) #5
9[5 x double]*8B$
"
	full_text

[5 x double]* %71
9[5 x double]*8B$
"
	full_text

[5 x double]* %43
Xcall8BN
L
	full_text?
=
;call void @save_vector(double* %72, double* nonnull %46) #5
-double*8B

	full_text

double* %72
-double*8B

	full_text

double* %46
5icmp8B+
)
	full_text

%73 = icmp sgt i32 %4, 1
;br8B3
1
	full_text$
"
 br i1 %73, label %74, label %204
#i18B

	full_text


i1 %73
6sext8B,
*
	full_text

%75 = sext i32 %52 to i64
%i328B

	full_text
	
i32 %52
ogetelementptr8B\
Z
	full_textM
K
I%76 = getelementptr inbounds [5 x double], [5 x double]* %9, i64 0, i64 1
8[5 x double]*8B#
!
	full_text

[5 x double]* %9
ogetelementptr8B\
Z
	full_textM
K
I%77 = getelementptr inbounds [5 x double], [5 x double]* %9, i64 0, i64 2
8[5 x double]*8B#
!
	full_text

[5 x double]* %9
ogetelementptr8B\
Z
	full_textM
K
I%78 = getelementptr inbounds [5 x double], [5 x double]* %9, i64 0, i64 3
8[5 x double]*8B#
!
	full_text

[5 x double]* %9
ogetelementptr8B\
Z
	full_textM
K
I%79 = getelementptr inbounds [5 x double], [5 x double]* %9, i64 0, i64 4
8[5 x double]*8B#
!
	full_text

[5 x double]* %9
pgetelementptr8B]
[
	full_textN
L
J%80 = getelementptr inbounds [5 x double], [5 x double]* %11, i64 0, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %11
pgetelementptr8B]
[
	full_textN
L
J%81 = getelementptr inbounds [5 x double], [5 x double]* %11, i64 0, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %11
pgetelementptr8B]
[
	full_textN
L
J%82 = getelementptr inbounds [5 x double], [5 x double]* %11, i64 0, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %11
pgetelementptr8B]
[
	full_textN
L
J%83 = getelementptr inbounds [5 x double], [5 x double]* %11, i64 0, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %11
pgetelementptr8B]
[
	full_textN
L
J%84 = getelementptr inbounds [5 x double], [5 x double]* %11, i64 0, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %11
'br8B

	full_text

br label %85
Ephi8B<
:
	full_text-
+
)%86 = phi i64 [ %75, %74 ], [ %202, %85 ]
%i648B

	full_text
	
i64 %75
&i648B

	full_text


i64 %202
Oload8BE
C
	full_text6
4
2%87 = load double, double* %46, align 16, !tbaa !8
-double*8B

	full_text

double* %46
Nload8BD
B
	full_text5
3
1%88 = load double, double* %76, align 8, !tbaa !8
-double*8B

	full_text

double* %76
Oload8BE
C
	full_text6
4
2%89 = load double, double* %77, align 16, !tbaa !8
-double*8B

	full_text

double* %77
Nload8BD
B
	full_text5
3
1%90 = load double, double* %78, align 8, !tbaa !8
-double*8B

	full_text

double* %78
Oload8BE
C
	full_text6
4
2%91 = load double, double* %79, align 16, !tbaa !8
-double*8B

	full_text

double* %79
¢getelementptr8Bé
ã
	full_text~
|
z%92 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %30, i64 %86, i64 %48, i64 %50, i64 0
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %30
%i648B

	full_text
	
i64 %86
%i648B

	full_text
	
i64 %48
%i648B

	full_text
	
i64 %50
Nload8BD
B
	full_text5
3
1%93 = load double, double* %92, align 8, !tbaa !8
-double*8B

	full_text

double* %92
ögetelementptr8BÜ
É
	full_textv
t
r%94 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %86, i64 2, i64 0, i64 0
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %86
Nload8BD
B
	full_text5
3
1%95 = load double, double* %94, align 8, !tbaa !8
-double*8B

	full_text

double* %94
Afsub8B7
5
	full_text(
&
$%96 = fsub double -0.000000e+00, %95
+double8B

	full_text


double %95
dcall8BZ
X
	full_textK
I
G%97 = call double @llvm.fmuladd.f64(double %96, double %87, double %93)
+double8B

	full_text


double %96
+double8B

	full_text


double %87
+double8B

	full_text


double %93
ögetelementptr8BÜ
É
	full_textv
t
r%98 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %86, i64 2, i64 1, i64 0
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %86
Nload8BD
B
	full_text5
3
1%99 = load double, double* %98, align 8, !tbaa !8
-double*8B

	full_text

double* %98
Bfsub8B8
6
	full_text)
'
%%100 = fsub double -0.000000e+00, %99
+double8B

	full_text


double %99
fcall8B\
Z
	full_textM
K
I%101 = call double @llvm.fmuladd.f64(double %100, double %88, double %97)
,double8B

	full_text

double %100
+double8B

	full_text


double %88
+double8B

	full_text


double %97
õgetelementptr8Bá
Ñ
	full_textw
u
s%102 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %86, i64 2, i64 2, i64 0
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %86
Pload8BF
D
	full_text7
5
3%103 = load double, double* %102, align 8, !tbaa !8
.double*8B

	full_text

double* %102
Cfsub8B9
7
	full_text*
(
&%104 = fsub double -0.000000e+00, %103
,double8B

	full_text

double %103
gcall8B]
[
	full_textN
L
J%105 = call double @llvm.fmuladd.f64(double %104, double %89, double %101)
,double8B

	full_text

double %104
+double8B

	full_text


double %89
,double8B

	full_text

double %101
õgetelementptr8Bá
Ñ
	full_textw
u
s%106 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %86, i64 2, i64 3, i64 0
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %86
Pload8BF
D
	full_text7
5
3%107 = load double, double* %106, align 8, !tbaa !8
.double*8B

	full_text

double* %106
Cfsub8B9
7
	full_text*
(
&%108 = fsub double -0.000000e+00, %107
,double8B

	full_text

double %107
gcall8B]
[
	full_textN
L
J%109 = call double @llvm.fmuladd.f64(double %108, double %90, double %105)
,double8B

	full_text

double %108
+double8B

	full_text


double %90
,double8B

	full_text

double %105
õgetelementptr8Bá
Ñ
	full_textw
u
s%110 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %86, i64 2, i64 4, i64 0
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %86
Pload8BF
D
	full_text7
5
3%111 = load double, double* %110, align 8, !tbaa !8
.double*8B

	full_text

double* %110
Cfsub8B9
7
	full_text*
(
&%112 = fsub double -0.000000e+00, %111
,double8B

	full_text

double %111
gcall8B]
[
	full_textN
L
J%113 = call double @llvm.fmuladd.f64(double %112, double %91, double %109)
,double8B

	full_text

double %112
+double8B

	full_text


double %91
,double8B

	full_text

double %109
Pstore8BE
C
	full_text6
4
2store double %113, double* %80, align 16, !tbaa !8
,double8B

	full_text

double %113
-double*8B

	full_text

double* %80
Ostore8BD
B
	full_text5
3
1store double %113, double* %92, align 8, !tbaa !8
,double8B

	full_text

double %113
-double*8B

	full_text

double* %92
£getelementptr8Bè
å
	full_text
}
{%114 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %30, i64 %86, i64 %48, i64 %50, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %30
%i648B

	full_text
	
i64 %86
%i648B

	full_text
	
i64 %48
%i648B

	full_text
	
i64 %50
Pload8BF
D
	full_text7
5
3%115 = load double, double* %114, align 8, !tbaa !8
.double*8B

	full_text

double* %114
õgetelementptr8Bá
Ñ
	full_textw
u
s%116 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %86, i64 2, i64 0, i64 1
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %86
Pload8BF
D
	full_text7
5
3%117 = load double, double* %116, align 8, !tbaa !8
.double*8B

	full_text

double* %116
Cfsub8B9
7
	full_text*
(
&%118 = fsub double -0.000000e+00, %117
,double8B

	full_text

double %117
gcall8B]
[
	full_textN
L
J%119 = call double @llvm.fmuladd.f64(double %118, double %87, double %115)
,double8B

	full_text

double %118
+double8B

	full_text


double %87
,double8B

	full_text

double %115
õgetelementptr8Bá
Ñ
	full_textw
u
s%120 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %86, i64 2, i64 1, i64 1
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %86
Pload8BF
D
	full_text7
5
3%121 = load double, double* %120, align 8, !tbaa !8
.double*8B

	full_text

double* %120
Cfsub8B9
7
	full_text*
(
&%122 = fsub double -0.000000e+00, %121
,double8B

	full_text

double %121
gcall8B]
[
	full_textN
L
J%123 = call double @llvm.fmuladd.f64(double %122, double %88, double %119)
,double8B

	full_text

double %122
+double8B

	full_text


double %88
,double8B

	full_text

double %119
õgetelementptr8Bá
Ñ
	full_textw
u
s%124 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %86, i64 2, i64 2, i64 1
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %86
Pload8BF
D
	full_text7
5
3%125 = load double, double* %124, align 8, !tbaa !8
.double*8B

	full_text

double* %124
Cfsub8B9
7
	full_text*
(
&%126 = fsub double -0.000000e+00, %125
,double8B

	full_text

double %125
gcall8B]
[
	full_textN
L
J%127 = call double @llvm.fmuladd.f64(double %126, double %89, double %123)
,double8B

	full_text

double %126
+double8B

	full_text


double %89
,double8B

	full_text

double %123
õgetelementptr8Bá
Ñ
	full_textw
u
s%128 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %86, i64 2, i64 3, i64 1
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %86
Pload8BF
D
	full_text7
5
3%129 = load double, double* %128, align 8, !tbaa !8
.double*8B

	full_text

double* %128
Cfsub8B9
7
	full_text*
(
&%130 = fsub double -0.000000e+00, %129
,double8B

	full_text

double %129
gcall8B]
[
	full_textN
L
J%131 = call double @llvm.fmuladd.f64(double %130, double %90, double %127)
,double8B

	full_text

double %130
+double8B

	full_text


double %90
,double8B

	full_text

double %127
õgetelementptr8Bá
Ñ
	full_textw
u
s%132 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %86, i64 2, i64 4, i64 1
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %86
Pload8BF
D
	full_text7
5
3%133 = load double, double* %132, align 8, !tbaa !8
.double*8B

	full_text

double* %132
Cfsub8B9
7
	full_text*
(
&%134 = fsub double -0.000000e+00, %133
,double8B

	full_text

double %133
gcall8B]
[
	full_textN
L
J%135 = call double @llvm.fmuladd.f64(double %134, double %91, double %131)
,double8B

	full_text

double %134
+double8B

	full_text


double %91
,double8B

	full_text

double %131
Ostore8BD
B
	full_text5
3
1store double %135, double* %81, align 8, !tbaa !8
,double8B

	full_text

double %135
-double*8B

	full_text

double* %81
Pstore8BE
C
	full_text6
4
2store double %135, double* %114, align 8, !tbaa !8
,double8B

	full_text

double %135
.double*8B

	full_text

double* %114
£getelementptr8Bè
å
	full_text
}
{%136 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %30, i64 %86, i64 %48, i64 %50, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %30
%i648B

	full_text
	
i64 %86
%i648B

	full_text
	
i64 %48
%i648B

	full_text
	
i64 %50
Pload8BF
D
	full_text7
5
3%137 = load double, double* %136, align 8, !tbaa !8
.double*8B

	full_text

double* %136
õgetelementptr8Bá
Ñ
	full_textw
u
s%138 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %86, i64 2, i64 0, i64 2
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %86
Pload8BF
D
	full_text7
5
3%139 = load double, double* %138, align 8, !tbaa !8
.double*8B

	full_text

double* %138
Cfsub8B9
7
	full_text*
(
&%140 = fsub double -0.000000e+00, %139
,double8B

	full_text

double %139
gcall8B]
[
	full_textN
L
J%141 = call double @llvm.fmuladd.f64(double %140, double %87, double %137)
,double8B

	full_text

double %140
+double8B

	full_text


double %87
,double8B

	full_text

double %137
õgetelementptr8Bá
Ñ
	full_textw
u
s%142 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %86, i64 2, i64 1, i64 2
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %86
Pload8BF
D
	full_text7
5
3%143 = load double, double* %142, align 8, !tbaa !8
.double*8B

	full_text

double* %142
Cfsub8B9
7
	full_text*
(
&%144 = fsub double -0.000000e+00, %143
,double8B

	full_text

double %143
gcall8B]
[
	full_textN
L
J%145 = call double @llvm.fmuladd.f64(double %144, double %88, double %141)
,double8B

	full_text

double %144
+double8B

	full_text


double %88
,double8B

	full_text

double %141
õgetelementptr8Bá
Ñ
	full_textw
u
s%146 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %86, i64 2, i64 2, i64 2
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %86
Pload8BF
D
	full_text7
5
3%147 = load double, double* %146, align 8, !tbaa !8
.double*8B

	full_text

double* %146
Cfsub8B9
7
	full_text*
(
&%148 = fsub double -0.000000e+00, %147
,double8B

	full_text

double %147
gcall8B]
[
	full_textN
L
J%149 = call double @llvm.fmuladd.f64(double %148, double %89, double %145)
,double8B

	full_text

double %148
+double8B

	full_text


double %89
,double8B

	full_text

double %145
õgetelementptr8Bá
Ñ
	full_textw
u
s%150 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %86, i64 2, i64 3, i64 2
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %86
Pload8BF
D
	full_text7
5
3%151 = load double, double* %150, align 8, !tbaa !8
.double*8B

	full_text

double* %150
Cfsub8B9
7
	full_text*
(
&%152 = fsub double -0.000000e+00, %151
,double8B

	full_text

double %151
gcall8B]
[
	full_textN
L
J%153 = call double @llvm.fmuladd.f64(double %152, double %90, double %149)
,double8B

	full_text

double %152
+double8B

	full_text


double %90
,double8B

	full_text

double %149
õgetelementptr8Bá
Ñ
	full_textw
u
s%154 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %86, i64 2, i64 4, i64 2
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %86
Pload8BF
D
	full_text7
5
3%155 = load double, double* %154, align 8, !tbaa !8
.double*8B

	full_text

double* %154
Cfsub8B9
7
	full_text*
(
&%156 = fsub double -0.000000e+00, %155
,double8B

	full_text

double %155
gcall8B]
[
	full_textN
L
J%157 = call double @llvm.fmuladd.f64(double %156, double %91, double %153)
,double8B

	full_text

double %156
+double8B

	full_text


double %91
,double8B

	full_text

double %153
Pstore8BE
C
	full_text6
4
2store double %157, double* %82, align 16, !tbaa !8
,double8B

	full_text

double %157
-double*8B

	full_text

double* %82
Pstore8BE
C
	full_text6
4
2store double %157, double* %136, align 8, !tbaa !8
,double8B

	full_text

double %157
.double*8B

	full_text

double* %136
£getelementptr8Bè
å
	full_text
}
{%158 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %30, i64 %86, i64 %48, i64 %50, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %30
%i648B

	full_text
	
i64 %86
%i648B

	full_text
	
i64 %48
%i648B

	full_text
	
i64 %50
Pload8BF
D
	full_text7
5
3%159 = load double, double* %158, align 8, !tbaa !8
.double*8B

	full_text

double* %158
õgetelementptr8Bá
Ñ
	full_textw
u
s%160 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %86, i64 2, i64 0, i64 3
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %86
Pload8BF
D
	full_text7
5
3%161 = load double, double* %160, align 8, !tbaa !8
.double*8B

	full_text

double* %160
Cfsub8B9
7
	full_text*
(
&%162 = fsub double -0.000000e+00, %161
,double8B

	full_text

double %161
gcall8B]
[
	full_textN
L
J%163 = call double @llvm.fmuladd.f64(double %162, double %87, double %159)
,double8B

	full_text

double %162
+double8B

	full_text


double %87
,double8B

	full_text

double %159
õgetelementptr8Bá
Ñ
	full_textw
u
s%164 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %86, i64 2, i64 1, i64 3
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %86
Pload8BF
D
	full_text7
5
3%165 = load double, double* %164, align 8, !tbaa !8
.double*8B

	full_text

double* %164
Cfsub8B9
7
	full_text*
(
&%166 = fsub double -0.000000e+00, %165
,double8B

	full_text

double %165
gcall8B]
[
	full_textN
L
J%167 = call double @llvm.fmuladd.f64(double %166, double %88, double %163)
,double8B

	full_text

double %166
+double8B

	full_text


double %88
,double8B

	full_text

double %163
õgetelementptr8Bá
Ñ
	full_textw
u
s%168 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %86, i64 2, i64 2, i64 3
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %86
Pload8BF
D
	full_text7
5
3%169 = load double, double* %168, align 8, !tbaa !8
.double*8B

	full_text

double* %168
Cfsub8B9
7
	full_text*
(
&%170 = fsub double -0.000000e+00, %169
,double8B

	full_text

double %169
gcall8B]
[
	full_textN
L
J%171 = call double @llvm.fmuladd.f64(double %170, double %89, double %167)
,double8B

	full_text

double %170
+double8B

	full_text


double %89
,double8B

	full_text

double %167
õgetelementptr8Bá
Ñ
	full_textw
u
s%172 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %86, i64 2, i64 3, i64 3
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %86
Pload8BF
D
	full_text7
5
3%173 = load double, double* %172, align 8, !tbaa !8
.double*8B

	full_text

double* %172
Cfsub8B9
7
	full_text*
(
&%174 = fsub double -0.000000e+00, %173
,double8B

	full_text

double %173
gcall8B]
[
	full_textN
L
J%175 = call double @llvm.fmuladd.f64(double %174, double %90, double %171)
,double8B

	full_text

double %174
+double8B

	full_text


double %90
,double8B

	full_text

double %171
õgetelementptr8Bá
Ñ
	full_textw
u
s%176 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %86, i64 2, i64 4, i64 3
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %86
Pload8BF
D
	full_text7
5
3%177 = load double, double* %176, align 8, !tbaa !8
.double*8B

	full_text

double* %176
Cfsub8B9
7
	full_text*
(
&%178 = fsub double -0.000000e+00, %177
,double8B

	full_text

double %177
gcall8B]
[
	full_textN
L
J%179 = call double @llvm.fmuladd.f64(double %178, double %91, double %175)
,double8B

	full_text

double %178
+double8B

	full_text


double %91
,double8B

	full_text

double %175
Ostore8BD
B
	full_text5
3
1store double %179, double* %83, align 8, !tbaa !8
,double8B

	full_text

double %179
-double*8B

	full_text

double* %83
Pstore8BE
C
	full_text6
4
2store double %179, double* %158, align 8, !tbaa !8
,double8B

	full_text

double %179
.double*8B

	full_text

double* %158
£getelementptr8Bè
å
	full_text
}
{%180 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %30, i64 %86, i64 %48, i64 %50, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %30
%i648B

	full_text
	
i64 %86
%i648B

	full_text
	
i64 %48
%i648B

	full_text
	
i64 %50
Pload8BF
D
	full_text7
5
3%181 = load double, double* %180, align 8, !tbaa !8
.double*8B

	full_text

double* %180
õgetelementptr8Bá
Ñ
	full_textw
u
s%182 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %86, i64 2, i64 0, i64 4
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %86
Pload8BF
D
	full_text7
5
3%183 = load double, double* %182, align 8, !tbaa !8
.double*8B

	full_text

double* %182
Cfsub8B9
7
	full_text*
(
&%184 = fsub double -0.000000e+00, %183
,double8B

	full_text

double %183
gcall8B]
[
	full_textN
L
J%185 = call double @llvm.fmuladd.f64(double %184, double %87, double %181)
,double8B

	full_text

double %184
+double8B

	full_text


double %87
,double8B

	full_text

double %181
õgetelementptr8Bá
Ñ
	full_textw
u
s%186 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %86, i64 2, i64 1, i64 4
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %86
Pload8BF
D
	full_text7
5
3%187 = load double, double* %186, align 8, !tbaa !8
.double*8B

	full_text

double* %186
Cfsub8B9
7
	full_text*
(
&%188 = fsub double -0.000000e+00, %187
,double8B

	full_text

double %187
gcall8B]
[
	full_textN
L
J%189 = call double @llvm.fmuladd.f64(double %188, double %88, double %185)
,double8B

	full_text

double %188
+double8B

	full_text


double %88
,double8B

	full_text

double %185
õgetelementptr8Bá
Ñ
	full_textw
u
s%190 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %86, i64 2, i64 2, i64 4
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %86
Pload8BF
D
	full_text7
5
3%191 = load double, double* %190, align 8, !tbaa !8
.double*8B

	full_text

double* %190
Cfsub8B9
7
	full_text*
(
&%192 = fsub double -0.000000e+00, %191
,double8B

	full_text

double %191
gcall8B]
[
	full_textN
L
J%193 = call double @llvm.fmuladd.f64(double %192, double %89, double %189)
,double8B

	full_text

double %192
+double8B

	full_text


double %89
,double8B

	full_text

double %189
õgetelementptr8Bá
Ñ
	full_textw
u
s%194 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %86, i64 2, i64 3, i64 4
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %86
Pload8BF
D
	full_text7
5
3%195 = load double, double* %194, align 8, !tbaa !8
.double*8B

	full_text

double* %194
Cfsub8B9
7
	full_text*
(
&%196 = fsub double -0.000000e+00, %195
,double8B

	full_text

double %195
gcall8B]
[
	full_textN
L
J%197 = call double @llvm.fmuladd.f64(double %196, double %90, double %193)
,double8B

	full_text

double %196
+double8B

	full_text


double %90
,double8B

	full_text

double %193
õgetelementptr8Bá
Ñ
	full_textw
u
s%198 = getelementptr inbounds [3 x [5 x [5 x double]]], [3 x [5 x [5 x double]]]* %38, i64 %86, i64 2, i64 4, i64 4
Q[3 x [5 x [5 x double]]]*8B0
.
	full_text!

[3 x [5 x [5 x double]]]* %38
%i648B

	full_text
	
i64 %86
Pload8BF
D
	full_text7
5
3%199 = load double, double* %198, align 8, !tbaa !8
.double*8B

	full_text

double* %198
Cfsub8B9
7
	full_text*
(
&%200 = fsub double -0.000000e+00, %199
,double8B

	full_text

double %199
gcall8B]
[
	full_textN
L
J%201 = call double @llvm.fmuladd.f64(double %200, double %91, double %197)
,double8B

	full_text

double %200
+double8B

	full_text


double %91
,double8B

	full_text

double %197
Pstore8BE
C
	full_text6
4
2store double %201, double* %84, align 16, !tbaa !8
,double8B

	full_text

double %201
-double*8B

	full_text

double* %84
Pstore8BE
C
	full_text6
4
2store double %201, double* %180, align 8, !tbaa !8
,double8B

	full_text

double %201
.double*8B

	full_text

double* %180
`call8BV
T
	full_textG
E
Ccall void @copy_vector(double* nonnull %46, double* nonnull %55) #5
-double*8B

	full_text

double* %46
-double*8B

	full_text

double* %55
6add8B-
+
	full_text

%202 = add nsw i64 %86, -1
%i648B

	full_text
	
i64 %86
7icmp8B-
+
	full_text

%203 = icmp sgt i64 %86, 0
%i648B

	full_text
	
i64 %86
<br8B4
2
	full_text%
#
!br i1 %203, label %85, label %204
$i18B

	full_text
	
i1 %203
Zcall8BP
N
	full_textA
?
=call void @llvm.lifetime.end.p0i8(i64 40, i8* nonnull %12) #5
%i8*8B

	full_text
	
i8* %12
[call8BQ
O
	full_textB
@
>call void @llvm.lifetime.end.p0i8(i64 200, i8* nonnull %17) #5
%i8*8B

	full_text
	
i8* %17
Zcall8BP
N
	full_textA
?
=call void @llvm.lifetime.end.p0i8(i64 40, i8* nonnull %16) #5
%i8*8B

	full_text
	
i8* %16
[call8BQ
O
	full_textB
@
>call void @llvm.lifetime.end.p0i8(i64 200, i8* nonnull %15) #5
%i8*8B

	full_text
	
i8* %15
[call8BQ
O
	full_textB
@
>call void @llvm.lifetime.end.p0i8(i64 200, i8* nonnull %14) #5
%i8*8B

	full_text
	
i8* %14
[call8BQ
O
	full_textB
@
>call void @llvm.lifetime.end.p0i8(i64 200, i8* nonnull %13) #5
%i8*8B

	full_text
	
i8* %13
$ret8B

	full_text


ret void
$i328	B

	full_text


i32 %3
,double*8	B

	full_text


double* %0
,double*8	B

	full_text


double* %1
$i328	B

	full_text


i32 %4
$i328	B

	full_text


i32 %2
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
$i648	B

	full_text


i64 32
$i328	B

	full_text


i32 -2
$i328	B

	full_text


i32 -1
&i328	B

	full_text


i32 4875
#i648	B

	full_text	

i64 3
#i648	B

	full_text	

i64 4
$i648	B

	full_text


i64 -1
#i328	B

	full_text	

i32 3
#i648	B

	full_text	

i64 1
#i648	B

	full_text	

i64 0
$i648	B

	full_text


i64 40
#i328	B

	full_text	

i32 1
#i328	B

	full_text	

i32 0
$i648	B

	full_text


i64 25
5double8	B'
%
	full_text

double -0.000000e+00
#i648	B

	full_text	

i64 2
%i648	B

	full_text
	
i64 200
$i648	B

	full_text


i64 50        	
 		                       !    "# "" $$ %& %% '( '' )) *+ *, ** -. -/ 01 02 00 34 35 67 66 89 8: 88 ;< ;; => =? == @A @@ BC BB DE DD FG FF HH IJ II KL KK MN MM OP OQ OO RS RR TU TT VW VV XY XZ XX [\ [[ ]^ ]] _` __ ab aa cd cc ef eg eh ee ij ik ii lm ln lo ll pq pr pp st su ss vw vx vv yy zz {| {{ }~ } }} ÄÅ ÄÄ ÇÉ Ç
Ñ ÇÇ ÖÜ ÖÖ áà áä ââ ãç åå éè é
ê éé ëí ë
ì ëë îï î
ñ îî óò ó
ô óó öõ ö
ú öö ùû ù
ü ùù †° †
¢ †
£ †
§ †† •¶ •
ß •• ®© ®
™ ®
´ ®® ¨≠ ¨
Æ ¨
Ø ¨¨ ∞± ∞
≤ ∞
≥ ∞∞ ¥µ ¥
∂ ¥¥ ∑∏ ∑
π ∑∑ ∫ª ∫
º ∫∫ Ωæ ΩΩ ø¿ ø
¡ øø ¬√ ¬
ƒ ¬¬ ≈∆ ≈
« ≈≈ »… »À    ÃÕ Ã
Œ ÃÃ œ– œ
— œœ “” “
‘ ““ ’÷ ’
◊ ’’ ÿŸ ÿ
⁄ ÿÿ €‹ €
› €€ ﬁﬂ ﬁ
‡ ﬁ
· ﬁ
‚ ﬁﬁ „‰ „
Â „„ ÊÁ Ê
Ë Ê
È ÊÊ ÍÎ Í
Ï Í
Ì ÍÍ ÓÔ Ó
 ÓÓ ÒÚ Ò
Û ÒÒ Ùı Ù
ˆ ÙÙ ˜¯ ˜
˘ ˜˜ ˙˙ ˚¸ ˚˛ ˝˝ ˇÄ ˇˇ ÅÇ ÅÅ ÉÑ ÉÉ ÖÜ ÖÖ áà áá âä ââ ãå ãã çé çç èê èè ëì í
î íí ïñ ïï óò óó ôö ôô õú õõ ùû ùù ü† ü
° ü
¢ ü
£ üü §• §§ ¶ß ¶
® ¶¶ ©™ ©© ´
¨ ´´ ≠Æ ≠
Ø ≠
∞ ≠≠ ±≤ ±
≥ ±± ¥µ ¥¥ ∂
∑ ∂∂ ∏π ∏
∫ ∏
ª ∏∏ ºΩ º
æ ºº ø¿ øø ¡
¬ ¡¡ √ƒ √
≈ √
∆ √√ «» «
… ««  À    Ã
Õ ÃÃ Œœ Œ
– Œ
— ŒŒ “” “
‘ ““ ’÷ ’’ ◊
ÿ ◊◊ Ÿ⁄ Ÿ
€ Ÿ
‹ ŸŸ ›ﬁ ›
ﬂ ›› ‡· ‡
‚ ‡‡ „‰ „
Â „
Ê „
Á „„ ËÈ ËË ÍÎ Í
Ï ÍÍ ÌÓ ÌÌ Ô
 ÔÔ ÒÚ Ò
Û Ò
Ù ÒÒ ıˆ ı
˜ ıı ¯˘ ¯¯ ˙
˚ ˙˙ ¸˝ ¸
˛ ¸
ˇ ¸¸ ÄÅ Ä
Ç ÄÄ ÉÑ ÉÉ Ö
Ü ÖÖ áà á
â á
ä áá ãå ã
ç ãã éè éé ê
ë êê íì í
î í
ï íí ñó ñ
ò ññ ôö ôô õ
ú õõ ùû ù
ü ù
† ùù °¢ °
£ °° §• §
¶ §§ ß® ß
© ß
™ ß
´ ßß ¨≠ ¨¨ ÆØ Æ
∞ ÆÆ ±≤ ±± ≥
¥ ≥≥ µ∂ µ
∑ µ
∏ µµ π∫ π
ª ππ ºΩ ºº æ
ø ææ ¿¡ ¿
¬ ¿
√ ¿¿ ƒ≈ ƒ
∆ ƒƒ «» «« …
  …… ÀÃ À
Õ À
Œ ÀÀ œ– œ
— œœ “” ““ ‘
’ ‘‘ ÷◊ ÷
ÿ ÷
Ÿ ÷÷ ⁄€ ⁄
‹ ⁄⁄ ›ﬁ ›› ﬂ
‡ ﬂﬂ ·‚ ·
„ ·
‰ ·· ÂÊ Â
Á ÂÂ ËÈ Ë
Í ËË ÎÏ Î
Ì Î
Ó Î
Ô ÎÎ Ò  ÚÛ Ú
Ù ÚÚ ıˆ ıı ˜
¯ ˜˜ ˘˙ ˘
˚ ˘
¸ ˘˘ ˝˛ ˝
ˇ ˝˝ ÄÅ ÄÄ Ç
É ÇÇ ÑÖ Ñ
Ü Ñ
á ÑÑ àâ à
ä àà ãå ãã ç
é çç èê è
ë è
í èè ìî ì
ï ìì ñó ññ ò
ô òò öõ ö
ú ö
ù öö ûü û
† ûû °¢ °° £
§ ££ •¶ •
ß •
® •• ©™ ©
´ ©© ¨≠ ¨
Æ ¨¨ Ø∞ Ø
± Ø
≤ Ø
≥ ØØ ¥µ ¥¥ ∂∑ ∂
∏ ∂∂ π∫ ππ ª
º ªª Ωæ Ω
ø Ω
¿ ΩΩ ¡¬ ¡
√ ¡¡ ƒ≈ ƒƒ ∆
« ∆∆ »… »
  »
À »» ÃÕ Ã
Œ ÃÃ œ– œœ —
“ —— ”‘ ”
’ ”
÷ ”” ◊ÿ ◊
Ÿ ◊◊ ⁄€ ⁄⁄ ‹
› ‹‹ ﬁﬂ ﬁ
‡ ﬁ
· ﬁﬁ ‚„ ‚
‰ ‚‚ ÂÊ ÂÂ Á
Ë ÁÁ ÈÍ È
Î È
Ï ÈÈ ÌÓ Ì
Ô ÌÌ Ò 
Ú  ÛÙ Û
ı ÛÛ ˆ˜ ˆˆ ¯˘ ¯¯ ˙˚ ˙
˝ ¸¸ ˛
ˇ ˛˛ Ä
Å ÄÄ Ç
É ÇÇ Ñ
Ö ÑÑ Ü
á ÜÜ àâ )ä 5ã Då Hå yå zå ˙ç /  
	           !  #$ &% () +" ,* ./ 1' 20 4" 76 9/ :' <; >8 ?= A@ CB ED G JD LK NI PM Q SD UT WR YV Z \  ^] `% ba d5 f_ gc h[ je kI mR n[ oM qI rV tR ue w[ x |{ ~R  ÅÄ É[ Ñ Üz àH äΩ çF èå êÖ íé ìF ïå ñI òî ôF õå úR ûö ü5 °å ¢_ £c §[ ¶† ßÖ ©Ä ™[ ´Ö ≠{ ÆI ØI ±R ≤[ ≥î µI ∂ö ∏R π† ª[ ºå æ{ ¿R ¡Ä √[ ƒΩ ∆â «≈ …H ÀF Õ  ŒÖ –Ã —F ”  ‘I ÷“ ◊F Ÿ  ⁄R ‹ÿ ›5 ﬂ  ‡_ ·c ‚[ ‰ﬁ ÂÖ ÁÄ Ë[ ÈÖ Î{ ÏI ÌI Ô[ “ ÚI Ûÿ ıR ˆﬁ ¯[ ˘˙ ¸y ˛ Ä Ç Ñ Ü à ä å é ê˝ ìˆ î[ ñˇ òÅ öÉ úÖ û5 †í °_ ¢c £ü •F ßí ®¶ ™© ¨´ Æï Ø§ ∞F ≤í ≥± µ¥ ∑∂ πó ∫≠ ªF Ωí æº ¿ø ¬¡ ƒô ≈∏ ∆F »í …« À  ÕÃ œõ –√ —F ”í ‘“ ÷’ ÿ◊ ⁄ù €Œ ‹Ÿ ﬁá ﬂŸ ·ü ‚5 ‰í Â_ Êc Á„ ÈF Îí ÏÍ ÓÌ Ô Úï ÛË ÙF ˆí ˜ı ˘¯ ˚˙ ˝ó ˛Ò ˇF Åí ÇÄ ÑÉ ÜÖ àô â¸ äF åí çã èé ëê ìõ îá ïF óí òñ öô úõ ûù üí †ù ¢â £ù •„ ¶5 ®í ©_ ™c ´ß ≠F Øí ∞Æ ≤± ¥≥ ∂ï ∑¨ ∏F ∫í ªπ Ωº øæ ¡ó ¬µ √F ≈í ∆ƒ »«  … Ãô Õ¿ ŒF –í —œ ”“ ’‘ ◊õ ÿÀ ŸF €í ‹⁄ ﬁ› ‡ﬂ ‚ù „÷ ‰· Êã Á· Èß Í5 Ïí Ì_ Óc ÔÎ ÒF Ûí ÙÚ ˆı ¯˜ ˙ï ˚ ¸F ˛í ˇ˝ ÅÄ ÉÇ Öó Ü˘ áF âí äà åã éç êô ëÑ íF îí ïì óñ ôò õõ úè ùF üí †û ¢° §£ ¶ù ßö ®• ™ç ´• ≠Î Æ5 ∞í ±_ ≤c ≥Ø µF ∑í ∏∂ ∫π ºª æï ø¥ ¿F ¬í √¡ ≈ƒ «∆ …ó  Ω ÀF Õí ŒÃ –œ “— ‘ô ’» ÷F ÿí Ÿ◊ €⁄ ›‹ ﬂõ ‡” ·F „í ‰‚ ÊÂ ËÁ Íù Îﬁ ÏÈ Óè ÔÈ ÒØ Ú[ ÙÄ ıí ˜í ˘¯ ˚ ˝ ˇ Å É Ö	 á- ¸- /3 ¸3 5á  á â˚ ˝˚ ¸ã åë í»  » å˙ í˙ ¸ ññ à òò öö ëë ìì èè íí êê ôô éé õõ óó ïï îî∑ ìì ∑œ êê œp ìì pù öö ùO êê O√ öö √i ëë ië êê ë„ ëë „˛ õõ ˛¸ õõ ¸¨ òò ¨· öö · éé } ïï }Ñ öö Ñµ öö µ» öö »Ç ññ Ç∞ íí ∞• öö •Ÿ öö Ÿ€ êê € éé Ä õõ ÄÈ öö È∏ öö ∏í öö íó êê ó÷ öö ÷ éé ¥ ìì ¥Ó ôô ÓÒ ìì ÒÑ õõ Ñø ïï ø’ êê ’À öö À˘ öö ˘¸ öö ¸Û ññ ÛÇ õõ Ç• ëë •˜ îî ˜ù êê ùö öö öÍ òò Í éé ¬ ññ ¬ èè Œ öö Œè öö èX êê Xs ìì s” öö ”Ω öö Ω¿ öö ¿l íí l éé ® óó ®Ê óó Ê éé ∫ îî ∫ﬁ öö ﬁÜ õõ ÜÙ ìì ÙÒ öö Ò≠ öö ≠v îî vá öö á$ èè $	ú ]	ú _	ú a	ú c	ù )	ù /	ù y	û 6	û ;	û H	ü @
† É
† ç
† «
† ã
† œ
† Î
† Ú
† ˝
† à
† ì
† ì
† û
† ◊
° Ö
° è
° “
° ñ
° ⁄
° û
° Ø
° ∂
° ¡
° Ã
° ◊
° ‚
° ‚
¢ ˆ	£ z	§  	§ %
§ å
§ î
§ Ω
§ “
§ ˇ
§ â
§ ±
§ „
§ Í
§ ı
§ ı
§ Ä
§ ã
§ ñ
§ π
§ ˝
§ ¡	• I	• I	• R	• R	• [	• [	• e	• e	• {	• {
• Ä
• Ä
• Ö
• Ö
• é
• é
• î
• ö
• †
• Ã
• Ã
• “
• ÿ
• ﬁ
• ˇ
• Å
• É
• Ö
• á
• á
• â
• ã
• ç
• è
• ü
• ¶
• ¶
• ±
• º
• «
• “
• Í
• Æ
• Ú
• ∂
• ¯¶ ¶ ¶ ¸¶ Äß ß ß ß ß ß ß 
ß ˙® $	© K™ ´™ ∂™ ¡™ Ã™ ◊™ Ô™ ˙™ Ö™ ê™ õ™ ≥™ æ™ …™ ‘™ ﬂ™ ˜™ Ç™ ç™ ò™ £™ ª™ ∆™ —™ ‹™ Á
´ ö
´ ÿ
´ Å
´ ã
´ ¶
´ ±
´ º
´ º
´ «
´ “
´ Í
´ ı
´ Ä
´ Ä
´ ã
´ ñ
´ ß
´ Æ
´ Æ
´ π
´ π
´ ƒ
´ ƒ
´ ƒ
´ œ
´ œ
´ ⁄
´ ⁄
´ Ú
´ ˝
´ à
´ à
´ ì
´ û
´ ∂
´ ¡
´ Ã
´ Ã
´ ◊
´ ‚¨ ¨ ¨ ¨ ¨ ˛¨ Ç¨ Ñ¨ Ü	≠ T"	
z_solve"
llvm.lifetime.start.p0i8"
_Z13get_global_idj"
load_matrix"
load_vector"

p_binvcrhs"
save_matrix"
save_vector"
copy_matrix"
copy_vector"
p_matvec_sub"
p_matmul_sub"
	p_binvrhs"
llvm.fmuladd.f64"
llvm.lifetime.end.p0i8*ä
npb-BT-z_solve.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02Å

wgsize
>
 
transfer_bytes_log1p
ùÆúA

devmap_label


transfer_bytes	
ÿîÁò

wgsize_log1p
ùÆúA