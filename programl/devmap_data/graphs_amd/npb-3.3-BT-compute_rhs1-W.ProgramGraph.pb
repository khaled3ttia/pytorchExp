

[external]
LcallBD
B
	full_text5
3
1%11 = tail call i64 @_Z13get_global_idj(i32 2) #3
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
LcallBD
B
	full_text5
3
1%14 = tail call i64 @_Z13get_global_idj(i32 0) #3
6truncB-
+
	full_text

%15 = trunc i64 %14 to i32
#i64B

	full_text
	
i64 %14
5icmpB-
+
	full_text

%16 = icmp slt i32 %12, %9
#i32B

	full_text
	
i32 %12
6truncB-
+
	full_text

%17 = trunc i64 %13 to i32
#i64B

	full_text
	
i64 %13
5icmpB-
+
	full_text

%18 = icmp slt i32 %17, %8
#i32B

	full_text
	
i32 %17
/andB(
&
	full_text

%19 = and i1 %16, %18
!i1B

	full_text


i1 %16
!i1B

	full_text


i1 %18
5icmpB-
+
	full_text

%20 = icmp slt i32 %15, %7
#i32B

	full_text
	
i32 %15
/andB(
&
	full_text

%21 = and i1 %19, %20
!i1B

	full_text


i1 %19
!i1B

	full_text


i1 %20
8brB2
0
	full_text#
!
br i1 %21, label %22, label %60
!i1B

	full_text


i1 %21
Wbitcast8BJ
H
	full_text;
9
7%23 = bitcast double* %0 to [25 x [25 x [5 x double]]]*
Qbitcast8BD
B
	full_text5
3
1%24 = bitcast double* %1 to [25 x [25 x double]]*
Qbitcast8BD
B
	full_text5
3
1%25 = bitcast double* %2 to [25 x [25 x double]]*
Qbitcast8BD
B
	full_text5
3
1%26 = bitcast double* %3 to [25 x [25 x double]]*
Qbitcast8BD
B
	full_text5
3
1%27 = bitcast double* %4 to [25 x [25 x double]]*
Qbitcast8BD
B
	full_text5
3
1%28 = bitcast double* %5 to [25 x [25 x double]]*
Qbitcast8BD
B
	full_text5
3
1%29 = bitcast double* %6 to [25 x [25 x double]]*
1shl8B(
&
	full_text

%30 = shl i64 %11, 32
%i648B

	full_text
	
i64 %11
9ashr8B/
-
	full_text 

%31 = ashr exact i64 %30, 32
%i648B

	full_text
	
i64 %30
1shl8B(
&
	full_text

%32 = shl i64 %13, 32
%i648B

	full_text
	
i64 %13
9ashr8B/
-
	full_text 

%33 = ashr exact i64 %32, 32
%i648B

	full_text
	
i64 %32
1shl8B(
&
	full_text

%34 = shl i64 %14, 32
%i648B

	full_text
	
i64 %14
9ashr8B/
-
	full_text 

%35 = ashr exact i64 %34, 32
%i648B

	full_text
	
i64 %34
�
�
	full_text~
|
z%36 = getelementptr inbounds [25 x [25 x [5 x double]]], [25 x [25 x [5 x double]]]* %23, i64 %31, i64 %33, i64 %35, i64 0
U[25 x [25 x [5 x double]]]*8B2
0
	full_text#
!
[25 x [25 x [5 x double]]]* %23
%i648B

	full_text
	
i64 %31
%i648B

	full_text
	
i64 %33
%i648B

	full_text
	
i64 %35
Nload8BD
B
	full_text5
3
1%37 = load double, double* %36, align 8, !tbaa !8
-double*8B

	full_text

double* %36
�
�
	full_text~
|
z%38 = getelementptr inbounds [25 x [25 x [5 x double]]], [25 x [25 x [5 x double]]]* %23, i64 %31, i64 %33, i64 %35, i64 1
U[25 x [25 x [5 x double]]]*8B2
0
	full_text#
!
[25 x [25 x [5 x double]]]* %23
%i648B

	full_text
	
i64 %31
%i648B

	full_text
	
i64 %33
%i648B

	full_text
	
i64 %35
Nload8BD
B
	full_text5
3
1%39 = load double, double* %38, align 8, !tbaa !8
-double*8B

	full_text

double* %38
�
�
	full_text~
|
z%40 = getelementptr inbounds [25 x [25 x [5 x double]]], [25 x [25 x [5 x double]]]* %23, i64 %31, i64 %33, i64 %35, i64 2
U[25 x [25 x [5 x double]]]*8B2
0
	full_text#
!
[25 x [25 x [5 x double]]]* %23
%i648B

	full_text
	
i64 %31
%i648B

	full_text
	
i64 %33
%i648B

	full_text
	
i64 %35
Nload8BD
B
	full_text5
3
1%41 = load double, double* %40, align 8, !tbaa !8
-double*8B

	full_text

double* %40
�
�
	full_text~
|
z%42 = getelementptr inbounds [25 x [25 x [5 x double]]], [25 x [25 x [5 x double]]]* %23, i64 %31, i64 %33, i64 %35, i64 3
U[25 x [25 x [5 x double]]]*8B2
0
	full_text#
!
[25 x [25 x [5 x double]]]* %23
%i648B

	full_text
	
i64 %31
%i648B

	full_text
	
i64 %33
%i648B

	full_text
	
i64 %35
Nload8BD
B
	full_text5
3
1%43 = load double, double* %42, align 8, !tbaa !8
-double*8B

	full_text

double* %42
@fdiv8B6
4
	full_text'
%
#%44 = fdiv double 1.000000e+00, %37
+double8B

	full_text


double %37
�
x
	full_textk
i
g%45 = getelementptr inbounds [25 x [25 x double]], [25 x [25 x double]]* %28, i64 %31, i64 %33, i64 %35
I[25 x [25 x double]]*8B,
*
	full_text

[25 x [25 x double]]* %28
%i648B

	full_text
	
i64 %31
%i648B

	full_text
	
i64 %33
%i648B

	full_text
	
i64 %35
Nstore8BC
A
	full_text4
2
0store double %44, double* %45, align 8, !tbaa !8
+double8B

	full_text


double %44
-double*8B

	full_text

double* %45
7fmul8B-
+
	full_text

%46 = fmul double %39, %44
+double8B

	full_text


double %39
+double8B

	full_text


double %44
�
x
	full_textk
i
g%47 = getelementptr inbounds [25 x [25 x double]], [25 x [25 x double]]* %24, i64 %31, i64 %33, i64 %35
I[25 x [25 x double]]*8B,
*
	full_text

[25 x [25 x double]]* %24
%i648B

	full_text
	
i64 %31
%i648B

	full_text
	
i64 %33
%i648B

	full_text
	
i64 %35
Nstore8BC
A
	full_text4
2
0store double %46, double* %47, align 8, !tbaa !8
+double8B

	full_text


double %46
-double*8B

	full_text

double* %47
7fmul8B-
+
	full_text

%48 = fmul double %44, %41
+double8B

	full_text


double %44
+double8B

	full_text


double %41
�
x
	full_textk
i
g%49 = getelementptr inbounds [25 x [25 x double]], [25 x [25 x double]]* %25, i64 %31, i64 %33, i64 %35
I[25 x [25 x double]]*8B,
*
	full_text

[25 x [25 x double]]* %25
%i648B

	full_text
	
i64 %31
%i648B

	full_text
	
i64 %33
%i648B

	full_text
	
i64 %35
Nstore8BC
A
	full_text4
2
0store double %48, double* %49, align 8, !tbaa !8
+double8B

	full_text


double %48
-double*8B

	full_text

double* %49
7fmul8B-
+
	full_text

%50 = fmul double %44, %43
+double8B

	full_text


double %44
+double8B

	full_text


double %43
�
x
	full_textk
i
g%51 = getelementptr inbounds [25 x [25 x double]], [25 x [25 x double]]* %26, i64 %31, i64 %33, i64 %35
I[25 x [25 x double]]*8B,
*
	full_text

[25 x [25 x double]]* %26
%i648B

	full_text
	
i64 %31
%i648B

	full_text
	
i64 %33
%i648B

	full_text
	
i64 %35
Nstore8BC
A
	full_text4
2
0store double %50, double* %51, align 8, !tbaa !8
+double8B

	full_text


double %50
-double*8B

	full_text

double* %51
7fmul8B-
+
	full_text

%52 = fmul double %41, %41
+double8B

	full_text


double %41
+double8B

	full_text


double %41
icall8B_
]
	full_textP
N
L%53 = tail call double @llvm.fmuladd.f64(double %39, double %39, double %52)
+double8B

	full_text


double %39
+double8B

	full_text


double %39
+double8B

	full_text


double %52
icall8B_
]
	full_textP
N
L%54 = tail call double @llvm.fmuladd.f64(double %43, double %43, double %53)
+double8B

	full_text


double %43
+double8B

	full_text


double %43
+double8B

	full_text


double %53
@fmul8B6
4
	full_text'
%
#%55 = fmul double %54, 5.000000e-01
+double8B

	full_text


double %54
7fmul8B-
+
	full_text

%56 = fmul double %44, %55
+double8B

	full_text


double %44
+double8B

	full_text


double %55
�
x
	full_textk
i
g%57 = getelementptr inbounds [25 x [25 x double]], [25 x [25 x double]]* %29, i64 %31, i64 %33, i64 %35
I[25 x [25 x double]]*8B,
*
	full_text

[25 x [25 x double]]* %29
%i648B

	full_text
	
i64 %31
%i648B

	full_text
	
i64 %33
%i648B

	full_text
	
i64 %35
Nstore8BC
A
	full_text4
2
0store double %56, double* %57, align 8, !tbaa !8
+double8B

	full_text


double %56
-double*8B

	full_text

double* %57
7fmul8B-
+
	full_text

%58 = fmul double %44, %56
+double8B

	full_text


double %44
+double8B

	full_text


double %56
�
x
	full_textk
i
g%59 = getelementptr inbounds [25 x [25 x double]], [25 x [25 x double]]* %27, i64 %31, i64 %33, i64 %35
I[25 x [25 x double]]*8B,
*
	full_text

[25 x [25 x double]]* %27
%i648B

	full_text
	
i64 %31
%i648B

	full_text
	
i64 %33
%i648B

	full_text
	
i64 %35
Nstore8BC
A
	full_text4
2
0store double %58, double* %59, align 8, !tbaa !8
+double8B

	full_text


double %58
-double*8B

	full_text

double* %59
'br8B

	full_text

br label %60
$ret8B

	full_text


ret void
,double*8B

	full_text


double* %6
,double*8B

	full_text


double* %1
,double*8B

	full_text


double* %2
$i328B

	full_text


i32 %7
,double*8B

	full_text


double* %4
,double*8B

	full_text


double* %3
$i328B

	full_text


i32 %8
$i328B

	full_text


i32 %9
,double*8B

	full_text


double* %0
,double*8B

	full_text


double* %5
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
4double8B&
$
	full_text

double 1.000000e+00
4double8B&
$
	full_text

double 5.000000e-01
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
#i648B

	full_text	

i64 2
#i328B

	full_text	

i32 2
#i328B

	full_text	

i32 0
#i648B

	full_text	

i64 1
$i648B

	full_text


i64 32
#i648B

	full_text	

i64 3       	  
 

 
� �
� �
� �� �� �
� �� �� �
� �� �� �
� �
� �
� �� �� �
� �� �� � � 	� � � 	� 	� � �    	 
 
compute_rhs1"
_Z13get_global_idj"
llvm.fmuladd.f64*�
npb-BT-compute_rhs1_W.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282�

transfer_bytes
��
 
transfer_bytes_log1p
ϭ�A

wgsize_log1p
ϭ�A

wgsize
0

devmap_label
 