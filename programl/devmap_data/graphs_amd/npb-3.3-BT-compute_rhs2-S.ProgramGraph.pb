

[external]
KcallBC
A
	full_text4
2
0%6 = tail call i64 @_Z13get_global_idj(i32 2) #2
4truncB+
)
	full_text

%7 = trunc i64 %6 to i32
"i64B

	full_text


i64 %6
KcallBC
A
	full_text4
2
0%8 = tail call i64 @_Z13get_global_idj(i32 1) #2
KcallBC
A
	full_text4
2
0%9 = tail call i64 @_Z13get_global_idj(i32 0) #2
5truncB,
*
	full_text

%10 = trunc i64 %9 to i32
"i64B

	full_text


i64 %9
4icmpB,
*
	full_text

%11 = icmp slt i32 %7, %4
"i32B

	full_text


i32 %7
5truncB,
*
	full_text

%12 = trunc i64 %8 to i32
"i64B

	full_text


i64 %8
5icmpB-
+
	full_text

%13 = icmp slt i32 %12, %3
#i32B

	full_text
	
i32 %12
/andB(
&
	full_text

%14 = and i1 %11, %13
!i1B

	full_text


i1 %11
!i1B

	full_text


i1 %13
5icmpB-
+
	full_text

%15 = icmp slt i32 %10, %2
#i32B

	full_text
	
i32 %10
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
8brB2
0
	full_text#
!
br i1 %16, label %17, label %51
!i1B

	full_text


i1 %16
Wbitcast8BJ
H
	full_text;
9
7%18 = bitcast double* %0 to [13 x [13 x [5 x double]]]*
Wbitcast8BJ
H
	full_text;
9
7%19 = bitcast double* %1 to [13 x [13 x [5 x double]]]*
0shl8B'
%
	full_text

%20 = shl i64 %6, 32
$i648B

	full_text


i64 %6
9ashr8B/
-
	full_text 

%21 = ashr exact i64 %20, 32
%i648B

	full_text
	
i64 %20
0shl8B'
%
	full_text

%22 = shl i64 %8, 32
$i648B

	full_text


i64 %8
9ashr8B/
-
	full_text 

%23 = ashr exact i64 %22, 32
%i648B

	full_text
	
i64 %22
0shl8B'
%
	full_text

%24 = shl i64 %9, 32
$i648B

	full_text


i64 %9
9ashr8B/
-
	full_text 

%25 = ashr exact i64 %24, 32
%i648B

	full_text
	
i64 %24
�
�
	full_text~
|
z%26 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %18, i64 %21, i64 %23, i64 %25, i64 0
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %18
%i648B

	full_text
	
i64 %21
%i648B

	full_text
	
i64 %23
%i648B

	full_text
	
i64 %25
Abitcast8B4
2
	full_text%
#
!%27 = bitcast double* %26 to i64*
-double*8B

	full_text

double* %26
Hload8B>
<
	full_text/
-
+%28 = load i64, i64* %27, align 8, !tbaa !8
'i64*8B

	full_text


i64* %27
�
�
	full_text~
|
z%29 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %19, i64 %21, i64 %23, i64 %25, i64 0
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %19
%i648B

	full_text
	
i64 %21
%i648B

	full_text
	
i64 %23
%i648B

	full_text
	
i64 %25
Abitcast8B4
2
	full_text%
#
!%30 = bitcast double* %29 to i64*
-double*8B

	full_text

double* %29
Hstore8B=
;
	full_text.
,
*store i64 %28, i64* %30, align 8, !tbaa !8
%i648B

	full_text
	
i64 %28
'i64*8B

	full_text


i64* %30
�
�
	full_text~
|
z%31 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %18, i64 %21, i64 %23, i64 %25, i64 1
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %18
%i648B

	full_text
	
i64 %21
%i648B

	full_text
	
i64 %23
%i648B

	full_text
	
i64 %25
Abitcast8B4
2
	full_text%
#
!%32 = bitcast double* %31 to i64*
-double*8B

	full_text

double* %31
Hload8B>
<
	full_text/
-
+%33 = load i64, i64* %32, align 8, !tbaa !8
'i64*8B

	full_text


i64* %32
�
�
	full_text~
|
z%34 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %19, i64 %21, i64 %23, i64 %25, i64 1
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %19
%i648B

	full_text
	
i64 %21
%i648B

	full_text
	
i64 %23
%i648B

	full_text
	
i64 %25
Abitcast8B4
2
	full_text%
#
!%35 = bitcast double* %34 to i64*
-double*8B

	full_text

double* %34
Hstore8B=
;
	full_text.
,
*store i64 %33, i64* %35, align 8, !tbaa !8
%i648B

	full_text
	
i64 %33
'i64*8B

	full_text


i64* %35
�
�
	full_text~
|
z%36 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %18, i64 %21, i64 %23, i64 %25, i64 2
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %18
%i648B

	full_text
	
i64 %21
%i648B

	full_text
	
i64 %23
%i648B

	full_text
	
i64 %25
Abitcast8B4
2
	full_text%
#
!%37 = bitcast double* %36 to i64*
-double*8B

	full_text

double* %36
Hload8B>
<
	full_text/
-
+%38 = load i64, i64* %37, align 8, !tbaa !8
'i64*8B

	full_text


i64* %37
�
�
	full_text~
|
z%39 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %19, i64 %21, i64 %23, i64 %25, i64 2
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %19
%i648B

	full_text
	
i64 %21
%i648B

	full_text
	
i64 %23
%i648B

	full_text
	
i64 %25
Abitcast8B4
2
	full_text%
#
!%40 = bitcast double* %39 to i64*
-double*8B

	full_text

double* %39
Hstore8B=
;
	full_text.
,
*store i64 %38, i64* %40, align 8, !tbaa !8
%i648B

	full_text
	
i64 %38
'i64*8B

	full_text


i64* %40
�
�
	full_text~
|
z%41 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %18, i64 %21, i64 %23, i64 %25, i64 3
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %18
%i648B

	full_text
	
i64 %21
%i648B

	full_text
	
i64 %23
%i648B

	full_text
	
i64 %25
Abitcast8B4
2
	full_text%
#
!%42 = bitcast double* %41 to i64*
-double*8B

	full_text

double* %41
Hload8B>
<
	full_text/
-
+%43 = load i64, i64* %42, align 8, !tbaa !8
'i64*8B

	full_text


i64* %42
�
�
	full_text~
|
z%44 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %19, i64 %21, i64 %23, i64 %25, i64 3
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %19
%i648B

	full_text
	
i64 %21
%i648B

	full_text
	
i64 %23
%i648B

	full_text
	
i64 %25
Abitcast8B4
2
	full_text%
#
!%45 = bitcast double* %44 to i64*
-double*8B

	full_text

double* %44
Hstore8B=
;
	full_text.
,
*store i64 %43, i64* %45, align 8, !tbaa !8
%i648B

	full_text
	
i64 %43
'i64*8B

	full_text


i64* %45
�
�
	full_text~
|
z%46 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %18, i64 %21, i64 %23, i64 %25, i64 4
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %18
%i648B

	full_text
	
i64 %21
%i648B

	full_text
	
i64 %23
%i648B

	full_text
	
i64 %25
Abitcast8B4
2
	full_text%
#
!%47 = bitcast double* %46 to i64*
-double*8B

	full_text

double* %46
Hload8B>
<
	full_text/
-
+%48 = load i64, i64* %47, align 8, !tbaa !8
'i64*8B

	full_text


i64* %47
�
�
	full_text~
|
z%49 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %19, i64 %21, i64 %23, i64 %25, i64 4
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %19
%i648B

	full_text
	
i64 %21
%i648B

	full_text
	
i64 %23
%i648B

	full_text
	
i64 %25
Abitcast8B4
2
	full_text%
#
!%50 = bitcast double* %49 to i64*
-double*8B

	full_text

double* %49
Hstore8B=
;
	full_text.
,
*store i64 %48, i64* %50, align 8, !tbaa !8
%i648B

	full_text
	
i64 %48
'i64*8B

	full_text


i64* %50
'br8B

	full_text

br label %51
$ret8B

	full_text


ret void
,double*8B

	full_text


double* %0
$i328B

	full_text


i32 %4
,double*8B

	full_text


double* %1
$i328B

	full_text


i32 %3
$i328B

	full_text


i32 %2
-; undefined function B

	full_text

 
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
$i648B

	full_text


i64 32
#i328B

	full_text	

i32 1
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
i64 4
#i648B

	full_text	

i64 0
#i648B

	full_text	

i64 2       	  
 

 
� �� �� 	� � 	� 	�    	 
 
compute_rhs2"
_Z13get_global_idj*�
npb-BT-compute_rhs2_S.clu
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
 
transfer_bytes_log1p
�fA

wgsize_log1p
�fA

devmap_label
 

transfer_bytes
��n