
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
1addB*
(
	full_text

%10 = add nsw i32 %4, 1
5icmpB-
+
	full_text

%11 = icmp slt i32 %10, %7
#i32B

	full_text
	
i32 %10
"i32B

	full_text


i32 %7
8brB2
0
	full_text#
!
br i1 %11, label %54, label %12
!i1B

	full_text


i1 %11
7trunc8B,
*
	full_text

%13 = trunc i64 %9 to i32
$i648B

	full_text


i64 %9
7trunc8B,
*
	full_text

%14 = trunc i64 %8 to i32
$i648B

	full_text


i64 %8
3add8B*
(
	full_text

%15 = add nsw i32 %3, 1
8icmp8B.
,
	full_text

%16 = icmp slt i32 %15, %14
%i328B

	full_text
	
i32 %15
%i328B

	full_text
	
i32 %14
3add8B*
(
	full_text

%17 = add nsw i32 %2, 1
8icmp8B.
,
	full_text

%18 = icmp slt i32 %17, %13
%i328B

	full_text
	
i32 %17
%i328B

	full_text
	
i32 %13
/or8B'
%
	full_text

%19 = or i1 %16, %18
#i18B

	full_text


i1 %16
#i18B

	full_text


i1 %18
:br8B2
0
	full_text#
!
br i1 %19, label %54, label %20
#i18B

	full_text


i1 %19
Wbitcast8BJ
H
	full_text;
9
7%21 = bitcast double* %0 to [13 x [13 x [5 x double]]]*
Wbitcast8BJ
H
	full_text;
9
7%22 = bitcast double* %1 to [13 x [13 x [5 x double]]]*
0shl8B'
%
	full_text

%23 = shl i64 %6, 32
$i648B

	full_text


i64 %6
9ashr8B/
-
	full_text 

%24 = ashr exact i64 %23, 32
%i648B

	full_text
	
i64 %23
0shl8B'
%
	full_text

%25 = shl i64 %8, 32
$i648B

	full_text


i64 %8
9ashr8B/
-
	full_text 

%26 = ashr exact i64 %25, 32
%i648B

	full_text
	
i64 %25
0shl8B'
%
	full_text

%27 = shl i64 %9, 32
$i648B

	full_text


i64 %9
9ashr8B/
-
	full_text 

%28 = ashr exact i64 %27, 32
%i648B

	full_text
	
i64 %27
�
�
	full_text~
|
z%29 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %21, i64 %24, i64 %26, i64 %28, i64 0
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %21
%i648B

	full_text
	
i64 %24
%i648B

	full_text
	
i64 %26
%i648B

	full_text
	
i64 %28
Abitcast8B4
2
	full_text%
#
!%30 = bitcast double* %29 to i64*
-double*8B

	full_text

double* %29
Hload8B>
<
	full_text/
-
+%31 = load i64, i64* %30, align 8, !tbaa !8
'i64*8B

	full_text


i64* %30
�
�
	full_text~
|
z%32 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %22, i64 %24, i64 %26, i64 %28, i64 0
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %22
%i648B

	full_text
	
i64 %24
%i648B

	full_text
	
i64 %26
%i648B

	full_text
	
i64 %28
Abitcast8B4
2
	full_text%
#
!%33 = bitcast double* %32 to i64*
-double*8B

	full_text

double* %32
Hstore8B=
;
	full_text.
,
*store i64 %31, i64* %33, align 8, !tbaa !8
%i648B

	full_text
	
i64 %31
'i64*8B

	full_text


i64* %33
�
�
	full_text~
|
z%34 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %21, i64 %24, i64 %26, i64 %28, i64 1
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %21
%i648B

	full_text
	
i64 %24
%i648B

	full_text
	
i64 %26
%i648B

	full_text
	
i64 %28
Abitcast8B4
2
	full_text%
#
!%35 = bitcast double* %34 to i64*
-double*8B

	full_text

double* %34
Hload8B>
<
	full_text/
-
+%36 = load i64, i64* %35, align 8, !tbaa !8
'i64*8B

	full_text


i64* %35
�
�
	full_text~
|
z%37 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %22, i64 %24, i64 %26, i64 %28, i64 1
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %22
%i648B

	full_text
	
i64 %24
%i648B

	full_text
	
i64 %26
%i648B

	full_text
	
i64 %28
Abitcast8B4
2
	full_text%
#
!%38 = bitcast double* %37 to i64*
-double*8B

	full_text

double* %37
Hstore8B=
;
	full_text.
,
*store i64 %36, i64* %38, align 8, !tbaa !8
%i648B

	full_text
	
i64 %36
'i64*8B

	full_text


i64* %38
�
�
	full_text~
|
z%39 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %21, i64 %24, i64 %26, i64 %28, i64 2
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %21
%i648B

	full_text
	
i64 %24
%i648B

	full_text
	
i64 %26
%i648B

	full_text
	
i64 %28
Abitcast8B4
2
	full_text%
#
!%40 = bitcast double* %39 to i64*
-double*8B

	full_text

double* %39
Hload8B>
<
	full_text/
-
+%41 = load i64, i64* %40, align 8, !tbaa !8
'i64*8B

	full_text


i64* %40
�
�
	full_text~
|
z%42 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %22, i64 %24, i64 %26, i64 %28, i64 2
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %22
%i648B

	full_text
	
i64 %24
%i648B

	full_text
	
i64 %26
%i648B

	full_text
	
i64 %28
Abitcast8B4
2
	full_text%
#
!%43 = bitcast double* %42 to i64*
-double*8B

	full_text

double* %42
Hstore8B=
;
	full_text.
,
*store i64 %41, i64* %43, align 8, !tbaa !8
%i648B

	full_text
	
i64 %41
'i64*8B

	full_text


i64* %43
�
�
	full_text~
|
z%44 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %21, i64 %24, i64 %26, i64 %28, i64 3
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %21
%i648B

	full_text
	
i64 %24
%i648B

	full_text
	
i64 %26
%i648B

	full_text
	
i64 %28
Abitcast8B4
2
	full_text%
#
!%45 = bitcast double* %44 to i64*
-double*8B

	full_text

double* %44
Hload8B>
<
	full_text/
-
+%46 = load i64, i64* %45, align 8, !tbaa !8
'i64*8B

	full_text


i64* %45
�
�
	full_text~
|
z%47 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %22, i64 %24, i64 %26, i64 %28, i64 3
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %22
%i648B

	full_text
	
i64 %24
%i648B

	full_text
	
i64 %26
%i648B

	full_text
	
i64 %28
Abitcast8B4
2
	full_text%
#
!%48 = bitcast double* %47 to i64*
-double*8B

	full_text

double* %47
Hstore8B=
;
	full_text.
,
*store i64 %46, i64* %48, align 8, !tbaa !8
%i648B

	full_text
	
i64 %46
'i64*8B

	full_text


i64* %48
�
�
	full_text~
|
z%49 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %21, i64 %24, i64 %26, i64 %28, i64 4
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %21
%i648B

	full_text
	
i64 %24
%i648B

	full_text
	
i64 %26
%i648B

	full_text
	
i64 %28
Abitcast8B4
2
	full_text%
#
!%50 = bitcast double* %49 to i64*
-double*8B

	full_text

double* %49
Hload8B>
<
	full_text/
-
+%51 = load i64, i64* %50, align 8, !tbaa !8
'i64*8B

	full_text


i64* %50
�
�
	full_text~
|
z%52 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %22, i64 %24, i64 %26, i64 %28, i64 4
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %22
%i648B

	full_text
	
i64 %24
%i648B

	full_text
	
i64 %26
%i648B

	full_text
	
i64 %28
Abitcast8B4
2
	full_text%
#
!%53 = bitcast double* %52 to i64*
-double*8B

	full_text

double* %52
Hstore8B=
;
	full_text.
,
*store i64 %51, i64* %53, align 8, !tbaa !8
%i648B

	full_text
	
i64 %51
'i64*8B

	full_text


i64* %53
'br8B

	full_text

br label %54
$ret8B

	full_text


ret void
,double*8B

	full_text


double* %1
$i328B

	full_text


i32 %2
$i328B

	full_text


i32 %4
,double*8B

	full_text


double* %0
$i328B

	full_text


i32 %3
-; undefined function B

	full_text

 
#i328B

	full_text	

i32 2
#i328B

	full_text	

i32 1
$i648B

	full_text


i64 32
#i648B

	full_text	

i64 3
#i648B

	full_text	

i64 2
#i648B

	full_text	

i64 0
#i648B

	full_text	

i64 4
#i328B

	full_text	

i32 0
#i648B

	full_text	

i64 1       	  
 

� �
� �
� �� �� �� �� �
� �� �� � � � �    	  
 �
  � � � � �� ��  ��  �� � � 	� 	� 	� 	� 	� !	� #	� %	� '	� )	� d	� m	� Q	� Z	� +	� 4	� w
� �� 	� >	� G"
compute_rhs2"
_Z13get_global_idj*�
npb-SP-compute_rhs2.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02

devmap_label
 

transfer_bytes
��1

wgsize
<
 
transfer_bytes_log1p
��YA

wgsize_log1p
��YA