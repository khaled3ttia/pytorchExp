

[external]
KcallBC
A
	full_text4
2
0%5 = tail call i64 @_Z13get_global_idj(i32 1) #3
4truncB+
)
	full_text

%6 = trunc i64 %5 to i32
"i64B

	full_text


i64 %5
KcallBC
A
	full_text4
2
0%7 = tail call i64 @_Z13get_global_idj(i32 0) #3
3icmpB+
)
	full_text

%8 = icmp slt i32 %6, %3
"i32B

	full_text


i32 %6
4truncB+
)
	full_text

%9 = trunc i64 %7 to i32
"i64B

	full_text


i64 %7
4icmpB,
*
	full_text

%10 = icmp slt i32 %9, %2
"i32B

	full_text


i32 %9
.andB'
%
	full_text

%11 = and i1 %8, %10
 i1B

	full_text	

i1 %8
!i1B

	full_text


i1 %10
8brB2
0
	full_text#
!
br i1 %11, label %12, label %48
!i1B

	full_text


i1 %11
Wbitcast8BJ
H
	full_text;
9
7%13 = bitcast double* %0 to [13 x [13 x [5 x double]]]*
5icmp8B+
)
	full_text

%14 = icmp sgt i32 %1, 0
:br8B2
0
	full_text#
!
br i1 %14, label %15, label %48
#i18B

	full_text


i1 %14
0shl8B'
%
	full_text

%16 = shl i64 %5, 32
$i648B

	full_text


i64 %5
9ashr8B/
-
	full_text 

%17 = ashr exact i64 %16, 32
%i648B

	full_text
	
i64 %16
0shl8B'
%
	full_text

%18 = shl i64 %7, 32
$i648B

	full_text


i64 %7
9ashr8B/
-
	full_text 

%19 = ashr exact i64 %18, 32
%i648B

	full_text
	
i64 %18
5zext8B+
)
	full_text

%20 = zext i32 %1 to i64
0and8B'
%
	full_text

%21 = and i64 %20, 1
%i648B

	full_text
	
i64 %20
4icmp8B*
(
	full_text

%22 = icmp eq i32 %1, 1
:br8B2
0
	full_text#
!
br i1 %22, label %40, label %23
#i18B

	full_text


i1 %22
6sub8B-
+
	full_text

%24 = sub nsw i64 %20, %21
%i648B

	full_text
	
i64 %20
%i648B

	full_text
	
i64 %21
'br8B

	full_text

br label %25
Bphi8B9
7
	full_text*
(
&%26 = phi i64 [ 0, %23 ], [ %37, %25 ]
%i648B

	full_text
	
i64 %37
Dphi8B;
9
	full_text,
*
(%27 = phi i64 [ %24, %23 ], [ %38, %25 ]
%i648B

	full_text
	
i64 %24
%i648B

	full_text
	
i64 %38
�
�
	full_text~
|
z%28 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %13, i64 %17, i64 %19, i64 %26, i64 0
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %13
%i648B

	full_text
	
i64 %17
%i648B

	full_text
	
i64 %19
%i648B

	full_text
	
i64 %26
Wstore8BL
J
	full_text=
;
9store double 1.000000e+00, double* %28, align 8, !tbaa !8
-double*8B

	full_text

double* %28
�
�
	full_text~
|
z%29 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %13, i64 %17, i64 %19, i64 %26, i64 1
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %13
%i648B

	full_text
	
i64 %17
%i648B

	full_text
	
i64 %19
%i648B

	full_text
	
i64 %26
�
�
	full_text~
|
z%30 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %13, i64 %17, i64 %19, i64 %26, i64 4
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %13
%i648B

	full_text
	
i64 %17
%i648B

	full_text
	
i64 %19
%i648B

	full_text
	
i64 %26
@bitcast8B3
1
	full_text$
"
 %31 = bitcast double* %29 to i8*
-double*8B

	full_text

double* %29
ecall8B[
Y
	full_textL
J
Hcall void @llvm.memset.p0i8.i64(i8* align 8 %31, i8 0, i64 24, i1 false)
%i8*8B

	full_text
	
i8* %31
Wstore8BL
J
	full_text=
;
9store double 1.000000e+00, double* %30, align 8, !tbaa !8
-double*8B

	full_text

double* %30
.or8B&
$
	full_text

%32 = or i64 %26, 1
%i648B

	full_text
	
i64 %26
�
�
	full_text~
|
z%33 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %13, i64 %17, i64 %19, i64 %32, i64 0
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %13
%i648B

	full_text
	
i64 %17
%i648B

	full_text
	
i64 %19
%i648B

	full_text
	
i64 %32
Wstore8BL
J
	full_text=
;
9store double 1.000000e+00, double* %33, align 8, !tbaa !8
-double*8B

	full_text

double* %33
�
�
	full_text~
|
z%34 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %13, i64 %17, i64 %19, i64 %32, i64 1
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %13
%i648B

	full_text
	
i64 %17
%i648B

	full_text
	
i64 %19
%i648B

	full_text
	
i64 %32
�
�
	full_text~
|
z%35 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %13, i64 %17, i64 %19, i64 %32, i64 4
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %13
%i648B

	full_text
	
i64 %17
%i648B

	full_text
	
i64 %19
%i648B

	full_text
	
i64 %32
@bitcast8B3
1
	full_text$
"
 %36 = bitcast double* %34 to i8*
-double*8B

	full_text

double* %34
ecall8B[
Y
	full_textL
J
Hcall void @llvm.memset.p0i8.i64(i8* align 8 %36, i8 0, i64 24, i1 false)
%i8*8B

	full_text
	
i8* %36
Wstore8BL
J
	full_text=
;
9store double 1.000000e+00, double* %35, align 8, !tbaa !8
-double*8B

	full_text

double* %35
4add8B+
)
	full_text

%37 = add nsw i64 %26, 2
%i648B

	full_text
	
i64 %26
1add8B(
&
	full_text

%38 = add i64 %27, -2
%i648B

	full_text
	
i64 %27
5icmp8B+
)
	full_text

%39 = icmp eq i64 %38, 0
%i648B

	full_text
	
i64 %38
:br8B2
0
	full_text#
!
br i1 %39, label %40, label %25
#i18B

	full_text


i1 %39
Bphi8B9
7
	full_text*
(
&%41 = phi i64 [ 0, %15 ], [ %37, %25 ]
%i648B

	full_text
	
i64 %37
5icmp8B+
)
	full_text

%42 = icmp eq i64 %21, 0
%i648B

	full_text
	
i64 %21
:br8B2
0
	full_text#
!
br i1 %42, label %48, label %43
#i18B

	full_text


i1 %42
�
�
	full_text~
|
z%44 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %13, i64 %17, i64 %19, i64 %41, i64 0
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %13
%i648B

	full_text
	
i64 %17
%i648B

	full_text
	
i64 %19
%i648B

	full_text
	
i64 %41
Wstore8BL
J
	full_text=
;
9store double 1.000000e+00, double* %44, align 8, !tbaa !8
-double*8B

	full_text

double* %44
�
�
	full_text~
|
z%45 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %13, i64 %17, i64 %19, i64 %41, i64 1
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %13
%i648B

	full_text
	
i64 %17
%i648B

	full_text
	
i64 %19
%i648B

	full_text
	
i64 %41
�
�
	full_text~
|
z%46 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %13, i64 %17, i64 %19, i64 %41, i64 4
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %13
%i648B

	full_text
	
i64 %17
%i648B

	full_text
	
i64 %19
%i648B

	full_text
	
i64 %41
@bitcast8B3
1
	full_text$
"
 %47 = bitcast double* %45 to i8*
-double*8B

	full_text

double* %45
ecall8B[
Y
	full_textL
J
Hcall void @llvm.memset.p0i8.i64(i8* align 8 %47, i8 0, i64 24, i1 false)
%i8*8B

	full_text
	
i8* %47
Wstore8BL
J
	full_text=
;
9store double 1.000000e+00, double* %46, align 8, !tbaa !8
-double*8B

	full_text

double* %46
'br8B

	full_text

br label %48
$ret8B

	full_text


ret void
$i328B

	full_text


i32 %2
$i328B

	full_text


i32 %3
,double*8B

	full_text


double* %0
$i328B

	full_text


i32 %1
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
$i648B

	full_text


i64 32
#i328B

	full_text	

i32 1
#i648B

	full_text	

i64 0
#i648B

	full_text	

i64 2
$i648B

	full_text


i64 -2
#i648B

	full_text	

i64 4
!i88B

	full_text

i8 0
$i648B

	full_text


i64 24
#i328B

	full_text	

i32 0
4double8B&
$
	full_text

double 1.000000e+00
%i18B

	full_text


i1 false
#i648B

	full_text	

i64 1        	
 		  
 	 
initialize1"
_Z13get_global_idj"
llvm.memset.p0i8.i64*�
npb-SP-initialize1.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282
 
transfer_bytes_log1p
��YA

wgsize
<

devmap_label
 

wgsize_log1p
��YA

transfer_bytes
��1