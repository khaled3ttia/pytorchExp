

[external]
8allocaB.
,
	full_text

%6 = alloca double, align 8
<bitcastB1
/
	full_text"
 
%7 = bitcast double* %6 to i8*
*double*B

	full_text


double* %6
XcallBP
N
	full_textA
?
=call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %7) #4
"i8*B

	full_text


i8* %7
KcallBC
A
	full_text4
2
0%8 = tail call i64 @_Z13get_global_idj(i32 0) #5
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
4icmpB,
*
	full_text

%10 = icmp slt i32 %9, %4
"i32B

	full_text


i32 %9
8brB2
0
	full_text#
!
br i1 %10, label %11, label %32
!i1B

	full_text


i1 %10
0shl8B'
%
	full_text

%12 = shl i64 %8, 32
$i648B

	full_text


i64 %8
9ashr8B/
-
	full_text 

%13 = ashr exact i64 %12, 32
%i648B

	full_text
	
i64 %12
^getelementptr8BK
I
	full_text<
:
8%14 = getelementptr inbounds double, double* %1, i64 %13
%i648B

	full_text
	
i64 %13
Abitcast8B4
2
	full_text%
#
!%15 = bitcast double* %14 to i64*
-double*8B

	full_text

double* %14
Hload8B>
<
	full_text/
-
+%16 = load i64, i64* %15, align 8, !tbaa !8
'i64*8B

	full_text


i64* %15
@bitcast8B3
1
	full_text$
"
 %17 = bitcast double* %6 to i64*
,double*8B

	full_text


double* %6
Hstore8B=
;
	full_text.
,
*store i64 %16, i64* %17, align 8, !tbaa !8
%i648B

	full_text
	
i64 %16
'i64*8B

	full_text


i64* %17
4mul8B+
)
	full_text

%18 = mul nsw i32 %9, %3
$i328B

	full_text


i32 %9
3add8B*
(
	full_text

%19 = add nsw i32 %2, 1
5icmp8B+
)
	full_text

%20 = icmp sgt i32 %3, 0
:br8B2
0
	full_text#
!
br i1 %20, label %21, label %32
#i18B

	full_text


i1 %20
5zext8B+
)
	full_text

%22 = zext i32 %3 to i64
'br8B

	full_text

br label %23
Bphi8B9
7
	full_text*
(
&%24 = phi i64 [ 0, %21 ], [ %30, %23 ]
%i648B

	full_text
	
i64 %30
8trunc8B-
+
	full_text

%25 = trunc i64 %24 to i32
%i648B

	full_text
	
i64 %24
2add8B)
'
	full_text

%26 = add i32 %18, %25
%i328B

	full_text
	
i32 %18
%i328B

	full_text
	
i32 %25
2mul8B)
'
	full_text

%27 = mul i32 %26, %19
%i328B

	full_text
	
i32 %26
%i328B

	full_text
	
i32 %19
6sext8B,
*
	full_text

%28 = sext i32 %27 to i64
%i328B

	full_text
	
i32 %27
ygetelementptr8Bf
d
	full_textW
U
S%29 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %0, i64 %28, i32 0
%i648B

	full_text
	
i64 %28
vcall8Bl
j
	full_text]
[
Ycall void @vranlc(i32 128, double* nonnull %6, double 0x41D2309CE5400000, double* %29) #4
,double*8B

	full_text


double* %6
-double*8B

	full_text

double* %29
8add8B/
-
	full_text 

%30 = add nuw nsw i64 %24, 1
%i648B

	full_text
	
i64 %24
7icmp8B-
+
	full_text

%31 = icmp eq i64 %30, %22
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %22
:br8B2
0
	full_text#
!
br i1 %31, label %32, label %23
#i18B

	full_text


i1 %31
Xcall8BN
L
	full_text?
=
;call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %7) #4
$i8*8B

	full_text


i8* %7
$ret8B

	full_text


ret void
$i328B

	full_text


i32 %2
$i328B

	full_text


i32 %3
,double*8B

	full_text


double* %1
$i328B

	full_text


i32 %4
6struct*8B'
%
	full_text

%struct.dcomplex* %0
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
:double8B,
*
	full_text

double 0x41D2309CE5400000
#i328B

	full_text	

i32 0
#i648B

	full_text	

i64 1
$i648B

	full_text


i64 32
%i328B

	full_text
	
i32 128
#i648B

	full_text	

i64 8
#i328B

	full_text	

i32 1
#i648B

	full_text	

i64 0        	
 		                      !  " #% $$ &' && () (* (( +, +- ++ ./ .. 01 00 23 24 22 56 55 78 79 77 :; := << >? @ @ @ "A B 	C 0    
	           !5 %$ ' )& *( , -+ /. 1 30 4$ 65 8" 97 ; =  <  "  <# $: <: $ DD EE FF GG >2 FF 2 DD  EE < GG <H 2I I I 0J 5K K L 2M M <N N O $"
compute_initial_conditions"
llvm.lifetime.start.p0i8"
_Z13get_global_idj"
vranlc"
llvm.lifetime.end.p0i8*?
$npb-FT-compute_initial_conditions.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02?

transfer_bytes
???

wgsize


wgsize_log1p
@s?A
 
transfer_bytes_log1p
@s?A

devmap_label
 