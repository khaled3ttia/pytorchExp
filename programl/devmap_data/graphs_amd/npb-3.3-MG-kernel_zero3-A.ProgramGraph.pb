

[external]
KcallBC
A
	full_text4
2
0%6 = tail call i64 @_Z13get_global_idj(i32 0) #2
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
3icmpB+
)
	full_text

%8 = icmp slt i32 %7, %1
"i32B

	full_text


i32 %7
6brB0
.
	full_text!

br i1 %8, label %9, label %21
 i1B

	full_text	

i1 %8
Ncall8BD
B
	full_text5
3
1%10 = tail call i64 @_Z13get_global_idj(i32 1) #2
8trunc8B-
+
	full_text

%11 = trunc i64 %10 to i32
%i648B

	full_text
	
i64 %10
Ncall8BD
B
	full_text5
3
1%12 = tail call i64 @_Z13get_global_idj(i32 2) #2
8trunc8B-
+
	full_text

%13 = trunc i64 %12 to i32
%i648B

	full_text
	
i64 %12
5mul8B,
*
	full_text

%14 = mul nsw i32 %13, %2
%i328B

	full_text
	
i32 %13
2add8B)
'
	full_text

%15 = add i32 %14, %11
%i328B

	full_text
	
i32 %14
%i328B

	full_text
	
i32 %11
1mul8B(
&
	full_text

%16 = mul i32 %15, %1
%i328B

	full_text
	
i32 %15
0add8B'
%
	full_text

%17 = add i32 %7, %4
$i328B

	full_text


i32 %7
2add8B)
'
	full_text

%18 = add i32 %17, %16
%i328B

	full_text
	
i32 %17
%i328B

	full_text
	
i32 %16
6sext8B,
*
	full_text

%19 = sext i32 %18 to i64
%i328B

	full_text
	
i32 %18
^getelementptr8BK
I
	full_text<
:
8%20 = getelementptr inbounds double, double* %0, i64 %19
%i648B

	full_text
	
i64 %19
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %20, align 8, !tbaa !8
-double*8B

	full_text

double* %20
'br8B

	full_text

br label %21
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
$i328B

	full_text


i32 %1
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
#i328B

	full_text	

i32 0
#i328B

	full_text	

i32 1
4double8B&
$
	full_text

double 0.000000e+00
#i328B

	full_text	

i32 2       	
 		                       " # $ $ %     
   	          !  ! ! && &&  &&  && ' ( ) * "
kernel_zero3"
_Z13get_global_idj*?
npb-MG-kernel_zero3.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282?

wgsize_log1p
???A

wgsize
@

transfer_bytes	
????
 
transfer_bytes_log1p
???A

devmap_label
