

[external]
KcallBC
A
	full_text4
2
0%5 = tail call i64 @_Z13get_global_idj(i32 0) #2
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
0%7 = tail call i64 @_Z13get_global_idj(i32 1) #2
4truncB+
)
	full_text

%8 = trunc i64 %7 to i32
"i64B

	full_text


i64 %7
3icmpB+
)
	full_text

%9 = icmp slt i32 %8, %3
"i32B

	full_text


i32 %8
4icmpB,
*
	full_text

%10 = icmp slt i32 %6, %2
"i32B

	full_text


i32 %6
.andB'
%
	full_text

%11 = and i1 %10, %9
!i1B

	full_text


i1 %10
 i1B

	full_text	

i1 %9
8brB2
0
	full_text#
!
br i1 %11, label %12, label %23
!i1B

	full_text


i1 %11
0shl8B'
%
	full_text

%13 = shl i64 %5, 32
$i648B

	full_text


i64 %5
9ashr8B/
-
	full_text 

%14 = ashr exact i64 %13, 32
%i648B

	full_text
	
i64 %13
\getelementptr8BI
G
	full_text:
8
6%15 = getelementptr inbounds float, float* %0, i64 %14
%i648B

	full_text
	
i64 %14
Lload8BB
@
	full_text3
1
/%16 = load float, float* %15, align 4, !tbaa !9
+float*8B

	full_text


float* %15
4mul8B+
)
	full_text

%17 = mul nsw i32 %8, %2
$i328B

	full_text


i32 %8
5add8B,
*
	full_text

%18 = add nsw i32 %17, %6
%i328B

	full_text
	
i32 %17
$i328B

	full_text


i32 %6
6sext8B,
*
	full_text

%19 = sext i32 %18 to i64
%i328B

	full_text
	
i32 %18
\getelementptr8BI
G
	full_text:
8
6%20 = getelementptr inbounds float, float* %1, i64 %19
%i648B

	full_text
	
i64 %19
Lload8BB
@
	full_text3
1
/%21 = load float, float* %20, align 4, !tbaa !9
+float*8B

	full_text


float* %20
6fsub8B,
*
	full_text

%22 = fsub float %21, %16
)float8B

	full_text

	float %21
)float8B

	full_text

	float %16
Lstore8BA
?
	full_text2
0
.store float %22, float* %20, align 4, !tbaa !9
)float8B

	full_text

	float %22
+float*8B

	full_text


float* %20
'br8B

	full_text

br label %23
$ret8B

	full_text


ret void
$i328B

	full_text


i32 %2
$i328B

	full_text


i32 %3
*float*8B

	full_text

	float* %0
*float*8B

	full_text

	float* %1
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
$i648B

	full_text


i64 32        	
 		                       !" !! #$ #% ## &' &( && )+ 	+ , - .     
	              "! $ %# ' (  *) * * // //  // 0 1 2 2 "
reduce_kernel"
_Z13get_global_idj*?
-polybench-gpu-1.0-covariance-reduce_kernel.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02?

transfer_bytes
???

devmap_label


wgsize_log1p
???A
 
transfer_bytes_log1p
???A

wgsize
?