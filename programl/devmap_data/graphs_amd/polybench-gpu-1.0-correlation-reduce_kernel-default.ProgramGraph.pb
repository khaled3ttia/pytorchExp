

[external]
KcallBC
A
	full_text4
2
0%7 = tail call i64 @_Z13get_global_idj(i32 0) #2
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
KcallBC
A
	full_text4
2
0%9 = tail call i64 @_Z13get_global_idj(i32 1) #2
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
5icmpB-
+
	full_text

%11 = icmp slt i32 %10, %5
#i32B

	full_text
	
i32 %10
4icmpB,
*
	full_text

%12 = icmp slt i32 %8, %4
"i32B

	full_text


i32 %8
/andB(
&
	full_text

%13 = and i1 %12, %11
!i1B

	full_text


i1 %12
!i1B

	full_text


i1 %11
8brB2
0
	full_text#
!
br i1 %13, label %14, label %30
!i1B

	full_text


i1 %13
0shl8B'
%
	full_text

%15 = shl i64 %7, 32
$i648B

	full_text


i64 %7
9ashr8B/
-
	full_text 

%16 = ashr exact i64 %15, 32
%i648B

	full_text
	
i64 %15
\getelementptr8BI
G
	full_text:
8
6%17 = getelementptr inbounds float, float* %0, i64 %16
%i648B

	full_text
	
i64 %16
Lload8BB
@
	full_text3
1
/%18 = load float, float* %17, align 4, !tbaa !9
+float*8B

	full_text


float* %17
5mul8B,
*
	full_text

%19 = mul nsw i32 %10, %4
%i328B

	full_text
	
i32 %10
5add8B,
*
	full_text

%20 = add nsw i32 %19, %8
%i328B

	full_text
	
i32 %19
$i328B

	full_text


i32 %8
6sext8B,
*
	full_text

%21 = sext i32 %20 to i64
%i328B

	full_text
	
i32 %20
\getelementptr8BI
G
	full_text:
8
6%22 = getelementptr inbounds float, float* %2, i64 %21
%i648B

	full_text
	
i64 %21
Lload8BB
@
	full_text3
1
/%23 = load float, float* %22, align 4, !tbaa !9
+float*8B

	full_text


float* %22
6fsub8B,
*
	full_text

%24 = fsub float %23, %18
)float8B

	full_text

	float %23
)float8B

	full_text

	float %18
Lstore8BA
?
	full_text2
0
.store float %24, float* %22, align 4, !tbaa !9
)float8B

	full_text

	float %24
+float*8B

	full_text


float* %22
Icall8B?
=
	full_text0
.
,%25 = tail call float @_Z4sqrtf(float %3) #2
\getelementptr8BI
G
	full_text:
8
6%26 = getelementptr inbounds float, float* %1, i64 %16
%i648B

	full_text
	
i64 %16
Lload8BB
@
	full_text3
1
/%27 = load float, float* %26, align 4, !tbaa !9
+float*8B

	full_text


float* %26
6fmul8B,
*
	full_text

%28 = fmul float %25, %27
)float8B

	full_text

	float %25
)float8B

	full_text

	float %27
Cfdiv8B9
7
	full_text*
(
&%29 = fdiv float %24, %28, !fpmath !13
)float8B

	full_text

	float %24
)float8B

	full_text

	float %28
Lstore8BA
?
	full_text2
0
.store float %29, float* %22, align 4, !tbaa !9
)float8B

	full_text

	float %29
+float*8B

	full_text


float* %22
'br8B

	full_text

br label %30
$ret8B

	full_text


ret void
$i328B

	full_text


i32 %5
*float*8B

	full_text

	float* %0
$i328B

	full_text


i32 %4
*float*8B

	full_text

	float* %2
(float8B

	full_text


float %3
*float*8B

	full_text

	float* %1
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
#i328B

	full_text	

i32 0
$i648B

	full_text


i64 32
#i328B

	full_text	

i32 1        	
 		                       !" !! #$ #% ## &' &( && )) *+ ** ,- ,, ./ .0 .. 12 13 11 45 46 44 79 : ; 	; < = )> *    
	              "! $ %# ' ( +* -) /, 0# 2. 31 5 6  87 8 ?? 8 @@ ?? ) @@ ) ?? A B B C "
reduce_kernel"
_Z13get_global_idj"

_Z4sqrtf*?
.polybench-gpu-1.0-correlation-reduce_kernel.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02?

wgsize
?

transfer_bytes
???

devmap_label


wgsize_log1p
"??A
 
transfer_bytes_log1p
"??A