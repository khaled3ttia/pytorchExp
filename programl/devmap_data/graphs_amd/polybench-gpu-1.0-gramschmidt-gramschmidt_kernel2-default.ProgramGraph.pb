
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
3icmpB+
)
	full_text

%9 = icmp slt i32 %8, %4
"i32B

	full_text


i32 %8
7brB1
/
	full_text"
 
br i1 %9, label %10, label %23
 i1B

	full_text	

i1 %9
4mul8B+
)
	full_text

%11 = mul nsw i32 %8, %5
$i328B

	full_text


i32 %8
5add8B,
*
	full_text

%12 = add nsw i32 %11, %3
%i328B

	full_text
	
i32 %11
6sext8B,
*
	full_text

%13 = sext i32 %12 to i64
%i328B

	full_text
	
i32 %12
\getelementptr8BI
G
	full_text:
8
6%14 = getelementptr inbounds float, float* %0, i64 %13
%i648B

	full_text
	
i64 %13
Lload8BB
@
	full_text3
1
/%15 = load float, float* %14, align 4, !tbaa !9
+float*8B

	full_text


float* %14
4mul8B+
)
	full_text

%16 = mul nsw i32 %5, %3
5add8B,
*
	full_text

%17 = add nsw i32 %16, %3
%i328B

	full_text
	
i32 %16
6sext8B,
*
	full_text

%18 = sext i32 %17 to i64
%i328B

	full_text
	
i32 %17
\getelementptr8BI
G
	full_text:
8
6%19 = getelementptr inbounds float, float* %1, i64 %18
%i648B

	full_text
	
i64 %18
Lload8BB
@
	full_text3
1
/%20 = load float, float* %19, align 4, !tbaa !9
+float*8B

	full_text


float* %19
Cfdiv8B9
7
	full_text*
(
&%21 = fdiv float %15, %20, !fpmath !13
)float8B

	full_text

	float %15
)float8B

	full_text

	float %20
\getelementptr8BI
G
	full_text:
8
6%22 = getelementptr inbounds float, float* %2, i64 %13
%i648B

	full_text
	
i64 %13
Lstore8BA
?
	full_text2
0
.store float %21, float* %22, align 4, !tbaa !9
)float8B

	full_text

	float %21
+float*8B

	full_text


float* %22
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
i32 %3
$i328B

	full_text


i32 %5
*float*8B

	full_text

	float* %0
*float*8B

	full_text

	float* %1
*float*8B

	full_text

	float* %2
$i328B

	full_text


i32 %4
-; undefined function B

	full_text

 
#i328B

	full_text	

i32 0      	  
 

                      !  "    #% 
% % & & ' ( ) *     	 
           ! "  $# $ $ ++ ++ , "
gramschmidt_kernel2"
_Z13get_global_idj*?
4polybench-gpu-1.0-gramschmidt-gramschmidt_kernel2.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02?

transfer_bytes
???0

devmap_label


wgsize_log1p
k?A

wgsize
?
 
transfer_bytes_log1p
k?A