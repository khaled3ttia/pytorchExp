
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
3icmpB+
)
	full_text

%7 = icmp slt i32 %6, %3
"i32B

	full_text


i32 %6
6brB0
.
	full_text!

br i1 %7, label %8, label %17
 i1B

	full_text	

i1 %7
/shl8B&
$
	full_text

%9 = shl i64 %5, 32
$i648B

	full_text


i64 %5
8ashr8B.
,
	full_text

%10 = ashr exact i64 %9, 32
$i648B

	full_text


i64 %9
\getelementptr8BI
G
	full_text:
8
6%11 = getelementptr inbounds float, float* %0, i64 %10
%i648B

	full_text
	
i64 %10
Lload8BB
@
	full_text3
1
/%12 = load float, float* %11, align 4, !tbaa !8
+float*8B

	full_text


float* %11
\getelementptr8BI
G
	full_text:
8
6%13 = getelementptr inbounds float, float* %1, i64 %10
%i648B

	full_text
	
i64 %10
Lload8BB
@
	full_text3
1
/%14 = load float, float* %13, align 4, !tbaa !8
+float*8B

	full_text


float* %13
6fadd8B,
*
	full_text

%15 = fadd float %12, %14
)float8B

	full_text

	float %12
)float8B

	full_text

	float %14
\getelementptr8BI
G
	full_text:
8
6%16 = getelementptr inbounds float, float* %2, i64 %10
%i648B

	full_text
	
i64 %10
Lstore8BA
?
	full_text2
0
.store float %15, float* %16, align 4, !tbaa !8
)float8B

	full_text

	float %15
+float*8B

	full_text


float* %16
'br8B

	full_text

br label %17
$ret8B

	full_text


ret void
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

	float* %2
*float*8B

	full_text

	float* %1
-; undefined function B

	full_text

 
$i648B

	full_text


i64 32
#i328B

	full_text	

i32 0      	  
 

                     !     	 
  
    
        "" "" # # 
$ "
	VectorAdd"
_Z13get_global_idj*?
!nvidia-4.2-VectorAdd-VectorAdd.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02?

wgsize_log1p
]??A

transfer_bytes
???A

devmap_label

 
transfer_bytes_log1p
]??A

wgsize
?