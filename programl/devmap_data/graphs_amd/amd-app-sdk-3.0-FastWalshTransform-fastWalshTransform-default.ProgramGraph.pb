

[external]
KcallBC
A
	full_text4
2
0%3 = tail call i64 @_Z13get_global_idj(i32 0) #2
4truncB+
)
	full_text

%4 = trunc i64 %3 to i32
"i64B

	full_text


i64 %3
/uremB'
%
	full_text

%5 = urem i32 %4, %1
"i32B

	full_text


i32 %4
,shlB%
#
	full_text

%6 = shl i32 %1, 1
/udivB'
%
	full_text

%7 = udiv i32 %4, %1
"i32B

	full_text


i32 %4
-mulB&
$
	full_text

%8 = mul i32 %6, %7
"i32B

	full_text


i32 %6
"i32B

	full_text


i32 %7
-addB&
$
	full_text

%9 = add i32 %8, %5
"i32B

	full_text


i32 %8
"i32B

	full_text


i32 %5
.addB'
%
	full_text

%10 = add i32 %9, %1
"i32B

	full_text


i32 %9
3zextB+
)
	full_text

%11 = zext i32 %9 to i64
"i32B

	full_text


i32 %9
ZgetelementptrBI
G
	full_text:
8
6%12 = getelementptr inbounds float, float* %0, i64 %11
#i64B

	full_text
	
i64 %11
JloadBB
@
	full_text3
1
/%13 = load float, float* %12, align 4, !tbaa !7
)float*B

	full_text


float* %12
4zextB,
*
	full_text

%14 = zext i32 %10 to i64
#i32B

	full_text
	
i32 %10
ZgetelementptrBI
G
	full_text:
8
6%15 = getelementptr inbounds float, float* %0, i64 %14
#i64B

	full_text
	
i64 %14
JloadBB
@
	full_text3
1
/%16 = load float, float* %15, align 4, !tbaa !7
)float*B

	full_text


float* %15
4faddB,
*
	full_text

%17 = fadd float %13, %16
'floatB

	full_text

	float %13
'floatB

	full_text

	float %16
JstoreBA
?
	full_text2
0
.store float %17, float* %12, align 4, !tbaa !7
'floatB

	full_text

	float %17
)float*B

	full_text


float* %12
4fsubB,
*
	full_text

%18 = fsub float %13, %16
'floatB

	full_text

	float %13
'floatB

	full_text

	float %16
JstoreBA
?
	full_text2
0
.store float %18, float* %15, align 4, !tbaa !7
'floatB

	full_text

	float %18
)float*B

	full_text


float* %15
"retB

	full_text


ret void
$i328B

	full_text


i32 %1
*float*8B

	full_text

	float* %0
-; undefined function B

	full_text

 
#i328B

	full_text	

i32 0
#i328B

	full_text	

i32 1        	
 	 		                      !  "    #$ #% ## &' &( && )* * * * + +     
 	            ! " $ %# ' ( ) ,, ,, - . "
fastWalshTransform"
_Z13get_global_idj*?
FastWalshTransform_Kernels.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282

transfer_bytes
? 
 
transfer_bytes_log1p
?A

wgsize_log1p
?A

wgsize
?

devmap_label
 