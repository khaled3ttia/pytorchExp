

[external]
KcallBC
A
	full_text4
2
0%6 = tail call i64 @_Z13get_global_idj(i32 0) #3
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
%8 = icmp slt i32 %7, %2
"i32B

	full_text


i32 %7
6brB0
.
	full_text!

br i1 %8, label %9, label %22
 i1B

	full_text	

i1 %8
0shl8B'
%
	full_text

%10 = shl i64 %6, 32
$i648B

	full_text


i64 %6
9ashr8B/
-
	full_text 

%11 = ashr exact i64 %10, 32
%i648B

	full_text
	
i64 %10
\getelementptr8BI
G
	full_text:
8
6%12 = getelementptr inbounds float, float* %1, i64 %11
%i648B

	full_text
	
i64 %11
wgetelementptr8Bd
b
	full_textU
S
Q%13 = getelementptr inbounds %struct.latLong, %struct.latLong* %0, i64 %11, i32 0
%i648B

	full_text
	
i64 %11
Lload8BB
@
	full_text3
1
/%14 = load float, float* %13, align 4, !tbaa !9
+float*8B

	full_text


float* %13
5fsub8B+
)
	full_text

%15 = fsub float %3, %14
)float8B

	full_text

	float %14
wgetelementptr8Bd
b
	full_textU
S
Q%16 = getelementptr inbounds %struct.latLong, %struct.latLong* %0, i64 %11, i32 1
%i648B

	full_text
	
i64 %11
Mload8BC
A
	full_text4
2
0%17 = load float, float* %16, align 4, !tbaa !14
+float*8B

	full_text


float* %16
5fsub8B+
)
	full_text

%18 = fsub float %4, %17
)float8B

	full_text

	float %17
6fmul8B,
*
	full_text

%19 = fmul float %18, %18
)float8B

	full_text

	float %18
)float8B

	full_text

	float %18
ecall8B[
Y
	full_textL
J
H%20 = tail call float @llvm.fmuladd.f32(float %15, float %15, float %19)
)float8B

	full_text

	float %15
)float8B

	full_text

	float %15
)float8B

	full_text

	float %19
Jcall8B@
>
	full_text1
/
-%21 = tail call float @_Z4sqrtf(float %20) #3
)float8B

	full_text

	float %20
Mstore8BB
@
	full_text3
1
/store float %21, float* %12, align 4, !tbaa !15
)float8B

	full_text

	float %21
+float*8B

	full_text


float* %12
'br8B

	full_text

br label %22
$ret8B

	full_text


ret void
5struct*8B&
$
	full_text

%struct.latLong* %0
(float8B

	full_text


float %3
(float8B

	full_text


float %4
$i328B

	full_text


i32 %2
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
-; undefined function B
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
i32 1      	  
 

                       !" !! #$ #% ## &( ( ) * + ,     	 
 
   
          "! $ %  '& ' ' -- .. // --  // ! .. !0 0 1 1 
2 "
NearestNeighbor"
_Z13get_global_idj"

_Z4sqrtf"
llvm.fmuladd.f32*?
!rodinia-3.1-nn-NearestNeighbor.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02

wgsize
 

transfer_bytes
??
 
transfer_bytes_log1p
?_RA

wgsize_log1p
?_RA

devmap_label
 