

[external]
KcallBC
A
	full_text4
2
0%5 = tail call i64 @_Z13get_global_idj(i32 0) #3
-shlB&
$
	full_text

%6 = shl i64 %5, 32
"i64B

	full_text


i64 %5
5ashrB-
+
	full_text

%7 = ashr exact i64 %6, 32
"i64B

	full_text


i64 %6
XgetelementptrBG
E
	full_text8
6
4%8 = getelementptr inbounds float, float* %0, i64 %7
"i64B

	full_text


i64 %7
HloadB@
>
	full_text1
/
-%9 = load float, float* %8, align 4, !tbaa !8
(float*B

	full_text

	float* %8
YgetelementptrBH
F
	full_text9
7
5%10 = getelementptr inbounds float, float* %1, i64 %7
"i64B

	full_text


i64 %7
JloadBB
@
	full_text3
1
/%11 = load float, float* %10, align 4, !tbaa !8
)float*B

	full_text


float* %10
acallBY
W
	full_textJ
H
F%12 = tail call float @llvm.fmuladd.f32(float %3, float %11, float %9)
'floatB

	full_text

	float %11
&floatB

	full_text


float %9
YgetelementptrBH
F
	full_text9
7
5%13 = getelementptr inbounds float, float* %2, i64 %7
"i64B

	full_text


i64 %7
JstoreBA
?
	full_text2
0
.store float %12, float* %13, align 4, !tbaa !8
'floatB

	full_text

	float %12
)float*B

	full_text


float* %13
"retB

	full_text


ret void
*float*8B

	full_text

	float* %0
(float8B

	full_text


float %3
*float*8B

	full_text

	float* %1
*float*8B

	full_text

	float* %2
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
#i328B

	full_text	

i32 0
$i648B

	full_text


i64 32       	  
 

              
     	 
                "
Triad"
_Z13get_global_idj"
llvm.fmuladd.f32*?
shoc-1.1.5-Triad-Triad.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282?

wgsize_log1p
˦?A

devmap_label
 

transfer_bytes
???8

wgsize
?
 
transfer_bytes_log1p
˦?A